"""Zero-shot pick-and-place on the real SO-101: VLM pointing -> homography -> IK -> scripted motion.

One attempt:
  1. photo from the overhead camera (arm in its rest pose, out of the way)
  2. Qwen3-VL points at the toy and at the box                       (zero_shot/pointing.py)
  3. pixel -> table (x, y) with the mappings fitted from the demos    (zero_shot/homography.json:
     H_pixel_to_table for the toy, H_pixel_to_place for the box)
  4. IK for: above the toy, at the toy, above the box                 (zero_shot/kinematics.py)
  5. scripted motion, open loop:
       rest -> open gripper above toy -> straight down -> close -> straight up to PLACE_Z (11 cm)
            -> lift to ~CARRY_Z (17 cm) -> across to above the box at that height (the hanging toy
            clears the box wall) -> down to PLACE_Z -> open -> rest

Safety (the arm is driven by a script, not by a person, so every limit is checked before moving):
  * targets must lie in the area the toy / box can be in (WORKSPACE), else the attempt is skipped
  * IK must reach the target (error below IK_TOL), else skipped
  * every point of every planned path is checked with forward kinematics: tip never below MIN_Z,
    and never below CARRY_MIN_Z while carrying the toy across to the box
  * joints move at most --speed deg/s (default 20, slow); the command never jumps
  * --step waits for Enter before every segment (use it for the first runs)
  * Ctrl+C: the arm goes slowly back to rest before torque is released (not dropped mid-air)

Usage (inside the `lerobot` conda env, from the repo root):
    python -m zero_shot.run_grasp --dry-run              # camera + VLM + IK only, robot not touched
    python -m zero_shot.run_grasp --step                 # real arm, Enter before each segment
    python -m zero_shot.run_grasp --trials 40 --name zero_shot   # evaluation: logs every attempt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from zero_shot.kinematics import ARM_JOINTS, fk, ik, solve_grasp

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ARM_JOINTS + ["gripper"]

# Poses measured from the 50 demonstrations (degrees; gripper 0 = closed .. 100 = open).
REST = np.array([-1.3, -102.1, 95.9, 77.6, -89.3, 2.2])   # mean first frame of the demos
GRIP_OPEN = 40.0                                         # demos open to ~39 before grasping (--open)
GRIP_CLOSED = 0.0                                        # demos command ~1 when holding the toy
GRASP_Z = -0.008                                         # tip height at the grasp moment in the demos (m, --grasp-z)
CLOSE_EARLY = 0.0                                        # start closing this far above GRASP_Z, while still descending (m, --close-early)
HOVER_DZ = 0.06                                          # approach from 6 cm above the toy
PLACE_Z = 0.11                                           # demos release ~10 cm above the table
CARRY_Z = 0.17                                           # carry height: demos peak at 13.7 cm (90%: 16.6) while carrying
WRIST_ROLL = -90.0                                       # same as at rest and in the demos

# Where targets may be (m, base frame): demo grasp spots / release point +- a margin.
WORKSPACE = {"toy": ((0.15, 0.34), (-0.14, 0.05)), "box": ((0.15, 0.38), (0.05, 0.28))}
MIN_Z = -0.025        # lowest tip height allowed anywhere on a path (rest pose itself is -1.1 cm)
CARRY_MIN_Z = 0.14    # while moving across to the box (the toy hangs several cm under the tip)
IK_TOL = {"grasp": 0.005, "hover": 0.012, "place": 0.015, "carry": 0.035}  # carry poses are in free air;
# above ~13 cm the 5-joint arm cannot keep the gripper vertical, so they may be a few cm off (paths are still checked)
TOY_PROMPT, BOX_PROMPT = "the blue plush toy", "the center of the cardboard box"


# ---------------------------------------------------------------- perception
def pixel_to_table(H: np.ndarray, uv) -> np.ndarray:
    p = H @ np.array([uv[0], uv[1], 1.0])
    return p[:2] / p[2]


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray:
    for _ in range(5):  # drop buffered frames so the photo is current
        cap.read()
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("overhead camera returned no frame")
    return bgr


# ---------------------------------------------------------------- planning
def in_workspace(kind: str, xy) -> bool:
    (x0, x1), (y0, y1) = WORKSPACE[kind]
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def vertical_path(xy, z_from, z_to, q_start, n=8) -> list[np.ndarray]:
    """Joint waypoints moving the tip straight up/down, each IK warm-started from the previous one."""
    path, q = [], q_start[:5].copy()
    for z in np.linspace(z_from, z_to, n + 1)[1:]:
        q, err = ik([xy[0], xy[1], z], q)
        if err > IK_TOL["hover"]:
            raise ValueError(f"no IK on the vertical path at z={z:.3f} (error {err * 100:.1f} cm)")
        path.append(q.copy())
    return path


def plan(toy_xy, box_xy) -> list[tuple[str, np.ndarray]]:
    """Named key poses (6 values: 5 arm joints + gripper). Raises ValueError if anything is unsafe."""
    for kind, xy in (("toy", toy_xy), ("box", box_xy)):
        if not in_workspace(kind, xy):
            raise ValueError(f"{kind} at ({xy[0] * 100:.1f}, {xy[1] * 100:.1f}) cm is outside the workspace")
    q_grasp, e = solve_grasp([*toy_xy, GRASP_Z], WRIST_ROLL)
    if e > IK_TOL["grasp"]:
        raise ValueError(f"no IK for the grasp (error {e * 100:.1f} cm)")
    q_hover, e = ik([*toy_xy, GRASP_Z + HOVER_DZ], q_grasp)
    if e > IK_TOL["hover"]:
        raise ValueError(f"no IK above the toy (error {e * 100:.1f} cm)")
    q_place, e = solve_grasp([*box_xy, PLACE_Z], WRIST_ROLL)
    if e > IK_TOL["place"]:
        raise ValueError(f"no IK above the box (error {e * 100:.1f} cm)")
    q_carry, e = ik([*box_xy, CARRY_Z], q_place)
    if e > IK_TOL["carry"]:
        raise ValueError(f"no IK high above the box (error {e * 100:.1f} cm)")

    g = lambda q, grip: np.r_[q[:5], grip]  # noqa: E731
    poses = [("above toy, open", g(q_hover, GRIP_OPEN))]
    def grip_at(q):  # open, or already closing during the last CLOSE_EARLY of the descent (pinches a soft toy)
        if CLOSE_EARLY <= 0:
            return GRIP_OPEN
        f = np.clip((GRASP_Z + CLOSE_EARLY - fk(q)[2, 3]) / CLOSE_EARLY, 0, 1)
        return GRIP_OPEN + f * (GRIP_CLOSED - GRIP_OPEN)
    poses += [("down", g(q, grip_at(q))) for q in vertical_path(toy_xy, GRASP_Z + HOVER_DZ, GRASP_Z, q_hover, n=12)]
    poses += [("close", g(poses[-1][1], GRIP_CLOSED))]
    poses += [("up", g(q, GRIP_CLOSED)) for q in vertical_path(toy_xy, GRASP_Z, PLACE_Z, poses[-1][1], n=12)]
    q_lift, e = ik([*toy_xy, CARRY_Z], poses[-1][1][:5])
    if e > IK_TOL["carry"]:
        raise ValueError(f"no IK high above the toy (error {e * 100:.1f} cm)")
    poses += [("lift", g(q_lift, GRIP_CLOSED)), ("above box", g(q_carry, GRIP_CLOSED)), ("lower", g(q_place, GRIP_CLOSED)),
              ("release", g(q_place, GRIP_OPEN)), ("rest", REST.copy())]
    return poses


def interpolate(q_from: np.ndarray, q_to: np.ndarray, speed: float, hz: float) -> list[np.ndarray]:
    """Straight line in joint space at <= `speed` deg/s (gripper counts as a joint)."""
    n = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) / speed * hz)))
    return [q_from + (q_to - q_from) * (k / n) for k in range(1, n + 1)]


def check_paths(start: np.ndarray, poses, speed: float, hz: float) -> float:
    """FK along every interpolated command; raises if the tip would go below MIN_Z. Returns duration (s)."""
    q, steps = start, 0
    for name, target in poses:
        for c in interpolate(q, target, speed, hz):
            z = fk(c[:5])[2, 3]
            floor = CARRY_MIN_Z if name == "above box" else MIN_Z
            if z < floor:
                raise ValueError(f"segment '{name}' would take the tip to z={z * 100:.1f} cm")
            steps += 1
        q = target
    return steps / hz


# ---------------------------------------------------------------- execution
class Arm:
    def __init__(self, cfg: dict, hz: float):
        os.environ.setdefault("HF_LEROBOT_CALIBRATION", str(ROOT / "config" / "calibration"))
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=cfg["follower"]["port"], id=cfg["follower"]["id"],
                                                       calibration_dir=ROOT / "config" / "calibration" / "robots" / "so_follower"))
        self.robot.connect(calibrate=False)
        self.hz = hz

    def read(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS])

    def move(self, target: np.ndarray, speed: float) -> None:
        for c in interpolate(self.read(), target, speed, self.hz):
            t = time.perf_counter()
            self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, c)})
            time.sleep(max(0.0, 1 / self.hz - (time.perf_counter() - t)))

    def close(self, speed: float) -> None:
        """Back to rest slowly, then release torque (at rest the arm is folded and supported)."""
        try:
            self.move(REST, speed)
        finally:
            self.robot.disconnect()


def report(arm: Arm, target: np.ndarray, label: str) -> dict:
    """Where the arm really is vs. where it was told to go (a soft toy can stop the descent)."""
    time.sleep(0.5)
    q = arm.read()
    z_cmd, z_real = fk(target[:5])[2, 3], fk(q[:5])[2, 3]
    print(f"    [{label}] commanded {np.round(target, 1)}")
    print(f"    [{label}] measured  {np.round(q, 1)}")
    print(f"    [{label}] tip z commanded {z_cmd * 100:.1f} cm, measured {z_real * 100:.1f} cm")
    return {f"{label}_z_cmd": round(z_cmd * 100, 1), f"{label}_z_real": round(z_real * 100, 1),
            f"{label}_q_real": np.round(q, 1).tolist()}


def open_camera(cam: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam["index_or_path"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.get("fourcc", "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    return cap


def snapshot(cams: dict, prefix: Path) -> None:
    """Save a frame from each camera, e.g. to see where the fingers ended up relative to the toy."""
    frames = []
    for name, cap in cams.items():
        try:
            frames.append(cv2.resize(grab_frame(cap), (640, 480)))
        except RuntimeError:
            pass
    if frames:
        cv2.imencode(".jpg", np.hstack(frames))[1].tofile(f"{prefix}.jpg")


def execute(arm: Arm, poses, speed: float, step: bool, cams: dict | None = None, prefix: Path | None = None) -> dict:
    """Run the key poses; returns measurements at the bottom of the descent and after closing.
    With `cams`, saves overhead + wrist pictures above the toy, at the bottom and after closing."""
    last, info = None, {}
    shot = (lambda tag: snapshot(cams, Path(f"{prefix}_{tag}"))) if cams else (lambda tag: None)  # noqa: E731
    for k, (name, target) in enumerate(poses):
        if step and name != last:
            input(f"  Enter -> {name} ")
        arm.move(target, speed)
        if name in ("close", "release"):
            time.sleep(0.6)  # let the gripper finish closing / the toy drop
        if name == "above toy, open":
            shot("1_above")
        if name == "down" and poses[k + 1][0] != "down":
            info.update(report(arm, target, "bottom"))
            shot("2_bottom")
        if name == "close":
            info.update(report(arm, target, "closed"))
            shot("3_closed")
        last = name
    return info


def annotate(bgr, toy_uv, box_uv, text: str) -> np.ndarray:
    img = bgr.copy()
    for uv, color in ((toy_uv, (0, 0, 255)), (box_uv, (0, 255, 255))):
        if uv is not None:
            cv2.drawMarker(img, (int(uv[0]), int(uv[1])), color, cv2.MARKER_CROSS, 24, 2)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def main() -> None:
    global GRIP_OPEN, GRASP_Z, CLOSE_EARLY  # command-line overrides of the defaults used by plan()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="camera + VLM + planning only; the arm is not connected")
    ap.add_argument("--step", action="store_true", help="wait for Enter before each motion segment")
    ap.add_argument("--speed", type=float, default=20.0, help="max joint speed, deg/s (default 20 = slow)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--name", default="zero_shot_test", help="log name: outputs/zero_shot/<name>/")
    ap.add_argument("--image", help="plan from this image instead of the camera (implies --dry-run)")
    ap.add_argument("--open", type=float, default=GRIP_OPEN, help=f"gripper opening 0-100 (default {GRIP_OPEN:.0f})")
    ap.add_argument("--close-early", type=float, default=0.0,
                    help="start closing the gripper this many cm above the grasp height, while still going down")
    ap.add_argument("--grasp-z", type=float, default=GRASP_Z * 100, help=f"tip height at the grasp, cm (default {GRASP_Z * 100})")
    args = ap.parse_args()
    dry = args.dry_run or args.image is not None
    GRIP_OPEN, GRASP_Z, CLOSE_EARLY = args.open, args.grasp_z / 100, args.close_early / 100
    hz = 30.0

    cfg = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
    calib = json.loads((ROOT / "zero_shot" / "homography.json").read_text())
    H_toy, H_box = np.array(calib["H_pixel_to_table"]), np.array(calib["H_pixel_to_place"])
    out = ROOT / "outputs" / "zero_shot" / args.name
    out.mkdir(parents=True, exist_ok=True)

    from zero_shot.pointing import Pointer
    print("loading Qwen3-VL ...")
    pointer = Pointer()
    cap = None
    if args.image is None:
        cam = cfg["cameras"]["top"]
        cap = open_camera(cam)
    arm = None if dry else Arm(cfg, hz)
    wrist = open_camera(cfg["cameras"]["wrist"]) if arm is not None and "wrist" in cfg["cameras"] else None

    log_path = out / "attempts.csv"
    new_log = not log_path.exists()
    first = 1 if new_log else len(log_path.read_text(encoding="utf-8").splitlines())  # continue numbering
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["trial", "time", "toy_u", "toy_v", "box_u", "box_v", "toy_x_cm", "toy_y_cm",
                         "box_x_cm", "box_y_cm", "vlm_s", "motion_s", "status", "result", "open", "grasp_z_cm", "measured"])
    try:
        if arm is not None:
            print("moving to rest ...")
            arm.move(REST, args.speed)
        for k in range(args.trials):
            trial = first + k
            if not dry:
                spot = "①②③④⑤"[(trial - 1) % 5]  # evaluation order of docs/08: ①..⑤, repeated
                input(f"\n[trial {trial}] put the toy on spot {spot}, then Enter (Ctrl+C to stop) ")
            bgr = grab_frame(cap) if cap is not None else cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            toy_uv, _, t1 = pointer.point(rgb, TOY_PROMPT, "toy")
            box_uv, _, t2 = pointer.point(rgb, BOX_PROMPT, "box")
            row = [trial, time.strftime("%H:%M:%S"), *(np.round(toy_uv, 1) if toy_uv else ["", ""]),
                   *(np.round(box_uv, 1) if box_uv else ["", ""])]
            status, motion_s, toy_xy, box_xy = "ok", 0.0, None, None
            try:
                if toy_uv is None or box_uv is None:
                    raise ValueError("VLM gave no point")
                toy_xy, box_xy = pixel_to_table(H_toy, toy_uv), pixel_to_table(H_box, box_uv)
                print(f"  toy px {np.round(toy_uv)} -> ({toy_xy[0] * 100:.1f}, {toy_xy[1] * 100:.1f}) cm | "
                      f"box px {np.round(box_uv)} -> ({box_xy[0] * 100:.1f}, {box_xy[1] * 100:.1f}) cm | VLM {t1 + t2:.1f} s")
                poses = plan(toy_xy, box_xy)
                start = arm.read() if arm is not None else REST
                motion_s = check_paths(start, poses, args.speed, hz)
                print(f"  plan OK: {len(poses)} key poses, {motion_s:.1f} s of motion at {args.speed:.0f} deg/s")
                for name, q in poses:
                    if name not in ("down", "up"):
                        print(f"    {name:<16} {np.round(q, 1)}  tip z {fk(q[:5])[2, 3] * 100:5.1f} cm")
            except ValueError as e:
                status = f"skipped: {e}"
                print(f"  SKIPPED: {e}")
            cv2.imencode(".jpg", annotate(bgr, toy_uv, box_uv, f"trial {trial}: {status[:60]}"))[1].tofile(
                str(out / f"trial_{trial:03d}.jpg"))
            result, measured = "", {}
            if status == "ok" and arm is not None:
                t = time.time()
                cams = {"top": cap, **({"wrist": wrist} if wrist is not None else {})}
                measured = execute(arm, poses, args.speed, args.step, cams, out / f"trial_{trial:03d}")
                motion_s = time.time() - t
                result = input("  result? 1 = success, 2 = grasped not placed, 3 = not grasped: ").strip()
            xy = [*(np.round(toy_xy * 100, 1) if toy_xy is not None else ["", ""]),
                  *(np.round(box_xy * 100, 1) if box_xy is not None else ["", ""])]
            writer.writerow(row + xy + [round(t1 + t2, 2), round(motion_s, 1), status, result,
                             GRIP_OPEN, round(GRASP_Z * 100, 1), json.dumps(measured)])
            log.flush()
    except KeyboardInterrupt:
        print("\nstopped by Ctrl+C")
    finally:
        log.close()
        if cap is not None:
            cap.release()
        if arm is not None and wrist is not None:
            wrist.release()
        if arm is not None:
            print("returning to rest ...")
            arm.close(args.speed)
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
