"""Training-free baseline: Claude Haiku 5.5 decides every step of the pick-and-place through tools.

The model gets the task in words and four tools, and nothing else: no demonstrations, no scripted
grasp, no constants taken from the recorded data.

    look()                         overhead image with a 5 cm table grid in cm + wrist image + state
    move_to(x_cm, y_cm, z_cm)      move the gripper tip in a straight line (table coordinates)
    gripper(opening)               0 = closed ... 100 = fully open
    done(success, reason)          end the attempt

move_to and gripper return the new images and state right away, so one model call = one action.
What turns a position into joint angles is plain geometry, measured without the demos:
  * camera -> table: zero_shot/camera_calib.json (touch calibration, zero_shot/fit_touch_calib.py)
  * position -> joints: inverse kinematics on the SO-101 URDF, seeds spread over the joint ranges
  * "home" = the folded pose the arm is in when the script starts
Safety layer (the model can't override it): workspace box, tip never below the table, joint speed
limit, Cartesian paths checked with forward kinematics before moving; a rejected move comes back
to the model as an error it can react to.

Usage (inside the `lerobot` conda env, from the repo root; needs ANTHROPIC_API_KEY):
    python -m zero_shot.agent --dry-run --frame data/Yufeng1216/rollout_eval_pi05:0
        no robot: images come from a recorded frame, moves are only planned (checks the plumbing)
    python -m zero_shot.agent --trials 1 --step                real robot, Enter before every motion
    python -m zero_shot.agent --trials 1                       real robot, one attempt
    python -m zero_shot.agent --trials 5 --name vlm_agent      one evaluation block (8 blocks = 40 trials, results noted by hand
        like the other policies); each run continues from the next trial number
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from zero_shot.fit_touch_calib import distort, pixel_to_table  # noqa: F401  (pixel_to_table: for debugging)
from zero_shot.kinematics import ARM_JOINTS, fk, ik, solve_seeded

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ARM_JOINTS + ["gripper"]
MODEL = "claude-haiku-5-5"
CALIB = json.loads(Path(__file__).with_name("camera_calib.json").read_text(encoding="utf-8"))
TABLE_Z = CALIB["table_z_m"]                       # table height in the arm frame (m)
K = CALIB["distortion"]["k"]
H_INV = np.linalg.inv(np.array(CALIB["H_undistorted_pixel_to_table"]))
# Fingertip gap (cm) at each gripper opening, measured with a ruler (zero_shot/measure_gripper.py).
_GAP = json.loads(Path(__file__).with_name("gripper_geometry.json").read_text(encoding="utf-8"))["fingertip_gap_cm"]
GAP_OPENINGS, GAP_CM = [float(k) for k in _GAP], [float(v) for v in _GAP.values()]

# What the model may ask for (table coordinates, cm; z above the table). Covers the area that was
# touch-calibrated: the five toy spots and the box.
LIMITS = {"x": (5.0, 38.0), "y": (-20.0, 33.0), "z": (0.0, 25.0)}
SPEED = 45.0          # deg/s, max joint speed of any command
HZ = 30.0
IK_TOL = 0.012        # m: a target the solver misses by more than this is rejected
FLOOR = -0.005        # m below the table the tip may never go (contact with a soft toy pushes it a bit)

SYSTEM = """You control a small robot arm (SO-101, 5 joints + a two-finger gripper) on a white table, seen by an overhead camera and a camera on the gripper.
Task: pick up the blue plush toy and put it in the cardboard box.

Coordinates are table coordinates in centimetres: x grows away from the arm's base, y grows to the left in the overhead image (towards the box), z is the height of the fingertips above the table (0 = touching the table).
The gripper has one fixed finger and one moving finger; opening the gripper swings the moving finger away to the side, and closing brings it back against the fixed finger. The gripper position you give and get is the point midway between the two fingertips at the current opening, so an object centred there is between the fingers.
The overhead image has a 5 cm grid with these coordinates written on it; the red dot is the table point straight below that midpoint, and the short red line joins the two fingertips.
Sizes, measured: the toy is about 3 cm tall and 6-7 cm wide; the box walls are 8 cm tall. A toy held in the gripper hangs up to about 3 cm below the fingertips. Gap between the fingertips at opening 25 / 50 / 75 / 100: 4.5 / 9 / 12.7 / 15 cm.

Tools: move_to moves the gripper in a straight line and keeps the fingers pointing down when it can; gripper sets the opening (0 closed, 100 fully open); look returns fresh images; done ends the attempt.
After every move or gripper action you get new images and the state, including the measured gripper opening: if you closed the gripper on something, the measured opening stays above the commanded one.
Moves that are unsafe or out of reach are refused with a message; then choose another one.
You have a limited number of actions, so think about each one. When the toy has been released inside the box, or you are sure it can't be done, call done."""

TOOLS = [
    {"name": "look", "description": "Return the current overhead and wrist images and the robot state.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "move_to", "description": "Move the gripper (the point midway between the fingertips) in a straight line to (x, y, z) in table centimetres. z is the fingertip height above the table.",
     "input_schema": {"type": "object", "properties": {
         "x_cm": {"type": "number"}, "y_cm": {"type": "number"}, "z_cm": {"type": "number"}},
         "required": ["x_cm", "y_cm", "z_cm"], "additionalProperties": False}},
    {"name": "gripper", "description": "Set the gripper opening: 0 = closed, 100 = fully open.",
     "input_schema": {"type": "object", "properties": {"opening": {"type": "number"}},
                      "required": ["opening"], "additionalProperties": False}},
    {"name": "done", "description": "End the attempt.",
     "input_schema": {"type": "object", "properties": {
         "success": {"type": "boolean"}, "reason": {"type": "string"}},
         "required": ["success", "reason"], "additionalProperties": False}},
]
TOOLS = [{**t, "strict": True} for t in TOOLS]


# ---------------------------------------------------------------- geometry
def table_to_pixel(xy_m) -> np.ndarray:
    und = cv2.perspectiveTransform(np.float64(np.atleast_2d(xy_m))[None], H_INV)[0]
    return distort(und, K)


def gap_m(opening: float) -> float:
    return float(np.interp(opening, GAP_OPENINGS, GAP_CM)) / 100


def jaw_dir(q) -> np.ndarray:
    """Horizontal unit vector from the fixed fingertip towards the moving one (URDF: the moving jaw
    sits on the gripper_link's +x side, and the tip frame is that frame turned 180 deg about y)."""
    v = -fk(np.asarray(q)[:5])[:3, 0]
    v[2] = 0.0
    return v / np.linalg.norm(v)


def fingertips_m(q) -> tuple[np.ndarray, np.ndarray]:
    """(fixed fingertip, moving fingertip) in the arm frame, m. `q[5]` is the gripper opening."""
    fixed = fk(np.asarray(q)[:5])[:3, 3]
    return fixed, fixed + jaw_dir(q) * gap_m(q[5])


def tip_cm(q) -> np.ndarray:
    """The agent's gripper position: midway between the fingertips, table cm (z above the table)."""
    a, b = fingertips_m(q)
    p = (a + b) / 2
    return np.array([p[0] * 100, p[1] * 100, (p[2] - TABLE_Z) * 100])


def draw_grid(bgr: np.ndarray, q=None) -> np.ndarray:
    """Thin 5 cm grid of table coordinates (cm) on the overhead image, lines half-transparent."""
    lines = bgr.copy()
    labels = []
    for x in np.arange(5, 40, 5) / 100:
        ys = np.linspace(-0.20, 0.33, 50)
        cv2.polylines(lines, [table_to_pixel(np.c_[np.full_like(ys, x), ys]).astype(np.int32)], False, (0, 170, 0), 1)
        labels.append((f"x{int(round(x * 100))}", table_to_pixel([[x, -0.20]])[0] + (3, 0), (0, 130, 0)))
    for y in np.arange(-20, 35, 5) / 100:
        xs = np.linspace(0.05, 0.38, 50)
        cv2.polylines(lines, [table_to_pixel(np.c_[xs, np.full_like(xs, y)]).astype(np.int32)], False, (200, 110, 0), 1)
        labels.append((f"y{int(round(y * 100))}", table_to_pixel([[0.38, y]])[0] + (-10, -4), (190, 80, 0)))
    img = cv2.addWeighted(lines, 0.6, bgr, 0.4, 0)
    for text, (u, v), color in labels:
        cv2.putText(img, text, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    if q is not None:
        a, b = fingertips_m(q)
        pa, pb, pc = table_to_pixel([a[:2], b[:2], ((a + b) / 2)[:2]]).astype(int)
        cv2.line(img, tuple(pa), tuple(pb), (0, 0, 255), 2)
        cv2.circle(img, tuple(pc), 5, (0, 0, 255), -1)
    return img


def jpeg_block(bgr: np.ndarray) -> dict:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode()}}


# ---------------------------------------------------------------- robot (real or simulated)
class Robot:
    """The follower arm + both cameras. With dry_run, nothing is connected: images come from a
    recorded frame and the 'measured' joints are just the last command."""

    def __init__(self, dry_run: bool, frame: tuple[np.ndarray, np.ndarray] | None = None, step: bool = False):
        self.dry = dry_run
        self.step = step                   # wait for Enter before every motion (first real runs)
        cfg = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
        if dry_run:
            self.frame = frame
            self.q = np.array([0.0, -100.0, 97.0, 75.0, -90.0, 2.0])   # a folded pose, for the simulation only
        else:
            os.environ.setdefault("HF_LEROBOT_CALIBRATION", str(ROOT / "config" / "calibration"))
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            self.robot = SO101Follower(SO101FollowerConfig(port=cfg["follower"]["port"], id=cfg["follower"]["id"],
                                                           calibration_dir=ROOT / "config" / "calibration" / "robots" / "so_follower"))
            self.robot.connect(calibrate=False)
            self.caps = {}
            for name in ("top", "wrist"):
                c = cfg["cameras"][name]
                cap = cv2.VideoCapture(c["index_or_path"])
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
                self.caps[name] = cap
            self.q = self.read()
        self.home = self.q.copy()          # the pose the arm was in at start: folded, out of the way

    def read(self) -> np.ndarray:
        if self.dry:
            return self.q.copy()
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS])

    def images(self) -> tuple[np.ndarray, np.ndarray]:
        if self.dry:
            return self.frame
        out = []
        for name in ("top", "wrist"):
            for _ in range(4):               # drop buffered frames
                ok, img = self.caps[name].read()
            out.append(img if ok else np.zeros((480, 640, 3), np.uint8))
        return tuple(out)

    def send(self, q: np.ndarray) -> None:
        self.q = q.copy()
        if not self.dry:
            self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})

    def play(self, waypoints: list[np.ndarray]) -> None:
        """Joint-space interpolation through the waypoints at <= SPEED deg/s."""
        cur = self.read() if not self.dry else self.q
        for w in waypoints:
            n = max(1, int(np.ceil(np.max(np.abs(w - cur)) / SPEED * HZ)))
            for k in range(1, n + 1):
                t = time.perf_counter()
                self.send(cur + (w - cur) * k / n)
                if not self.dry:
                    time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t)))
            cur = w
        if not self.dry:
            # the servos trail the commands (the first real run measured the tip up to 4 cm short of a
            # just-finished move), so wait until the arm has arrived before taking pictures
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0:
                time.sleep(0.1)
                if np.max(np.abs(self.read()[:5] - cur[:5])) < 1.5:
                    break
            time.sleep(0.2)

    def close(self) -> None:
        if not self.dry:
            self.robot.disconnect()
            for cap in self.caps.values():
                cap.release()


# ---------------------------------------------------------------- motion planning (the safety layer)
def _solve(p, q_seed, seeded: bool):
    """IK for point p (m): fingers down if possible, else allow a tilt (high up the arm can't stay vertical)."""
    for w in (0.05, 0.01):
        q, err = solve_seeded(p, q_seed, orient_weight=w) if seeded else ik(p, q_seed[:5], orient_weight=w)
        if err <= IK_TOL:
            return q
    return None


def _line(q_from: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
    """Tip along a straight line, solved every 1 cm, each IK warm-started from the previous point."""
    start = fk(q_from[:5])[:3, 3]
    n = max(1, int(np.ceil(np.linalg.norm(goal - start) / 0.01)))
    path, q = [], q_from.copy()
    for p in np.linspace(start, goal, n + 1)[1:]:
        s = _solve(p, q, seeded=False)
        if s is None:
            return None
        q = np.r_[s, q_from[5]]
        path.append(q)
    return path


def _clear(q_from: np.ndarray, path: list[np.ndarray]) -> bool:
    """Forward kinematics every 2 deg along the joint interpolation: tip stays above the table
    (or, when starting below that line, e.g. folded at home, never goes lower than where it starts)."""
    floor = min(TABLE_Z + FLOOR, fk(q_from[:5])[2, 3] - 0.003)
    cur = q_from
    for w in path:
        m = max(1, int(np.ceil(np.max(np.abs(w - cur)) / 2.0)))
        if any(fk((cur + (w - cur) * k / m)[:5])[2, 3] < floor for k in range(1, m + 1)):
            return False
        cur = w
    return True


def plan_line(q_now: np.ndarray, target_cm) -> list[np.ndarray]:
    """Plan for the agent's gripper position (midway between the fingertips at the current opening):
    convert it to a position of the fixed fingertip (the IK tip), which depends on where the jaw
    points at the end pose, so solve twice."""
    x, y, z = target_cm
    for name, v in zip("xyz", target_cm):
        lo, hi = LIMITS[name]
        if not lo <= v <= hi:
            raise ValueError(f"{name}={v:.1f} cm is outside the allowed range {lo:.0f}..{hi:.0f} cm")
    centre = np.array([x / 100, y / 100, TABLE_Z + z / 100])
    half = gap_m(q_now[5]) / 2
    q_est = q_now
    for _ in range(2):
        tip = centre - jaw_dir(q_est) * half
        path = plan_tip(q_now, tip, target_cm)
        q_est = path[-1]
    return path


def plan_tip(q_now: np.ndarray, goal: np.ndarray, target_cm) -> list[np.ndarray]:
    """Waypoints (6 joints, gripper unchanged) to the target, safest option first:
      1. straight line of the tip (what the model asked for);
      2. if that has no IK (e.g. leaving the folded home pose, shoulder at its limit): joint
         interpolation to a solution found from seeds;
      3. if that would dip towards the table: first straight up to 15 cm, then the line.
    Every option is checked with forward kinematics; raises ValueError if none is safe."""
    x, y, z = target_cm
    path = _line(q_now, goal)
    if path is not None and _clear(q_now, path):
        return path
    s = _solve(goal, q_now, seeded=True)
    if s is None:
        raise ValueError(f"({x:.1f}, {y:.1f}, {z:.1f}) cm is out of the arm's reach")
    direct = [np.r_[s, q_now[5]]]
    if _clear(q_now, direct):
        return direct
    start = fk(q_now[:5])[:3, 3]
    up = np.array([start[0], start[1], TABLE_Z + 0.15])
    su = _solve(up, q_now, seeded=True)
    if su is not None:
        first = [np.r_[su, q_now[5]]]
        rest = _line(first[0], goal)
        if rest is not None and _clear(q_now, first) and _clear(first[0], rest):
            return first + rest
    raise ValueError("every path there would push the gripper into the table or the arm; approach from higher up")


def go_home(robot: Robot) -> None:
    """Up to 15 cm above where the tip is (clears the box walls), then joint interpolation home."""
    q = robot.read()
    t = tip_cm(q)
    try:
        if t[2] < 15:
            robot.play(plan_line(q, (np.clip(t[0], *LIMITS["x"]), np.clip(t[1], *LIMITS["y"]), 15.0)))
    except ValueError:
        pass
    robot.play([np.r_[robot.home[:5], robot.read()[5]], robot.home])


# ---------------------------------------------------------------- the agent
def observation(robot: Robot, note: str, log_dir: Path | None, step: int) -> list[dict]:
    top, wrist = robot.images()
    q = robot.read()
    t = tip_cm(q)
    top_g = draw_grid(top, q)
    if log_dir is not None:
        cv2.imencode(".jpg", np.hstack([top_g, wrist]))[1].tofile(str(log_dir / f"step_{step:02d}.jpg"))
    state = (f"{note}\nGripper (midway between the fingertips) at x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} cm. "
             f"Gripper opening: commanded {robot.q[5]:.0f}, measured {q[5]:.0f}.")
    return [{"type": "text", "text": "Overhead camera (grid in cm):"}, jpeg_block(top_g),
            {"type": "text", "text": "Wrist camera:"}, jpeg_block(wrist), {"type": "text", "text": state}]


def run_tool(robot: Robot, name: str, args: dict, log_dir, step) -> tuple[list[dict] | str, bool, bool]:
    """Returns (tool_result content, is_error, finished)."""
    if name == "done":
        return "Attempt ended.", False, True
    if name == "look":
        return observation(robot, "Current view.", log_dir, step), False, False
    if name == "gripper":
        g = float(np.clip(args["opening"], 0, 100))
        if robot.step:
            input(f"       Enter -> gripper {g:.0f} ")
        q = robot.read()
        robot.play([np.r_[q[:5], g]])
        time.sleep(0.4 if not robot.dry else 0)
        return observation(robot, f"Gripper set to {g:.0f}.", log_dir, step), False, False
    if name == "move_to":
        target = (args["x_cm"], args["y_cm"], args["z_cm"])
        try:
            path = plan_line(robot.read(), target)
        except ValueError as e:
            return f"Move refused: {e}.", True, False
        if robot.step:
            input(f"       Enter -> move to {target} ({len(path)} waypoints) ")
        robot.play(path)
        note = f"Moved to ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) cm."
        miss = np.linalg.norm(tip_cm(robot.read()) - np.array(target))
        if not robot.dry and miss > 2.0:
            # the servos stop when something is in the way (e.g. the toy or a finger against the box wall)
            note += f" The gripper stopped {miss:.1f} cm short of the target: something may be blocking it."
        return observation(robot, note, log_dir, step), False, False
    return f"Unknown tool {name}.", True, False


def attempt(client, robot: Robot, task_note: str, log_dir: Path, max_actions: int, effort: str) -> dict:
    messages = [{"role": "user", "content": observation(robot, task_note, log_dir, 0)}]
    record = {"actions": [], "usage": {"input": 0, "output": 0}, "claimed": None, "stop": None}
    t0 = time.time()
    finished = False
    for step in range(1, max_actions + 1):
        resp = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM, tools=TOOLS,
                                      output_config={"effort": effort}, messages=messages)
        record["usage"]["input"] += resp.usage.input_tokens
        record["usage"]["output"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})   # unchanged: thinking blocks stay valid
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "refusal" or not calls:
            record["stop"] = resp.stop_reason
            print(f"  [{step}] model stopped ({resp.stop_reason}): {text[:200]}")
            break
        results = []
        for b in calls:
            print(f"  [{step}] {b.name}({json.dumps(b.input)})" + (f"  -- {text[:160]}" if text else ""))
            content, is_err, fin = run_tool(robot, b.name, b.input, log_dir, step)
            record["actions"].append({"step": step, "tool": b.name, "input": b.input, "error": is_err,
                                      "result": content if isinstance(content, str) else content[-1]["text"],
                                      "t": round(time.time() - t0, 1)})
            if is_err:
                print(f"       -> {content}")
            if b.name == "done":
                record["claimed"] = b.input
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_err})
            finished = finished or fin
            # written after every action, so an attempt stopped with Ctrl+C still leaves its log
            (log_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        messages.append({"role": "user", "content": results})
        if finished:
            record["stop"] = "done"
            break
    else:
        record["stop"] = "max_actions"
    record["seconds"] = round(time.time() - t0, 1)
    # Haiku 5.5: $0.10 / $0.50 per million input / output tokens (prompts under 100k tokens)
    record["cost_usd"] = round(record["usage"]["input"] * 0.10e-6 + record["usage"]["output"] * 0.50e-6, 4)
    return record


def load_frame(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """'<dataset dir>:<episode>' -> first overhead and wrist frames of that episode (BGR)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from align_camera import reference_frame
    ds, ep = spec.rsplit(":", 1)
    repo = Path(ds).relative_to("data").as_posix() if ds.startswith("data") else ds
    return reference_frame(repo, "top", int(ep)), reference_frame(repo, "wrist", int(ep))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frame", default="data/Yufeng1216/rollout_eval_pi05:0", help="dry run: <dataset>:<episode>")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--name", default="vlm_agent_test")
    ap.add_argument("--max-actions", type=int, default=25)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--step", action="store_true", help="real robot: press Enter before every motion")
    ap.add_argument("--reset-time", type=int, default=25, help="seconds between trials to put the toy on the next spot")
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    out = ROOT / "outputs" / "zero_shot" / args.name
    out.mkdir(parents=True, exist_ok=True)
    robot = Robot(args.dry_run, load_frame(args.frame) if args.dry_run else None, step=args.step)
    # Trial numbers continue from what is already in this folder, so the 40 trials can be run in
    # several sittings with the same command; the spot follows the number (①..⑤ repeating).
    done_before = [json.loads(p.read_text(encoding="utf-8")).get("trial", 0) for p in out.glob("trial_*/record.json")]
    first = max(done_before, default=0) + 1
    try:
        for trial in range(first, first + args.trials):
            spot = "①②③④⑤"[(trial - 1) % 5]
            if not args.dry_run and trial > first:
                # reset phase like robot.ps1 eval (25 s): put the toy on the next spot, no key needed
                for left in range(args.reset_time, 0, -1):
                    print(f"\r  reset: put the toy on spot {spot} ... {left:2d} s ", end="", flush=True)
                    time.sleep(1)
                print()
            print(f"\nTrial {trial}: spot {spot}")
            log_dir = out / f"trial_{trial:02d}_{datetime.now():%Y%m%d_%H%M%S}"
            log_dir.mkdir()
            rec = attempt(client, robot, "Start. The arm is in its folded home pose.", log_dir, args.max_actions, args.effort)
            rec.update({"trial": trial, "spot": spot, "model": MODEL, "effort": args.effort, "dry_run": args.dry_run})
            (log_dir / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            if not args.dry_run:
                go_home(robot)
            print(f"Trial {trial} ({spot}): {rec['stop']}, claimed {rec['claimed']}, "
                  f"{len(rec['actions'])} actions, {rec['seconds']} s, ${rec['cost_usd']}")
    finally:
        if not args.dry_run:
            try:
                go_home(robot)
            finally:
                robot.close()


if __name__ == "__main__":
    main()
