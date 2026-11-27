"""Camera-to-arm calibration without the demonstrations: touch marks on the table with the gripper tip.

The VLM agent sees the overhead image but moves the arm in table coordinates (metres, robot base
frame), so it needs the map pixel -> table point. That map is a homography (the table is a plane),
fitted from pairs (pixel of a mark, tip position when touching it).

Procedure (inside the `lerobot` conda env, from the repo root):
    python zero_shot/calibrate_touch.py
1. Put 6-8 small flat marks (bits of tape or paper, ~1 cm) on the table, spread over the area where
   the toy and the box go, not in a line. Move the arm out of the way and press Enter: the script
   saves a clean overhead photo (the pixel of each mark is read from it afterwards).
2. Drive the follower with the leader arm. Touch the centre of mark 1 with the gripper tip (the point
   between the closed fingertips, gripper closed), hold still and press Enter. Repeat for every mark,
   in an order you note down. Type `u` + Enter to undo the last point. To finish, first bring the arm
   back to its folded rest pose with the leader, then type `q` + Enter (torque is released on exit).
3. The tip position comes from forward kinematics (zero_shot/kinematics.py, the SO-101 URDF), so
   nothing is taken from the demos. Results: outputs/zero_shot/calib_touch/<time>/points.json + photos.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from kinematics import ARM_JOINTS, fk  # noqa: E402

JOINTS = ARM_JOINTS + ["gripper"]
CAL = ROOT / "config" / "calibration"


def stdin_lines() -> queue.Queue:
    """Read lines from the keyboard in a background thread, so teleoperation keeps running."""
    q: queue.Queue = queue.Queue()

    def reader():
        for line in sys.stdin:
            q.put(line.strip().lower())

    threading.Thread(target=reader, daemon=True).start()
    return q


def main() -> None:
    os.environ.setdefault("HF_LEROBOT_CALIBRATION", str(CAL))
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    cfg = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
    out = ROOT / "outputs" / "zero_shot" / "calib_touch" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    cam = cfg["cameras"]["top"]
    cap = cv2.VideoCapture(cam["index_or_path"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.get("fourcc", "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

    def photo(name: str) -> None:
        for _ in range(5):          # drop buffered frames, keep the newest
            ok, img = cap.read()
        if ok:
            cv2.imencode(".jpg", img)[1].tofile(str(out / name))  # imwrite fails on non-ASCII paths

    follower = SO101Follower(SO101FollowerConfig(port=cfg["follower"]["port"], id=cfg["follower"]["id"],
                                                 calibration_dir=CAL / "robots" / "so_follower"))
    leader = SO101Leader(SOLeaderTeleopConfig(port=cfg["leader"]["port"], id=cfg["leader"]["id"],
                                              calibration_dir=CAL / "teleoperators" / "so_leader"))
    follower.connect(calibrate=False)
    leader.connect(calibrate=False)
    lines = stdin_lines()
    points: list[dict] = []
    print("Teleoperation is on. Move the arm out of the camera's way and press Enter for the clean photo.")
    clean_done = False
    try:
        while True:
            t = time.perf_counter()
            follower.send_action(leader.get_action())
            try:
                cmd = lines.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd is not None:
                if not clean_done:
                    photo("clean.jpg")
                    clean_done = True
                    print("Saved clean.jpg. Now touch mark 1 with the tip and press Enter (u = undo; rest pose, then q = finish).")
                elif cmd == "q":
                    break
                elif cmd == "u" and points:
                    gone = points.pop()
                    print(f"  undone point {gone['index']}")
                else:
                    time.sleep(0.3)  # let the arm settle under the last command
                    obs = follower.get_observation()
                    q = np.array([obs[f"{j}.pos"] for j in JOINTS])
                    xyz = fk(q[:5])[:3, 3]
                    i = len(points) + 1
                    photo(f"touch_{i}.jpg")
                    points.append({"index": i, "tip_xyz_m": np.round(xyz, 4).tolist(), "joints_deg": np.round(q, 2).tolist()})
                    print(f"  point {i}: tip x={xyz[0] * 100:.1f} y={xyz[1] * 100:.1f} z={xyz[2] * 100:.1f} cm. Next mark, Enter.")
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t)))
    finally:
        (out / "points.json").write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")
        print(f"saved {len(points)} points to {out}")
        leader.disconnect()
        follower.disconnect()
        cap.release()


if __name__ == "__main__":
    main()
