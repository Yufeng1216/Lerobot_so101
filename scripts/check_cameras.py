"""Pre-flight check: open every camera in config/robot.json exactly as LeRobot will.

Why this exists: `lerobot-record` creates the dataset folder BEFORE it connects the
cameras. If a camera then fails (e.g. "failed to set fps=30 (actual_fps=0.0)"), an empty
dataset is left behind and the next run refuses to start because the name is taken.
Running this first catches the camera problem before anything is written.

Exit code 0 = all cameras OK, 1 = at least one failed (robot.ps1 then stops).

Usage (inside the `lerobot` conda env):
    python scripts/check_cameras.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CONFIG = Path(__file__).resolve().parent.parent / "config" / "robot.json"
RETRIES = 3  # the wrist camera occasionally fails once right after another program released it


def check(name: str, cfg: dict) -> bool:
    cam_cfg = OpenCVCameraConfig(
        index_or_path=cfg["index_or_path"],
        fps=cfg["fps"],
        width=cfg["width"],
        height=cfg["height"],
        fourcc=cfg.get("fourcc"),
    )
    for attempt in range(1, RETRIES + 1):
        cam = OpenCVCamera(cam_cfg)
        try:
            cam.connect()
            frame = cam.read()
            print(f"  {name:<6} index {cfg['index_or_path']}: OK  {frame.shape[1]}x{frame.shape[0]}")
            return True
        except Exception as e:  # report and retry: we only want to know whether it works
            print(f"  {name:<6} index {cfg['index_or_path']}: attempt {attempt} failed: {e}")
            time.sleep(1.0)
        finally:
            if cam.is_connected:
                cam.disconnect()
    return False


def main() -> int:
    cameras = json.loads(CONFIG.read_text(encoding="utf-8"))["cameras"]
    print("Checking cameras from config/robot.json ...")
    ok = all([check(name, cfg) for name, cfg in cameras.items()])
    if not ok:
        print(
            "\nA camera failed. Re-seat its USB cable (the wrist camera's cable gets tugged by the arm),\n"
            "then run `python scripts/find_hardware.py` to confirm the indices haven't changed."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
