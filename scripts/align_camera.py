"""Live tool to put a camera back exactly where it was during recording.

If a camera gets bumped, the policy sees a shifted scene it was never trained on
(distribution shift). This shows, in one window, the recorded frame blended with the
live view, plus the estimated shift in pixels. Nudge the camera until the ghost image
of the fixed objects (box, table edge) disappears and the shift is close to 0.

The shift is estimated from ORB feature matches between the two images, so moving
things (the toy, the arm, your hand) slightly disturb it; trust the fixed objects in
the overlay first.

The overlay is shown in a Rerun window (the same viewer lerobot-record uses):
LeRobot installs the headless build of OpenCV, which cannot open windows itself.

Usage (inside the `lerobot` conda env; reads camera index from config/robot.json):
    python scripts/align_camera.py                 # top camera
    python scripts/align_camera.py --camera wrist
Press Ctrl+C in the terminal to quit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import rerun as rr

ROOT = Path(__file__).resolve().parent.parent


def reference_frame(repo_id: str, camera: str, episode: int) -> np.ndarray:
    """First frame of `episode` from the recorded dataset, decoded from its MP4."""
    ds = ROOT / "data" / repo_id
    key = f"observation.images.{camera}"
    episodes = pd.concat(pd.read_parquet(f) for f in sorted((ds / "meta" / "episodes").rglob("*.parquet")))
    row = episodes.iloc[episode]
    start = float(row[f"videos/{key}/from_timestamp"])
    chunk, file = int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"])
    path = ds / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    for frame in av.open(str(path)).decode(video=0):
        if frame.time is not None and frame.time >= start:
            return frame.to_ndarray(format="bgr24")
    raise RuntimeError(f"no frame at t={start} in {path}")


def estimate_shift(ref_gray: np.ndarray, live_gray: np.ndarray, orb, matcher) -> tuple[float, float] | None:
    """Median displacement (dx, dy) of RANSAC-inlier ORB matches, or None if too few."""
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(live_gray, None)
    if d1 is None or d2 is None:
        return None
    matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(matches) < 20:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    _, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if inliers is None or inliers.sum() < 15:
        return None
    delta = (p2 - p1)[inliers.ravel() == 1]
    dx, dy = np.median(delta, axis=0)
    return float(dx), float(dy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="top", help="camera name in config/robot.json")
    parser.add_argument("--repo-id", default="Yufeng1216/so101_pick_place")
    parser.add_argument("--episode", type=int, default=-1, help="reference episode (default: the last one)")
    args = parser.parse_args()

    cam_cfg = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))["cameras"][args.camera]
    ref = reference_frame(args.repo_id, args.camera, args.episode)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(cam_cfg["index_or_path"])
    if cam_cfg.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_cfg["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {cam_cfg['index_or_path']}: run scripts/find_hardware.py")

    orb = cv2.ORB_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rr.init(f"align_{args.camera}", spawn=True)  # opens the Rerun viewer
    print("Align until the ghost of the box disappears and the shift is ~0. Ctrl+C to quit.")
    try:
        while True:
            ok, live = cap.read()
            if not ok:
                continue
            overlay = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
            shift = estimate_shift(ref_gray, cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), orb, matcher)
            if shift is None:
                text, color = "shift: (not enough matches)", (0, 0, 255)
            else:
                dx, dy = shift
                good = abs(dx) <= 3 and abs(dy) <= 3
                text = f"shift dx={dx:+.1f}px dy={dy:+.1f}px {'OK' if good else ''}"
                color = (0, 255, 0) if good else (0, 200, 255)
            cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            print(f"\r{text:<45}", end="", flush=True)
            # Rerun expects RGB; OpenCV images are BGR.
            rr.log("overlay", rr.Image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))
            rr.log("live", rr.Image(cv2.cvtColor(live, cv2.COLOR_BGR2RGB)))
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
