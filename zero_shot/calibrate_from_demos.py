"""Fit the pixel -> table homography from the recorded demos (no calibration session needed).

A homography H maps the overhead image to the table plane: [x, y, 1] ~ H @ [u, v, 1]. It needs
pairs of (pixel, table point). The 50 demos already contain them:
  * pixel: in the first overhead frame of each demo, the blue toy is the only blue thing on the
    white table, so a colour threshold gives its centre (u, v);
  * table point: at the moment the gripper closes on the toy, forward kinematics of the recorded
    joint angles gives the gripper tip (x, y) in the arm's base frame.
The toy has not moved between those two moments, so each demo is one correspondence. RANSAC drops
the few where the grasp detection or the colour mask is off.

Output: zero_shot/homography.json (H, plus the fit error). Also used as ground truth for the VLM:
the colour centres are where the VLM should point.

Usage:  python -m zero_shot.calibrate_from_demos
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zero_shot.check_fk import DATA, grasp_frame
from zero_shot.kinematics import fk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from align_camera import reference_frame  # noqa: E402  (first frame of an episode, BGR)

OUT = Path(__file__).with_name("homography.json")


def toy_pixel(bgr: np.ndarray) -> tuple[float, float] | None:
    """Centre of the largest saturated-blue blob (the toy), or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 120, 50), (130, 255, 255))   # OpenCV hue 100-130 = blue
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (float(centroids[i][0]), float(centroids[i][1])) if stats[i, cv2.CC_STAT_AREA] > 300 else None


def main() -> None:
    df = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "data").rglob("*.parquet")))
    pairs = []
    for ep, g in df.groupby("episode_index"):
        state = np.stack(g["observation.state"].to_numpy())
        i = grasp_frame(state[:, 5])
        if i is None:
            continue
        tip = fk(state[i, :5])[:3, 3]
        if tip[2] > 0.04:          # closed over the box, not on the table (see check_fk.py)
            continue
        uv = toy_pixel(reference_frame("Yufeng1216/so101_pick_place", "top", int(ep)))
        if uv is not None:
            pairs.append((int(ep), *uv, *tip[:2]))
    p = np.array(pairs)
    H, inliers = cv2.findHomography(p[:, 1:3], p[:, 3:5], cv2.RANSAC, ransacReprojThreshold=0.015)
    pred = cv2.perspectiveTransform(p[:, None, 1:3], H)[:, 0]
    err = np.linalg.norm(pred - p[:, 3:5], axis=1)
    ok = inliers.ravel() == 1
    print(f"{len(p)} pixel/table pairs, {ok.sum()} RANSAC inliers")
    print(f"table-position error from pixels: median {np.median(err[ok]) * 100:.1f} cm, max {err[ok].max() * 100:.1f} cm (inliers)")
    OUT.write_text(json.dumps({
        "H_pixel_to_table": H.tolist(),
        "image_size": [640, 480],
        "pairs": len(p), "inliers": int(ok.sum()),
        "median_error_m": float(np.median(err[ok])),
        "note": "fitted from demo first frames (blue-blob centre) and FK at the grasp moment",
    }, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
