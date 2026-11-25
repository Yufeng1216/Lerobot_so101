"""Fit the pixel -> table mappings from the recorded demos (no calibration session needed).

A homography H maps the overhead image to the table plane: [x, y, 1] ~ H @ [u, v, 1]. It needs
pairs of (pixel, table point). The 50 demos already contain them:
  * pixel: in the first overhead frame of each demo, the blue toy is the only blue thing on the
    white table, so a colour threshold gives its centre (u, v);
  * table point: at the moment the gripper closes on the toy, forward kinematics of the recorded
    joint angles gives the gripper tip (x, y) in the arm's base frame.
The toy has not moved between those two moments, so each demo is one correspondence. RANSAC drops
the few where the grasp detection or the colour mask is off.

Two mappings are saved:
  * H_pixel_to_table: from the grasp pairs only. Accurate (~1 cm) where the toy can be, but those
    points all lie in one 10 x 10 cm patch, and far from it the perspective terms blow up: at the
    box it was off by ~24 cm. Used for the toy.
  * H_pixel_to_place: grasp pairs + release pairs (the frame where the gripper opens over the box:
    toy pixel in that frame <-> tip (x, y) by FK). The toy is ~9 cm above the table then, so these
    points carry a few cm of parallax, but they pin the mapping down over the box. Used for the box,
    where a few cm do not matter (the box opening is ~20 cm wide).

Output: zero_shot/homography.json. The colour centres are also the ground truth for the VLM test.

Usage:  python -m zero_shot.calibrate_from_demos
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
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


def frame_at(episodes: pd.DataFrame, ep: int, index: int, camera: str = "top") -> np.ndarray | None:
    """Overhead frame `index` (0-based) of episode `ep`, decoded from the dataset MP4 (BGR)."""
    key = f"observation.images.{camera}"
    row = episodes.loc[ep]
    path = (DATA / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}"
            / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
    t = float(row[f"videos/{key}/from_timestamp"]) + index / 30
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t - 1, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time >= t - 1e-3:
                return frame.to_ndarray(format="bgr24")
    return None


def apply(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.c_[uv, np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:]


def main() -> None:
    df = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "data").rglob("*.parquet")))
    episodes = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "meta" / "episodes").rglob("*.parquet")))
    episodes = episodes.set_index("episode_index")
    grasp, release = [], []
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
            grasp.append((*uv, *tip[:2]))
        # release: first frame after the grasp where the gripper opens again (+10 over its closed value)
        held = state[i + 20, 5]
        opened = i + 20 + np.flatnonzero(state[i + 20:, 5] > held + 10)
        if len(opened):
            frame = frame_at(episodes, int(ep), int(opened[0]))
            uv = toy_pixel(frame) if frame is not None else None
            if uv is not None:
                release.append((*uv, *fk(state[opened[0], :5])[:2, 3]))
    grasp, release = np.array(grasp), np.array(release)

    H, inl = cv2.findHomography(grasp[:, :2], grasp[:, 2:], cv2.RANSAC, ransacReprojThreshold=0.015)
    ok = inl.ravel() == 1
    err = np.linalg.norm(apply(H, grasp[:, :2]) - grasp[:, 2:], axis=1)
    print(f"toy mapping: {len(grasp)} grasp pairs, {ok.sum()} inliers, error median "
          f"{np.median(err[ok]) * 100:.1f} cm, max {err[ok].max() * 100:.1f} cm")
    both = np.r_[grasp, release]
    Hp, inl_p = cv2.findHomography(both[:, :2], both[:, 2:], cv2.RANSAC, ransacReprojThreshold=0.02)
    err_r = np.linalg.norm(apply(Hp, release[:, :2]) - release[:, 2:], axis=1)
    err_r_old = np.linalg.norm(apply(H, release[:, :2]) - release[:, 2:], axis=1)
    print(f"place mapping: + {len(release)} release pairs, {int(inl_p.sum())} inliers; at the release point the "
          f"error is median {np.median(err_r) * 100:.1f} cm (toy mapping there: {np.median(err_r_old) * 100:.1f} cm)")
    OUT.write_text(json.dumps({
        "H_pixel_to_table": H.tolist(),
        "H_pixel_to_place": Hp.tolist(),
        "image_size": [640, 480],
        "grasp_pairs": len(grasp), "grasp_inliers": int(ok.sum()), "median_error_m": float(np.median(err[ok])),
        "release_pairs": len(release), "place_median_error_at_release_m": float(np.median(err_r)),
        "note": "toy: demo first frames (blue-blob centre) <-> FK at the grasp; place: + release frames <-> FK",
    }, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
