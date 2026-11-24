"""Quality report for a recorded LeRobot dataset (read-only).

For every episode it prints duration, how much of it the arm was idle, how long you
waited before moving and after finishing, and how wide the gripper opened. Episodes
that look broken (almost no motion, or never closed the gripper) are flagged.

Optionally saves a contact sheet per episode (6 evenly spaced frames from each camera)
to outputs/dataset_check/<dataset>/ so you can eyeball success/failure quickly.

Usage (inside the `lerobot` conda env):
    python scripts/inspect_dataset.py Yufeng1216/so101_pick_place
    python scripts/inspect_dataset.py Yufeng1216/so101_pick_place --frames      # also save contact sheets
    python scripts/inspect_dataset.py Yufeng1216/so101_pick_place --frames --episodes 0 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

IDLE_THRESHOLD = 0.3  # sum over joints of |action change| per frame, in normalized units
GRIPPER = 5  # index of the gripper in the 6-dim action vector (0 = closed, 100 = open)


def lerobot_home() -> Path:
    # Same rule as scripts/env.ps1: datasets live in <repo>/data unless HF_LEROBOT_HOME says otherwise.
    if os.getenv("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"])
    return Path(__file__).resolve().parent.parent / "data"


def load_frames_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def report(df: pd.DataFrame, fps: int) -> None:
    print(f"{'ep':>3} {'len':>6} {'idle':>5} {'wait@start':>10} {'wait@end':>8} {'grip open':>9}  flags")
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].values)
        speed = np.abs(np.diff(a, axis=0)).sum(axis=1)
        moving = np.flatnonzero(speed > IDLE_THRESHOLD)
        idle = (speed <= IDLE_THRESHOLD).mean()
        start = moving[0] / fps if len(moving) else float("nan")
        end = (len(a) - 1 - moving[-1]) / fps if len(moving) else float("nan")
        grip_open = a[:, GRIPPER].max()

        flags = []
        if idle > 0.9:
            flags.append("NO MOTION - delete")
        if grip_open < 15:
            flags.append("gripper never opened")
        if start > 3:
            flags.append("slow start")
        if end > 3:
            flags.append("idle tail")
        print(
            f"{ep:>3} {len(a) / fps:>5.1f}s {idle:>5.0%} {start:>9.1f}s {end:>7.1f}s {grip_open:>9.0f}  {', '.join(flags)}"
        )


def contact_sheets(root: Path, df: pd.DataFrame, cameras: list[str], episodes: list[int] | None, out: Path) -> None:
    import av
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    # All episodes of a chunk live in one MP4 per camera; frame i of the file = global index i
    # for a single-file dataset. Decode each camera once.
    frames = {}
    for cam in cameras:
        video_files = sorted((root / "videos" / cam).rglob("*.mp4"))
        frames[cam] = [f.to_ndarray(format="bgr24") for vf in video_files for f in av.open(str(vf)).decode(video=0)]

    bounds = df.groupby("episode_index")["index"].agg(["min", "max"])
    for ep, (lo, hi) in bounds.iterrows():
        if episodes and ep not in episodes:
            continue
        picks = np.linspace(lo, hi, 6).astype(int)
        rows = []
        for cam in cameras:
            tiles = []
            for i in picks:
                tile = cv2.resize(frames[cam][i], (320, 240))
                label = f"{cam.split('.')[-1]} t={(i - lo) / 30:.0f}s"
                cv2.putText(tile, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
        path = out / f"episode_{ep:03d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
    print(f"\nContact sheets saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="e.g. Yufeng1216/so101_pick_place")
    parser.add_argument("--root", type=Path, help="dataset folder (default: $HF_LEROBOT_HOME/<repo_id>)")
    parser.add_argument("--frames", action="store_true", help="save per-episode contact sheets")
    parser.add_argument("--episodes", type=int, nargs="*", help="only these episodes for --frames")
    args = parser.parse_args()

    root = args.root or lerobot_home() / args.repo_id
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    print(f"{args.repo_id}: {info['total_episodes']} episodes, {info['total_frames']} frames @ {fps} fps")
    print(f"cameras: {cameras}\n")

    df = load_frames_table(root)
    report(df, fps)
    if args.frames:
        out = Path("outputs/dataset_check") / args.repo_id.replace("/", "__")
        contact_sheets(root, df, cameras, args.episodes, out)


if __name__ == "__main__":
    main()
