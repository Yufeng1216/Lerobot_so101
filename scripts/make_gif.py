"""Turn one recorded episode (e.g. an evaluation trial) into a GIF for the README.

Both cameras side by side (top | wrist), sped up, at a small size so the README stays light.
LeRobot stores many episodes in one MP4 per camera; the episode's start/end time inside that
file comes from the dataset's meta/episodes table, and ffmpeg cuts that slice out.

Usage (inside the `lerobot` conda env):
    python scripts/make_gif.py --repo-id Yufeng1216/rollout_eval_act --episode 18 --out assets/act.gif
Episodes are numbered from 0 (trial #19 in results/act.md is --episode 18).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("top", "wrist")


def episode_clip(ds: Path, episode: int, camera: str) -> tuple[Path, float, float]:
    """(video file, start s, end s) of `episode` for one camera."""
    meta = pd.concat(pd.read_parquet(f) for f in sorted((ds / "meta" / "episodes").rglob("*.parquet")))
    row = meta[meta["episode_index"] == episode].iloc[0]
    key = f"videos/observation.images.{camera}"
    chunk, file = int(row[f"{key}/chunk_index"]), int(row[f"{key}/file_index"])
    path = ds / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    return path, float(row[f"{key}/from_timestamp"]), float(row[f"{key}/to_timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed-up")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    parser.add_argument("--height", type=int, default=240, help="height of each camera view in pixels")
    args = parser.parse_args()

    ds = ROOT / "data" / args.repo_id
    inputs, labels = [], []
    for i, cam in enumerate(CAMERAS):
        path, start, end = episode_clip(ds, args.episode, cam)
        inputs += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(path)]
        # Speed up, drop to the GIF frame rate, shrink.
        labels.append(f"[{i}:v]setpts=PTS/{args.speed},fps={args.fps},scale=-2:{args.height}[v{i}]")

    # GIFs only have 256 colours: build a palette from the clip first, then use it (much less banding).
    graph = ";".join(labels) + (
        ";[v0][v1]hstack=inputs=2,split[a][b];[a]palettegen=max_colors=128[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-loop", "0", str(args.out)],
                   check=True)
    print(f"saved {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
