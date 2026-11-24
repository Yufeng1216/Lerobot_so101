"""Measure how fast training batches can be loaded, for different DataLoader settings.

Training speed = max(data loading time, GPU time) per step. On this laptop the ACT smoke
test spent ~1.0 s/step waiting for data and only ~0.4 s computing, so data loading (CPU
video decoding) is the bottleneck. This script loads batches exactly like ACT training
(current frame of every camera + the next 100 actions) and reports batches/second for each
combination of worker count and video decoding backend.

Usage (inside the `lerobot` conda env):
    python scripts/bench_dataloader.py
    python scripts/bench_dataloader.py --workers 4 8 12 --backends torchcodec pyav --batches 60
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

# Same dataset location rule as scripts/env.ps1.
os.environ.setdefault("HF_LEROBOT_HOME", str(Path(__file__).resolve().parent.parent / "data"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def bench(repo_id: str, backend: str, workers: int, batch_size: int, n_batches: int, chunk: int) -> float:
    fps = 30
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [i / fps for i in range(chunk)]},  # ACT: 100-step action chunk
        video_backend=backend,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        pin_memory=True,
        multiprocessing_context="spawn" if workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(5):  # warm-up: worker start-up and first decodes
        next(it)
    start = time.perf_counter()
    for _ in range(n_batches):
        next(it)
    return n_batches / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="Yufeng1216/so101_pick_place")
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--backends", nargs="+", default=["torchcodec", "pyav"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--chunk", type=int, default=100)
    args = parser.parse_args()

    print(f"{'backend':<11} {'workers':>7} {'batches/s':>10} {'s/batch':>8}")
    for backend in args.backends:
        for workers in args.workers:
            rate = bench(args.repo_id, backend, workers, args.batch_size, args.batches, args.chunk)
            print(f"{backend:<11} {workers:>7} {rate:>10.2f} {1 / rate:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
