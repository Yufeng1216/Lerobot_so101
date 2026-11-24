"""`lerobot-train`, made to work for a normal (non-admin) Windows user.

The problem: after every checkpoint, LeRobot points `checkpoints/last` at the newest
checkpoint with a *symbolic link*. On Windows, creating symlinks needs admin rights or
Developer Mode; without them the run crashes with `WinError 1314` at the FIRST
checkpoint save - typically hours into training.

The fix: on Windows, make `last` a *directory junction* instead. A junction behaves like
a symlink for directories (opening `checkpoints/last/...` lands in the newest step) but
needs no special privilege. Everything else is the stock `lerobot-train`.

Usage: exactly like lerobot-train (scripts/train.ps1 calls it for you):
    python scripts/lerobot_train_win.py --policy.type=act --dataset.repo_id=...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import lerobot.common.train_utils as train_utils
import lerobot.scripts.lerobot_train as lerobot_train


def update_last_checkpoint_junction(checkpoint_dir: Path) -> None:
    """Drop-in replacement for lerobot.common.train_utils.update_last_checkpoint."""
    import _winapi  # Windows-only stdlib module

    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if last.is_junction() or last.is_symlink():
        os.rmdir(last)  # removes the link itself, never the checkpoint it points to
    # Junction targets must be absolute; the checkpoint folders sit next to `last`.
    _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last))


if sys.platform == "win32":
    # Patch both the defining module and the name lerobot_train imported from it.
    train_utils.update_last_checkpoint = update_last_checkpoint_junction
    lerobot_train.update_last_checkpoint = update_last_checkpoint_junction


if __name__ == "__main__":
    lerobot_train.main()
