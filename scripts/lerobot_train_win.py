"""`lerobot-train`, made to work for a normal (non-admin) Windows user.

Two Windows-only problems in LeRobot 0.7.0, both patched at start-up (LeRobot itself is
not modified; everything else is the stock `lerobot-train`):

1. Checkpoint link. After every checkpoint, LeRobot points `checkpoints/last` at the
   newest checkpoint with a *symbolic link*. On Windows, creating symlinks needs admin
   rights or Developer Mode; without them the run crashes with `WinError 1314` at the
   FIRST checkpoint save - typically hours into training.
   Fix: make `last` a *directory junction* instead. It behaves like a symlink for
   directories but needs no special privilege.

2. Hub model names. `--policy.path=lerobot/smolvla_base` is stored as a `Path`, and on
   Windows `str(Path("lerobot/smolvla_base"))` is `lerobot\\smolvla_base`, which the Hub
   rejects ("Repo id must use alphanumeric chars ..."). Fine-tuning any pretrained policy
   from the Hub fails. Fix: if the name is not an existing local folder, turn the
   backslashes back into slashes before loading.

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


def _fix_hub_id(name_or_path):
    """'lerobot\\smolvla_base' -> 'lerobot/smolvla_base', unless it is a real local path."""
    name = str(name_or_path)
    if "\\" in name and not os.path.exists(name):
        return name.replace("\\", "/")
    return name_or_path


def hub_safe_from_pretrained(original):
    """Wrap a `from_pretrained` classmethod so Hub ids keep their forward slash on Windows.

    The name arrives either positionally or as a keyword (the policy and the processor
    pipeline call it differently), so both are fixed.
    """

    def from_pretrained(cls, *args, **kwargs):
        if args:
            args = (_fix_hub_id(args[0]), *args[1:])
        for key in ("pretrained_name_or_path", "pretrained_model_name_or_path"):
            if key in kwargs:
                kwargs[key] = _fix_hub_id(kwargs[key])
        return original(cls, *args, **kwargs)

    return classmethod(from_pretrained)


if sys.platform == "win32":
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    # (1) Patch both the defining module and the name lerobot_train imported from it.
    train_utils.update_last_checkpoint = update_last_checkpoint_junction
    lerobot_train.update_last_checkpoint = update_last_checkpoint_junction
    # (2) Both loaders that read a pretrained policy from the Hub: the weights
    # (PreTrainedPolicy, inherited by SmolVLA) and the pre/post-processing pipelines
    # (DataProcessorPipeline; PolicyProcessorPipeline is an alias of it).
    for klass in (PreTrainedPolicy, DataProcessorPipeline):
        klass.from_pretrained = hub_safe_from_pretrained(klass.__dict__["from_pretrained"].__func__)


if __name__ == "__main__":
    lerobot_train.main()
