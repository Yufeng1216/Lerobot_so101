"""Fine-tune pi0.5 on Hugging Face Jobs (rented cloud GPU), submitted from the Windows laptop.

pi0.5 (~3.6 B params) needs far more than 8 GB, so it trains on an HF Jobs GPU. LeRobot can
submit a training run there by itself (`--job.target=<hardware>`): it writes the training config
to a private Hub repo, starts a job that runs `lerobot-train` in the official LeRobot GPU image,
and the finished model is pushed to that repo. Costs are billed per second to the HF account.

The arguments are built here in Python rather than in PowerShell because the camera rename map
is JSON, and Windows PowerShell 5.1 strips the quotes inside it.

Usage (inside the `lerobot` conda env, from the repo root, after `hf auth login`):
    python cloud/hf_jobs/train_pi05.py --dry-run                      # print the job config, submit nothing
    python cloud/hf_jobs/train_pi05.py --steps 200 --timeout 40m      # speed / memory test
    python cloud/hf_jobs/train_pi05.py --steps 3000 --timeout 9h      # full run (~8.3 s/step on A100)

Measured on a100-large (batch 32, float32 full fine-tune, gradient checkpointing): ~8.3-8.4 s/step,
66.9 GB GPU memory, 4.1 B trainable parameters. Pricing: `hf jobs hardware`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import lerobot_train_win  # noqa: E402  (applies the Windows Hub-id fix before the config is parsed)
import lerobot.scripts.lerobot_train as lerobot_train  # noqa: E402
from lerobot.jobs import hf as hf_jobs  # noqa: E402

# Our camera names -> the ones pi05_base was pretrained with (the third camera is left empty).
RENAME = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


def _fix_windows_paths(original):
    """On Windows the Hub id is serialized as `lerobot\\pi05_base`; the Linux job would reject it.

    Wraps build_remote_config_file so the JSON uploaded for the job has the forward slash back.
    """

    def build(cfg, repo_id, dest, tags=None):
        path = original(cfg, repo_id, dest, tags=tags)
        data = json.loads(path.read_text())
        for parent, key in ((data.get("policy", {}), "pretrained_path"), (data, "output_dir")):
            if isinstance(parent.get(key), str):
                parent[key] = parent[key].replace("\\", "/")
        path.write_text(json.dumps(data, indent=4))
        return path

    return build


hf_jobs.build_remote_config_file = _fix_windows_paths(hf_jobs.build_remote_config_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--flavor", default="a100-large", help="HF Jobs hardware, see `hf jobs hardware`")
    ap.add_argument("--timeout", default="40m", help="hard stop for the job (billing stops too)")
    ap.add_argument("--num-workers", type=int, default=10,
                    help="data-loading processes; the a100-large job has 12 vCPUs (default 4 left the GPU waiting ~1.3 s/step)")
    ap.add_argument("--dry-run", action="store_true", help="print the config the job would get, submit nothing")
    args = ap.parse_args()

    sys.argv = [
        "lerobot-train",
        "--policy.path=lerobot/pi05_base",
        f"--rename_map={json.dumps(RENAME)}",
        "--dataset.repo_id=Yufeng1216/so101_pick_place",
        # The job pushes the model to a private repo <this name>_<UTC timestamp>.
        "--policy.repo_id=Yufeng1216/pi05_so101_pick_place",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--policy.scheduler_decay_steps={args.steps}",
        "--policy.gradient_checkpointing=true",   # trade compute for memory
        # One checkpoint, at the end: a full pi0.5 checkpoint with optimizer state is ~66 GB, and the
        # job pushes the final model to the Hub itself. Keep --timeout well above the expected run time.
        f"--save_freq={args.steps}",
        f"--num_workers={args.num_workers}",
        "--log_freq=10" if args.steps <= 500 else "--log_freq=100",
        "--eval_steps=0",                        # no held-out eval on the paid GPU
        "--job_name=pi05_so101_pick_place",
        f"--job.target={args.flavor}",
        f"--job.timeout={args.timeout}",
    ]

    if args.dry_run:
        def show(cfg, *a, **k):
            cfg.validate()  # loads pi05_base's config, as the real submit does
            with tempfile.TemporaryDirectory() as tmp:
                path = hf_jobs.build_remote_config_file(cfg, "Yufeng1216/DRY_RUN", Path(tmp) / "c.json")
                print(path.read_text())
        lerobot_train.submit_to_hf = show

    lerobot_train.main()


if __name__ == "__main__":
    main()
