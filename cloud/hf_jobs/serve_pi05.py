"""Serve pi0.5 from a Hugging Face Jobs GPU for real-robot evaluation (remote inference).

The laptop's 8 GB GPU can't hold pi0.5, so the policy runs in `lerobot-policy-server` on an HF Jobs
GPU, and the laptop drives the arm with `robot.ps1 eval -Remote`. The job is started with SSH enabled;
an SSH tunnel maps the server's port 7447 inside the job to 127.0.0.1:7447 on the laptop, so nothing
is exposed publicly and the traffic is encrypted.

Usage (inside the `lerobot` conda env, from the repo root, after `hf auth login` and with your SSH
public key added at https://huggingface.co/settings/keys):
    python cloud/hf_jobs/serve_pi05.py --dry-run          # print the job command, submit nothing
    python cloud/hf_jobs/serve_pi05.py --timeout 4h       # start the server, print the tunnel command
    hf jobs cancel <job_id>                               # when done: billing stops

Flavor: a10g-large (24 GB A10G, 46 GB RAM, $1.50/h). The float32 checkpoint is 16.6 GB, so it fits on
the 24 GB GPU unchanged (same precision as training); a10g-small has only 15 GB of RAM to load it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "Yufeng1216/pi05_so101_pick_place"
MODEL_DIR = "/tmp/pi05"
PORT = 7447


def server_config() -> str:
    """The same config scripts/make_server_config.py writes, with the model path inside the job."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pi05.json"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_server_config.py"),
             str(ROOT / "outputs/train/pi05_so101_pick_place/pretrained_model"),
             "--dataset", str(ROOT / "data/Yufeng1216/so101_pick_place"),
             "--model-path", MODEL_DIR, "--port", str(PORT), "--out", str(out)],
            check=True,
        )
        return out.read_text(encoding="utf-8")


# Runs inside the job. The image's own LeRobot may be newer; the remote-inference protocol is
# versioned, so install the laptop's exact version (0.7.0) first.
JOB_SCRIPT = f"""
set -euo pipefail
PY="$(command -v python)"
if command -v uv >/dev/null; then uv pip install --python "$PY" -q "lerobot[pi,remote]==0.7.0";
else "$PY" -m pip install -q "lerobot[pi,remote]==0.7.0"; fi
"$PY" -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
hf download {MODEL_REPO} --local-dir {MODEL_DIR} --quiet
printf '%s' "$SERVER_CONFIG" > /tmp/pi05.json
printf '%s' "$LAUNCHER" > /tmp/pi05_server_launch.py
exec "$PY" /tmp/pi05_server_launch.py --config_path=/tmp/pi05.json
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flavor", default="a10g-large", help="HF Jobs hardware, see `hf jobs hardware`")
    ap.add_argument("--timeout", default="4h", help="hard stop for the job (billing stops too)")
    ap.add_argument("--image", default="huggingface/lerobot-gpu:latest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = server_config()
    if args.dry_run:
        print(f"image={args.image} flavor={args.flavor} timeout={args.timeout}\n{JOB_SCRIPT}\nSERVER_CONFIG={cfg}")
        return

    api = HfApi()
    job = api.run_job(
        image=args.image,
        command=["bash", "-c", JOB_SCRIPT],
        env={"SERVER_CONFIG": cfg, "PYTHONUNBUFFERED": "1",
             # lerobot-policy-server plus pi0.5's missing input check (see pi05_server_launch.py)
             "LAUNCHER": (Path(__file__).parent / "pi05_server_launch.py").read_text(encoding="utf-8")},
        flavor=args.flavor,
        timeout=args.timeout,
        name="pi05_policy_server",
        ssh=True,
    )
    print(f"job {job.id} submitted ({args.flavor}, timeout {args.timeout})")
    print(f"logs:   hf jobs logs {job.id}")
    print(f"stop:   hf jobs cancel {job.id}")

    # Wait until the container runs, then print the tunnel command (the server itself needs a few
    # more minutes to install, download 16.6 GB and warm up; watch the logs for "Policy server ready").
    while True:
        info = api.inspect_job(job_id=job.id)
        stage = str(info.status.stage)
        if info.status.ssh_url and "RUNNING" in stage:
            break
        if any(s in stage for s in ("COMPLETED", "ERROR", "CANCELED", "DELETED")):
            raise SystemExit(f"job ended before running: {stage} {info.status.message or ''}")
        time.sleep(10)
    u = urlsplit(info.status.ssh_url)
    port = f" -p {u.port}" if u.port else ""
    print("\ntunnel (keep this window open while evaluating):")
    print(f"  ssh -N -L {PORT}:127.0.0.1:{PORT}{port} {u.username}@{u.hostname}")


if __name__ == "__main__":
    main()
