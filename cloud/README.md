# Remote training and inference

**English** · [简体中文](README.zh-CN.md)

Everything in `scripts/` runs on the Windows laptop (robot, recording, local training, deployment).
This folder holds what runs **elsewhere**, for policies that do not fit the laptop's 8 GB GPU.

| Folder | Where | Used for |
|---|---|---|
| `slurm/` | a Linux GPU cluster with the Slurm scheduler | Diffusion Policy at its default size (278 M params); SmolVLA at batch 32 |
| `hf_jobs/` | rented Hugging Face GPUs (billed per second) | π₀.₅ (4.1 B trainable params, needs an 80 GB GPU in float32) |

## `slurm/`

| File | Purpose |
|---|---|
| `setup_env.sh` | one-time environment install on the login node (same versions as the laptop) |
| `site.env.example` | template for your account / partition / time limit; copy to `site.env` (git-ignored) |
| `submit.sh` | checks the account against the allow-list and submits `train.sbatch` |
| `train.sbatch` | the job: activates the env, downloads the dataset, runs `lerobot-train` with the policy preset; resumes automatically when re-submitted |

```bash
# on your laptop, from the repository root
scp -r cloud <user>@<cluster-login-host>:~/Lerobot/

# on the cluster
cd ~/Lerobot
bash cloud/slurm/setup_env.sh                      # once
cp cloud/slurm/site.env.example cloud/slurm/site.env && nano cloud/slurm/site.env
bash cloud/slurm/submit.sh diffusion             # or: smolvla
squeue -u $USER
tail -f logs/diffusion_<jobid>.log
```

When the job reaches its time limit, run the same `submit.sh` command again: it continues
from the last checkpoint. When training is done, copy the model back to the laptop for deployment:

| Policy | Folder to copy |
|---|---|
| diffusion | `outputs/train/diffusion_so101_pick_place/checkpoints/last/pretrained_model_ema/` |
| smolvla | `outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model/` |

SmolVLA downloads `lerobot/smolvla_base` (about 2 GB) from the Hub on first use, so the compute
node needs internet access.

Notes on the Linux environment (all handled in the scripts):

- PyTorch is the CUDA 12.6 build: the CUDA 12.8 builds no longer include kernels for V100 GPUs.
- Videos are decoded with PyAV (`--dataset.video_backend=pyav`); torchcodec's Linux wheel is built for CUDA 13.
- `LD_LIBRARY_PATH` puts conda's C++ runtime before the system one.

## `hf_jobs/`

`train_pi05.py` submits a π₀.₅ fine-tuning run to [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
from the Windows laptop, using LeRobot's built-in `--job.target=<hardware>`.

```bash
python cloud/hf_jobs/train_pi05.py --dry-run                     # print the job config, submit nothing
python cloud/hf_jobs/train_pi05.py --steps 200 --timeout 40m     # speed / memory test
python cloud/hf_jobs/train_pi05.py --steps 3000 --timeout 9h     # the run used here
python cloud/hf_jobs/serve_pi05.py --timeout 4h                  # evaluation: π₀.₅ policy server on an a10g-large, prints the SSH tunnel
```

Before the first run:

- a Hugging Face token with **write** access (fine-grained tokens need the Jobs permission), and credit on the account;
- accept the license of [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224): π₀.₅ uses its tokenizer.

Measured on `a100-large` ($2.50/h): ~8.3 s/step at batch 32 (float32 full fine-tune, gradient checkpointing),
66.9 GB GPU memory. The default 30k steps would take ~70 h, so this project trains 3,000 steps with the
learning-rate schedule shortened to match; everything else is the LeRobot default. `--timeout` does not count
time spent queuing for a GPU.

On Windows, LeRobot writes the model id into the job config as `lerobot\pi05_base`; the script restores the
forward slash before the config is uploaded, otherwise the Linux job cannot find the model.

The 3,000-step run finished in 7 h 6 min (≈ $18), final loss 0.008; the model was pushed to the repo the job created, later renamed and made public as
[`Yufeng1216/pi05_so101_pick_place`](https://huggingface.co/Yufeng1216/pi05_so101_pick_place). Training curve, download command and the deployment plan (the model does not fit the laptop
GPU, so for the evaluation `serve_pi05.py` runs `lerobot-policy-server` on an HF Jobs A10G, reached through an SSH tunnel) are in [docs/07_pi05.md](../docs/07_pi05.md).

`hf jobs ps` / `hf jobs logs` print characters the Windows GBK console cannot encode; set `PYTHONUTF8=1` first
(`scripts/env.ps1` does).
