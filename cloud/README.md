# Remote training

Everything in `scripts/` runs on the Windows laptop (robot, recording, local training, deployment).
This folder holds what runs **elsewhere**, for policies that do not fit the laptop's 8 GB GPU.

| Folder | Where | Used for |
|---|---|---|
| `slurm/` | a Linux GPU cluster with the Slurm scheduler | Diffusion Policy at its default size (278 M params); SmolVLA at batch 32 |

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
