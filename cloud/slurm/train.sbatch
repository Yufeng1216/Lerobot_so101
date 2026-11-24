#!/bin/bash
# One Slurm training job. Submit it through submit.sh (which sets account, partition, time):
#   bash cloud/slurm/submit.sh diffusion
#   squeue -u $USER                       # job state (PD = waiting, R = running)
#   tail -f logs/<policy>_<jobid>.log
#
# The settings mirror scripts/train.ps1 (the laptop presets), except where the larger GPU allows
# the paper's defaults. Re-submitting the same policy after a time-out continues from the last
# checkpoint.
#
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail
POLICY="${1:?policy name missing}"

# Second safety net (submit.sh already checked): only the allowed allocation may run this.
case "${SLURM_JOB_ACCOUNT:-}" in
    "${ALLOWED_ACCOUNT_PREFIX:?not set: submit through cloud/slurm/submit.sh}"*) ;;
    *) echo "ERROR: Slurm account '${SLURM_JOB_ACCOUNT:-}' is not allowed for this project." >&2; exit 1 ;;
esac
echo ">> Slurm account: $SLURM_JOB_ACCOUNT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lerobot
# Use conda's C++ runtime before the system's /lib64 one (see setup_env.sh).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# The dataset is public on the Hub and is downloaded on first use into the working directory.
export HF_LEROBOT_HOME="$PWD/data"

RUN="${POLICY}_so101_pick_place"
OUT="outputs/train/$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -d "$OUT/checkpoints/last" ]; then
    echo ">> Resuming $RUN"
    srun lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true
    exit
fi

COMMON=(
    --dataset.repo_id=Yufeng1216/so101_pick_place
    --dataset.eval_split=0.1
    --dataset.video_backend=pyav        # torchcodec's Linux wheel needs CUDA 13; see setup_env.sh
    --eval_steps=2000
    --output_dir="$OUT"
    --job_name="$RUN"
    --policy.device=cuda
    --policy.repo_id="Yufeng1216/$RUN"
    --policy.push_to_hub=false          # no Hub login on the cluster; copy the model back instead
    --num_workers=8
    --log_freq=100
)

case "$POLICY" in
    diffusion)
        # Paper-default U-Net (down_dims 512/1024/2048, 278M params): needs > 8 GB, fits a 16 GB V100.
        PRESET=(
            --policy.type=diffusion
            --policy.resize_shape=[240,320]
            --policy.crop_ratio=0.95
            --ema.type=warmup
            --batch_size=8
            --steps=100000
            --save_freq=20000
        )
        ;;
    smolvla)
        # Fine-tune the pretrained SmolVLA (450M, 100M trainable). Batch 32 overflows 8 GB but fits
        # 16 GB. The rename_map maps our camera names onto the ones smolvla_base was trained with.
        PRESET=(
            --policy.path=lerobot/smolvla_base
            '--rename_map={"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
            --batch_size=32
            --steps=20000
            --policy.scheduler_decay_steps=20000
            --save_freq=5000                # short gaps: a time-out loses at most 5k steps
        )
        ;;
    *)
        echo "Unknown policy '$POLICY'. Known: diffusion, smolvla" >&2
        exit 1
        ;;
esac

srun lerobot-train "${COMMON[@]}" "${PRESET[@]}"
