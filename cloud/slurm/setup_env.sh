#!/bin/bash
# One-time environment setup on a Linux Slurm cluster (run on a login node, ~15 min).
#
#   bash cloud/slurm/setup_env.sh
#
# Creates the same environment as the Windows laptop (docs/00_environment.md):
# conda env `lerobot`, Python 3.12, ffmpeg 7 from conda-forge, LeRobot 0.7.0 (pinned).
# Miniforge is installed in $HOME so nothing depends on the cluster's module versions.
set -euo pipefail

CONDA_DIR="$HOME/miniforge3"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo ">> Installing Miniforge into $CONDA_DIR"
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm /tmp/miniforge.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^lerobot "; then
    echo ">> Creating conda env lerobot"
    conda create -y -n lerobot python=3.12
fi
conda activate lerobot
conda install -y -c conda-forge "ffmpeg=7.1.1"      # FFmpeg libraries for video decoding
# PyTorch 2.11 (same as the laptop) built for CUDA 12.6. Not cu128: those wheels dropped Volta
# (sm_70), so on V100 GPUs every CUDA op fails with "no kernel image is available".
# Not the default PyPI wheel either: a CUDA 13 build that needs a newer GPU driver.
pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0+cu126" "torchvision==0.26.0+cu126"
# Extras: training (dataset + video decoding + Trackio), diffusion (diffusers), smolvla (transformers).
pip install "lerobot[training,diffusion,smolvla]==0.7.0"
# The Linux torchcodec 0.17 wheel is built for CUDA 13 (needs libnvrtc.so.13) and cannot load
# next to torch cu128. Remove it: LeRobot then decodes videos with pyav (--dataset.video_backend=pyav).
pip uninstall -y torchcodec
# conda's ICU library (pulled in by ffmpeg) needs a newer C++ runtime than the cluster's
# /lib64/libstdc++.so.6. Install conda's own; the sbatch script puts it first on the library path.
conda install -y -c conda-forge "libstdcxx-ng>=13"

echo ">> Check:"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -c "import sqlite3, av, diffusers, transformers, lerobot, torch; import lerobot.scripts.lerobot_train; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda build', torch.version.cuda, '| av', av.__version__, '| train imports OK')"
echo ">> Done. GPU availability can only be checked inside a GPU job (login nodes have no GPU)."
