# 00 · Environment setup on Windows

**English** · [简体中文](00_environment.zh-CN.md)

Goal: LeRobot 0.7.0 and PyTorch with CUDA on **native Windows 11**, with direct access to the servo serial ports and the USB cameras.

## Why not WSL2 / a VM / ROS

Robot projects usually assume Linux, but what we actually need is just three things: serial ports, cameras and a GPU.
So first check which option gives us all three:

| Option | Servo serial port | USB camera | CUDA training |
|---|---|---|---|
| **Native Windows** ✅ | COM ports work directly | OpenCV works directly | yes |
| WSL2 | must be attached by hand with `usbipd-win` | the default kernel has no UVC driver, you build your own | yes |
| VMware VM | USB pass-through works | yes | **no** (the VM cannot see the GPU) |

Only native Windows gets all three without extra work. And LeRobot itself **does not depend on ROS at all**: servos are
read and written over pyserial, cameras go through OpenCV, datasets are Parquet + MP4, training is plain PyTorch.
So having no ROS on Windows costs us nothing.

## 1. Create the conda environment

LeRobot 0.7.0 needs **Python ≥ 3.12**. `ffmpeg` has to come from conda-forge into the environment, because TorchCodec
loads its DLLs to decode videos.

```powershell
conda create -y -n lerobot -c conda-forge --override-channels python=3.12 ffmpeg=7.1.1
conda activate lerobot
```

PS: `CondaHTTPError: HTTP 000 CONNECTION FAILED` is just a network hiccup. Run it again.

## 2. Install PyTorch with CUDA

Install **PyTorch first, then LeRobot**. The other way round, pip pulls in a CPU-only PyTorch on the way.
The laptop has an RTX 3070 with driver 596.x, so we use the CUDA 12.8 build:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install LeRobot (pinned)

```powershell
pip install "lerobot[core-scripts,training,feetech,smolvla,pi,diffusion]==0.7.0"
```

| extra | What it brings |
|---|---|
| `core-scripts` | the CLI tools (`lerobot-record`, `lerobot-teleoperate`, ...) and what they need: datasets, visualisation, keyboard listener |
| `training` | `lerobot-train`, accelerate, trackio (training curves) |
| `feetech` | serial driver for the Feetech servos (STS3215 on the SO-101) |
| `diffusion` | `diffusers`, for Diffusion Policy |
| `smolvla` / `pi` | `transformers` and friends, for SmolVLA and π₀.₅ |

PS: the version is pinned to `==0.7.0` for reproducibility. LeRobot moves fast, and CLI flags change even between minor versions.

## 4. Check

```powershell
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

`True` and `0.7.0` means you are good.

## 5. Log in to Hugging Face

Datasets and models live on the Hugging Face Hub. Create a token with **write** access at
<https://huggingface.co/settings/tokens>, then:

```powershell
hf auth login
hf auth whoami
```

Put your user name in `hf_user` in `config/robot.json`. **Never write the token into any file in the repo.**

PS: run `hf auth login` inside the `lerobot` environment. From conda's base environment you get a different `hf`.

## 6. Where to put the project

**Not in a synced folder like OneDrive.** Each training checkpoint is hundreds of MB to several GB; the sync client
uploads them while training runs, and can lock a file while it is being written, which crashes the run.
If you want a backup, copy it by hand after training.

`scripts/env.ps1` makes LeRobot keep datasets in the project's `data/` and calibration files in `config/calibration/`
(by default both go to the global cache `$HF_HOME/lerobot`). Downloaded pretrained models stay in the global cache,
since they are shared by every project.

## 7. Get the code

Code and docs are on GitHub; datasets and models are on the Hugging Face Hub (GitHub caps a single file at 100 MB).

```powershell
git clone https://github.com/Yufeng1216/Lerobot_so101.git
cd Lerobot_so101
copy config\robot.example.json config\robot.json   # then fill in your COM ports, camera indices and HF user
```

PS: Git for Windows ships Git Credential Manager. The first `git push` opens a browser to authorise; afterwards it
lands on a blank `127.0.0.1:xxxx/?code=...` page. That is expected, just close it.
