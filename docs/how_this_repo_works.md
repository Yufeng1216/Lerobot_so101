# How this repo works: what every file is for

**English** · [简体中文](how_this_repo_works.zh-CN.md)

Written for someone who knows deep learning but not the engineering tools. After this page you'll know why every file in the repo exists.

## 1. A mental model first

We all know the pipeline for training a model: **data → model → training → evaluation**. Robot imitation learning is exactly the same,
except the data isn't downloaded; **we record it ourselves with the arm**:

```
hand moves the leader ──teleoperation──▶ the follower follows
        │                                    │
        │ every 1/30 s we record:             │
        │   observation = two camera images + 6 joint angles (state)
        │   action      = the leader's 6 joint angles at that moment
        ▼
   dataset of (observation, action) sequences  ──▶  train a policy π(action | observation)
                                                          │
                                                          ▼
                               deploy: the policy reads cameras + joints and drives the follower
```

So the project is really **supervised learning (behavior cloning)**: images and state in, regress actions.
ACT, Diffusion Policy, SmolVLA and π₀.₅ differ only in "how to model p(action | observation)".
The engineering part (serial ports, cameras, command line) is just plumbing that moves data in and out; follow the tutorial.

## 2. Is Windows instead of Linux "not proper"?

No, for concrete reasons:

- LeRobot depends neither on ROS nor on anything Linux-only. Servos go over serial, cameras through OpenCV, training is PyTorch.
- On Windows all 12 servos read fine, the cameras capture, and the GPU runs CUDA.
- The only difference is the port name: `/dev/ttyACM0` on Linux, `COM3` on Windows. When the official docs say `/dev/tty...`, use the COM port.
- What matters is **understanding the policies, getting results and explaining the trade-offs**, not the operating system.

## 3. File types at a glance

### Two kinds first

- **Text files**: human-readable characters; open them in Notepad or VS Code and you can read and edit them.
  Code, configs and docs are text. git is good at them: it shows exactly "what changed on which line".
- **Binary files**: bytes meant for programs; in Notepad they look like garbage.
  Images, videos, tabular data and model weights are binary. They're usually big and **don't go into git**.

PS: Windows hides extensions by default. In File Explorer → View → Show, tick "File name extensions" and "Hidden items"
to see suffixes like `.py`, `.json` and dot-files like `.git`, `.gitignore`.

### A. Code and scripts (text, in git)

| File | What it is | How to use it | Example here |
|---|---|---|---|
| `.py` | Python source | run with `python scripts/xxx.py`; edit in VS Code | `scripts/scan_motors.py` |
| `.ps1` | PowerShell script: several commands packed into one, Windows' "batch file" | run in PowerShell with `.\scripts\robot.ps1 ...` | `scripts/robot.ps1`, `scripts/env.ps1` |
| `.sh` | bash script: the Linux equivalent of `.ps1` | run in a Linux terminal with `bash cloud/slurm/xxx.sh` | `cloud/slurm/setup_env.sh` |
| `.sbatch` | also a bash script; its leading `#SBATCH` lines request resources from the Slurm queue | submitted with `sbatch xxx.sbatch` (here via `submit.sh`) | `cloud/slurm/train.sbatch` |

### B. Configuration (text)

| File | What it is | Example here |
|---|---|---|
| `.json` | structured data, `{"key": value}`, almost the same as a Python dict. Programs read their settings from it | `config/robot.json` (COM ports, camera indices), `config/calibration/**/*.json` (calibration) |
| `.gitignore` | tells git **which files not to upload**: large files, secrets, machine-specific config. One rule per line; `data/` means the whole folder | repo root |
| `.gitattributes` | tells git **how to treat certain files**. Here it normalises line endings (CRLF on Windows, LF on Linux) | repo root |
| `.env` | one `NAME=value` per line, read by scripts as variables. Usually personal settings, so not in git; a `.example` template is committed instead | `cloud/slurm/site.env.example` |

### C. Documentation (text, in git)

| File | What it is | Example here |
|---|---|---|
| `.md` | Markdown (English `xxx.md`, Chinese `xxx.zh-CN.md`): `#` is a heading, `**bold**`, triple backticks wrap code, `\| a \| b \|` is a table. GitHub renders it as a web page; `Ctrl+Shift+V` previews it in VS Code | `README.md`, `docs/*.md` |
| `LICENSE` (no extension) | the open-source licence: whether and how others may use the code. MIT here (use it freely, keep the attribution) | repo root |

### D. Files inside the dataset (`data/Yufeng1216/so101_pick_place/`, not in git, a copy on the Hub)

| File | Type | What it is |
|---|---|---|
| `meta/info.json` | text | the dataset's "spec sheet": fps, total frames, total episodes, shape and dtype of each key, video codec |
| `meta/stats.json` | text | per-key statistics (mean, std, min, max, quantiles), used to normalise inputs and outputs in training |
| `meta/tasks.parquet` | binary | table of task strings (just one sentence here) |
| `meta/episodes/.../*.parquet` | binary | per-episode index: first and last frame, start and end time in the video |
| `data/.../*.parquet` | binary | **the core table**: one row per frame, columns `action`, `observation.state`, `timestamp`, `episode_index`, ... |
| `videos/<camera>/.../*.mp4` | binary | camera frames, AV1-encoded, several episodes per file |

`.parquet` is a **columnar table** format, smaller and faster than CSV. To peek inside:

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/Yufeng1216/so101_pick_place/data/chunk-000/file-000.parquet'); print(df.head()); print(df.columns)"
```

### E. Training outputs (`outputs/train/<run>/checkpoints/<step>/pretrained_model/`, not in git, final models on the Hub)

| File | Type | What it is |
|---|---|---|
| `model.safetensors` | binary | **the weights** (about 200 MB for ACT). Same role as PyTorch's `.pt`, but only stores tensors and can't hide executable code, so it's safer and loads faster |
| `config.json` | text | policy architecture: chunk_size, layers, input/output features. Read first to build the network when loading |
| `train_config.json` | text | **all** the settings of this run (dataset, steps, learning rate, ...). Reproducible, and what `-Resume` reads |
| `policy_preprocessor.json` and `*_normalizer_processor.safetensors` | text + binary | the input preprocessing steps and the means/variances used to normalise (from `stats.json`). Must be used together with the weights |
| `policy_postprocessor.json` and `*_unnormalizer_processor.safetensors` | text + binary | output post-processing: turns the network's normalised actions back into real joint angles |
| `training_state/` (next to `pretrained_model/`) | text + binary | `optimizer_state.safetensors` (Adam's moments), `rng_state.safetensors` (random state), `training_step.json` (which step). Only needed to resume training |

### F. Other runtime files (`outputs/`, not in git)

| File | What it is |
|---|---|
| `.jpg` | images: camera snapshots (`outputs/hardware_check/`), dataset thumbnails (`outputs/dataset_check/`) |
| `.log` | logs: everything a program printed, saved for later (`outputs/logs/`) |

### G. The hidden `.git/` folder

git's "database" with every past version. **Don't edit or delete it by hand**; without it the folder is no longer a git repo.

## 4. Directory layout

```
Lerobot/
├── README.md              the front page (English; Chinese in README.zh-CN.md): what was done and how it went
├── .gitignore             what not to upload
├── config/
│   ├── robot.example.json config template (in git)
│   ├── robot.json         this computer's real config: COM ports, camera indices (not in git, differs per machine)
│   └── calibration/       calibration of both arms (in git: the dataset depends on it)
├── scripts/               tools that run on the Windows laptop: hardware, recording, local training, deployment
│   ├── env.ps1            shared environment: makes LeRobot keep datasets and calibration inside this repo
│   ├── find_hardware.py   lists serial ports and cameras, and takes one photo per camera
│   ├── scan_motors.py     read-only check of the 6 servo IDs and angles
│   ├── check_cameras.py   camera pre-check before recording
│   ├── inspect_dataset.py data-quality check after recording (idle fraction, waiting at the start, thumbnails)
│   ├── robot.ps1          reads config/robot.json and builds the calibrate/teleop/record/deploy commands
│   ├── train.ps1          training entry point: per-policy presets
│   ├── lerobot_train_win.py  lerobot-train with the Windows fixes (see docs/04_act.md)
│   ├── align_camera.py    puts a bumped camera back where it was, against a frame from the recording
│   ├── make_gif.py        cuts an evaluation trial into a GIF for the README
│   └── plot_results.py    per-spot success chart from results/*.md
├── cloud/                 training elsewhere (models that don't fit the laptop's 8 GB)
│   ├── README.md          how to use it
│   ├── slurm/             Linux GPU cluster (Slurm queue): environment setup, job submission
│   │   ├── site.env.example  template for personal settings like the account (the real site.env stays local)
│   │   ├── setup_env.sh   one-time environment setup
│   │   ├── submit.sh      checks the account, then submits
│   │   └── train.sbatch   the training job itself
│   └── hf_jobs/train_pi05.py  submits π₀.₅ training to rented Hugging Face GPUs
├── zero_shot/             the training-free baselines (VLM + IK): kinematics, homography calibration, local VLM pointing, real-robot grasp script run_grasp.py
├── results/               per-trial real-robot evaluation sheets
├── assets/                GIFs and charts for the README
├── docs/                  the tutorial, one file per stage (English and Chinese)
├── data/                  recorded datasets (not in git, copies on the HF Hub)
├── outputs/               runtime output: snapshots, checkpoints, logs (not in git)
└── notes/                 personal dev log (local only)
```

**Local vs. remote:** everything in `scripts/` runs on the laptop; everything in `cloud/` runs on other machines.
When both train the same policy the settings match; the only difference is that a bigger GPU can use the paper's default model size.

Why aren't datasets and models in git? git is built for **text code**; hundreds of MB of video and weights bloat the repo and make cloning slow,
and GitHub caps files at 100 MB. The usual practice: code on GitHub, datasets and models on the Hugging Face Hub, linked from the README.

Why both `robot.example.json` and `robot.json`?
COM ports and camera indices are **specific to this computer** and change on another one, so the real config isn't in git;
but someone cloning the repo needs to see what it looks like, so a template is committed. That's standard practice in open-source projects.

## 5. A few "environment" concepts

| Concept | Analogy | Here |
|---|---|---|
| **PowerShell** | Windows' command line (terminal) where you type commands | every command runs here |
| **conda environment** | a separate Python + set of libraries, isolated from others. Like one Python per project | named `lerobot`; run `conda activate lerobot` in every new terminal |
| **pip** | Python's package manager: `pip install xxx` | installed `lerobot==0.7.0` |
| **CLI command** | commands added by an installed library, e.g. `lerobot-record` | really just a Python function inside the lerobot package |
| **`--robot.port=COM3`** | a command-line argument. LeRobot uses dots for nested config: `robot.port` is `port` under `robot` | same idea as `--lr=1e-4` in training |
| **git / GitHub** | git records every change to the code; GitHub hosts git repos | this folder is pushed to GitHub |
| **Hugging Face Hub** | "GitHub for models and datasets" | the recorded dataset and trained models live here |

## 6. The routine at the start of each session

```powershell
cd D:\桌面\Lerobot
conda activate lerobot
. .\scripts\env.ps1                  # only needed when calling lerobot-xxx directly; robot.ps1 / train.ps1 do it for you
python scripts/find_hardware.py      # check that ports and camera indices haven't changed
```
