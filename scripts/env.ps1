# Shared environment for every script in this repo (dot-sourced: `. $PSScriptRoot\env.ps1`).
#
# LeRobot stores datasets under $HF_LEROBOT_HOME and calibration files under
# $HF_LEROBOT_CALIBRATION. By default both live in the global Hugging Face cache
# ($HF_HOME/lerobot). Pointing them into the repo keeps the whole project in one folder:
#
#   data/                 recorded datasets   (git-ignored; the copy of record is on the HF Hub)
#   config/calibration/   arm calibration     (tracked in git: small, and datasets depend on it)
#
# Downloaded pretrained models (ResNet weights, SmolVLA, pi0.5 ...) still go to the global
# cache ($HF_HOME / $TORCH_HOME): they are shared downloads, not part of this project.

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:HF_LEROBOT_HOME = Join-Path $repoRoot "data"
$env:HF_LEROBOT_CALIBRATION = Join-Path $repoRoot "config\calibration"
# Python's UTF-8 mode. Without it, text files default to the Windows code page (cp936 on a
# Chinese system). LeRobot writes a temporary ffmpeg file list holding the absolute video paths;
# if the repo path has non-ASCII characters (e.g. a Chinese "Desktop" folder), ffmpeg can't read that list and
# saving the 2nd episode of a recording fails with "[Errno 22] Invalid argument: ...ffconcat".
$env:PYTHONUTF8 = "1"
# huggingface_hub warns on every download that Windows can't make symlinks; it works without them.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
