<#
.SYNOPSIS
  Train a policy on the recorded dataset with `lerobot-train`, using the settings this
  project settled on for each policy (documented in docs/04_act.md, docs/05_..., ...).

.EXAMPLE
  .\scripts\train.ps1 act                          # full ACT run
  .\scripts\train.ps1 act -Steps 200 -Name smoke   # quick pipeline check (a few minutes)
  .\scripts\train.ps1 act -Resume                  # continue the last ACT run from its last checkpoint
  .\scripts\train.ps1 act -DryRun                  # print the command only

  Extra arguments go straight to lerobot-train, e.g.
  .\scripts\train.ps1 act --batch_size=4
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("act", "diffusion")]   # smolvla / pi05 are added as we reach them
    [string]$Policy,

    [string]$Dataset = "so101_pick_place",   # dataset name under your HF user
    [int]$Steps = 0,                         # 0 = the policy's preset below
    [string]$Name,                           # run name; default "<policy>_<dataset>"
    [switch]$Resume,
    [switch]$NoPush,                         # don't upload the final model to the Hub
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1")   # datasets -> data/, calibration -> config/calibration/
$root = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content (Join-Path $root "config/robot.json") -Raw | ConvertFrom-Json

if (-not $Name) { $Name = "$($Policy)_$($Dataset)" }
$outputDir = "outputs/train/$Name"

if ($Resume) {
    # Everything (policy, dataset, optimizer state, step counter) is restored from the checkpoint.
    $argsList = @(
        "--config_path=$outputDir/checkpoints/last/pretrained_model/train_config.json",
        "--resume=true"
    )
}
else {
    # Settings shared by every policy.
    $argsList = @(
        "--dataset.repo_id=$($cfg.hf_user)/$Dataset",
        # Hold out 10% of episodes and report their loss every 2k steps: a validation curve,
        # so over-fitting is visible instead of guessed.
        "--dataset.eval_split=0.1",
        "--eval_steps=2000",
        "--output_dir=$outputDir",
        "--job_name=$Name",
        "--policy.device=cuda",
        "--policy.repo_id=$($cfg.hf_user)/$Name",
        "--trackio.enable=true",
        "--trackio.project=so101",
        "--num_workers=4",
        "--log_freq=100"
    )
    if ($NoPush) { $argsList += "--policy.push_to_hub=false" }

    switch ($Policy) {
        "act" {
            # ACT defaults (chunk_size=100, ResNet18, CVAE with kl_weight=10, lr=1e-5) are the
            # paper's and work for SO-101; only the schedule is set here.
            # batch 8 fits comfortably in 8 GB of VRAM with two 480x640 cameras.
            if ($Steps -eq 0) { $Steps = 50000 }
            $argsList += @(
                "--policy.type=act",
                "--batch_size=8",
                "--steps=$Steps",
                "--save_freq=10000"
            )
        }
        "diffusion" {
            # Diffusion Policy defaults (n_obs_steps=2, horizon=64, n_action_steps=32, DDPM with
            # 100 steps, U-Net down_dims 512/1024/2048, lr=1e-4 cosine) follow the paper.
            # Changed for an 8 GB laptop GPU and a 50-episode dataset:
            #   resize_shape 240x320 : 4x fewer pixels per camera -> fits in VRAM and trains faster
            #   crop_ratio 0.95      : random 95% crop during training = data augmentation (paper)
            #   ema warmup           : exponential moving average of weights, standard for diffusion
            if ($Steps -eq 0) { $Steps = 100000 }
            $argsList += @(
                "--policy.type=diffusion",
                "--policy.resize_shape=[240,320]",
                "--policy.crop_ratio=0.95",
                # U-Net channels halved (default 512/1024/2048 = 278M params): with fp32 weights,
                # grads, Adam and the EMA copy that alone is ~5.5 GB, so 8 GB overflowed into
                # shared system memory and training slowed to ~2 min/step.
                "--policy.down_dims=[256,512,1024]",
                "--ema.type=warmup",
                "--batch_size=8",
                "--steps=$Steps",
                "--save_freq=20000"
            )
        }
    }
}

if ($Extra) { $argsList += $Extra }

Write-Host "> lerobot-train (via scripts/lerobot_train_win.py) $($argsList -join ' ')" -ForegroundColor Cyan
if ($DryRun) { return }
if (-not $Resume -and (Test-Path $outputDir)) {
    throw "$outputDir already exists. Use -Resume to continue it, or -Name to start a separate run."
}
# Same as `lerobot-train`, but checkpoints/last is a junction instead of a symlink, so a
# non-admin Windows user doesn't crash at the first checkpoint (see lerobot_train_win.py).
# "Continue": Windows PowerShell 5.1 turns a native program's stderr lines (harmless
# warnings from torch) into errors when output is redirected; "Stop" would abort the run.
$ErrorActionPreference = "Continue"
python (Join-Path $PSScriptRoot "lerobot_train_win.py") @argsList
exit $LASTEXITCODE
