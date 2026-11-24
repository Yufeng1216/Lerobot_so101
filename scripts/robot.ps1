<#
.SYNOPSIS
  Thin wrapper around the LeRobot CLI that fills in ports, IDs and cameras from config/robot.json,
  so every stage of the tutorial uses exactly the same hardware settings.

.EXAMPLE
  .\scripts\robot.ps1 calibrate-follower
  .\scripts\robot.ps1 calibrate-leader
  .\scripts\robot.ps1 teleop                 # leader drives follower, no camera
  .\scripts\robot.ps1 teleop -Cameras        # ... with the camera stream shown in Rerun
  .\scripts\robot.ps1 record -Name so101_pick_place -Episodes 10
  .\scripts\robot.ps1 record -Name so101_pick_place -Episodes 10 -Resume   # add 10 more episodes
  .\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model
  .\scripts\robot.ps1 eval -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Name act
      # 20 recorded trials (docs/08_evaluation.md) -> local dataset <hf_user>/rollout_eval_act

  Any extra arguments are passed straight through to the LeRobot command, e.g.
  .\scripts\robot.ps1 record -Name test -Episodes 2 --dataset.episode_time_s=20
  Add -DryRun to print the command without running it.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("calibrate-follower", "calibrate-leader", "teleop", "record", "rollout", "eval")]
    [string]$Command,

    [switch]$Cameras,          # teleop: also stream the cameras
    [string]$Name,             # record: dataset name (repo id becomes <hf_user>/<Name>); eval: policy label
    [int]$Episodes = 50,       # record: number of episodes (with -Resume: number of EXTRA episodes); eval: default 20
    [int]$EpisodeTime = 30,    # record: max seconds per episode (press Right Arrow to end early)
    [int]$ResetTime = 25,      # record: max seconds to put the scene back (press Right Arrow to skip the rest)
    [switch]$Resume,           # record: append to an existing dataset instead of creating it
    [switch]$NoPush,           # record: keep the dataset local, don't upload to the Hub
    [string]$Policy,           # rollout / eval: local checkpoint dir or Hub repo id
    [int]$Duration = 0,        # rollout: seconds, 0 = until Ctrl+C
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1")   # datasets -> data/, calibration -> config/calibration/
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root "config/robot.json"
if (-not (Test-Path $configPath)) {
    throw "config/robot.json not found. Copy config/robot.example.json to config/robot.json and edit it."
}
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json

function Get-CameraArg {
    # Builds: { top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG} }
    # fourcc is optional: MJPG makes the camera send compressed frames, which halves USB bandwidth
    # and keeps two cameras on one USB controller from failing to open together.
    $parts = foreach ($p in $cfg.cameras.PSObject.Properties) {
        $c = $p.Value
        $s = "$($p.Name): {type: $($c.type), index_or_path: $($c.index_or_path), width: $($c.width), height: $($c.height), fps: $($c.fps)"
        if ($c.fourcc) { $s += ", fourcc: $($c.fourcc)" }
        "$s}"
    }
    "{ " + ($parts -join ", ") + " }"
}

$follower = @(
    "--robot.type=$($cfg.follower.type)",
    "--robot.port=$($cfg.follower.port)",
    "--robot.id=$($cfg.follower.id)"
)
$leader = @(
    "--teleop.type=$($cfg.leader.type)",
    "--teleop.port=$($cfg.leader.port)",
    "--teleop.id=$($cfg.leader.id)"
)
$cameraArg = "--robot.cameras=$(Get-CameraArg)"

switch ($Command) {
    "calibrate-follower" { $exe = "lerobot-calibrate"; $argsList = $follower }
    "calibrate-leader"   { $exe = "lerobot-calibrate"; $argsList = $leader }
    "teleop" {
        $exe = "lerobot-teleoperate"
        $argsList = $follower + $leader
        if ($Cameras) { $argsList += @($cameraArg, "--display_data=true") }
    }
    "record" {
        if (-not $Name) { throw "record needs -Name, e.g. -Name pick_toy" }
        $exe = "lerobot-record"
        $repoId = "$($cfg.hf_user)/$Name"
        $argsList = $follower + $leader + @(
            $cameraArg,
            "--dataset.repo_id=$repoId",
            # Keep the exact name (LeRobot would otherwise append _YYYYMMDD_HHMMSS),
            # so training commands and resume can refer to it by a stable name.
            "--dataset.no_stamp=true",
            "--dataset.num_episodes=$Episodes",
            "--dataset.episode_time_s=$EpisodeTime",
            "--dataset.reset_time_s=$ResetTime",
            "--dataset.single_task=$($cfg.task)",
            # Encode video while recording instead of dumping PNGs first: saving an episode is instant.
            "--dataset.streaming_encoding=true",
            "--display_data=true"
        )
        if ($NoPush) { $argsList += "--dataset.push_to_hub=false" }
        if ($Resume) {
            # Datasets live in $HF_LEROBOT_HOME/<repo_id> (default: $HF_HOME/lerobot/<repo_id>).
            $lerobotHome = if ($env:HF_LEROBOT_HOME) { $env:HF_LEROBOT_HOME } else { Join-Path $env:HF_HOME "lerobot" }
            $argsList += @("--resume=true", "--dataset.root=$(Join-Path $lerobotHome $repoId)")
        }
    }
    "rollout" {
        if (-not $Policy) { throw "rollout needs -Policy <checkpoint dir or hub repo id>" }
        $exe = "lerobot-rollout"
        $argsList = @("--strategy.type=base", "--policy.path=$Policy") + $follower + @(
            $cameraArg,
            "--task=$($cfg.task)",
            "--display_data=true"
        )
        if ($Duration -gt 0) { $argsList += "--duration=$Duration" }
    }
    "eval" {
        # The evaluation protocol of docs/08: the policy drives each trial, the leader arm is
        # used during the reset phase to put the follower back. Every trial is recorded as an
        # episode (video for the write-up). Keyboard: Right Arrow = end trial / reset early,
        # Left Arrow = redo the trial, Esc = stop. The dataset stays local (no Hub upload).
        if (-not $Policy) { throw "eval needs -Policy <checkpoint dir or hub repo id>" }
        if (-not $Name) { throw "eval needs -Name <policy label>, e.g. -Name act" }
        if (-not $PSBoundParameters.ContainsKey("Episodes")) { $Episodes = 20 }
        $exe = "lerobot-rollout"
        # LeRobot requires rollout datasets to be named rollout_*.
        $repoId = "$($cfg.hf_user)/rollout_eval_$Name"
        $argsList = @("--strategy.type=episodic", "--policy.path=$Policy") + $follower + $leader + @(
            $cameraArg,
            "--dataset.repo_id=$repoId",
            "--dataset.no_stamp=true",
            "--dataset.num_episodes=$Episodes",
            "--dataset.episode_time_s=$EpisodeTime",
            "--dataset.reset_time_s=$ResetTime",
            "--dataset.single_task=$($cfg.task)",
            "--dataset.streaming_encoding=true",
            "--dataset.push_to_hub=false",
            "--display_data=true"
        )
    }
}

if ($Extra) { $argsList += $Extra }

Write-Host "> $exe $($argsList -join ' ')" -ForegroundColor Cyan
if ($DryRun) { return }

if ($Command -eq "record") {
    # lerobot-record creates the dataset folder before connecting the cameras, so a camera
    # failure would leave an empty dataset behind. Catch both problems before starting.
    $lerobotHome = if ($env:HF_LEROBOT_HOME) { $env:HF_LEROBOT_HOME } else { Join-Path $env:HF_HOME "lerobot" }
    $datasetDir = Join-Path $lerobotHome $repoId
    if (-not $Resume -and (Test-Path $datasetDir)) {
        throw "Dataset folder already exists: $datasetDir`nUse -Resume to add episodes to it, or pick another -Name. If it is an empty leftover from a failed run (meta\info.json says total_episodes 0), delete that folder."
    }
    $ErrorActionPreference = "Continue"  # OpenCV prints warnings on stderr; don't treat them as fatal
    python (Join-Path $PSScriptRoot "check_cameras.py")
    if ($LASTEXITCODE -ne 0) { throw "Camera pre-flight check failed; nothing was recorded." }
}
# "Continue": Windows PowerShell 5.1 turns a native program's stderr lines into errors when
# output is redirected; with "Stop" a harmless warning would abort the run. Exit code still counts.
$ErrorActionPreference = "Continue"
& $exe @argsList
exit $LASTEXITCODE
