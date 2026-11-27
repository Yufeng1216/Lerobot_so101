# Code walkthrough: from hardware discovery to training and evaluation

**English** · [简体中文](code_walkthrough.zh-CN.md)

This page goes through every script in the repo **in the order the project uses them**: what problem it solves, what each
part of the code does, why it's written that way, and what the LeRobot internals it calls are doing.

Division of labour with the other docs: `00`–`09` say **how to operate**; this page says **what the code does**.
Python / PowerShell syntax is explained the first time it appears (marked 📘).

| Stage | Script | In one line |
|---|---|---|
| 1 hardware | [`find_hardware.py`](#1-scriptsfind_hardwarepy) | list serial ports and cameras, one photo per camera |
| 1 hardware | [`scan_motors.py`](#2-scriptsscan_motorspy) | read-only check of the 6 servo IDs |
| 2 config | [`env.ps1`](#3-scriptsenvps1) | make LeRobot keep data and calibration inside this repo |
| 2 config | [`robot.ps1`](#4-scriptsrobotps1) | build calibrate / teleop / record / deploy commands from the config file |
| 3 data | [`check_cameras.py`](#5-scriptscheck_cameraspy) | camera pre-check before recording |
| 3 data | [`inspect_dataset.py`](#6-scriptsinspect_datasetpy) | data-quality report after recording |
| 4 training | [`train.ps1`](#7-scriptstrainps1) | training entry point and per-policy presets |
| 4 training | [`lerobot_train_win.py`](#8-scriptslerobot_train_winpy) | fixes the checkpoint-link problem on Windows |
| 5 remote | [`cloud/slurm/*`](#10-cloudslurm-training-on-a-gpu-cluster) | environment setup and job submission on a Linux GPU cluster |
| 5 remote | [`cloud/hf_jobs/train_pi05.py`](#13-cloudhf_jobstrain_pi05py) | submit π₀.₅ training to rented HF GPUs |
| 6 evaluation | [`make_server_config.py`, `serve_pi05.py`](#15-remote-inference-make_server_configpy-and-cloudhf_jobsserve_pi05py) | run π₀.₅ on an HF Jobs GPU and drive the arm from the laptop |
| 6 evaluation | [`make_gif.py`](#11-scriptsmake_gifpy) | cut an evaluation trial into a GIF for the README |
| 6 evaluation | [`plot_results.py`](#12-scriptsplot_resultspy) | per-spot success bar chart from `results/*.md` |
| 7 zero-shot | [`zero_shot/*`](#14-zero_shot-the-training-free-baseline-vlm-agent) | kinematics, touch calibration, gripper measurement, the Claude Haiku agent |
| internals | [What LeRobot does inside](#9-lerobot-internals-what-the-commands-we-call-do) | recording loop, dataset, training loop, ACT forward |

---

## 1. `scripts/find_hardware.py`

**Problem:** after plugging in two servo boards and the cameras, Windows assigns COM numbers and camera indices but doesn't tell you which is which.

### The docstring at the top

```python
"""List the serial ports and cameras this computer can see.
...
"""
```

📘 Triple quotes `"""..."""` are a multi-line string in Python. At the very top of a file or function it's called a **docstring**,
documentation for humans. `python scripts/find_hardware.py --help` prints it (see `argparse` below).

### Imports

```python
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from serial.tools import list_ports
```

- 📘 `import x` loads a library; `from x import y` takes only `y` from it.
- `from __future__ import annotations`: postpones evaluation of type hints (like `-> None`); a habit of modern Python, no effect on the logic.
- `argparse`: parses command-line arguments (like `--max-cameras 6`).
- `Path`: a class for file paths; `Path("outputs") / "a.jpg"` builds the right path on both Windows and Linux.
- `cv2`: OpenCV, to read cameras.
- `serial.tools.list_ports`: pyserial's port enumerator.

### Table of known chips

```python
KNOWN_CHIPS = {
    (0x1A86, 0x55D3): "WCH CH343 (Waveshare / Feetech bus servo board)",
    (0x1A86, 0x7523): "WCH CH340",
    ...
}
```

📘 `{key: value}` is a **dict**. The keys here are **tuples** `(VID, PID)`. `0x1A86` is a hexadecimal number.

Every USB device has two numbers: **VID** (vendor, `1A86` = WCH) and **PID** (product).
COM numbers change when you re-plug, but VID:PID is burned into the chip, so it's the most reliable way to recognise a board.
Our two boards happen to use different chips (CH343 and CH340), which is how we tell them apart.

### Listing serial ports

```python
def list_serial_ports() -> None:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    for p in ports:
        chip = KNOWN_CHIPS.get((p.vid, p.pid), "unknown chip") if p.vid else "no USB VID/PID"
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid else "----:----"
        print(f"  {p.device:<8} {vidpid}  ...")
```

- 📘 `def name(args) -> return_type:` defines a function. `-> None` means it returns nothing.
- 📘 `sorted(list, key=lambda p: p.device)`: sort by `p.device` (e.g. `COM3`).
  `lambda p: p.device` is a small **anonymous function**: takes `p`, returns `p.device`.
- 📘 `dict.get(key, default)`: returns the default instead of raising when the key is missing.
- 📘 `A if condition else B`: an if-else on one line.
- 📘 **f-string**: `{expression}` inside `f"..."` is replaced by its value. `{p.vid:04X}` means "4 digits, zero-padded, upper-case hex";
  `{p.device:<8}` means "left-aligned, 8 characters wide", to line up the columns.

### Taking photos

```python
def list_cameras(max_index: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(15):
            ok, frame = cap.read()
        ...
        cap.release()
        cv2.imwrite(str(path), frame)
```

- `mkdir(parents=True, exist_ok=True)`: create the folder, parents too if needed, no error if it exists.
- 📘 `for idx in range(5)`: `idx` takes 0, 1, 2, 3, 4. `continue` skips the rest of this iteration.
- `cv2.VideoCapture(idx)` opens a camera by index; on Windows OpenCV uses the MSMF backend by default, same as LeRobot.
- **Why read 15 frames before saving**: right after opening, auto-exposure hasn't settled and the first frames are often black.
- 📘 `for _ in range(15)`: `_` is the conventional name for "I don't need this variable".
- 📘 `ok, frame = cap.read()`: the function returns two values, assigned to two variables (**tuple unpacking**).
- `cap.release()`: frees the camera. **If you don't, the next program (e.g. LeRobot) can't open it.**

### Entry point

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, ...)
    parser.add_argument("--max-cameras", type=int, default=5, help="...")
    args = parser.parse_args()
    list_serial_ports()
    list_cameras(args.max_cameras, args.out)

if __name__ == "__main__":
    main()
```

- `argparse`: declares which arguments the script takes, their types and defaults. `args.max_cameras` is the user's value (`-` becomes `_`).
- 📘 `if __name__ == "__main__":` runs `main()` only when the file is **run directly**, not when another file `import`s it.
  On Windows this line is **essential**: Windows multiprocessing only supports spawn, so each child process is a fresh Python
  that re-imports your script. Without the guard, the data-loader workers in training would each start more workers, endlessly.

---

## 2. `scripts/scan_motors.py`

**Problem:** servo IDs live in each servo's own EEPROM and survive another computer or OS. So we only need to confirm all 6 IDs are right, **without changing anything on the servos**.

### Background: the servo bus

The SO-101's 6 Feetech STS3215 servos are **chained on one cable** (daisy chain) and share one serial port. Every command from the host carries a target ID,
and only the servo with that ID answers. So the 6 servos need distinct IDs (1–6) and the same baud rate as the host (1 Mbps).
Both live in **the servo's own EEPROM** and survive power-off, so a new computer or OS needs no re-setup.

### Joint table

```python
SO101_MOTORS = {"shoulder_pan": 1, "shoulder_lift": 2, ..., "gripper": 6}
STS3215_MODEL_NUMBER = 777
```

The ID assignment matches LeRobot's `lerobot/robots/so_follower/so_follower.py` exactly.
`777` is what an STS3215 reports in its model register (same as LeRobot's `_model_number("sts3215")`).

### Building the bus

```python
def build_bus(port: str) -> SerialMotorsBus:
    motors = {name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
              for name, motor_id in SO101_MOTORS.items()}
    return SerialMotorsBus(port=port, motors=motors)
```

- 📘 `{k: v for ... in ...}` is a **dict comprehension**: builds a whole dict with one loop.
- 📘 `dict.items()` yields keys and values together: `name`, `motor_id`.
- `SerialMotorsBus`: LeRobot 0.7.0's servo-bus class (the serial I/O underneath is the Rust library `rustypot`).
  The `FeetechMotorsBus` of older tutorials is deprecated in 0.7.0.

### Scanning

```python
model = bus.ping(name, num_retry=2)
if model is None:
    print("MISSING"); continue
pos = bus.read("Present_Position", name, normalize=False, num_retry=2)
```

- `ping` reads the servo's model register: the model number if it answers, `None` if not.
- `read("Present_Position", ..., normalize=False)` reads the **raw** encoder value (0–4095 per turn).
  `normalize=False` because there is no calibration file yet to convert to −100..100.
- `num_retry=2`: the serial line occasionally drops a packet; retry twice.

### Two key "read-only" choices

```python
bus.connect(handshake=False)
...
finally:
    bus.disconnect(disable_torque=False)
```

1. `handshake=False`: by default LeRobot checks every servo on connect and **exits with an error if one is missing**.
   We want "list which are present and which are missing", so we turn the handshake off and ping one by one.
2. `disable_torque=False`: by default LeRobot **writes** "torque off" on disconnect. That's a write;
   this script promises to be read-only, so it writes nothing on disconnect.
3. 📘 `try: ... finally: ...`: the `finally` block **always runs**, error or not.
   It guarantees the port gets closed; otherwise the next program can't open that COM port.

### `--watch` mode

```python
pos = bus.sync_read("Present_Position", names, normalize=False, num_retry=2)
print("\r  " + ..., end="", flush=True)
```

- `sync_read`: one command reads several servos at once, much faster than one by one.
- `"\r"` (carriage return) moves the cursor back to the start of the line and `end=""` suppresses the newline: each new line overwrites the old one, which looks like a live display.
- 📘 `except KeyboardInterrupt:` catches Ctrl+C so the program exits cleanly instead of printing a long traceback.

The same idea tells leader and follower apart: read both ports for 15 s while you move the leader, and sum each joint's max − min.
COM4 changed by 4545 in total, COM3 by 0, so COM4 is the leader.

---

## 3. `scripts/env.ps1`

```powershell
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:HF_LEROBOT_HOME = Join-Path $repoRoot "data"
$env:HF_LEROBOT_CALIBRATION = Join-Path $repoRoot "config\calibration"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

- 📘 PowerShell variables start with `$`. `$PSScriptRoot` is **the folder of the running script** (`scripts`);
  `Split-Path -Parent` gives its parent, the repo root. So the paths are right whatever directory you run from.
- 📘 `$env:NAME = value` sets an **environment variable**: an OS-level "global setting"
  visible to the current process and every child it starts (e.g. Python).
- LeRobot reads these two in `lerobot/utils/constants.py`:

```python
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path))
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration"))
```

  `os.getenv("NAME", default)`: use the variable if set, the default otherwise. So **without touching any LeRobot code**,
  two environment variables move the data and calibration.

- `PYTHONUTF8 = "1"` turns on Python's **UTF-8 mode**. On Chinese-locale Windows, Python writes text files in cp936 by default;
  when LeRobot saves the 2nd and later episodes it writes a temporary ffmpeg concat list (absolute video paths) without specifying an encoding.
  With non-ASCII characters in the repo path (like "桌面"), ffmpeg can't read the list and fails with
  `[Errno 22] Invalid argument: ...ffconcat`. In UTF-8 mode every file without an explicit encoding is written as UTF-8 and the problem goes away.

- These settings **only apply to the current PowerShell window** and vanish when it closes; other projects are unaffected.
  `robot.ps1` and `train.ps1` both load it first with `. (Join-Path $PSScriptRoot "env.ps1")`.
- 📘 The leading `.` (dot + space) is **dot-sourcing**: run another script in the **current** scope,
  so the variables it sets stay set. Run it without the dot and the variables disappear when that script ends.

---

## 4. `scripts/robot.ps1`

**Problem:** LeRobot's commands are very long (a record command has over a dozen arguments), easy to mistype, and
**calibration, teleoperation, recording and deployment must use exactly the same ports, IDs and camera settings**; otherwise the calibration file isn't found, or the data and the deployment don't match.

**Idea:** write the hardware settings once in `config/robot.json`; the script reads it and builds the full command.

### Parameters

```powershell
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("calibrate-follower", "calibrate-leader", "teleop", "record", "rollout", "eval")]
    [string]$Command,
    [switch]$Cameras,
    [int]$Episodes = 50,
    ...
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
```

- 📘 `param(...)` declares the script's parameters, like Python's `argparse`.
- `Mandatory = $true, Position = 0`: required, and can be given positionally first, hence `robot.ps1 record`.
- `ValidateSet(...)`: only these values are allowed; PowerShell lists them if you mistype.
- `[switch]`: a flag, `$true` when present (e.g. `-Cameras`), `$false` otherwise.
- `[int]$Episodes = 50`: an integer with default 50.
- `ValueFromRemainingArguments`: **anything not claimed by the parameters above goes into `$Extra`**.
  So the `--policy...` in `robot.ps1 rollout ... --policy.n_action_steps=1` is passed to LeRobot unchanged.

### Reading the config

```powershell
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
```

- 📘 `|` is a **pipe**: the output of the left command goes to the right one.
- `Get-Content -Raw` reads the whole file as text and `ConvertFrom-Json` turns it into an object,
  so `$cfg.follower.port` gives `"COM3"`.

### Building the camera argument

```powershell
function Get-CameraArg {
    $parts = foreach ($p in $cfg.cameras.PSObject.Properties) {
        $c = $p.Value
        $s = "$($p.Name): {type: $($c.type), index_or_path: $($c.index_or_path), ...}"
        ...
    }
    "{ " + ($parts -join ", ") + " }"
}
```

- `$cfg.cameras.PSObject.Properties`: iterate over the keys of the JSON object (`top`, `wrist`).
- 📘 `$(expression)` inside a double-quoted string is replaced by its value, like a Python f-string.
- 📘 The last "bare" value in a function is its return value (PowerShell doesn't need `return`).
- The result has the format LeRobot expects: `{ top: {type: opencv, index_or_path: 1, ...}, wrist: {...} }`,
  which LeRobot parses into `OpenCVCameraConfig` objects with the `draccus` library.

### Assembling per command

```powershell
switch ($Command) {
    "teleop" {
        $exe = "lerobot-teleoperate"
        $argsList = $follower + $leader
        if ($Cameras) { $argsList += @($cameraArg, "--display_data=true") }
    }
    "record" { ... }
}
```

- 📘 `@(a, b)` is an array; `+` concatenates arrays; `+=` appends.
- Why a few key arguments in the `record` branch:

| Argument | Why |
|---|---|
| `--dataset.no_stamp=true` | by default LeRobot appends `_YYYYMMDD_HHMMSS` to the dataset name. With this off the name is always `so101_pick_place`, so training commands and `-Resume` can refer to it directly |
| `--dataset.streaming_encoding=true` | encode frames to video while recording. Otherwise thousands of PNGs are written first and encoded after each episode, which takes a long time |
| `--dataset.root=...` with `-Resume` | LeRobot needs the exact on-disk location to resume recording |

`rollout` and `eval` both call `lerobot-rollout`; the difference is `--strategy.type`:

| Command | strategy | What it does |
|---|---|---|
| `rollout` | `base` | the policy drives the follower for `-Duration` seconds, nothing recorded. For trying things out |
| `eval` | `episodic` | following the [08 protocol](08_evaluation.md): 20 trials by default, a reset phase in between (the leader brings the follower back), each trial recorded as an episode |

Details of `eval`:
- The dataset is called `rollout_eval_<Name>`: LeRobot requires rollout datasets to start with `rollout_` and errors otherwise.
- Without `-Episodes` the default is 20, not record's 50: `$PSBoundParameters.ContainsKey("Episodes")` tells whether the user passed it explicitly.
- `--dataset.push_to_hub=false`: evaluation videos stay local.
- **Camera renaming for VLAs**: SmolVLA and π₀.₅ were fine-tuned with `--rename_map` (top → camera1, ...), and the mapping is stored in the model's `policy_preprocessor.json`. But `lerobot-rollout` checks camera names *before* the preprocessor runs and only skips that check when `--rename_map` is given. So `eval` reads the `rename_observations_processor` step from that file and passes the same mapping on; for ACT and Diffusion there is none and nothing is added. The inner quotes are written as `\"` for the PowerShell 5.1 reason explained in §7.
- **`-Remote`** (π₀.₅, which doesn't fit the laptop GPU): instead of `--policy.path`, pass `--inference.type=remote` with the deployment name (the policy `type` from the local `config.json`) and the semantics string that `make_server_config.py` writes, so client and server agree; images go as JPEG. `--inference.refill_seconds=1.0` asks for the next chunk 1 s before the current one runs out (default 0.5 s), so a cloud round trip of up to about 1 s doesn't make the arm pause. The camera mapping is still read from the local checkpoint, because the server drops its own rename step. See §15.
- **Automatic append**: if an evaluation dataset of that name exists (has `meta\info.json`), `--resume=true --dataset.root=...` are added.
  So one policy can be tested in several runs (`-Episodes 5` each) without remembering `-Resume`. Unlike `record`: there a name clash is usually a mistake and needs an explicit `-Resume`;
  for evaluation the same name means "the same experiment".

### Two checks before running (record only)

```powershell
if (-not $Resume -and (Test-Path $datasetDir)) { throw "..." }
python (Join-Path $PSScriptRoot "check_cameras.py")
if ($LASTEXITCODE -ne 0) { throw "..." }
```

**Why:** `lerobot_record.py` creates the dataset folder first and connects the cameras and arm second.
If a camera fails, an empty shell folder is left behind and the next recording with that name is refused.
So we check **before** calling LeRobot: stop if the folder exists; stop if a camera won't open.

- 📘 `throw "message"`: raise an error; the script stops at once.
- 📘 `$LASTEXITCODE`: the **exit code** of the last external program. By convention 0 means success, anything else failure.

### Running

```powershell
Write-Host "> $exe $($argsList -join ' ')" -ForegroundColor Cyan
if ($DryRun) { return }
$ErrorActionPreference = "Continue"
& $exe @argsList
exit $LASTEXITCODE
```

- First print the full command in cyan, so you see exactly what the script runs for you; `-DryRun` only prints, for checking.
- 📘 `& $exe` runs the program named in a variable. `@argsList` (**splatting**) expands the array into separate arguments,
  so an argument containing spaces (like the task description) is passed as one piece.
- Why `$ErrorActionPreference = "Continue"`: when output is redirected, Windows PowerShell 5.1 treats **every line** an external program writes to stderr
  (harmless warnings included) as an error. With `"Stop"`, one PyTorch notice would kill training.
  Real failures are still detected through the exit code.
- `exit $LASTEXITCODE`: pass LeRobot's exit code through, so the caller knows whether it worked.

---

## 5. `scripts/check_cameras.py`

**Problem:** the wrist camera's cable gets tugged by the arm and the camera occasionally fails to open (`failed to set fps=30 (actual_fps=0.0)`).

```python
cam_cfg = OpenCVCameraConfig(index_or_path=cfg["index_or_path"], fps=cfg["fps"],
                             width=cfg["width"], height=cfg["height"], fourcc=cfg.get("fourcc"))
for attempt in range(1, RETRIES + 1):
    cam = OpenCVCamera(cam_cfg)
    try:
        cam.connect()
        frame = cam.read()
        return True
    except Exception as e:
        print(f"... attempt {attempt} failed: {e}")
        time.sleep(1.0)
    finally:
        if cam.is_connected:
            cam.disconnect()
return False
```

- **It uses LeRobot's own `OpenCVCamera` class**, not raw `cv2`. So the check follows exactly the same path as recording:
  whatever would fail during recording (e.g. setting the fps) also fails here.
- `cfg.get("fourcc")`: use `fourcc` if the config has it, otherwise `None` (automatic).
  `MJPG` makes the camera send compressed frames, so two cameras on one USB hub don't run out of bandwidth.
- 📘 `try / except Exception as e`: catch the error, keep its message in `e`, don't crash, retry.
- 📘 `all(...)` around `[check(n, c) for n, c in cameras.items()]`: passes only if **every** camera passes.
  The list checks every camera before deciding, so the second camera's result is printed even if the first fails.
- `sys.exit(0 or 1)`: hand the result to `robot.ps1` as the exit code (the `$LASTEXITCODE` above).

---

## 6. `scripts/inspect_dataset.py`

**Problem:** with 50 episodes recorded you can't watch every video. We need an automatic report that picks out the problem episodes.

### Reading the data

```python
files = sorted((root / "data").rglob("*.parquet"))
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
```

- `rglob("*.parquet")`: find all parquet files recursively (data may be split over several chunks and files).
- 📘 A pandas **DataFrame** is a table: one row per frame, one column per field (`action`, `episode_index`, ...).
  `pd.concat` stacks several tables.

### Per-episode metrics

```python
for ep, g in df.groupby("episode_index"):
    a = np.stack(g["action"].values)                 # shape (frames, 6)
    speed = np.abs(np.diff(a, axis=0)).sum(axis=1)   # shape (frames-1,)
    moving = np.flatnonzero(speed > IDLE_THRESHOLD)
    idle = (speed <= IDLE_THRESHOLD).mean()
    start = moving[0] / fps
    end = (len(a) - 1 - moving[-1]) / fps
```

This is the heart of the script, line by line:

1. `df.groupby("episode_index")`: group by episode; `ep` is the index, `g` all frames of that episode.
2. `np.stack(...)`: each frame's `action` in the parquet file is an array of 6; `stack` builds a `(frames, 6)` matrix.
3. `np.diff(a, axis=0)`: subtract neighbouring frames: how much the action changes from each frame to the next.
4. `np.abs(...).sum(axis=1)`: add the absolute changes of the 6 joints: each frame's "amount of motion". It's just the L1 norm.
5. `speed > 0.3`: "moving" only above a threshold (0.3 in normalised units: above sensor noise, below the slowest deliberate motion).
6. `(speed <= 0.3).mean()`: the mean of a boolean array = the fraction of True = the **idle fraction**.
7. `np.flatnonzero(...)`: positions of all "moving" frames. The first one / fps = **seconds waited at the start**;
   the distance from the last one to the end / fps = **seconds idle at the end**.
8. `a[:, GRIPPER].max()`: maximum of column 5 (gripper) = how far the gripper opened (0 = closed, 100 = fully open).

Then flags follow simple rules, e.g. `idle > 0.9` → `NO MOTION - delete`.

### Thumbnails

```python
frames[cam] = [f.to_ndarray(format="bgr24") for vf in video_files for f in av.open(str(vf)).decode(video=0)]
picks = np.linspace(lo, hi, 6).astype(int)
```

- `av` (PyAV) decodes the MP4, each frame into a NumPy array. `bgr24` is OpenCV's colour order.
- 📘 A nested comprehension: loop over each video file, then over each frame in it.
- `np.linspace(lo, hi, 6)`: 6 evenly spaced positions between the episode's first and last frame.
- `np.hstack` joins side by side (6 frames per row), `np.vstack` stacks rows (one per camera), saved as one jpg.

---

## 7. `scripts/train.ps1`

Same structure as `robot.ps1`: parameters → read config → build arguments → run. The point is **why each argument is set as it is**.

### Arguments shared by all policies

| Argument | Meaning | Why |
|---|---|---|
| `--dataset.eval_split=0.1` + `--eval_steps=2000` | hold out 10% of episodes (5) from training and compute their loss every 2000 steps | a validation curve shows whether we overfit |
| `--output_dir=outputs/train/<name>` | where checkpoints and logs go | |
| `--policy.repo_id=Yufeng1216/<name>` | the Hub repo the model is pushed to at the end | |
| `--trackio.enable=true` | log loss curves to a local database | view with `trackio show` |

### Resuming

```powershell
if ($Resume) {
    $argsList = @("--config_path=$outputDir/checkpoints/last/pretrained_model/train_config.json", "--resume=true")
}
```

Resuming **specifies no arguments again**: every setting comes from the saved `train_config.json`,
and optimizer state, LR schedule and step counter come from `training_state/`, so it's as if training was never interrupted.

### Per-policy presets

```powershell
switch ($Policy) {
    "act" {
        if ($Steps -eq 0) { $Steps = 50000 }
        $argsList += @("--policy.type=act", "--batch_size=8", "--steps=$Steps", "--save_freq=10000")
    }
    "diffusion" {
        if ($Steps -eq 0) { $Steps = 100000 }
        $argsList += @(
            "--policy.type=diffusion",
            "--policy.resize_shape=[240,320]",   # half-size images, saves memory
            "--policy.crop_ratio=0.95",          # random crops in training = data augmentation
            "--policy.down_dims=[256,512,1024]", # U-Net channels halved: the default 278M overflows 8 GB
            "--ema.type=warmup",                 # also keep a moving average of the weights
            "--batch_size=8", "--steps=$Steps", "--save_freq=20000"
        )
    }
    "smolvla" {
        if ($Steps -eq 0) { $Steps = 30000 }
        $argsList += @(
            "--policy.path=lerobot/smolvla_base",          # fine-tune from pretrained weights
            '--rename_map={\"observation.images.top\":\"observation.images.camera1\",...}',
            "--batch_size=8", "--steps=$Steps",
            "--policy.scheduler_decay_steps=$Steps",       # align LR decay with the total steps
            "--save_freq=10000"
        )
    }
}
```

`-Steps` defaults to 0, meaning "use this policy's recommended value"; a smoke test overrides it with `-Steps 200`.
One branch per policy. π₀.₅ trains in the cloud, not here.

- `--policy.xxx`: the policy's own architecture parameters (the fields in `configuration_diffusion.py`).
  A list on the command line is written `[240,320]`; LeRobot's config parser (draccus) turns it into a Python tuple.
- `--ema.xxx`: a training-loop parameter, not part of the policy. ACT doesn't use EMA, so it only appears in the diffusion branch.
- The diffusion branch halves the U-Net channels so it can train on 8 GB; the final model of this project was trained at the default size on the cluster (§10).
- Why these values: [05_diffusion_policy.md](05_diffusion_policy.md) §5.

**Two new things in the smolvla branch:**

- `--policy.path` vs `--policy.type`: `type` **creates** a randomly initialised model with the default architecture;
  `path` **loads** architecture and weights from an existing checkpoint (local folder or Hub name). Fine-tuning needs `path`.
- The value of `--rename_map` is a JSON dict containing double quotes. When Windows PowerShell 5.1 passes arguments to an external program (python here),
  it **swallows** plain double quotes, so python would receive `{observation.images.top:...}` and fail to parse it.
  Written as `\"`, PowerShell passes it through and python reads `\"` as a double quote.
  Wrapping the whole argument in single quotes `'...'` stops PowerShell from interpreting `$` and backticks inside.

---

## 8. `scripts/lerobot_train_win.py`

**Problem:** every time LeRobot saves a checkpoint it points `checkpoints/last` at the newest one with a **symbolic link**.
A normal Windows user may not create symlinks (`WinError 1314`), so training crashes at the first save.

### Symlink vs directory junction

| | symlink | directory junction |
|---|---|---|
| What it does | a "shortcut": open it and you are in the target | same, but folders only |
| Windows permission | admin or Developer Mode | **any user** |
| Target path | may be relative | must be absolute |

All LeRobot needs is for `last` to "open like a folder", so a junction is enough.

### The code

```python
import lerobot.common.train_utils as train_utils
import lerobot.scripts.lerobot_train as lerobot_train

def update_last_checkpoint_junction(checkpoint_dir: Path) -> None:
    import _winapi
    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if last.is_junction() or last.is_symlink():
        os.rmdir(last)
    _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last))

if sys.platform == "win32":
    train_utils.update_last_checkpoint = update_last_checkpoint_junction
    lerobot_train.update_last_checkpoint = update_last_checkpoint_junction

if __name__ == "__main__":
    lerobot_train.main()
```

- This is a **monkey patch**: at run time a function inside a module is **replaced** with our own version, without editing the library's source.
  Pro: no fork of LeRobot, no conflicts when upgrading. Con: if LeRobot renames the function, the patch silently stops working.
- **Why replace it in two places**: the original lives in `train_utils`, but `lerobot_train.py` does
  `from ... import update_last_checkpoint`, which copies a reference into its own module. Patch only one and the other still has the old function.
  (`grep` over the source finds every place that calls it.)
- `os.rmdir(last)`: remove the old junction. `rmdir` on a junction **removes only the junction**, not the checkpoint it points to.
- `_winapi.CreateJunction`: the standard library's Windows-only junction function. The target must be absolute, hence `.resolve()`.
- `if sys.platform == "win32"`: only patch on Windows; on Linux the same file behaves exactly like the original.
- Finally call LeRobot's own `main()`. Command-line arguments pass through unchanged, so this file is used exactly like `lerobot-train`.

### Second patch: the slash in Hub model names

**Problem:** fine-tuning SmolVLA uses `--policy.path=lerobot/smolvla_base` (a Hub name, "user/model").
LeRobot stores it as a `Path`, and on Windows `str(Path("lerobot/smolvla_base"))` becomes `lerobot\smolvla_base`.
The Hub rejects backslashes: `Repo id must use alphanumeric chars ...`. On Linux the separator is `/` anyway, so only Windows breaks.

```python
def _fix_hub_id(name_or_path):
    name = str(name_or_path)
    if "\\" in name and not os.path.exists(name):
        return name.replace("\\", "/")
    return name_or_path

def hub_safe_from_pretrained(original):
    def from_pretrained(cls, *args, **kwargs):
        if args:
            args = (_fix_hub_id(args[0]), *args[1:])
        for key in ("pretrained_name_or_path", "pretrained_model_name_or_path"):
            if key in kwargs:
                kwargs[key] = _fix_hub_id(kwargs[key])
        return original(cls, *args, **kwargs)
    return classmethod(from_pretrained)

for klass in (PreTrainedPolicy, DataProcessorPipeline):
    klass.from_pretrained = hub_safe_from_pretrained(klass.__dict__["from_pretrained"].__func__)
```

- **The rule:** a backslash in the name and **no such local path** means it's a mangled Hub name, so turn `\` back into `/`.
  A real local folder (e.g. `outputs\train\...\pretrained_model`) is left alone.
- **Two loaders to patch:** a pretrained policy is **weights** (loaded by `PreTrainedPolicy.from_pretrained`) plus **pre/post-processing config**
  (loaded by `DataProcessorPipeline.from_pretrained`, e.g. normalisation stats). Patch only the first and the weights load,
  but the processor config fails with the same error. So both.
- The name is sometimes positional (`from_pretrained("lerobot/x")`) and sometimes a keyword (`from_pretrained(pretrained_model_name_or_path=...)`),
  and the two loaders even name the keyword differently, so `*args, **kwargs` handles both.
- It's a **decorator**-style wrapper: `hub_safe_from_pretrained` takes the original function and returns a new one that "fixes the name, then calls the original".
- `from_pretrained` is a **classmethod** (its first argument is the class `cls`, not an instance).
  `klass.__dict__["from_pretrained"]` is the classmethod object itself and `.__func__` the plain function inside;
  we wrap that and turn it back into a classmethod with `classmethod(...)`.
- Patching the base classes is enough: SmolVLA doesn't override `from_pretrained`, so calls walk up the inheritance chain to the patched version.

---

## 9. LeRobot internals: what the commands we call do

### The main loop of `lerobot-record` (`lerobot/scripts/lerobot_record.py`)

```
create the dataset (or resume) → connect the leader → connect the follower and cameras
while recorded < num_episodes:
    🔊 "Recording episode N"
    record_loop(duration = episode_time_s):   # every 1/30 s
        obs = robot.get_observation()         # follower's 6 joints + both cameras
        act = teleop.get_action()             # leader's 6 joints
        robot.send_action(act)                # follower follows
        dataset.add_frame({obs, act, task})   # into the current episode's buffer
    🔊 "Reset the environment"
    record_loop(duration = reset_time_s, nothing written)
    if ← was pressed: clear the buffer, re-record
    else:             dataset.save_episode()  # write parquet, close this video segment
upload to the Hub
```

This explains facts the tutorial relies on: **an episode is saved only after its reset** (so ← during reset still discards it);
the last episode has no reset; during reset the leader still drives the follower.

### `LeRobotDataset.__getitem__`: fetching one training sample

1. Take frame i's `observation.state`, `action`, etc. from the parquet file.
2. Fetch neighbouring frames according to `delta_timestamps` (ACT wants the next 100 actions); past the end of the episode the last frame is repeated,
   and an `action_is_pad` mask tells the loss which positions are padding.
3. Locate and decode the matching image frames in the MP4 by timestamp.
4. Return a dict: `{"observation.images.top": tensor, ..., "action": (100, 6) tensor, ...}`.

### The training loop of `lerobot-train` (`lerobot/scripts/lerobot_train.py`)

```
read dataset metadata (stats.json) → split episodes into train/validation by eval_split
make_policy()               # build the network from policy.type; ResNet18 loads ImageNet weights
make_pre_post_processors()  # normalisation (mean/std from stats.json) and de-normalisation
for step in range(steps):
    batch = next(dataloader)
    batch = preprocessor(batch)      # normalise, move to GPU
    loss, info = policy.forward(batch)
    loss.backward(); gradient clipping; optimizer.step(); scheduler.step()
    every log_freq steps: print loss, grdn (gradient norm), data_s, updt_s
    every eval_steps steps: eval_loss on the validation episodes
    every save_freq steps: save a checkpoint, update last
at the end: push_to_hub
```

In the log, `data_s` is time spent waiting for data and `updt_s` the GPU forward/backward time (both averaged between log lines).
`data_s` is inflated in the first 100 steps because the worker start-up time (about 19 s each on Windows) is averaged in;
after warm-up a batch takes about 0.06 s (torchcodec decoding, 4 workers), so the GPU is the bottleneck.

### ACT's `forward` (`lerobot/policies/act/modeling_act.py`)

```
training:
  VAE encoder input [CLS, joint state, 100 true actions] → CLS output → μ, log σ² → sample z (32-D)
inference:
  z = 0
both:
  each image → ResNet18 → 512×15×20 feature map → 300 tokens (+ 2D sinusoidal positional encoding)
  encoder input = [z token, joint-state token, top's 300 tokens, wrist's 300 tokens]
  Transformer encoder (4 layers) → Transformer decoder (1 layer, 100 learned queries) → Linear → (100, 6)
loss = L1(predicted, true actions, ignoring action_is_pad positions) + 10 × KL(q(z) ‖ N(0, I))
```

These two terms are `l1_loss` and `kld_loss` in the log. The model itself is explained in [04_act.md](04_act.md) §3–4.

---

## 10. `cloud/slurm/`: training on a GPU cluster

These are **bash scripts** that run on the Linux cluster, not on Windows. Usage in [cloud/README.md](../cloud/README.md).
Division of work: `setup_env.sh` sets up the environment once; `submit.sh` reads the personal settings, checks the account and submits; `train.sbatch` is the job itself.

### Personal settings kept out of the code: `site.env`

```bash
ACCOUNT=...                  # billing account
PARTITION=gpu                # GPU partition
TIME=08:00:00                # no longer than the account allows
ALLOWED_ACCOUNT_PREFIX=...   # only accounts starting with this are allowed
```

Account names differ per person, so the real `site.env` is excluded by `.gitignore` and only the template `site.env.example` is committed;
the same idea as `config/robot.json` / `robot.example.json`.

### `setup_env.sh`

```bash
set -euo pipefail
```

bash's "safe mode": `-e` stop when any command fails; `-u` error on undefined variables; `-o pipefail` a pipe fails if any part fails.

```bash
if [ ! -x "$CONDA_DIR/bin/conda" ]; then ... fi
```

Install Miniforge only if `~/miniforge3/bin/conda` doesn't exist (`!` negates, `-x` means "exists and is executable"). So the script **can be re-run**; done steps are skipped.

The versions match the laptop (Python 3.12, LeRobot 0.7.0, torch 2.11), with three Linux-specific measures explained in the script's comments:

| Measure | Why |
|---|---|
| PyTorch built for **CUDA 12.6** (`cu126`) | the CUDA 12.8 build no longer contains GPU code for the V100 (compute capability 7.0) and fails with `no kernel image is available` |
| remove torchcodec, train with `--dataset.video_backend=pyav` | the Linux torchcodec wheel is built against CUDA 13 and doesn't match the cu126 PyTorch |
| install conda's `libstdcxx-ng` and put `$CONDA_PREFIX/lib` first in `LD_LIBRARY_PATH` | otherwise the cluster's older system C++ runtime is loaded first and importing `sqlite3` fails with `CXXABI_1.3.15 not found` |

### `submit.sh`

```bash
POLICY="${1:?usage: ...}"
```

`$1` is the first command-line argument (e.g. `diffusion`). `${1:?message}` means "if missing, print the message and exit".

```bash
case "$ACCOUNT" in
    "$ALLOWED_ACCOUNT_PREFIX"*) ;;
    *) echo "ERROR: ..." >&2; exit 1 ;;
esac
```

bash pattern matching: let it through if the account starts with the allowed prefix (`*` matches anything after it), otherwise exit with an error (`>&2` writes to stderr).
When you have several billing accounts but only one may be used for this project, this prevents picking the wrong one.

```bash
sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" --job-name="$POLICY" train.sbatch "$POLICY"
```

The personal settings go to `sbatch` as command-line arguments, so `train.sbatch` contains no account information.

### `train.sbatch`

```bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.log
```

Comments to bash, **resource requests** to Slurm: 1 GPU, 8 CPU cores (for the data loaders), log to `logs/<jobname>_<jobid>.log`.

When the job starts it **checks the account once more** (`SLURM_JOB_ACCOUNT` is set by Slurm), a second safeguard.

```bash
if [ -d "$OUT/checkpoints/last" ]; then
    srun lerobot-train --config_path=.../train_config.json --resume=true
    exit
fi
```

Cluster jobs have a time limit. When a job is stopped and resubmitted, this finds the existing checkpoint and continues, like `train.ps1 -Resume`.

```bash
COMMON=( --dataset.repo_id=... --eval_steps=2000 ... )
case "$POLICY" in
    diffusion) PRESET=( --policy.type=diffusion ... ) ;;
    smolvla)   PRESET=( --policy.path=lerobot/smolvla_base '--rename_map={...}' --batch_size=32 ... ) ;;
esac
srun lerobot-train "${COMMON[@]}" "${PRESET[@]}"
```

Like `train.ps1`: "arguments shared by all policies" plus "per-policy preset". `( ... )` is a bash array and `"${COMMON[@]}"` expands it into separate arguments.
The only difference of the cluster's diffusion preset from the laptop's is **no halving of the U-Net channels**: the V100's 16 GB fits the paper's default 278M parameters.
The smolvla preset differs from the laptop's in batch 32 instead of 8, 20k steps instead of 30k, and a checkpoint every 5000 steps. The V100 runs bf16 slowly (2.47 s/step), so 20k steps take about 14 h, split over two 8 h jobs: resubmit after the first times out and it resumes from the last checkpoint, losing at most 5000 steps.
The `rename_map` argument is wrapped in single quotes: in bash, `"` inside single quotes passes through unchanged (in PowerShell 5.1 it must be written `\"`, see §7).

PS: `.sh` and `.sbatch` files need Linux line endings (LF), or bash fails with `$'\r': command not found`.
`*.sh text eol=lf` in `.gitattributes` guarantees that.

## 11. `scripts/make_gif.py`

**Problem:** the README needs animations of the robot actually working. Evaluation videos are in the dataset, but LeRobot concatenates many episodes
into **one MP4** per camera, so they can't be used directly.

### Finding the episode's segment in the video

```python
row = meta[meta["episode_index"] == episode].iloc[0]
key = f"videos/observation.images.{camera}"
path = ds / "videos" / ... / f"file-{file:03d}.mp4"
return path, row[f"{key}/from_timestamp"], row[f"{key}/to_timestamp"]
```

The dataset's `meta/episodes/*.parquet` is a table with one row per episode, recording which video file it is in and from which second to which.

### Cutting, joining and converting to GIF with ffmpeg

```python
inputs += ["-ss", start, "-to", end, "-i", path]            # cut one segment per camera
labels.append(f"[{i}:v]setpts=PTS/{speed},fps={fps},scale=-2:{height}[v{i}]")
graph = ";".join(labels) + ";[v0][v1]hstack=inputs=2,split[a][b];[a]palettegen[p];[b][p]paletteuse"
```

| Filter | Effect |
|---|---|
| `setpts=PTS/2` | divide timestamps by 2, i.e. play at 2× speed |
| `fps=10` | down to 10 frames per second (more frames = bigger GIF) |
| `scale=-2:240` | height 240, width in proportion (`-2` keeps it even) |
| `hstack` | put the two cameras side by side |
| `palettegen` + `paletteuse` | a GIF has only 256 colours. First compute the best palette for the whole clip, then convert with it: far less banding |

- 📘 `filter_complex` is ffmpeg's filter graph: `[0:v]` is the video of the first input, names like `[v0]` are our own intermediate results, and `;` separates the steps.
- 📘 `subprocess.run([...], check=True)`: run an external program from Python; `check=True` raises if it fails.
- Each GIF is about 3 MB.

## 12. `scripts/plot_results.py`

**Problem:** the README chart has to be updated after every policy's evaluation. Drawn by hand it easily disagrees with the sheets.

**Idea:** read only the tables in `results/<policy>.md` and generate `assets/results.svg`. Re-run it after each evaluation.

```python
ROW = re.compile(r"^\|\s*\d+\s*\|\s*([①②③④⑤])\s*\|\s*(✅|🟡|❌)\s*\|")
```

- 📘 A regular expression matching rows like `| 12 | ② | ✅ |`. `\d+` is digits, `\s*` any whitespace, `( ... )` the parts to extract:
  here the spot and the result.
- The SVG is written as text: SVG is a text format that describes graphics in XML; `<rect>` is a rectangle (a bar), `<text>` is text.
  GitHub displays SVG directly, so no matplotlib needed.
- Only the file names listed in `ORDER` are read (`act.md`, `diffusion.md`, ...); sheets with other names are not drawn.
- `trials_label()`: "how many trials per spot" in the title is counted from the data instead of hard-coded. `8 trials each` when all policies have the same count,
  a range (e.g. `4-8 trials each`) otherwise, so the title never disagrees with the chart.

## 13. `cloud/hf_jobs/train_pi05.py`

**Problem:** π₀.₅ has 4.1B trainable parameters; full float32 fine-tuning needs about 67 GB of VRAM. Neither the laptop nor a V100 can hold it, so we rent an A100 from Hugging Face.

**LeRobot has remote training built in:** add `--job.target=a100-large` to the training command and `lerobot-train` doesn't train locally; it:
1. resolves the config locally and checks the dataset is on the Hub;
2. writes the config as `train_config.json` and uploads it to a new private model repo;
3. starts a job on HF that runs `lerobot-train --config_path=<that repo>` in the official image;
4. pushes the trained model to that repo.

### Why build the arguments in Python

```python
f"--rename_map={json.dumps(RENAME)}"
```

`rename_map` is JSON and PowerShell 5.1 swallows its quotes (§7). Building `sys.argv` in Python and calling `lerobot_train.main()` avoids the quoting problem.

### Fixing Windows paths

```python
def build(cfg, repo_id, dest, tags=None):
    path = original(cfg, repo_id, dest, tags=tags)
    ...parent[key] = parent[key].replace("\\", "/")
hf_jobs.build_remote_config_file = _fix_windows_paths(hf_jobs.build_remote_config_file)
```

LeRobot stores `--policy.path` as a `Path`, which on Windows ends up in the JSON as `lerobot\pi05_base`. The remote side is Linux and can't read it,
so the job would fail immediately. This wraps the function that writes the config and, after it writes the file, turns `\` back into `/` in `pretrained_path` and `output_dir`.
Same problem as the patch in §8, only at the step of "sending the config to another machine".

### `--dry-run`

Replaces the submit function with "just print the config", so the JSON being sent can be checked for free. Dry-run after every change.

### Key arguments

| Argument | Value | Why |
|---|---|---|
| `steps` / `scheduler_decay_steps` | 3000 / 3000 | the default 30k steps take ~70 h ($175). Only the step count changes, and the LR schedule completes over the new length; everything else stays default |
| `gradient_checkpointing` | true | don't keep intermediate activations, recompute them in the backward pass: time for memory, mathematically identical |
| `num_workers` | 10 | with the default 4 workers the GPU waited 1.3 s per step for data; the machine has 12 CPUs |
| `save_freq` | = steps | a full checkpoint with optimizer state is ~66 GB, so save only once at the end |
| `eval_steps` | 0 | no validation on a paid GPU |
| `--job.timeout` | 9h | ~7.3 h expected plus margin; queueing doesn't count |

---

## 14. `zero_shot/`: the training-free baseline (VLM agent)

Design and results are in [09_zero_shot_vlm.md](09_zero_shot_vlm.md); this section is about the code. Run the modules as `python -m zero_shot.<file>`
from the repo root (`-m` makes Python treat `zero_shot` as a package, so its files can import each other); the two calibration scripts also run as plain files.

| File | Role |
|---|---|
| `kinematics.py` | FK / IK on the SO-101 URDF |
| `calibrate_touch.py` | touch calibration: teleoperate, touch tape squares, record tip positions |
| `fit_touch_calib.py` | fit lens distortion + homography → `camera_calib.json` |
| `measure_gripper.py` | fingertip gap at each opening → `gripper_geometry.json` |
| `agent.py` | the Claude Haiku agent: tools, motion and safety layer, the loop |

### `kinematics.py`: forward / inverse kinematics

**Reading the URDF:** URDF is an XML description of a robot. Each `<joint>` has a parent link, a child link, `<origin xyz rpy>` (the joint's position and orientation in the parent frame)
and `<axis>` (the rotation axis). Walking from the gripper tip back through the parents to the base gives the chain of 6 joints.

**Forward kinematics (FK):** starting at the base, multiply for each joint the two 4×4 matrices "fixed offset × rotation by q about the axis";
the result is the tip's position and orientation in the base frame:

```python
for j in CHAIN:
    T = T @ j.origin                      # fixed offset (URDF xyz / rpy)
    if j.movable:
        R[:3, :3] = _axis_angle(j.axis, angles[j.name])   # rotate by the joint angle about the axis (Rodrigues)
        T = T @ R
```

**Inverse kinematics (IK):** given a target, find the joint angles. No closed form, so iterate:

```python
r = residual(q)                 # [position error (m), 3-D; 0.05 × (gripper direction − straight down), 3-D]
J[:, k] = (residual(q + dq) - r) / 0.01         # numerical Jacobian: move each joint 0.01°, see how r changes
step = J.T @ solve(J @ J.T + λ² I, -r)          # damped least squares (DLS)
```

- 📘 **Jacobian** J: the derivative of the residual with respect to each joint angle. Newton's method would use J's inverse directly, but with the arm stretched out or near a limit J is nearly singular
  and the step becomes absurdly large. Adding λ²I (damping) keeps the step bounded: that's DLS (also known as Levenberg–Marquardt).
- **λ depends on the units**: 1° of joint motion moves the tip only about 3 mm, so J's entries are about 3e-3 and λ has to be around 1e-3; with a large λ (e.g. 0.05) λ² dwarfs JJᵀ and the solver barely moves.
- **Why only 4 joints**: wrist_roll only spins the gripper about its own axis; it changes neither the tip position nor the downward direction, so it stays fixed.
- **`solve_seeded`, the starting points**: from the folded pose shoulder_lift sits exactly at its −100° limit; the first step is clipped back to the limit and it's stuck.
  So IK is also started from an even grid (shoulder_lift, elbow_flex ∈ {−60, −20, 20, 60}°, wrist_flex ∈ {30, 70}°, 32 seeds, shoulder_pan aimed at the target with `arctan2`),
  plus the current pose. Keep the smallest error; among answers within 1 mm of it, the one whose largest joint change from now is smallest (so the arm doesn't swing round for nothing).
  Nothing here comes from the demos.

### `calibrate_touch.py`: touch calibration

- **Teleoperation inside the script**: the leader arm (`SO101Leader`) and the follower are connected directly, and every 1/30 s `follower.send_action(leader.get_action())`.
  The script has to own both serial ports itself, because it also reads the follower's joints at the moment of a touch (`lerobot-teleoperate` would hold the ports).
- 📘 **Reading the keyboard without stopping the loop**: `input()` would block teleoperation. A background thread (`threading.Thread(daemon=True)`) reads lines from `sys.stdin`
  and puts them in a `queue.Queue`; the loop checks it with `get_nowait()` every cycle. `daemon=True` lets Python exit without waiting for that thread.
- On Enter: wait 0.3 s for the arm to settle, read the joints, FK → tip xyz, save a photo `touch_<n>.jpg`. `u` undoes the last point, `q` ends;
  the first Enter saves `clean.jpg` (arm out of the way) for finding the tape squares.
- **Exit**: disconnecting releases torque, so the instructions say to fold the arm to its rest pose with the leader *before* typing `q`.

### `fit_touch_calib.py`: distortion + homography

- **Finding the squares** (done once and saved as `pairs.json`): grey image minus its Gaussian blur (σ = 15 px) = how much darker each pixel is than its surroundings;
  > 45 levels → binary; `cv2.connectedComponentsWithStats` gives each blob's area, size and centroid; keep square-ish blobs of 40–500 px.
  Each touch is matched to the nearest square after mapping it through a first rough fit, then refitted (a few rounds).
- `undistort`: the inverse of $u_d = c + (u - c)(1 + k_1 r^2 + k_2 r^4)$ has no closed form, so 30 rounds of fixed-point iteration.
- `fit`: 📘 `scipy.optimize.minimize(..., method="Nelder-Mead")` searches $(k_1, k_2)$ without gradients (a simplex of 3 points that reflects / shrinks towards lower cost);
  inside the cost, `cv2.findHomography(..., 0)` fits $H$ by plain least squares on the undistorted points and returns the mean squared error in metres.
- **Leave-one-out**: refit without point *t*, predict *t*; the median of these is the honest accuracy (8.3 mm here). Printed per point, so a bad touch stands out.
- Output `camera_calib.json`: $k$, $H$, the table height (median tip z over the touches) and both error figures.

### `measure_gripper.py`

Lift to (20, 0, 12) cm with the agent's own planner, step the opening through 0 / 25 / 50 / 75 / 100, and ask for the ruler reading each time → `gripper_geometry.json`.
In a `finally` block the gripper goes back to its start value and the arm goes home, even after Ctrl+C.

### `agent.py`: the agent

**Geometry helpers.**
- `table_to_pixel`: the inverse direction of the calibration (table → undistorted pixel via $H^{-1}$, then `distort`), used to draw the grid and the fingertip marks.
- `jaw_dir`: the direction from the fixed to the moving fingertip = the tip frame's $-x$ axis (`-fk(q)[:3, 0]`), flattened onto the table.
- `fingertips_m`: the fixed fingertip (FK) and the moving one (fixed + direction × gap at the opening `q[5]`, `np.interp` over the measured table).
- `tip_cm`: the midpoint of the two fingertips, in table cm; this is the "gripper position" the model sees and gives.

**Planning a move (`plan_line` → `plan_tip`).** The model's target is a midpoint; the IK tip is the fixed finger, so the target is shifted by half the gap against `jaw_dir`.
`jaw_dir` depends on the final pose, so: plan, take the end pose, recompute the shift, plan again. `plan_tip` then tries, in order:
1. `_line`: the tip along a straight line, IK every 1 cm, each point warm-started from the previous one; fingers down (`orient_weight` 0.05), or tilted (0.01) where vertical is impossible;
2. `solve_seeded` + one joint-space jump (e.g. leaving the folded home pose, where the line fails);
3. straight up to 15 cm, then the line.
Each candidate goes through `_clear`: FK every 2° along the interpolation, tip never below the table (or, starting below that, never lower than the start). None safe → `ValueError`, which becomes the tool's error message.

**`Robot`.** Follower + both cameras via OpenCV (4 frames dropped before each picture, so it isn't a stale buffered one). `play` interpolates in joint space at ≤ 45°/s,
then **waits until the joints are within 1.5° of the target** (≤ 2 s): the servos trail the commands, and the first real run showed the arm 2–4 cm short when the picture was taken right away.
With `--dry-run` nothing is connected: the images are a recorded frame and "measured" joints are the last command, which tests the API plumbing and the planner for a cent.
`home` is the pose read at start.

**Tools (`TOOLS`, `run_tool`).** JSON-schema tool definitions with `strict: True` (the API then guarantees the arguments match the schema, e.g. numbers really are numbers).
`move_to` / `gripper` act, then return `observation`: overhead image with the grid (`draw_grid`), wrist image, and a text line with the gripper position and the commanded and measured opening.
If a move ends > 2 cm from its target, the text says it may be blocked. `--step` asks for Enter before each motion.

**The loop (`attempt`).** A manual tool loop on the Messages API:

```python
resp = client.messages.create(model="claude-haiku-5-5", system=SYSTEM, tools=TOOLS,
                              output_config={"effort": effort}, messages=messages, max_tokens=4000)
messages.append({"role": "assistant", "content": resp.content})     # appended unchanged
... run every tool_use block, collect tool_result blocks ...
messages.append({"role": "user", "content": results})
```

- The assistant's content is appended **unchanged**: Haiku 5.5 thinks adaptively and its thinking blocks must be sent back exactly as received; editing earlier turns invalidates them.
- 📘 **Images in a tool result**: a `tool_result`'s `content` can be a list of blocks, here text + `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", ...}}`.
- A response may contain several tool calls (e.g. `move_to` then `gripper`); they run in order and all results go back in **one** user message.
- Stops on `done`, a response without tool calls (or `stop_reason == "refusal"`), or `--max-actions` (25).
- `record.json` is rewritten after every action (each call, its result, the model's verdict, tokens, cost at $0.10 / $0.50 per M), so a run stopped with Ctrl+C still leaves its log.
- After each attempt `go_home`: straight up to 15 cm (clears the box walls), then joint interpolation home; also in `finally`, so Ctrl+C never drops the arm mid-air.
- The API key is read by the SDK from `ANTHROPIC_API_KEY`; the code never handles it. `anthropic.Anthropic(max_retries=8)`: a dropped connection is retried with backoff
  (the SDK default of 2 once let a connection reset end a trial); this changes nothing in what the model sees.

**Running an evaluation (`main`).** Trial numbers continue from the `trial_*/record.json` already in the output folder, so 8 runs of `--trials 5` make trials 1–40,
and the spot follows the number (①..⑤). Between trials a 25 s countdown (`--reset-time`) replaces a key press.
`--video` starts a thread that reads both cameras at 30 fps and writes them side by side to an MP4 (`cv2.VideoWriter`); while it runs, `images()` takes its newest frames
under a `threading.Lock` instead of reading the cameras itself, because two threads must not read the same `VideoCapture`.

---

## 15. Remote inference: `make_server_config.py` and `cloud/hf_jobs/serve_pi05.py`

π₀.₅ needs about 17 GB of GPU memory in float32, and the laptop has 8 GB. LeRobot 0.7 can split a rollout in two:
`lerobot-policy-server` loads the model on a GPU machine, and `lerobot-rollout --inference.type=remote` runs the arm and cameras on the laptop.
The two talk over [Zenoh](https://zenoh.io/) on TCP port 7447. We put the server on an HF Jobs GPU and reach it through an SSH tunnel.

### `scripts/make_server_config.py`: the server's contract

The server won't guess: every input must be declared with its name, shape, dtype and a "semantics" string, and the action with its joint order.
The script reads all of it from files that already have it:

- the checkpoint's `policy_preprocessor.json` → the rename map (`observation.images.top` → `observation.images.base_0_rgb`, ...);
- the dataset's `meta/info.json` → camera size (480×640×3, `uint8`, `kind: rgb`), joint names for the state and the action, and the fps.

Each input is declared under the name **the model** expects, because the server replaces the checkpoint's rename step with an empty one.
The laptop maps its camera names onto those with `--rename_map` (§4). `SEMANTICS = "so101_follower_v1"` is just a label that both sides must agree on.
`--model-path` is where the checkpoint will be on the server (`/tmp/pi05` in the job). The listen endpoint `tcp/127.0.0.1:7447` is local to the server machine, so it can only be reached through the tunnel.

### `cloud/hf_jobs/serve_pi05.py`: start the server on HF Jobs

`HfApi().run_job(...)` with `ssh=True` starts a container on `a10g-large` and gives it an SSH address, `<job_id>@ssh.hf.jobs`. The SSH key is the one added at huggingface.co/settings/keys.
The container runs a short bash script:

1. Install `lerobot[pi,remote]==0.7.0`. The image ships a newer LeRobot, but the remote-inference protocol is versioned, so the server must match the laptop.
2. Download the public checkpoint into `/tmp/pi05` with `hf download` (16.6 GB).
3. Write the server config (passed in as the `SERVER_CONFIG` environment variable) and the launcher (`LAUNCHER`) to files.
4. `exec` the launcher. The server becomes the container's main process, so `hf jobs cancel` stops it.

The submit script then waits until the job is `RUNNING` and prints the exact `ssh -N -L 7447:127.0.0.1:7447 ...` command. 📘 `-N` means "run no remote command, only forward ports"; `-L local:host:remote` forwards the laptop's port 7447 to 127.0.0.1:7447 inside the container.

Why `a10g-large` rather than `a10g-small`: both have a 24 GB A10G, but loading a 16.6 GB float32 checkpoint needs more than the small flavor's 15 GB of RAM.

### `cloud/hf_jobs/pi05_server_launch.py`: one missing override

Before serving, LeRobot 0.7 calls `policy.validate_chunk_input_features(...)`. The default check insists that the declared inputs match the checkpoint's `input_features` exactly: same names, same shapes.
π₀.₅'s config lists three 3×224×224 cameras and a 32-dim state. The robot sends two 480×640×3 cameras and 6 joints, so the first job stopped with
`Canonical feature names must exactly match policy input_features`.

Nothing is actually wrong. At run time π₀.₅ resizes each image to 224×224 with padding, pads the state to 32 dims, and fills a missing camera with an empty image that is masked out of attention.
That is exactly what happens in local inference. SmolVLA is in the same situation and overrides the check; π₀.₅ doesn't yet.
The launcher replaces the check with the same rules as SmolVLA's:

- every non-camera input is present;
- only known camera names;
- at least one camera;
- cameras are H×W×3 RGB;
- the state is no longer than `max_state_dim`.

Then it calls the normal `lerobot-policy-server` `main()`. The model, the processors and the server logic are untouched.
