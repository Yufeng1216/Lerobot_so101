# 02 · Calibration and teleoperation

**English** · [简体中文](02_calibration_teleop.zh-CN.md)

Goal: make the leader and follower "speak the same language", then drive the follower live with the leader.

Our setup (`config/robot.json`): **follower = COM3** (CH340), **leader = COM4** (CH343).

## 1. What calibration does

Each servo has a 12-bit magnetic encoder; 0–4095 is one turn. The problem is that the same physical pose gives
different raw readings on the two arms: the servos were mounted at different angles, and the leader's gear ratio
differs from the follower's. So sending the leader's raw readings to the follower does not line up.

So calibration records two things per joint:

1. **Homing offset**: defines "the middle of the joint's range" as the midpoint;
2. **Range (min / max)**: you move the joint to both ends by hand and the lowest and highest readings are stored.

With these, LeRobot **normalises** raw readings to `[-100, 100]` (`[0, 100]` for the gripper). It is really the same
idea as input normalisation in deep learning: once both arms are normalised, "leader elbow at 30" means "follower elbow
at 30", and teleoperation maps one to one. A trained policy also transfers to another calibrated arm of the same model,
because what it learned are normalised angles.

The result is saved as JSON, named after `--robot.id` / `--teleop.id`:

- `config/calibration/robots/so_follower/so101_follower.json`
- `config/calibration/teleoperators/so_leader/so101_leader.json`

PS: by default LeRobot keeps calibration in the global cache `$HF_HOME/lerobot/calibration`. `scripts/env.ps1` sets
`HF_LEROBOT_CALIBRATION` so it reads and writes this folder in the repo instead. LeRobot finds the file by id, so
**every later command must use the same id** (here always `so101_follower` and `so101_leader`); `scripts/robot.ps1`
takes care of that for you.

## 2. Calibrate the follower (COM3)

```powershell
conda activate lerobot
.\scripts\robot.ps1 calibrate-follower
```

Same as:

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower
```

The terminal prints a pose sketch, then:

1. **Put the arm in the middle pose**: upper arm straight up, forearm horizontal pointing forward, wrist in line with
   the forearm, gripper closed (as in the ASCII sketch). Torque is off, so you can move it freely. Press **Enter**.
2. **Move every joint through its full range**: a live table shows MIN / POS / MAX per joint. Move each joint
   (except `wrist_roll`) slowly from one end to the other and watch MIN and MAX spread out.
   - `wrist_roll` turns all the way round; the full range is used, no need to move it.
   - Stop at the mechanical limit. **Don't force it.**
3. Press **Enter** when done. The calibration file is saved.

PS: if you forget a joint, its MIN equals its MAX and you get `a motor did not move`. Just run it again.

## 3. Calibrate the leader (COM4)

```powershell
.\scripts\robot.ps1 calibrate-leader
```

Exactly the same steps. The leader's "gripper" is the trigger on the orange handle.

### Why the calibration files are in git

A recorded dataset is **normalised with this calibration**. Change the calibration and the angles in the dataset no
longer match the arm, and a trained policy loses its hand-eye coordination. So the calibration files are versioned
together with the code. **Don't recalibrate casually**; if you must (say, a servo was replaced), re-evaluate the
datasets and models made before.

PS: when you run `lerobot-calibrate` and friends directly instead of through `scripts/robot.ps1`, run
`. .\scripts\env.ps1` first, or LeRobot looks for calibration in the global cache.

Our calibration (encoder ticks, 0–4095 = one turn):

| Joint | Follower range | Leader range |
|---|---|---|
| shoulder_pan | 883 – 3319 | 828 – 3233 |
| shoulder_lift | 865 – 3273 | 819 – 3225 |
| elbow_flex | 820 – 3062 | 874 – 3090 |
| wrist_flex | 809 – 3151 | 914 – 3237 |
| wrist_roll | 0 – 4095 (full turn) | 0 – 4095 (full turn) |
| gripper | 1777 – 3315 | 2026 – 3275 |

## 4. Teleoperation (no cameras)

⚠️ The moment teleoperation starts, the follower **jumps** to the leader's current pose. So before starting, put the
leader in roughly the same pose as the follower, and keep a hand near the follower's power supply; unplug it if anything goes wrong.

```powershell
.\scripts\robot.ps1 teleop
```

Move the leader; the follower should follow. `Ctrl+C` to stop.

Check:
- [ ] all 6 joints move the right way (leader goes left, follower goes left)
- [ ] the gripper opens and closes and can hold the toy
- [ ] no obvious jitter or lag

## 5. Teleoperation with camera views

```powershell
.\scripts\robot.ps1 teleop -Cameras
```

With `-Cameras` the script adds `--robot.cameras=...` and `--display_data=true`, and LeRobot opens a **Rerun** window
with the top and wrist views and the joint curves. **This is everything the policy will "see" later**, so check here before recording:

- [ ] the top view covers the whole workspace (toy, box, follower)
- [ ] the wrist view shows the gripper tips and the object
- [ ] the two views are not swapped

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not connect on port 'COM3'`, or Windows error 31 "device is not functioning" | The CH340 is stuck. Unplug the USB cable, wait 3 s, plug it back in |
| every servo MISSING in the scan | The servo board has no power (USB only carries data) |
| follower moves the opposite way, or by the wrong amount | The middle pose was off during calibration; recalibrate that arm |
| camera won't open / black image | Its index changed. Run `python scripts/find_hardware.py`, look at the snapshots, fix `config/robot.json` |
| `Failed to open OpenCVCamera(2)` although single snapshots work | Two possible causes: ① two cameras streaming uncompressed frames run out of USB bandwidth, so the config sets `"fourcc": "MJPG"` (compressed); ② the wrist camera's cable gets pulled loose as the arm moves. Leave slack, tie it to the arm, and plug it straight into the laptop if needed |
