# 01 · Hardware: serial ports, servo IDs, cameras

**English** · [简体中文](01_hardware.zh-CN.md)

Goal: find out **which COM port is the leader arm and which is the follower**, confirm that all 6 servo IDs are set,
and find the indices of the two cameras.

## Overview

```
        ┌──────────────┐  USB   ┌──────────────────────┐  3-pin daisy chain  ┌──────────────────┐
laptop ─┤ COMx (CH343) ├────────┤ servo board + 12V/5V ├─────────────────────┤ 6 × STS3215 servo │  follower
        │ COMy (CH340) ├────────┤ servo board + 5V     ├─────────────────────┤ 6 × STS3215 servo │  leader (hand-held)
        │ USB cam ×2   ├── top: overhead camera on a desk-lamp arm
        │              ├── wrist: camera on the follower's gripper
        └──────────────┘
```

- **Follower**: the arm with the gripper, driven by the policy or by the leader.
- **Leader**: the arm with the orange handle, moved by hand for teleoperation. Its servos have no torque; they only read angles.

## 1. List serial ports and cameras

```powershell
conda activate lerobot
python scripts/find_hardware.py
```

Example output:

```
=== Serial ports ===
  COM3     1A86:7523  serial=-            WCH CH340
  COM4     1A86:55D3  serial=5B79032576   WCH CH343 (Waveshare / Feetech bus servo board)
```

The two boards use different USB-serial chips (CH340 and CH343). After moving a cable to another USB port the COM
number may change, but the chip does not. So if the COM numbers change, run this script again and match them by chip.

PS: the CH340 needs WCH's CH341SER driver. Windows 11 normally installs it through Windows Update; if Device Manager
shows a yellow warning sign, download it from WCH's website.

## 2. Tell leader and follower apart

LeRobot ships a tool for this: unplug one arm when asked, and it tells you which port disappeared.

```powershell
lerobot-find-port
```

The more direct way is `--watch` from step 3: move one arm by hand and see which port's readings change.
Here it is **COM3 = follower, COM4 = leader**.

Write the result into `config/robot.json`:

```powershell
Copy-Item config/robot.example.json config/robot.json
notepad config/robot.json
```

## 3. Check the servo IDs (read-only, writes nothing)

`lerobot-setup-motors` writes each servo's ID (shoulder_pan=1 … gripper=6) and baud rate (1 Mbps) into **the servo's
own EEPROM**. It survives power-off, another computer, another OS. So once the IDs are set you never set them again;
just scan to confirm:

```powershell
python scripts/scan_motors.py --port COM3
python scripts/scan_motors.py --port COM4
```

All 6 should say `ok`:

```
  joint          id  status     model  position  voltage
  shoulder_pan    1  ok           777      2048    12.1 V
  ...
  gripper         6  ok           777      2010    12.1 V
All 6 servos answered with the right IDs: no need to run lerobot-setup-motors.
```

`--watch` shows the raw encoder readings (0–4095) live. Move a joint by hand and you can see that ID and joint match:

```powershell
python scripts/scan_motors.py --port COM3 --watch
```

### If a servo is MISSING

1. Check the wiring first: power, USB, the 3-pin daisy-chain cables. On the Waveshare board both jumpers must be on **channel B (USB)**.
2. Wiring is fine and it is still missing: the ID really is not set. Use the official procedure, which asks you to connect one servo at a time:

```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM3
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=COM4
```

## 4. Find the camera indices

`find_hardware.py` saves one snapshot per camera it can open to `outputs/hardware_check/camera_<index>.jpg`.
Open them, see which index is the overhead view (`top`) and which is the wrist view (`wrist`); the laptop's built-in camera is not used.
Put the indices into `cameras.top.index_or_path` and `cameras.wrist.index_or_path` in `config/robot.json`.

⚠️ The two external cameras are the same model (JYU2C-2083) with identical device names, so only the snapshots tell
them apart. And OpenCV camera indices on Windows are not stable: re-plugging, another USB port or a reboot can swap them. So:

1. always plug the two cameras into the same USB ports;
2. **run `find_hardware.py` and look at the snapshots before every recording session.**

PS: if top and wrist get swapped, one dataset ends up with the two views mixed, and a policy trained on it is basically useless.

LeRobot's equivalent tool:

```powershell
lerobot-find-cameras opencv
```
