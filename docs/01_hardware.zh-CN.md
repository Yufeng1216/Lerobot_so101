# 01 · 硬件识别：串口、舵机 ID、摄像头

[English](01_hardware.md) · **简体中文**

目标：搞清楚**哪个 COM 口是主臂（leader）、哪个是从臂（follower）**，确认 6 个舵机的 ID 都已经设好，再找到两个摄像头的编号。

## 硬件一览

```
        ┌──────────────┐  USB   ┌────────────────────┐  3-pin 菊花链  ┌───────────────────┐
笔记本 ─┤ COMx (CH343) ├────────┤ 舵机驱动板 + 12V/5V ├───────────────┤ 6 × STS3215 舵机   │  从臂 follower
        │ COMy (CH340) ├────────┤ 舵机驱动板 + 5V     ├───────────────┤ 6 × STS3215 舵机   │  主臂 leader（手持）
        │ USB 摄像头 ×2 ├── top：台灯支架上的俯视摄像头
        │              ├── wrist：装在从臂夹爪上的腕部摄像头
        └──────────────┘
```

- **从臂（follower）**：带夹爪的那条，由策略或主臂驱动。
- **主臂（leader）**：带橙色手柄的那条，人手拿着做遥操作。它的舵机不上力矩，只负责读角度。

## 1. 列出串口和摄像头

```powershell
conda activate lerobot
python scripts/find_hardware.py
```

输出示例：

```
=== Serial ports ===
  COM3     1A86:7523  serial=-            WCH CH340
  COM4     1A86:55D3  serial=5B79032576   WCH CH343 (Waveshare / Feetech bus servo board)
```

两块板子用的 USB 转串口芯片不一样（CH340 和 CH343）。换了 USB 口，COM 号可能会变，但芯片不会变。
所以 COM 号变了的话，重跑一次这个脚本，按芯片对回去就行。

PS：CH340 要用 WCH 的 CH341SER 驱动，Windows 11 一般会通过 Windows Update 自动装。设备管理器里出现黄色感叹号的话，去 WCH 官网下载。

## 2. 区分主臂和从臂

LeRobot 自带一个工具：按提示拔掉其中一条臂的 USB 线，它会告诉你消失的是哪个口。

```powershell
lerobot-find-port
```

更直观的办法是用第 3 步的 `--watch`：用手掰哪条臂，哪个口的读数就会变。我们这里是 **COM3 = 从臂，COM4 = 主臂**。

把结果写进 `config/robot.json`：

```powershell
Copy-Item config/robot.example.json config/robot.json
notepad config/robot.json
```

## 3. 确认舵机 ID（只读，不写任何东西）

`lerobot-setup-motors` 会把舵机 ID（shoulder_pan=1 … gripper=6）和波特率（1 Mbps）写进**舵机自己的 EEPROM**。
断电、换电脑、换系统都不会丢。所以 ID 设过一次就不用再设，先扫一遍确认就好：

```powershell
python scripts/scan_motors.py --port COM3
python scripts/scan_motors.py --port COM4
```

正常的话 6 个都是 `ok`：

```
  joint          id  status     model  position  voltage
  shoulder_pan    1  ok           777      2048    12.1 V
  ...
  gripper         6  ok           777      2010    12.1 V
All 6 servos answered with the right IDs: no need to run lerobot-setup-motors.
```

加 `--watch` 可以实时看原始编码器读数（0–4095）。用手掰关节，就能确认 ID 和关节是不是对得上：

```powershell
python scripts/scan_motors.py --port COM3 --watch
```

### 如果有舵机 MISSING

1. 先查线：电源、USB、3-pin 菊花链线有没有插紧。Waveshare 驱动板的两个跳线帽要在 **B 通道（USB）**。
2. 线没问题还是缺，说明 ID 确实没设。按官方流程重设，它会提示你每次只接一个舵机：

```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM3
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=COM4
```

## 4. 找摄像头编号

`find_hardware.py` 会给每个能打开的摄像头存一张截图到 `outputs/hardware_check/camera_<编号>.jpg`。
打开截图，看哪个编号是俯视（`top`）、哪个是腕部（`wrist`），笔记本自带的摄像头不用。
然后把编号分别填进 `config/robot.json` 的 `cameras.top.index_or_path` 和 `cameras.wrist.index_or_path`。

⚠️ 两个外接摄像头是同一型号（JYU2C-2083），设备名一模一样，只能靠截图区分。而且 Windows 下 OpenCV 的摄像头编号不固定，
重新插拔、换 USB 口、重启之后都可能互换。所以：

1. 两个摄像头固定插在同一组 USB 口上；
2. **每次录数据前都先跑一次 `find_hardware.py`，看截图确认编号。**

PS：top 和 wrist 一旦录反，同一个数据集里两路画面就混在一起了，训出来的策略基本没法用。

LeRobot 自带的等价工具：

```powershell
lerobot-find-cameras opencv
```
