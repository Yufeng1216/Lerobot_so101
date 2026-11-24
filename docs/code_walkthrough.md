# 代码讲解：从硬件识别到训练评估

这篇按**项目推进的时间顺序**，把仓库里每一个脚本逐段讲清楚：它解决什么问题、每段代码在做什么、
为什么这样写，以及它调用的 LeRobot 内部代码在做什么。

和其他教程的分工：`00`–`08` 讲**怎么操作**；这篇讲**代码在干什么**。
Python / PowerShell 语法在第一次出现时解释（标为 📘）。

| 阶段 | 脚本 | 一句话 |
|---|---|---|
| 1 硬件 | [`find_hardware.py`](#1-scriptsfind_hardwarepy) | 列出串口和相机，给每个相机拍照 |
| 1 硬件 | [`scan_motors.py`](#2-scriptsscan_motorspy) | 只读地确认 6 个舵机的 ID |
| 2 配置 | [`env.ps1`](#3-scriptsenvps1) | 让 LeRobot 把数据、标定存进本仓库 |
| 2 配置 | [`robot.ps1`](#4-scriptsrobotps1) | 从配置文件拼出标定 / 遥操作 / 录制 / 部署命令 |
| 3 数据 | [`check_cameras.py`](#5-scriptscheck_cameraspy) | 录制前的相机预检查 |
| 3 数据 | [`inspect_dataset.py`](#6-scriptsinspect_datasetpy) | 录制后的数据质量报告 |
| 4 训练 | [`bench_dataloader.py`](#7-scriptsbench_dataloaderpy) | 找出最快的数据加载设置 |
| 4 训练 | [`train.ps1`](#8-scriptstrainps1) | 训练入口和每个策略的参数预设 |
| 4 训练 | [`lerobot_train_win.py`](#9-scriptslerobot_train_winpy) | 修复 Windows 上的 checkpoint 链接问题 |
| 5 集群 | [`cluster/*`](#11-clustercluster集群训练) | 在 Slurm cluster 上装环境、提交大模型训练 |
| 内部 | [LeRobot 内部做了什么](#10-lerobot-内部我们调用的命令在做什么) | 录制循环、数据集、训练循环、ACT 前向 |

---

## 1. `scripts/find_hardware.py`

**问题：** 插上两块舵机驱动板和三个相机后，Windows 给它们分配了 COM 号和相机编号，但不告诉你哪个是哪个。

### 文件开头的文档字符串

```python
"""List the serial ports and cameras this computer can see.
...
"""
```

📘 三引号 `"""..."""` 是 Python 的多行字符串。放在文件或函数最开头时叫 **docstring**（文档字符串），
是给人看的说明。`python scripts/find_hardware.py --help` 会把它打印出来（见下面的 `argparse`）。

### 导入

```python
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from serial.tools import list_ports
```

- 📘 `import x` 加载一个库；`from x import y` 只取其中的 `y`。
- `from __future__ import annotations`：让类型标注（如 `-> None`）延迟解析，是写新式 Python 的习惯写法，不影响运行逻辑。
- `argparse`：解析命令行参数（`--max-cameras 6` 这种）。
- `Path`：处理文件路径的类，`Path("outputs") / "a.jpg"` 会自动拼成正确的路径，Windows 和 Linux 通用。
- `cv2`：OpenCV，读相机。
- `serial.tools.list_ports`：pyserial 自带的串口枚举工具。

### 已知芯片表

```python
KNOWN_CHIPS = {
    (0x1A86, 0x55D3): "WCH CH343 (Waveshare / Feetech bus servo board)",
    (0x1A86, 0x7523): "WCH CH340",
    ...
}
```

📘 `{键: 值}` 是**字典**（dict）。这里的键是一个**元组** `(VID, PID)`。`0x1A86` 是十六进制数字。

每个 USB 设备都有两个编号：**VID**（厂商号，`1A86` = 沁恒 WCH）和 **PID**（产品号）。
COM 号会随插拔变化，但 VID:PID 是写死在芯片里的，所以用它来认板子最可靠。
本项目的两块板正好是不同芯片（CH343 和 CH340），这就是我们区分它们的依据。

### 列串口

```python
def list_serial_ports() -> None:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    for p in ports:
        chip = KNOWN_CHIPS.get((p.vid, p.pid), "unknown chip") if p.vid else "no USB VID/PID"
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid else "----:----"
        print(f"  {p.device:<8} {vidpid}  ...")
```

- 📘 `def 名字(参数) -> 返回类型:` 定义函数。`-> None` 表示不返回值。
- 📘 `sorted(列表, key=lambda p: p.device)`：按 `p.device`（如 `COM3`）排序。
  `lambda p: p.device` 是一个**匿名小函数**：输入 `p`，返回 `p.device`。
- 📘 `字典.get(键, 默认值)`：查不到键时返回默认值，而不是报错。
- 📘 `A if 条件 else B`：一行写完的 if-else。
- 📘 **f-string**：`f"..."` 里的 `{表达式}` 会被替换成值。`{p.vid:04X}` 表示"4 位、不足补 0、大写十六进制"；
  `{p.device:<8}` 表示"左对齐、占 8 个字符宽"，用来把输出排成整齐的列。

### 拍照

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

- `mkdir(parents=True, exist_ok=True)`：建文件夹；上级不存在就一起建，已存在也不报错。
- 📘 `for idx in range(5)`：`idx` 依次取 0、1、2、3、4。`continue` 跳过本次循环剩下的部分。
- `cv2.VideoCapture(idx)` 按编号打开相机；OpenCV 在 Windows 上默认用 MSMF 后端，和 LeRobot 一致。
- **为什么读 15 帧才保存**：相机刚打开时自动曝光还没稳定，前几帧常常是全黑的。我们第一次截图就遇到了全黑。
- 📘 `for _ in range(15)`：`_` 是"我不需要这个变量"的惯用名。
- 📘 `ok, frame = cap.read()`：函数返回两个值，分别赋给两个变量（**元组解包**）。
- `cap.release()`：释放相机。**不释放的话，下一个程序（比如 LeRobot）就打不开这个相机。**

### 入口

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

- `argparse`：声明这个脚本接受哪些参数、类型、默认值。`args.max_cameras` 就是用户传入的值（`-` 会自动变成 `_`）。
- 📘 `if __name__ == "__main__":`：只有**直接运行**这个文件时才执行 `main()`；被别的文件 `import` 时不执行。
  后面 `bench_dataloader.py` 会讲到，在 Windows 上这一行**必不可少**。

---

## 2. `scripts/scan_motors.py`

**问题：** 舵机 ID 是之前在虚拟机里设置的，现在换到 Windows，要确认 6 个 ID 是否都还在，而且**不能改动舵机的任何设置**。

### 背景：舵机总线

SO-101 的 6 个 Feetech STS3215 舵机**串在一根线上**（菊花链），共用一个串口。主机发出的每条指令都带一个目标 ID，
只有 ID 匹配的舵机才回应。所以 6 个舵机必须有不同的 ID（1–6），波特率也必须和主机一致（1 Mbps）。
这两样都存在**舵机自己的 EEPROM** 里，断电不丢，所以换电脑、换系统都不用重设。

### 关节表

```python
SO101_MOTORS = {"shoulder_pan": 1, "shoulder_lift": 2, ..., "gripper": 6}
STS3215_MODEL_NUMBER = 777
```

ID 分配和 LeRobot 源码 `lerobot/robots/so_follower/so_follower.py` 完全一致。
`777` 是 STS3215 在型号寄存器里报告的值，我们用 LeRobot 的 `_model_number("sts3215")` 实测确认过。

### 建总线

```python
def build_bus(port: str) -> SerialMotorsBus:
    motors = {name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
              for name, motor_id in SO101_MOTORS.items()}
    return SerialMotorsBus(port=port, motors=motors)
```

- 📘 `{k: v for ... in ...}` 是**字典推导式**：用一个循环一次性生成整个字典。
- 📘 `字典.items()` 同时取出键和值：`name`、`motor_id`。
- `SerialMotorsBus`：LeRobot 0.7.0 的舵机总线类（底层用 Rust 写的 `rustypot` 库读写串口）。
  旧版教程里的 `FeetechMotorsBus` 在 0.7.0 已标记为弃用。

### 扫描

```python
model = bus.ping(name, num_retry=2)
if model is None:
    print("MISSING"); continue
pos = bus.read("Present_Position", name, normalize=False, num_retry=2)
```

- `ping` 读舵机的型号寄存器：有回应就返回型号号码，没回应返回 `None`。
- `read("Present_Position", ..., normalize=False)`：读**原始**编码器值（0–4095 为一圈）。
  `normalize=False` 是因为此时还没有标定文件，没法换算成 −100~100。
- `num_retry=2`：串口偶尔丢包，失败了再试两次。

### 两个"只读"的关键设计

```python
bus.connect(handshake=False)
...
finally:
    bus.disconnect(disable_torque=False)
```

1. `handshake=False`：LeRobot 默认连接时会检查所有舵机，**缺一个就直接报错退出**。
   我们想要的是"列出哪些在、哪些缺"，所以关掉握手，自己逐个 ping。
2. `disable_torque=False`：LeRobot 默认断开时会**写入**"关闭力矩"。这是一次写操作；
   我们承诺脚本只读，所以断开时什么都不写。
3. 📘 `try: ... finally: ...`：不管 `try` 里是否出错，`finally` 里的代码**一定会执行**。
   保证串口一定被关闭，否则下一个程序打不开这个 COM 口。

### `--watch` 模式

```python
pos = bus.sync_read("Present_Position", names, normalize=False, num_retry=2)
print("\r  " + ..., end="", flush=True)
```

- `sync_read`：一条指令同时读多个舵机，比逐个读快得多。
- `"\r"`（回车符）把光标移回行首，`end=""` 不换行：新的一行数字会覆盖旧的，在终端里看起来就是"实时刷新"。
- 📘 `except KeyboardInterrupt:`：捕获 Ctrl+C，让程序正常退出而不是打印一大段报错。

我们就是用同样的思路区分主臂和从臂的：同时读两个口 15 秒，你晃主臂，记录每个关节的最大值减最小值，
COM4 的变化合计 4545，COM3 是 0，所以 COM4 是主臂。

---

## 3. `scripts/env.ps1`

```powershell
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:HF_LEROBOT_HOME = Join-Path $repoRoot "data"
$env:HF_LEROBOT_CALIBRATION = Join-Path $repoRoot "config\calibration"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

- 📘 PowerShell 变量以 `$` 开头。`$PSScriptRoot` 是**当前脚本所在的文件夹**（`scripts`），
  `Split-Path -Parent` 取它的上一级，也就是仓库根目录。这样无论你在哪个目录运行脚本，路径都对。
- 📘 `$env:名字 = 值`：设置**环境变量**。环境变量是操作系统层面的"全局设置"，
  当前进程和它启动的所有子进程（比如 Python）都能读到。
- LeRobot 在 `lerobot/utils/constants.py` 里读这两个变量：

```python
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path))
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration"))
```

  `os.getenv("名字", 默认值)`：有这个环境变量就用它，没有就用默认值。所以我们**不改 LeRobot 的任何代码**，
  只设两个环境变量，就把数据和标定的位置改了。

- 这样设置**只对当前 PowerShell 窗口有效**，关掉窗口就没了，不会影响电脑上的其他项目。
  `robot.ps1` 和 `train.ps1` 开头都用 `. (Join-Path $PSScriptRoot "env.ps1")` 加载它。
- 📘 开头的 `.`（点号 + 空格）叫 **dot-source**：在**当前**作用域里运行另一个脚本，
  它设置的变量在当前脚本里继续有效。如果不加点号直接运行，变量会在那个脚本结束时消失。

---

## 4. `scripts/robot.ps1`

**问题：** LeRobot 的命令非常长（一条录制命令有十几个参数），每次手打容易错，
而且**标定、遥操作、录制、部署必须用完全相同的端口、ID、相机设置**，否则标定文件找不到，或者数据和部署对不上。

**思路：** 硬件设置只在 `config/robot.json` 里写一次，脚本读它、拼出完整命令。

### 参数声明

```powershell
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("calibrate-follower", "calibrate-leader", "teleop", "record", "rollout")]
    [string]$Command,
    [switch]$Cameras,
    [int]$Episodes = 50,
    ...
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
```

- 📘 `param(...)` 声明脚本接受的参数，作用和 Python 的 `argparse` 一样。
- `Mandatory = $true, Position = 0`：必填，并且可以不写参数名直接放在第一个位置，所以能写成 `robot.ps1 record`。
- `ValidateSet(...)`：只允许这几个值，打错了 PowerShell 会直接提示可选项。
- `[switch]`：开关参数，写了就是 `$true`（如 `-Cameras`），不写就是 `$false`。
- `[int]$Episodes = 50`：整数参数，默认值 50。
- `ValueFromRemainingArguments`：**没被上面任何参数认领的东西都收进 `$Extra`**。
  所以 `robot.ps1 rollout ... --policy.n_action_steps=1` 里的 `--policy...` 会原样传给 LeRobot。

### 读配置

```powershell
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
```

- 📘 `|` 是**管道**：把左边命令的输出交给右边。
- `Get-Content -Raw` 读出整个文件的文本，`ConvertFrom-Json` 把 JSON 文本变成对象，
  之后可以用 `$cfg.follower.port` 取到 `"COM3"`。

### 拼相机参数

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

- `$cfg.cameras.PSObject.Properties`：遍历 JSON 对象的每个键（`top`、`wrist`）。
- 📘 双引号字符串里的 `$($表达式)` 会被替换成值，相当于 Python 的 f-string。
- 📘 函数最后"裸露"的值就是返回值（PowerShell 不需要写 `return`）。
- 结果是 LeRobot 要的格式：`{ top: {type: opencv, index_or_path: 1, ...}, wrist: {...} }`。
  LeRobot 用 `draccus` 库把它解析成 `OpenCVCameraConfig` 对象。

### 按命令组装

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

- 📘 `@(a, b)` 是数组；`+` 把两个数组拼起来；`+=` 往数组末尾追加。
- `record` 分支里几个关键参数的原因：

| 参数 | 原因 |
|---|---|
| `--dataset.no_stamp=true` | LeRobot 默认在数据集名后加 `_年月日_时分秒`。关掉它，名字就是固定的 `so101_pick_place`，训练命令和 `-Resume` 都能直接引用 |
| `--dataset.streaming_encoding=true` | 边录边把画面编码成视频。否则会先存成上千张 PNG，每条结束后再统一编码，要等很久 |
| `-Resume` 时加 `--dataset.root=...` | LeRobot 恢复录制时必须知道数据集在硬盘上的确切位置 |

### 执行前的两道检查（只针对 record）

```powershell
if (-not $Resume -and (Test-Path $datasetDir)) { throw "..." }
python (Join-Path $PSScriptRoot "check_cameras.py")
if ($LASTEXITCODE -ne 0) { throw "..." }
```

**为什么需要：** 我们读了 `lerobot_record.py` 的源码，发现它的执行顺序是
"先创建数据集文件夹 → 再连接相机和机械臂"。相机一失败，就留下一个空壳文件夹，下次同名录制会被拒绝。
所以在调用 LeRobot **之前**先检查：文件夹已存在就停；相机打不开也停。

- 📘 `throw "消息"`：抛出错误，脚本立即停止。
- 📘 `$LASTEXITCODE`：上一个外部程序的**退出码**。约定 0 表示成功，非 0 表示失败。

### 运行

```powershell
Write-Host "> $exe $($argsList -join ' ')" -ForegroundColor Cyan
if ($DryRun) { return }
$ErrorActionPreference = "Continue"
& $exe @argsList
exit $LASTEXITCODE
```

- 先用青色打印完整命令：你能看到脚本到底替你执行了什么；`-DryRun` 只打印不执行，用来检查。
- 📘 `& $exe`：运行变量里存着的程序名。`@argsList`（**splatting**）把数组展开成一个个独立参数。
  这样即使参数里有空格（比如任务描述），也会被当成一个整体传过去。
- `$ErrorActionPreference = "Continue"` 的原因：Windows PowerShell 5.1 在输出被重定向时，
  会把外部程序写到 stderr 的**每一行**（包括无害的警告）都当成错误。如果设成 `"Stop"`，
  一句 PyTorch 的提示就会把训练停掉，我们的第一次冒烟测试就是这样失败的。
  真正的失败仍然能通过退出码判断。
- `exit $LASTEXITCODE`：把 LeRobot 的退出码原样传出去，调用者才知道成功与否。

---

## 5. `scripts/check_cameras.py`

**问题：** 腕部相机的线会被机械臂扯到，偶尔打不开（`failed to set fps=30 (actual_fps=0.0)`）。

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

- **用的是 LeRobot 自己的 `OpenCVCamera` 类**，而不是直接用 `cv2`。这样检查的路径和录制时完全一样：
  录制时会失败的情况（比如设置 fps 失败），预检查时也一定会失败。
- `cfg.get("fourcc")`：配置里有 `fourcc` 就用，没有就是 `None`（自动选择）。
  `MJPG` 让相机传压缩画面，两个相机共用一个 USB hub 时不容易因为带宽不足而打不开。
- 📘 `try / except Exception as e`：捕获错误，把错误信息存进 `e`，程序不崩溃，接着重试。
- 📘 `[check(n, c) for n, c in cameras.items()]` 外面再套 `all(...)`：**所有**相机都通过才算通过。
  先用列表把每个相机都检查一遍再判断，这样即使第一个失败了，第二个的结果也会打印出来。
- `sys.exit(0 或 1)`：把结果作为退出码交给 `robot.ps1`（就是上面 `$LASTEXITCODE` 读到的值）。

---

## 6. `scripts/inspect_dataset.py`

**问题：** 录了 50 条，不可能每条都看一遍视频。需要一个自动报告，把有问题的 episode 挑出来。

### 读数据

```python
files = sorted((root / "data").rglob("*.parquet"))
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
```

- `rglob("*.parquet")`：递归找所有 parquet 文件（数据可能被分成多个 chunk 和文件）。
- 📘 `pd`（pandas）的 **DataFrame** 就是一张表：每行一帧，每列一个字段（`action`、`episode_index`……）。
  `pd.concat` 把多张表上下拼起来。

### 逐个 episode 计算指标

```python
for ep, g in df.groupby("episode_index"):
    a = np.stack(g["action"].values)                 # 形状 (帧数, 6)
    speed = np.abs(np.diff(a, axis=0)).sum(axis=1)   # 形状 (帧数-1,)
    moving = np.flatnonzero(speed > IDLE_THRESHOLD)
    idle = (speed <= IDLE_THRESHOLD).mean()
    start = moving[0] / fps
    end = (len(a) - 1 - moving[-1]) / fps
```

这是整个脚本的核心，逐行拆开：

1. `df.groupby("episode_index")`：按 episode 分组，`ep` 是编号，`g` 是这个 episode 的所有帧。
2. `np.stack(...)`：parquet 里每帧的 `action` 是一个长度为 6 的数组，`stack` 把它们叠成 `(帧数, 6)` 的矩阵。
3. `np.diff(a, axis=0)`：相邻两帧相减，得到每一帧到下一帧的动作变化量。
4. `np.abs(...).sum(axis=1)`：6 个关节变化量的绝对值加起来，得到每一帧的"运动量"。
   和你熟悉的 L1 范数是一回事。
5. `speed > 0.3`：运动量超过阈值才算"在动"（0.3 是归一化单位，比传感器噪声大，比最慢的有意动作小）。
6. `(speed <= 0.3).mean()`：布尔数组求平均 = True 的比例 = **空闲占比**。
7. `np.flatnonzero(...)`：所有"在动"的帧的位置。第一个位置 / fps = **开头等了几秒**；
   最后一个位置到结尾的距离 / fps = **结尾空等了几秒**。
8. `a[:, GRIPPER].max()`：第 5 列（夹爪）的最大值 = 夹爪最多张开了多少（0 = 闭合，100 = 全开）。

然后按规则打标记，比如 `idle > 0.9` 标为 `NO MOTION - delete`。

### 缩略图

```python
frames[cam] = [f.to_ndarray(format="bgr24") for vf in video_files for f in av.open(str(vf)).decode(video=0)]
picks = np.linspace(lo, hi, 6).astype(int)
```

- `av`（PyAV）解码 MP4，每一帧转成 NumPy 数组。`bgr24` 是 OpenCV 习惯的颜色顺序。
- 📘 嵌套列表推导：先遍历每个视频文件，再遍历文件里的每一帧。
- `np.linspace(lo, hi, 6)`：在这个 episode 的第一帧和最后一帧之间均匀取 6 个位置。
- `np.hstack` 横着拼（6 帧一行），`np.vstack` 竖着拼（两个相机两行），再存成一张 jpg。

---

## 7. `scripts/bench_dataloader.py`

**问题：** 冒烟测试显示每步约 1.4 秒，其中约 1 秒在等数据（`data_s`）。要不要加 worker、换解码器？

**结果（50 条数据集，batch 8，ACT 的取数方式）：**

| 解码器 | 4 进程 | 8 进程 | 12 进程 |
|---|---|---|---|
| **torchcodec** | **0.062 s/batch** | 0.081 | 0.073 |
| pyav | 0.135 | 0.141 | 0.127 |

结论：默认的 torchcodec + 4 进程已经最快；进程再多反而变慢（进程之间争抢 CPU 和内存带宽）。
稳定状态下数据加载远快于 GPU 计算（约 0.4 s/步），所以冒烟测试里的 1 秒是启动开销，不是瓶颈（详见第 10 节）。

### 加载方式和 ACT 训练完全一致

```python
dataset = LeRobotDataset(
    repo_id,
    delta_timestamps={"action": [i / fps for i in range(chunk)]},
    video_backend=backend,
)
```

- `delta_timestamps`：告诉数据集"取样本时，每个键要哪些**相对时间**的帧"。
  `action: [0, 1/30, 2/30, ..., 99/30]` 表示从当前帧开始的 100 个动作，正是 ACT 的动作块。
  图像没写，就只取当前帧。
- 所以每个样本 = 2 张 480×640 图像（要从视频里解码）+ 6 维状态 + 100×6 的动作块。
- `video_backend`：用哪个库解码视频。`torchcodec`（PyTorch 官方）或 `pyav`（FFmpeg 的 Python 封装）。

### DataLoader

```python
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, persistent_workers=workers > 0, prefetch_factor=4,
    pin_memory=True, multiprocessing_context="spawn",
)
```

- `num_workers`：开几个**子进程**并行读数据。CPU 解码是瓶颈，多开几个进程就能并行解码。
- `persistent_workers`：每个 epoch 结束后不销毁子进程。Windows 上启动一个子进程要约 19 秒，不能反复重启。
- `prefetch_factor=4`：每个子进程提前准备 4 个 batch，GPU 一要就有。
- `pin_memory`：把数据放在"锁页内存"里，拷到 GPU 更快。
- `multiprocessing_context="spawn"`：Windows **只支持** spawn 方式：每个子进程是一个全新的 Python，
  会重新 `import` 你的脚本。这就是为什么 `if __name__ == "__main__":` 在 Windows 上**必不可少**：
  不加的话，每个子进程 import 脚本时又会启动一批子进程，无限套娃。

### 计时

```python
for _ in range(5): next(it)          # 预热：子进程启动、第一次解码
start = time.perf_counter()
for _ in range(n_batches): next(it)
return n_batches / (time.perf_counter() - start)
```

先丢掉前 5 个 batch，因为它们包含了子进程的启动时间，不代表稳定速度。
`time.perf_counter()` 是高精度计时器，专门用来测耗时。

---

## 8. `scripts/train.ps1`

结构和 `robot.ps1` 相同：参数 → 读配置 → 拼参数 → 运行。重点是**每个参数为什么这样设**。

### 所有策略共用的参数

| 参数 | 含义 | 为什么 |
|---|---|---|
| `--dataset.eval_split=0.1` + `--eval_steps=2000` | 留出 10% 的 episode（5 条）不参与训练，每 2000 步算一次它们的 loss | 得到验证曲线，看得出是否过拟合 |
| `--output_dir=outputs/train/<名字>` | checkpoint 和日志存在这里 | |
| `--policy.repo_id=Yufeng1216/<名字>` | 训练结束后模型推到 Hub 的这个仓库 | |
| `--trackio.enable=true` | 记录 loss 曲线到本地数据库 | 用 `trackio show` 查看 |

### 恢复训练

```powershell
if ($Resume) {
    $argsList = @("--config_path=$outputDir/checkpoints/last/pretrained_model/train_config.json", "--resume=true")
}
```

恢复时**不重新指定任何参数**：所有设置都从上次保存的 `train_config.json` 读，
优化器状态、学习率调度、步数计数器从 `training_state/` 恢复，保证和没中断过一样。

### 策略预设

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
            "--policy.resize_shape=[240,320]",   # 图像缩小一半，省显存
            "--policy.crop_ratio=0.95",          # 训练时随机裁剪 = 数据增强
            "--policy.down_dims=[256,512,1024]", # U-Net 通道减半：默认 278M 参数在 8 GB 显存里溢出
            "--ema.type=warmup",                 # 额外保存一份权重的滑动平均
            "--batch_size=8", "--steps=$Steps", "--save_freq=20000"
        )
    }
}
```

`-Steps` 默认是 0，表示"用这个策略的推荐值"；冒烟测试时传 `-Steps 200` 覆盖它。
每个策略一个分支；SmolVLA、π₀.₅ 之后也加在这里。

- `--policy.xxx`：改策略本身的结构参数（对应 `configuration_diffusion.py` 里的字段）。
  命令行里的列表写成 `[240,320]`，LeRobot 的配置解析器（draccus）会把它转成 Python 的元组。
- `--ema.xxx`：训练流程的参数，不属于策略。ACT 不用 EMA，所以只在 diffusion 分支里出现。
- 为什么这三个参数这样设，见 [05_diffusion_policy.md](05_diffusion_policy.md) 第 5 节。

---

## 9. `scripts/lerobot_train_win.py`

**问题：** LeRobot 每保存一次 checkpoint，会把 `checkpoints/last` 设成指向最新 checkpoint 的**符号链接**。
Windows 普通用户没有创建符号链接的权限（`WinError 1314`），训练会在第一次保存时崩溃。

### 符号链接 vs 目录联接

| | 符号链接（symlink） | 目录联接（junction） |
|---|---|---|
| 作用 | 一个"快捷方式"，打开它就进入目标 | 同左，但只能指向文件夹 |
| Windows 权限 | 需要管理员或开发者模式 | **普通用户就能建** |
| 目标路径 | 可以是相对路径 | 必须是绝对路径 |

对 LeRobot 来说，`last` 只需要"能当文件夹打开"，所以联接完全够用。

### 代码

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

- 这种技巧叫 **monkey patch（猴子补丁）**：程序运行时把某个模块里的函数**替换**成我们自己的版本，
  而不修改库的源文件。好处：不用 fork LeRobot，升级 LeRobot 也不会冲突；坏处：如果 LeRobot 改了函数名，补丁就失效。
- **为什么要替换两处**：LeRobot 原函数定义在 `train_utils` 里，而 `lerobot_train.py` 写的是
  `from ... import update_last_checkpoint`，相当于复制了一份引用到自己的模块里。只改一处，另一处还是旧函数。
  （我们用 `grep` 在源码里找到了所有调用它的地方。）
- `os.rmdir(last)`：删除旧联接。对联接使用 `rmdir` **只删联接本身**，不会删掉它指向的 checkpoint。我们测试过。
- `_winapi.CreateJunction`：Python 标准库里 Windows 专用的建联接函数。目标必须是绝对路径，所以用 `.resolve()`。
- `if sys.platform == "win32"`：只在 Windows 上替换，同一份代码在 Linux 上行为和原版完全一样。
- 最后调用 LeRobot 原本的 `main()`。命令行参数原样传给它，所以这个文件的用法和 `lerobot-train` 完全相同。

---

## 10. LeRobot 内部：我们调用的命令在做什么

### `lerobot-record` 的主循环（`lerobot/scripts/lerobot_record.py`）

```
创建数据集（或 resume）→ 连接主臂 → 连接从臂和相机
while 已录条数 < num_episodes:
    🔊 "Recording episode N"
    record_loop(时长 = episode_time_s):        # 每 1/30 秒一次
        obs = robot.get_observation()         # 读从臂 6 个关节 + 两路相机
        act = teleop.get_action()             # 读主臂 6 个关节
        robot.send_action(act)                # 让从臂跟随
        dataset.add_frame({obs, act, task})   # 写入当前 episode 的缓冲区
    🔊 "Reset the environment"
    record_loop(时长 = reset_time_s, 不写数据集)
    if 按了 ←: 清空缓冲区，重录
    else:      dataset.save_episode()        # 写 parquet、结束这一段视频
上传到 Hub
```

从这里能看出几件我们教程里用到的事实：**episode 在复位结束后才保存**（所以复位时按 ← 还能丢弃）；
最后一条没有复位阶段；复位时主臂仍然在控制从臂。

### `LeRobotDataset.__getitem__`：训练时取一个样本

1. 从 parquet 里取出第 i 帧的 `observation.state`、`action` 等。
2. 按 `delta_timestamps` 取出相邻帧（ACT 要往后 100 步的 action）；超出 episode 末尾的部分用最后一帧填充，
   并返回一个 `action_is_pad` 掩码，告诉 loss 哪些位置是填充的、不要算进去。
3. 根据时间戳，在 MP4 里定位并解码对应的图像帧。
4. 返回一个字典：`{"observation.images.top": 张量, ..., "action": (100, 6) 张量, ...}`。

### `lerobot-train` 的训练循环（`lerobot/scripts/lerobot_train.py`）

```
读数据集元信息（stats.json）→ 按 eval_split 划分训练/验证 episode
make_policy()           # 按 policy.type 建网络；ResNet18 加载 ImageNet 预训练权重
make_pre_post_processors()  # 归一化（用 stats.json 的 mean/std）和反归一化
for step in range(steps):
    batch = next(dataloader)
    batch = preprocessor(batch)      # 归一化，移到 GPU
    loss, info = policy.forward(batch)
    loss.backward(); 梯度裁剪; optimizer.step(); scheduler.step()
    每 log_freq 步：打印 loss、grdn（梯度范数）、data_s、updt_s
    每 eval_steps 步：在验证 episode 上算 eval_loss
    每 save_freq 步：保存 checkpoint，更新 last
训练结束：push_to_hub
```

日志里的 `data_s` 是等数据的时间，`updt_s` 是 GPU 前向反向的时间（都是两次日志之间的平均值）。
冒烟测试第 100 步时 `data_s≈0.97`、`updt_s≈0.42`，我们一开始据此判断"瓶颈在数据加载"。
但 `bench_dataloader.py` 测出稳定状态下每个 batch 只要 0.06 秒：那 0.97 秒其实是
**4 个 worker 进程的启动时间（每个约 19 秒）被摊到了前 100 步里**。所以真正的瓶颈是 GPU。
教训：**看性能数据要区分"启动开销"和"稳定状态"**，只看前 100 步的平均值会被误导。

### ACT 的 `forward`（`lerobot/policies/act/modeling_act.py`）

```
训练时：
  VAE 编码器输入 [CLS, 关节状态, 100 步真实动作] → 取 CLS 输出 → μ, log σ² → 采样 z（32 维）
推理时：
  z = 0
共同部分：
  每张图 → ResNet18 → 512×15×20 特征图 → 300 个 token（加 2D 正弦位置编码）
  编码器输入 = [z 的 token, 关节状态 token, top 的 300 个 token, wrist 的 300 个 token]
  Transformer 编码器（4 层）→ Transformer 解码器（1 层，100 个可学习 query）→ Linear → (100, 6)
loss = L1(预测动作, 真实动作，忽略 action_is_pad 的位置) + 10 × KL(q(z) ‖ N(0, I))
```

日志里的 `l1_loss` 和 `kld_loss` 就是这两项。

---

## 11. `cluster/`：集群训练

这两个文件是 **bash 脚本**（Linux 的 shell 语言），在集群上运行，不在 Windows 上运行。
用法见 [05_diffusion_policy.md](05_diffusion_policy.md) 第 6.5 节。

### `setup_env.sh`：一次性装环境

```bash
set -euo pipefail
```

bash 脚本的"安全模式"：`-e` 任何一条命令失败就停止；`-u` 用到未定义的变量就报错；
`-o pipefail` 管道里任何一段失败都算失败。没有这一行，前面装失败了脚本还会继续往下跑。

```bash
if [ ! -x "$CONDA_DIR/bin/conda" ]; then ... fi
```

如果 `~/miniforge3/bin/conda` 不存在（`!` 取反，`-x` 表示"存在且可执行"），就下载安装 Miniforge。
所以这个脚本**可以重复运行**：装过的步骤会跳过。下面的 `conda env list | grep -q` 也是同样的思路。

装的东西和笔记本上一一对应：Python 3.12、conda-forge 的 ffmpeg 7.1.1、`lerobot==0.7.0`。
版本对齐，两边训练出的结果才能比较。

### `train_diffusion_full.sbatch`：训练任务

```bash
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
```

对 bash 来说这些是注释；对 `sbatch` 来说这是**资源申请**：用哪个账号计费、哪个分区（哪类 GPU）、
几块 GPU、最长跑多久、日志写到哪（`%x` = 任务名，`%j` = 任务编号）。

```bash
export HF_LEROBOT_HOME="$PWD/data"
```

和 Windows 上 `env.ps1` 的作用一样：让数据集下载到项目目录里。第一次运行时，
`lerobot-train` 发现本地没有 `Yufeng1216/so101_pick_place`，会自动从 Hub 下载（数据集是公开的，不需要登录）。

```bash
if [ -d "outputs/train/$RUN/checkpoints/last" ]; then
    srun lerobot-train --config_path=... --resume=true
    exit
fi
```

集群任务有时间上限。如果超时被停掉，**原样再提交一次**，这里检测到已有 checkpoint，就接着训，
和 `train.ps1 -Resume` 一样。`srun` 让命令在 Slurm 分配到的资源上运行。

最后的 `lerobot-train` 参数和 `train.ps1` 的 diffusion 预设**逐项相同**，只是**没有** `--policy.down_dims`
（于是用默认的 512/1024/2048），这样两次训练只差模型容量一个变量。行尾的 `\` 表示"命令没写完，下一行接着"。

> 这两个文件必须用 Linux 的换行符（LF）。Windows 的换行符（CRLF）会让 bash 报
> `$'\r': command not found`。`.gitattributes` 里的 `*.sh text eol=lf` 保证了这一点。

---

## 更新记录

这篇会随项目推进持续更新：每新增或修改一个脚本（Diffusion / SmolVLA / π₀.₅ 的训练预设、评估脚本……），
就在这里补上对应的讲解。
