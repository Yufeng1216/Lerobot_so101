# 00 · Windows 环境搭建

目标：在 **原生 Windows 11** 上装好 LeRobot 0.7.0 + 带 CUDA 的 PyTorch，能直接访问舵机串口和 USB 摄像头。

## 为什么不用 WSL2 / 虚拟机 / ROS

| 方案 | 舵机串口 | USB 摄像头 | CUDA 训练 |
|---|---|---|---|
| **原生 Windows** ✅ | COM 口直接可用 | OpenCV 直接可用 | 可用 |
| WSL2 | 需要 `usbipd-win` 手动挂载 | 默认内核缺 UVC 驱动，需自编内核 | 可用 |
| VMware 虚拟机 | USB 直通可用 | 可用 | **不可用**（虚拟机拿不到 GPU） |

LeRobot **完全不依赖 ROS**：舵机通过串口（pyserial）直接读写，摄像头走 OpenCV，
数据集是 Parquet + MP4，训练是纯 PyTorch。所以 Windows 上没有 ROS 也不受影响。

## 1. 创建 conda 环境

LeRobot 0.7.0 要求 **Python ≥ 3.12**。`ffmpeg` 必须从 conda-forge 装进环境里，
TorchCodec 解码视频时要用到它的 DLL。

```powershell
conda create -y -n lerobot -c conda-forge --override-channels python=3.12 ffmpeg=7.1.1
conda activate lerobot
```

> 如果中途报 `CondaHTTPError: HTTP 000 CONNECTION FAILED`，是网络抖动，直接重跑即可。

## 2. 安装带 CUDA 的 PyTorch

**先装 PyTorch，再装 LeRobot**，否则 pip 可能装成 CPU 版。
笔记本是 RTX 3070，驱动 596.x，用 CUDA 12.8 的轮子：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. 安装 LeRobot（固定版本）

```powershell
pip install "lerobot[core-scripts,training,feetech,smolvla,pi,diffusion]==0.7.0"
```

各个 extra 的作用：

| extra | 内容 |
|---|---|
| `core-scripts` | `lerobot-record` / `lerobot-teleoperate` 等命令行工具及其依赖（数据集、可视化、键盘监听） |
| `training` | `lerobot-train`、accelerate、trackio 训练曲线 |
| `feetech` | Feetech 舵机（SO-101 用的 STS3215）串口驱动 |
| `diffusion` | Diffusion Policy 需要的 `diffusers` |
| `smolvla` / `pi` | SmolVLA、π₀.₅ 需要的 `transformers` 等 |

> 固定 `==0.7.0` 是为了可复现：LeRobot 迭代很快，命令行参数在小版本之间也会变。

## 4. 验证

```powershell
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

应看到 `True` 和 `0.7.0`。

## 5. 登录 Hugging Face（录数据前需要）

数据集和模型都推到 Hugging Face Hub。去 <https://huggingface.co/settings/tokens> 创建一个
**write** 权限的 token，然后：

```powershell
hf auth login
hf auth whoami
```

把用户名写进 `config/robot.json` 的 `hf_user` 字段。**token 不要写进任何仓库文件。**
