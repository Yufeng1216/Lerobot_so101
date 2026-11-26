# 00 · Windows 环境搭建

[English](00_environment.md) · **简体中文**

目标：在**原生 Windows 11** 上装好 LeRobot 0.7.0 和带 CUDA 的 PyTorch，并且能直接访问舵机串口和 USB 摄像头。

## 为什么不用 WSL2 / 虚拟机 / ROS

机器人项目一般默认用 Linux，但我们要的其实只有三样东西：串口、摄像头、GPU。所以先看各方案能不能拿到这三样：

| 方案 | 舵机串口 | USB 摄像头 | CUDA 训练 |
|---|---|---|---|
| **原生 Windows** ✅ | COM 口直接可用 | OpenCV 直接可用 | 可用 |
| WSL2 | 要用 `usbipd-win` 手动挂载 | 默认内核没有 UVC 驱动，要自己编内核 | 可用 |
| VMware 虚拟机 | USB 直通可用 | 可用 | **不可用**（虚拟机拿不到 GPU） |

只有原生 Windows 三样都能直接用。而 LeRobot 本身**完全不依赖 ROS**：舵机用 pyserial 直接读写，摄像头走 OpenCV，
数据集是 Parquet + MP4，训练是纯 PyTorch。所以在 Windows 上没有 ROS 也没关系。

## 1. 创建 conda 环境

LeRobot 0.7.0 要求 **Python ≥ 3.12**。`ffmpeg` 要从 conda-forge 装进环境里，因为 TorchCodec 解码视频时要用到它的 DLL。

```powershell
conda create -y -n lerobot -c conda-forge --override-channels python=3.12 ffmpeg=7.1.1
conda activate lerobot
```

PS：中途报 `CondaHTTPError: HTTP 000 CONNECTION FAILED` 只是网络抖动，重跑一次就好。

## 2. 安装带 CUDA 的 PyTorch

一定要**先装 PyTorch，再装 LeRobot**。反过来的话，pip 会顺手装一个 CPU 版的 PyTorch。
笔记本是 RTX 3070，驱动 596.x，所以用 CUDA 12.8 的版本：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. 安装 LeRobot（固定版本）

```powershell
pip install "lerobot[core-scripts,training,feetech,smolvla,pi,diffusion]==0.7.0"
```

| extra | 装了什么 |
|---|---|
| `core-scripts` | `lerobot-record`、`lerobot-teleoperate` 这些命令行工具，以及它们要用的数据集、可视化、键盘监听 |
| `training` | `lerobot-train`、accelerate、trackio（训练曲线） |
| `feetech` | Feetech 舵机（SO-101 用的 STS3215）的串口驱动 |
| `diffusion` | Diffusion Policy 要用的 `diffusers` |
| `smolvla` / `pi` | SmolVLA、π₀.₅ 要用的 `transformers` 等 |

PS：版本固定成 `==0.7.0` 是为了可复现。LeRobot 更新很快，小版本之间命令行参数都会变。

## 4. 验证

```powershell
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

看到 `True` 和 `0.7.0` 就说明装好了。

## 5. 登录 Hugging Face

数据集和模型都放在 Hugging Face Hub 上。去 <https://huggingface.co/settings/tokens> 建一个有 **write** 权限的 token，然后：

```powershell
hf auth login
hf auth whoami
```

把用户名填进 `config/robot.json` 的 `hf_user`。**token 不要写进仓库里的任何文件。**

PS：`hf auth login` 要在 `lerobot` 环境里运行。在 conda 的 base 环境里运行的话，用到的是另一个版本的 `hf`。

## 6. 项目放在哪里

项目文件夹**不要放在 OneDrive 这类同步盘里**。训练时每个 checkpoint 有几百 MB 到几 GB，同步软件会边训练边上传，
写文件时还可能把文件锁住，导致训练报错。要备份的话，训练完再手动复制。

`scripts/env.ps1` 会让 LeRobot 把数据集存进项目的 `data/`，把标定文件存进 `config/calibration/`
（默认它们都放在全局缓存 `$HF_HOME/lerobot` 里）。下载的预训练模型还是放在全局缓存，因为那是所有项目共享的。

## 7. 获取代码

代码和教程在 GitHub 上，数据集和模型在 Hugging Face Hub 上（GitHub 单个文件最大 100 MB，放不下）。

```powershell
git clone https://github.com/Yufeng1216/Lerobot_so101.git
cd Lerobot_so101
copy config\robot.example.json config\robot.json   # 再填上自己的 COM 口、相机编号和 HF 用户名
```

PS：Windows 版 Git 自带 Git Credential Manager。第一次 `git push` 时会打开浏览器授权，授权后跳到一个
`127.0.0.1:xxxx/?code=...` 的空白页，这是正常的，关掉就行。
