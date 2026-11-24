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

> `hf auth login` 要在 `lerobot` 环境里运行（`conda activate lerobot` 之后）。
> 如果在 conda 的 base 环境里运行，用的是另一个版本的 `hf`。

## 6. 项目放在哪里

项目文件夹是 `D:\桌面\Lerobot`，**不要放在 OneDrive 等同步盘里**：训练时每个 checkpoint 有几百 MB 到
几 GB，同步软件会一边训练一边上传，还可能在写文件时锁住文件导致训练报错。需要备份时，训练结束后再手动复制。

`scripts/env.ps1` 让 LeRobot 把数据集存进项目的 `data/`、标定存进 `config/calibration/`
（默认它们会放在全局缓存 `$HF_HOME/lerobot` 里）。下载的预训练模型仍然放在全局缓存，因为那是共享的下载，不属于本项目。

## 7. Git 和 GitHub

代码和教程放 GitHub，数据集和模型放 Hugging Face Hub（GitHub 单文件限制 100 MB）。

### 第一次上传

1. 在 <https://github.com/new> 新建仓库（本项目：`Yufeng1216/Lerobot_so101`，先设为 Private，有结果后再公开）。
   如果勾选了 "Add a license"，GitHub 会先生成一次提交，本地要先接上它再推送（见下面第 3 步）。
2. 本地初始化并提交：

```powershell
cd D:\桌面\Lerobot
git init -b main
git add -A
git commit -m "Initial project setup"
git remote add origin https://github.com/Yufeng1216/Lerobot_so101.git
```

3. 如果远程已经有提交（比如 LICENSE），先把本地提交"接"在它后面，再推送：

```powershell
git fetch origin
git rebase origin/main
git push -u origin main
```

`rebase` 的意思是：把我本地的提交挪到远程最新提交的后面，历史保持一条直线，不会覆盖远程的任何东西。

### 登录 GitHub 的坑

- Windows 版 Git 自带 **Git Credential Manager (GCM)**：第一次 `git push` 时会打开浏览器授权，
  授权后跳到一个 `127.0.0.1:xxxx/?code=...` 的**空白页，这是正常的**，表示授权已完成，关掉即可。
- GCM 的登录和 `gh`（GitHub CLI）的登录是**两套独立的系统**：`gh auth status` 显示未登录，不代表 `git push` 不行。
  查看 GCM 存了哪些账号：`git credential-manager github list`。
- **账号改过名**：GCM 里记的是改名前的用户名（本项目是 `Yufeng1216` → `Yufeng1216`）。
  如果远程地址写成 `https://Yufeng1216@github.com/...`，GCM 会以为是另一个账号，要求重新登录。
  解决：远程地址里**不要带用户名**，用 `https://github.com/Yufeng1216/Lerobot_so101.git`。
- "Repository not found" 在私有仓库上有两种可能：仓库还没建，或者当前凭据的账号没有权限看到它。
- 电脑上可能有两个 `gh`（官方 GitHub CLI 和一个同名的 npm 包）。要用官方版本时写全路径：
  `& "C:\Program Files\GitHub CLI\gh.exe" auth login`。
