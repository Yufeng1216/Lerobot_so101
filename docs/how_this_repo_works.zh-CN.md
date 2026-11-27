# 项目导读：每个文件是干什么的

[English](how_this_repo_works.md) · **简体中文**

写给"深度学习会、工程工具不熟"的人。读完这一页，仓库里每个文件为什么存在就都清楚了。

## 1. 先建立一个心智模型

训练一个模型的流程我们都很熟：**数据 → 模型 → 训练 → 评估**。机器人模仿学习一模一样，
只是数据不是从网上下载，而是**自己用机械臂录出来的**：

```
人手拿主臂(leader) ──遥操作──▶ 从臂(follower) 跟着动
        │                              │
        │ 每 1/30 秒记录一次：          │
        │   观测 = 两路相机图像 + 6 个关节角度（state）
        │   动作 = 主臂此刻的 6 个关节角度（action）
        ▼
   数据集 (observation, action) 序列  ──▶  训练策略 π(action | observation)
                                                    │
                                                    ▼
                               部署：策略看相机 + 关节，输出动作，驱动从臂
```

所以这个项目本质上就是**有监督学习（行为克隆）**：输入图像和状态，回归动作。
ACT、Diffusion Policy、SmolVLA、π₀.₅ 的区别，只在"怎么建模 p(action | observation)"上。
工程部分（串口、相机、命令行）其实只是把数据搬进搬出的管道，跟着教程走就行。

## 2. 用 Windows 而不是 Linux，会不会"不正规"？

不会。原因很具体：

- LeRobot 不依赖 ROS，也不依赖任何 Linux 专属的东西。舵机走串口，相机走 OpenCV，训练是 PyTorch。
- 在 Windows 上 12 个舵机都能读到，相机都能拍照，GPU 能跑 CUDA。
- 唯一的差别是串口名字：Linux 叫 `/dev/ttyACM0`，Windows 叫 `COM3`。官方文档里看到 `/dev/tty...`，换成 COM 口就行。
- 所以重要的是**理解策略、做出结果、讲清楚取舍**，而不是用什么操作系统。

## 3. 文件类型速查

### 先分清两大类

- **文本文件**：里面是人能读的字符，用记事本或 VS Code 打开就能看懂、能改。
  代码、配置、文档都是文本文件。git 擅长管理它们：能精确显示"第几行改了什么"。
- **二进制文件**：里面是给程序读的字节，用记事本打开是一堆乱码。
  图片、视频、表格数据、模型权重都是二进制文件。它们通常很大，**不放进 git**。

PS：Windows 默认隐藏扩展名。打开资源管理器 → 查看 → 显示 → 勾选"文件扩展名"和"隐藏的项目"，
才能看到 `.py`、`.json` 这些后缀，以及 `.git`、`.gitignore` 这种以点开头的文件。

### A. 代码和脚本（文本，进 git）

| 文件 | 是什么 | 怎么用 | 本项目例子 |
|---|---|---|---|
| `.py` | Python 源代码 | `python scripts/xxx.py` 运行；VS Code 打开编辑 | `scripts/scan_motors.py` |
| `.ps1` | PowerShell 脚本：把一串命令打包成一个，Windows 上的"批处理" | 在 PowerShell 里 `.\scripts\robot.ps1 ...` 运行 | `scripts/robot.ps1`、`scripts/env.ps1` |
| `.sh` | bash 脚本：Linux 上的"批处理"，作用和 `.ps1` 一样 | 在 Linux 终端里 `bash cloud/slurm/xxx.sh` 运行 | `cloud/slurm/setup_env.sh` |
| `.sbatch` | 也是 bash 脚本，开头的 `#SBATCH` 行是给集群排队系统 Slurm 的资源申请 | `sbatch xxx.sbatch` 提交（本项目通过 `submit.sh`） | `cloud/slurm/train.sbatch` |

### B. 配置（文本）

| 文件 | 是什么 | 本项目例子 |
|---|---|---|
| `.json` | 结构化数据：`{"键": 值}`，和 Python 的 dict 几乎一样。程序读它来获取设置 | `config/robot.json`（COM 口、相机编号），`config/calibration/**/*.json`（标定） |
| `.gitignore` | 告诉 git **哪些文件不要上传**：大文件、密钥、本机配置。每行一个规则，`data/` 表示整个文件夹 | 根目录 |
| `.gitattributes` | 告诉 git **怎么处理某类文件**。本项目用它统一换行符（Windows 用 CRLF，Linux 用 LF） | 根目录 |
| `.env` | 一行一个 `名字=值` 的设置，脚本读进来当变量用。常放因人而异的设置，所以通常不进 git，只上传一个 `.example` 模板 | `cloud/slurm/site.env.example` |

### C. 文档（文本，进 git）

| 文件 | 是什么 | 本项目例子 |
|---|---|---|
| `.md` | Markdown（英文版 `xxx.md`，中文版 `xxx.zh-CN.md`）：`#` 是标题，`**粗体**`，三个反引号包起来的是代码块，`\| a \| b \|` 是表格。GitHub 会自动渲染成网页。VS Code 里按 `Ctrl+Shift+V` 预览 | `README.md`、`docs/*.md` |
| `LICENSE`（无扩展名） | 开源许可证：别人能不能用、怎么用你的代码。本项目是 MIT（几乎可以随便用，保留署名即可） | 根目录 |

### D. 数据集里的文件（`data/Yufeng1216/so101_pick_place/`，不进 git，Hub 上有一份）

| 文件 | 类型 | 是什么 |
|---|---|---|
| `meta/info.json` | 文本 | 数据集"说明书"：fps、总帧数、总 episode 数、每个键的形状和类型、视频编码格式 |
| `meta/stats.json` | 文本 | 每个键的统计量（mean、std、min、max、分位数）。训练时用来归一化输入和输出 |
| `meta/tasks.parquet` | 二进制 | 任务描述字符串表（本项目只有一句） |
| `meta/episodes/.../*.parquet` | 二进制 | 每个 episode 的索引：从第几帧到第几帧、在视频里的起止时间 |
| `data/.../*.parquet` | 二进制 | **核心数据表**：每帧一行，列是 `action`、`observation.state`、`timestamp`、`episode_index`…… |
| `videos/<相机>/.../*.mp4` | 二进制 | 相机画面，AV1 编码。多个 episode 拼在同一个文件里 |

`.parquet` 是一种**列式存储的表格**格式，比 CSV 小、读得快。想看里面的内容：

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/Yufeng1216/so101_pick_place/data/chunk-000/file-000.parquet'); print(df.head()); print(df.columns)"
```

### E. 训练产物（`outputs/train/<run>/checkpoints/<步数>/pretrained_model/`，不进 git，最终模型推到 Hub）

| 文件 | 类型 | 是什么 |
|---|---|---|
| `model.safetensors` | 二进制 | **模型权重**（ACT 约 200 MB）。`.safetensors` 和 PyTorch 的 `.pt` 作用一样，但只存张量、不能藏可执行代码，所以更安全，加载也更快 |
| `config.json` | 文本 | 策略结构参数：chunk_size、层数、输入输出特征。加载模型时先读它来搭网络 |
| `train_config.json` | 文本 | 这次训练的**全部**设置（数据集、步数、学习率……）。可复现，也是 `-Resume` 时读取的文件 |
| `policy_preprocessor.json` 和 `*_normalizer_processor.safetensors` | 文本 + 二进制 | 输入预处理的步骤，以及归一化用的均值和方差（从 `stats.json` 来）。部署时必须和权重一起用 |
| `policy_postprocessor.json` 和 `*_unnormalizer_processor.safetensors` | 文本 + 二进制 | 输出后处理：把网络输出的归一化动作还原成真实的关节角度 |
| `training_state/`（与 `pretrained_model/` 同级） | 文本 + 二进制 | `optimizer_state.safetensors`（Adam 的动量等）、`rng_state.safetensors`（随机数状态）、`training_step.json`（训到第几步）。只有恢复训练时用 |

### F. 运行时产生的其他文件（`outputs/`，不进 git）

| 文件 | 是什么 |
|---|---|
| `.jpg` | 图片：相机截图（`outputs/hardware_check/`）、数据集缩略图（`outputs/dataset_check/`） |
| `.log` | 日志：程序运行时打印的全部文字，存成文件方便事后查看（`outputs/logs/`） |

### G. 隐藏的 `.git/` 文件夹

git 的"数据库"，存着所有历史版本。**不要手动修改或删除**；删了它，这个文件夹就不再是 git 仓库了。

## 4. 目录结构

```
Lerobot/
├── README.md              项目门面（英文；中文版 README.zh-CN.md）：做了什么、结果如何
├── .gitignore             不上传哪些文件
├── config/
│   ├── robot.example.json 配置模板（会上传到 GitHub）
│   ├── robot.json         这台电脑的真实配置：COM 口、相机编号（不上传，每台电脑不一样）
│   └── calibration/       两条臂的标定文件（上传：数据集依赖它）
├── scripts/               在笔记本（Windows）上运行的工具：硬件、录制、本地训练、部署
│   ├── env.ps1            所有脚本共用的环境设置：让 LeRobot 把数据集、标定存进本仓库
│   ├── find_hardware.py   列出串口和相机，并给每个相机拍一张照片
│   ├── scan_motors.py     只读地检查 6 个舵机的 ID 和角度
│   ├── check_cameras.py   录制前的相机预检查
│   ├── inspect_dataset.py 录完后检查数据质量（空闲占比、开头等待、缩略图）
│   ├── robot.ps1          读 config/robot.json，拼好标定/遥操作/录制/部署命令
│   ├── train.ps1          训练入口：每个策略的参数预设
│   ├── lerobot_train_win.py  Windows 修复版 lerobot-train（见 docs/04_act.zh-CN.md）
│   ├── align_camera.py    相机被碰歪后，对照录制时的画面把它摆回原位
│   ├── make_server_config.py  远程推理的服务端配置（π₀.₅ 在云端 GPU 上跑）
│   ├── make_gif.py        把一次评估试验剪成 README 用的 GIF
│   └── plot_results.py    从 results/*.md 画各点位成功率图
├── cloud/                 在别的机器上训练（笔记本 8 GB 显存放不下的模型）
│   ├── README.md          使用说明
│   ├── slurm/             Linux GPU 集群（Slurm 排队系统）：装环境、提交训练任务
│   │   ├── site.env.example  账号等个人设置的模板（真实的 site.env 不上传）
│   │   ├── setup_env.sh   一次性装环境
│   │   ├── submit.sh      检查账号后提交任务
│   │   └── train.sbatch   训练任务本身
│   └── hf_jobs/           租用的 Hugging Face GPU：π₀.₅ 训练（train_pi05.py）和评估用的推理服务端（serve_pi05.py）
├── zero_shot/             免训练基线（VLM 智能体 + IK）：运动学、触碰标定、夹爪测量、Claude Haiku 智能体 agent.py
├── results/               每个策略真机评估的逐次记录表
├── assets/                README 用的 GIF 和图
├── docs/                  教程，一个阶段一个文件（中英文各一份）
├── data/                  录好的数据集（不上传 git，HF Hub 上有一份）
├── outputs/               运行时产生的东西：截图、训练 checkpoint、日志（不上传）
└── notes/                 个人开发日志（只在本地，不上传）
```

**本地和云端怎么分：** `scripts/` 里的东西都在笔记本上运行；`cloud/` 里的东西在别的机器上运行。
两边训练同一个策略时，参数保持一致，唯一的区别是更大的显卡可以用论文的默认模型大小。

为什么数据集和模型不放进 git？git 是为**文本代码**设计的，几百 MB 的视频和权重放进去会让仓库臃肿、
clone 很慢，GitHub 也限制单文件 100 MB。业界做法是：代码放 GitHub，数据集和模型放 Hugging Face Hub，
README 里互相链接。

为什么有 `robot.example.json` 和 `robot.json` 两个文件？
COM 口和相机编号是**这台电脑特有的**，换台电脑就变了，所以真实配置不进 git；
但别人 clone 仓库时需要知道配置长什么样，所以上传一个模板。这是开源项目的通用做法。

## 5. 几个"环境"概念

| 概念 | 类比 | 本项目里 |
|---|---|---|
| **PowerShell** | Windows 的命令行（终端），你在里面输入命令 | 所有命令都在这里运行 |
| **conda 环境** | 一个独立的 Python + 一套库，互不干扰。像给每个项目单独一个 Python | 环境名 `lerobot`，每次开终端先 `conda activate lerobot` |
| **pip** | Python 的包管理器，`pip install xxx` 装库 | 装了 `lerobot==0.7.0` |
| **CLI 命令** | 装完库后多出来的命令，比如 `lerobot-record` | 本质上是在运行 lerobot 包里的某个 Python 函数 |
| **`--robot.port=COM3`** | 命令行参数。LeRobot 用点号表示嵌套配置：`robot.port` 就是配置里 `robot` 下面的 `port` | 和训练时的 `--lr=1e-4` 一个道理 |
| **git / GitHub** | git 记录代码的每一次修改；GitHub 是放 git 仓库的网站 | 最后把这个文件夹推到 GitHub |
| **Hugging Face Hub** | 放模型和数据集的"GitHub" | 录好的数据集、训好的模型都传到这里 |

## 6. 每次开始干活的固定流程

```powershell
cd D:\桌面\Lerobot
conda activate lerobot
. .\scripts\env.ps1                  # 只在直接用 lerobot-xxx 命令时需要；robot.ps1 / train.ps1 会自动做
python scripts/find_hardware.py      # 确认串口和相机编号没变
```
