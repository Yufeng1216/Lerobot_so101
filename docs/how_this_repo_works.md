# 项目导读：每个文件是干什么的

写给"深度学习会、工程工具不熟"的自己。读完这一页，仓库里每个文件你都知道它为什么存在。

## 1. 先建立一个心智模型

你已经熟悉训练一个模型的流程：**数据 → 模型 → 训练 → 评估**。机器人模仿学习一模一样，
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
ACT、Diffusion Policy、SmolVLA、π₀.₅ 的区别，只在"怎么建模 p(action | observation)"上，
这正是你的强项。工程部分（串口、相机、命令行）只是把数据搬进搬出的管道，跟着教程走就行。

## 2. 用 Windows 而不是 Linux，会不会"不正规"？

不会。原因很具体：

- LeRobot 不依赖 ROS，也不依赖任何 Linux 专属的东西。舵机走串口，相机走 OpenCV，训练是 PyTorch。
- 你在 Windows 上已经验证过：12 个舵机全部能读到，3 个相机都能拍照，GPU 能跑 CUDA。
- 唯一的差别是串口名字：Linux 叫 `/dev/ttyACM0`，Windows 叫 `COM3`。官方文档里看到 `/dev/tty...`，换成 COM 口就行。
- 招聘方看的是你**理解策略、做出了结果、能讲清楚取舍**，不是你用什么操作系统。

## 3. 文件类型速查

| 扩展名 | 是什么 | 用什么打开 | 本项目里的例子 |
|---|---|---|---|
| `.py` | Python 源代码。用 `python 文件名.py` 运行 | VS Code | `scripts/scan_motors.py` |
| `.ps1` | PowerShell 脚本，Windows 的"批处理"。把一长串命令打包成一个 | VS Code；运行：`.\scripts\robot.ps1 ...` | `scripts/robot.ps1` |
| `.json` | 配置数据，`{"键": 值}` 的结构，和 Python 的 dict 几乎一样 | 任何文本编辑器 | `config/robot.json` |
| `.md` | Markdown 文档：`#` 是标题，`**粗体**`，``` 包起来的是代码块。GitHub 会自动渲染成网页 | VS Code（`Ctrl+Shift+V` 预览） | `README.md`、`docs/*.md` |
| `.gitignore` | 告诉 git **哪些文件不要上传**（大文件、密钥、本机配置） | 文本编辑器 | 根目录 `.gitignore` |
| `.jpg` | 图片 | 看图软件 | `outputs/hardware_check/camera_1.jpg` |
| `.parquet` | 表格数据（列式存储），LeRobot 用它存关节角度和动作 | Python（pandas） | 录数据后出现 |
| `.mp4` | 视频，LeRobot 把相机画面压成视频以节省空间 | 播放器 | 录数据后出现 |
| `.safetensors` | 模型权重（和 `.pt` 类似但更安全、更快加载） | Python | 训练后出现 |

## 4. 目录结构

```
Lerobot/
├── README.md              项目门面（英文，给招聘方看）：做了什么、结果如何
├── .gitignore             不上传哪些文件
├── config/
│   ├── robot.example.json 配置模板（会上传到 GitHub）
│   ├── robot.json         你这台电脑的真实配置：COM 口、相机编号（不上传，每台电脑不一样）
│   └── calibration/       两条臂的标定文件（上传：数据集依赖它）
├── scripts/               自己写的小工具
│   ├── env.ps1            所有脚本共用的环境设置：让 LeRobot 把数据集、标定存进本仓库
│   ├── find_hardware.py   列出串口和相机，并给每个相机拍一张照片
│   ├── scan_motors.py     只读地检查 6 个舵机的 ID 和角度
│   ├── check_cameras.py   录制前的相机预检查
│   ├── inspect_dataset.py 录完后检查数据质量（空闲占比、开头等待、缩略图）
│   ├── robot.ps1          读 config/robot.json，拼好标定/遥操作/录制/部署命令
│   ├── train.ps1          训练入口：每个策略的参数预设
│   ├── bench_dataloader.py 测试不同数据加载设置的速度（训练前调优用）
│   └── lerobot_train_win.py  Windows 修复版 lerobot-train（见 docs/04_act.md）
├── docs/                  中文教程，一个阶段一个文件
├── data/                  录好的数据集（不上传 git，HF Hub 上有一份）
└── outputs/               运行时产生的东西：截图、训练 checkpoint、日志（不上传）
```

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
