# SO-101 模仿学习：ACT → Diffusion Policy → SmolVLA → π₀.₅

[English](README.md) · **简体中文**

用 [🤗 LeRobot](https://github.com/huggingface/lerobot) 在低成本的 **SO-101** 机械臂上复现四个主流的机器人学习策略，全流程都在真机上做：
遥操作采集数据 → 训练 → 真机部署和评估。另外和一个**完全不用示范的零样本基线**对比
（现成的视觉语言模型负责感知，逆运动学负责动作）。

**任务：** 从 5 个标记点之一抓起毛绒玩具，放进快递盒。
**数据集：** 50 条遥操作示范，[`Yufeng1216/so101_pick_place`](https://huggingface.co/datasets/Yufeng1216/so101_pick_place)。

| ACT（点位 ⑤） | Diffusion Policy（点位 ②） |
|---|---|
| ![ACT 运行](assets/act_success.gif) | ![Diffusion Policy 运行](assets/diffusion_success.gif) |

<sub>评估时录下的自主运行片段，2 倍速。每段左边是俯视相机，右边是腕部相机。</sub>

| 策略 | 论文 | 参数量 | 类型 | 在哪训练 |
|---|---|---|---|---|
| **ACT** | [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) | ~52 M | CVAE + Transformer，action chunking | 本地 RTX 3070（8 GB） |
| **Diffusion Policy** | [Chi et al., 2023](https://arxiv.org/abs/2303.04137) | ~278 M | 对动作块做 DDPM/DDIM 去噪 | 学校 GPU 集群（Slurm） |
| **SmolVLA** | [Shukor et al., 2025](https://arxiv.org/abs/2506.01844) | ~450 M | 小型 VLM + flow matching 动作专家 | 学校 GPU 集群（Slurm） |
| **π₀.₅** | [Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054) | ~4.1 B | VLA，PaliGemma + flow matching 动作专家 | 租用的 A100 80 GB（Hugging Face Jobs） |
| **零样本 VLM + IK**（基线） | — | ~2 B（VLM） | Qwen3-VL 指出玩具和盒子 → 桌面单应性 → IK → 固定抓取脚本 | 不训练 |

## 结果

所有策略都用同一套协议在真机上评估（[docs/08_evaluation.zh-CN.md](docs/08_evaluation.zh-CN.md)）：**40 次试验**（5 个起始点 × 8），每次 30 秒，
光照和录示范时一样（晚上，只开台灯）。每个策略 40 次连续测完（8 组 × 5 次），换策略之间机械臂冷却 20 分钟。
成功 = 玩具放进盒子并松开。逐次记录在 [`results/`](results/)。

> **状态：** ACT 和 Diffusion Policy 已测完（各 40 次；两者比较 Fisher 精确检验 p = 0.025）。接下来是 SmolVLA、π₀.₅ 和零样本基线。

| 策略 | 步数 × batch | 看过的样本（epoch） | 训练时长 | 成功 | 95% 区间 | 抓起没放进 | 没抓起 | 每个动作块推理 | 模型 | 片段 |
|---|---|---|---|---|---|---|---|---|---|---|
| ACT | 50k × 8 | 400k (≈27) | 4.0 h，RTX 3070 Laptop | **14/40**（[记录](results/act.zh-CN.md)） | 22–51% | 4 | 22 | ~47 ms | [🤗](https://huggingface.co/Yufeng1216/act_so101_pick_place) | [gif](assets/act_success.gif) |
| Diffusion Policy | 100k × 8 | 800k (≈53) | ≈4 h，Tesla V100 16 GB | **25/40**（[记录](results/diffusion.zh-CN.md)） | 47–76% | 7 | 8 | ~200 ms（DDIM 10 步） | [🤗](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place) | [gif](assets/diffusion_success.gif) |
| SmolVLA | 20k × 32 | 640k (≈43) | ≈14 h，Tesla V100 16 GB | – | – | – | – | ~0.38 s（RTX 3070 Laptop） | – | – |
| π₀.₅ | 3k × 32 | 96k (≈6) | 7.1 h，A100 80 GB（≈ $18） | – | – | – | – | – | – | – |
| 零样本 VLM + IK | 0（不用示范） | 0 | – | – | – | – | – | – | – | – |

<sub>epoch 按 1.5 万帧的训练集计算（16,717 帧的 90%）。95% 区间：Wilson 区间。</sub>

## 硬件

- SO-101 从臂（6 个飞特 STS3215 舵机）+ 遥操作用的 SO-101 主臂
- 两块 USB 舵机驱动板（WCH CH343 和 CH340 串口芯片）
- 两个 USB 相机（640×480 @ 30 fps）：`top`（俯视，装在台灯支架上）和 `wrist`（装在从臂夹爪上）
- 笔记本：Windows 11，RTX 3070 Laptop GPU（8 GB）

## 软件

- 原生 Windows 11（不用 WSL / 虚拟机：串口和相机直接可用，也能用 CUDA）
- Python 3.12，conda 环境 `lerobot`
- LeRobot **0.7.0**（固定版本），PyTorch + CUDA 12.8

## 仓库结构

```
config/          robot.example.json → 复制成 robot.json（填自己的 COM 口、相机编号、HF 用户名）
scripts/         在 Windows 笔记本上运行：硬件检查、录数据、本地训练、部署
cloud/           在别的机器上运行：Slurm 集群和租用的 Hugging Face GPU，给显存超过 8 GB 的模型用
zero_shot/       不用示范的基线：SO-101 运动学、像素→桌面单应性、本地 VLM 指向、真机抓取脚本
docs/            分步教程，一个阶段一个文件（英文 *.md，中文 *.zh-CN.md）
results/         真机评估的逐次记录表
assets/          README 用的媒体（运行 GIF）
```

## 教程

模型相关的章节（4–7、9）会先从基础讲清楚模型的结构和原理，再给命令。工程工具不熟的话，先看 [docs/how_this_repo_works.zh-CN.md](docs/how_this_repo_works.zh-CN.md)（每个文件和工具是干什么的）。

| # | 阶段 | 文档 |
|---|---|---|
| 0 | Windows 环境搭建 | [docs/00_environment.zh-CN.md](docs/00_environment.zh-CN.md) |
| 1 | 硬件：串口、舵机 ID、相机 | [docs/01_hardware.zh-CN.md](docs/01_hardware.zh-CN.md) |
| 2 | 标定与遥操作 | [docs/02_calibration_teleop.zh-CN.md](docs/02_calibration_teleop.zh-CN.md) |
| 3 | 录制数据集 | [docs/03_data_collection.zh-CN.md](docs/03_data_collection.zh-CN.md) |
| 4 | ACT | [docs/04_act.zh-CN.md](docs/04_act.zh-CN.md) |
| 5 | Diffusion Policy | [docs/05_diffusion_policy.zh-CN.md](docs/05_diffusion_policy.zh-CN.md) |
| 6 | SmolVLA | [docs/06_smolvla.zh-CN.md](docs/06_smolvla.zh-CN.md) |
| 7 | π₀.₅ | [docs/07_pi05.zh-CN.md](docs/07_pi05.zh-CN.md) |
| 8 | 评估协议与对比 | [docs/08_evaluation.zh-CN.md](docs/08_evaluation.zh-CN.md) |
| 9 | 零样本基线：VLM + IK，不用示范 | [docs/09_zero_shot_vlm.zh-CN.md](docs/09_zero_shot_vlm.zh-CN.md) |
| – | 代码讲解（每个脚本逐段讲） | [docs/code_walkthrough.zh-CN.md](docs/code_walkthrough.zh-CN.md) |

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 致谢

基于 [LeRobot](https://github.com/huggingface/lerobot)（Apache-2.0）和开源机械臂 [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100)。
