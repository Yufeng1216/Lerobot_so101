# 06 · SmolVLA：小型视觉-语言-动作模型

论文：Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025
（[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)）。

前两个策略（ACT、Diffusion）都是**从零开始**、只用我们的 50 条示范训练。SmolVLA 不一样：
它是一个**预训练好的大模型**，我们只做**微调**（fine-tuning）。这和 NLP 里"拿预训练的 BERT / LLM 在自己的数据上微调"是同一个思路。

## 1. VLA 是什么

**VLA = Vision-Language-Action**：输入图像和一句自然语言指令，输出机器人动作。

```
图像（俯视 + 腕部） ┐
任务指令（文字）    ├─▶  VLA  ─▶  未来 50 步的关节动作
关节状态           ┘
```

和 ACT / Diffusion 的区别：

| | ACT / Diffusion | SmolVLA |
|---|---|---|
| 初始化 | 只有图像编码器（ResNet18）用 ImageNet 预训练 | **整个模型**在大量社区机器人数据上预训练过 |
| 输入 | 图像 + 关节状态 | 图像 + 关节状态 + **语言指令** |
| 训练 | 从零开始 | 在预训练权重上微调 |

本项目只有一个任务，语言指令始终是同一句话（"Pick up the blue plush toy and place it in the cardboard box."），
所以语言在这里发挥不了"区分任务"的作用。我们关心的是：**预训练带来的先验，能不能让 50 条示范学得更好**。

## 2. 结构：VLM + 动作专家

SmolVLA 由两部分组成（约 4.5 亿参数）：

```
 图像 ─ SigLIP 视觉编码器 ─┐
 指令 ─ 分词 ──────────────┼─▶ SmolVLM2（语言模型，只用前 16 层）─▶ 每层的特征
 关节状态 ─ Linear ────────┘                                         │
                                                                     ▼（交叉注意力）
 噪声动作 + 时间 t ─────────────────────────────▶ 动作专家（小 Transformer）─▶ 速度场 v
```

1. **VLM（视觉语言模型）**：SmolVLM2-500M。图像经 SigLIP 编码成视觉 token，和指令的文字 token 一起送进语言模型。
   为了快，只用语言模型的**前 16 层**（论文发现中间层的特征对动作已经足够）。
2. **动作专家（action expert）**：一个更小的 Transformer（宽度是 VLM 的 0.75 倍），
   交替使用**交叉注意力**（看 VLM 的特征）和**自注意力**（动作 token 之间互相看）。

### Flow matching：和 Diffusion 很像，但更直接

动作专家用 **flow matching** 生成 50 步的动作块，可以看成 Diffusion 的"直线版"：

- 训练：在真实动作 $A$ 和噪声 $\epsilon$ 之间取一个点 $A^t = t\,\epsilon + (1-t)\,A$，
  让网络预测从噪声指向数据的**速度** $v = \epsilon - A$，损失是 $\lVert v_\theta(A^t, t, O) - v \rVert^2$。
- 推理：从纯噪声出发，沿预测的速度场积分 **10 步**（`num_steps=10`）得到动作。
  Diffusion Policy 默认要去噪 100 步，所以 SmolVLA 推理快得多。

## 3. 微调时训练哪些部分

默认配置（我们不改）：

| 部分 | 是否训练 | 配置项 |
|---|---|---|
| SigLIP 视觉编码器 | 冻结 | `freeze_vision_encoder=True` |
| 语言模型 | 冻结 | `train_expert_only=True` |
| 动作专家 | **训练** | |
| 关节状态的投影层 | **训练** | `train_state_proj=True` |

只训练约 1 亿参数，所以虽然模型有 4.5 亿参数，梯度和优化器状态只占一小部分显存。

## 4. 相机名对应：`rename_map`

预训练模型认识的相机名是 `camera1`、`camera2`、`camera3`，我们的数据集里叫 `top` 和 `wrist`。
用 `--rename_map` 告诉 LeRobot 怎么对应：

| 我们的相机 | 对应到 | 为什么 |
|---|---|---|
| `observation.images.top` | `observation.images.camera1` | 预训练数据里 camera1 多是俯视或正面视角 |
| `observation.images.wrist` | `observation.images.camera2` | camera2 多是腕部视角 |
| （没有） | `camera3` | 缺少的相机直接跳过 |

## 5. 本项目的训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.path` | `lerobot/smolvla_base` | 从官方预训练权重开始（不是 `policy.type`，那样会从零训练） |
| `rename_map` | top → camera1，wrist → camera2 | 见第 4 节 |
| `batch_size` | 8（集群上 32，见第 7 节） | 官方示例用 64，8 GB 显存放不下；batch 小，学习率不变，步数相应多一些 |
| `steps` | 30,000 | 约 16 个 epoch |
| `scheduler_decay_steps` | 和 `steps` 相同 | 默认按 3 万步衰减学习率，改步数时要一起改 |
| `save_freq` | 10,000 | 3 个 checkpoint |
| 其余 | 默认 | 学习率 1e-4，1000 步预热，余弦衰减到 2.5e-6 |

## 6. 冒烟测试

```powershell
.\scripts\train.ps1 smolvla -Steps 300 -Name smolvla_smoke -NoPush --eval_steps=150 --trackio.enable=false
```

第一次运行会下载预训练模型（约 2 GB），存在 Hugging Face 的缓存目录里。

实测（RTX 3070 Laptop）：

| batch | 每步耗时 | 显存 | 结论 |
|---|---|---|---|
| 8 | 0.35 s | 2.9 GB | 可用；300 步内 eval_loss 0.465 → 0.379 |
| 32 | 约 800 s | 7.85 GB | 超出 8 GB，Windows 偷用内存，慢 2000 倍，不可用 |

可训练参数 100M / 总参数 450M。batch 8 跑 3 万步约 3 小时。

## 7. 正式训练

两种做法，配置不同：

| | 笔记本（8 GB） | GPU 集群（16 GB V100） |
|---|---|---|
| batch | 8 | 32 |
| 步数 | 30,000 | 20,000（样本数约为笔记本的 2.7 倍） |
| save_freq | 10,000 | 5,000（超时重提交最多损失 5k 步） |
| 实测每步 | 0.35 s | 2.47 s（V100 没有 bf16 加速），2 万步约 14 小时：8 小时超时后再提交一次，自动续训 |
| 命令 | `.\scripts\train.ps1 smolvla` | `bash cloud/slurm/submit.sh smolvla` |

笔记本：

```powershell
hf auth whoami
.\scripts\train.ps1 smolvla
```

集群：用法见 [`cloud/README.md`](../cloud/README.md)。V100 不原生支持 bfloat16（SmolVLA 的 VLM 部分用 bf16 存），能跑但会慢一些，开始后先看日志里的 `step_s`。训练完把 `checkpoints/last/pretrained_model` 拷回笔记本。

## 8. 在真机上运行

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model -Duration 30
```

`robot.ps1` 会自动把 `config/robot.json` 里的任务描述作为语言指令传给模型。

## 9. 和前两个策略对比时要看什么

- **成功率**：同样按 [08_evaluation.md](08_evaluation.md) 跑 20 次。
- **预训练的作用**：同样 50 条示范，从预训练权重微调的 SmolVLA 和从零训练的 ACT / Diffusion 相比如何。
- **推理速度**：flow matching 10 步积分，比 Diffusion 的 100 步去噪快，但模型本身比 ACT 大很多。
