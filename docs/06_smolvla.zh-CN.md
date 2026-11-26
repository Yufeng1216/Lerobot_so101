# 06 · SmolVLA：小型视觉-语言-动作模型

[English](06_smolvla.md) · **简体中文**

论文：Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025
（[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)）。

前两个策略（ACT、Diffusion）都是**从零开始**、只用我们的 50 条示范训练的。SmolVLA 不一样：它是一个**预训练好的大模型**，我们只做**微调**。
其实这和 NLP 里"先在大语料上 pretrain，再在自己的任务上 fine-tune"是同一个思路，只是语料换成了机器人数据。

## 1. VLA 是什么

**VLA = Vision-Language-Action**：输入图像和一句自然语言指令，输出机器人动作。

```
图像（俯视 + 腕部） ┐
任务指令（文字）    ├─▶  VLA  ─▶  未来 50 步的关节动作
关节状态           ┘
```

出发点是：VLM（视觉语言模型）已经在海量图文数据上学会了"看懂画面"和"听懂指令"。既然如此，与其让机器人策略从零学视觉，
不如直接拿 VLM 当骨干，只在后面接一个输出动作的模块。

| | ACT / Diffusion | SmolVLA |
|---|---|---|
| 初始化 | 只有 ResNet18 用 ImageNet 预训练 | VLM 在图文数据上预训练，**整个 VLA** 又在社区上传的机器人数据上预训练 |
| 输入 | 图像 + 关节状态 | 图像 + 关节状态 + **语言指令** |
| 训练 | 从零开始 | 在预训练权重上微调 |

PS：我们只有一个任务，指令一直是同一句话（"Pick up the blue plush toy and place it in the cardboard box."），所以语言在这里起不到"区分任务"的作用。
我们关心的是：**预训练带来的先验，能不能让 50 条示范学得更好**。

## 2. 整体结构：VLM 做前缀，动作专家做后缀

SmolVLA（约 4.5 亿参数）由两个 Transformer 组成，它们**一层一层并排**跑：

```
 前缀（prefix）：VLM 处理                         后缀（suffix）：动作专家处理
┌───────────────────────────────────────────┐   ┌──────────────────────────────┐
│ 图像 token（每张 64 个）× 2 路相机           │   │ 50 个动作 token               │
│ 指令 token（分词后最多 48 个）               │   │ = 噪声动作 + 时间 t 的编码     │
│ 状态 token（1 个）                          │   │                              │
└───────────────────────────────────────────┘   └──────────────────────────────┘
        SmolVLM2 的前 16 层（宽 960）          ◀──注意力──▶   动作专家 16 层（宽 720）
                                                                   │
                                                     Linear → 50 × 6 的速度场 v
```

下面一块块拆开看。

### 2.1 VLM：SmolVLM2-500M

VLM 的结构其实就是 "ViT + 投影层 + 语言模型"：

1. **视觉编码器 SigLIP**（一个 ViT，12 层，宽 768）：图像先等比缩放并补边到 512×512，切成 16×16 的 patch，得到 32×32 = 1024 个 patch token。
2. **Pixel shuffle**：1024 个 token 对语言模型来说太多了。所以把相邻 4×4 个 token 在通道维拼起来，token 数变成 1024 / 16 = **64 个**，每个的维度变大 16 倍。
   这是一种"用通道换空间"的压缩，信息没丢，只是重新排列了。
3. **投影层（connector）**：一个线性层把这 64 个 token 映射到语言模型的宽度 960，于是图像 token 和文字 token 处在同一个 embedding 空间里，可以直接拼成一个序列。
4. **语言模型 SmolLM2**（32 层，宽 960，15 个注意力头，5 个 KV 头，即 GQA）：SmolVLA **只用前 16 层**。
   论文发现中间层的特征对动作已经够用了，砍掉一半层数，推理快一倍。

关节状态（6 维，补零到 32 维）经过一个线性层 `state_proj` 变成 **1 个 token**，放在前缀最后。

### 2.2 动作专家（action expert）

动作专家也是一个 Transformer，层数和 VLM 一样（16 层），但宽度只有 0.75 倍（720）。它的输入是 50 个动作 token：

- 噪声动作（50 × 32，6 维补零到 32）经 `action_in_proj` 变成 50 个 720 维向量；
- 时间 $t$ 用正弦编码，和每个动作向量拼起来，再过一个两层 MLP。所以每个动作 token 都"知道"现在的噪声有多大。

输出端用 `action_out_proj` 把 50 个 token 映射回 50 × 32，取前 6 维，就是预测的速度场 $v$。

### 2.3 两个 Transformer 怎么交流：交替的 cross-attention 和 self-attention

每一层里，VLM 先算出前缀 token 的 K、V。动作专家这一层分两种：

- **cross-attention 层**：动作 token 的 Q 去看**前缀**的 K、V（经过一个线性层映射到专家的维度）。也就是说"根据画面、指令和状态，决定动作"。
- **self-attention 层**（每 2 层一次）：动作 token 和前缀 token 拼在一起做注意力，动作 token 之间也能互相看到，让 50 步动作彼此协调。

注意力还有一个 mask 规则（"块状因果"）：

| 谁能看谁 | 图像 + 指令 | 状态 | 动作 |
|---|---|---|---|
| 图像 + 指令 | ✓（互相看） | ✗ | ✗ |
| 状态 | ✓ | ✓ | ✗ |
| 动作 | ✓ | ✓ | 只看自己和前面的动作 |

**前缀永远看不到动作**。这一点很关键：推理时前缀只需要算**一次**（存成 KV cache），之后 10 步去噪只跑动作专家，所以很快。

## 3. Flow matching：Diffusion 的"直线版"

动作专家用 **flow matching** 生成动作块。思路和 Diffusion 一样，都是"从噪声变到数据"，但路线更简单。

### 3.1 出发点

DDPM 的加噪路线是一条弯曲的路（$\sqrt{\bar\alpha_k}$ 和 $\sqrt{1-\bar\alpha_k}$ 非线性地变化），所以推理时要走很多小步。
既然如此，能不能直接让噪声和数据之间走**直线**？

### 3.2 训练

取真实动作 $A$ 和噪声 $\epsilon \sim \mathcal N(0, I)$，在它们之间线性插值（$t=1$ 是纯噪声，$t=0$ 是数据）：

$$A^t = t\,\epsilon + (1-t)\,A$$

沿这条直线的速度是常数 $\frac{dA^t}{dt} = \epsilon - A$。所以让网络预测这个速度：

$$\mathcal L = \big\| v_\theta(A^t, t, O) - (\epsilon - A) \big\|^2$$

$t$ 不是均匀采样的，而是从 Beta(1.5, 1) 采样，**偏向噪声大的一端**（$t$ 接近 1）。因为那一端最难：要从一团噪声里决定往哪个峰走。

### 3.3 推理

从 $A^1 = \epsilon$ 出发，用欧拉法沿速度场积分 10 步（`num_steps = 10`，步长 $\Delta t = -0.1$）：

$$A^{t+\Delta t} = A^t + \Delta t \cdot v_\theta(A^t, t, O)$$

| | DDPM（Diffusion Policy） | Flow matching（SmolVLA） |
|---|---|---|
| 网络预测什么 | 加进去的噪声 $\epsilon$ | 从数据指向噪声的速度 $\epsilon - A$ |
| 加噪路线 | 弯曲（由噪声表决定） | 直线 |
| 推理步数 | 默认 100（DDIM 可以 10） | 10 |

PS：路线是直的，所以少走几步误差也不大，这就是 flow matching 推理快的原因。多模态的能力和 Diffusion 一样，因为每次都从不同的噪声出发。

## 4. 微调时训练哪些部分

默认配置（我们不改）：

| 部分 | 训不训 | 配置项 |
|---|---|---|
| SigLIP 视觉编码器 | 冻结 | `freeze_vision_encoder=True` |
| 语言模型（前 16 层） | 冻结 | `train_expert_only=True` |
| 动作专家 | **训练** | |
| 状态投影、动作投影、时间 MLP | **训练** | `train_state_proj=True` |

只训约 1 亿参数。所以虽然模型有 4.5 亿参数，梯度和优化器状态只占一小部分显存。
PS：其实这和 LoRA 的想法类似：大模型的通用知识冻住不动，只训一小部分参数去适配新任务。

## 5. 相机名对应：`rename_map`

预训练模型认识的相机名是 `camera1`、`camera2`、`camera3`，我们的数据集里叫 `top` 和 `wrist`。所以要用 `--rename_map` 告诉 LeRobot 怎么对应：

| 我们的相机 | 对应到 | 为什么 |
|---|---|---|
| `observation.images.top` | `observation.images.camera1` | 预训练数据里 camera1 多是俯视或正面视角 |
| `observation.images.wrist` | `observation.images.camera2` | camera2 多是腕部视角 |
| （没有） | `camera3` | 缺少的相机直接跳过 |

PS：`rename_map` 会存进模型的预处理配置（`policy_preprocessor.json`），所以部署时自动套用，不用再传。

## 6. 训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.path` | `lerobot/smolvla_base` | 从官方预训练权重开始（不是 `policy.type`，那样会从零训） |
| `rename_map` | top → camera1，wrist → camera2 | 见第 5 节 |
| `batch_size` | 32 | 官方示例用 64；32 在 16 GB 的 V100 上放得下 |
| `steps` | 20,000 | 64 万个样本，约 43 个 epoch |
| `scheduler_decay_steps` | 和 `steps` 相同 | 默认按 3 万步衰减学习率，改步数要一起改 |
| `save_freq` | 5,000 | 集群任务超时重新提交时最多损失 5k 步 |
| 其余 | 默认 | 学习率 1e-4，1000 步预热，余弦衰减到 2.5e-6 |

PS：笔记本（8 GB）上 batch 32 放不下（显存溢出到系统内存，慢上千倍），batch 8 可以跑（每步 0.35 s，2.9 GB）。
所以 `train.ps1 smolvla` 是 batch 8、3 万步的笔记本版本；正式模型是在集群上用 batch 32 训的。

## 7. 训练

先跑个冒烟测试，第一次运行会下载预训练模型（约 2 GB，存在 Hugging Face 的缓存里）：

```powershell
.\scripts\train.ps1 smolvla -Steps 300 -Name smolvla_smoke -NoPush --eval_steps=150 --trackio.enable=false
```

正式训练在集群上（用法见 [cloud/README.zh-CN.md](../cloud/README.zh-CN.md)）：

```bash
bash cloud/slurm/submit.sh smolvla
```

PS：V100 不原生支持 bfloat16（SmolVLA 的 VLM 部分用 bf16 存），能跑但会慢一些：每步 2.47 s，2 万步约 14 小时。
8 小时超时后再提交一次，会自动接着训。

训完只把部署要用的 `pretrained_model` 拷回笔记本（约 0.9 GB，不含优化器状态），在**笔记本**上运行：

```powershell
scp -r <用户名>@<集群传输节点>:~/Lerobot/outputs/train/smolvla_so101_pick_place/checkpoints/020000/pretrained_model outputs/train/smolvla_so101_pick_place/
```

### 训练结果

- 2 万步训满，V100 上共约 14 小时。
- **离线检查**：从 5 条示范里各取一帧（抓取前、抓取中、搬运中、放下时），预测 50 步动作块，和示范里记录的动作比，
  每个关节的平均绝对误差（度；夹爪是 0–100）：

| | 底座 | 大臂 | 小臂 | 腕俯仰 | 腕旋转 | 夹爪 | 平均 |
|---|---|---|---|---|---|---|---|
| SmolVLA 预测 | 1.4 | 3.0 | 2.7 | 1.9 | 2.3 | 0.5 | **2.0** |
| 对照：原地不动 | 14.2 | 29.3 | 31.2 | 13.1 | 3.2 | 7.3 | 16.4 |

  PS：这些帧都在训练集里，只能说明模型加载正确、训练有效，说明不了泛化能力（同样的检查 π₀.₅ 是 1.7，见 [07_pi05.zh-CN.md](07_pi05.zh-CN.md)）。
- **推理速度**：笔记本 GPU 上一个动作块约 0.38 秒（第一次 1.2 秒，含预热），显存 0.9 GB。一块能执行 1.67 秒，所以可以直接在笔记本上跑真机。

## 8. 在真机上运行

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/smolvla_so101_pick_place/pretrained_model -Duration 30
```

`robot.ps1` 会自动把 `config/robot.json` 里的任务描述当作语言指令传给模型。

## 9. 和前两个策略比的时候看什么

- **成功率**：同样按 [08_evaluation.zh-CN.md](08_evaluation.zh-CN.md) 测 40 次。
- **预训练的作用**：同样 50 条示范，从预训练权重微调的 SmolVLA 和从零训练的 ACT / Diffusion 比，谁更好。
- **推理速度**：flow matching 10 步积分 + 前缀只算一次，比 Diffusion 的去噪快；但模型本身比 ACT 大很多。
