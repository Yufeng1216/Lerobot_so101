# 04 · ACT：Action Chunking with Transformers

论文：Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023
（[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)）。

ACT 是四个策略里**最先训练**的：参数少（本项目配置下 51.6M）、训练快、50 条示范通常就能学会。
它也是后面三个策略的**基线**。

## 1. 问题设定

行为克隆：给定观测 $o_t$（两路图像 + 6 维关节状态），预测动作 $a_t$（6 维目标关节角度）。
最朴素的做法是训练一个网络 $\pi(a_t \mid o_t)$，每一步预测一个动作，用 MSE 回归。实际效果很差，原因有两个：

1. **误差累积（compounding error）**：每一步的小误差会让机器人进入训练数据里没见过的状态，
   下一步误差更大，滚雪球。单步预测时，一条 300 步的 episode 要连续做对 300 次决策。
2. **示范不一致**：人在同一个画面下，有时先停顿再抓，有时直接抓。对 MSE 来说这是
   同一个输入对应多个标签，模型只会学到它们的平均值：一个"犹豫不决"的动作。

## 2. ACT 的两个核心想法

### (1) Action chunking：一次预测未来 k 步

ACT 预测的是 $\pi(a_{t:t+k} \mid o_t)$，一次输出 **k = 100 步**动作（30 Hz 下约 3.3 秒）。

- 决策次数减少到原来的 1/k，误差累积的"有效步数"大大缩短。
- 一段动作作为整体预测，内部自然连贯，相当于在动作序列上建模，而不是逐点建模。
  类比：语言模型一次生成一个短语，而不是每个字母都重新决定。

### (2) CVAE：用隐变量吸收示范中的"风格差异"

训练时有一个额外的 **CVAE 编码器**，它看到"真实的未来动作序列 + 当前关节状态"，
把它压缩成一个 32 维的隐变量 $z$（风格向量：快一点还是慢一点、从哪边绕过去）。
解码器以 $z$ 为条件来重建动作序列。

- 训练：$z \sim q_\phi(z \mid a_{t:t+k}, s_t)$，损失 = **L1 重建误差 + β · KL(q ‖ N(0, I))**，β = `kl_weight` = 10。
- 推理：没有未来动作可以看，直接令 **z = 0**（先验的均值），得到一个"标准风格"的动作序列。

这样，示范之间的不一致被 $z$ 解释掉了，解码器不必学一个模糊的平均值。

## 3. 网络结构

```
 top 图像 ─┐
           ├─ ResNet18（ImageNet 预训练）→ 特征图 → 展平成 token（每张图 15×20 = 300 个）
wrist 图像 ┘
关节状态 s_t ─ Linear → 1 个 token
隐变量 z     ─ Linear → 1 个 token          （训练时来自 CVAE 编码器，推理时 z = 0）
                 │
                 ▼
     Transformer 编码器（4 层，d=512，8 头）  ← 所有 token 做自注意力
                 │
                 ▼
     Transformer 解码器（1 层）
       query = 100 个可学习的位置 query      ← 每个 query 负责输出一步动作
       cross-attention 到编码器输出
                 │
                 ▼
     Linear → 100 × 6 的动作序列 a_{t:t+100}
```

和你熟悉的模型对应起来：这基本就是 **DETR**（目标检测 Transformer）的结构，
把"100 个物体 query"换成了"100 个时间步 query"。

## 4. 推理时怎么执行动作块

默认（`n_action_steps = 100`）：预测 100 步 → 全部执行完 → 再看一次画面、预测下一块。
每 3.3 秒才"看"一次，反应慢，块与块的交界处可能有跳变。

**Temporal ensembling**（论文里的做法，可选）：**每一步**都重新预测一个 100 步的块，
于是同一时刻 t 会有多个块对它给出预测；按指数权重 $w_i = e^{-m \cdot i}$ 对它们加权平均后再执行。
动作更平滑，反应更快，代价是每步都要跑一次网络（ACT 足够小，30 Hz 跑得动）。
部署时用 `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` 开启。两种方式我们都会测。

## 5. 本项目的训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.type` | `act` | |
| `chunk_size` | 100（默认） | 3.3 秒，大约是一次"抓"或"放"的时长 |
| `kl_weight` | 10（默认） | 论文值 |
| `optimizer_lr` | 1e-5（默认，backbone 也是 1e-5） | 论文值；Transformer + 预训练 backbone，小学习率更稳 |
| `batch_size` | 8 | 8 GB 显存下两路 480×640 图像的稳妥值 |
| `steps` | 50,000 | 约 45 条 × 330 帧 ≈ 15k 帧，相当于约 27 个 epoch |
| `save_freq` | 10,000 | 保存 5 个 checkpoint，可以比较不同训练阶段在真机上的效果 |
| `dataset.eval_split` | 0.1 | 留出 10% 的 episode 做**验证集**，每 2000 步算一次验证 loss |

关于验证 loss：在模仿学习里，验证 loss 低**不等于**真机成功率高（误差累积只有在闭环里才出现），
但它能告诉你是否严重过拟合。**最终以真机成功率为准**，这是后面要比较的指标。

## 6. 先跑一个冒烟测试（几分钟）

正式训练要好几个小时。先用 200 步确认整条管线（读数据 → 解码视频 → 前向反向 → 保存 checkpoint）
在 Windows 上没问题，同时测出训练速度，以便估计总时长：

```powershell
.\scripts\train.ps1 act -Steps 200 -Name act_smoke -NoPush --eval_steps=100
```

200 步结束时会保存一次 checkpoint，`--eval_steps=100` 让验证集评估也跑两次，
这样管线的每个环节都被覆盖到。确认没问题后可以删掉 `outputs/train/act_smoke`。

> **Windows 坑：符号链接权限。** LeRobot 每次保存 checkpoint 都会创建一个 `checkpoints/last`
> 符号链接指向最新的 checkpoint，而 Windows 普通用户没有创建符号链接的权限（`WinError 1314`），
> 训练会在**第一次保存时崩溃**。`scripts/train.ps1` 通过 `scripts/lerobot_train_win.py`
> 启动训练，把这个链接换成不需要特权的**目录联接（junction）**，其他行为和 `lerobot-train` 完全一样。
> （另一种解决办法是在 Windows 设置里打开"开发者模式"。）
>
> 联接只能指向**绝对路径**：如果之后把整个项目文件夹挪到别处，旧 run 里的 `last` 会失效。
> 这时直接用带编号的目录（如 `checkpoints/050000/pretrained_model`）即可，模型本身不受影响。

实测（RTX 3070 Laptop，batch 8）：200 步约 5 分钟，显存 3.7 GB；每步约 1.4 秒，
其中约 1 秒在**等数据**（CPU 解码两路视频），只有约 0.4 秒在做 GPU 计算。

## 7. 正式训练

```powershell
.\scripts\train.ps1 act
```

等价的完整命令会以青色打印在终端开头。训练产物：

```
outputs/train/act_so101_pick_place/
├── train_config.json                 本次训练的完整配置（可复现）
└── checkpoints/
    ├── 010000/pretrained_model/      第 1 万步的模型
    │   ├── model.safetensors         权重
    │   ├── config.json               策略结构参数
    │   ├── train_config.json
    │   └── policy_preprocessor*.json / policy_postprocessor*.json  归一化统计量等前后处理
    ├── 020000/ ...
    └── last  →  指向最新的 checkpoint（目录联接）
```

训练结束后，模型会自动上传到 `https://huggingface.co/Yufeng1216/act_so101_pick_place`。

中断了可以接着训：

```powershell
.\scripts\train.ps1 act -Resume
```

### 看训练曲线

训练用 Trackio 记录 loss，在另一个终端里运行：

```powershell
conda activate lerobot
trackio show --project so101
```

浏览器里会打开一个面板。重点看：

- `train/loss`：总损失 = `l1_loss` + 10 × `kld_loss`，应该稳定下降
- `train/l1_loss`：动作重建误差（归一化空间里），主要看它
- `train/kld_loss`：应该很小但不为 0；如果一直是 0，说明 CVAE 的隐变量没被用上（posterior collapse）
- `eval/loss`：验证集 loss。如果训练 loss 一直降、验证 loss 却开始上升，就是过拟合了

## 8. 在真机上运行

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30
```

`-Policy` 也可以填 Hub 上的模型名 `Yufeng1216/act_so101_pick_place`。

安全注意：

- 第一次运行时手放在从臂电源旁边，策略可能做出意料之外的动作。
- 策略启动时从臂会从当前姿态移动到策略输出的第一个动作，先把从臂摆在**起始姿态**。
- 场景要和录数据时一样：相机位置、快递盒、标记点、光照都不要动。

试试 temporal ensembling：

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
```

## 9. 评估

按 [08_evaluation.md](08_evaluation.md) 的协议：5 个标记点 × 每点 4 次 = 20 次，记录成功率。
四个策略都用同一个协议，结果填进 README 的结果表。
