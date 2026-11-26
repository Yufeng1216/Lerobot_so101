# 04 · ACT：Action Chunking with Transformers

[English](04_act.md) · **简体中文**

论文：Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023
（[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)）。

ACT 是四个策略里**最先训练**的：参数少（这里的配置是 51.6M），训练快，50 条示范一般就能学会。它也是后面三个策略的**基线**。

## 1. 问题设定

行为克隆（behavior cloning）就是监督学习：给定观测 $o_t$（两路图像 + 6 维关节状态 $s_t$），预测动作 $a_t$（6 维目标关节角度），
标签来自人的示范。最直接的做法是训一个网络 $\pi(a_t \mid o_t)$，每一步预测一个动作，用 MSE 回归。但实际效果很差，原因有两个：

1. **误差累积（compounding error）**：训练时网络看到的都是"人走过的状态"，部署时它看到的是"自己走过的状态"。
   每一步的小误差会把机器人带到训练数据里没见过的状态，下一步误差就更大，越滚越大。单步预测的话，一条 300 步的 episode 要连续做对 300 次决策。
2. **示范不一致（多模态）**：同一个画面下，人有时先停一下再抓，有时直接抓；有时从左边绕，有时从右边绕。
   对 MSE 来说这是同一个输入对应多个标签，而 MSE 的最优解是条件均值 $\mathbb{E}[a \mid o]$，所以模型只能学到它们的平均值，
   也就是一个"犹豫不决"的动作（左右平均就撞到正中间的障碍物）。

ACT 的两个核心想法正好分别对应这两个问题：action chunking 对付误差累积，CVAE 对付多模态。

## 2. 想法一：Action chunking

既然误差是一步步累积的，那就少做几次决策。ACT 预测的不是一个动作，而是一整段：

$$\pi(a_{t:t+k} \mid o_t),\quad k = 100$$

一次输出 **100 步**动作（30 Hz 下约 3.3 秒），输出张量形状是 `(batch, 100, 6)`。

- 决策次数变成原来的 1/k，误差累积的"有效步数"一下就短了 100 倍。
- 一段动作作为整体预测，内部自然是连贯的。网络学的是"一段轨迹长什么样"，而不是 100 个互相独立的点。
  Example：语言模型如果每个字母都独立决定，拼出来的词很容易不成词；一次生成一个短语就通顺得多。
- PS：人的动作本来也是分段的（"伸过去""抓""抬起来"），3.3 秒差不多就是其中一段的长度。

## 3. 想法二：CVAE

### 3.1 先从 VAE 说起

多模态的问题在于：光看 $o_t$，标签 $a_{t:t+k}$ 是不确定的。既然如此，就引入一个隐变量 $z$ 来表示"这次是哪种风格"，
让 $a$ 在给定 $(o, z)$ 之后变得确定：

$$p(a \mid o) = \int p(a \mid o, z)\, p(z)\, dz$$

这就是 VAE（Variational Autoencoder）的思路。问题是训练时不知道每条示范的 $z$ 是多少，积分也算不出来。
所以 VAE 再训一个**编码器** $q_\phi(z \mid a, o)$：它看到答案（真实的动作序列），猜出这条示范的 $z$。训练目标是证据下界（ELBO）：

$$\log p(a \mid o) \ \ge\ \underbrace{\mathbb{E}_{z\sim q_\phi}\big[\log p_\theta(a \mid o, z)\big]}_{\text{重建：用猜出的 } z \text{ 还原动作}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z \mid a, o)\,\|\,p(z)\big)}_{\text{正则：} z \text{ 的分布别离先验太远}}$$

- 重建项：如果 $p_\theta$ 取拉普拉斯分布，$-\log p$ 就是 **L1 误差**，ACT 用的就是 L1。
- KL 项：先验取标准正态 $p(z)=\mathcal{N}(0,I)$，编码器输出高斯 $\mathcal{N}(\mu, \sigma^2)$，KL 有解析解：
  $\mathrm{KL} = -\tfrac12 \sum_j \big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$。
  PS：这就是 NLP 里讲的那个 KL divergence，这里拿它来约束"$z$ 的分布要像标准正态"。
- 采样不可导怎么办？用**重参数化**：$z = \mu + \sigma \odot \epsilon,\ \epsilon \sim \mathcal{N}(0, I)$，梯度就能穿过 $\mu$ 和 $\sigma$。

"C"（Conditional）只是说生成是以观测 $o$ 为条件的。

### 3.2 ACT 里的 CVAE

ACT 的损失就是上面那个 ELBO 取负号：

$$\mathcal{L} = \underbrace{\frac{1}{k}\sum_{i}\big\|a_{t+i} - \hat a_{t+i}\big\|_1}_{\texttt{l1\_loss}} \;+\; \beta \cdot \underbrace{\mathrm{KL}\big(q_\phi(z \mid a_{t:t+k}, s_t)\,\|\,\mathcal{N}(0,I)\big)}_{\texttt{kld\_loss}},\qquad \beta=\texttt{kl\_weight}=10$$

和标准 VAE 有两个不同：

- **编码器不看图像**，只看关节状态 $s_t$ 和动作序列 $a_{t:t+k}$。图像里的信息让解码器自己去用，$z$ 只负责"风格"。
- **推理时直接令 $z = 0$**（先验的均值），而不是从 $\mathcal{N}(0,I)$ 里采样。所以部署时 ACT 是确定性的：同一个观测永远给同一段动作。

PS：$\beta$ 大一点，$z$ 里装的信息就少一点。如果 `kld_loss` 一直是 0，说明编码器把 $z$ 压成了先验、解码器完全不用它（posterior collapse），这时 CVAE 就退化成了普通回归。

## 4. 网络结构

整体分三块：CVAE 编码器（只在训练时用）、主编码器、解码器。

### 4.1 CVAE 编码器（训练时）

```
[CLS] token        ─ 可学习的 embedding         ─┐
关节状态 s_t (6)    ─ Linear → 512              ─┤  共 1 + 1 + 100 = 102 个 token
动作序列 a (100×6)  ─ Linear → 512（每步一个 token）─┘  + 正弦位置编码
                         │
          Transformer 编码器（4 层，d=512，8 头）
                         │
            取 [CLS] 的输出 → Linear → (μ, log σ²)，各 32 维
                         │
            z = μ + σ·ε（重参数化）
```

这就是 BERT 的做法：用一个 [CLS] token 汇总整段序列的信息。

### 4.2 主编码器

```
 top 图像   (3×480×640) ─┐
                         ├─ ResNet18 去掉最后的 pool 和 fc → 特征图 512×15×20
wrist 图像 (3×480×640) ─┘     （下采样 32 倍：480/32=15，640/32=20）
                               → 1×1 卷积 → 展平成 300 个 512 维 token（每张图）
                               + 2D 正弦位置编码（告诉 Transformer 每个 token 在图上的哪个位置）
关节状态 s_t ─ Linear → 1 个 token
隐变量 z     ─ Linear → 1 个 token          （训练时来自 CVAE 编码器，推理时 z = 0）
                         │
         共 1 + 1 + 300 + 300 = 602 个 token
                         │
     Transformer 编码器（4 层，d=512，8 头，FFN 3200）← 所有 token 两两做 self-attention
```

- ResNet18 用 ImageNet 预训练权重初始化，训练时一起微调（backbone 学习率也是 1e-5）。
- 两个相机共用同一个 ResNet。Transformer 怎么知道一个 token 来自哪张图、哪个位置？靠位置编码。
- Self-attention 让"夹爪附近的图像块"和"关节状态"直接互相看到，这是 CNN 单独做不到的。

### 4.3 解码器

```
     100 个 query（输入是全 0，加上 100 个可学习的位置 embedding）
                         │
     Transformer 解码器（1 层）
       self-attention：100 个 query 之间互相看（让相邻时间步协调）
       cross-attention：query 去看主编码器输出的 602 个 token
                         │
     Linear(512 → 6) → 100 × 6 的动作序列 â_{t:t+100}
```

这其实就是 **DETR**：DETR 用 100 个"物体 query"各自去图像特征里找一个物体，ACT 用 100 个"时间步 query"各自去观测特征里找"第 i 步该怎么动"。
和语言模型的解码器不同，这里**没有因果 mask、也不是自回归**：100 步是一次并行输出的。

PS：LeRobot 默认解码器只有 1 层（论文代码里写了 7 层，但由于一个实现细节实际只用到第 1 层，LeRobot 照搬了这个行为）。
总参数 51.6M，其中 ResNet18 约 11M。

### 4.4 训练时一步在干什么

1. 从数据集取一个时刻 $t$：两张图、$s_t$、以及未来 100 步的动作 $a_{t:t+100}$（episode 结尾不够 100 步的部分用 padding，并用 `action_is_pad` 把它们从 loss 里去掉）。
2. CVAE 编码器看 $(s_t, a_{t:t+100})$，输出 $\mu, \sigma$，采样 $z$。
3. 主编码器 + 解码器用 $(图像, s_t, z)$ 预测 $\hat a_{t:t+100}$。
4. loss = L1 + 10 × KL，反向传播，AdamW 更新。

PS：关节和动作在进网络前都按数据集统计量做了 mean/std 归一化，所以 L1 是在归一化空间里算的，6 个关节的量纲一致。

## 5. 推理时怎么执行动作块

默认（`n_action_steps = 100`）：预测 100 步 → 全部执行完 → 再看一次画面，预测下一块。
Limit：每 3.3 秒才"看"一次，这期间玩具被碰歪了它也不知道；而且块和块的交界处可能跳一下。

所以论文里还有一个可选的做法，**temporal ensembling**：**每一步**都重新预测一个 100 步的块，于是同一时刻 $t$ 会有好几个块对它给出预测，
按指数权重 $w_i = e^{-m \cdot i}$ 加权平均后再执行（$i=0$ 是最早的那次预测，$m$ = `temporal_ensemble_coeff`）。
动作更平滑，反应更快，代价是每步都要跑一次网络（ACT 够小，30 Hz 跑得动）。
部署时用 `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` 打开。

## 6. 训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.type` | `act` | |
| `chunk_size` | 100（默认） | 3.3 秒，差不多是一次"抓"或"放"的时长 |
| `kl_weight` | 10（默认） | 论文值 |
| `optimizer_lr` | 1e-5（默认，backbone 也是 1e-5） | 论文值；Transformer + 预训练 backbone，小学习率更稳 |
| `batch_size` | 8 | 8 GB 显存下两路 480×640 图像的稳妥值 |
| `steps` | 50,000 | 约 45 条 × 330 帧 ≈ 15k 帧，约 27 个 epoch |
| `save_freq` | 10,000 | 存 5 个 checkpoint |
| `dataset.eval_split` | 0.1 | 留 10% 的 episode 做**验证集**，每 2000 步算一次验证 loss |

PS：在模仿学习里，验证 loss 低**不等于**真机成功率高，因为误差累积只有在闭环里才会出现。
它只能告诉你有没有严重过拟合，**最终还是看真机成功率**。

## 7. 先跑一个冒烟测试（几分钟）

正式训练要好几个小时。所以先用 200 步确认整条管线（读数据 → 解码视频 → 前向反向 → 存 checkpoint）在 Windows 上没问题，顺便测出速度来估计总时长：

```powershell
.\scripts\train.ps1 act -Steps 200 -Name act_smoke -NoPush --eval_steps=100
```

200 步结束时会存一次 checkpoint，`--eval_steps=100` 让验证也跑两次，这样每个环节都覆盖到了。没问题就可以删掉 `outputs/train/act_smoke`。

PS（Windows）：LeRobot 每次存 checkpoint 都会建一个 `checkpoints/last` 符号链接指向最新的那个，但 Windows 普通用户没有建符号链接的权限
（`WinError 1314`），训练会在**第一次保存时崩掉**。所以 `scripts/train.ps1` 通过 `scripts/lerobot_train_win.py` 启动训练，
把这个链接换成不需要特权的**目录联接（junction）**，其他行为和 `lerobot-train` 完全一样。联接只能指向**绝对路径**，
所以把项目文件夹挪走之后 `last` 会失效，这时直接用带编号的目录（比如 `checkpoints/050000/pretrained_model`）就行。

速度（RTX 3070 Laptop，batch 8）：显存 3.7 GB，稳定后每步 0.28 秒，瓶颈在 GPU 而不是数据加载。混合精度（`--policy.use_amp=true`）没有加速，所以用默认的 fp32。

PS：前 100 步日志里的 `data_s` 会很大（约 1 秒），这是 Windows 上数据加载进程的**启动时间**被摊进了平均值，不是真的慢。看速度要看稳定以后的值。

## 8. 训练前准备（笔记本）

- **插着电源**：用电池时 GPU 会降频，可能慢一半。
- **关掉自动睡眠**：Windows 设置 → 系统 → 电源 → "接通电源时进入睡眠"改成"从不"（屏幕可以关）。睡眠会让训练停住。
- 训练时别跑游戏、剪辑这类占 CPU/GPU 的程序；机械臂可以断电。
- 关掉录数据时留下的 Rerun 窗口，它也占显存。

## 9. 正式训练

```powershell
.\scripts\train.ps1 act
```

完整的等价命令会以青色打印在终端开头。训练产物：

```
outputs/train/act_so101_pick_place/
├── train_config.json                 这次训练的完整配置（可复现）
└── checkpoints/
    ├── 010000/pretrained_model/      第 1 万步的模型
    │   ├── model.safetensors         权重
    │   ├── config.json               策略结构参数
    │   ├── train_config.json
    │   └── policy_preprocessor*.json / policy_postprocessor*.json  归一化统计量等前后处理
    ├── 020000/ ...
    └── last  →  指向最新的 checkpoint（目录联接）
```

训完后模型会自动上传到 `https://huggingface.co/Yufeng1216/act_so101_pick_place`。

中断了可以接着训：

```powershell
.\scripts\train.ps1 act -Resume
```

PS：训练结束时上传失败（比如 HF 登录过期，报 `401`）的话，模型还完整地存在本地，重新登录后手动上传就行：

```powershell
hf upload Yufeng1216/act_so101_pick_place outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model .
```

所以长时间训练之前，先运行 `hf auth whoami` 确认一下登录状态。

### 训练结果

50,000 步用时 **4 小时 01 分**（3.7 步/秒）。验证集 loss（每 2000 步一次）：

| 步数 | 2k | 4k | 6k | 8k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|---|
| `eval_loss` | 0.406 | 0.405 | 0.396 | 0.379 | 0.385 | 0.387 | 0.380 | 0.387 | 0.381 |

- 验证 loss **在 8k 步（约 4 个 epoch）之后就基本不降了**，一直在 0.38 附近小幅波动。这和"模仿学习一般 5–10 个 epoch 就收敛"的经验一致。
- 它也**没有上升**，所以继续训练没有造成明显的过拟合。
- 但验证 loss 不降**不代表**后面的训练没用：训练 loss 还在降，动作可能变得更精细，而这只有在真机闭环里才看得出来。

### 看训练曲线

训练用 Trackio 记 loss，在另一个终端里运行：

```powershell
conda activate lerobot
trackio show --project so101
```

浏览器里会打开一个面板，重点看：

- `train/loss`：总损失 = `l1_loss` + 10 × `kld_loss`，应该稳定下降
- `train/l1_loss`：动作重建误差（归一化空间里），主要看它
- `train/kld_loss`：应该很小但不为 0。一直是 0 的话，说明 CVAE 的隐变量没被用上（posterior collapse）
- `eval/loss`：验证 loss。训练 loss 一直降、验证 loss 却开始升，就是过拟合了

## 10. 在真机上运行

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30
```

PS：`-Policy` 也可以填 Hub 上的模型名 `Yufeng1216/act_so101_pick_place`。但在 Windows 上，LeRobot 可能把名字里的 `/` 转成 `\`，导致找不到模型
（见 code_walkthrough 第 8 节），所以建议填本地路径。

安全注意：

- 第一次运行时手放在从臂电源旁边，策略可能做出意料之外的动作。
- 策略一启动，从臂会从当前姿态移到策略输出的第一个动作，所以先把从臂摆在**起始姿态**。
- 场景要和录数据时一样：相机位置、快递盒、标记点、光照都不要动。

试试 temporal ensembling：

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
```

## 11. 评估

按 [08_evaluation.zh-CN.md](08_evaluation.zh-CN.md) 的协议测 40 次，记录成功率，结果填进 README 的结果表。
