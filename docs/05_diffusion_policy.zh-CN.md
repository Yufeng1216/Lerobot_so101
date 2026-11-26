# 05 · Diffusion Policy：用扩散模型生成动作

[English](05_diffusion_policy.md) · **简体中文**

论文：Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023
（[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)）。

一句话：**把图像扩散模型里的"生成一张图"换成"生成一段动作"**，条件从"一句文字"换成"当前的图像和关节状态"。
下面先把扩散模型本身从头讲一遍，再看它怎么用在动作上。

## 1. 为什么要用扩散模型生成动作

回到 [04_act.zh-CN.md](04_act.zh-CN.md) 的第二个问题：**示范不一致**。同一个画面下，人有时从左边绕、有时从右边绕，
所以正确的动作分布是**多峰的**（multimodal）：

```
     概率
      │   ▲               ▲
      │  ╱ ╲             ╱ ╲
      │ ╱   ╲           ╱   ╲
      └────────────────────────── 动作
        从左绕   (平均值)   从右绕
                    ↑ MSE 回归会输出这里：直直撞上去
```

- **MSE 回归**只能输出条件均值，正好落在两个峰中间，是一个谁都没做过的动作。
- **ACT** 用 CVAE 的隐变量 $z$ 区分不同的峰，但推理时固定 $z = 0$，等于只取了一个"标准风格"，表达能力有限。
- **扩散模型**直接学整个分布 $p(A \mid O)$，推理时从噪声出发**采样**，每次落在某一个峰上。它本来就是为生成多峰的高维数据（图像）设计的。

所以问题变成：怎么学一个高维的条件分布，并且能从里面采样？

## 2. 扩散模型（DDPM）是怎么工作的

### 2.1 出发点：把"生成"拆成很多次"去噪"

直接从 $\mathcal N(0,I)$ 一步映射到一段合理的动作很难（这就是 GAN、VAE 在做的事）。
DDPM 的想法是：既然一步太难，就拆成 $K$ 小步，每一步只去掉一点点噪声。每一小步都简单，网络就学得会。

### 2.2 前向过程：一点点加噪声（固定的，不用学）

记真实的动作序列为 $A^0$（64 步 × 6 维）。每一步加一点高斯噪声：

$$q(A^k \mid A^{k-1}) = \mathcal N\big(\sqrt{1-\beta_k}\,A^{k-1},\ \beta_k I\big),\qquad k = 1,\dots,K$$

$\beta_k$ 是事先定好的很小的数（噪声表，这里是余弦表 `squaredcos_cap_v2`，$K=100$）。
因为高斯加高斯还是高斯，可以一步跳到任意第 $k$ 步，记 $\bar\alpha_k = \prod_{i\le k}(1-\beta_i)$：

$$A^k = \sqrt{\bar\alpha_k}\,A^0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\qquad \epsilon \sim \mathcal N(0, I)$$

$k$ 越大，$\bar\alpha_k$ 越接近 0，$A^k$ 就越接近纯噪声。$k = K$ 时基本就是 $\mathcal N(0,I)$。

### 2.3 反向过程：学一个去噪网络

生成就是把前向过程倒过来：从 $A^K \sim \mathcal N(0,I)$ 出发，一步步去噪回 $A^0$。
反向的每一步也近似是高斯，它的均值只取决于"这一步里的噪声是多少"。所以只要训一个网络 $\epsilon_\theta(A^k, k, O)$ 去**预测加进去的噪声**：

$$\mathcal L = \mathbb E_{A^0,\,k,\,\epsilon}\Big[\big\| \epsilon - \epsilon_\theta(A^k, k, O) \big\|^2\Big]$$

于是训练一步就是：

1. 从数据里取一个样本 $(O, A^0)$，随机选一个 $k \in \{1,\dots,100\}$，采样 $\epsilon$；
2. 用上面的闭式公式加噪得到 $A^k$；
3. 网络看 $(A^k, k, O)$，预测 $\epsilon$，算 MSE。

PS：损失还是 MSE，但这次 MSE 不会把两个峰平均掉。因为网络预测的是**某一个具体噪声样本**，而不是动作本身；
从不同的初始噪声出发，反向过程会走到不同的峰上。

推理时（DDPM 采样），每一步用预测的噪声算出上一步的均值，再加一点新的随机噪声：

$$A^{k-1} = \frac{1}{\sqrt{1-\beta_k}}\Big(A^k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_\theta(A^k,k,O)\Big) + \sigma_k z,\quad z\sim\mathcal N(0,I)$$

走完 100 步得到 $A^0$。Limit：要跑 100 次网络，**比 ACT（一次前向）慢得多**，这是扩散策略最主要的代价。

### 2.4 DDIM：同一个网络，少走几步

DDIM（Song et al., 2021）发现：训练目标完全不用改，只要把采样换成一个**确定性**的更新（不再加随机噪声 $z$），
就可以**跳着走**，比如只走 100 → 90 → … → 0 这 10 步。每一步先用 $\epsilon_\theta$ 估计出干净的 $\hat A^0$，再直接算出目标步的 $A$：

$$\hat A^0 = \frac{A^k - \sqrt{1-\bar\alpha_k}\,\epsilon_\theta}{\sqrt{\bar\alpha_k}},\qquad A^{k'} = \sqrt{\bar\alpha_{k'}}\,\hat A^0 + \sqrt{1-\bar\alpha_{k'}}\,\epsilon_\theta$$

所以 DDIM 是推理时的加速，**同一个训练好的模型**，换一个参数就能用。真机上我们就用 DDIM 10 步（第 8 节）。

## 3. 用在动作上：三个时间长度

条件 $O$ 和要生成的 $A$ 分别覆盖多长时间？这是 Diffusion Policy 最容易混的地方。默认值是 2 / 64 / 32：

```
时间 →        t-1   t   t+1 ...                                   t+62
观测          [■■■■■■]                                               n_obs_steps = 2：看最近 2 帧
预测动作      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]  horizon = 64：一次生成 64 步
实际执行            [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                n_action_steps = 32：只执行前 32 步
                                                    ↑ 然后重新观测、重新生成
```

- **看 2 帧**：两帧的差能读出速度，单帧做不到。
- **生成 64 步，只执行 32 步**：后半段只是为了让前半段"知道之后要去哪"，保证连贯；执行一半就重新规划，能及时修正。
  这就是论文说的 **receding horizon**（滚动时域），和 ACT 的 action chunking 是同一个思想。
- 30 Hz 下 32 步 ≈ 1 秒重新规划一次，比 ACT 默认的 3.3 秒勤快得多。

PS：动作和状态用 min/max 归一化到 [−1, 1]（不是 ACT 的 mean/std），这样和 $\mathcal N(0,I)$ 的尺度对得上；采样时还会把结果裁到 [−1, 1] 里（`clip_sample`）。

## 4. 网络结构

整体是"条件编码器 + 去噪网络"两块：

```
 top 图像   (3×240×320) ─ ResNet18 ─ Spatial Softmax（32 个关键点）─ Linear ─ 64 维 ─┐
 wrist 图像 (3×240×320) ─ ResNet18 ─ Spatial Softmax（32 个关键点）─ Linear ─ 64 维 ─┼─ 拼接 = 134 维/帧
 关节状态 (6)                                                                    ─┘     × 2 帧 = 268 维
                                                                                          │
 噪声步数 k ─ 正弦编码 ─ MLP ─ 128 维 ───────────────────────────────────────── 拼接 ─▶ 条件向量 c（396 维）
                                                                                          │
 噪声动作 A^k（64×6）──▶  1D 时序 U-Net（每个残差块都用 c 做 FiLM）  ◀──────────────────────┘
                 │
                 ▼
          预测的噪声 ε̂（64×6）
```

### 4.1 条件编码器：ResNet18 + Spatial Softmax

每个相机一个独立的 ResNet18（`use_separate_rgb_encoder_per_camera`），ImageNet 预训练初始化。
和 ACT 不同的是，这里不把特征图展平成 token，而是接一个 **Spatial Softmax**：

1. 用 1×1 卷积把特征图变成 32 个通道，每个通道对应一个"关键点"；
2. 对每个通道在空间上做 softmax，得到一张概率图 $p_c(u, v)$；
3. 求坐标的期望 $(\mathbb E[u], \mathbb E[v])$，就得到这个关键点的位置。

于是每张图变成 32 个 (x, y) 坐标 = 64 个数。它输出的是"东西在**哪里**"，而不是"东西**长什么样**"，正好适合操作任务。
PS：这个结构来自 Levine 等人的 visuomotor 策略，相当于让网络自己学 32 个"值得盯着看的点"。

### 4.2 去噪网络：1D 时序 U-Net

U-Net 本来是图像分割和图像扩散里的结构：编码器一路下采样，解码器一路上采样，同层之间有 skip connection。
这里把 2D 卷积换成**沿时间轴的 1D 卷积**，6 个关节就是 6 个输入通道：

```
A^k (6 通道 × 64 步)
  ├─ 残差块 ×2（512 通道）──────────────────────────┐ skip
  ├─ 下采样（步长 2）→ 32 步                        │
  ├─ 残差块 ×2（1024 通道）────────────────┐ skip   │
  ├─ 下采样 → 16 步                        │        │
  ├─ 残差块 ×2（2048 通道）                │        │
  ├─ 中间 残差块 ×2                        │        │
  ├─ 上采样 → 32 步，拼接 skip ◀───────────┘        │
  ├─ 上采样 → 64 步，拼接 skip ◀────────────────────┘
  └─ 1D 卷积 → 6 通道 × 64 步 = ε̂
```

每个残差块是 Conv1d(核 5) → GroupNorm → Mish，两层，再加残差连接。
沿时间轴卷积的好处是天然地让相邻时间步的动作平滑，下采样之后又能看到整段 64 步的大趋势。

### 4.3 条件怎么注入：FiLM

条件向量 $c$ 要告诉 U-Net "现在看到了什么、噪声有多大"。Diffusion Policy 用的是 **FiLM**（Feature-wise Linear Modulation）：
每个残差块里，用一个线性层从 $c$ 算出每个通道的缩放 $\gamma$ 和偏移 $\beta$，然后

$$h \leftarrow \gamma(c) \odot h + \beta(c)$$

也就是说，观测不是作为输入拼进去的，而是去"调节"每一层特征的增益和偏置。

| 部件 | 图像扩散（如 Stable Diffusion） | Diffusion Policy |
|---|---|---|
| 生成的对象 | 一张图（2D 像素网格） | 一段动作（64 步 × 6 维） |
| 去噪网络 | 2D U-Net | **1D U-Net**（沿时间轴卷积） |
| 条件 | 文字，经 CLIP 编码 | 图像 + 关节状态，经 ResNet18 + Spatial Softmax 编码 |
| 条件注入 | cross-attention | **FiLM** |

默认通道数 `down_dims = (512, 1024, 2048)` 时总参数约 2.78 亿，绝大部分在 U-Net 里。

## 5. 训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.type` | `diffusion` | |
| `n_obs_steps / horizon / n_action_steps` | 2 / 64 / 32（默认） | 论文值 |
| `resize_shape` | **[240, 320]** | 480×640 缩小一半，像素少 4 倍，训练快很多。论文本身用的分辨率更低（96×96 到 240×320） |
| `crop_ratio` | **0.95** | 训练时随机裁出 95% 的区域（推理时裁中心）。论文里的**数据增强**，让模型对相机的轻微偏移更鲁棒 |
| `ema.type` | **warmup** | 另存一份权重的**指数滑动平均**（EMA）。扩散模型的标准做法，EMA 权重生成的动作更平滑 |
| `optimizer_lr` | 1e-4，余弦退火，500 步预热（默认） | 论文值 |
| `down_dims` | 512 / 1024 / 2048（默认） | 278M 参数，要 16 GB 以上显存（见第 6 节） |
| `batch_size` | 8 | 和 ACT 一样 |
| `steps` | 100,000 | 扩散模型要学整个分布，比回归难，一般要更多步；约 53 个 epoch |
| `save_freq` | 20,000 | 5 个 checkpoint |

PS：开了 EMA 之后，checkpoint 里除了 `pretrained_model/` 还会多一个 `pretrained_model_ema/`。上传到 Hub 的、部署时用的都是 EMA 那份。

## 6. 显存：为什么不在笔记本上训

默认大小的 U-Net 光是 fp32 的权重、梯度、Adam 的两个状态和 EMA 副本就要约 5.5 GB，再加上激活值，8 GB 的笔记本显卡放不下。

PS：显存不够时，Windows 不一定报 `CUDA out of memory`。驱动会把放不下的部分挪到系统内存里，训练不报错，但慢上百倍
（实测每步约 2 分钟）。所以 `nvidia-smi` 显示显存几乎占满、日志里 `updt_s` 异常大，就是这个原因。

所以默认大小放到**更大显存的 GPU**（16 GB 的 V100，Slurm 集群）上训。只有 8 GB 显卡又想在本地训的话，
`train.ps1` 的 diffusion 预设把 U-Net 通道数减半成 (256, 512, 1024)（89M 参数，2.9 GB，每步 0.13 s，论文里也用过这组），其他设置不变。

## 7. 正式训练（Linux GPU 集群）

用 `cloud/slurm/` 里的脚本，参数和第 5 节一样。完整步骤见 [cloud/README.zh-CN.md](../cloud/README.zh-CN.md)，核心三步：

```bash
bash cloud/slurm/setup_env.sh          # 一次性装环境
bash cloud/slurm/submit.sh diffusion   # 提交训练任务
tail -f logs/diffusion_<任务编号>.log   # 看训练日志
```

Tesla V100 16 GB，batch 8：

| `step_s`（每步） | `data_s`（等数据） | 显存 | 10 万步总时长 |
|---|---|---|---|
| 0.127 s | 0.003 s | 6.5 GB | 约 4 小时（含验证） |

- 集群任务有时长上限。被停掉之后再运行一次 `submit.sh diffusion`，会从上一个 checkpoint 接着训。
- Diffusion 的 `train/loss` 是噪声预测的 MSE，和 ACT 的 loss **不能直接比**，只看它自己的趋势。

训完只把 EMA 权重拷回笔记本（约 1 GB，不拷优化器状态），在笔记本上运行：

```powershell
scp -r <用户名>@<集群传输节点>:~/Lerobot/outputs/train/diffusion_so101_pick_place/checkpoints/last/pretrained_model_ema outputs/train/diffusion_so101_pick_place/
```

训好的模型：[Yufeng1216/diffusion_so101_pick_place](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place)。

## 8. 在真机上运行：用 DDIM

先离线算一下速度：拿数据集里的画面喂给模型（RTX 3070 Laptop）：

| 采样器 | 每次生成 | 和示范动作的平均绝对误差（8 帧 × 6 个关节） |
|---|---|---|
| DDPM 100 步（训练时的默认） | 约 1.5 s | 3.2 |
| DDIM 10 步 | 约 0.16 s | 3.3 |

每次生成的 32 步动作（`n_action_steps`）在 30 Hz 下只够执行约 1.07 s。DDPM 100 步生成比执行还慢，从臂每执行完一段就得停下来等；
DDIM 10 步快 10 倍，误差几乎不变。所以**真机上用 DDIM**：

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/diffusion_so101_pick_place/pretrained_model_ema -Duration 30 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
```

PS：单帧误差在 0.4 到 7 之间，误差最大的是示范刚开始、手臂从静止起步的时候（肘关节差十几到二十多）。
起步时机本来就不确定，所以这里偏差大不一定是错。这只是离线对比，真正的指标是真机成功率。

## 9. 和 ACT 比的时候看什么

- **成功率**：同样按 [08_evaluation.zh-CN.md](08_evaluation.zh-CN.md) 测 40 次。
- **动作果不果断**：在"从左抓、从右抓都行"的位置，ACT 可能犹豫，扩散策略应该更果断。
- **推理延迟**：每次生成要多久，会不会影响动作的连贯性。
