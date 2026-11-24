# 05 · Diffusion Policy：用扩散模型生成动作

论文：Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023
（[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)）。

如果你学过图像扩散模型（DDPM、Stable Diffusion），这一篇几乎没有新概念：
**把"生成一张图片"换成"生成一段动作序列"**，条件从"文字"换成"当前的图像和关节状态"。

## 1. 为什么要用扩散模型生成动作

回顾 [04_act.md](04_act.md) 的第二个问题：**示范不一致**。同一个画面下，人有时从左边绕、有时从右边绕。
这时正确的动作分布是**多峰的**（multimodal）：

```
     概率
      │   ▲               ▲
      │  ╱ ╲             ╱ ╲
      │ ╱   ╲           ╱   ╲
      └────────────────────────── 动作
        从左绕   (平均值)   从右绕
                    ↑ MSE 回归会输出这里：直直撞上去
```

- **MSE 回归**只能输出平均值，正好落在两个峰之间，是一个谁都没做过的动作。
- **ACT** 用 CVAE 的隐变量 z 来区分不同的峰，但推理时固定 z = 0，表达能力有限。
- **扩散模型**直接学习整个分布 $p(a \mid o)$，推理时从噪声出发**采样**，每次落在某一个峰上。
  它本来就是为了生成多峰的高维数据（图像）而设计的。

## 2. 训练和推理

记 $A^0$ 为真实动作序列（horizon = 64 步 × 6 维），$O$ 为观测条件（最近 2 帧的图像特征 + 关节状态）。

**训练**（和 DDPM 完全一样）：

1. 随机选一个噪声步数 $k \in \{1, \dots, 100\}$，采样噪声 $\epsilon \sim \mathcal N(0, I)$；
2. 加噪：$A^k = \sqrt{\bar\alpha_k}\,A^0 + \sqrt{1-\bar\alpha_k}\,\epsilon$；
3. 网络预测噪声，损失 $\mathcal L = \lVert \epsilon - \epsilon_\theta(A^k, k, O) \rVert^2$。

**推理**：从纯噪声 $A^{100} \sim \mathcal N(0, I)$ 开始，用 $\epsilon_\theta$ 一步步去噪，得到 $A^0$。
默认去噪 100 步，所以**推理比 ACT 慢得多**（ACT 只做一次前向）。这是扩散策略的主要代价。

`prediction_type = epsilon`（预测噪声）、`beta_schedule = squaredcos_cap_v2`（余弦噪声表）
都是 LeRobot 默认，也是论文的设置。

## 3. 三个时间长度：n_obs_steps、horizon、n_action_steps

这是 Diffusion Policy 最容易混淆的地方。默认值是 2 / 64 / 32：

```
时间 →        t-1   t   t+1 ...                                   t+62
观测          [■■■■■■]                                               n_obs_steps = 2：看最近 2 帧
预测动作      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]  horizon = 64：一次生成 64 步
实际执行            [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                n_action_steps = 32：只执行前 32 步
                                                    ↑ 然后重新观测、重新生成
```

- **看 2 帧**：能从两帧的差异中读出速度信息，单帧做不到。
- **生成 64 步，只执行 32 步**：后半段只是为了让前半段"知道之后要去哪"，保证连贯；
  执行一半就重新规划，能及时修正。这就是论文说的 **receding horizon**（滚动时域），
  和 ACT 的 action chunking 是同一个思想。
- 30 Hz 下 32 步 ≈ 1 秒重新规划一次，比 ACT 默认的 3.3 秒频繁。

## 4. 网络结构

```
 top 图像   ─ ResNet18 ─ Spatial Softmax（32 个关键点）─┐
 wrist 图像 ─ ResNet18 ─ Spatial Softmax（32 个关键点）─┼─ 拼接 ×2 帧 ─▶ 全局条件向量 O
 关节状态                                              ─┘
                                                          │
 噪声动作 A^k（64×6）──▶  1D 时序 U-Net  ◀──  FiLM 条件  ◀──┤
 噪声步数 k ─ 正弦编码 ─────────────────────────────────────┘
                 │
                 ▼
          预测的噪声 ε（64×6）
```

几个和图像扩散的对应关系：

| 部件 | 图像扩散 | Diffusion Policy |
|---|---|---|
| 去噪网络 | 2D U-Net（在像素网格上卷积） | **1D U-Net**（沿时间轴卷积，通道 = 6 个关节） |
| 条件注入 | cross-attention 注入文字 | **FiLM**：条件向量生成每层的缩放 γ 和偏移 β，$h \leftarrow \gamma \odot h + \beta$ |
| 条件编码器 | CLIP 文本编码器 | 每路相机一个 ResNet18（`use_separate_rgb_encoder_per_camera`） |

**Spatial Softmax** 把 ResNet 的特征图变成 32 个"关键点"的 (x, y) 坐标：对每个通道做 softmax 得到
一张概率图，再求坐标的期望。输出是"东西在哪里"，而不是"东西长什么样"，正适合操作任务。

U-Net 的通道数 `down_dims = (512, 1024, 2048)` 让参数量达到约 2.6 亿，其中大部分在 U-Net 里。

## 5. 本项目的训练配置

| 参数 | 值 | 为什么 |
|---|---|---|
| `policy.type` | `diffusion` | |
| `n_obs_steps / horizon / n_action_steps` | 2 / 64 / 32（默认） | 论文值 |
| `resize_shape` | **[240, 320]** | 把 480×640 缩小一半，像素少 4 倍。8 GB 显存放得下，训练也更快。论文本身用的分辨率更低（96×96 到 240×320） |
| `crop_ratio` | **0.95** | 训练时随机裁出 95% 的区域（推理时裁中心）。这是论文里的**数据增强**，让模型对相机的轻微偏移更鲁棒 |
| `ema.type` | **warmup** | 保存一份权重的**指数滑动平均**（EMA）。扩散模型的标准做法，EMA 权重生成的动作更平滑稳定 |
| `optimizer_lr` | 1e-4，余弦退火，500 步预热（默认） | 论文值 |
| `batch_size` | 8 | 和 ACT 一样（显存实测见下） |
| `steps` | 100,000 | 扩散模型要学整个分布，比回归难，通常需要更多步；约 53 个 epoch |
| `save_freq` | 20,000 | 5 个 checkpoint |

> **关于 EMA：** 开启后，checkpoint 里除了 `pretrained_model/`，还会多一个 `pretrained_model_ema/`。
> 上传到 Hub 的是 EMA 权重，部署时也用 EMA 那份。

## 6. 冒烟测试

```powershell
.\scripts\train.ps1 diffusion -Steps 200 -Name diffusion_smoke -NoPush --eval_steps=100 --trackio.enable=false
```

实测（RTX 3070 Laptop，batch 8）：

| 配置 | 参数量 | 显存 | 每步 |
|---|---|---|---|
| 默认 U-Net `down_dims=(512,1024,2048)` | 278M | **> 8 GB，溢出** | **约 124 s**（不可用） |
| 本项目 `down_dims=(256,512,1024)` | 89M | 2.9 GB | **0.13 s** |

> **踩坑：显存溢出不会报错，只会变得极慢。** 默认配置下，光是 fp32 的权重、梯度、Adam 的两份状态
> 和 EMA 副本就约 278M × 4 字节 × 5 ≈ 5.5 GB，再加上激活值就超过了 8 GB。
> Windows 的显卡驱动不会报 `CUDA out of memory`，而是悄悄把放不下的部分挪到**系统内存**（"共享 GPU 内存"），
> 每一步都要通过 PCIe 来回搬数据，速度慢了约 1000 倍。
> 判断方法：`nvidia-smi` 显示显存几乎占满，同时 `updt_s` 异常大。
>
> 解决：把 U-Net 通道数减半，参数量变成原来的约 1/3。
> 这组通道数论文里也用过；50 条示范的数据量下，89M 参数（仍是 ACT 的 1.7 倍）已经足够。
> 图像编码器（ResNet18 + Spatial Softmax）没有改动。
> 如果真机效果不理想，可以在云端用默认大小再训一次做对照（模型容量消融）。

所以第 5 节的配置表里还要加一行：

| 参数 | 值 | 为什么 |
|---|---|---|
| `down_dims` | **[256, 512, 1024]** | 默认值在 8 GB 显存下溢出，见上 |

100,000 步 × 0.13 s ≈ **3.6 小时**（加上验证约 4 小时）。

## 6.5 对照实验：在 Slurm cluster 集群上训练默认大小（278M）

减半是被显存逼的，不能想当然地认为"小模型就够了"。过拟合可以靠正则化控制：
本配置里已经有随机裁剪、EMA、weight decay 和验证集监控。所以我们在 university 的 **Slurm cluster**
集群上用**默认的 U-Net**（278M）再训一次，其余设置和笔记本上完全一样。两次的结果放在一起，
就是一个**模型容量消融实验**：

| 训练 | `down_dims` | 参数量 | 硬件 |
|---|---|---|---|
| 笔记本 | 256 / 512 / 1024 | 89M | RTX 3070 Laptop 8 GB |
| 集群 | 512 / 1024 / 2048（默认） | 278M | V100 16 GB 或 A40 48 GB |

### 集群是什么，和笔记本有什么不同

- **登录节点**（login node）：SSH 登录后所在的机器。只用来编辑文件、装环境、提交任务，**没有 GPU，不能在上面训练**。
- **Slurm**：集群的排队系统。你把"要几块 GPU、多少内存、跑多久、运行什么命令"写进一个 `.sbatch` 文件，
  用 `sbatch` 提交，Slurm 分到资源后在**计算节点**上运行它。你可以断开 SSH，任务照样跑。
- `.sbatch` 文件就是一个 bash 脚本。开头的 `#SBATCH ...` 行看起来是注释，但 Slurm 会把它们读成资源申请。
- 集群是 Linux，所以不需要 Windows 的目录联接补丁，直接用 `lerobot-train`。

### 步骤（登录和两步验证只能你自己做）

```bash
# 1) 在笔记本上，把集群脚本传上去（-xfer 是专门用来传文件的节点）
scp -r cluster <uniqname>@cluster-host:~/Lerobot/

# 2) 登录（需要密码 + Duo 两步验证）
ssh <uniqname>@cluster-host
cd ~/Lerobot

# 3) 一次性装环境，约 15 分钟
bash cluster/setup_env.sh

# 4) 查看自己有哪些计费账号，把名字填进 sbatch 文件的 --account=
my_accounts
nano cluster/train_diffusion_full.sbatch

# 5) 提交
mkdir -p logs                        # Slurm 不会自动创建日志目录
sbatch cluster/train_diffusion_full.sbatch
squeue -u $USER                      # PD = 排队中，R = 运行中
tail -f logs/diffusion_full_*.log    # 看训练日志，Ctrl+C 退出查看（不会停止训练）
```

- 任务最长 12 小时。超时被停掉的话，**原样再提交一次**，脚本会发现已有 checkpoint，自动接着训。
- 训练完把模型拷回笔记本（在笔记本上运行）：

```bash
scp -r <uniqname>@cluster-host:~/Lerobot/outputs/train/diffusion_full_so101_pick_place/checkpoints/last/pretrained_model_ema outputs/train/diffusion_full_so101_pick_place/
```

## 7. 正式训练

开始之前先确认 HF 登录没过期（ACT 就在这里吃过亏）：

```powershell
hf auth whoami
.\scripts\train.ps1 diffusion
```

训练曲线同样在 Trackio 里看（`trackio show --project so101`）。Diffusion 的 `train/loss` 是噪声预测的 MSE，
它的数值和 ACT 的 loss **不能直接比较**，只看它自己的趋势。

## 8. 在真机上运行

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/diffusion_so101_pick_place/checkpoints/last/pretrained_model_ema -Duration 30
```

推理时每次生成要去噪 100 步。如果从臂动作一顿一顿的（生成跟不上 30 Hz），可以换成更快的 DDIM 采样器，
只去噪 10 步：

```powershell
... --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
```

## 9. 和 ACT 对比时要看什么

- **成功率**：同样按 [08_evaluation.md](08_evaluation.md) 跑 20 次。
- **动作是否果断**：在毛绒玩具"抓左边还是抓右边都可以"的位置，ACT 可能犹豫，扩散策略应该更果断。
- **推理延迟**：每次生成要多久，是否影响动作的连贯性。
