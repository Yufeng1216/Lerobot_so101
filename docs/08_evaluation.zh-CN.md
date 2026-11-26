# 08 · 评估协议

[English](08_evaluation.md) · **简体中文**

所有策略都用**同一套协议**在真机上评估，结果才能横向比较。标准事先写死，避免"看到结果再调整标准"。

## 1. 场景

和录数据时完全一样：相机位置、快递盒位置（胶带标记）、5 个玩具起始点、光照、任务描述。

**光照：晚上，只开台灯，拉上窗帘。** 示范是在这种条件下录的（暖色台灯）。白天的自然光偏蓝，颜色一变就是分布偏移；
而且白天光线一直在变，不同时间测的策略没法公平比较。台灯的**角度**也要和录示范时一样：灯头一转，阴影就落到玩具另一侧了。

**机械臂底座固定在桌上**（C 型夹或双面胶）。底座一动，机械臂相对相机和桌面的坐标系就变了：
学习型策略看到的是没见过的画面，零样本基线（目标在底座坐标系里算）会直接抓偏。

## 2. 流程

- 5 个起始点 × 每点 8 次 = **每个策略 40 次试验**（为什么是 40，见第 3 节）
- 每次试验开始前：从臂在起始姿态，玩具放在标记点上，盒子是空的
- 每次最长 **30 秒**
- 顺序：①②③④⑤ 为一组（`-Episodes 5`），**一个策略 8 组连续测完，再换下一个**
- 每组开始前看一眼底座有没有动
- 换策略之间关掉扭矩、**冷却 20 分钟**，让舵机温度回到和开始时一样；冷却后重新检查灯光、相机和底座

运行（每次试验都录成一条 episode，留作展示视频）：

```powershell
.\scripts\robot.ps1 eval -Policy <checkpoint 目录或 Hub 模型> -Name <策略名> -Episodes 5
```

| 策略 | 命令 |
|---|---|
| ACT | `.\scripts\robot.ps1 eval -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Name act -Episodes 5` |
| Diffusion | `.\scripts\robot.ps1 eval -Policy outputs/train/diffusion_so101_pick_place/pretrained_model_ema -Name diffusion -Episodes 5 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10` |
| SmolVLA | `.\scripts\robot.ps1 eval -Policy outputs/train/smolvla_so101_pick_place/pretrained_model -Name smolvla -Episodes 5` |

PS：同名的评估数据集已经存在时会自动接着录，不会覆盖。所以一个策略运行 8 次同一条命令就是 40 次。

- 背后是 `lerobot-rollout --strategy.type=episodic`：策略控制从臂做一次试验，然后进入复位阶段，这时用**主臂**把从臂带回起始姿态、把玩具放到下一个点。
- 按键：→ 提前结束这次试验或复位；← 作废这次、重做；Esc 停止。
- 录下的数据集叫 `Yufeng1216/rollout_eval_<策略名>`，存在 `data/` 里，不上传。LeRobot 规定 rollout 录的数据集名字必须以 `rollout_` 开头。
- 主臂和从臂都要通电（12V 电源），不然会报 `Missing motor IDs`。

## 3. 判定标准和统计

| 结果 | 定义 |
|---|---|
| ✅ 成功 | 30 秒内玩具完全进入盒子并松开 |
| 🟡 部分成功 | 抓起来了，但没放进盒子（掉在外面，或者时间到了还拿着） |
| ❌ 失败 | 没抓起来 |

主指标是**成功率**（成功 / 40）。部分成功单独统计，用来分析失败是出在抓取还是放置。

**次数少的时候误差很大**，所以报告时一起给出 95% 置信区间（Wilson 区间）。比较两个策略用 Fisher 精确检验：

```powershell
python -c "from scipy.stats import fisher_exact; print(fisher_exact([[4,6],[9,1]])[1])"   # [[成功,失败],[成功,失败]]
```

为什么是 40 次？Example：只测 20 次的话，10/20 和 15/20 的区间是 30%–70% 和 53%–89%，p ≈ 0.19，分不出高下。
10 次就更弱了：4/10 和 9/10 的 p ≈ 0.06，也只能算趋势。PS：单个点位 8 次的结论也不强（2/8 的区间是 7%–59%），所以主要看总成功率。

PS：各策略的训练步数、batch 都不同，**步数不能横向比较**。我们比的是"各自按标准配方训练后的真机效果"，同时列出看过的样本总数和训练时长，方便对照预算。

## 4. 记录表

每个策略一张（`results/<策略名>.md`）：

| # | 点位 | 结果 | 用时 (s) | 备注（失败原因） |
|---|---|---|---|---|
| 1 | ① | | | |
| 2 | ② | | | |
| … | | | | |
| 40 | ⑤ | | | |

汇总：成功率、各点位成功率、平均用时、主要失败模式。

## 5. 评估完：出 GIF 和图

```powershell
# 挑一次成功的试验做 GIF（--episode 从 0 开始，记录表第 5 次就是 4）
python scripts/make_gif.py --repo-id Yufeng1216/rollout_eval_act --episode 4 --out assets/act_success.gif
# 从 results/*.md 画各点位成功率
python scripts/plot_results.py
```

两个脚本的代码讲解见 [code_walkthrough.zh-CN.md](code_walkthrough.zh-CN.md) 第 11、12 节。
