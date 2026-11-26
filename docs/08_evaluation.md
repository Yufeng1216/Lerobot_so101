# 08 · 评估协议

四个策略用**同一套协议**在真机上评估，结果才能横向比较。这一页在训练第一个策略之前就定下来，
避免"看到结果再调整标准"。

## 场景

与录数据时完全相同：相机位置、快递盒位置（胶带标记）、5 个玩具起始点、光照、任务描述。

**光照固定为晚上、只开台灯、拉上窗帘**——示范数据是在这种条件下录的（晚上 22 点，暖色灯光）。
白天的自然光偏蓝：下午录的评估画面红/蓝比约 1.30，示范数据是 1.58。颜色变了就是分布偏移，
而且白天光线随时间变化，不同时间测的策略没法公平比较。
台灯的**角度**也要和录示范时一样：灯头转了，阴影会落到玩具另一侧，颜色数值却看不出来。

**机械臂底座要固定在桌上**（C 型夹或双面胶）。底座一动，机械臂相对相机和桌面的坐标系就变了：
学习型策略看到的是没见过的画面，零样本基线（目标在底座坐标系里算）会抓偏。

## 流程

- 5 个起始点 × 每点 8 次 = **每个策略 40 次试验**（20 次的置信区间太宽，见下面"判定标准"）
- 每次试验开始前：从臂在起始姿态，玩具放在标记点上，盒子是空的
- 每测完一组，看一眼底座有没有移位
- 每次最长 **30 秒**
- 顺序：①②③④⑤ 为一组（`-Episodes 5`），**一个策略 8 组连续测完，再换下一个**
- 换策略之间关掉扭矩、**冷却 20 分钟**，让舵机温度回到和开始时一样；冷却后重新检查灯光、相机和底座

运行（每次试验都录成一条 episode，留作展示视频）：

```powershell
.\scripts\robot.ps1 eval -Policy <checkpoint 目录或 Hub 模型> -Name <策略名>
```

例如测 ACT，每次一组 5 次，运行 8 次；ACT 测完、冷却后再测 Diffusion：

```powershell
.\scripts\robot.ps1 eval -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Name act -Episodes 5
.\scripts\robot.ps1 eval -Policy outputs/train/diffusion_so101_pick_place/pretrained_model_ema -Name diffusion -Episodes 5 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
# 同名的评估数据集已存在时会自动接着录，不会覆盖
```

- 背后是 `lerobot-rollout --strategy.type=episodic`：策略控制从臂做一次试验，然后进入复位阶段，
  这时用**主臂**把从臂带回起始姿态、把玩具放到下一个点。
- 键盘：→ 提前结束本次试验或复位；← 作废本次、重做；Esc 停止。
- 录下的数据集叫 `Yufeng1216/rollout_eval_<策略名>`，存在 `data/` 里，不上传。
  LeRobot 规定 rollout 录的数据集名字必须以 `rollout_` 开头。
- 主臂和从臂都要通电（12V 电源），否则会报 `Missing motor IDs`。

## 判定标准

| 结果 | 定义 |
|---|---|
| ✅ 成功 | 30 秒内玩具完全进入盒子并松开 |
| 🟡 部分成功 | 抓起来了，但没放进盒子（掉在外面、时间到了还拿着） |
| ❌ 失败 | 没抓起来 |

主指标是**成功率**（成功 / 40）。部分成功单独统计，用来分析失败原因（抓取问题还是放置问题）。

**次数少时误差很大**，报告时一起给出 95% 置信区间（Wilson 区间）。比较两个策略用 Fisher 精确检验：

```powershell
python -c "from scipy.stats import fisher_exact; print(fisher_exact([[4,6],[9,1]])[1])"   # [[成功,失败],[成功,失败]]
```

为什么要 40 次：如果只测 20 次，10/20 和 15/20 的区间是 30%～70% 和 53%～89%，p ≈ 0.19，分不出高下。
例如目前每个策略有效的只有 10 次：4/10 和 9/10，p ≈ 0.06，仍只算趋势。
每个点位 8 次时单点结论也弱（2/8 的区间是 7%～59%）。

各策略训练步数不同、batch 不同，**步数不能横向比较**；比的是"各自按标准配方训练后的真机效果"，
同时列出看过的样本总数和训练时长，方便对照预算。

## 记录表

每个策略填一张（复制到 `results/<策略名>.md`）：

| # | 点位 | 结果 | 用时 (s) | 备注（失败原因） |
|---|---|---|---|---|
| 1 | ① | | | |
| 2 | ② | | | |
| … | | | | |
| 40 | ⑤ | | | |

汇总：成功率、各点位成功率、平均用时、主要失败模式。

## 评估完：出 GIF 和图

```powershell
# 挑一次成功的试验做 GIF（--episode 从 0 开始，记录表第 5 次就是 4）
python scripts/make_gif.py --repo-id Yufeng1216/rollout_eval_act --episode 4 --out assets/act_success.gif
# 从 results/*.md 重新画各点位成功率
python scripts/plot_results.py
```

两个脚本的代码讲解见 [code_walkthrough.md](code_walkthrough.md#11-scriptsmake_gifpy)。
