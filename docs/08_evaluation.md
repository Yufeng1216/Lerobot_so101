# 08 · 评估协议

四个策略用**同一套协议**在真机上评估，结果才能横向比较。这一页在训练第一个策略之前就定下来，
避免"看到结果再调整标准"。

## 场景

与录数据时完全相同：相机位置、快递盒位置（胶带标记）、5 个玩具起始点、光照、任务描述。

## 流程

- 5 个起始点 × 每点 4 次 = **20 次试验**
- 每次试验开始前：从臂在起始姿态，玩具放在标记点上，盒子是空的
- 每次最长 **30 秒**
- 顺序：①②③④⑤ 循环 4 轮（避免一个点连续测导致的偏差）

运行（每次试验都录成一条 episode，留作展示视频）：

```powershell
.\scripts\robot.ps1 eval -Policy <checkpoint 目录或 Hub 模型> -Name <策略名>
```

例如 ACT：

```powershell
.\scripts\robot.ps1 eval -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Name act
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

主指标是**成功率**（成功 / 20）。部分成功单独统计，用来分析失败原因（抓取问题还是放置问题）。

## 记录表

每个策略填一张（复制到 `results/<策略名>.md`）：

| # | 点位 | 结果 | 用时 (s) | 备注（失败原因） |
|---|---|---|---|---|
| 1 | ① | | | |
| 2 | ② | | | |
| … | | | | |
| 20 | ⑤ | | | |

汇总：成功率、各点位成功率、平均用时、主要失败模式。

## 评估完：出 GIF 和图

```powershell
# 挑一次成功的试验做 GIF（--episode 从 0 开始，记录表第 19 次就是 18）
python scripts/make_gif.py --repo-id Yufeng1216/rollout_eval_act --episode 18 --out assets/act_success.gif
# 从 results/*.md 重新画各点位成功率
python scripts/plot_results.py
```

两个脚本的代码讲解见 [code_walkthrough.md](code_walkthrough.md#12-scriptsmake_gifpy)。
