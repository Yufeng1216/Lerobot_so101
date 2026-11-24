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

运行（录下评估视频，以后做展示用）：

```powershell
lerobot-rollout --strategy.type=episodic `
  --policy.path=<checkpoint 或 Hub 模型> `
  --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower `
  --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG} }" `
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=so101_leader `
  --dataset.repo_id=Yufeng1216/eval_<策略名> `
  --dataset.num_episodes=20 --dataset.episode_time_s=30 --dataset.reset_time_s=25 `
  --dataset.single_task="Pick up the blue plush toy and place it in the cardboard box."
```

（PowerShell 里一条命令换行用反引号 `` ` ``。）评估数据集名字必须以 `eval_` 开头，这是 LeRobot 的约定。
复位阶段可以用主臂把从臂摆回起始姿态。

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
