# 02 · 标定与遥操作

目标：让主臂和从臂"说同一种语言"，然后用主臂实时控制从臂。

本机配置（`config/robot.json`）：**从臂 follower = COM3**（CH340），**主臂 leader = COM4**（CH343）。

## 1. 标定在做什么？

每个舵机内部有一个 12 位磁编码器，读数是 0–4095（一圈）。但同一个物理姿态，
两条臂的原始读数并不一样：舵机装配时的角度不同，主臂的减速比也和从臂不同。

标定就是给每个关节记录两件事：

1. **零点（homing offset）**：把"关节在活动范围中间"定义为中点；
2. **活动范围（min / max）**：你手动把关节转到两头，记下最小、最大读数。

之后 LeRobot 把原始读数**归一化**到 `[-100, 100]`（夹爪是 `[0, 100]`）。
这和你在深度学习里做的输入归一化是一回事：只要主臂和从臂各自归一化过，
"主臂肘关节 30" 就等于 "从臂肘关节 30"，遥操作才能一一对应。
训好的策略也能换到另一台标定过的同型号机械臂上用，因为它学到的是归一化后的角度。

> 标定结果存成 JSON 文件，以 `--robot.id` / `--teleop.id` 命名，存放在仓库的
> `config/calibration/robots/so_follower/so101_follower.json` 和
> `config/calibration/teleoperators/so_leader/so101_leader.json`。
> （LeRobot 默认存在全局缓存 `$HF_HOME/lerobot/calibration` 里；`scripts/env.ps1` 设置了
> `HF_LEROBOT_CALIBRATION`，让它直接读写仓库里的这个文件夹，并且随 git 一起保存。）
> **之后所有命令都要用同一个 id**（本项目固定为 `so101_follower` 和 `so101_leader`），
> LeRobot 是靠 id 找到对应标定文件的。`scripts/robot.ps1` 已经替你保证了这一点。

## 2. 标定从臂（COM3）

```powershell
conda activate lerobot
.\scripts\robot.ps1 calibrate-follower
```

它等价于：

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower
```

终端会打印一张姿态示意图，然后：

1. **摆到中间姿态**：大臂竖直向上，小臂水平朝前，手腕和小臂成一条直线，夹爪闭合
   （就是终端里那张 ASCII 图的样子）。此时舵机没有力矩，可以随便掰。摆好后按 **Enter**。
2. **逐个关节转满全程**：终端会显示一张实时表格，列出每个关节的 MIN / POS / MAX。
   把每个关节（除了 `wrist_roll`）从一头慢慢转到另一头，看着 MIN 和 MAX 被撑开。
   - `wrist_roll`（手腕旋转）可以转整圈，程序直接用全量程，不需要记录。
   - 转到机械限位即可，**不要硬掰**。
3. 所有关节都转过之后按 **Enter**，标定文件就保存了。

> 如果某个关节忘了转，它的 MIN 和 MAX 会相同，程序会报错 `a motor did not move`，重跑一次即可。

## 3. 标定主臂（COM4）

```powershell
.\scripts\robot.ps1 calibrate-leader
```

步骤完全相同。主臂的"夹爪"是橙色手柄上的扳机。

### 标定文件为什么要进 git

**录好的数据集是用这份标定归一化的**：标定一变，旧数据集里的角度就和机械臂对不上了，
训好的策略也会"手眼不协调"。所以标定文件存在仓库的 `config/calibration/` 里，随代码一起版本管理。
**不要随便重新标定**；如果必须重标（比如换了舵机），之前的数据集和模型都要重新评估。

> 注意：直接运行 `lerobot-calibrate` 等命令（不通过 `scripts/robot.ps1`）时，
> 要先运行 `. .\scripts\env.ps1`，否则 LeRobot 会去全局缓存里找标定文件。

本机的标定结果（编码器刻度，0–4095 为一圈）：

| 关节 | 从臂范围 | 主臂范围 |
|---|---|---|
| shoulder_pan | 883 – 3319 | 828 – 3233 |
| shoulder_lift | 865 – 3273 | 819 – 3225 |
| elbow_flex | 820 – 3062 | 874 – 3090 |
| wrist_flex | 809 – 3151 | 914 – 3237 |
| wrist_roll | 0 – 4095（整圈） | 0 – 4095（整圈） |
| gripper | 1777 – 3315 | 2026 – 3275 |

## 4. 遥操作（不带相机）

**安全提示：** 遥操作一开始，从臂会**立刻**移动到主臂当前的姿态。启动前先把主臂摆成和从臂差不多的姿态，
手放在从臂电源附近，出问题直接拔电源。

```powershell
.\scripts\robot.ps1 teleop
```

现在动主臂，从臂应该同步跟随。按 `Ctrl+C` 结束。

检查清单：
- [ ] 6 个关节方向都对（主臂往左，从臂也往左）
- [ ] 夹爪开合正常，能夹住蓝色玩具
- [ ] 没有明显抖动或延迟

## 5. 遥操作（带相机画面）

```powershell
.\scripts\robot.ps1 teleop -Cameras
```

加了 `-Cameras` 后，脚本会追加 `--robot.cameras=...` 和 `--display_data=true`。
LeRobot 会弹出 **Rerun** 可视化窗口，实时显示 top、wrist 两路画面和关节曲线。
**这就是之后策略"看到"的全部输入**，录数据前在这里确认：

- [ ] top 画面能看到整个工作区（玩具、杯子、从臂）
- [ ] wrist 画面能看到夹爪尖和被抓物体
- [ ] 两路画面没有互换

## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `Could not connect on port 'COM3'`，或 Windows 错误 31 "设备没有发挥作用" | CH340 卡住了。拔掉 USB 等 3 秒再插回去 |
| 扫描时舵机全部 MISSING | 驱动板没接电源（USB 只负责通信，不供电） |
| 从臂动作方向和主臂相反 / 幅度不对 | 标定时中间姿态没摆准，重新标定那条臂 |
| 相机打不开 / 画面全黑 | 编号变了。跑 `python scripts/find_hardware.py` 看截图，改 `config/robot.json` |
| `Failed to open OpenCVCamera(2)`，但单独拍照又正常 | 本机遇到过一次。两个原因：① 两个相机同时传未压缩画面，USB 带宽不够，所以配置里给相机加了 `"fourcc": "MJPG"`（压缩传输）；② 腕部相机的线随机械臂运动被扯松。线要留出余量并固定在臂上，必要时直接插笔记本的 USB 口 |
