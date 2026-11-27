# 08 · Evaluation protocol

**English** · [简体中文](08_evaluation.zh-CN.md)

Every policy is evaluated on the real robot with **the same protocol**, otherwise results can't be compared. The criteria are fixed in advance, so nobody "adjusts the standard after seeing the results".

## 1. Scene

Exactly as during recording: camera positions, box position (taped), the 5 toy start spots, lighting, task description.

**Lighting: night, desk lamp only, curtains closed.** The demos were recorded like this (warm desk lamp). Daylight is bluer, and a colour change is a distribution shift;
daylight also keeps changing, so policies tested at different times can't be compared fairly. The lamp's **angle** must match the demos too: turn the lamp head and the shadow falls on the other side of the toy.

**The arm's base is fixed to the table** (C-clamp or double-sided tape). If the base moves, the arm's frame relative to the camera and table changes:
learned policies see a view they never saw, and the zero-shot baseline (whose targets are computed in the base frame) grasps beside the toy.

## 2. Procedure

- 5 start spots × 8 each = **40 trials per policy** (why 40: see §3)
- before every trial: follower in the start pose, toy on its mark, box empty
- at most **30 s** per trial
- order: ①②③④⑤ is one block (`-Episodes 5`); **a policy runs all 8 blocks in a row, then the next policy**
- check the base before each block
- between policies, turn torque off and **cool down for 20 minutes** so the servos are back at the starting temperature; then re-check lighting, cameras and base

Run it (every trial is recorded as an episode, kept as video):

```powershell
.\scripts\robot.ps1 eval -Policy <checkpoint dir or Hub model> -Name <policy name> -Episodes 5
```

| Policy | Command |
|---|---|
| ACT | `.\scripts\robot.ps1 eval -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Name act -Episodes 5` |
| Diffusion | `.\scripts\robot.ps1 eval -Policy outputs/train/diffusion_so101_pick_place/pretrained_model_ema -Name diffusion -Episodes 5 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10` |
| SmolVLA | `.\scripts\robot.ps1 eval -Policy outputs/train/smolvla_so101_pick_place/pretrained_model -Name smolvla -Episodes 5` |
| π₀.₅ | `.\scripts\robot.ps1 eval -Policy outputs/train/pi05_so101_pick_place/pretrained_model -Name pi05 -Remote -Episodes 5` (server on a cloud GPU, [docs/07](07_pi05.md) §8) |
| VLM agent | `python -m zero_shot.agent --trials 5 --name vlm_agent` (its own script, [docs/09](09_zero_shot_vlm.md) §4) |

PS: if the evaluation dataset of that name already exists, recording continues in it instead of overwriting. So running the same command 8 times gives 40 trials.

- Under the hood it's `lerobot-rollout --strategy.type=episodic`: the policy drives the follower for one trial, then a reset phase follows in which you bring the follower back to the start pose with the **leader** and put the toy on the next spot.
- Keys: → end this trial or the reset early; ← discard this trial and redo it; Esc stop.
- The recording is the dataset `Yufeng1216/rollout_eval_<policy name>` in `data/`; after the evaluation it was uploaded to the Hub (links in `results/`). LeRobot requires rollout datasets to start with `rollout_`.
- Both arms need power (12 V supply), otherwise you get `Missing motor IDs`.

## 3. Criteria and statistics

| Result | Definition |
|---|---|
| ✅ success | within 30 s the toy is fully inside the box and released |
| 🟡 partial | grasped but not placed (dropped outside, or still holding it when time runs out) |
| ❌ failure | not grasped |

The main metric is the **success rate** (successes / 40). Partials are counted separately to see whether failures come from grasping or placing.

**Few trials mean large uncertainty**, so every rate is reported with a 95% confidence interval (Wilson). Two policies are compared with Fisher's exact test:

```powershell
python -c "from scipy.stats import fisher_exact; print(fisher_exact([[4,6],[9,1]])[1])"   # [[success,failure],[success,failure]]
```

Why 40? Example: with 20 trials, 10/20 vs 15/20 gives intervals of 30%–70% and 53%–89%, p ≈ 0.19: no winner.
With 10 it is weaker still: 4/10 vs 9/10 gives p ≈ 0.06, only a trend. PS: per-spot conclusions from 8 trials are weak too (2/8 is 7%–59%), so the overall rate is what counts.

PS: training steps and batch sizes differ between policies, so **steps can't be compared across rows**. What we compare is "real-robot performance when each is trained with its standard recipe", listing samples seen and training time so the budgets can be compared.

## 4. Sheet

One per policy (`results/<policy name>.md`):

| # | Spot | Result | Time (s) | Note (failure reason) |
|---|---|---|---|---|
| 1 | ① | | | |
| 2 | ② | | | |
| … | | | | |
| 40 | ⑤ | | | |

Summary: success rate, success per spot, mean time, main failure modes.

## 5. After the evaluation: GIF and chart

```powershell
# a GIF of one successful trial (--episode starts at 0, so trial 5 on the sheet is 4)
python scripts/make_gif.py --repo-id Yufeng1216/rollout_eval_act --episode 4 --out assets/act_success.gif
# per-spot success chart from results/*.md
python scripts/plot_results.py
```

Both scripts are explained in [code_walkthrough.md](code_walkthrough.md) §11–12.
