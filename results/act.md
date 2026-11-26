# ACT real-robot evaluation (first 10 trials, 2026-11-24; the rest to follow)

**English** · [简体中文](act.zh-CN.md)

- Model: `outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model` (50k steps, [Hub](https://huggingface.co/Yufeng1216/act_so101_pick_place))
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md): night, desk lamp only, curtains closed (as when the demos were recorded); 5 spots × 8 rounds, at most 30 s per trial. Trials 1–10 alternated with Diffusion every 5 trials; trials 11–40 run in one go
- Recording: local dataset `Yufeng1216/rollout_eval_act` (under `data/`, 10 episodes)
- Result: ✅ success / 🟡 partial (grasped, not placed) / ❌ failure (not grasped)
- "Time" is how long the episode was recorded: after a success → ends it early, so for successes it ≈ the time to finish

| # | Spot | Result | Time (s) | Note |
|---|---|---|---|---|
| 1 | ① | ❌ | 28.9 | block 1 |
| 2 | ② | ❌ | 29.6 | block 1 |
| 3 | ③ | ❌ | 29.5 | block 1 |
| 4 | ④ | ✅ | 9.9 | block 1 |
| 5 | ⑤ | ✅ | 14.9 | block 1 |
| 6 | ① | ❌ | 29.2 | block 2 |
| 7 | ② | ❌ | 29.3 | block 2 |
| 8 | ③ | ❌ | 29.6 | block 2 |
| 9 | ④ | ✅ | 12.3 | block 2 |
| 10 | ⑤ | ✅ | 12.9 | block 2 |

## Summary (first 10 trials)

| Metric | Value |
|---|---|
| **Success rate** | **4/10 = 40%** (95% CI 17–69%) |
| Partial | 0/10 |
| Failure (not grasped) | 6/10 |
| Success per spot | ① 0/2 · ② 0/2 · ③ 0/2 · ④ 2/2 · ⑤ 2/2 |
