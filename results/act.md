# ACT real-robot evaluation (40 trials: 1–10 on 2026-11-24, 11–40 on 2026-11-26)

**English** · [简体中文](act.zh-CN.md)

- Model: `outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model` (50k steps, [Hub](https://huggingface.co/Yufeng1216/act_so101_pick_place))
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md): night, desk lamp only, curtains closed (as when the demos were recorded); 5 spots × 8 rounds, at most 30 s per trial. Trials 1–10 alternated with Diffusion every 5 trials; trials 11–40 run in one go
- Recording: local dataset `Yufeng1216/rollout_eval_act` (under `data/`, 40 episodes)
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
| 11 | ① | ❌ | 28.7 | block 3 |
| 12 | ② | ❌ | 29.4 | block 3 |
| 13 | ③ | ✅ | 27.6 | block 3 |
| 14 | ④ | ❌ | 29.5 | block 3 |
| 15 | ⑤ | ✅ | 11.8 | block 3 |
| 16 | ① | ✅ | 11.8 | block 4 |
| 17 | ② | ❌ | 29.5 | block 4 |
| 18 | ③ | ❌ | 29.6 | block 4 |
| 19 | ④ | 🟡 | 29.6 | block 4 |
| 20 | ⑤ | 🟡 | 29.4 | block 4 |
| 21 | ① | ❌ | 29.3 | block 5 |
| 22 | ② | ❌ | 29.6 | block 5 |
| 23 | ③ | ❌ | 29.6 | block 5 |
| 24 | ④ | ✅ | 12.6 | block 5 |
| 25 | ⑤ | 🟡 | 29.5 | block 5 |
| 26 | ① | ❌ | 29.3 | block 6 |
| 27 | ② | ❌ | 29.6 | block 6 |
| 28 | ③ | ❌ | 29.4 | block 6 |
| 29 | ④ | ✅ | 10.8 | block 6 |
| 30 | ⑤ | ✅ | 13.3 | block 6 |
| 31 | ① | ❌ | 29.0 | block 7 |
| 32 | ② | ❌ | 29.6 | block 7 |
| 33 | ③ | ❌ | 29.5 | block 7 |
| 34 | ④ | ✅ | 11.3 | block 7 |
| 35 | ⑤ | 🟡 | 29.5 | block 7 |
| 36 | ① | ❌ | 29.2 | block 8 |
| 37 | ② | ❌ | 29.5 | block 8 |
| 38 | ③ | ✅ | 18.4 | block 8 |
| 39 | ④ | ✅ | 10.0 | block 8 |
| 40 | ⑤ | ✅ | 11.4 | block 8 |

## Summary (40 trials)

| Metric | Value |
|---|---|
| **Success rate** | **14/40 = 35%** (95% CI 22–51%) |
| Partial (grasped, not placed) | 4/40 |
| Failure (not grasped) | 22/40 |
| Success per spot | ① 1/8 · ② 0/8 · ③ 2/8 · ④ 6/8 · ⑤ 5/8 |
| First 10 / last 30 | 4/10 · 10/30 |

- 21 of the 22 failures (not grasped) are at spots ①–③; at ④⑤ the toy is grasped in 15 of 16 trials.
- All 4 partials (grasped, not placed) are at ④⑤.
