# Diffusion Policy real-robot evaluation (40 trials: 1–10 on 2026-11-24, 11–40 on 2026-11-26)

**English** · [简体中文](diffusion.zh-CN.md)

- Model: `outputs/train/diffusion_so101_pick_place/pretrained_model_ema` (278M, 100k steps, EMA weights, [Hub](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place))
- Sampling: 10-step DDIM (`--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10`), about 200 ms per inference
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md): night, desk lamp only, curtains closed (as when the demos were recorded); 5 spots × 8 rounds, at most 30 s per trial. Trials 1–10 alternated with ACT every 5 trials; trials 11–40 run in one go
- Recording: local dataset `Yufeng1216/rollout_eval_diffusion` (under `data/`, 40 episodes)
- Result: ✅ success / 🟡 partial (grasped, not placed) / ❌ failure (not grasped)
- "Time" = recorded frames / 30. The control loop actually runs at about 25 Hz (it pauses briefly while each action segment is generated), so real time ≈ the number × 1.17

| # | Spot | Result | Time (s) | Note |
|---|---|---|---|---|
| 1 | ① | ✅ | 10.1 | block 1 |
| 2 | ② | ✅ | 11.1 | block 1 |
| 3 | ③ | ✅ | 12.7 | block 1 |
| 4 | ④ | ✅ | 26.2 | block 1 |
| 5 | ⑤ | ✅ | 14.3 | block 1 |
| 6 | ① | ✅ | 12.3 | block 2 |
| 7 | ② | ✅ | 26.1 | block 2 |
| 8 | ③ | ✅ | 13.7 | block 2 |
| 9 | ④ | ✅ | 19.5 | block 2 |
| 10 | ⑤ | ❌ | 26.1 | block 2 |
| 11 | ① | ✅ | 11.1 | block 3 |
| 12 | ② | ✅ | 12.0 | block 3 |
| 13 | ③ | 🟡 | 25.3 | block 3 |
| 14 | ④ | ✅ | 16.1 | block 3 |
| 15 | ⑤ | ❌ | 25.3 | block 3 |
| 16 | ① | ✅ | 13.3 | block 4 |
| 17 | ② | 🟡 | 25.3 | block 4 |
| 18 | ③ | 🟡 | 25.4 | block 4 |
| 19 | ④ | ✅ | 21.7 | block 4 |
| 20 | ⑤ | ❌ | 25.3 | block 4 |
| 21 | ① | 🟡 | 25.8 | block 5 |
| 22 | ② | 🟡 | 26.1 | block 5 |
| 23 | ③ | 🟡 | 26.2 | block 5 |
| 24 | ④ | ❌ | 26.1 | block 5 |
| 25 | ⑤ | ❌ | 26.1 | block 5 |
| 26 | ① | ❌ | 25.6 | block 6 |
| 27 | ② | ✅ | 12.0 | block 6 |
| 28 | ③ | ✅ | 11.4 | block 6 |
| 29 | ④ | ✅ | 15.2 | block 6 |
| 30 | ⑤ | ❌ | 26.1 | block 6 |
| 31 | ① | ✅ | 11.2 | block 7 |
| 32 | ② | ✅ | 13.8 | block 7 |
| 33 | ③ | 🟡 | 26.1 | block 7 |
| 34 | ④ | ✅ | 13.0 | block 7 |
| 35 | ⑤ | ✅ | 12.1 | block 7 |
| 36 | ① | ✅ | 8.8 | block 8 |
| 37 | ② | ✅ | 12.7 | block 8 |
| 38 | ③ | ✅ | 10.1 | block 8 |
| 39 | ④ | ✅ | 11.8 | block 8 |
| 40 | ⑤ | ❌ | 25.9 | block 8 |

## Summary (40 trials)

| Metric | Value |
|---|---|
| **Success rate** | **25/40 = 62.5%** (95% CI 47–76%) |
| Partial (grasped, not placed) | 7/40 |
| Failure (not grasped) | 8/40 |
| Success per spot | ① 6/8 · ② 6/8 · ③ 4/8 · ④ 7/8 · ⑤ 2/8 |
| First 10 / last 30 | 9/10 · 16/30 (Fisher p = 0.06) |

- 6 of the 8 failures (not grasped) are at spot ⑤.
- All 7 partials (grasped, not placed) are at ①–③, 4 of them at ③.
- vs. ACT (14/40): Fisher's exact test p = 0.025 on success; on "grasped at all" (32/40 vs 18/40) p = 0.002.
