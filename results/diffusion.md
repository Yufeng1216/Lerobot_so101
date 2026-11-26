# Diffusion Policy real-robot evaluation (first 10 trials, 2026-11-24; the rest to follow)

**English** · [简体中文](diffusion.zh-CN.md)

- Model: `outputs/train/diffusion_so101_pick_place/pretrained_model_ema` (278M, 100k steps, EMA weights, [Hub](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place))
- Sampling: 10-step DDIM (`--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10`), about 200 ms per inference
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md): night, desk lamp only, curtains closed (as when the demos were recorded); 5 spots × 8 rounds, at most 30 s per trial. Trials 1–10 alternated with ACT every 5 trials; trials 11–40 run in one go
- Recording: local dataset `Yufeng1216/rollout_eval_diffusion` (under `data/`, 10 episodes)
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

## Summary (first 10 trials)

| Metric | Value |
|---|---|
| **Success rate** | **9/10 = 90%** (95% CI 60–98%) |
| Partial | 0/10 |
| Failure (not grasped) | 1/10 |
| Success per spot | ① 2/2 · ② 2/2 · ③ 2/2 · ④ 2/2 · ⑤ 1/2 |
