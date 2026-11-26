# SmolVLA real-robot evaluation (40 trials, 2026-11-26)

**English** · [简体中文](smolvla.zh-CN.md)

- Model: `outputs/train/smolvla_so101_pick_place/pretrained_model` (fine-tuned from `lerobot/smolvla_base`, 20k steps × batch 32)
- Inference: 10 flow-matching steps, about 0.38 s per 50-step chunk on the laptop GPU; cameras renamed top → camera1, wrist → camera2 (`--rename_map`, added by `robot.ps1`)
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md): night, desk lamp only, curtains closed; 5 spots × 8 rounds, at most 30 s per trial, all 40 in one go
- Recording: local dataset `Yufeng1216/rollout_eval_smolvla` (under `data/`, 40 episodes)
- Result: ✅ success / 🟡 partial (grasped, not placed) / ❌ failure (not grasped)
- "Time" = recorded frames / 30. The control loop runs at about 25 Hz, so a trial that runs the full 30 s shows about 24 s

| # | Spot | Result | Time (s) | Note |
|---|---|---|---|---|
| 1 | ① | ✅ | 9.8 | block 1 |
| 2 | ② | ❌ | 24.6 | block 1 |
| 3 | ③ | ✅ | 10.4 | block 1 |
| 4 | ④ | ❌ | 24.5 | block 1 |
| 5 | ⑤ | ✅ | 11.8 | block 1 |
| 6 | ① | ❌ | 24.2 | block 2 |
| 7 | ② | ✅ | 14.1 | block 2 |
| 8 | ③ | ✅ | 10.5 | block 2 |
| 9 | ④ | ✅ | 11.7 | block 2 |
| 10 | ⑤ | ❌ | 24.5 | block 2 |
| 11 | ① | ❌ | 24.2 | block 3 |
| 12 | ② | ❌ | 24.6 | block 3 |
| 13 | ③ | ✅ | 11.1 | block 3 |
| 14 | ④ | ✅ | 11.9 | block 3 |
| 15 | ⑤ | ✅ | 10.0 | block 3 |
| 16 | ① | ❌ | 24.2 | block 4 |
| 17 | ② | ❌ | 24.5 | block 4 |
| 18 | ③ | ✅ | 11.6 | block 4 |
| 19 | ④ | ✅ | 10.8 | block 4 |
| 20 | ⑤ | ✅ | 13.1 | block 4 |
| 21 | ① | ❌ | 24.2 | block 5 |
| 22 | ② | ❌ | 24.6 | block 5 |
| 23 | ③ | ❌ | 23.4 | block 5 |
| 24 | ④ | ✅ | 10.3 | block 5 |
| 25 | ⑤ | ✅ | 10.1 | block 5 |
| 26 | ① | ❌ | 24.1 | block 6 |
| 27 | ② | ✅ | 10.2 | block 6 |
| 28 | ③ | ✅ | 11.8 | block 6 |
| 29 | ④ | ✅ | 10.9 | block 6 |
| 30 | ⑤ | ✅ | 13.4 | block 6 |
| 31 | ① | ❌ | 24.0 | block 7 |
| 32 | ② | ❌ | 24.5 | block 7 |
| 33 | ③ | ✅ | 14.3 | block 7 |
| 34 | ④ | ✅ | 11.7 | block 7 |
| 35 | ⑤ | ✅ | 11.1 | block 7 |
| 36 | ① | ✅ | 10.0 | block 8 |
| 37 | ② | ✅ | 13.6 | block 8 |
| 38 | ③ | ✅ | 10.9 | block 8 |
| 39 | ④ | ✅ | 10.0 | block 8 |
| 40 | ⑤ | ✅ | 24.1 | block 8 |

## Summary (40 trials)

| Metric | Value |
|---|---|
| **Success rate** | **26/40 = 65%** (95% CI 50–78%) |
| Partial (grasped, not placed) | 0/40 |
| Failure (not grasped) | 14/40 |
| Success per spot | ① 2/8 · ② 3/8 · ③ 7/8 · ④ 7/8 · ⑤ 7/8 |
| First 20 / last 20 | 12/20 · 14/20 |

- 11 of the 14 failures are at spots ①②; at ③–⑤ it succeeds 21 of 24 times.
- No partials: whenever the toy was grasped, it ended up in the box.
- vs. ACT (14/40): Fisher's exact test p = 0.013. vs. Diffusion Policy (25/40): p = 1.0, no difference.
