# VLM agent (Claude Haiku 5.5) real-robot evaluation (40 trials, 2026-11-27)

**English** · [简体中文](vlm_agent.zh-CN.md)

- Agent: `zero_shot/agent.py` at tag `vlm-agent-frozen` (later commits changed only logging, the reset countdown and network retries); `claude-haiku-5-5`, adaptive thinking at effort `medium`; no training, nothing taken from the demos ([docs/09](../docs/09_zero_shot_vlm.md))
- Protocol: [docs/08_evaluation.md](../docs/08_evaluation.md) (night, desk lamp only; 5 spots × 8 rounds), except the limit: **25 model calls** per trial instead of 30 s, because every action is one API call
- Scene: camera and lighting as for the other policies (< 2 px shift); the blue-and-white object on the floor at the bottom edge of the overhead view (also there for π₀.₅, not for ACT, Diffusion or SmolVLA) was there for these trials too
- Logs: every step's images and a `record.json` per trial (each tool call, the model's own verdict, time, tokens, cost) in [`Yufeng1216/vlm_agent_eval_logs`](https://huggingface.co/datasets/Yufeng1216/vlm_agent_eval_logs)
- Result: ✅ success / 🟡 partial (grasped, not placed) / ❌ failure (not grasped)
- "Time" = wall-clock seconds of the attempt (API calls + arm motion); "Calls" = model calls

| # | Spot | Result | Time (s) | Calls | Note |
|---|---|---|---|---|---|
| 1 | ① | ✅ | 171 | 16 | block 1 |
| 2 | ② | ✅ | 148 | 12 | block 1 |
| 3 | ③ | ✅ | 103 | 9 | block 1 |
| 4 | ④ | ✅ | 177 | 17 | block 1 |
| 5 | ⑤ | ✅ | 86 | 10 | block 1 |
| 6 | ① | ✅ | 45 | 6 | block 2 |
| 7 | ② | ✅ | 86 | 10 | block 2 |
| 8 | ③ | ✅ | 82 | 10 | block 2 |
| 9 | ④ | ✅ | 218 | 22 | block 2 |
| 10 | ⑤ | 🟡 | 252 | 25 | block 2; hit the 25-call limit holding the toy |
| 11 | ① | ✅ | 59 | 8 | block 3 |
| 12 | ② | ✅ | 106 | 13 | block 3 |
| 13 | ③ | ✅ | 88 | 9 | block 3 |
| 14 | ④ | ❌ | 112 | 14 | block 3 |
| 15 | ⑤ | ✅ | 167 | 17 | block 3 |
| 16 | ① | ✅ | 57 | 7 | block 4 |
| 17 | ② | ✅ | 63 | 7 | block 4 |
| 18 | ③ | ✅ | 49 | 6 | block 4 |
| 19 | ④ | ❌ | 156 | 18 | block 4 |
| 20 | ⑤ | ✅ | 178 | 20 | block 4 |
| 21 | ① | ✅ | 92 | 12 | block 5 |
| 22 | ② | ❌ | 87 | 10 | block 5 |
| 23 | ③ | ✅ | 79 | 10 | block 5 |
| 24 | ④ | ❌ | 218 | 22 | block 5 |
| 25 | ⑤ | ✅ | 81 | 9 | block 5 |
| 26 | ① | ❌ | 96 | 10 | block 6 |
| 27 | ② | ✅ | 138 | 15 | block 6 |
| 28 | ③ | ✅ | 86 | 9 | block 6 |
| 29 | ④ | 🟡 | 144 | 17 | block 6; the model reported success; the toy was not in the box |
| 30 | ⑤ | ✅ | 96 | 11 | block 6 |
| 31 | ① | ✅ | 62 | 8 | block 7 |
| 32 | ② | ✅ | 94 | 11 | block 7 |
| 33 | ③ | ✅ | 97 | 11 | block 7 |
| 34 | ④ | ✅ | 138 | 15 | block 7; rerun: the first try broke off on a dropped API connection (not counted) |
| 35 | ⑤ | ❌ | 174 | 20 | block 7 |
| 36 | ① | ✅ | 62 | 8 | block 8 |
| 37 | ② | ❌ | 156 | 17 | block 8 |
| 38 | ③ | ❌ | 231 | 25 | block 8; hit the 25-call limit |
| 39 | ④ | ✅ | 118 | 13 | block 8 |
| 40 | ⑤ | 🟡 | 174 | 19 | block 8; the model reported failure: grasped, not placed |

## Summary (40 trials)

| Metric | Value |
|---|---|
| **Success rate** | **29/40 = 72.5%** (95% CI 57–84%) |
| Partial (grasped, not placed) | 3/40 |
| Failure (not grasped) | 8/40 |
| Success per spot | ① 7/8 · ② 6/8 · ③ 7/8 · ④ 4/8 · ⑤ 5/8 |
| First 20 / last 20 | 17/20 · 12/20 |
| Time per trial | median 100 s (45–252 s); successes 92 s, others 156 s |
| Model calls per trial | median 11.5 (successes 10) |
| API cost | $1.24 in total, $0.03 per trial |

- Weakest at ④ (4/8): the spot nearest the box, where the toy sits close to the wall and the fingers stop short or push it away.
- The model's own verdict matched the outcome in 37 of the 38 trials where it called `done`; once (trial 29) it reported success with the toy outside the box. Two trials ran out of calls.
- vs. ACT (14/40): Fisher's exact test p = 0.002. vs. Diffusion Policy (25/40): p = 0.47; vs. SmolVLA (26/40): p = 0.63; vs. π₀.₅ (34/40): p = 0.27.
