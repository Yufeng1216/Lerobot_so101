# SO-101 Imitation Learning: ACT → Diffusion Policy → SmolVLA → π₀.₅

Reproducing four modern robot-learning policies on a low-cost **SO-101** arm with
[🤗 LeRobot](https://github.com/huggingface/lerobot), end to end on real hardware:
teleoperated data collection → training → deployment and evaluation on the real robot —
and comparing them against a **zero-shot baseline** that uses no demonstrations at all
(an off-the-shelf vision-language model for perception + inverse kinematics for motion).

**Task:** pick up a plush toy from one of 5 marked spots and place it in a cardboard box.
**Dataset:** 50 teleoperated demonstrations, [`Yufeng1216/so101_pick_place`](https://huggingface.co/datasets/Yufeng1216/so101_pick_place).

| ACT (spot ⑤) | Diffusion Policy (spot ②) |
|---|---|
| ![ACT rollout](assets/act_success.gif) | ![Diffusion Policy rollout](assets/diffusion_success.gif) |

<sub>Autonomous rollouts recorded during evaluation, 2× speed. Left of each clip: overhead camera; right: wrist camera.</sub>

> 中文教程见 [`docs/`](docs/)（所有教程用中文写成，命令可直接复制运行）。

| Policy | Paper | Params | Type | Trained on |
|---|---|---|---|---|
| **ACT** | [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) | ~52 M | CVAE + Transformer, action chunking | local RTX 3070 (8 GB) |
| **Diffusion Policy** | [Chi et al., 2023](https://arxiv.org/abs/2303.04137) | ~278 M | DDPM/DDIM denoising over action chunks | university GPU cluster (Slurm) |
| **SmolVLA** | [Shukor et al., 2025](https://arxiv.org/abs/2506.01844) | ~450 M | small VLM + flow-matching action expert | university GPU cluster (Slurm) |
| **π₀.₅** | [Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054) | ~4.1 B | VLA, PaliGemma + flow-matching expert | rented A100 80 GB (Hugging Face Jobs) |
| **Zero-shot VLM + IK** (baseline) | — | ~2 B (VLM) | Qwen3-VL points at toy/box → table homography → IK → scripted grasp | no training |

## Results

Every policy is evaluated on the real robot with the same protocol
([docs/08_evaluation.md](docs/08_evaluation.md)): **40 trials** (5 start spots × 8), 30 s each, under the same
lighting the demonstrations were recorded in (night, desk lamp only). Policies take turns in blocks of 5 trials, so
slow drifts (lighting, toy wear, motor temperature) hit every policy equally.
Success = toy released inside the box. Per-trial sheets are in [`results/`](results/).

> **Status:** only the first 10 trials per policy are valid so far. Around trial 20 of the session the arm's base
> was knocked out of place (found afterwards from the recorded camera frames, see below); the remaining 30 trials
> per policy are being re-run with the base clamped to the table. Numbers below are provisional.

| Policy | Steps × batch | Samples seen (epochs) | Train time | Success (valid trials) | 95% CI | Grasped, not placed | Not grasped | Inference / chunk | Model | Clip |
|---|---|---|---|---|---|---|---|---|---|---|
| ACT | 50k × 8 | 400k (≈27) | 4.0 h, RTX 3070 Laptop | **4/10** ([sheet](results/act.md)) | 17–69% | 0 | 6 | ~47 ms | [🤗](https://huggingface.co/Yufeng1216/act_so101_pick_place) | [gif](assets/act_success.gif) |
| Diffusion Policy | 100k × 8 | 800k (≈53) | ≈4 h, Tesla V100 16 GB | **9/10** ([sheet](results/diffusion.md)) | 60–98% | 0 | 1 | ~200 ms (DDIM, 10 steps) | [🤗](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place) | [gif](assets/diffusion_success.gif) |
| SmolVLA | 20k × 32 | 640k (≈43) | training | – | – | – | – | – | – | – |
| π₀.₅ | 3k × 32 | 96k (≈6) | 7.1 h, A100 80 GB (≈ $18) | – | – | – | – | – | – | – |
| Zero-shot VLM + IK | 0 (no demos) | 0 | – | – | – | – | – | – | – | – |

<sub>Epochs over the 15k-frame training split (90% of 16,717 frames). 95% CI: Wilson score interval.</sub>

![Per-spot success rate](assets/results.svg)

### How to read this comparison

- **Each policy uses its own standard recipe** (from the paper / LeRobot defaults), on the same 50
  demonstrations and the same evaluation protocol. This answers *"which method works better when deployed
  as intended on this task?"*, not *"which architecture is better at equal compute"*.
- **Training steps are not comparable across rows.** Batch sizes differ (8 vs 32), model sizes differ by ~70×
  (52 M to 4.1 B), and SmolVLA / π₀.₅ start from pretrained weights while ACT and Diffusion start from scratch.
  *Samples seen* and *train time* are listed so the budgets can be compared directly.
- **Why 40 trials.** With 20 trials, 50% vs 75% would still have overlapping 95% intervals (30–70% vs 53–89%,
  Fisher's exact test p ≈ 0.19). With the 10 valid trials so far, 4/10 vs 9/10 gives p ≈ 0.06: a trend only.

**Observations so far**

- **Diffusion Policy is ahead of ACT** (9/10 vs 4/10 in the valid trials). Every failure so far is a missed grasp;
  once the toy is in the gripper, both policies reach the box.
- **ACT fails at spots ①–③ and succeeds at ④–⑤** in every valid trial (0/6 vs 4/4); Diffusion misses only one
  trial, at ⑤. With 2 trials per spot this is a lead, not a result.
- **The setup itself has to be checked during the evaluation, not only before it.** The base of the arm moved by
  ~0.5–1 cm partway through the session and nobody noticed at the time. It showed up afterwards: with the camera
  fixed, the arm's image position at its rest pose stays within ~2 px across all 50 demos, but jumped by 6–13 px
  from one block onward. All later trials were discarded. The same shift also made the zero-shot baseline
  grasp beside the toy, since its targets are computed in the base frame.
- **Sampler matters for Diffusion.** The default 100-step DDPM takes ~1.5 s per action chunk on the laptop GPU,
  longer than the 1.07 s the chunk lasts, so the arm would stall; 10-step DDIM is ~10× faster with the same
  offline action error ([docs/05](docs/05_diffusion_policy.md#8-在真机上运行)).

## Hardware

- SO-101 follower arm (6× Feetech STS3215) + SO-101 leader arm for teleoperation
- Two USB bus-servo controller boards (WCH CH343 and CH340 USB-serial chips)
- Two USB cameras (640×480 @ 30 fps): `top` (overhead, on a desk-lamp arm) and `wrist` (on the follower's gripper)
- Laptop: Windows 11, RTX 3070 Laptop GPU (8 GB)

## Software

- Windows 11, native (no WSL/VM: serial ports and cameras work directly, and CUDA is available)
- Python 3.12, conda env `lerobot`
- LeRobot **0.7.0** (pinned), PyTorch with CUDA 12.8

## Repository layout

```
config/          robot.example.json  → copy to robot.json (your COM ports, camera indices, HF user)
scripts/         runs on the Windows laptop: hardware checks, recording, local training, deployment
cloud/           runs elsewhere: Slurm cluster and rented Hugging Face GPUs, for models that need more than 8 GB
zero_shot/       the no-demonstration baseline: SO-101 kinematics, pixel→table homography, local VLM pointing, real-robot grasp script
docs/            step-by-step tutorial, one file per stage
results/         per-trial real-robot evaluation sheets
assets/          README media (rollout GIFs, results chart)
```

## Tutorial

New to the tooling? Start with [docs/how_this_repo_works.md](docs/how_this_repo_works.md) (what every file and tool is for).

| # | Stage | Doc |
|---|---|---|
| 0 | Environment setup on Windows | [docs/00_environment.md](docs/00_environment.md) |
| 1 | Hardware: ports, motor IDs, cameras | [docs/01_hardware.md](docs/01_hardware.md) |
| 2 | Calibration and teleoperation | [docs/02_calibration_teleop.md](docs/02_calibration_teleop.md) |
| 3 | Recording a dataset | [docs/03_data_collection.md](docs/03_data_collection.md) |
| 4 | ACT | [docs/04_act.md](docs/04_act.md) |
| 5 | Diffusion Policy | [docs/05_diffusion_policy.md](docs/05_diffusion_policy.md) |
| 6 | SmolVLA | [docs/06_smolvla.md](docs/06_smolvla.md) |
| 7 | π₀.₅ | [docs/07_pi05.md](docs/07_pi05.md) |
| 8 | Evaluation protocol and comparison | [docs/08_evaluation.md](docs/08_evaluation.md) |
| 9 | Zero-shot baseline: VLM + IK, no demonstrations | [docs/09_zero_shot_vlm.md](docs/09_zero_shot_vlm.md) |
| – | Code walkthrough (every script explained) | [docs/code_walkthrough.md](docs/code_walkthrough.md) |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) and the
[SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) open-source arm design.
