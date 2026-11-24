# SO-101 Imitation Learning: ACT → Diffusion Policy → SmolVLA → π₀.₅

Reproducing four modern robot-learning policies on a low-cost **SO-101** arm with
[🤗 LeRobot](https://github.com/huggingface/lerobot), end to end on real hardware:
teleoperated data collection → training → deployment and evaluation on the real robot —
and comparing them against a **zero-shot baseline** that uses no demonstrations at all
(an off-the-shelf vision-language model for perception + inverse kinematics for motion).

**Task:** pick up a plush toy from one of 5 marked spots and place it in a cardboard box.
**Dataset:** 50 teleoperated demonstrations, [`Yufeng1216/so101_pick_place`](https://huggingface.co/datasets/Yufeng1216/so101_pick_place).

> 中文教程见 [`docs/`](docs/)（所有教程用中文写成，命令可直接复制运行）。

| Policy | Paper | Params | Type | Trained on |
|---|---|---|---|---|
| **ACT** | [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) | ~52 M | CVAE + Transformer, action chunking | local RTX 3070 (8 GB) |
| **Diffusion Policy** | [Chi et al., 2023](https://arxiv.org/abs/2303.04137) | ~278 M | DDPM/DDIM denoising over action chunks | university GPU cluster (Slurm) |
| **SmolVLA** | [Shukor et al., 2025](https://arxiv.org/abs/2506.01844) | ~450 M | small VLM + flow-matching action expert | local / cloud GPU |
| **π₀.₅** | [Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054) | ~3 B | VLA, PaliGemma + flow-matching expert | cloud GPU (≥ 24 GB) |
| **Zero-shot VLM + IK** (baseline) | — | ~2 B (VLM) | Qwen3-VL points at toy/box → table homography → IK → scripted grasp | no training |

## Results

_Filled in as each stage is completed._ Every policy is evaluated on the same task with
the same protocol (see [docs/08_evaluation.md](docs/08_evaluation.md)).

| Policy | Train steps | Train time | Success rate (n=20) | Model | Rollout video |
|---|---|---|---|---|---|
| ACT | 50k (≈27 epochs) | 4.0 h (RTX 3070 Laptop) | **50%** (10/20; [details](results/act.md)) | [act_so101_pick_place](https://huggingface.co/Yufeng1216/act_so101_pick_place) | – |
| Diffusion Policy | 100k (≈53 epochs) | ≈4 h (Tesla V100 16 GB) | – | [diffusion_so101_pick_place](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place) | – |
| SmolVLA | – | – | – | – | – |
| π₀.₅ | – | – | – | – | – |
| Zero-shot VLM + IK | 0 (no demos) | – | – | – | – |

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
cloud/           runs elsewhere: Slurm GPU-cluster scripts for models that need more than 8 GB
docs/            step-by-step tutorial, one file per stage
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
| 7 | π₀.₅ | docs/07_pi05.md |
| 8 | Evaluation protocol and comparison | [docs/08_evaluation.md](docs/08_evaluation.md) |
| 9 | Zero-shot baseline: VLM + IK, no demonstrations | [docs/09_zero_shot_vlm.md](docs/09_zero_shot_vlm.md) |
| – | Code walkthrough (every script explained) | [docs/code_walkthrough.md](docs/code_walkthrough.md) |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) and the
[SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) open-source arm design.
