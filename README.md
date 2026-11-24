# SO-101 Imitation Learning: ACT → Diffusion Policy → SmolVLA → π₀.₅

Reproducing four modern robot-learning policies on a low-cost **SO-101** arm with
[🤗 LeRobot](https://github.com/huggingface/lerobot), end to end on real hardware:
teleoperated data collection → training → deployment and evaluation on the real robot.

> 中文教程见 [`docs/`](docs/)（所有教程用中文写成，命令可直接复制运行）。

| Policy | Paper | Params | Type | Trained on |
|---|---|---|---|---|
| **ACT** | [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) | ~52 M | CVAE + Transformer, action chunking | local RTX 3070 (8 GB) |
| **Diffusion Policy** | [Chi et al., 2023](https://arxiv.org/abs/2303.04137) | ~263 M | DDPM/DDIM denoising over action chunks | local RTX 3070 (8 GB) |
| **SmolVLA** | [Shukor et al., 2025](https://arxiv.org/abs/2506.01844) | ~450 M | small VLM + flow-matching action expert | local / cloud GPU |
| **π₀.₅** | [Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054) | ~3 B | VLA, PaliGemma + flow-matching expert | cloud GPU (≥ 24 GB) |

## Results

_Filled in as each stage is completed._ Every policy is evaluated on the same task with
the same protocol (see [docs/08_evaluation.md](docs/08_evaluation.md)).

| Policy | Train steps | Train time | Success rate (n=20) | Model | Rollout video |
|---|---|---|---|---|---|
| ACT | – | – | – | – | – |
| Diffusion Policy | – | – | – | – | – |
| SmolVLA | – | – | – | – | – |
| π₀.₅ | – | – | – | – | – |

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
scripts/         helper scripts (hardware discovery, motor scan, PowerShell wrappers for each stage)
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
| 5 | Diffusion Policy | docs/05_diffusion_policy.md |
| 6 | SmolVLA | docs/06_smolvla.md |
| 7 | π₀.₅ | docs/07_pi05.md |
| 8 | Evaluation protocol and comparison | [docs/08_evaluation.md](docs/08_evaluation.md) |

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) and the
[SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) open-source arm design.
