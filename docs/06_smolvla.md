# 06 · SmolVLA: a small vision-language-action model

**English** · [简体中文](06_smolvla.zh-CN.md)

Paper: Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025
([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)).

The first two policies (ACT, Diffusion) are trained **from scratch** on our 50 demos only. SmolVLA is different: it is a
**pretrained large model**, and we only **fine-tune** it. It is really the same idea as in NLP, "pretrain on a big corpus,
then fine-tune on your task", except that the corpus is robot data.

## 1. What a VLA is

**VLA = Vision-Language-Action**: images and a natural-language instruction in, robot actions out.

```
images (top + wrist)  ┐
instruction (text)    ├─▶  VLA  ─▶  the next 50 joint actions
joint state           ┘
```

The starting point: a VLM (vision-language model) has already learned to "understand a scene" and "follow an instruction"
from huge amounts of image-text data. So rather than learning vision from scratch for a robot policy, use the VLM as the
backbone and attach a module that outputs actions.

| | ACT / Diffusion | SmolVLA |
|---|---|---|
| Initialisation | only ResNet18 is pretrained (ImageNet) | the VLM is pretrained on image-text data, and **the whole VLA** is pretrained again on community robot data |
| Input | images + joint state | images + joint state + **language instruction** |
| Training | from scratch | fine-tuned from pretrained weights |

PS: we only have one task, and the instruction is always the same sentence ("Pick up the blue plush toy and place it in the cardboard box."),
so language can't "tell tasks apart" here. What we care about is whether **the prior from pretraining helps learn better from 50 demos**.

## 2. Overall structure: the VLM handles the prefix, the action expert the suffix

SmolVLA (~450M parameters) is two Transformers running **side by side, layer by layer**:

```
 prefix: processed by the VLM                   suffix: processed by the action expert
┌───────────────────────────────────────────┐   ┌──────────────────────────────┐
│ image tokens (64 each) × 2 cameras         │   │ 50 action tokens              │
│ instruction tokens (up to 48)              │   │ = noisy actions + encoding    │
│ state token (1)                            │   │   of the time t               │
└───────────────────────────────────────────┘   └──────────────────────────────┘
        first 16 layers of SmolVLM2 (width 960) ◀──attention──▶ action expert, 16 layers (width 720)
                                                                   │
                                                     Linear → velocity field v, 50 × 6
```

Let's take it apart piece by piece.

### 2.1 The VLM: SmolVLM2-500M

A VLM is really "ViT + projection + language model":

1. **Vision encoder SigLIP** (a ViT, 12 layers, width 768): the image is resized and padded to 512×512 and cut into 16×16 patches: 32×32 = 1024 patch tokens.
2. **Pixel shuffle**: 1024 tokens is too many for the language model. So each 4×4 group of neighbouring tokens is stacked along the channel dimension: 1024 / 16 = **64 tokens**, each 16× wider.
   It trades space for channels; no information is lost, only rearranged.
3. **Projection (connector)**: a linear layer maps the 64 tokens to the language model's width (960), so image tokens and word tokens live in the same embedding space and can be concatenated into one sequence.
4. **Language model SmolLM2** (32 layers, width 960, 15 attention heads, 5 KV heads, i.e. GQA): SmolVLA **only uses the first 16 layers**.
   The paper found the middle layers' features are already enough for actions; dropping half the layers makes inference twice as fast.

The joint state (6-D, zero-padded to 32) goes through a linear layer `state_proj` into **1 token** at the end of the prefix.

### 2.2 The action expert

The action expert is also a Transformer, with as many layers as the VLM (16) but 0.75× its width (720). Its input is 50 action tokens:

- the noisy actions (50 × 32; 6-D zero-padded to 32) go through `action_in_proj` into 50 vectors of 720-D;
- the time $t$ gets a sinusoidal encoding, concatenated to each action vector, then a two-layer MLP. So every action token "knows" how noisy it is.

At the output, `action_out_proj` maps the 50 tokens back to 50 × 32; the first 6 dimensions are the predicted velocity field $v$.

### 2.3 How the two Transformers talk: alternating cross- and self-attention

In every layer the VLM first computes K and V for the prefix tokens. The expert's layer is one of two kinds:

- **cross-attention layer**: the action tokens' Q attends to the **prefix**'s K and V (mapped to the expert's width by a linear layer). In words: "decide the action from the view, the instruction and the state".
- **self-attention layer** (every other layer): action and prefix tokens attend jointly, so action tokens also see each other and the 50 steps stay coordinated.

On top of that there is a mask rule ("block-causal"):

| Who sees whom | images + instruction | state | actions |
|---|---|---|---|
| images + instruction | ✓ (each other) | ✗ | ✗ |
| state | ✓ | ✓ | ✗ |
| actions | ✓ | ✓ | itself and earlier actions only |

**The prefix never sees the actions.** That matters: at inference the prefix is computed **once** (kept as a KV cache), and the 10 denoising steps only run the action expert, which is fast.

## 3. Flow matching: the "straight-line" diffusion

The action expert generates the chunk with **flow matching**. Same idea as diffusion, "go from noise to data", but along a simpler path.

### 3.1 Starting point

DDPM's noising path is curved ($\sqrt{\bar\alpha_k}$ and $\sqrt{1-\bar\alpha_k}$ change non-linearly), so inference needs many small steps.
So why not let noise and data be connected by a **straight line**?

### 3.2 Training

Take a real action chunk $A$ and noise $\epsilon \sim \mathcal N(0, I)$ and interpolate linearly ($t=1$ is pure noise, $t=0$ is data):

$$A^t = t\,\epsilon + (1-t)\,A$$

The velocity along that line is constant, $\frac{dA^t}{dt} = \epsilon - A$. So the network predicts that velocity:

$$\mathcal L = \big\| v_\theta(A^t, t, O) - (\epsilon - A) \big\|^2$$

$t$ is not sampled uniformly but from Beta(1.5, 1), **favouring the noisy end** ($t$ near 1), because that end is the hardest: deciding, from a blob of noise, which mode to head for.

### 3.3 Inference

Start from $A^1 = \epsilon$ and integrate the velocity field with Euler's method for 10 steps (`num_steps = 10`, step $\Delta t = -0.1$):

$$A^{t+\Delta t} = A^t + \Delta t \cdot v_\theta(A^t, t, O)$$

| | DDPM (Diffusion Policy) | Flow matching (SmolVLA) |
|---|---|---|
| What the network predicts | the added noise $\epsilon$ | the velocity from data to noise, $\epsilon - A$ |
| Noising path | curved (set by the noise schedule) | straight |
| Inference steps | 100 by default (10 with DDIM) | 10 |

PS: the path is straight, so a few large steps don't add much error; that's why flow matching is fast at inference. It is just as multimodal as diffusion, because every sample starts from different noise.

## 4. What gets trained during fine-tuning

Defaults (we keep them):

| Part | Trained? | Setting |
|---|---|---|
| SigLIP vision encoder | frozen | `freeze_vision_encoder=True` |
| language model (first 16 layers) | frozen | `train_expert_only=True` |
| action expert | **trained** | |
| state/action projections, time MLP | **trained** | `train_state_proj=True` |

About 100M parameters are trained. So although the model has 450M, gradients and optimizer state only take a small share of memory.
PS: it is the same idea as LoRA: freeze the big model's general knowledge and train only a small part to adapt to the new task.

## 5. Camera names: `rename_map`

The pretrained model knows cameras called `camera1`, `camera2`, `camera3`; our dataset calls them `top` and `wrist`. `--rename_map` tells LeRobot how they match:

| Our camera | Mapped to | Why |
|---|---|---|
| `observation.images.top` | `observation.images.camera1` | camera1 is mostly an overhead or front view in the pretraining data |
| `observation.images.wrist` | `observation.images.camera2` | camera2 is mostly a wrist view |
| (none) | `camera3` | missing cameras are skipped |

PS: `rename_map` is saved in the model's preprocessing config (`policy_preprocessor.json`), so it applies automatically at deployment.

## 6. Training config

| Parameter | Value | Why |
|---|---|---|
| `policy.path` | `lerobot/smolvla_base` | start from the official pretrained weights (not `policy.type`, which would train from scratch) |
| `rename_map` | top → camera1, wrist → camera2 | see §5 |
| `batch_size` | 32 | the official example uses 64; 32 fits a 16 GB V100 |
| `steps` | 20,000 | 640k samples, about 43 epochs |
| `scheduler_decay_steps` | same as `steps` | the default decays over 30k steps; change both together |
| `save_freq` | 5,000 | resubmitting after a cluster time-out loses at most 5k steps |
| rest | default | lr 1e-4, 1000 warm-up steps, cosine decay to 2.5e-6 |

PS: on the laptop (8 GB) batch 32 doesn't fit (VRAM spills into system RAM and gets thousands of times slower); batch 8 works (0.35 s per step, 2.9 GB).
So `train.ps1 smolvla` is a laptop version with batch 8 and 30k steps; the real model was trained on the cluster with batch 32.

## 7. Training

Smoke test first. The first run downloads the pretrained model (about 2 GB, into the Hugging Face cache):

```powershell
.\scripts\train.ps1 smolvla -Steps 300 -Name smolvla_smoke -NoPush --eval_steps=150 --trackio.enable=false
```

The real run is on the cluster (see [cloud/README.md](../cloud/README.md)):

```bash
bash cloud/slurm/submit.sh smolvla
```

PS: the V100 has no native bfloat16 (SmolVLA keeps its VLM in bf16); it works but is slower: 2.47 s per step, about 14 h for 20k steps.
After the 8 h time-out, submit again and it continues automatically.

Copy back only the `pretrained_model` needed for deployment (about 0.9 GB, no optimizer state). On the **laptop**:

```powershell
scp -r <user>@<cluster-transfer-host>:~/Lerobot/outputs/train/smolvla_so101_pick_place/checkpoints/020000/pretrained_model outputs/train/smolvla_so101_pick_place/
```

### Results

- All 20k steps done, about 14 h on the V100.
- **Offline check**: one frame from each of 5 demos (before grasp, grasping, carrying, releasing); predict a 50-step chunk and compare with the recorded actions.
  Mean absolute error per joint (degrees; gripper 0–100):

| | pan | lift | elbow | wrist flex | wrist roll | gripper | mean |
|---|---|---|---|---|---|---|---|
| SmolVLA | 1.4 | 3.0 | 2.7 | 1.9 | 2.3 | 0.5 | **2.0** |
| baseline: hold still | 14.2 | 29.3 | 31.2 | 13.1 | 3.2 | 7.3 | 16.4 |

  PS: these frames are in the training set, so this only shows the model loads correctly and training worked, not that it generalises (the same check gives 1.7 for π₀.₅, see [07_pi05.md](07_pi05.md)).
- **Inference speed**: about 0.38 s per chunk on the laptop GPU (1.2 s the first time, with warm-up), 0.9 GB of VRAM. A chunk lasts 1.67 s, so it runs on the robot straight from the laptop.

## 8. Running on the robot

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/smolvla_so101_pick_place/pretrained_model -Duration 30
```

`robot.ps1` passes the task description from `config/robot.json` to the model as the language instruction.

## 9. What to look at when comparing with the first two

- **Success rate**: 40 trials following [08_evaluation.md](08_evaluation.md).
- **What pretraining buys**: with the same 50 demos, how does fine-tuned SmolVLA compare with ACT / Diffusion trained from scratch?
- **Inference speed**: 10 flow-matching steps with the prefix computed once is faster than Diffusion's denoising, but the model itself is much bigger than ACT.
