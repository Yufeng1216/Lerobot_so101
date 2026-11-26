# 05 · Diffusion Policy: generating actions with a diffusion model

**English** · [简体中文](05_diffusion_policy.zh-CN.md)

Paper: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023
([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)).

In one sentence: **take an image diffusion model and replace "generate an image" with "generate a segment of actions"**,
and the condition "a sentence of text" with "the current images and joint state".
Below we first build up diffusion models from scratch, then see how they are applied to actions.

## 1. Why generate actions with a diffusion model

Back to the second problem in [04_act.md](04_act.md): **inconsistent demos**. From the same view the human sometimes goes
around on the left, sometimes on the right, so the right action distribution is **multimodal**:

```
     probability
      │   ▲               ▲
      │  ╱ ╲             ╱ ╲
      │ ╱   ╲           ╱   ╲
      └────────────────────────── action
       go left  (average)  go right
                    ↑ MSE regression outputs this: straight into the obstacle
```

- **MSE regression** can only output the conditional mean, right between the two modes: an action nobody ever did.
- **ACT** uses the CVAE latent $z$ to separate the modes, but fixes $z = 0$ at inference, which just picks one "standard style". Limited.
- **A diffusion model** learns the whole distribution $p(A \mid O)$ and **samples** from it at inference, starting from noise and landing on one mode each time. It was designed to generate multimodal high-dimensional data (images) in the first place.

So the question becomes: how do we learn a high-dimensional conditional distribution and sample from it?

## 2. How a diffusion model (DDPM) works

### 2.1 Starting point: turn "generation" into many small "denoising" steps

Mapping $\mathcal N(0,I)$ to a sensible action segment in one step is hard (that is what GANs and VAEs try).
DDPM's idea: if one step is too hard, split it into $K$ small steps, each removing only a little noise. Each small step is easy, so a network can learn it.

### 2.2 Forward process: add noise bit by bit (fixed, nothing learned)

Let the real action sequence be $A^0$ (64 steps × 6). Each step adds a little Gaussian noise:

$$q(A^k \mid A^{k-1}) = \mathcal N\big(\sqrt{1-\beta_k}\,A^{k-1},\ \beta_k I\big),\qquad k = 1,\dots,K$$

$\beta_k$ are small fixed numbers (the noise schedule; here the cosine schedule `squaredcos_cap_v2`, $K=100$).
Gaussian plus Gaussian is Gaussian, so we can jump to any step $k$ directly. With $\bar\alpha_k = \prod_{i\le k}(1-\beta_i)$:

$$A^k = \sqrt{\bar\alpha_k}\,A^0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\qquad \epsilon \sim \mathcal N(0, I)$$

The larger $k$, the closer $\bar\alpha_k$ is to 0 and the closer $A^k$ is to pure noise. At $k = K$ it is essentially $\mathcal N(0,I)$.

### 2.3 Reverse process: learn a denoiser

Generation runs the forward process backwards: start from $A^K \sim \mathcal N(0,I)$ and denoise step by step back to $A^0$.
Each reverse step is also approximately Gaussian, and its mean only depends on "how much noise is in this step". So we
just train a network $\epsilon_\theta(A^k, k, O)$ to **predict the noise that was added**:

$$\mathcal L = \mathbb E_{A^0,\,k,\,\epsilon}\Big[\big\| \epsilon - \epsilon_\theta(A^k, k, O) \big\|^2\Big]$$

So one training step is:

1. take a sample $(O, A^0)$ from the data, pick a random $k \in \{1,\dots,100\}$, sample $\epsilon$;
2. add noise with the closed form above to get $A^k$;
3. the network sees $(A^k, k, O)$, predicts $\epsilon$; compute the MSE.

PS: it is still an MSE loss, but this time MSE does not average the modes away. The network predicts **one specific noise
sample**, not the action itself; starting from different initial noise, the reverse process ends up on different modes.

At inference (DDPM sampling), each step uses the predicted noise to compute the previous step's mean, plus a little fresh noise:

$$A^{k-1} = \frac{1}{\sqrt{1-\beta_k}}\Big(A^k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_\theta(A^k,k,O)\Big) + \sigma_k z,\quad z\sim\mathcal N(0,I)$$

After 100 steps we have $A^0$. Limit: 100 network passes, **much slower than ACT (one forward pass)**. That is the main price of a diffusion policy.

### 2.4 DDIM: same network, fewer steps

DDIM (Song et al., 2021) showed that without changing training at all, you can swap the sampler for a **deterministic**
update (no fresh noise $z$) that can **skip steps**, e.g. only 100 → 90 → … → 0, 10 steps. Each step first uses
$\epsilon_\theta$ to estimate the clean $\hat A^0$, then computes $A$ at the target step directly:

$$\hat A^0 = \frac{A^k - \sqrt{1-\bar\alpha_k}\,\epsilon_\theta}{\sqrt{\bar\alpha_k}},\qquad A^{k'} = \sqrt{\bar\alpha_{k'}}\,\hat A^0 + \sqrt{1-\bar\alpha_{k'}}\,\epsilon_\theta$$

So DDIM is an inference-time speed-up for **the same trained model**, switched on with one flag. On the robot we use 10-step DDIM (§8).

## 3. Applying it to actions: three time lengths

How much time do the condition $O$ and the generated $A$ cover? This is the easiest part of Diffusion Policy to mix up. Defaults are 2 / 64 / 32:

```
time →        t-1   t   t+1 ...                                   t+62
observe       [■■■■■■]                                               n_obs_steps = 2: look at the last 2 frames
predict       [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]  horizon = 64: generate 64 steps at once
execute             [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                n_action_steps = 32: execute only the first 32
                                                    ↑ then observe and generate again
```

- **2 frames**: the difference between two frames gives velocity; one frame can't.
- **Generate 64, execute 32**: the second half only lets the first half "know where it is heading", for coherence; re-planning
  halfway lets it correct in time. The paper calls this **receding horizon**; it is the same idea as ACT's action chunking.
- 32 steps at 30 Hz ≈ re-planning every second, much more often than ACT's default 3.3 s.

PS: actions and states are min/max-normalised to [−1, 1] (not mean/std as in ACT) so they match the scale of $\mathcal N(0,I)$; samples are also clipped to [−1, 1] (`clip_sample`).

## 4. Architecture

Two parts: a condition encoder and a denoising network.

```
 top image   (3×240×320) ─ ResNet18 ─ Spatial Softmax (32 keypoints) ─ Linear ─ 64-D ─┐
 wrist image (3×240×320) ─ ResNet18 ─ Spatial Softmax (32 keypoints) ─ Linear ─ 64-D ─┼─ concat = 134-D per frame
 joint state (6)                                                                   ─┘     × 2 frames = 268-D
                                                                                            │
 noise step k ─ sinusoidal encoding ─ MLP ─ 128-D ───────────────────────────── concat ─▶ condition c (396-D)
                                                                                            │
 noisy actions A^k (64×6) ──▶  1D temporal U-Net (FiLM with c in every residual block)  ◀───┘
                 │
                 ▼
          predicted noise ε̂ (64×6)
```

### 4.1 Condition encoder: ResNet18 + Spatial Softmax

Each camera has its own ResNet18 (`use_separate_rgb_encoder_per_camera`), initialised from ImageNet.
Unlike ACT, the feature map is not flattened into tokens; it goes into a **Spatial Softmax**:

1. a 1×1 conv turns the feature map into 32 channels, one per "keypoint";
2. a softmax over space per channel gives a probability map $p_c(u, v)$;
3. the expected coordinates $(\mathbb E[u], \mathbb E[v])$ are the keypoint's position.

So each image becomes 32 (x, y) coordinates = 64 numbers. It says **where** things are rather than **what they look like**, which suits manipulation.
PS: this comes from Levine et al.'s visuomotor policies; it lets the network learn 32 "points worth tracking" by itself.

### 4.2 Denoiser: a 1D temporal U-Net

U-Net is the structure from image segmentation and image diffusion: an encoder that downsamples, a decoder that upsamples,
and skip connections between matching levels. Here 2D convolutions become **1D convolutions along time**, and the 6 joints are 6 input channels:

```
A^k (6 channels × 64 steps)
  ├─ residual block ×2 (512 ch) ─────────────────────┐ skip
  ├─ downsample (stride 2) → 32 steps                 │
  ├─ residual block ×2 (1024 ch) ───────────┐ skip    │
  ├─ downsample → 16 steps                  │         │
  ├─ residual block ×2 (2048 ch)            │         │
  ├─ middle residual block ×2               │         │
  ├─ upsample → 32 steps, concat skip ◀─────┘         │
  ├─ upsample → 64 steps, concat skip ◀───────────────┘
  └─ 1D conv → 6 channels × 64 steps = ε̂
```

Each residual block is Conv1d (kernel 5) → GroupNorm → Mish, twice, plus a residual connection.
Convolving along time naturally smooths neighbouring actions, and after downsampling the network sees the trend of the whole 64-step segment.

### 4.3 Injecting the condition: FiLM

The condition $c$ has to tell the U-Net "what is seen and how noisy this step is". Diffusion Policy uses **FiLM**
(Feature-wise Linear Modulation): in every residual block a linear layer maps $c$ to a per-channel scale $\gamma$ and shift $\beta$, and

$$h \leftarrow \gamma(c) \odot h + \beta(c)$$

So the observation is not concatenated as an input; it "modulates" the gain and bias of every layer's features.

| Part | Image diffusion (e.g. Stable Diffusion) | Diffusion Policy |
|---|---|---|
| What is generated | an image (2D pixel grid) | an action segment (64 steps × 6) |
| Denoiser | 2D U-Net | **1D U-Net** (convolution along time) |
| Condition | text, encoded by CLIP | images + joint state, encoded by ResNet18 + Spatial Softmax |
| Injection | cross-attention | **FiLM** |

With the default `down_dims = (512, 1024, 2048)` there are about 278M parameters, almost all in the U-Net.

## 5. Training config

| Parameter | Value | Why |
|---|---|---|
| `policy.type` | `diffusion` | |
| `n_obs_steps / horizon / n_action_steps` | 2 / 64 / 32 (default) | paper values |
| `resize_shape` | **[240, 320]** | half of 480×640, 4× fewer pixels, much faster. The paper itself uses even lower resolutions (96×96 to 240×320) |
| `crop_ratio` | **0.95** | random 95% crop in training (centre crop at inference). The paper's **data augmentation**, makes the model robust to small camera shifts |
| `ema.type` | **warmup** | keep an **exponential moving average** of the weights (EMA). Standard for diffusion; EMA weights give smoother actions |
| `optimizer_lr` | 1e-4, cosine, 500 warm-up steps (default) | paper values |
| `down_dims` | 512 / 1024 / 2048 (default) | 278M parameters, needs 16 GB+ of VRAM (§6) |
| `batch_size` | 8 | same as ACT |
| `steps` | 100,000 | learning a whole distribution is harder than regression and usually takes more steps; about 53 epochs |
| `save_freq` | 20,000 | 5 checkpoints |

PS: with EMA on, each checkpoint has a `pretrained_model_ema/` next to `pretrained_model/`. The EMA one is what gets uploaded and what we deploy.

## 6. Memory: why not train on the laptop

At the default size, fp32 weights, gradients, Adam's two states and the EMA copy alone take about 5.5 GB; add activations and an 8 GB laptop GPU can't hold it.

PS: when VRAM runs out, Windows doesn't necessarily say `CUDA out of memory`. The driver moves what doesn't fit into system RAM;
training runs without error but a hundred times slower (about 2 minutes per step here). So if `nvidia-smi` shows VRAM nearly full and `updt_s` in the log is huge, that's why.

So the default size trains on **a GPU with more memory** (a 16 GB V100 on a Slurm cluster). If you only have 8 GB and want to train locally,
the diffusion preset in `train.ps1` halves the U-Net channels to (256, 512, 1024) (89M parameters, 2.9 GB, 0.13 s per step; the paper uses this size too), everything else unchanged.

## 7. Training (Linux GPU cluster)

Use the scripts in `cloud/slurm/`, with the settings of §5. Full steps in [cloud/README.md](../cloud/README.md); the core is three commands:

```bash
bash cloud/slurm/setup_env.sh          # one-time environment setup
bash cloud/slurm/submit.sh diffusion   # submit the training job
tail -f logs/diffusion_<jobid>.log     # follow the log
```

Tesla V100 16 GB, batch 8:

| `step_s` (per step) | `data_s` (waiting for data) | VRAM | 100k steps |
|---|---|---|---|
| 0.127 s | 0.003 s | 6.5 GB | about 4 h (incl. validation) |

- Cluster jobs have a time limit. When one is stopped, run `submit.sh diffusion` again and it continues from the last checkpoint.
- Diffusion's `train/loss` is the noise-prediction MSE; it is **not comparable** with ACT's loss, only its own trend matters.

Copy only the EMA weights back to the laptop (about 1 GB, no optimizer state). On the laptop:

```powershell
scp -r <user>@<cluster-transfer-host>:~/Lerobot/outputs/train/diffusion_so101_pick_place/checkpoints/last/pretrained_model_ema outputs/train/diffusion_so101_pick_place/
```

Trained model: [Yufeng1216/diffusion_so101_pick_place](https://huggingface.co/Yufeng1216/diffusion_so101_pick_place).

## 8. Running on the robot: use DDIM

First measure speed offline by feeding dataset frames to the model (RTX 3070 Laptop):

| Sampler | Per generation | Mean absolute error vs. the demo actions (8 frames × 6 joints) |
|---|---|---|
| DDPM, 100 steps (training default) | about 1.5 s | 3.2 |
| DDIM, 10 steps | about 0.16 s | 3.3 |

The 32 actions generated each time (`n_action_steps`) last only about 1.07 s at 30 Hz. 100-step DDPM is slower than that,
so the arm would stop and wait after every segment; 10-step DDIM is 10× faster with practically the same error. So **use DDIM on the robot**:

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/diffusion_so101_pick_place/pretrained_model_ema -Duration 30 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
```

PS: per-frame errors range from 0.4 to 7; the largest is at the very start of a demo when the arm starts from rest (elbow off by 10–20+).
When to start moving is genuinely uncertain, so a large error there isn't necessarily wrong. This is only an offline check; the real metric is the real-robot success rate.

## 9. What to look at when comparing with ACT

- **Success rate**: 40 trials following [08_evaluation.md](08_evaluation.md).
- **Decisiveness**: at spots where "grasp from the left or from the right" both work, ACT may hesitate; the diffusion policy should commit.
- **Inference latency**: how long each generation takes and whether it hurts smoothness.
