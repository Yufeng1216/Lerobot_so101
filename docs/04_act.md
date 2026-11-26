# 04 · ACT: Action Chunking with Transformers

**English** · [简体中文](04_act.zh-CN.md)

Paper: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023
([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)).

ACT is the **first** of the four policies we train: small (51.6M parameters in our config), fast to train, and 50 demos
are usually enough. It is also the **baseline** for the other three.

## 1. The problem

Behavior cloning is just supervised learning: given the observation $o_t$ (two images + the 6-D joint state $s_t$),
predict the action $a_t$ (6 target joint angles); labels come from human demos. The obvious approach is a network
$\pi(a_t \mid o_t)$ that predicts one action per step, trained with MSE. In practice it works badly, for two reasons:

1. **Compounding error**: in training the network only sees states the human visited; at deployment it sees states it
   visited itself. A small error at one step takes the robot to a state not in the data, the next error is bigger, and it snowballs.
   With one-step prediction, a 300-step episode needs 300 correct decisions in a row.
2. **Inconsistent (multimodal) demos**: from the same view, the human sometimes pauses before grasping and sometimes doesn't;
   sometimes goes around on the left, sometimes on the right. For MSE that is one input with several labels, and the MSE
   optimum is the conditional mean $\mathbb{E}[a \mid o]$, so the model learns their average: a "hesitating" action
   (average left and right and you hit the obstacle in the middle).

ACT's two key ideas map onto these two problems: action chunking for compounding error, a CVAE for multimodality.

## 2. Idea one: action chunking

If error accumulates step by step, make fewer decisions. ACT predicts a whole segment instead of one action:

$$\pi(a_{t:t+k} \mid o_t),\quad k = 100$$

That is **100 actions** at once (about 3.3 s at 30 Hz); the output tensor is `(batch, 100, 6)`.

- 1/k as many decisions, so the "effective number of steps" for error to pile up is 100× shorter.
- A segment predicted as a whole is coherent inside. The network learns "what a stretch of trajectory looks like", not 100 independent points.
  Example: a language model that picked every letter independently would often spell non-words; generating a phrase at a time reads much better.
- PS: human motion is segmented anyway ("reach", "grasp", "lift"); 3.3 s is about one such segment.

## 3. Idea two: a CVAE

### 3.1 Start from the VAE

The trouble with multimodality is that, given only $o_t$, the label $a_{t:t+k}$ is uncertain. So introduce a latent
variable $z$ for "which style this time", such that $a$ becomes determined once $(o, z)$ is given:

$$p(a \mid o) = \int p(a \mid o, z)\, p(z)\, dz$$

That is the idea of a VAE (Variational Autoencoder). The problem: in training we don't know each demo's $z$, and the
integral can't be computed. So the VAE trains an **encoder** $q_\phi(z \mid a, o)$ as well: it sees the answer (the true
action sequence) and guesses that demo's $z$. The objective is the evidence lower bound (ELBO):

$$\log p(a \mid o) \ \ge\ \underbrace{\mathbb{E}_{z\sim q_\phi}\big[\log p_\theta(a \mid o, z)\big]}_{\text{reconstruction: rebuild } a \text{ from the guessed } z} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z \mid a, o)\,\|\,p(z)\big)}_{\text{regulariser: keep } z \text{ close to the prior}}$$

- Reconstruction: with a Laplace $p_\theta$, $-\log p$ is the **L1 error**, which is what ACT uses.
- KL: with a standard-normal prior $p(z)=\mathcal{N}(0,I)$ and a Gaussian encoder $\mathcal{N}(\mu, \sigma^2)$, the KL has a closed form:
  $\mathrm{KL} = -\tfrac12 \sum_j \big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$.
  PS: this is the same KL divergence as in information theory; here it forces "the distribution of $z$ to look like a standard normal".
- Sampling is not differentiable? Use the **reparameterisation trick**: $z = \mu + \sigma \odot \epsilon,\ \epsilon \sim \mathcal{N}(0, I)$, so gradients flow through $\mu$ and $\sigma$.

"C" (conditional) just means generation is conditioned on the observation $o$.

### 3.2 The CVAE inside ACT

ACT's loss is the negative of that ELBO:

$$\mathcal{L} = \underbrace{\frac{1}{k}\sum_{i}\big\|a_{t+i} - \hat a_{t+i}\big\|_1}_{\texttt{l1\_loss}} \;+\; \beta \cdot \underbrace{\mathrm{KL}\big(q_\phi(z \mid a_{t:t+k}, s_t)\,\|\,\mathcal{N}(0,I)\big)}_{\texttt{kld\_loss}},\qquad \beta=\texttt{kl\_weight}=10$$

Two differences from a textbook VAE:

- **The encoder doesn't see the images**, only the joint state $s_t$ and the action sequence $a_{t:t+k}$. The decoder uses the image information itself; $z$ only carries "style".
- **At inference $z = 0$** (the prior mean) instead of a sample from $\mathcal{N}(0,I)$. So the deployed ACT is deterministic: the same observation always gives the same chunk.

PS: a larger $\beta$ puts less information into $z$. If `kld_loss` stays at 0, the encoder has collapsed $z$ onto the prior
and the decoder ignores it (posterior collapse); the CVAE has then degenerated into plain regression.

## 4. Architecture

Three parts: the CVAE encoder (training only), the main encoder, and the decoder.

### 4.1 CVAE encoder (training)

```
[CLS] token         ─ learned embedding               ─┐
joint state s_t (6) ─ Linear → 512                    ─┤  1 + 1 + 100 = 102 tokens
actions a (100×6)   ─ Linear → 512 (one token per step)─┘  + sinusoidal positional encoding
                         │
          Transformer encoder (4 layers, d=512, 8 heads)
                         │
            output at [CLS] → Linear → (μ, log σ²), 32-D each
                         │
            z = μ + σ·ε (reparameterisation)
```

This is BERT's trick: a [CLS] token that summarises the whole sequence.

### 4.2 Main encoder

```
 top image   (3×480×640) ─┐
                          ├─ ResNet18 without the final pool and fc → feature map 512×15×20
wrist image (3×480×640) ─┘     (downsampled 32×: 480/32=15, 640/32=20)
                               → 1×1 conv → flatten into 300 tokens of 512-D (per image)
                               + 2D sinusoidal positional encoding (where on the image each token is)
joint state s_t ─ Linear → 1 token
latent z        ─ Linear → 1 token          (from the CVAE encoder in training, z = 0 at inference)
                         │
         1 + 1 + 300 + 300 = 602 tokens
                         │
     Transformer encoder (4 layers, d=512, 8 heads, FFN 3200) ← self-attention over all tokens
```

- ResNet18 starts from ImageNet weights and is fine-tuned along with everything else (backbone learning rate also 1e-5).
- Both cameras share one ResNet. How does the Transformer know which image and which position a token comes from? Positional encoding.
- Self-attention lets "image patches near the gripper" and "joint state" look at each other directly, which a CNN alone cannot do.

### 4.3 Decoder

```
     100 queries (input all zeros, plus 100 learned positional embeddings)
                         │
     Transformer decoder (1 layer)
       self-attention: the 100 queries look at each other (neighbouring time steps coordinate)
       cross-attention: the queries look at the 602 encoder output tokens
                         │
     Linear(512 → 6) → action sequence â_{t:t+100}, 100 × 6
```

This is really **DETR**: DETR sends 100 "object queries" into the image features, each finding one object; ACT sends
100 "time-step queries" into the observation features, each finding "how to move at step i".
Unlike a language-model decoder there is **no causal mask and nothing autoregressive**: all 100 steps come out in parallel.

PS: LeRobot's default decoder has a single layer. The original ACT code lists 7, but a bug there means only the first is
used, and LeRobot matches that behaviour. 51.6M parameters in total, about 11M of them in ResNet18.

### 4.4 What one training step does

1. Sample a time $t$ from the dataset: two images, $s_t$, and the next 100 actions $a_{t:t+100}$ (near the end of an episode the missing steps are padded and masked out of the loss with `action_is_pad`).
2. The CVAE encoder reads $(s_t, a_{t:t+100})$, outputs $\mu, \sigma$, and samples $z$.
3. Main encoder + decoder predict $\hat a_{t:t+100}$ from (images, $s_t$, $z$).
4. loss = L1 + 10 × KL, backprop, AdamW step.

PS: joints and actions are mean/std-normalised with dataset statistics before entering the network, so the L1 is computed in normalised space and all 6 joints are on the same scale.

## 5. Executing chunks at inference

Default (`n_action_steps = 100`): predict 100 steps → execute all of them → look again and predict the next chunk.
Limit: the policy only "looks" every 3.3 s, so it won't notice if the toy gets bumped in between, and the boundary between chunks can jump.

So the paper has an optional alternative, **temporal ensembling**: predict a new 100-step chunk at **every** step. Then each time $t$
gets predictions from several chunks, which are averaged with exponential weights $w_i = e^{-m \cdot i}$ ($i=0$ is the
oldest prediction, $m$ = `temporal_ensemble_coeff`). Smoother and more reactive, at the cost of running the network every
step (ACT is small enough for 30 Hz). Turn it on at deployment with `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`.

## 6. Training config

| Parameter | Value | Why |
|---|---|---|
| `policy.type` | `act` | |
| `chunk_size` | 100 (default) | 3.3 s, about one "grasp" or "place" |
| `kl_weight` | 10 (default) | paper value |
| `optimizer_lr` | 1e-5 (default; backbone also 1e-5) | paper value; a Transformer with a pretrained backbone is more stable with a small learning rate |
| `batch_size` | 8 | safe for two 480×640 images on 8 GB |
| `steps` | 50,000 | ~45 episodes × 330 frames ≈ 15k frames, about 27 epochs |
| `save_freq` | 10,000 | 5 checkpoints |
| `dataset.eval_split` | 0.1 | hold out 10% of episodes as a **validation set**, validation loss every 2000 steps |

PS: in imitation learning a low validation loss **does not mean** a high real-robot success rate, because compounding error only shows up in closed loop.
It only tells you whether you are badly overfitting. **The real-robot success rate is what counts.**

## 7. Smoke test first (a few minutes)

Real training takes hours. So first run 200 steps to check that the whole pipeline (read data → decode video → forward/backward → save checkpoint) works on Windows, and measure the speed to estimate the total time:

```powershell
.\scripts\train.ps1 act -Steps 200 -Name act_smoke -NoPush --eval_steps=100
```

A checkpoint is saved at step 200 and `--eval_steps=100` runs validation twice, so every stage is covered. If it works, delete `outputs/train/act_smoke`.

PS (Windows): every time LeRobot saves a checkpoint it creates a symlink `checkpoints/last` pointing to the newest one, but
a normal Windows user may not create symlinks (`WinError 1314`), so training **crashes at the first save**. That is why
`scripts/train.ps1` starts training through `scripts/lerobot_train_win.py`, which replaces the link with a **directory junction**
(no privilege needed); everything else behaves exactly like `lerobot-train`. A junction stores an **absolute path**, so if you
move the project folder `last` breaks; then just use the numbered folder (e.g. `checkpoints/050000/pretrained_model`).

Speed (RTX 3070 Laptop, batch 8): 3.7 GB of VRAM, 0.28 s per step once warmed up; the bottleneck is the GPU, not data loading.
Mixed precision (`--policy.use_amp=true`) gave no speed-up, so we keep the default fp32.

PS: in the first 100 steps `data_s` looks large (about 1 s). That is the **start-up time** of the data-loader processes on Windows averaged in, not real slowness. Read speeds after warm-up.

## 8. Before training (laptop)

- **Plug in the charger**: on battery the GPU throttles and can be half as fast.
- **Turn off sleep**: Windows Settings → System → Power → "When plugged in, put my device to sleep after" → Never (the screen may turn off). Sleep pauses training.
- Don't run games or video editing while training; the arm can be powered off.
- Close the Rerun window left from recording; it also uses VRAM.

## 9. Training

```powershell
.\scripts\train.ps1 act
```

The full equivalent command is printed in cyan at the top. Output:

```
outputs/train/act_so101_pick_place/
├── train_config.json                 the full config of this run (reproducible)
└── checkpoints/
    ├── 010000/pretrained_model/      model at step 10k
    │   ├── model.safetensors         weights
    │   ├── config.json               policy architecture
    │   ├── train_config.json
    │   └── policy_preprocessor*.json / policy_postprocessor*.json  normalisation stats and other pre/post-processing
    ├── 020000/ ...
    └── last  →  the newest checkpoint (directory junction)
```

When it finishes, the model is uploaded to `https://huggingface.co/Yufeng1216/act_so101_pick_place`.

Interrupted? Resume:

```powershell
.\scripts\train.ps1 act -Resume
```

PS: if the upload at the end fails (e.g. the HF login expired, `401`), the model is still complete on disk. Log in again and upload by hand:

```powershell
hf upload Yufeng1216/act_so101_pick_place outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model .
```

So before a long run, check the login with `hf auth whoami`.

### Results

50,000 steps in **4 h 01 min** (3.7 steps/s). Validation loss (every 2000 steps):

| Step | 2k | 4k | 6k | 8k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|---|
| `eval_loss` | 0.406 | 0.405 | 0.396 | 0.379 | 0.385 | 0.387 | 0.380 | 0.387 | 0.381 |

- Validation loss **stops falling after about 8k steps (≈4 epochs)** and hovers around 0.38. That matches the rule of thumb that imitation learning converges in 5–10 epochs.
- It **doesn't rise** either, so longer training did not cause visible overfitting.
- But a flat validation loss **doesn't mean** the later steps were useless: training loss keeps falling and the motion may get finer, which only shows in closed loop on the robot.

### Training curves

Training logs to Trackio. In another terminal:

```powershell
conda activate lerobot
trackio show --project so101
```

A dashboard opens in the browser. Watch:

- `train/loss`: total = `l1_loss` + 10 × `kld_loss`, should fall steadily
- `train/l1_loss`: action reconstruction error (normalised space), the main one to watch
- `train/kld_loss`: small but not 0. Stuck at 0 means the CVAE latent is unused (posterior collapse)
- `eval/loss`: validation loss. Training loss keeps falling while validation loss rises = overfitting

## 10. Running on the robot

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30
```

PS: `-Policy` also accepts the Hub name `Yufeng1216/act_so101_pick_place`. On Windows, though, LeRobot may turn the `/` in
the name into `\` and fail to find the model (see code walkthrough §8), so a local path is safer.

Safety:

- The first time, keep a hand near the follower's power supply; the policy may do something unexpected.
- At start the follower moves from wherever it is to the policy's first action, so put it in the **start pose** first.
- The scene must match recording: camera positions, box, marks and lighting unchanged.

Try temporal ensembling:

```powershell
.\scripts\robot.ps1 rollout -Policy outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model -Duration 30 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
```

## 11. Evaluation

40 trials following the protocol in [08_evaluation.md](08_evaluation.md); the success rate goes into the README results table.
