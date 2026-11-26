# 09 · Zero-shot baseline: vision-language model + inverse kinematics (no demonstrations)

**English** · [简体中文](09_zero_shot_vlm.zh-CN.md)

## 1. Why this experiment

The four policies so far (ACT, Diffusion, SmolVLA, π₀.₅) are all **end-to-end imitation learning**: they learn "image → action" from 50 demos.
The other route is **no training at all**: let a general vision-language model (VLM) understand the scene, then execute with classical geometry and a fixed motion script.

Putting both routes on **the same task with the same evaluation protocol** answers a practical question:

> How much do 50 human demos actually add over "an off-the-shelf model + geometry"? And where does each one fail?

## 2. Pipeline

```
overhead camera frame (640×480)
   │
   ▼  ① perception: VLM (Qwen3-VL-2B-Instruct, running locally)
   │     prompts: "Point to the blue plush toy" / "Point to the center of the cardboard box"
   │     output: pixel coordinates (u, v)
   ▼  ② pixel → table: homography H (calibrated from the demos; one for the toy, one for the box)
   │     (u, v) ──H──▶ (x, y), in the arm's base frame, metres
   ▼  ③ inverse kinematics (IK): SO-101 URDF + damped least squares
   │     "gripper tip at (x, y, z), pointing down" → 5 joint angles
   ▼  ④ fixed motion script (open loop)
         open → 6 cm above the toy → straight down → close → straight up to 11 cm → lift to ~17 cm →
         over to the box at that height → down to 11 cm → release → back to the start pose
```

Let's go through what each block does.

## 3. Perception: how a VLM "points" at something

### 3.1 Qwen3-VL's structure

Like SmolVLM and PaliGemma, Qwen3-VL-2B is "ViT + projection + language model":

1. **Vision encoder** (ViT, 24 layers, width 1024, patch 16): it accepts **any resolution** without resizing to a square. A 640×480 image becomes 40×30 = 1200 patches.
   Positions use 2D rotary position embeddings (RoPE), so the model knows each patch's row and column.
2. **Merger**: each 2×2 group of neighbouring patches becomes one token (the same idea as SmolVLM's pixel shuffle): 1200 → **300 visual tokens**, then projected to the language model's width.
3. **Language model** (Qwen3, 28 layers, width 2048, GQA with 16/8 heads): visual and text tokens form one sequence and the answer is generated **autoregressively**.

### 3.2 Coordinates are just text

The key point: Qwen3-VL saw lots of "point at the object" data (grounding) during pretraining, so it can **output coordinates as text**. Our prompt:

```
Point to the blue plush toy in the image. Answer only with JSON in this format: [{"point_2d": [x, y], "label": "toy"}]
```

It answers `[{"point_2d": [512, 233], "label": "toy"}]`. Those numbers are ordinary tokens, no different from generating a sentence.
PS: Qwen3-VL outputs **relative coordinates on a 0–1000 scale**, not pixels; multiply by the image width/height and divide by 1000.

Limit: the numbers are discrete tokens, so precision is capped, and the model gives a single point, not the object's size or orientation. For "roughly where is the toy" that is enough (§6).

### 3.3 Why this model

| Consideration | Reason |
|---|---|
| outputs coordinates directly | no separate detector needed |
| 2B parameters | ~4 GB in bf16, fits the 8 GB GPU, ~1.7 s per query |
| runs locally | free, and no images of the room go to an external server |

## 4. Geometry: from pixels to the table

### 4.1 Homography

The camera gives a 2D image; the arm needs 3D coordinates. In general that takes depth, but our case is special: **the toy always lies on the table**, and the table is a plane.
A plane seen through a perspective projection onto another plane (the image) is mapped by exactly one 3×3 matrix $H$, in homogeneous coordinates:

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix},\qquad x = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}},\quad y = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

$H$ is defined up to scale, so it has 8 degrees of freedom. Each "pixel ↔ table point" pair gives 2 equations, so **4 pairs solve it**; more pairs go into least squares.
The $h_{31}, h_{32}$ in the denominator are the perspective effect (near things look bigger). So **no depth camera** and no camera intrinsics are needed.

### 4.2 Where the point pairs come from: the demos themselves

Solving $H$ needs pairs. We didn't run a separate calibration; we took them from the 50 demos:

- **pixel**: in each demo's first frame the blue toy is the only blue thing on the white table, so an HSV colour threshold + the largest connected component gives its centre $(u, v)$;
- **table point**: at the moment the gripper closes on the toy, the recorded joint angles go through **forward kinematics (FK, §4.3)** to give the gripper tip's $(x, y)$.

The toy doesn't move between those two moments, so each demo is one pair. `cv2.findHomography` solves $H$ with **RANSAC**:
pick 4 random pairs, solve an $H$, count how many points agree with it (error < 1.5 cm), repeat many times, and keep the one with most agreement. Wrongly detected points drop out automatically.

PS: the table grasp points all sit near the 5 marks (about 10×10 cm), and far from them $H$ extrapolates badly (24 cm off at the box).
So the box gets a second mapping, `H_pixel_to_place`, fitted with the "moment of release" pairs added (the toy in the image ↔ the gripper position by FK).
At release the toy is about 9 cm above the table, so there are a few cm of parallax, but the box opening is about 20 cm wide; good enough.

### 4.3 Forward kinematics (FK)

The arm is a chain of "link + joint". The URDF gives each joint's fixed offset (translation + rotation) from its parent link and its rotation axis.
So the gripper tip's pose in the base frame is a product of 4×4 homogeneous transforms:

$$T_{\text{tip}}(q) = \prod_{i=1}^{5} \Big( T_i^{\text{fixed}} \cdot \mathrm{Rot}(\text{axis}_i,\ q_i) \Big) \cdot T^{\text{tip}}$$

The top-left 3×3 of $T$ is the orientation, the top-right 3×1 the position.

PS: in LeRobot 0.7 the SO-101 joint readings are in degrees with 0° = middle of the calibrated range, the same zero as the official `so101_new_calib.urdf`, so they go straight in.
We checked against the demos: over 35 table grasps, FK puts the gripper tip at **−0.8 ± 1.0 cm** height (tips pressing slightly into the soft table mat), and the grasp positions form 5 clusters, exactly the 5 taped marks.
With a wrong zero or direction these numbers would be all over the place.

### 4.4 Inverse kinematics (IK): damped least squares

IK goes the other way: given "tip at $p^*$, gripper pointing down", find the joint angles $q$. The SO-101 has only 5 joints (and wrist_roll only spins the gripper about itself, not moving the tip),
so an arbitrary orientation is impossible, but "position + pointing down" is solvable.

There is no closed form, so we iterate. Define the 6-D residual:

$$r(q) = \begin{bmatrix} p(q) - p^* \\ w \cdot \big(z_{\text{approach}}(q) - [0, 0, -1]^T\big) \end{bmatrix}$$

The first 3 entries are the position error, the last 3 the difference between "the gripper's approach direction" and "straight down", weighted by $w = 0.05$.
Each step uses a numerical Jacobian $J = \partial r / \partial q$ (move each joint 0.01° and see how the residual changes), then

$$\Delta q = -J^T \big(J J^T + \lambda^2 I\big)^{-1} r$$

- Without $\lambda$ this is Gauss-Newton; but near a singular pose (e.g. the arm stretched out) $JJ^T$ is nearly singular and $\Delta q$ explodes.
  Adding $\lambda^2 I$ makes the step shrink automatically near singularities. That is **damped least squares** (the Levenberg-Marquardt idea).
- PS: $\lambda$ has to match the units: 1° of joint motion moves the tip only about 3 mm, so Jacobian entries are around $10^{-3}$ and $\lambda \approx 10^{-3}$.
- After each step the angles are clipped to the URDF joint limits.
- The initial guess matters: from the folded start pose shoulder_lift sits right at its −100° limit and the solver gets stuck. So several typical grasp poses from the demos are used as seeds, keeping the best result.

Result: **0.1 mm** error on table targets, gripper pointing straight down.

## 5. Execution: a fixed motion script

`run_grasp.py` chains the key poses solved by IK:

```
start pose → 6 cm above the toy (open) → straight down (IK every 0.5 cm, so the descent is a straight line) → close
           → straight up to 11 cm → lift to ~17 cm → over to the box at that height → down to 11 cm → release → back to the start pose
```

Why carry at 17 cm: the toy hangs several cm under the gripper. In the demos the human lifted the tip to 13.7 cm on average while carrying
(90% below 16.6 cm), so 17 cm clears the box wall in every demo. Above about 13 cm the 5-joint arm can no longer keep the gripper exactly vertical,
so the carry poses are solved with a looser tolerance (a few cm); they are in free air, and the path check below still guards them.

- Between key poses it interpolates linearly in joint space and sends targets at 30 Hz to `SO101Follower.send_action()`, at most 20°/s per joint by default.
  That is **exactly the same low-level interface and calibration files** as the learned policies use, so the comparison is fair.
- **Checked before moving**: targets within a sensible area, small IK error, and along every interpolated point of the path the tip stays above 2.5 cm below the table surface (and above 14 cm while carrying, so the toy doesn't hit the box wall).
  If anything fails, nothing moves.
- `--step` pauses before each segment until you press Enter; Ctrl+C first returns slowly to the start pose, then disconnects.

Offline: with the toy at each demo's grasp point and the box at its release point, a safe path is planned in **35 of 35** cases; while carrying, the tip never drops below 16.6 cm; one full attempt takes about 25 s.

## 6. Offline validation

Code in [`zero_shot/`](../zero_shot/); explained in [code_walkthrough.md](code_walkthrough.md) §14.

| Module | Test | Result |
|---|---|---|
| FK | tip height at grasp time in the demos (`python -m zero_shot.check_fk`) | −0.8 ± 1.0 cm; 5 clusters = the 5 marks |
| IK | table targets | 0.1 mm error, gripper straight down; 4–6 mm when hovering 10 cm above (wrist at its 95° limit) |
| toy mapping `H_pixel_to_table` | 35 pairs, 25 RANSAC inliers | pixel → table error median 0.6 cm, max 1.4 cm |
| box mapping `H_pixel_to_place` | release pairs added | median 1.3 cm at the release point |
| VLM pointing at the toy | first frame of the 50 demos vs. the colour-detected centre | 50/50 answered; error median 0.5 cm, 90th percentile 1.0 cm, max 1.5 cm |
| VLM pointing at the box centre | same | always inside the box, varying by ~3 cm front to back ("centre" is vague in itself); fine for dropping into the box |

⇒ **Perception and geometry are not the bottleneck** (about 1 cm error, far smaller than the toy). On the robot, success will mostly depend on whether the fixed grasp holds a soft toy and whether the toy gets bumped during open-loop execution.

## 7. Known limitations (state them in the report)

- **Overhead RGB only, no depth**: objects are assumed to lie on the table, grasp height is fixed. True for our task; false for stacked or raised objects.
- **One fixed grasp**: only "straight down from above". For a soft plush toy it may be worse than a learned policy. That's exactly a comparison worth reporting.
- **Open loop**: it doesn't look while moving, so it can't tell if the toy got bumped or slipped.
- **Depends on the base position**: every target is computed in the base frame, and the mappings were calibrated for where the base was. Move the base 1 cm and the gripper lands "precisely" 1 cm off.

## 8. Running and evaluating

```powershell
python -m zero_shot.run_grasp --dry-run                         # photo, pointing and planning only; the arm doesn't move
python -m zero_shot.run_grasp --speed 15 --open 60              # real robot, one automatic attempt
python -m zero_shot.run_grasp --trials 40 --name zero_shot      # evaluation: enter the result after each attempt
```

Each attempt leaves in `outputs/zero_shot/<name>/`: `attempts.csv` (pixels, table coordinates, timing, result, plus the **measured** joint angles and tip height at the bottom and after closing),
a photo annotated with the VLM's points, and overhead + wrist photos at three moments ("above the toy / lowest point / after closing").

Tunable: `--open` (opening 0–100; the demos average ~39), `--close-early` (start closing this many cm above the grasp height; helps with soft toys), `--grasp-z` (grasp height, cm), `--speed` (°/s).

Evaluation is **exactly the same** as for the other policies: 5 spots × 8 = 40 trials, the same success / partial / failure criteria ([08_evaluation.md](08_evaluation.md)).
Additionally record VLM time and motion time per attempt, and attribute failures to: wrong point / no IK solution / missed grasp / dropped while carrying / placed off target.
