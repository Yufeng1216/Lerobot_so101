# 09 · Training-free baseline: a VLM agent (Claude Haiku 5.5) + inverse kinematics

**English** · [简体中文](09_zero_shot_vlm.zh-CN.md)

PS: status: calibration, gripper measurement and the agent are done; prompt and settings are frozen (commit tagged `vlm-agent-frozen`), the 40 trials are running.

## 1. Why this experiment

The four policies so far (ACT, Diffusion, SmolVLA, π₀.₅) are all **trained on our 50 demos**: they learn "image → action".
The other route is **no training at all**: a general vision-language model (VLM) that has never seen robot data looks at the cameras and decides,
step by step, how to do the task. Putting it on **the same task with the same evaluation protocol** answers a practical question:

> How much do 50 human demos and training actually add over "an off-the-shelf model that just decides"? And where does each one fail?

To keep the comparison honest, nothing in this baseline comes from the demos. The camera calibration, the heights and the gripper size are measured on the setup,
the IK starts from an even grid of poses, and every setting is frozen before the 40 trials.

## 2. Design

### 2.1 Why the model doesn't output joint angles

An untrained model doesn't know this arm's kinematics. If it had to output joint angles it would inch forward look-move-look for dozens of steps:
a long-horizon problem that needs a very large model. With a Cartesian interface the same task takes about 5–10 decisions
(above the toy, down, close, up, over the box, release), which a small, cheap model can handle.

IK plays the role of a driver here (like a car's steering turning "go left" into a wheel angle); it makes no task decisions.
Published training-free systems are built the same way (Code as Policies, VoxPoser, MOKA, ReKep, Gemini Robotics-ER): the large model decides, a generic Cartesian / IK layer executes.

| | Who decides |
|---|---|
| where to grasp, how wide to open, how high to lift, the path, when to release, whether to retry, when to give up | **the VLM** |
| turning "gripper to (x, y, z)" into joint angles | IK (§3.6) |
| workspace, never into the table, joint speed limit | safety layer (§3.7) |

### 2.2 Tools

| Tool | What it does |
|---|---|
| `look()` | returns the overhead and wrist images and the state |
| `move_to(x, y, z)` | moves the gripper in a straight line; (x, y, z) is the point **midway between the fingertips**, in table cm, z = height above the table |
| `gripper(opening)` | 0 = closed … 100 = fully open |
| `done(success, reason)` | ends the attempt with the model's own verdict |

`move_to` and `gripper` return the new images and state right away, so **one model call = one action**: the loop is closed, and the model sees every result before deciding the next step.

### 2.3 What the model sees

- **Overhead image with a 5 cm grid** in table coordinates (cm), so the model reads positions off the picture instead of guessing pixel scales.
  A red dot marks the table point under the gripper, a short red line joins the two fingertips.
- **Wrist image**: the toy between (or beside) the fingers, up close.
- **State text**: gripper position, and the gripper opening both commanded and measured.
  If the fingers closed on the toy, the measured opening stays well above the command (the toy is in the way); if they close to ~1, they missed.
  That one number is how the model knows whether it is holding anything.
- **Blocked moves**: if the gripper ends more than 2 cm from where it was sent (the servos stop when a finger or the held toy hits the box wall), the result says so.

The prompt contains only facts: the task, the coordinate convention, the gripper's layout, sizes measured with a ruler (toy 3 cm tall and 6–7 cm wide, so a held toy hangs up to ~3 cm below the fingertips; box walls 8 cm;
fingertip gap at each opening) and what each tool does. **How to grasp is not written anywhere**; the model plans that itself.

### 2.4 Model and cost

Claude Haiku 5.5 through the Anthropic API (`claude-haiku-5-5`), adaptive thinking at effort `medium`, called with the official Python SDK; $0.10 / $0.50 per million input / output tokens.
Each call carries two images (~800 tokens). A 30-action attempt cost **about $0.02**, so 40 attempts are well under a dollar.
The key is a user-level environment variable `ANTHROPIC_API_KEY` set by the user; it never appears in the repo, in logs or in the chat.

## 3. Building blocks

The model reasons in table coordinates, so we need pixel ↔ table (§3.1–3.3), the gripper's geometry (§3.4), and FK / IK to drive the arm (§3.5–3.6).

### 3.1 From pixels to the table: homography

The camera gives a 2D image; the arm needs 3D coordinates. In general that takes depth, but our case is special: **the toy and the box stand on the table**, and the table is a plane.
A plane seen through a perspective projection onto another plane (the image) is mapped by exactly one 3×3 matrix $H$, in homogeneous coordinates:

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix},\qquad x = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}},\quad y = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

$H$ is defined up to scale, so it has 8 degrees of freedom. Each "pixel ↔ table point" pair gives 2 equations, so **4 pairs solve it**; more pairs go into least squares.
The $h_{31}, h_{32}$ in the denominator are the perspective effect (near things look bigger). So **no depth camera** is needed.

### 3.2 Lens distortion

Limit: a homography assumes a pinhole camera, where straight lines stay straight. Our overhead camera is wide-angle: the table edge visibly bends near the image border (barrel distortion).
With a plain homography, points away from the calibration marks were off by **1–3 cm**. So: undistort first, then apply $H$.

The standard radial model, around the image centre $c$ (in units of $f$ = 400 px):

$$u_d = c + (u - c)\,\big(1 + k_1 r^2 + k_2 r^4\big),\qquad r = \frac{\lVert u - c \rVert}{f}$$

$u$ is where an ideal pinhole camera would see the point, $u_d$ where ours does. $k_1 < 0$ is barrel distortion. We need the inverse (pixel → ideal pixel), which has no closed form,
so it is solved by fixed-point iteration ($u \leftarrow c + (u_d - c) / (1 + k_1 r^2 + k_2 r^4)$, 30 rounds). $k_1, k_2$ and $H$ are fitted **together** from the calibration points:
for each candidate $(k_1, k_2)$, undistort, fit $H$ by least squares, measure the error; Nelder–Mead searches over $(k_1, k_2)$. Result: $k_1 = -0.47$, $k_2 = 0.28$.

### 3.3 Touch calibration (no demos)

1. Put small dark tape squares on the table where the toy and the box go, spread out, not in a line, **all visible to the overhead camera** (one hidden behind the box wall is useless).
2. Take a photo with the arm out of the way; the script finds each square as a dark blob against the local background (pixel = blob centroid).
3. Teleoperate the follower and touch the centre of each square with the closed fingertips; at each touch the script records the joint angles, and FK (§3.5) gives the tip position.

Each square thus gives one "pixel ↔ table point" pair. We touched 16; 14 were used (one square was grey tape the detector missed; at one, deep inside the box with the arm fully stretched, the touch was 3 cm off).

| | |
|---|---|
| fit error (median) | ~6 mm |
| **leave-one-out error** (fit without a point, predict it; median) | **8.3 mm** |
| table height in the arm frame (median tip z over the touches) | −0.7 cm |

Leave-one-out is the honest number: it is the error at a place the fit has not seen. With only the first 6 squares (all in one corner of the image) it was 1–3 cm, because a homography
is only pinned down where there are points; spreading them over the whole working area fixed that.
The toy is 6–7 cm wide and the fingers open up to 15 cm, so ~1 cm is within the margin, and the wrist camera lets the model correct the rest.

PS: the calibration stays valid only while the camera and the arm's base don't move. The base is clamped; if the camera gets bumped, re-align it (`scripts/align_camera.py`) or calibrate again (~15 minutes).

### 3.4 Gripper geometry

The SO-101 gripper has **one fixed finger and one moving finger**. The point the URDF calls the tip (`gripper_frame_link`) sits on the fixed finger.
Limit: if `move_to` positioned that point, a model aiming at the toy's centre puts the fixed finger *on* the toy, the toy is never between the fingers, and every grasp misses.
So `move_to` positions the **midpoint between the fingertips at the current opening** instead:

- **Gap** between the fingertips, measured with a ruler at openings 0 / 25 / 50 / 75 / 100: 0 / 4.5 / 9 / 12.7 / 15 cm (linear interpolation between them).
- **Direction** of the moving finger: from the URDF, the moving jaw sits on the $+x$ side of `gripper_link`, and the tip frame is that frame turned 180° about $y$, so the direction is $-x$ of the tip frame, projected on the table.
- midpoint = fixed fingertip + direction × gap / 2. Since the direction depends on the final pose, the target is converted and solved twice.

### 3.5 Forward kinematics (FK)

The arm is a chain of "link + joint". The URDF gives each joint's fixed offset (translation + rotation) from its parent link and its rotation axis.
So the tip's pose in the base frame is a product of 4×4 homogeneous transforms:

$$T_{\text{tip}}(q) = \prod_{i=1}^{5} \Big( T_i^{\text{fixed}} \cdot \mathrm{Rot}(\text{axis}_i,\ q_i) \Big) \cdot T^{\text{tip}}$$

The top-left 3×3 of $T$ is the orientation, the top-right 3×1 the position.

PS: in LeRobot 0.7 the SO-101 joint readings are in degrees with 0° = middle of the calibrated range, the same zero as the official `so101_new_calib.urdf`, so they go straight in.
The touch calibration doubles as a check: the 11 touches on the bare table all come out at the same height, **−0.8 ± 0.2 cm**, across the whole working area.
With a wrong zero or a flipped axis, the height would swing by centimetres from one side to the other.

### 3.6 Inverse kinematics (IK): damped least squares

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
- High up (above ~13 cm) a 5-joint arm can't keep the gripper vertical everywhere; then $w$ drops to 0.01 and the gripper may tilt a little.
- **Seeds**: from the folded start pose shoulder_lift sits at its −100° limit and the solver gets stuck. So it also starts from an **even grid** over the joint ranges
  (shoulder_lift and elbow_flex at −60 / −20 / 20 / 60°, wrist_flex at 30 / 70°, shoulder_pan pointed at the target) and keeps the best; among equally good answers, the one closest to the current pose.

### 3.7 Motion and safety layer

The model can't bypass any of this; a refused move comes back to it as an error message, and it picks another.

- **Workspace**: x 5–38 cm, y −20–33 cm, z 0–25 cm (the touch-calibrated area).
- **Straight lines**: the path is solved every 1 cm along the line, each point warm-started from the previous one, so the gripper really moves straight (e.g. straight down onto the toy).
  If that fails (e.g. leaving the folded home pose), the target is solved from the seeds and reached by joint interpolation; if that would dip towards the table, it goes straight up to 15 cm first.
- **Checked before moving**: forward kinematics every 2° along the commanded motion; the tip must stay above the table.
- **Speed**: every joint at most 45°/s.
- **Wait until arrived**: the servos trail the commands, so after each motion the script waits until the joints are within 1.5° of the target (at most 2 s) before taking pictures; otherwise the model would judge a pose the arm hasn't reached yet.
- **Home** = the folded pose the arm is in when the script starts (read from the robot, not from the demos). After each attempt: straight up to 15 cm, then home.

## 4. Running it

Everything from the repo root, in the `lerobot` conda env.

```powershell
python zero_shot/calibrate_touch.py                                  # touch the tape squares (§3.3)
python zero_shot/fit_touch_calib.py outputs/zero_shot/calib_touch/<time>   # -> zero_shot/camera_calib.json
python -m zero_shot.measure_gripper                                  # fingertip gap at 0..100 (§3.4)
python -m zero_shot.agent --dry-run --frame data/Yufeng1216/rollout_eval_pi05:0   # no robot: recorded frame, moves only planned
python -m zero_shot.agent --trials 1 --step                          # real arm, Enter before every motion
python -m zero_shot.agent --trials 5 --name vlm_agent                # one evaluation block of 5 (×8); trial numbers continue
```

`fit_touch_calib.py` needs a `pairs.json` next to the touches (touch number → pixel of its square in `clean.jpg`). Every attempt logs each step's images and a `record.json`
(every action, the model's own verdict, time, tokens, cost) under `outputs/zero_shot/<name>/`.

PS: protocol: the agent calls the API once per action, so one attempt takes about 1–3 minutes, longer than the 30 s limit of the trained policies.
Its limit is therefore **25 model calls** (a call may contain several tool calls, e.g. move then close) instead of 30 seconds, and the time is recorded; this is the one place it differs from [08_evaluation.md](08_evaluation.md).
