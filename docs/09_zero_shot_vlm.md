# 09 · Training-free baseline: a VLM agent (Claude Haiku 5.5) + inverse kinematics

**English** · [简体中文](09_zero_shot_vlm.zh-CN.md)

PS: status: design and building blocks ready; the agent loop, a demo-free calibration and the evaluation are next.

## 1. Why this experiment

The four policies so far (ACT, Diffusion, SmolVLA, π₀.₅) are all **trained on our 50 demos**: they learn "image → action".
The other route is **no training at all**: a general vision-language model (VLM) that has never seen robot data looks at the cameras and decides,
step by step, how to do the task. Putting it on **the same task with the same evaluation protocol** answers a practical question:

> How much do 50 human demos and training actually add over "an off-the-shelf model that just decides"? And where does each one fail?

To keep the comparison honest, nothing in this baseline is taken from the demos: calibration and heights are measured on the setup, and every setting is frozen before the 40 trials.

## 2. Design

### 2.1 Why the model doesn't output joint angles

An untrained model doesn't know this arm's kinematics, so if it had to output joint angles it would inch forward look-move-look for dozens of steps:
a long-horizon problem that needs a very large model. With a Cartesian interface the same task takes 5–10 decisions
(above the toy, down, close, up, over the box, release), which a small, cheap model can handle.

IK plays the role of a driver here (like a car's steering turning "go left" into a wheel angle); it makes no task decisions.
This is also how published training-free systems are built (Code as Policies, VoxPoser, MOKA, ReKep, Gemini Robotics-ER): the large model decides, a generic Cartesian / IK layer executes.

| | Who decides |
|---|---|
| where to grasp, how high to lift, the path, when to release, whether to retry | **the VLM** |
| turning "gripper to (x, y, z)" into joint angles | IK (§3.4) |
| joint limits, speed limit, never below the table | safety checks |

### 2.2 Tools given to the model

| Tool | What it does |
|---|---|
| `look()` | returns the overhead and wrist images and the current gripper position (x, y, z) |
| `move_to(x, y, z)` | moves the gripper tip there, pointing down (IK + joint-speed limit + path check) |
| `gripper(opening)` | opens (0–100) or closes the gripper |
| `done(result)` | ends the attempt |

The loop is closed: after every move the model looks again, so it can correct a missed grasp or a slipping toy.

### 2.3 Model and cost

Claude Haiku 5.5 through the Anthropic API: it reads images and calls tools; $0.10 / $0.50 per million input / output tokens.
About ten calls per attempt, so 40 attempts cost well under a dollar. The account and key are set up by the user (`ant auth login` or `ANTHROPIC_API_KEY`) and never written into the repo.

## 3. Building blocks

The model reasons in table coordinates (centimetres in the arm's base frame), so it needs a pixel → table mapping, and `move_to` needs FK and IK.

### 3.1 From pixels to the table: homography

The camera gives a 2D image; the arm needs 3D coordinates. In general that takes depth, but our case is special: **the toy and the box stand on the table**, and the table is a plane.
A plane seen through a perspective projection onto another plane (the image) is mapped by exactly one 3×3 matrix $H$, in homogeneous coordinates:

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix},\qquad x = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}},\quad y = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

$H$ is defined up to scale, so it has 8 degrees of freedom. Each "pixel ↔ table point" pair gives 2 equations, so **4 pairs solve it**; more pairs go into least squares.
The $h_{31}, h_{32}$ in the denominator are the perspective effect (near things look bigger). So **no depth camera** and no camera intrinsics are needed.

### 3.2 Calibration without the demos

To find $H$, touch 6–8 points spread over the table with the gripper tip (teleoperated, or by moving the arm by hand with torque off), read the tip position by FK (§3.3),
and mark the same points in the overhead image. That is a standard camera–robot calibration, done once for the setup, with no task demonstrations involved.
The heights the model needs (box wall, toy) are measured with a ruler.

### 3.3 Forward kinematics (FK)

The arm is a chain of "link + joint". The URDF gives each joint's fixed offset (translation + rotation) from its parent link and its rotation axis.
So the gripper tip's pose in the base frame is a product of 4×4 homogeneous transforms:

$$T_{\text{tip}}(q) = \prod_{i=1}^{5} \Big( T_i^{\text{fixed}} \cdot \mathrm{Rot}(\text{axis}_i,\ q_i) \Big) \cdot T^{\text{tip}}$$

The top-left 3×3 of $T$ is the orientation, the top-right 3×1 the position.

PS: in LeRobot 0.7 the SO-101 joint readings are in degrees with 0° = middle of the calibrated range, the same zero as the official `so101_new_calib.urdf`, so they go straight in.
We checked against the demos: over 35 table grasps, FK puts the gripper tip at **−0.8 ± 1.0 cm** height (tips pressing slightly into the soft table mat), and the grasp positions form 5 clusters, exactly the 5 taped marks.
With a wrong zero or direction these numbers would be all over the place.

### 3.4 Inverse kinematics (IK): damped least squares

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
- The initial guess matters: from the folded start pose shoulder_lift sits right at its −100° limit and the solver gets stuck. So several typical grasp poses from the demos are used as seeds, keeping the best result. (Today these seeds are typical grasp poses from the demos; for this baseline they will be replaced by seeds spread evenly over the joint ranges, so nothing comes from the demos.)

Result: **0.1 mm** error on table targets, gripper pointing straight down.

## 4. Plan

1. Calibrate as in §3.2; measure the box and toy heights.
2. Write the agent loop and test it offline on recorded frames (no arm).
3. A few real attempts, then freeze the prompt and settings and run the 40-trial protocol of [08_evaluation.md](08_evaluation.md).
