# 03 · Recording the dataset

**English** · [简体中文](03_data_collection.zh-CN.md)

Goal: teleoperate the task with the leader arm, record 50 demonstrations (episodes), and upload them to the Hugging Face Hub.
**All four policies share this one dataset**, so its quality decides every result after it.

## 1. What actually gets recorded

`lerobot-record` runs a 30 Hz loop. Every frame (1/30 s) it stores:

| Key | Shape | Meaning | Role in training |
|---|---|---|---|
| `observation.images.top` | 480×640×3 | overhead camera | input |
| `observation.images.wrist` | 480×640×3 | wrist camera | input |
| `observation.state` | 6 | the **follower's** 6 joint angles (normalised) | input |
| `action` | 6 | the **leader's** 6 joint angles = the target sent to the follower | label (what we predict) |
| `task` | string | task description, e.g. "Pick up the blue plush toy and place it in the cardboard box." | language input for SmolVLA / π₀.₅ |

In deep-learning terms, **this is just an (x, y) dataset**: x = images + state (+ language), y = action. Behavior cloning learns the mapping x → y.
PS: `action` comes from the leader and `state` from the follower. So what the policy learns is "how a human moves given this view".

## 2. What the dataset looks like on disk (LeRobotDataset v3)

```
data\Yufeng1216\so101_pick_place\        (set by scripts/env.ps1; not in git, a copy is on the Hub)
├── meta/
│   ├── info.json        the dataset's "spec sheet": fps, shape and dtype of every key, total frames, total episodes
│   ├── stats.json       per-key statistics (mean/std/min/max/quantiles), used for normalisation in training
│   ├── tasks.parquet    table of task strings
│   └── episodes/        per-episode index: start/end frame, length, ...
├── data/chunk-000/file-000.parquet     state, action and timestamps of every frame (episodes concatenated in one table)
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4     camera frames compressed as video (tens of times smaller than PNGs)
    └── observation.images.wrist/chunk-000/file-000.mp4
```

During training LeRobot decodes the frame at each timestamp from the MP4, lines it up with state/action from the parquet file, and builds a batch.

## 3. Log in to Hugging Face (once)

1. Create a token of type **Write** at <https://huggingface.co/settings/tokens>.
2. Run the commands below and paste the token when asked (nothing shows while you paste, that is normal). Answer `n` when asked to store it as a git credential:

```powershell
hf auth login
hf auth whoami
```

If `whoami` prints your user name, you are in. **The token stays on your machine; never write it into any file in the repo.**

## 4. Task and recording protocol

Task: **put the blue plush toy into the cardboard box**.

| Item | Setting |
|---|---|
| Episodes | 50 |
| Toy start positions | 5 spots × 10 each (taped on the table) |
| Box position | fixed (four corners taped) |
| Max length per episode | 30 s (press → to end early when done) |
| Reset time | 25 s (move the toy to the next spot, bring the leader back to the start pose) |
| Start pose | the follower starts every episode in the same "folded" pose |
| Cameras | fixed, never adjusted during recording |

Why this design:

- **Several start positions**: if the toy is always on the same spot, the policy just memorises one trajectory and fails anywhere else. So this is really about covering the input distribution, a kind of data augmentation.
- **Everything else fixed**: keep the problem simple enough that all four policies can learn it; only then is comparing them meaningful.
- **Consistent style**: grasp the same way every time (e.g. from above). Random variation between demos is "label noise" for a unimodal regression model like ACT.
  Diffusion Policy can handle multimodal demos, which is one more point to compare later.
- **Is the information enough?** Could a person do the task looking only at the camera views? If yes, the information is there.

### What one episode contains

One episode = doing the task description **once**, start to finish, in the "put it in" direction only:

```
┌──────────── recording (goes into the dataset) ─────────────────────────────────┐ ┌──── reset (not recorded) ────┐
start pose → above toy → down → grip → lift → above box → release → back to start pose → press →   take the toy out by hand,
                                                                                               put it on the next spot
```

- **Don't take the toy out of the box with the arm**; do it by hand during reset. Otherwise the dataset contains two opposite
  behaviours under the same sentence "place it in the box": the labels contradict each other, and a VLA also learns that language and action don't match.
- **End in the start pose**: every episode starts and ends the same way, so the policy also learns to fold back when done instead of stopping mid-air.

## 5. Teleoperation tips: why grasps keep slipping

The most common problem at first is "it won't hold the toy". Look at the data (gripper 0 = fully closed, 100 = fully open):

```
t= 5.3s  leader(action)= 44.1  follower(state)= 43.4   ← only 44% open
t= 9.3s  leader(action)=  1.4  follower(state)=  8.1   ← leader fully squeezed, follower stops at 8: the toy is held but barely squeezed
```

The reason is that **the follower's gripper force is limited on purpose**: LeRobot caps the gripper servo at 50% torque,
dropping to 25% on overload, so the servo doesn't burn out. A plush toy is soft, gets squeezed only a little, has little
friction, and slips out on lift. Don't raise that limit; fix it with technique:

1. **Open wide, then grasp**: open the trigger to 70–80% so both fingers **wrap around the toy's body**, not just pinch its edge.
2. **Come straight down**: move the gripper above the toy with the wrist pointing down, **then descend vertically**, let the tips touch the table, and close.
   Split it into "align → descend → close → lift", each step a bit slow.
3. **Keep squeezing**: hold the trigger fully in while carrying; let go and the force is gone.
4. **Aim with the wrist view in Rerun**: depth is hard to judge from the side; close when the toy is centred between the fingers in the wrist view.
5. **Practise for 10 minutes first**: just run `.\scripts\robot.ps1 teleop -Cameras`. Every recorded episode is teaching material, hesitation and corrections included.

## 6. Record a test set first (2 episodes, not uploaded)

```powershell
.\scripts\robot.ps1 record -Name so101_pick_toy_test -Episodes 2 -EpisodeTime 20 -ResetTime 10 -NoPush
```

PS: LeRobot refuses to create a dataset whose name already exists, so use a new name to re-record a test set.

Keys while recording (the terminal does not need focus):

| Key | Effect |
|---|---|
| **→** or `n` | end this episode early, save it, go to reset; during reset, end the reset early |
| **←** or `r` | this episode is bad, discard and re-record |
| **Esc** or `q` | stop recording, save and (if enabled) upload |

The computer announces each phase by voice (also printed), so keep the sound on. The timeline (from the `lerobot_record.py` source):

```
🔊 "Recording episode 0"   ← recording starts, move right away
│  [recording] up to episode_time_s: grasp → into the box → back to start pose
├─ press → (or time runs out)
🔊 "Reset the environment"
│  [reset] reset_time_s, not recorded; the leader still drives the follower. Press → to skip the rest
├─ 💾 only now is the previous episode saved
🔊 "Recording episode 1"
```

About re-recording (←):

- **An episode is saved only after its reset ends**, so ← during recording or during reset both discard it.
- ← during recording: stop recording → normal reset → 🔊 "Re-record episode" → record the same index again.
- ← during reset: the reset **ends at once** and re-recording **starts right away**. So reset the scene first, then press it.
- The **last episode of a batch** has no reset; → saves it and ends. If you don't like it, press ← while recording.
- Indices start at 0; with `-Resume` they continue from where they were.

Check two things afterwards:

1. The terminal ends with lines like `Cadence (episode 0): 29.88 Hz vs 30 Hz target`. **The real rate should be close to 30 Hz.**
   Clearly lower (say 20 Hz) means the computer can't keep up: close other programs or lower the resolution.
2. Replay episode 0 and see whether the follower repeats your motion (**the follower moves on its own during replay, stay clear**):

```powershell
lerobot-replay --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower --dataset.repo_id=Yufeng1216/so101_pick_toy_test --dataset.episode=0
```

If the test set looks fine you can delete it (`data\Yufeng1216\so101_pick_toy_test`).

## 7. The real recording (5 batches × 10, uploaded automatically)

### Setting up the scene

1. Mark the box's four corners with **small pieces of light tape** (or tape the box down). It does not move for the whole project.
2. The 5 toy spots: put small marks **right under** the toy (the toy covers them, so the camera barely sees them).
   Keep them within easy reach of the follower, 8–15 cm apart, toy always facing the same way.
3. Once the cameras are fixed, **don't touch them** from recording to the final evaluation, and keep the marks.
   A view that differs between training and testing is a distribution shift and directly costs success rate.
4. Teleoperate one grasp at each spot as a test.

### Recording

10 episodes per batch in the order ①①②②③③④④⑤⑤; 5 batches give 10 per spot. First batch:

```powershell
.\scripts\robot.ps1 record -Name so101_pick_place -Episodes 10 -ResetTime 25
```

Then append each further batch (`-Episodes` is **how many to add this time**, not the total):

```powershell
.\scripts\robot.ps1 record -Name so101_pick_place -Episodes 10 -ResetTime 25 -Resume
```

Rhythm:

- 25 s of reset is only the upper limit: place the toy, bring the leader back, then **press → to skip the rest**.
- Start moving **as soon as** you hear "Recording episode".
- Into the box → back to the start pose → press → **right away**.
- Keep hands and body out of the overhead view while recording.
- After each batch, wait for the prompt to come back (upload finished) before the next one.

### Check after every batch

```powershell
python scripts/inspect_dataset.py Yufeng1216/so101_pick_place --frames
```

For every episode it prints length, idle fraction, wait at the start, wait at the end and maximum gripper opening, and flags suspicious ones
(`NO MOTION`, `slow start`, `idle tail`, ...). `--frames` writes a thumbnail per episode to `outputs/dataset_check/`
(6 frames from each camera), so you see at a glance whether it succeeded.

Or look at it in the browser (both videos and the joint curves in sync):
<https://huggingface.co/spaces/lerobot/visualize_dataset?path=/Yufeng1216/so101_pick_place/episode_0>

Final dataset: 50 / 50 usable episodes, about 11 s on average, 18% idle, none with more than 3 s of waiting at the start.

PS: why does "waiting at the start" matter so much? The policy learns "this view → this action". If demos often idle for a few
seconds at the start, the "correct label" for the start view becomes "don't move", and the deployed policy may just sit in the start pose.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OpenCVCamera(2) failed to set fps=30 (actual_fps=0.0)` | The wrist camera occasionally drops out (its cable gets tugged as the arm moves). Re-plug and run again. `robot.ps1 record` runs `scripts/check_cameras.py` first and won't start if a camera fails |
| `Dataset folder already exists` | A failed run already created the folder. Open `<folder>\meta\info.json`: `total_episodes` 0 means an empty shell, delete it and start over; if it has data, add `-Resume` to continue |

PS: why the empty shell? `lerobot-record` creates the dataset first and connects cameras and arm second, so a failed connection leaves a folder with only `meta/info.json`.

## 9. Checklist for good data

- [ ] every episode **succeeds** (re-record failures with ←, don't keep them)
- [ ] smooth motion, no long pauses (too many pauses and the policy learns to stand still)
- [ ] the toy is visible in both cameras
- [ ] roughly the same number of episodes per start spot
- [ ] lighting and background stay the same throughout
