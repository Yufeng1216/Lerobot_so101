"""Check the forward kinematics against the recorded demonstrations, without the robot.

In every demo the gripper closes on the toy lying on the table. If the URDF, the joint zero and the
joint signs are right, then at that moment:
  * the gripper tip height z is about the same in every episode (the table), and
  * the (x, y) positions form 5 clusters, one per taped start spot (10 demos each).
Wrong signs or offsets show up immediately as scattered heights or a smeared cloud.

Usage:  python -m zero_shot.check_fk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zero_shot.kinematics import ARM_JOINTS, approach_axis, fk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Yufeng1216" / "so101_pick_place"
NAMES = ARM_JOINTS + ["gripper"]


def grasp_frame(gripper: np.ndarray) -> int | None:
    """Index where the gripper first closes after having opened (open > 25, closed < 60% of peak)."""
    opened = np.flatnonzero(gripper > 25)
    if len(opened) == 0:
        return None
    start = opened[0]
    peak = start + int(np.argmax(gripper[start:start + 150]))
    closing = np.flatnonzero(gripper[peak:] < 0.6 * gripper[peak])
    return peak + int(closing[0]) if len(closing) else None


def main() -> None:
    df = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "data").rglob("*.parquet")))
    rows = []
    for ep, g in df.groupby("episode_index"):
        state = np.stack(g["observation.state"].to_numpy())
        i = grasp_frame(state[:, 5])
        if i is None:
            continue
        T = fk(state[i, :5])
        rows.append((ep, i, *T[:3, 3], approach_axis(T)[2]))
    r = pd.DataFrame(rows, columns=["episode", "frame", "x", "y", "z", "approach_z"])
    # A few episodes close the gripper over the box first (z ~ 0.1 m): not a table grasp, skip them.
    table = r[r.z < 0.04].copy()
    print(f"table grasps: {len(table)}/{len(r)} episodes, tip height z = {table.z.mean() * 100:.1f} "
          f"+- {table.z.std() * 100:.1f} cm, approach_z = {table.approach_z.mean():.2f} (-1 = straight down)")
    table["spot"] = kmeans(table[["x", "y"]].to_numpy(), k=5)
    cm = table[["x", "y", "z"]] * 100
    cm["spot"] = table["spot"]
    print("\ngrasp position per start spot (cm, base frame):")
    print(cm.groupby("spot").agg(["mean", "std"]).round(1).assign(n=cm.groupby("spot").size()).to_string())


def kmeans(points: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
    """Plain k-means (farthest-point init), returns a cluster label per point."""
    centers = [points[0]]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(points - c, axis=1) for c in centers], axis=0)
        centers.append(points[np.argmax(d)])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(np.linalg.norm(points[:, None] - centers[None], axis=2), axis=1)
        centers = np.array([points[labels == i].mean(axis=0) for i in range(k)])
    return labels


if __name__ == "__main__":
    main()
