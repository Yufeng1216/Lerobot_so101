"""Forward and inverse kinematics of the SO-101 arm, read from the official URDF.

Joint angles use LeRobot's convention for the SO-101 follower (`use_degrees=True`, the default):
degrees, 0 = middle of the calibrated range. The official `so101_new_calib.urdf` is built for the
same zero, so the angles go straight into the URDF chain (verified against the recorded demos with
`python -m zero_shot.check_fk`).

Coordinates: metres, in the URDF base frame (`base_link`): origin under the arm's base, z up.

Why our own IK instead of a library: the arm has only 5 joints, and the grasp needs just two things
-- the gripper tip at (x, y, z) and the gripper pointing straight down. A damped-least-squares
solver over the URDF chain does that in ~60 lines of numpy, with no extra dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URDF = Path(__file__).with_name("so101_new_calib.urdf")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LINK = "gripper_frame_link"  # a point between the fingertips


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation about a unit axis (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


@dataclass
class Joint:
    name: str
    origin: np.ndarray       # 4x4 fixed transform parent -> joint frame
    axis: np.ndarray         # rotation axis in the joint frame
    movable: bool
    lower: float = -np.inf   # limits in radians
    upper: float = np.inf


def load_chain(urdf: Path = URDF, tip: str = TIP_LINK) -> list[Joint]:
    """Joints from base_link to `tip`, in order."""
    root = ET.parse(urdf).getroot()
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rpy_matrix(*rpy), xyz
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        lim = j.find("limit")
        joint = Joint(j.get("name"), T, axis, j.get("type") == "revolute")
        if lim is not None:
            joint.lower, joint.upper = float(lim.get("lower")), float(lim.get("upper"))
        by_child[j.find("child").get("link")] = (j.find("parent").get("link"), joint)
    chain, link = [], tip
    while link in by_child:
        link, joint = by_child[link]
        chain.append(joint)
    return chain[::-1]


CHAIN = load_chain()


def fk(q_deg: dict[str, float] | np.ndarray) -> np.ndarray:
    """4x4 pose of the gripper tip in the base frame. `q_deg`: the 5 arm joints in degrees."""
    q = q_deg if isinstance(q_deg, np.ndarray) else np.array([q_deg[n] for n in ARM_JOINTS])
    angles = dict(zip(ARM_JOINTS, np.radians(q)))
    T = np.eye(4)
    for j in CHAIN:
        T = T @ j.origin
        if j.movable:
            R = np.eye(4)
            R[:3, :3] = _axis_angle(j.axis, angles[j.name])
            T = T @ R
    return T


def approach_axis(T: np.ndarray) -> np.ndarray:
    """Direction the fingers point, in the base frame (the tip frame's z axis)."""
    return T[:3, 2]


def ik(target_xyz, q0_deg, down=(0.0, 0.0, -1.0), iters: int = 200, damping: float = 1e-3,
       orient_weight: float = 0.05) -> tuple[np.ndarray, float]:
    """Joint angles (deg) putting the tip at `target_xyz` with the fingers pointing along `down`.

    Damped least squares on the residual [position error (m), weighted approach-axis error].
    Units matter for the damping: one degree moves the tip only ~3 mm (Jacobian entries ~3e-3 m/deg),
    so `damping` must be ~1e-3; at 5e-2 it swamped J J^T and the solver crawled.
    wrist_roll is not used (it only spins the gripper about its own axis) and stays at q0.
    Joint limits from the URDF are enforced by clipping. Returns (angles, position error in m).
    """
    target, down = np.asarray(target_xyz, float), np.asarray(down, float)
    q = np.array(q0_deg, float)
    lo = np.degrees([j.lower for j in CHAIN if j.movable])[:5]
    hi = np.degrees([j.upper for j in CHAIN if j.movable])[:5]
    active = [0, 1, 2, 3]

    def residual(qq):
        T = fk(qq)
        return np.concatenate([T[:3, 3] - target, orient_weight * (approach_axis(T) - down)])

    for _ in range(iters):
        r = residual(q)
        if np.linalg.norm(r[:3]) < 1e-4 and np.linalg.norm(r[3:]) < 1e-3:
            break
        J = np.zeros((6, len(active)))
        for k, i in enumerate(active):  # numerical Jacobian, 0.01 deg steps
            dq = np.zeros(5)
            dq[i] = 0.01
            J[:, k] = (residual(q + dq) - r) / 0.01
        step = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), -r)
        q[active] = np.clip(q[active] + np.clip(step, -5, 5), lo[active], hi[active])
    return q, float(np.linalg.norm(fk(q)[:3, 3] - target))


def solve_grasp(target_xyz, wrist_roll: float = -90.0) -> tuple[np.ndarray, float]:
    """IK for a top-down grasp, tried from several starting poses; returns the best (angles, error).

    Starting from the folded rest pose fails: shoulder_lift sits at its -100 deg limit there and the
    solver gets stuck against it. The seeds are typical grasp poses from the recorded demos
    (shoulder_lift 0..40, elbow -20..40, wrist_flex 50..65), with shoulder_pan pointed at the target.
    """
    x, y, _ = target_xyz
    pan = -np.degrees(np.arctan2(y, x - CHAIN[0].origin[0, 3]))  # pan axis sits ~4 cm in front of base origin
    best = None
    for lift, elbow, wrist in [(20, 10, 60), (0, 40, 50), (40, -20, 60), (-30, 70, 60)]:
        q, err = ik(target_xyz, [pan, lift, elbow, wrist, wrist_roll])
        if best is None or err < best[1]:
            best = (q, err)
    return best


if __name__ == "__main__":
    print("chain:", [j.name for j in CHAIN])
    zero = fk(np.zeros(5))
    print("tip at all-zero joints (m):", np.round(zero[:3, 3], 3), " approach:", np.round(approach_axis(zero), 2))
