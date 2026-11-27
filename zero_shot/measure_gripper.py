"""Measure the gripper's geometry by hand (no demos): how far apart the fingertips are at each opening.

The SO-101 gripper has one fixed finger and one moving finger. The point the agent's move_to
positions (`gripper_frame_link` in the URDF) sits at the fixed finger, so to get the toy *between*
the fingers the agent has to know where the moving finger is. This script lifts the arm to a free
pose and steps the opening through 0, 25, 50, 75, 100; at each one, measure with a ruler the gap
between the two fingertips (inside edges, cm) and type it in.

    python -m zero_shot.measure_gripper
writes zero_shot/gripper_geometry.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zero_shot.agent import Robot, go_home, plan_line

OUT = Path(__file__).with_name("gripper_geometry.json")


def main() -> None:
    robot = Robot(dry_run=False)
    gaps = {}
    try:
        robot.play(plan_line(robot.read(), (20.0, 0.0, 12.0)))
        for g in (0, 25, 50, 75, 100):
            q = robot.read()
            robot.play([np.r_[q[:5], float(g)]])
            v = input(f"opening {g}: measured {robot.read()[5]:.0f}. Gap between the fingertips (cm)? ")
            gaps[g] = float(v)
    finally:
        q = robot.read()
        robot.play([np.r_[q[:5], robot.home[5]]])
        go_home(robot)
        robot.close()
    OUT.write_text(json.dumps({"fingertip_gap_cm": gaps}, indent=2), encoding="utf-8")
    print("wrote", OUT, gaps)


if __name__ == "__main__":
    main()
