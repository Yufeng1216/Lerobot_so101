"""How accurately does the VLM point at the toy? Measured offline on the 50 recorded demos.

For each demo's first overhead frame: ground truth = centre of the blue blob (calibrate_from_demos),
prediction = Qwen3-VL's point for "the blue plush toy". Reports the pixel error and, through the
homography, the error on the table in cm -- the number that decides whether a grasp can work.
Also asks for the box once per frame and reports how much those answers scatter.

Writes outputs/zero_shot/pointing_eval.csv and pointing_examples.jpg.

Usage:  python -m zero_shot.eval_pointing [--episodes 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zero_shot.calibrate_from_demos import toy_pixel
from zero_shot.pointing import Pointer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from align_camera import reference_frame  # noqa: E402

OUT = ROOT / "outputs" / "zero_shot"
H = np.array(json.loads(Path(__file__).with_name("homography.json").read_text())["H_pixel_to_table"])


def to_table(uv) -> np.ndarray:
    return cv2.perspectiveTransform(np.array([[uv]], float), H)[0, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = Pointer()
    rows, tiles = [], []
    for ep in range(args.episodes):
        bgr = reference_frame("Yufeng1216/so101_pick_place", "top", ep)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = toy_pixel(bgr)
        toy, toy_raw, dt = pointer.point(rgb, "the blue plush toy", "toy")
        box, _, _ = pointer.point(rgb, "the center of the cardboard box", "box")
        row = {"episode": ep, "seconds": round(dt, 2), "answer": toy_raw.strip()}
        if gt and toy:
            row.update(err_px=float(np.hypot(toy[0] - gt[0], toy[1] - gt[1])),
                       err_cm=float(np.linalg.norm(to_table(toy) - to_table(gt)) * 100))
        if box:
            row.update(box_u=box[0], box_v=box[1])
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items() if k != "answer"}, flush=True)
        if ep % 10 == 0:  # a few examples: green = colour ground truth, red = VLM toy, yellow = VLM box
            img = bgr.copy()
            for p, c in ((gt, (0, 255, 0)), (toy, (0, 0, 255)), (box, (0, 255, 255))):
                if p:
                    cv2.drawMarker(img, (int(p[0]), int(p[1])), c, cv2.MARKER_CROSS, 18, 2)
            tiles.append(cv2.resize(img, (320, 240)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "pointing_eval.csv", index=False)
    cv2.imencode(".jpg", np.hstack(tiles))[1].tofile(str(OUT / "pointing_examples.jpg"))
    ok = r.dropna(subset=["err_px"])
    print(f"\ntoy: answered {len(ok)}/{len(r)}; error median {ok.err_px.median():.1f} px / {ok.err_cm.median():.1f} cm, "
          f"90th pct {ok.err_cm.quantile(0.9):.1f} cm, max {ok.err_cm.max():.1f} cm; "
          f"within 2 cm: {(ok.err_cm < 2).mean() * 100:.0f}%")
    if "box_u" in r:
        print(f"box: spread (std) {r.box_u.std():.1f} px x {r.box_v.std():.1f} px")
    print(f"VLM time per query: median {r.seconds.median():.2f} s")


if __name__ == "__main__":
    main()
