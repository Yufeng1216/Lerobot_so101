"""Fit the pixel -> table map from a touch calibration (zero_shot/calibrate_touch.py), no demos involved.

Input: a calib_touch/<time>/ folder with points.json (tip positions by forward kinematics) and
pairs.json ({"<touch index>": [u, v]}: the pixel of the mark that touch went to, read from clean.jpg).

The overhead camera is wide-angle, so straight lines on the table bend near the image edges and a
plain homography leaves ~1-3 cm errors there. The model is therefore: undo radial lens distortion
(two coefficients k1, k2 around the image centre), then a homography to the table plane. Both are
fitted together by least squares; leave-one-out error is printed as the honest accuracy.

    python zero_shot/fit_touch_calib.py outputs/zero_shot/calib_touch/<time>
writes zero_shot/camera_calib.json, read by the VLM agent's tools.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).with_name("camera_calib.json")
CENTER, F = np.array([320.0, 240.0]), 400.0   # distortion is modelled around the image centre, in units of 400 px


def undistort(uv: np.ndarray, k) -> np.ndarray:
    """Distorted pixel -> pixel of an ideal pinhole camera (inverts u_d = u (1 + k1 r^2 + k2 r^4))."""
    x = (np.asarray(uv, float) - CENTER) / F
    xu = x.copy()
    for _ in range(30):
        r2 = (xu ** 2).sum(-1, keepdims=True)
        xu = x / (1 + k[0] * r2 + k[1] * r2 ** 2)
    return xu * F + CENTER


def distort(uv: np.ndarray, k) -> np.ndarray:
    x = (np.asarray(uv, float) - CENTER) / F
    r2 = (x ** 2).sum(-1, keepdims=True)
    return x * (1 + k[0] * r2 + k[1] * r2 ** 2) * F + CENTER


def fit(uv: np.ndarray, xy: np.ndarray, k0=(0.0, 0.0)):
    def cost(k):
        H, _ = cv2.findHomography(undistort(uv, k), xy, 0)
        return ((cv2.perspectiveTransform(undistort(uv, k)[None], H)[0] - xy) ** 2).sum(1).mean()

    k = minimize(cost, np.array(k0), method="Nelder-Mead").x
    H, _ = cv2.findHomography(undistort(uv, k), xy, 0)
    return k, H


def pixel_to_table(uv, k, H) -> np.ndarray:
    return cv2.perspectiveTransform(undistort(np.atleast_2d(uv), k)[None], H)[0]


def main() -> None:
    folder = Path(sys.argv[1])
    pts = {p["index"]: p for p in json.loads((folder / "points.json").read_text())["points"]}
    pairs = {int(i): uv for i, uv in json.loads((folder / "pairs.json").read_text()).items()}
    idx = sorted(pairs)
    uv = np.array([pairs[i] for i in idx], float)
    xyz = np.array([pts[i]["tip_xyz_m"] for i in idx])
    xy = xyz[:, :2]

    k, H = fit(uv, xy)
    err = np.linalg.norm(pixel_to_table(uv, k, H) - xy, axis=1) * 1000
    loo = []
    for t in range(len(idx)):
        keep = np.arange(len(idx)) != t
        kt, Ht = fit(uv[keep], xy[keep], k)
        loo.append(np.linalg.norm(pixel_to_table(uv[t], kt, Ht)[0] - xy[t]) * 1000)
    print("k1, k2 =", np.round(k, 3))
    print("fit error (mm):", dict(zip(idx, np.round(err, 1))))
    print("leave-one-out (mm):", dict(zip(idx, np.round(loo, 1))), f"median {np.median(loo):.1f}")
    table_z = float(np.median(xyz[:, 2]))
    print(f"table height in the arm frame: {table_z * 100:.1f} cm (median tip z when touching)")

    OUT.write_text(json.dumps({
        "source": str(folder.as_posix()),
        "distortion": {"center": CENTER.tolist(), "f": F, "k": k.tolist()},
        "H_undistorted_pixel_to_table": H.tolist(),
        "table_z_m": round(table_z, 4),
        "fit_error_mm_median": round(float(np.median(err)), 1),
        "leave_one_out_mm_median": round(float(np.median(loo)), 1),
    }, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
