"""List the serial ports and cameras this computer can see.

Read-only: it opens nothing on the serial ports and only grabs one frame
from each camera, saved under outputs/hardware_check/ so you can see which
index is which physical camera.

Usage (inside the `lerobot` conda env):
    python scripts/find_hardware.py
    python scripts/find_hardware.py --max-cameras 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from serial.tools import list_ports

# USB-to-serial chips commonly found on SO-100/SO-101 controller boards.
KNOWN_CHIPS = {
    (0x1A86, 0x55D3): "WCH CH343 (Waveshare / Feetech bus servo board)",
    (0x1A86, 0x7523): "WCH CH340",
    (0x1A86, 0x55D4): "WCH CH9102",
    (0x10C4, 0xEA60): "Silicon Labs CP210x",
    (0x0403, 0x6001): "FTDI FT232",
}


def list_serial_ports() -> None:
    print("=== Serial ports ===")
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("  (none found - is the USB cable plugged in and the driver installed?)")
    for p in ports:
        chip = KNOWN_CHIPS.get((p.vid, p.pid), "unknown chip") if p.vid else "no USB VID/PID"
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid else "----:----"
        print(f"  {p.device:<8} {vidpid}  serial={p.serial_number or '-':<12} {chip}  [{p.description}]")
    print()


def list_cameras(max_index: int, out_dir: Path) -> None:
    print(f"=== OpenCV cameras (index 0..{max_index - 1}) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    # On Windows the default backend (MSMF) is what LeRobot's OpenCVCamera uses too.
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        # The first frames are often black while auto-exposure settles: discard them.
        for _ in range(15):
            ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not ok:
            print(f"  index {idx}: opened but returned no frame")
            continue
        path = out_dir / f"camera_{idx}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  index {idx}: {w}x{h} @ {fps:.0f} fps  -> snapshot {path}")
    print("\nOpen the snapshots to see which index is the overhead camera and which is the laptop webcam.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-cameras", type=int, default=5, help="probe camera indices 0..N-1")
    parser.add_argument("--out", type=Path, default=Path("outputs/hardware_check"))
    args = parser.parse_args()

    list_serial_ports()
    list_cameras(args.max_cameras, args.out)


if __name__ == "__main__":
    main()
