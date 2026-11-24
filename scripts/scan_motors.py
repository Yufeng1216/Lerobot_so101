"""Check that an SO-101 arm's six servos answer on the expected IDs (read-only).

`lerobot-setup-motors` writes each servo's ID (shoulder_pan=1 ... gripper=6) and baud
rate (1 Mbps) into the servo's own EEPROM, so they survive power cycles and changing
computers. This script confirms that, without writing anything to the servos:

  * pings IDs 1-6 at 1 Mbps and reports which answer, with their model number,
  * reads the raw encoder position (0-4095) and supply voltage of each one.

If every ID answers, you can skip `lerobot-setup-motors`. If some are missing, rerun it.

Usage (inside the `lerobot` conda env):
    python scripts/scan_motors.py --port COM3
    python scripts/scan_motors.py --port COM3 --watch   # live positions, Ctrl+C to stop

Move each joint by hand while using --watch: the number that changes tells you which
servo is which, and the gripper should read roughly 2000 +/- 1000 across its range.
"""

from __future__ import annotations

import argparse
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

# Same ID map as lerobot/robots/so_follower/so_follower.py (leader and follower share it).
SO101_MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
STS3215_MODEL_NUMBER = 777


def build_bus(port: str) -> SerialMotorsBus:
    motors = {
        name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100) for name, motor_id in SO101_MOTORS.items()
    }
    return SerialMotorsBus(port=port, motors=motors)


def scan(bus: SerialMotorsBus) -> list[str]:
    print(f"Port {bus.port} @ {bus.default_baudrate} baud\n")
    print(f"  {'joint':<14} {'id':>2}  {'status':<10} {'model':>5}  {'position':>8}  {'voltage':>7}")
    found = []
    for name, motor_id in SO101_MOTORS.items():
        model = bus.ping(name, num_retry=2)
        if model is None:
            print(f"  {name:<14} {motor_id:>2}  {'MISSING':<10}")
            continue
        found.append(name)
        pos = bus.read("Present_Position", name, normalize=False, num_retry=2)
        try:
            volts = bus.read("Present_Voltage", name, normalize=False, num_retry=2) / 10
            volts_str = f"{volts:.1f} V"
        except Exception:  # register naming can differ across firmware; voltage is informational only
            volts_str = "n/a"
        status = "ok" if model == STS3215_MODEL_NUMBER else "odd model"
        print(f"  {name:<14} {motor_id:>2}  {status:<10} {model:>5}  {pos:>8}  {volts_str:>7}")

    print()
    if len(found) == len(SO101_MOTORS):
        print("All 6 servos answered with the right IDs: no need to run lerobot-setup-motors.")
    else:
        print(
            f"{len(SO101_MOTORS) - len(found)} servo(s) did not answer. Check the power supply and the 3-pin "
            "daisy-chain cables first; if they really are unset, run lerobot-setup-motors."
        )
    return found


def watch(bus: SerialMotorsBus, names: list[str]) -> None:
    print("\nLive raw positions (move the joints by hand, Ctrl+C to stop):\n")
    print("  " + "  ".join(f"{n:>13}" for n in names))
    try:
        while True:
            pos = bus.sync_read("Present_Position", names, normalize=False, num_retry=2)
            print("\r  " + "  ".join(f"{pos[n]:>13}" for n in names), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="serial port, e.g. COM3")
    parser.add_argument("--watch", action="store_true", help="stream raw positions after the scan")
    args = parser.parse_args()

    bus = build_bus(args.port)
    # handshake=False: don't raise on a missing motor, we want to report it instead.
    bus.connect(handshake=False)
    try:
        found = scan(bus)
        if args.watch and found:
            watch(bus, found)
    finally:
        # disable_torque=False: leave the servos exactly as they were (read-only script).
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
