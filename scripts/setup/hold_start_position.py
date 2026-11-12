#!/usr/bin/env python3
"""Move ARX arm(s) to the saved start pose and hold until interrupted."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from set_up_recording import (
    build_arx_controllers,
    build_qpos_commands,
    configured_controller_gain_snapshot,
    load_arx5_interface,
    load_existing_start_position,
    move_to_saved_start_position,
    resolve_arm_can_interfaces,
    send_hold_commands,
)
from hardware.safe_exit import ARXDamping, damping_exit  # noqa: E402 (path set above)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_POSITION = REPO_ROOT / "calibration" / "start_position.npz"
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def load_selected_start_positions(path, arm_can_interfaces):
    """Load and validate one seven-value saved pose for every selected arm."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Start-position file does not exist: {path}")

    available_qpos, saved_can_interfaces, _ = load_existing_start_position(path)
    missing = [arm for arm in arm_can_interfaces if arm not in available_qpos]
    if missing:
        raise ValueError(
            f"{path} has no saved start position for: {', '.join(missing)}"
        )

    selected_qpos = {}
    for arm, active_can_interface in arm_can_interfaces.items():
        qpos = np.asarray(available_qpos[arm], dtype=np.float64).reshape(-1)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError(
                f"{path} entry {arm}_qpos must contain exactly 7 finite values"
            )
        selected_qpos[arm] = qpos.copy()

        saved_can_interface = saved_can_interfaces.get(arm)
        if (
            saved_can_interface is not None
            and saved_can_interface != active_can_interface
        ):
            print(
                f"[WARN] Saved {arm} pose was recorded for {saved_can_interface}, "
                f"but this run uses {active_can_interface}."
            )

    return path, selected_qpos


def return_arms_home(controllers):
    """Attempt to home every connected arm even if another arm fails."""
    if not controllers:
        return

    print("Returning arm(s) to home before exit...")
    failures = []
    for arm, controller in controllers.items():
        try:
            controller.reset_to_home()
        except Exception as exc:
            failures.append(f"{arm}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))


def run_hold(args):
    """Connect, move to the saved pose, hold it, and always try to home."""
    if not np.isfinite(args.controller_fps) or args.controller_fps <= 0.0:
        raise ValueError("--controller-fps must be positive and finite")
    if not np.isfinite(args.move_duration) or args.move_duration <= 0.0:
        raise ValueError("--move-duration must be positive and finite")

    arm_can_interfaces = resolve_arm_can_interfaces(args)
    start_path, saved_qpos_by_arm = load_selected_start_positions(
        args.start_position,
        arm_can_interfaces,
    )

    print(f"Start position: {start_path}")
    for arm, can_interface in arm_can_interfaces.items():
        print(f"  {arm}: {can_interface}")

    controllers = {}
    failed = False
    try:
        # Ctrl+C or any failure: native damping until 'h', then reset_to_home.
        with damping_exit(lambda: ARXDamping(controllers.values()) if controllers else None):
            arx5 = load_arx5_interface()
            robot_config, controller_config, controllers = build_arx_controllers(
                arx5,
                arm_can_interfaces=arm_can_interfaces,
                controller_fps=args.controller_fps,
            )
            gain_snapshots = {
                arm: configured_controller_gain_snapshot(controller, controller_config)
                for arm, controller in controllers.items()
            }

            print(
                f"Moving arm(s) to the saved start position over "
                f"{args.move_duration:.1f}s..."
            )
            move_to_saved_start_position(
                arx5,
                robot_config,
                controllers,
                saved_qpos_by_arm,
                gain_snapshots,
                duration=args.move_duration,
                controller_fps=args.controller_fps,
            )

            hold_commands = build_qpos_commands(
                arx5,
                robot_config,
                saved_qpos_by_arm,
            )
            send_hold_commands(controllers, hold_commands)
            print("Holding the saved start position. Ctrl+C: damping, then 'h' returns home and exits.")

            hold_period = 1.0 / args.controller_fps
            while True:
                time.sleep(hold_period)
                send_hold_commands(controllers, hold_commands)
    except KeyboardInterrupt:
        failed = True
        print("\nExit requested.")
        return 130
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            try:
                return_arms_home(controllers)
            except Exception as exc:
                print(f"[WARN] Failed to return arm(s) home before exit: {exc}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Move ARX arm(s) to a saved recording start position, hold the arm "
            "joints and grippers there, then return home on exit."
        )
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which ARX arm(s) to hold (default: both)",
    )
    parser.add_argument(
        "--start-position",
        default=str(DEFAULT_START_POSITION),
        help=(
            "Saved .npz pose from set_up_recording.py "
            f"(default: {DEFAULT_START_POSITION})"
        ),
    )
    parser.add_argument(
        "--can-interface",
        default=None,
        help=(
            "CAN override for one arm; with --arm both, use a pair such as "
            "can0,can1"
        ),
    )
    parser.add_argument(
        "--left-can-interface",
        default=None,
        help="Left-arm CAN interface (default: can0)",
    )
    parser.add_argument(
        "--right-can-interface",
        default=None,
        help="Right-arm CAN interface (default: can1)",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=3.0,
        help="Seconds used to move smoothly to the start position (default: 3)",
    )
    parser.add_argument(
        "--controller-fps",
        type=float,
        default=50.0,
        help="Movement and hold-command rate in Hz (default: 50)",
    )
    return parser


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    args = build_parser().parse_args()
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS
    }
    for signum in HANDLED_SIGNALS:
        signal.signal(signum, _raise_keyboard_interrupt)
    try:
        return run_hold(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
