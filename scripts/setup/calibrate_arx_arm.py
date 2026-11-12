#!/usr/bin/env python3
"""Interactive ARX arm gain calibration, one motor at a time."""

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.hardware_utils import (
    resolve_arx_model,
    configure_controller,
    require_controller_hz,
    send_joint_target,
    configure_active_gains,
)
from hardware.safe_exit import ARXDamping, damping_exit


DEFAULT_CAN_INTERFACES = {
    "left": "can0",
    "right": "can1",
}

FIELD_ORDER = ("rate", "kp", "kd", "amplitude")
JOINT_MOTOR_COUNT = 6
GRIPPER_MOTOR_INDEX = 6
KEY_BUFFER = bytearray()


def load_arx5_interface():
    """Import the local ARX5 Python binding."""
    sdk_python_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hardware"
        / "arx5-sdk"
        / "python"
    )
    if str(sdk_python_dir) not in sys.path:
        sys.path.insert(0, str(sdk_python_dir))

    import arx5_interface as arx5

    return arx5


def build_controller(arx5, model, can_interface, controller_fps):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(resolve_arx_model(model))
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller",
        robot_config,
    )
    configure_active_gains(controller_config)
    controller_config.gravity_compensation = True
    controller_config.friction_compensation = False
    controller_config.background_send_recv = True
    require_controller_hz(controller_fps)
    configure_controller(controller_config)

    controller = arx5.Arx5JointController(robot_config, controller_config, can_interface)
    return robot_config, controller_config, controller


def clone_gain(arx5, gain, dof):
    new_gain = arx5.Gain(dof)
    new_gain.kp()[:] = np.asarray(gain.kp(), dtype=np.float64)
    new_gain.kd()[:] = np.asarray(gain.kd(), dtype=np.float64)
    new_gain.gripper_kp = float(gain.gripper_kp)
    new_gain.gripper_kd = float(gain.gripper_kd)
    return new_gain


def default_gain(arx5, controller_config, dof):
    gain = arx5.Gain(dof)
    default_kp = np.asarray(controller_config.default_kp, dtype=np.float64)
    default_kd = np.asarray(controller_config.default_kd, dtype=np.float64)
    kp_count = min(len(default_kp), len(gain.kp()))
    kd_count = min(len(default_kd), len(gain.kd()))
    gain.kp()[:kp_count] = default_kp[:kp_count]
    gain.kd()[:kd_count] = default_kd[:kd_count]
    gain.gripper_kp = float(controller_config.default_gripper_kp)
    gain.gripper_kd = float(controller_config.default_gripper_kd)
    return gain


def set_all_gain_values(gain, kp_values, kd_values):
    joint_count = min(JOINT_MOTOR_COUNT, len(gain.kp()))
    gain.kp()[:joint_count] = np.asarray(kp_values[:joint_count], dtype=np.float64)
    gain.kd()[:joint_count] = np.asarray(kd_values[:joint_count], dtype=np.float64)
    gain.gripper_kp = float(kp_values[GRIPPER_MOTOR_INDEX])
    gain.gripper_kd = float(kd_values[GRIPPER_MOTOR_INDEX])


def gain_to_arrays(gain):
    kp = np.asarray(gain.kp(), dtype=np.float64)[:JOINT_MOTOR_COUNT].tolist()
    kd = np.asarray(gain.kd(), dtype=np.float64)[:JOINT_MOTOR_COUNT].tolist()
    kp.append(float(gain.gripper_kp))
    kd.append(float(gain.gripper_kd))
    return kp, kd


def set_passive_mode(controller):
    gain = controller.get_gain()
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)


def make_joint_cmd(arx5, dof, positions, gripper_pos):
    cmd = arx5.JointState(dof)
    cmd.pos()[:] = np.asarray(positions[:dof], dtype=np.float64)
    cmd.vel()[:] = 0.0
    cmd.torque()[:] = 0.0
    cmd.gripper_pos = float(gripper_pos)
    cmd.gripper_vel = 0.0
    cmd.gripper_torque = 0.0
    return cmd


def get_motor_position(joint_state, motor_idx, dof):
    if motor_idx < JOINT_MOTOR_COUNT:
        return float(joint_state.pos()[motor_idx])
    return float(joint_state.gripper_pos)


def get_motor_velocity(joint_state, motor_idx, dof):
    if motor_idx < JOINT_MOTOR_COUNT:
        return float(joint_state.vel()[motor_idx])
    return float(joint_state.gripper_vel)


@contextmanager
def raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key_nonblocking(fd):
    if not KEY_BUFFER and not select.select([fd], [], [], 0.0)[0]:
        return None

    data = b""
    if select.select([fd], [], [], 0.0)[0]:
        data = os.read(fd, 1)
    for _ in range(7):
        if not select.select([fd], [], [], 0.002)[0]:
            break
        data += os.read(fd, 1)
    KEY_BUFFER.extend(data)

    if len(KEY_BUFFER) >= 3 and KEY_BUFFER[:2] == b"\x1b[":
        sequence = bytes(KEY_BUFFER[:3])
        del KEY_BUFFER[:3]
        return {
            b"\x1b[A": "up",
            b"\x1b[B": "down",
            b"\x1b[C": "right",
            b"\x1b[D": "left",
        }.get(sequence, "escape")
    if KEY_BUFFER[:1] == b"\x1b":
        if len(KEY_BUFFER) < 3:
            return None
        del KEY_BUFFER[:1]
        return "escape"

    ch = bytes(KEY_BUFFER[:1])
    del KEY_BUFFER[:1]
    if ch in (b"\r", b"\n"):
        return "enter"
    return ch.decode(errors="ignore").lower()


def adjust_value(value, direction, step, min_value, max_value):
    value += direction * step
    return min(max(value, min_value), max_value)


def format_status(motor_idx, selected_field, rate, kp, kd, amplitude):
    return (
        f"\rMotor {motor_idx} | selected={selected_field:<9} "
        f"rate={rate:5.3f}Hz kp={kp:8.3f} kd={kd:7.3f} amp={amplitude:6.4f} "
        "controls: left/right select, up/down adjust, n next, q quit"
    )


def init_rerun(enabled):
    if not enabled:
        return None
    try:
        import rerun as rr
    except ImportError:
        print("[!] rerun is not installed; continuing without Rerun visualization.")
        return None

    rr.init("arx_gain_calibration", spawn=True)
    return rr


def log_rerun(rr, sample_idx, motor_idx, target, actual, velocity, kp, kd, rate, amplitude):
    if rr is None:
        return

    rr.set_time("sample", sequence=sample_idx)
    prefix = f"motor_{motor_idx}"
    rr.log(f"{prefix}/target", rr.Scalars(target))
    rr.log(f"{prefix}/actual", rr.Scalars(actual))
    rr.log(f"{prefix}/error", rr.Scalars(target - actual))
    rr.log(f"{prefix}/velocity", rr.Scalars(velocity))
    rr.log(f"{prefix}/kp", rr.Scalars(kp))
    rr.log(f"{prefix}/kd", rr.Scalars(kd))
    rr.log(f"{prefix}/rate_hz", rr.Scalars(rate))
    rr.log(f"{prefix}/amplitude", rr.Scalars(amplitude))
    rr.log("active_motor", rr.Scalars(motor_idx))


def safe_home(controller, enabled=True):
    if controller is None or not enabled:
        return
    print("\n[home] Returning arm to home position...")
    controller.reset_to_home()
    print("[home] Arm returned home.")


def prompt_safe_to_start(motor_idx):
    print(f'\nMotor {motor_idx}: gravity compensation/passive mode active.')
    while True:
        answer = input(f"Is this position safe to start moving motor {motor_idx} around in? [y/N] ")
        if answer.strip().lower() == "y":
            return True
        print("Reposition the arm, then press y when safe, or Ctrl+C to quit.")


def apply_active_gain(controller, active_gain, motor_idx, dof, kp, kd):
    if motor_idx < JOINT_MOTOR_COUNT:
        active_gain.kp()[motor_idx] = float(kp)
        active_gain.kd()[motor_idx] = float(kd)
    else:
        active_gain.gripper_kp = float(kp)
        active_gain.gripper_kd = float(kd)
    controller.set_gain(active_gain)


def run_motor_calibration(
    arx5,
    controller,
    dof,
    motor_idx,
    base_gain,
    tuned_kp,
    tuned_kd,
    args,
    rr,
    sample_counter,
):
    set_passive_mode(controller)
    prompt_safe_to_start(motor_idx)

    center_state = controller.get_joint_state()
    center_positions = np.asarray(center_state.pos(), dtype=np.float64).copy()
    center_gripper = float(center_state.gripper_pos)
    center_value = get_motor_position(center_state, motor_idx, dof)

    active_gain = clone_gain(arx5, base_gain, dof)
    set_all_gain_values(active_gain, tuned_kp, tuned_kd)

    rate = float(args.initial_rate)
    amplitude = float(args.gripper_amplitude if motor_idx == GRIPPER_MOTOR_INDEX else args.amplitude)
    kp = float(tuned_kp[motor_idx])
    kd = float(tuned_kd[motor_idx])
    selected_idx = 0
    selected_field = FIELD_ORDER[selected_idx]

    apply_active_gain(controller, active_gain, motor_idx, dof, kp, kd)

    print(
        f"Freezing current pose and oscillating motor {motor_idx}. "
        "Use arrow keys to tune; press n for next motor."
    )
    print(format_status(motor_idx, selected_field, rate, kp, kd, amplitude), end="", flush=True)

    start_time = time.monotonic()
    last_status_time = 0.0
    next_tick = start_time
    dt = 1.0 / args.fps

    with raw_terminal():
        KEY_BUFFER.clear()
        stdin_fd = sys.stdin.fileno()
        while True:
            now = time.monotonic()
            key = read_key_nonblocking(stdin_fd)
            if key == "q":
                return False, sample_counter
            if key == "n":
                tuned_kp[motor_idx] = kp
                tuned_kd[motor_idx] = kd
                print()
                return True, sample_counter
            if key == "right":
                selected_idx = (selected_idx + 1) % len(FIELD_ORDER)
                selected_field = FIELD_ORDER[selected_idx]
            elif key == "left":
                selected_idx = (selected_idx - 1) % len(FIELD_ORDER)
                selected_field = FIELD_ORDER[selected_idx]
            elif key in ("up", "down"):
                direction = 1.0 if key == "up" else -1.0
                if selected_field == "rate":
                    rate = adjust_value(rate, direction, args.rate_step, args.min_rate, args.max_rate)
                elif selected_field == "kp":
                    kp = adjust_value(kp, direction, args.kp_step, args.min_kp, args.max_kp)
                    apply_active_gain(controller, active_gain, motor_idx, dof, kp, kd)
                elif selected_field == "kd":
                    kd = adjust_value(kd, direction, args.kd_step, args.min_kd, args.max_kd)
                    apply_active_gain(controller, active_gain, motor_idx, dof, kp, kd)
                elif selected_field == "amplitude":
                    max_amp = args.max_gripper_amplitude if motor_idx == GRIPPER_MOTOR_INDEX else args.max_amplitude
                    amplitude = adjust_value(amplitude, direction, args.amplitude_step, 0.0, max_amp)
            if key in ("left", "right", "up", "down"):
                print(
                    format_status(motor_idx, selected_field, rate, kp, kd, amplitude),
                    end="",
                    flush=True,
                )
                last_status_time = now

            elapsed = now - start_time
            offset = amplitude * np.sin(2.0 * np.pi * rate * elapsed)
            target_value = center_value + offset

            target_positions = center_positions.copy()
            target_gripper = center_gripper
            if motor_idx < JOINT_MOTOR_COUNT:
                target_positions[motor_idx] = target_value
            else:
                target_gripper = min(max(target_value, 0.0), args.gripper_width)
                target_value = target_gripper

            cmd = make_joint_cmd(arx5, dof, target_positions, target_gripper)
            send_joint_target(controller, cmd, dt)

            joint_state = controller.get_joint_state()
            actual = get_motor_position(joint_state, motor_idx, dof)
            velocity = get_motor_velocity(joint_state, motor_idx, dof)
            log_rerun(rr, sample_counter, motor_idx, target_value, actual, velocity, kp, kd, rate, amplitude)
            sample_counter += 1

            if now - last_status_time >= args.status_period:
                print(
                    format_status(motor_idx, selected_field, rate, kp, kd, amplitude),
                    end="",
                    flush=True,
                )
                last_status_time = now

            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactively tune ARX arm Kp/Kd motor by motor.")
    parser.add_argument("--model", type=resolve_arx_model, default="L5", help="ARX model name; L5 and Trossen_L5 use their respective SDK profiles. Default: L5.")
    parser.add_argument("--arm", choices=["left", "right"], default="left", help="Arm label for default CAN choice.")
    parser.add_argument("--can-interface", default=None, help="CAN interface override. Defaults to can0 for left, can1 for right.")
    parser.add_argument("--fps", type=float, default=50.0, help="Calibration command loop frequency.")
    parser.add_argument("--controller-fps", type=float, default=500.0, help="ARX background controller frequency.")
    parser.add_argument("--motor-start", type=int, default=0, help="First motor index to calibrate.")
    parser.add_argument("--motor-end", type=int, default=6, help="Last motor index to calibrate, inclusive.")
    parser.add_argument("--amplitude", type=float, default=0.05, help="Initial joint oscillation amplitude in radians.")
    parser.add_argument("--gripper-amplitude", type=float, default=0.01, help="Initial gripper oscillation amplitude in meters.")
    parser.add_argument("--max-amplitude", type=float, default=0.2, help="Maximum joint oscillation amplitude in radians.")
    parser.add_argument("--max-gripper-amplitude", type=float, default=0.03, help="Maximum gripper oscillation amplitude in meters.")
    parser.add_argument("--amplitude-step", type=float, default=0.005, help="Amplitude adjustment per keypress.")
    parser.add_argument("--initial-rate", type=float, default=0.15, help="Initial oscillation rate in Hz.")
    parser.add_argument("--min-rate", type=float, default=0.02, help="Minimum oscillation rate in Hz.")
    parser.add_argument("--max-rate", type=float, default=1.0, help="Maximum oscillation rate in Hz.")
    parser.add_argument("--rate-step", type=float, default=0.02, help="Rate adjustment per keypress in Hz.")
    parser.add_argument("--min-kp", type=float, default=0.0, help="Minimum Kp.")
    parser.add_argument("--max-kp", type=float, default=500.0, help="Maximum Kp.")
    parser.add_argument("--kp-step", type=float, default=2.0, help="Kp adjustment per keypress.")
    parser.add_argument("--min-kd", type=float, default=0.0, help="Minimum Kd.")
    parser.add_argument("--max-kd", type=float, default=50.0, help="Maximum Kd.")
    parser.add_argument("--kd-step", type=float, default=0.25, help="Kd adjustment per keypress.")
    parser.add_argument("--initial-kp", type=float, nargs=7, default=None, help="Initial Kp values for motors 0-6.")
    parser.add_argument("--initial-kd", type=float, nargs=7, default=None, help="Initial Kd values for motors 0-6.")
    parser.add_argument("--gripper-width", type=float, default=None, help="Override max gripper position in meters.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for JSON calibration results.")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun visualization.")
    parser.add_argument("--no-home-on-exit", action="store_true", help="Do not call reset_to_home() during cleanup.")
    parser.add_argument("--status-period", type=float, default=0.2, help="Terminal status refresh period in seconds.")
    return parser.parse_args()


def validate_args(args):
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.controller_fps <= 0:
        raise ValueError("--controller-fps must be positive")
    if args.motor_start < 0 or args.motor_end > 6 or args.motor_start > args.motor_end:
        raise ValueError("--motor-start/--motor-end must define an inclusive range within 0..6")
    if args.min_rate <= 0 or args.max_rate < args.min_rate:
        raise ValueError("rate bounds are invalid")
    if args.max_amplitude < 0 or args.max_gripper_amplitude < 0:
        raise ValueError("amplitude bounds must be non-negative")
    if args.max_kp < args.min_kp or args.max_kd < args.min_kd:
        raise ValueError("gain bounds are invalid")


def main():
    args = parse_args()
    validate_args(args)

    np.set_printoptions(precision=4, suppress=True)
    can_interface = args.can_interface or DEFAULT_CAN_INTERFACES[args.arm]

    arx5 = load_arx5_interface()
    robot_config, controller_config, controller = build_controller(
        arx5,
        model=args.model,
        can_interface=can_interface,
        controller_fps=args.controller_fps,
    )
    dof = robot_config.joint_dof
    if dof < JOINT_MOTOR_COUNT:
        raise RuntimeError(f"Expected at least {JOINT_MOTOR_COUNT} arm joints, got dof={dof}")

    if args.gripper_width is None:
        args.gripper_width = float(robot_config.gripper_width)

    base_gain = default_gain(arx5, controller_config, dof)
    tuned_kp, tuned_kd = gain_to_arrays(base_gain)
    if args.initial_kp is not None:
        tuned_kp = [float(value) for value in args.initial_kp]
    if args.initial_kd is not None:
        tuned_kd = [float(value) for value in args.initial_kd]

    rr = init_rerun(not args.no_rerun)
    sample_counter = 0
    completed = False

    print(f"Connected to {args.arm} ARX arm on {can_interface} using model {args.model}.")
    print("Ctrl+C or any failure: damping, then 'h' returns the arm home and quits.")

    failed = False
    try:
        with damping_exit(lambda: ARXDamping([controller])):
            for motor_idx in range(args.motor_start, args.motor_end + 1):
                should_continue, sample_counter = run_motor_calibration(
                    arx5=arx5,
                    controller=controller,
                    dof=dof,
                    motor_idx=motor_idx,
                    base_gain=base_gain,
                    tuned_kp=tuned_kp,
                    tuned_kd=tuned_kd,
                    args=args,
                    rr=rr,
                    sample_counter=sample_counter,
                )
                if not should_continue:
                    break
            else:
                completed = True
    except KeyboardInterrupt:
        failed = True
        print("\nCtrl+C detected.")
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:  # The failure exit already homed (or disabled in place).
            safe_home(controller, enabled=not args.no_home_on_exit)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "arm": args.arm,
        "can_interface": can_interface,
        "motor_start": args.motor_start,
        "motor_end": args.motor_end,
        "completed": completed,
        "kp": tuned_kp,
        "kd": tuned_kd,
    }

    print("\nCalibration result:")
    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved result to {args.output}")

    return 0 if completed else 130


if __name__ == "__main__":
    raise SystemExit(main())
