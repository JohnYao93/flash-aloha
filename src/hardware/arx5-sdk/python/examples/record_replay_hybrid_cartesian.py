"""Teach a joint trajectory and replay it in joint or hybrid Cartesian mode.

The arm is gravity compensated with arm gains off during the five-second teach
phase. It records 100 Hz joint-position knots, returns to the recorded starting
joints, and submits those knots as one timestamped trajectory. The native 500 Hz
controller loop interpolates them for either motor-local joint PD or hybrid
Cartesian replay with host Cartesian stiffness and motor-local joint damping.
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
import arx5_interface as arx5


DEFAULT_CARTESIAN_KP = np.array([200, 200, 200, 16, 16, 16], dtype=np.float64)
DEFAULT_MOTOR_KD = np.array([5, 5, 5, 1, 1, 1], dtype=np.float64)
DEFAULT_START_JOINT_TOLERANCE_RAD = 0.10
DEFAULT_START_TRANSLATION_TOLERANCE_MM = 30.0
DEFAULT_START_ROTATION_TOLERANCE_DEG = 10.0
DEFAULT_TRAJECTORY_KNOT_RATE_HZ = 100.0
RETURN_SETTLE_DURATION_S = 1.0
CARTESIAN_AXES = ("x", "y", "z", "rx", "ry", "rz")
JOINT_AXES = tuple(f"J{index}" for index in range(1, 7))
REPLAY_MODES = ("hybrid-cartesian", "joint", "comparison")


def validate_six(values, name: str, *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def build_velocity_limit_report(
    joint_velocities,
    joint_limits,
    cartesian_twists=None,
    cartesian_limits=None,
):
    """Build a serializable per-axis report for trajectory velocity limits."""
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    joint_limits = validate_six(joint_limits, "joint velocity limits")
    if joint_velocities.ndim != 2 or joint_velocities.shape[1] != 6:
        raise ValueError("joint velocities must have shape (samples, 6)")
    if not np.all(np.isfinite(joint_velocities)) or np.any(joint_limits <= 0.0):
        raise ValueError("joint velocities and limits must be finite with positive limits")

    groups = [("Joint", JOINT_AXES, joint_velocities, joint_limits, "rad/s")]
    if cartesian_twists is not None or cartesian_limits is not None:
        if cartesian_twists is None or cartesian_limits is None:
            raise ValueError("Cartesian twists and limits must be provided together")
        cartesian_twists = np.asarray(cartesian_twists, dtype=np.float64)
        cartesian_limits = validate_six(cartesian_limits, "Cartesian velocity limits")
        if cartesian_twists.ndim != 2 or cartesian_twists.shape[1] != 6:
            raise ValueError("Cartesian twists must have shape (samples, 6)")
        if not np.all(np.isfinite(cartesian_twists)) or np.any(cartesian_limits <= 0.0):
            raise ValueError("Cartesian twists and limits must be finite with positive limits")
        groups.append(("Cartesian", CARTESIAN_AXES, cartesian_twists, cartesian_limits, None))

    axes = []
    for group, names, velocities, limits, group_units in groups:
        peaks = np.max(np.abs(velocities), axis=0)
        for index, name in enumerate(names):
            units = group_units or ("m/s" if index < 3 else "rad/s")
            ratio = float(peaks[index] / limits[index])
            axes.append(
                {
                    "group": group,
                    "axis": name.upper(),
                    "units": units,
                    "requested_peak": float(peaks[index]),
                    "limit": float(limits[index]),
                    "ratio": ratio,
                    "exceeded": ratio > 1.0 + 1e-9,
                }
            )
    worst_ratio = max(axis["ratio"] for axis in axes)
    return {
        "exceeded": any(axis["exceeded"] for axis in axes),
        "time_scale": max(1.0, worst_ratio),
        "worst_ratio": worst_ratio,
        "axes": axes,
    }


def format_velocity_limit_report(report):
    """Format an intentionally prominent velocity-limit message."""
    border = "=" * 78
    if report["exceeded"]:
        lines = [
            border,
            "!!! VELOCITY LIMIT EXCEEDED — REPLAY WILL BE SLOWED !!!",
            f"Required time scaling: {report['time_scale']:.3f}x",
            "",
            f"{'Reference axis':<20} {'Requested peak':>16} {'Limit':>16} {'Ratio':>10}",
        ]
        for axis in report["axes"]:
            if not axis["exceeded"]:
                continue
            label = f"{axis['group']} {axis['axis']}"
            requested = f"{axis['requested_peak']:.3f} {axis['units']}"
            limit = f"{axis['limit']:.3f} {axis['units']}"
            lines.append(
                f"{label:<20} {requested:>16} {limit:>16} "
                f"{axis['ratio']:>8.3f}x  EXCEEDED"
            )
        lines.extend(
            [
                "",
                "All replay conditions in this run will use the slowed timeline.",
                border,
            ]
        )
    else:
        lines = [
            border,
            "VELOCITY LIMIT CHECK: PASS — RECORDED TIMING IS PRESERVED",
            f"Highest requested/limit ratio: {report['worst_ratio']:.3f}x",
            border,
        ]
    return "\n".join(lines)


def prepare_recorded_joint_trajectory(times, positions):
    """Preserve recorded position knots and estimate velocities only for limit checks."""
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (len(times), 6) or len(times) < 3:
        raise ValueError("joint recording must contain at least three six-joint samples")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)) or np.any(np.diff(times) < 0.0):
        raise ValueError("joint recording must be finite and have nondecreasing timestamps")
    # The Python recorder can occasionally observe the same coherent controller
    # feedback snapshot twice. Keep one copy rather than treating
    # that expected scheduling race as a malformed trajectory.
    unique = np.r_[True, np.diff(times) > 0.0]
    times = times[unique]
    positions = positions[unique]
    if len(times) < 3:
        raise ValueError("joint recording has fewer than three unique controller samples")
    replay_times = times - times[0]
    replay_positions = positions.copy()
    replay_velocities = np.gradient(
        replay_positions, replay_times, axis=0, edge_order=2
    )
    replay_velocities[0] = 0.0
    replay_velocities[-1] = 0.0
    return replay_times, replay_positions, replay_velocities


def scale_joint_reference_for_limits(times, positions, velocities, joint_velocity_limits):
    """Slow a recorded joint reference uniformly when a velocity limit requires it."""
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    joint_velocity_limits = validate_six(joint_velocity_limits, "joint velocity limits")
    if positions.shape != velocities.shape or positions.shape != (len(times), 6):
        raise ValueError("joint reference arrays have incompatible shapes")
    if np.any(joint_velocity_limits <= 0.0):
        raise ValueError("joint velocity limits must be positive")
    if not all(np.all(np.isfinite(values)) for values in (times, positions, velocities)):
        raise ValueError("joint reference must be finite")

    time_scale = max(1.0, float(np.max(np.abs(velocities) / joint_velocity_limits)))
    scaled_times = times * time_scale
    scaled_velocities = velocities / time_scale
    scaled_velocities[0] = 0.0
    scaled_velocities[-1] = 0.0
    return scaled_times, positions.copy(), scaled_velocities, time_scale


def calculate_fk_reference(
    kinematics,
    times,
    joint_positions,
    joint_velocities,
    joint_velocity_limits,
    cartesian_velocity_limits,
):
    """Convert a joint path to FK poses and Jacobian twists, slowing if required."""
    times = np.asarray(times, dtype=np.float64)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    joint_velocity_limits = validate_six(joint_velocity_limits, "joint velocity limits")
    cartesian_velocity_limits = validate_six(cartesian_velocity_limits, "Cartesian velocity limits")
    if np.any(joint_velocity_limits <= 0.0) or np.any(cartesian_velocity_limits <= 0.0):
        raise ValueError("velocity limits must be positive")
    if joint_positions.shape != joint_velocities.shape or joint_positions.shape != (len(times), 6):
        raise ValueError("FK trajectory arrays have incompatible shapes")

    poses = np.empty_like(joint_positions)
    twists = np.empty_like(joint_velocities)
    for index, (position, velocity) in enumerate(zip(joint_positions, joint_velocities)):
        jacobian = np.asarray(kinematics.jacobian(position), dtype=np.float64)
        pose = np.asarray(kinematics.pose(position), dtype=np.float64)
        if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
            raise ValueError(f"invalid Jacobian at FK sample {index}")
        poses[index] = validate_six(pose, f"FK pose {index}")
        twists[index] = jacobian @ velocity

    joint_ratio = float(np.max(np.abs(joint_velocities) / joint_velocity_limits))
    cartesian_ratio = float(np.max(np.abs(twists) / cartesian_velocity_limits))
    time_scale = max(1.0, joint_ratio, cartesian_ratio)
    scaled_times = times * time_scale
    scaled_twists = twists / time_scale
    scaled_twists[0] = 0.0
    scaled_twists[-1] = 0.0
    return scaled_times, poses, scaled_twists, time_scale


def wait_until_stationary(controller, duration_s=0.25, velocity_limit=0.10, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    stationary_start = None
    peak = math.inf
    while time.monotonic() < deadline:
        velocity = np.asarray(controller.get_joint_state().vel(), dtype=np.float64)
        peak = float(np.max(np.abs(velocity)))
        if peak <= velocity_limit:
            if stationary_start is None:
                stationary_start = time.monotonic()
            if time.monotonic() - stationary_start >= duration_s:
                return
        else:
            stationary_start = None
        time.sleep(0.01)
    raise RuntimeError(f"arm did not become stationary; latest peak velocity was {peak:.3f} rad/s")


def write_joint_recording(path: Path, times, positions, velocities, torques):
    columns = ["elapsed_s"]
    for prefix in ("joint_position", "joint_velocity", "joint_torque"):
        columns.extend(f"{prefix}_{joint}" for joint in range(1, 7))
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for index, elapsed in enumerate(times - times[0]):
            writer.writerow([elapsed, *positions[index], *velocities[index], *torques[index]])


def write_joint_reference(path: Path, times, positions, velocities):
    columns = ["replay_elapsed_s"]
    for prefix in ("joint_position", "joint_velocity"):
        columns.extend(f"{prefix}_{joint}" for joint in range(1, 7))
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for index, elapsed in enumerate(times):
            writer.writerow([elapsed, *positions[index], *velocities[index]])


def write_fk_reference(path: Path, times, joint_positions, poses, twists):
    columns = ["replay_elapsed_s"]
    columns.extend(f"joint_position_{joint}" for joint in range(1, 7))
    columns.extend(f"pose_{axis}" for axis in CARTESIAN_AXES)
    columns.extend(f"twist_{axis}" for axis in CARTESIAN_AXES)
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for index, elapsed in enumerate(times):
            writer.writerow([elapsed, *joint_positions[index], *poses[index], *twists[index]])


def replay_telemetry_columns():
    columns = [
        "elapsed_s",
        "sequence",
        "cartesian_pd_enabled",
        "desired_pose_x", "desired_pose_y", "desired_pose_z",
        "desired_pose_rx", "desired_pose_ry", "desired_pose_rz",
        "measured_pose_x", "measured_pose_y", "measured_pose_z",
        "measured_pose_rx", "measured_pose_ry", "measured_pose_rz",
    ]
    for prefix in (
        "joint_position", "joint_velocity", "joint_torque_measured",
        "desired_joint_velocity", "host_stiffness_torque", "gravity_torque",
        "final_t_ff", "packed_motor_kp", "packed_motor_kd", "motor_protocol_velocity",
    ):
        columns.extend(f"{prefix}_{joint}" for joint in range(1, 7))
    return columns


def capture_replay_row(controller, elapsed_s):
    telemetry = controller._get_control_cycle_telemetry()
    row = [
        elapsed_s,
        telemetry.sequence,
        int(bool(telemetry.cartesian_pd_enabled)),
        *telemetry.desired_pose,
        *telemetry.measured_pose,
        *telemetry.feedback_joint_position,
        *telemetry.feedback_joint_velocity,
        *telemetry.feedback_joint_torque,
        *telemetry.desired_joint_velocity_limited,
        *telemetry.host_cartesian_stiffness_torque,
        *telemetry.gravity_torque,
        *telemetry.final_joint_torque_command,
        *telemetry.final_motor_kp,
        *telemetry.final_motor_kd,
        *telemetry.final_motor_protocol_velocity,
    ]
    return [float(value) for value in row]


def joint_replay_telemetry_columns():
    columns = ["elapsed_s", "sequence", "cartesian_pd_enabled"]
    for prefix in (
        "desired_joint_position", "measured_joint_position",
        "desired_joint_velocity", "measured_joint_velocity",
        "joint_torque_measured", "gravity_torque", "final_t_ff",
        "packed_motor_kp", "packed_motor_kd", "motor_protocol_velocity",
    ):
        columns.extend(f"{prefix}_{joint}" for joint in range(1, 7))
    return columns


def capture_joint_replay_row(controller, elapsed_s):
    telemetry = controller._get_control_cycle_telemetry()
    row = [
        elapsed_s,
        telemetry.sequence,
        int(bool(telemetry.cartesian_pd_enabled)),
        *telemetry.final_joint_position_command,
        *telemetry.feedback_joint_position,
        *telemetry.final_joint_velocity_command,
        *telemetry.feedback_joint_velocity,
        *telemetry.feedback_joint_torque,
        *telemetry.gravity_torque,
        *telemetry.final_joint_torque_command,
        *telemetry.final_motor_kp,
        *telemetry.final_motor_kd,
        *telemetry.final_motor_protocol_velocity,
    ]
    return [float(value) for value in row]


def pose_error_norms(desired, measured):
    error = np.asarray(arx5._cartesian_pose_error(desired, measured), dtype=np.float64)
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def project_cartesian_stiffness_along_path(kinematics, joint_positions, cartesian_kp):
    """Summarize the local physical joint stiffness J.T diag(Kx) J along a path."""
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    cartesian_kp = validate_six(cartesian_kp, "Cartesian Kp", nonnegative=True)
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 6:
        raise ValueError("joint_positions must have shape (samples, 6)")

    matrices = []
    for index, position in enumerate(joint_positions):
        jacobian = np.asarray(kinematics.jacobian(position), dtype=np.float64)
        if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
            raise ValueError(f"invalid Jacobian at stiffness sample {index}")
        matrices.append(jacobian.T @ np.diag(cartesian_kp) @ jacobian)
    matrices = np.asarray(matrices)
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    return {
        "cartesian_kp": cartesian_kp.tolist(),
        "units": "Nm/rad (local physical approximation)",
        "start_matrix": matrices[0].tolist(),
        "start_diagonal": diagonal[0].tolist(),
        "median_diagonal": np.median(diagonal, axis=0).tolist(),
        "minimum_diagonal": np.min(diagonal, axis=0).tolist(),
        "maximum_diagonal": np.max(diagonal, axis=0).tolist(),
        "note": (
            "Kq = J.T diag(Kx) J is pose-dependent and generally coupled. "
            "The diagonal is only a rough fixed motor-Kp approximation."
        ),
    }


def return_to_recorded_start(
    controller,
    controller_config,
    start_joint_position,
    gripper_position,
    return_gain,
    return_kp,
    return_kd,
    return_speed,
    joint_tolerance_rad,
):
    """Move back to the taught start with stiff joint PD and validate the anchor."""
    current_state = controller.get_joint_state()
    return_duration = max(
        2.0,
        float(
            np.max(
                np.abs(
                    np.asarray(current_state.pos(), dtype=np.float64)
                    - start_joint_position
                )
            )
        )
        / return_speed,
    )
    controller.set_gain(return_gain)
    return_command = arx5.JointState(
        start_joint_position,
        np.zeros(6),
        np.zeros(6),
        gripper_position,
    )
    return_command.timestamp = controller.get_timestamp() + return_duration
    controller.set_joint_cmd(return_command)
    click.echo(
        f"Returning to the recorded start over {return_duration:.2f} seconds with stiff motor gains: "
        f"Kp={np.array2string(return_kp, precision=1)}, "
        f"Kd={np.array2string(return_kd, precision=1)}."
    )
    time.sleep(return_duration + RETURN_SETTLE_DURATION_S)
    wait_until_stationary(controller, timeout_s=3.0)
    returned_state = controller.get_joint_state()
    joint_error = float(
        np.max(
            np.abs(
                np.asarray(returned_state.pos(), dtype=np.float64)
                - start_joint_position
            )
        )
    )
    if joint_error > joint_tolerance_rad:
        raise RuntimeError(
            f"return-to-start joint error is {joint_error:.4f} rad; "
            f"limit is {joint_tolerance_rad:.4f} rad"
        )
    return returned_state, return_duration, joint_error


def run_hybrid_replay(
    controller,
    controller_config,
    cartesian_kp,
    motor_kd,
    replay_times,
    replay_poses,
    gripper_position,
):
    """Run one hybrid Cartesian condition and return rows plus reference offset."""
    controller.set_cartesian_pd_gain(arx5.CartesianPDGain(cartesian_kp, np.zeros(6)))
    controller.set_cartesian_damping_mode(arx5.CartesianDampingMode.MotorLocalJoint)
    controller.set_cartesian_motor_kd(motor_kd)
    controller.enable_cartesian_pd()
    time.sleep(2.0 * controller_config.controller_dt)
    packed = controller._get_control_cycle_telemetry()
    if not np.allclose(np.asarray(packed.final_motor_kp), 0.0, atol=1e-12):
        raise RuntimeError("hybrid replay packed nonzero motor Kp")
    if not np.allclose(np.asarray(packed.final_motor_kd), motor_kd, atol=1e-12):
        raise RuntimeError("hybrid replay did not pack the configured motor Kd")

    lead_time = 1.0
    trajectory_start = controller.get_timestamp() + lead_time
    commands = []
    for elapsed, pose in zip(replay_times, replay_poses):
        command = arx5.EEFState(pose, gripper_position)
        command.timestamp = trajectory_start + float(elapsed)
        commands.append(command)
    # Submit only the recorded-rate Cartesian knots. The C++ Cartesian
    # controller estimates knot twists and performs Hermite/SO(3) interpolation
    # every 2 ms; Python does not expand the trajectory to controller rate.
    controller.set_eef_traj(commands)

    reference_start_offset_s = trajectory_start - controller.get_timestamp()
    replay_rows = []
    replay_start = time.monotonic()
    nominal_end = replay_start + reference_start_offset_s + float(replay_times[-1])
    timeout = nominal_end + max(5.0, 0.5 * float(replay_times[-1]))
    settled_start = None
    next_telemetry = replay_start
    while time.monotonic() < timeout:
        now = time.monotonic()
        if now < next_telemetry:
            time.sleep(next_telemetry - now)
            now = time.monotonic()
        replay_rows.append(capture_replay_row(controller, now - replay_start))
        next_telemetry += 1.0 / 500.0
        if not controller.is_cartesian_pd_enabled():
            raise RuntimeError("hybrid Cartesian control disabled itself during replay")
        desired = controller.get_eef_cmd()
        desired_twist = np.asarray(controller.get_eef_cmd_twist(), dtype=np.float64)
        final_translation_error, final_rotation_error = pose_error_norms(
            replay_poses[-1], np.asarray(desired.pose_6d(), dtype=np.float64)
        )
        reference_done = (
            now >= nominal_end
            and final_translation_error < 1e-4
            and final_rotation_error < 1e-3
            and np.linalg.norm(desired_twist) < 1e-3
        )
        if reference_done:
            if settled_start is None:
                settled_start = now
            elif now - settled_start >= 0.5:
                return replay_rows, reference_start_offset_s
        else:
            settled_start = None
    raise RuntimeError("hybrid replay did not reach its final reference before timeout")


def run_joint_replay(
    controller,
    controller_config,
    joint_gain,
    joint_kp,
    joint_kd,
    replay_times,
    replay_joint_positions,
    replay_joint_velocities,
    gripper_position,
):
    """Run one direct joint condition and return rows, reference offset, and final error."""
    controller.set_gain(joint_gain)
    time.sleep(2.0 * controller_config.controller_dt)
    packed = controller._get_control_cycle_telemetry()
    if not np.allclose(np.asarray(packed.final_motor_kp), joint_kp, atol=1e-12):
        raise RuntimeError("joint replay did not pack the configured motor Kp")
    if not np.allclose(np.asarray(packed.final_motor_kd), joint_kd, atol=1e-12):
        raise RuntimeError("joint replay did not pack the configured motor Kd")

    lead_time = 1.0
    trajectory_start = controller.get_timestamp() + lead_time
    commands = []
    for elapsed, position, velocity in zip(
        replay_times, replay_joint_positions, replay_joint_velocities
    ):
        command = arx5.JointState(
            position,
            velocity,
            np.zeros(6),
            gripper_position,
        )
        command.timestamp = trajectory_start + float(elapsed)
        commands.append(command)
    controller.set_joint_traj(commands)

    reference_start_offset_s = trajectory_start - controller.get_timestamp()
    replay_rows = []
    replay_start = time.monotonic()
    nominal_end = replay_start + reference_start_offset_s + float(replay_times[-1])
    timeout = nominal_end + max(5.0, 0.5 * float(replay_times[-1]))
    settled_start = None
    hold_installed = False
    next_telemetry = replay_start
    while time.monotonic() < timeout:
        now = time.monotonic()
        if now < next_telemetry:
            time.sleep(next_telemetry - now)
            now = time.monotonic()
        replay_rows.append(capture_joint_replay_row(controller, now - replay_start))
        next_telemetry += 1.0 / 500.0
        if now >= nominal_end and not hold_installed:
            final_hold = arx5.JointState(
                replay_joint_positions[-1],
                np.zeros(6),
                np.zeros(6),
                gripper_position,
            )
            final_hold.timestamp = (
                controller.get_timestamp() + 2.0 * controller_config.controller_dt
            )
            controller.set_joint_cmd(final_hold)
            hold_installed = True
        desired = controller.get_joint_cmd()
        final_reference_error = float(
            np.max(
                np.abs(
                    np.asarray(desired.pos(), dtype=np.float64)
                    - replay_joint_positions[-1]
                )
            )
        )
        reference_done = (
            hold_installed
            and final_reference_error < 1e-4
            and np.linalg.norm(np.asarray(desired.vel(), dtype=np.float64)) < 1e-3
        )
        if reference_done:
            if settled_start is None:
                settled_start = now
            elif now - settled_start >= 0.5:
                final_state = controller.get_joint_state()
                final_error = (
                    np.asarray(final_state.pos(), dtype=np.float64)
                    - replay_joint_positions[-1]
                )
                return replay_rows, reference_start_offset_s, final_error
        else:
            settled_start = None
    raise RuntimeError("joint replay did not reach its final reference before timeout")


def _rows_as_columns(columns, rows):
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError("telemetry rows do not match their columns")
    return {name: data[:, index] for index, name in enumerate(columns)}


def _norm_metrics(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64) * scale
    return {
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def build_side_by_side_comparison(
    kinematics,
    replay_times,
    replay_joint_positions,
    replay_poses,
    hybrid_rows,
    hybrid_reference_start_s,
    joint_rows,
    joint_reference_start_s,
):
    """Resample both conditions at reference times and calculate common metrics."""
    hybrid = _rows_as_columns(replay_telemetry_columns(), hybrid_rows)
    joint = _rows_as_columns(joint_replay_telemetry_columns(), joint_rows)
    hybrid_sample_times = hybrid_reference_start_s + replay_times
    joint_sample_times = joint_reference_start_s + replay_times

    hybrid_joint_position = np.column_stack(
        [
            np.interp(
                hybrid_sample_times,
                hybrid["elapsed_s"],
                hybrid[f"joint_position_{index}"],
            )
            for index in range(1, 7)
        ]
    )
    joint_replay_position = np.column_stack(
        [
            np.interp(
                joint_sample_times,
                joint["elapsed_s"],
                joint[f"measured_joint_position_{index}"],
            )
            for index in range(1, 7)
        ]
    )
    hybrid_pose = np.column_stack(
        [
            np.interp(
                hybrid_sample_times,
                hybrid["elapsed_s"],
                hybrid[f"measured_pose_{axis}"],
            )
            for axis in CARTESIAN_AXES
        ]
    )
    joint_pose = np.asarray(
        [kinematics.pose(position) for position in joint_replay_position],
        dtype=np.float64,
    )

    hybrid_translation_error = []
    hybrid_rotation_error = []
    joint_translation_error = []
    joint_rotation_error = []
    for reference_pose, measured_hybrid, measured_joint in zip(
        replay_poses, hybrid_pose, joint_pose
    ):
        translation, rotation = pose_error_norms(reference_pose, measured_hybrid)
        hybrid_translation_error.append(translation)
        hybrid_rotation_error.append(rotation)
        translation, rotation = pose_error_norms(reference_pose, measured_joint)
        joint_translation_error.append(translation)
        joint_rotation_error.append(rotation)

    hybrid_translation_error = np.asarray(hybrid_translation_error)
    hybrid_rotation_error = np.asarray(hybrid_rotation_error)
    joint_translation_error = np.asarray(joint_translation_error)
    joint_rotation_error = np.asarray(joint_rotation_error)
    hybrid_joint_error = hybrid_joint_position - replay_joint_positions
    joint_replay_error = joint_replay_position - replay_joint_positions

    def condition_metrics(translation_error, rotation_error, joint_error):
        return {
            "translation_error_mm": _norm_metrics(translation_error, 1000.0),
            "rotation_error_deg": _norm_metrics(rotation_error, 180.0 / math.pi),
            "joint_error_vector_deg": _norm_metrics(
                np.linalg.norm(joint_error, axis=1), 180.0 / math.pi
            ),
            "joint_error_rmse_deg": (
                np.sqrt(np.mean(np.square(joint_error), axis=0))
                * 180.0
                / math.pi
            ).tolist(),
            "joint_error_maximum_deg": (
                np.max(np.abs(joint_error), axis=0) * 180.0 / math.pi
            ).tolist(),
        }

    summary = {
        "hybrid-cartesian": condition_metrics(
            hybrid_translation_error,
            hybrid_rotation_error,
            hybrid_joint_error,
        ),
        "joint": condition_metrics(
            joint_translation_error,
            joint_rotation_error,
            joint_replay_error,
        ),
    }
    values = {
        "replay_times": replay_times,
        "reference_joint_position": replay_joint_positions,
        "hybrid_joint_position": hybrid_joint_position,
        "joint_replay_position": joint_replay_position,
        "reference_pose": replay_poses,
        "hybrid_pose": hybrid_pose,
        "joint_pose": joint_pose,
        "hybrid_translation_error": hybrid_translation_error,
        "joint_translation_error": joint_translation_error,
        "hybrid_rotation_error": hybrid_rotation_error,
        "joint_rotation_error": joint_rotation_error,
    }
    return summary, values


def write_side_by_side_comparison(path, summary, values, stiffness_projection):
    """Write time-aligned CSV plus JSON/HTML summaries for the paired replay."""
    path = Path(path)
    csv_path = path.with_suffix(".csv")
    json_path = path.with_suffix(".json")
    html_path = path.with_suffix(".html")
    columns = ["replay_elapsed_s"]
    for prefix in (
        "reference_joint_position",
        "hybrid_joint_position",
        "joint_replay_position",
    ):
        columns.extend(f"{prefix}_{index}" for index in range(1, 7))
    for prefix in ("reference_pose", "hybrid_pose", "joint_pose"):
        columns.extend(f"{prefix}_{axis}" for axis in CARTESIAN_AXES)
    columns.extend(
        [
            "hybrid_translation_error_m",
            "joint_translation_error_m",
            "hybrid_rotation_error_rad",
            "joint_rotation_error_rad",
        ]
    )
    with csv_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for index, elapsed in enumerate(values["replay_times"]):
            writer.writerow(
                [
                    elapsed,
                    *values["reference_joint_position"][index],
                    *values["hybrid_joint_position"][index],
                    *values["joint_replay_position"][index],
                    *values["reference_pose"][index],
                    *values["hybrid_pose"][index],
                    *values["joint_pose"][index],
                    values["hybrid_translation_error"][index],
                    values["joint_translation_error"][index],
                    values["hybrid_rotation_error"][index],
                    values["joint_rotation_error"][index],
                ]
            )

    payload = {
        "conditions": summary,
        "equivalent_joint_stiffness": stiffness_projection,
        "interpretation": (
            "Both conditions replay the same recorded joint path and timing. "
            "Hybrid runs first; the arm returns to the recorded start before direct joint replay."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    metric_rows = []
    for label, metrics in summary.items():
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{metrics['translation_error_mm']['rms']:.3f}</td>"
            f"<td>{metrics['translation_error_mm']['p95']:.3f}</td>"
            f"<td>{metrics['translation_error_mm']['maximum']:.3f}</td>"
            f"<td>{metrics['rotation_error_deg']['rms']:.3f}</td>"
            f"<td>{metrics['rotation_error_deg']['maximum']:.3f}</td>"
            f"<td>{metrics['joint_error_vector_deg']['rms']:.3f}</td>"
            "</tr>"
        )
    configured = stiffness_projection["configured_cartesian_kp"]
    x_only = stiffness_projection["x_axis_only"]
    html_path.write_text(
        "<html><head><meta charset=\"utf-8\"><title>Teach/replay comparison</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.4rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}code,pre{background:#f4f4f4;padding:.2rem}</style>"
        "</head><body><h1>Hybrid Cartesian vs direct joint replay</h1>"
        f"<p>{html.escape(payload['interpretation'])}</p>"
        "<table><thead><tr><th>Condition</th><th>XYZ RMS (mm)</th><th>XYZ P95 (mm)</th>"
        "<th>XYZ max (mm)</th><th>rotation RMS (deg)</th><th>rotation max (deg)</th>"
        "<th>joint-vector RMS (deg)</th></tr></thead><tbody>"
        f"{''.join(metric_rows)}</tbody></table>"
        "<h2>Rough equivalent diagonal joint stiffness</h2>"
        "<p>The full matrix is <code>Kq = Jᵀ diag(Kx) J</code>; off-diagonal coupling and pose "
        "dependence mean these diagonal values are only an approximation.</p>"
        "<p>Configured Kx median diagonal (Nm/rad): <code>"
        f"{html.escape(str(configured['median_diagonal']))}</code></p>"
        f"<p>X-axis-only median diagonal (Nm/rad): <code>{html.escape(str(x_only['median_diagonal']))}</code></p>"
        "</body></html>"
    )
    return csv_path, json_path, html_path


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option(
    "--replay-mode",
    type=click.Choice(REPLAY_MODES),
    default="hybrid-cartesian",
    show_default=True,
    help=(
        "Replay with hybrid Cartesian impedance, replay the taught joints directly, "
        "or run hybrid then joint from the same recording."
    ),
)
@click.option("--duration", type=click.FloatRange(min=1.0), default=5.0, show_default=True)
@click.option(
    "--trajectory-rate",
    "--record-rate",
    "trajectory_rate",
    type=click.FloatRange(min=50.0),
    default=DEFAULT_TRAJECTORY_KNOT_RATE_HZ,
    show_default=True,
    help=(
        "Single shared rate used to record and replay position knots. The C++ controller "
        "interpolates between them at its 500 Hz control rate."
    ),
)
@click.option(
    "--cartesian-kp", "cartesian_kp", type=float, nargs=6,
    default=tuple(DEFAULT_CARTESIAN_KP), show_default=True,
    metavar="X Y Z RX RY RZ",
    help="Hybrid Cartesian stiffness: N/m for XYZ and Nm/rad for rotations.",
)
@click.option(
    "--motor-kd", type=float, nargs=6, default=tuple(DEFAULT_MOTOR_KD), show_default=True,
    metavar="J1 J2 J3 J4 J5 J6",
)
@click.option(
    "--joint-replay-kp",
    type=float,
    nargs=6,
    default=None,
    metavar="J1 J2 J3 J4 J5 J6",
    help=(
        "Direct joint-replay Kp. Joint-only mode defaults to the stiff return gains; "
        "comparison mode defaults to the path-median diagonal of J.T Kx J."
    ),
)
@click.option("--return-speed", type=click.FloatRange(min=0.05, max=1.0), default=0.35, show_default=True)
@click.option(
    "--start-joint-tolerance-rad",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_START_JOINT_TOLERANCE_RAD,
    show_default=True,
    help="Maximum joint mismatch accepted after the stiff return-to-start hold.",
)
@click.option(
    "--start-translation-tolerance-mm",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_START_TRANSLATION_TOLERANCE_MM,
    show_default=True,
    help="Maximum TCP translation mismatch accepted before hybrid replay.",
)
@click.option(
    "--start-rotation-tolerance-deg",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_START_ROTATION_TOLERANCE_DEG,
    show_default=True,
    help="Maximum TCP orientation mismatch accepted before hybrid replay.",
)
@click.option("--output-prefix", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--yes", "assume_yes", is_flag=True, help="Skip only the final replay-envelope confirmation.")
@click.option("--return-home/--no-return-home", default=True, show_default=True)
def main(
    model,
    interface,
    replay_mode,
    duration,
    trajectory_rate,
    cartesian_kp,
    motor_kd,
    joint_replay_kp,
    return_speed,
    start_joint_tolerance_rad,
    start_translation_tolerance_mm,
    start_rotation_tolerance_deg,
    output_prefix,
    assume_yes,
    return_home,
):
    """Record a taught joint path on MODEL/INTERFACE and replay it."""
    cartesian_kp = validate_six(cartesian_kp, "Cartesian Kp", nonnegative=True)
    motor_kd = validate_six(motor_kd, "motor Kd", nonnegative=True)
    if replay_mode in ("hybrid-cartesian", "comparison") and np.allclose(cartesian_kp, 0.0):
        raise click.ClickException("at least one Cartesian stiffness must be nonzero")
    if joint_replay_kp is not None:
        joint_replay_kp = validate_six(
            joint_replay_kp, "joint replay Kp", nonnegative=True
        )
        if np.any(joint_replay_kp > 500.0):
            raise click.ClickException("joint replay Kp exceeds the protocol maximum of 500")
    if np.any(motor_kd > 5.0):
        raise click.ClickException("motor Kd exceeds the protocol maximum of 5")

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if robot_config.joint_dof != 6:
        raise click.ClickException("teach/replay requires a six-joint model")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )
    controller_config.background_send_recv = True
    controller_config.gravity_compensation = True
    controller_config.friction_compensation = False
    controller_config.joint_friction_compensation = False
    controller_config.gripper_friction_compensation = False
    return_kp = np.asarray(controller_config.default_kp, dtype=np.float64).copy()
    return_kd = np.asarray(controller_config.default_kd, dtype=np.float64).copy()

    if output_prefix is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            "hybrid_cartesian_teach_replay"
            if replay_mode == "hybrid-cartesian"
            else "joint_teach_replay" if replay_mode == "joint" else "teach_replay_comparison"
        )
        output_prefix = Path.cwd() / f"{experiment_name}_{model}_{stamp}"
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    controller = None
    cartesian_enabled = False
    completed = False
    record_times = []
    record_positions = []
    record_velocities = []
    record_torques = []
    replay_rows = []
    metadata = {
        "model": model,
        "interface": interface,
        "replay_mode": replay_mode,
        "record_duration_s": duration,
        "requested_trajectory_knot_rate_hz": trajectory_rate,
        "requested_record_rate_hz": trajectory_rate,
        "requested_replay_knot_rate_hz": trajectory_rate,
        "record_replay_knot_rate_locked": True,
        "controller_loop_rate_hz": 1.0 / controller_config.controller_dt,
        "trajectory_position_processing": "recorded knots; no Python position interpolation",
        "joint_controller_interpolation": controller_config.interpolation_method,
        "cartesian_controller_interpolation": "cubic Hermite translation and SO(3) orientation",
        "cartesian_kp": cartesian_kp.tolist(),
        "cartesian_kd": [0.0] * 6,
        "motor_kd": motor_kd.tolist(),
        "return_kp": return_kp.tolist(),
        "return_kd": return_kd.tolist(),
        "return_settle_duration_s": RETURN_SETTLE_DURATION_S,
        "start_joint_tolerance_rad": start_joint_tolerance_rad,
        "start_translation_tolerance_mm": start_translation_tolerance_mm,
        "start_rotation_tolerance_deg": start_rotation_tolerance_deg,
        "failure": None,
    }
    if joint_replay_kp is not None:
        metadata["requested_joint_replay_kp"] = joint_replay_kp.tolist()

    try:
        controller = arx5.Arx5CartesianController(robot_config, controller_config, interface)
        controller.set_log_level(arx5.LogLevel.WARNING)
        controller.set_gravity_compensation(True)
        controller.set_friction_compensation(False)

        free_gain = arx5.Gain(robot_config.joint_dof)
        free_gain.gripper_kp = controller_config.default_gripper_kp
        free_gain.gripper_kd = controller_config.default_gripper_kd
        controller.set_gain(free_gain)

        while True:
            click.prompt(
                "Arm is gravity compensated with arm gains off. Move to the trajectory START, release it, then press Enter",
                default="",
                show_default=False,
            )
            try:
                wait_until_stationary(controller)
                break
            except RuntimeError as error:
                click.echo(str(error), err=True)

        click.echo(
            f"Recording begins in 3 seconds. Move the arm through the desired trajectory for {duration:g} seconds."
        )
        for count in (3, 2, 1):
            click.echo(str(count))
            time.sleep(1.0)

        record_start = time.monotonic()
        next_sample = record_start
        sample_dt = 1.0 / trajectory_rate
        while True:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(next_sample - now)
                now = time.monotonic()
            if now - record_start > duration and record_times:
                break
            state = controller.get_joint_state()
            record_times.append(float(state.timestamp))
            record_positions.append(np.asarray(state.pos(), dtype=np.float64).copy())
            record_velocities.append(np.asarray(state.vel(), dtype=np.float64).copy())
            record_torques.append(np.asarray(state.torque(), dtype=np.float64).copy())
            skipped = max(0, int((now - next_sample) // sample_dt))
            next_sample += (skipped + 1) * sample_dt
        click.echo(f"Recorded {len(record_times)} joint samples.")

        record_times_array = np.asarray(record_times)
        record_positions_array = np.asarray(record_positions)
        record_velocities_array = np.asarray(record_velocities)
        record_torques_array = np.asarray(record_torques)
        if not all(
            np.all(np.isfinite(values))
            for values in (record_times_array, record_positions_array, record_velocities_array, record_torques_array)
        ):
            raise RuntimeError("joint recording contains non-finite values")
        if np.any(record_positions_array < np.asarray(robot_config.joint_pos_min)) or np.any(
            record_positions_array > np.asarray(robot_config.joint_pos_max)
        ):
            raise RuntimeError("recorded trajectory crossed a configured joint limit")

        joint_csv = Path(f"{output_prefix}_recorded_joint.csv")
        write_joint_recording(
            joint_csv,
            record_times_array,
            record_positions_array,
            record_velocities_array,
            record_torques_array,
        )

        end_state = controller.get_joint_state()
        hold_end = arx5.JointState(
            np.asarray(end_state.pos(), dtype=np.float64),
            np.zeros(6),
            np.zeros(6),
            end_state.gripper_pos,
        )
        hold_end.timestamp = controller.get_timestamp()
        controller.set_joint_cmd(hold_end)
        damping_gain = arx5.Gain(
            np.zeros(6), motor_kd,
            controller_config.default_gripper_kp,
            controller_config.default_gripper_kd,
        )
        controller.set_gain(damping_gain)
        time.sleep(2.0 * controller_config.controller_dt)

        start_joint_position = record_positions_array[0]
        return_gain = arx5.Gain(
            return_kp, return_kd,
            controller_config.default_gripper_kp,
            controller_config.default_gripper_kd,
        )
        initial_anchor_tolerance = (
            min(start_joint_tolerance_rad, 0.05)
            if replay_mode == "comparison"
            else start_joint_tolerance_rad
        )
        returned_state, return_duration, joint_anchor_error = return_to_recorded_start(
            controller,
            controller_config,
            start_joint_position,
            end_state.gripper_pos,
            return_gain,
            return_kp,
            return_kd,
            return_speed,
            initial_anchor_tolerance,
        )

        # Preserve the recorded position samples as trajectory knots. Velocity
        # estimates below are used only for feasibility/retiming; both replay
        # paths interpolate the position knots inside the C++ controller loop.
        replay_times, replay_joint_positions, replay_joint_velocities = prepare_recorded_joint_trajectory(
            record_times_array,
            record_positions_array,
        )
        actual_knot_rate_hz = (len(replay_times) - 1) / replay_times[-1]
        metadata["replay_knot_count"] = len(replay_times)
        metadata["actual_record_and_replay_knot_rate_hz"] = actual_knot_rate_hz
        metadata["actual_record_knot_rate_hz"] = actual_knot_rate_hz
        metadata["actual_replay_knot_rate_hz"] = actual_knot_rate_hz
        click.echo(
            f"Prepared {len(replay_times)} recorded position knots at "
            f"{actual_knot_rate_hz:.1f} Hz; the C++ controller interpolates them at "
            f"{1.0 / controller_config.controller_dt:.0f} Hz."
        )
        metadata["recorded_start_joint_position"] = start_joint_position.tolist()
        metadata["returned_start_joint_position"] = np.asarray(returned_state.pos()).tolist()
        metadata["return_joint_error_max_rad"] = joint_anchor_error

        needs_hybrid = replay_mode in ("hybrid-cartesian", "comparison")
        needs_joint = replay_mode in ("joint", "comparison")
        stiffness_projection = None
        velocity_limit_report = None
        if needs_hybrid:
            kinematics = arx5._CartesianKinematics(
                robot_config.urdf_path,
                robot_config.joint_dof,
                robot_config.base_link_name,
                robot_config.eef_link_name,
            )
            replay_times, replay_poses, replay_twists, time_scale = calculate_fk_reference(
                kinematics,
                replay_times,
                replay_joint_positions,
                replay_joint_velocities,
                np.asarray(robot_config.joint_vel_max),
                np.asarray(robot_config.ee_vel_max),
            )
            velocity_limit_report = build_velocity_limit_report(
                replay_joint_velocities,
                np.asarray(robot_config.joint_vel_max),
                replay_twists * time_scale,
                np.asarray(robot_config.ee_vel_max),
            )
            replay_joint_velocities = replay_joint_velocities / time_scale
            replay_joint_velocities[0] = 0.0
            replay_joint_velocities[-1] = 0.0
            fk_csv = Path(f"{output_prefix}_fk_reference.csv")
            write_fk_reference(fk_csv, replay_times, replay_joint_positions, replay_poses, replay_twists)

            x_axis_kp = np.zeros(6, dtype=np.float64)
            x_axis_kp[0] = cartesian_kp[0]
            stiffness_projection = {
                "formula": "Kq = J.T diag(Kx) J",
                "configured_cartesian_kp": project_cartesian_stiffness_along_path(
                    kinematics, replay_joint_positions, cartesian_kp
                ),
                "x_axis_only": project_cartesian_stiffness_along_path(
                    kinematics, replay_joint_positions, x_axis_kp
                ),
            }
            metadata["equivalent_joint_stiffness"] = stiffness_projection

            measured_start_pose = np.asarray(
                kinematics.pose(np.asarray(returned_state.pos())), dtype=np.float64
            )
            translation_error, rotation_error = pose_error_norms(replay_poses[0], measured_start_pose)
            metadata["return_tcp_translation_error_m"] = translation_error
            metadata["return_tcp_rotation_error_rad"] = rotation_error
            translation_tolerance_m = start_translation_tolerance_mm / 1000.0
            rotation_tolerance_rad = math.radians(start_rotation_tolerance_deg)
            if translation_error > translation_tolerance_m or rotation_error > rotation_tolerance_rad:
                raise RuntimeError(
                    "returned TCP differs from the recorded start by "
                    f"{1000.0 * translation_error:.2f} mm and {math.degrees(rotation_error):.2f} deg; "
                    f"limits are {start_translation_tolerance_mm:g} mm and "
                    f"{start_rotation_tolerance_deg:g} deg"
                )

            translation_span = np.ptp(replay_poses[:, :3], axis=0)
            rotation_span = np.ptp(np.unwrap(replay_poses[:, 3:], axis=0), axis=0)
            click.echo(
                "FK path span: translation "
                f"{np.array2string(1000.0 * translation_span, precision=1)} mm; rotation "
                f"{np.array2string(np.degrees(rotation_span), precision=1)} deg."
            )
            click.echo(
                "Hybrid Cartesian Kp: " + np.array2string(cartesian_kp, precision=2)
                + "; motor Kd: " + np.array2string(motor_kd, precision=2)
            )
            click.echo(
                "Rough path-median diagonal joint Kp from J.T Kx J: "
                + np.array2string(
                    np.asarray(
                        stiffness_projection["configured_cartesian_kp"]["median_diagonal"]
                    ),
                    precision=1,
                )
            )
            click.echo(
                f"For X stiffness {cartesian_kp[0]:g} N/m alone, the path-median diagonal is: "
                + np.array2string(
                    np.asarray(stiffness_projection["x_axis_only"]["median_diagonal"]),
                    precision=2,
                )
            )
        else:
            unscaled_joint_velocities = replay_joint_velocities.copy()
            replay_times, replay_joint_positions, replay_joint_velocities, time_scale = (
                scale_joint_reference_for_limits(
                    replay_times,
                    replay_joint_positions,
                    replay_joint_velocities,
                    np.asarray(robot_config.joint_vel_max),
                )
            )
            velocity_limit_report = build_velocity_limit_report(
                unscaled_joint_velocities,
                np.asarray(robot_config.joint_vel_max),
            )
        if needs_joint:
            if replay_mode == "comparison" and joint_replay_kp is None:
                joint_replay_kp = np.asarray(
                    stiffness_projection["configured_cartesian_kp"]["median_diagonal"],
                    dtype=np.float64,
                )
            elif joint_replay_kp is None:
                joint_replay_kp = return_kp.copy()
            joint_replay_kd = motor_kd.copy()
            joint_gain = arx5.Gain(
                joint_replay_kp,
                joint_replay_kd,
                controller_config.default_gripper_kp,
                controller_config.default_gripper_kd,
            )
            metadata["joint_replay_kp"] = joint_replay_kp.tolist()
            metadata["joint_replay_kd"] = joint_replay_kd.tolist()
            metadata["joint_replay_kp_source"] = (
                "path-median diagonal of J.T Kx J"
                if replay_mode == "comparison" and "requested_joint_replay_kp" not in metadata
                else "explicit CLI" if "requested_joint_replay_kp" in metadata else "stiff return gains"
            )
            joint_reference_csv = Path(f"{output_prefix}_joint_reference.csv")
            write_joint_reference(
                joint_reference_csv,
                replay_times,
                replay_joint_positions,
                replay_joint_velocities,
            )
            joint_span = np.ptp(replay_joint_positions, axis=0)
            peak_joint_velocity = np.max(np.abs(replay_joint_velocities), axis=0)
            click.echo(
                "Joint path span: "
                f"{np.array2string(np.degrees(joint_span), precision=1)} deg; peak velocity "
                f"{np.array2string(peak_joint_velocity, precision=2)} rad/s."
            )
            click.echo(
                "Joint replay Kp: " + np.array2string(joint_replay_kp, precision=1)
                + "; Kd: " + np.array2string(joint_replay_kd, precision=1)
            )

        metadata["replay_time_scale"] = time_scale
        metadata["replay_duration_s"] = float(replay_times[-1])
        metadata["velocity_limit_check"] = velocity_limit_report
        metadata["submitted_replay_knot_rate_hz"] = (
            (len(replay_times) - 1) / replay_times[-1]
        )
        if not math.isclose(
            velocity_limit_report["time_scale"], time_scale, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError("velocity-limit report disagrees with trajectory time scaling")
        click.echo(format_velocity_limit_report(velocity_limit_report))
        click.echo(
            f"Submitted trajectory timeline: "
            f"{metadata['submitted_replay_knot_rate_hz']:.1f} knots/s."
        )
        if not assume_yes and not click.confirm(
            "The recorded path is feasible by construction, but collisions were not checked. Replay it now?"
        ):
            raise click.Abort()

        hybrid_rows = None
        joint_rows = None
        hybrid_reference_start_s = None
        joint_reference_start_s = None
        if needs_hybrid:
            click.echo("Starting hybrid Cartesian replay.")
            cartesian_enabled = True
            hybrid_rows, hybrid_reference_start_s = run_hybrid_replay(
                controller,
                controller_config,
                cartesian_kp,
                motor_kd,
                replay_times,
                replay_poses,
                returned_state.gripper_pos,
            )
            replay_rows.extend(hybrid_rows)
            hybrid_csv = Path(f"{output_prefix}_hybrid_replay.csv")
            with hybrid_csv.open("w", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(replay_telemetry_columns())
                writer.writerows(hybrid_rows)
            metadata["hybrid_reference_start_elapsed_s"] = hybrid_reference_start_s
            metadata["hybrid_replay_telemetry_samples"] = len(hybrid_rows)

        if replay_mode == "comparison":
            controller.disable_cartesian_pd()
            cartesian_enabled = False
            time.sleep(2.0 * controller_config.controller_dt)
            click.echo("Hybrid replay complete; returning to the common start before joint replay.")
            comparison_tolerance = min(start_joint_tolerance_rad, 0.05)
            second_returned_state, _, second_anchor_error = return_to_recorded_start(
                controller,
                controller_config,
                start_joint_position,
                returned_state.gripper_pos,
                return_gain,
                return_kp,
                return_kd,
                return_speed,
                comparison_tolerance,
            )
            anchor_delta = float(
                np.max(
                    np.abs(
                        np.asarray(second_returned_state.pos(), dtype=np.float64)
                        - np.asarray(returned_state.pos(), dtype=np.float64)
                    )
                )
            )
            if anchor_delta > 0.05:
                raise RuntimeError(
                    f"comparison anchors differ by {anchor_delta:.4f} rad; limit is 0.0500 rad"
                )
            second_start_pose = np.asarray(
                kinematics.pose(np.asarray(second_returned_state.pos())), dtype=np.float64
            )
            second_translation_error, second_rotation_error = pose_error_norms(
                replay_poses[0], second_start_pose
            )
            if (
                second_translation_error > start_translation_tolerance_mm / 1000.0
                or second_rotation_error > math.radians(start_rotation_tolerance_deg)
            ):
                raise RuntimeError(
                    "joint condition TCP anchor differs from the recorded start by "
                    f"{1000.0 * second_translation_error:.2f} mm and "
                    f"{math.degrees(second_rotation_error):.2f} deg"
                )
            metadata["comparison_second_returned_start_joint_position"] = (
                np.asarray(second_returned_state.pos()).tolist()
            )
            metadata["comparison_second_return_joint_error_max_rad"] = second_anchor_error
            metadata["comparison_anchor_delta_max_rad"] = anchor_delta
            metadata["comparison_second_return_tcp_translation_error_m"] = (
                second_translation_error
            )
            metadata["comparison_second_return_tcp_rotation_error_rad"] = second_rotation_error
            returned_state = second_returned_state

        if needs_joint:
            click.echo("Starting direct joint replay.")
            joint_rows, joint_reference_start_s, final_joint_error = run_joint_replay(
                controller,
                controller_config,
                joint_gain,
                joint_replay_kp,
                joint_replay_kd,
                replay_times,
                replay_joint_positions,
                replay_joint_velocities,
                returned_state.gripper_pos,
            )
            replay_rows.extend(joint_rows)
            joint_csv_replay = Path(f"{output_prefix}_joint_replay.csv")
            with joint_csv_replay.open("w", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(joint_replay_telemetry_columns())
                writer.writerows(joint_rows)
            metadata["joint_reference_start_elapsed_s"] = joint_reference_start_s
            metadata["joint_replay_telemetry_samples"] = len(joint_rows)
            metadata["final_joint_tracking_error_rad"] = final_joint_error.tolist()
            metadata["final_joint_tracking_error_max_rad"] = float(
                np.max(np.abs(final_joint_error))
            )

        if replay_mode == "comparison":
            comparison_summary, comparison_values = build_side_by_side_comparison(
                kinematics,
                replay_times,
                replay_joint_positions,
                replay_poses,
                hybrid_rows,
                hybrid_reference_start_s,
                joint_rows,
                joint_reference_start_s,
            )
            comparison_prefix = Path(f"{output_prefix}_comparison")
            comparison_csv, comparison_json, comparison_html = write_side_by_side_comparison(
                comparison_prefix,
                comparison_summary,
                comparison_values,
                stiffness_projection,
            )
            metadata["comparison_summary"] = comparison_summary
        completed = True
        click.echo(f"Joint recording: {joint_csv}")
        if needs_hybrid:
            click.echo(f"FK reference:    {fk_csv}")
            click.echo(f"Hybrid telemetry:{hybrid_csv}")
        if needs_joint:
            click.echo(f"Joint reference: {joint_reference_csv}")
            click.echo(f"Joint telemetry: {joint_csv_replay}")
        if replay_mode == "comparison":
            click.echo(f"Comparison CSV:  {comparison_csv}")
            click.echo(f"Comparison JSON: {comparison_json}")
            click.echo(f"Comparison HTML: {comparison_html}")
    except click.Abort:
        metadata["failure"] = "operator canceled"
        click.echo("Replay canceled.", err=True)
    except KeyboardInterrupt:
        metadata["failure"] = "operator interrupted"
        click.echo("Interrupted; entering damping.", err=True)
    except Exception as error:
        metadata["failure"] = f"{type(error).__name__}: {error}"
        click.echo(f"Experiment stopped: {metadata['failure']}", err=True)
    finally:
        if controller is not None:
            try:
                if cartesian_enabled and controller.is_cartesian_pd_enabled():
                    controller.disable_cartesian_pd()
                controller.set_to_damping()
                if completed and return_home:
                    click.echo("Replay complete; returning home.")
                    controller.reset_to_home()
                    time.sleep(0.6)
            except Exception as cleanup_error:
                metadata["cleanup_failure"] = str(cleanup_error)

        metadata["recorded_samples"] = len(record_times)
        metadata["replay_telemetry_samples"] = len(replay_rows)
        metadata_path = Path(f"{output_prefix}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        click.echo(f"Metadata:        {metadata_path}")

    if metadata["failure"] and metadata["failure"] != "operator interrupted":
        raise click.ClickException(metadata["failure"])


if __name__ == "__main__":
    main()
