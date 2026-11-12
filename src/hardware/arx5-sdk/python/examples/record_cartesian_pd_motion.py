"""Record the Cartesian PD sinusoid at Cartesian and joint levels.

The controller still evaluates its reference and sends motor commands at
500 Hz. This utility updates the sinusoidal target at a configurable command
rate (50 Hz by default), samples one atomically published controller-cycle
snapshot at 500 Hz, and writes a CSV plus a dependency-free HTML/SVG report.
The snapshot includes control-computation, CAN-call, aggregate motor-snapshot,
P/D wrench, task-torque, and final-command timing/signals.

Example:
    python examples/record_cartesian_pd_motion.py L5 can0 --axis rx
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
REPO_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))
import arx5_interface as arx5
from tune_cartesian_pd import (
    AXES,
    MOTION_FREQUENCY_HZ,
    MOTION_RAMP_DURATION_S,
    MOTION_ROTATION_AMPLITUDE_RAD,
    MOTION_TRANSLATION_AMPLITUDE_M,
    SinusoidalMotion,
    default_config_path,
    load_gain_config,
    preflight_sinusoidal_motion,
    validate_filter_cutoff,
    validate_gain_vector,
)


CARTESIAN_AXES = ("x", "y", "z", "rx", "ry", "rz")
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
HOME_FINAL_SETTLE_S = 0.6
CONTROL_MODES = ("host-cartesian", "hybrid-cartesian", "motor-local")
MOTOR_KP_MAX = 500.0
MOTOR_KD_MAX = 5.0
COMPARISON_MOTOR_KD = np.array([5, 5, 5, 1, 1, 1], dtype=np.float64)


def validate_motor_gain_vector(values, name, maximum) -> np.ndarray:
    """Validate gains before they reach the packed MIT-mode fields."""
    result = validate_gain_vector(values, name)
    if np.any(result > maximum):
        index = int(np.argmax(result))
        raise ValueError(
            f"{name} joint {index + 1} is {result[index]:g}; "
            f"the motor protocol maximum is {maximum:g}"
        )
    return result


def project_cartesian_gains_to_joint_diagonal(jacobian, kp, kd):
    """Return the diagonal motor gains matching J.T K J at one pose."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must be a finite 6x6 matrix")
    kp = validate_gain_vector(kp, "Cartesian Kp")
    kd = validate_gain_vector(kd, "Cartesian Kd")
    joint_kp = np.diag(jacobian.T @ np.diag(kp) @ jacobian)
    joint_kd = np.diag(jacobian.T @ np.diag(kd) @ jacobian)
    return joint_kp, joint_kd


class LocalJointCommandGenerator:
    """Map a Cartesian reference into a continuous local-PD joint waypoint."""

    def __init__(self, solver, kinematics, initial_joint_position, joint_velocity_max):
        self.solver = solver
        self.kinematics = kinematics
        self.previous_joint_position = np.asarray(
            initial_joint_position, dtype=np.float64
        ).copy()
        self.joint_velocity_max = np.asarray(
            joint_velocity_max, dtype=np.float64
        ).copy()
        if self.previous_joint_position.shape != (6,):
            raise ValueError("initial_joint_position must contain six values")
        if self.joint_velocity_max.shape != (6,) or np.any(
            self.joint_velocity_max <= 0.0
        ):
            raise ValueError("joint_velocity_max must contain six positive values")

    def make_command(self, pose, twist, gripper_position, timestamp):
        pose = _as_six(pose)
        twist = _as_six(twist)
        ik_status, joint_position = self.solver.inverse_kinematics(
            pose, self.previous_joint_position
        )
        joint_position = _as_dof(joint_position, 6)
        if ik_status != 0:
            try:
                status_name = self.solver.get_ik_status_name(ik_status)
            except (AttributeError, TypeError):
                status_name = "unknown"
            raise RuntimeError(
                f"live IK failed: {status_name} ({ik_status})"
            )

        jacobian = np.asarray(
            self.kinematics.jacobian(joint_position), dtype=np.float64
        )
        if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
            raise RuntimeError("live IK returned an invalid Jacobian")
        try:
            joint_velocity = np.linalg.solve(jacobian, twist)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("live IK reached a singular Jacobian") from exc
        if not np.all(np.isfinite(joint_velocity)):
            raise RuntimeError("live IK produced a non-finite joint velocity")
        speed_ratio = np.abs(joint_velocity) / self.joint_velocity_max
        if np.max(speed_ratio) > 1.0 + 1e-9:
            index = int(np.argmax(speed_ratio))
            raise RuntimeError(
                f"live IK requires joint {index + 1} velocity "
                f"{joint_velocity[index]:.3f} rad/s, above its "
                f"{self.joint_velocity_max[index]:.3f} rad/s limit"
            )

        command = arx5.JointState(
            joint_position,
            joint_velocity,
            np.zeros(6, dtype=np.float64),
            float(gripper_position),
        )
        command.timestamp = float(timestamp)
        self.previous_joint_position = joint_position.copy()
        return command


def recording_columns(joint_dof: int = 6) -> list[str]:
    columns = [
        "host_elapsed_s",
        "controller_time_s",
        "sample_dt_s",
        "schedule_lateness_s",
        "telemetry_read_duration_s",
        "command_update",
        "command_call_duration_s",
        "cartesian_pd_enabled",
        "cartesian_damping_mode",
        "cartesian_pseudoinverse_damping",
        "cartesian_joint_velocity_scale",
        "cartesian_joint_velocity_scaled",
        "control_cycle_sequence",
        "control_cycle_start_s",
        "control_compute_start_s",
        "control_compute_end_s",
        "feedback_used_timestamp_s",
        "motor_snapshot_read_start_s",
        "motor_snapshot_read_end_s",
        "feedback_publish_timestamp_s",
        "control_cycle_end_s",
    ]
    for prefix in (
        "source_pose",
        "source_twist",
        "desired_pose",
        "desired_twist",
        "measured_pose",
        "pose_error",
        "raw_twist",
        "filtered_twist",
        "proportional_wrench",
        "derivative_wrench",
        "clamped_wrench",
    ):
        columns.extend(f"{prefix}_{axis}" for axis in CARTESIAN_AXES)
    for prefix in (
        "control_joint_position",
        "control_joint_velocity",
        "control_joint_torque",
        "joint_position",
        "joint_velocity",
        "filtered_joint_velocity",
        "joint_torque_measured",
        "requested_cartesian_task_torque",
        "slewed_cartesian_task_torque",
        "host_cartesian_stiffness_torque",
        "host_cartesian_damping_torque",
        "gravity_torque",
        "desired_joint_velocity_unlimited",
        "desired_joint_velocity_limited",
        "packed_motor_kp",
        "packed_motor_kd",
        "motor_protocol_velocity",
        "joint_position_command",
        "joint_velocity_command",
        "joint_torque_command",
        "joint_send_call_start_s",
        "joint_send_call_end_s",
    ):
        columns.extend(f"{prefix}_{index + 1}" for index in range(joint_dof))
    return columns


def _as_six(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("recorded Cartesian values must contain six finite values")
    return result


def _as_dof(values, joint_dof: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (joint_dof,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"recorded joint values must contain {joint_dof} finite values"
        )
    return result


def capture_row(
    controller,
    kinematics,
    source_pose,
    source_twist,
    host_elapsed_s,
    sample_dt_s,
    schedule_lateness_s,
    command_update,
    command_call_duration_s,
):
    read_start = time.monotonic()
    telemetry = controller._get_control_cycle_telemetry()
    read_duration = time.monotonic() - read_start

    joint_dof = len(np.asarray(telemetry.feedback_joint_position))

    # The joint controller has no task-space control telemetry, but its joint
    # feedback is still one atomically published controller-cycle snapshot.
    # Derive the task-space measurement from that same snapshot for the
    # motor-local comparison instead of mixing several Python getters.
    if telemetry.has_cartesian_data:
        desired_pose = telemetry.desired_pose
        desired_twist = telemetry.desired_twist
        measured_pose = telemetry.measured_pose
        pose_error = telemetry.pose_error
        raw_twist = telemetry.raw_twist
        filtered_twist = telemetry.filtered_twist
        proportional_wrench = telemetry.proportional_wrench
        derivative_wrench = telemetry.derivative_wrench
        clamped_wrench = telemetry.clamped_wrench
    else:
        feedback_position = _as_dof(
            telemetry.feedback_joint_position, joint_dof
        )
        feedback_velocity = _as_dof(
            telemetry.feedback_joint_velocity, joint_dof
        )
        filtered_velocity = _as_dof(
            telemetry.filtered_joint_velocity, joint_dof
        )
        jacobian = np.asarray(
            kinematics.jacobian(feedback_position), dtype=np.float64
        )
        desired_pose = source_pose
        desired_twist = source_twist
        measured_pose = kinematics.pose(feedback_position)
        pose_error = np.zeros(6, dtype=np.float64)
        raw_twist = jacobian @ feedback_velocity
        filtered_twist = (
            jacobian @ filtered_velocity
            if np.any(filtered_velocity)
            else raw_twist.copy()
        )
        proportional_wrench = np.zeros(6, dtype=np.float64)
        derivative_wrench = np.zeros(6, dtype=np.float64)
        clamped_wrench = np.zeros(6, dtype=np.float64)

    row = [
        float(host_elapsed_s),
        float(telemetry.feedback_publish_timestamp_s),
        float(sample_dt_s),
        float(schedule_lateness_s),
        float(read_duration),
        int(bool(command_update)),
        float(command_call_duration_s),
        int(bool(telemetry.cartesian_pd_enabled)),
        int(getattr(telemetry, "cartesian_damping_mode", 0)),
        float(getattr(telemetry, "cartesian_pseudoinverse_damping", 0.0)),
        float(getattr(telemetry, "cartesian_joint_velocity_scale", 1.0)),
        int(bool(getattr(telemetry, "cartesian_joint_velocity_scaled", False))),
        int(telemetry.sequence),
        float(telemetry.cycle_start_s),
        float(telemetry.control_compute_start_s),
        float(telemetry.control_compute_end_s),
        float(telemetry.feedback_used_timestamp_s),
        float(telemetry.motor_snapshot_read_start_s),
        float(telemetry.motor_snapshot_read_end_s),
        float(telemetry.feedback_publish_timestamp_s),
        float(telemetry.cycle_end_s),
    ]
    row.extend(_as_six(source_pose))
    row.extend(_as_six(source_twist))
    for values in (
        desired_pose,
        desired_twist,
        measured_pose,
        pose_error,
        raw_twist,
        filtered_twist,
        proportional_wrench,
        derivative_wrench,
        clamped_wrench,
    ):
        row.extend(_as_six(values))
    for values in (
        telemetry.control_joint_position,
        telemetry.control_joint_velocity,
        telemetry.control_joint_torque,
        telemetry.feedback_joint_position,
        telemetry.feedback_joint_velocity,
        telemetry.filtered_joint_velocity,
        telemetry.feedback_joint_torque,
        telemetry.requested_cartesian_task_torque,
        telemetry.slewed_cartesian_task_torque,
        getattr(telemetry, "host_cartesian_stiffness_torque", np.zeros(joint_dof)),
        getattr(telemetry, "host_cartesian_damping_torque", np.zeros(joint_dof)),
        getattr(telemetry, "gravity_torque", np.zeros(joint_dof)),
        getattr(telemetry, "desired_joint_velocity_unlimited", np.zeros(joint_dof)),
        getattr(telemetry, "desired_joint_velocity_limited", np.zeros(joint_dof)),
        getattr(telemetry, "final_motor_kp", np.zeros(joint_dof)),
        getattr(telemetry, "final_motor_kd", np.zeros(joint_dof)),
        getattr(telemetry, "final_motor_protocol_velocity", np.zeros(joint_dof)),
        telemetry.final_joint_position_command,
        telemetry.final_joint_velocity_command,
        telemetry.final_joint_torque_command,
        telemetry.joint_send_call_start_s,
        telemetry.joint_send_call_end_s,
    ):
        row.extend(_as_dof(values, joint_dof))
    return [float(value) for value in row]


def write_csv(path: Path, columns: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        writer.writerows(rows)


def write_recording_artifacts(prefix: Path, columns, rows, metadata) -> None:
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    html_path = prefix.with_suffix(".html")
    write_csv(csv_path, columns, rows)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    write_html_report(html_path, columns, rows, metadata)
    click.echo(f"CSV:    {csv_path}")
    click.echo(f"Report: {html_path}")
    click.echo(f"Meta:   {json_path}")


def validate_paired_anchor(
    first_joint_position,
    first_pose,
    second_joint_position,
    second_pose,
) -> None:
    """Refuse a comparison whose two conditions do not share one anchor."""
    joint_delta = np.abs(_as_dof(second_joint_position, 6) - _as_dof(first_joint_position, 6))
    pose_delta = np.asarray(
        arx5._cartesian_pose_error(_as_six(first_pose), _as_six(second_pose)),
        dtype=np.float64,
    )
    if np.max(joint_delta) > 0.05:
        raise ValueError(f"paired joint anchor differs by {np.max(joint_delta):.4f} rad (limit 0.05 rad)")
    if np.linalg.norm(pose_delta[:3]) > 0.005:
        raise ValueError("paired TCP anchor translation differs by more than 5 mm")
    if np.linalg.norm(pose_delta[3:]) > np.deg2rad(3.0):
        raise ValueError("paired TCP anchor orientation differs by more than 3 degrees")


def condition_metrics(columns, rows, axis: str, frequency_hz=MOTION_FREQUENCY_HZ):
    """Compute controller-neutral metrics used by the paired report."""
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("condition metrics require at least two samples")
    index = {name: offset for offset, name in enumerate(columns)}
    axis_index = CARTESIAN_AXES.index(axis)
    time_values = data[:, index["host_elapsed_s"]]
    source = data[:, index[f"source_pose_{axis}"]]
    measured = data[:, index[f"measured_pose_{axis}"]]
    desired = data[:, index[f"desired_pose_{axis}"]]
    error = desired - measured
    omega = 2.0 * np.pi * frequency_hz

    def fit_amplitude_phase(values):
        design = np.column_stack((np.sin(omega * time_values), np.cos(omega * time_values), np.ones(len(time_values))))
        sine, cosine, _ = np.linalg.lstsq(design, values, rcond=None)[0]
        return float(np.hypot(sine, cosine)), float(np.arctan2(cosine, sine))

    source_amplitude, source_phase = fit_amplitude_phase(source)
    measured_amplitude, measured_phase = fit_amplitude_phase(measured)
    phase_lag = (source_phase - measured_phase + np.pi) % (2.0 * np.pi) - np.pi
    cross_axes = [name for name in CARTESIAN_AXES if name != axis]
    cross_motion = np.column_stack(
        [data[:, index[f"measured_pose_{name}"]] - data[0, index[f"measured_pose_{name}"]] for name in cross_axes]
    )

    def vector_stats(prefix):
        values = np.column_stack([data[:, index[f"{prefix}_{joint}"]] for joint in range(1, 7)])
        return {
            "rms": np.sqrt(np.mean(np.square(values), axis=0)).tolist(),
            "peak": np.max(np.abs(values), axis=0).tolist(),
        }

    return {
        "selected_axis_rms_tracking_error": float(np.sqrt(np.mean(np.square(error)))),
        "selected_axis_peak_tracking_error": float(np.max(np.abs(error))),
        "amplitude_ratio": measured_amplitude / source_amplitude if source_amplitude > 1e-12 else 0.0,
        "phase_lag_deg": float(np.rad2deg(phase_lag)),
        "cross_axis_rms_motion": np.sqrt(np.mean(np.square(cross_motion), axis=0)).tolist(),
        "cross_axis_peak_motion": np.max(np.abs(cross_motion), axis=0).tolist(),
        "joint_velocity": vector_stats("joint_velocity"),
        "measured_joint_torque": vector_stats("joint_torque_measured"),
        "host_stiffness_torque": vector_stats("host_cartesian_stiffness_torque"),
        "final_t_ff": vector_stats("joint_torque_command"),
        "packed_motor_kp": vector_stats("packed_motor_kp"),
        "packed_motor_kd": vector_stats("packed_motor_kd"),
        "packed_motor_velocity": vector_stats("motor_protocol_velocity"),
        "sample_count": int(len(data)),
        "missed_samples": int(np.count_nonzero(data[:, index["sample_dt_s"]] > 0.003)),
        "max_control_cycle_s": float(np.max(data[:, index["control_cycle_end_s"]] - data[:, index["control_cycle_start_s"]])),
        "selected_axis_index": axis_index,
    }


def write_paired_comparison_report(path: Path, hybrid_metrics: dict, motor_local_metrics: dict) -> None:
    """Write JSON and dependency-free HTML summaries without ranking either controller."""
    payload = {
        "conditions": {
            "hybrid-cartesian": hybrid_metrics,
            "motor-local": motor_local_metrics,
        },
        "interpretation": (
            "Both conditions use motor-local damping; this comparison isolates host Cartesian stiffness "
            "from motor-local joint stiffness and does not assert that either must outperform the other."
        ),
    }
    json_path = path.with_suffix(".json")
    html_path = path.with_suffix(".html")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    cards = []
    for label, metrics in payload["conditions"].items():
        cards.append(
            f"<h2>{html.escape(label)}</h2><pre>{html.escape(json.dumps(metrics, indent=2))}</pre>"
        )
    html_path.write_text(
        "<html><head><meta charset=\"utf-8\"><title>Cartesian stiffness comparison</title></head><body>"
        "<h1>Hybrid Cartesian vs motor-local stiffness</h1>"
        f"<p>{html.escape(payload['interpretation'])}</p>{''.join(cards)}</body></html>"
    )
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["condition", "rms_tracking_error", "peak_tracking_error", "amplitude_ratio", "phase_lag_deg",
             "missed_samples", "max_control_cycle_s"]
        )
        for label, metrics in payload["conditions"].items():
            writer.writerow(
                [label, metrics["selected_axis_rms_tracking_error"], metrics["selected_axis_peak_tracking_error"],
                 metrics["amplitude_ratio"], metrics["phase_lag_deg"], metrics["missed_samples"],
                 metrics["max_control_cycle_s"]]
            )


def return_home_after_recording(
    controller,
    control_mode: str = "host-cartesian",
    settle_duration_s: float = HOME_FINAL_SETTLE_S,
    sleep_fn=time.sleep,
) -> None:
    """Leave the test mode and complete the SDK's final home segment."""
    if control_mode in ("host-cartesian", "hybrid-cartesian"):
        controller.disable_cartesian_pd()
    elif control_mode == "motor-local":
        controller.set_to_damping()
    else:
        raise ValueError(f"unknown control mode: {control_mode}")
    controller.reset_to_home()
    # reset_to_home() schedules its last joint-space segment 0.5 s ahead and
    # returns immediately. Keep the controller alive until that segment lands.
    sleep_fn(settle_duration_s)


def _svg_plot(
    title: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    width: int = 1100,
    height: int = 230,
) -> str:
    left, right, top, bottom = 72, 18, 30, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    if x.size == 0:
        return ""
    stride = max(1, math.ceil(x.size / 2000))
    plotted_x = x[::stride]
    plotted_series = [(label, values[::stride]) for label, values in series]
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for _, values in plotted_series]
    )
    y_min = float(np.min(finite_values)) if finite_values.size else -1.0
    y_max = float(np.max(finite_values)) if finite_values.size else 1.0
    if abs(y_max - y_min) < 1e-12:
        padding = max(1e-6, abs(y_max) * 0.1)
        y_min -= padding
        y_max += padding
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding
    x_min, x_max = float(plotted_x[0]), float(plotted_x[-1])
    if x_max <= x_min:
        x_max = x_min + 1.0

    def point_x(value):
        return left + (float(value) - x_min) / (x_max - x_min) * plot_width

    def point_y(value):
        return top + (y_max - float(value)) / (y_max - y_min) * plot_height

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="20" class="title">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="plot-bg"/>',
    ]
    for grid_index in range(5):
        fraction = grid_index / 4.0
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.4g}</text>'
        )
    for index, (label, values) in enumerate(plotted_series):
        points = " ".join(
            f"{point_x(x_value):.2f},{point_y(y_value):.2f}"
            for x_value, y_value in zip(plotted_x, values)
            if np.isfinite(y_value)
        )
        color = COLORS[index % len(COLORS)]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.25"/>'
        )
        legend_x = left + index * 185
        parts.append(
            f'<line x1="{legend_x}" y1="{height - 14}" x2="{legend_x + 20}" y2="{height - 14}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{legend_x + 25}" y="{height - 10}" class="legend">{html.escape(label)}</text>'
        )
    parts.extend(
        [
            f'<text x="{left}" y="{top + plot_height + 17}" class="tick">{x_min:.3f} s</text>',
            f'<text x="{left + plot_width}" y="{top + plot_height + 17}" text-anchor="end" class="tick">{x_max:.3f} s</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def write_html_report(
    path: Path,
    columns: list[str],
    rows: list[list[float]],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(rows, dtype=np.float64)
    column = {name: data[:, index] for index, name in enumerate(columns)}
    elapsed = column["host_elapsed_s"]
    plots = []
    for axis in CARTESIAN_AXES:
        plots.append(
            _svg_plot(
                f"Cartesian pose {axis}",
                elapsed,
                [
                    ("source", column[f"source_pose_{axis}"]),
                    ("500 Hz desired", column[f"desired_pose_{axis}"]),
                    ("measured", column[f"measured_pose_{axis}"]),
                ],
            )
        )
        plots.append(
            _svg_plot(
                f"Cartesian twist {axis}",
                elapsed,
                [
                    ("source", column[f"source_twist_{axis}"]),
                    ("500 Hz desired", column[f"desired_twist_{axis}"]),
                    ("measured raw", column[f"raw_twist_{axis}"]),
                    ("measured filtered", column[f"filtered_twist_{axis}"]),
                ],
            )
        )
        plots.append(
            _svg_plot(
                f"Cartesian wrench {axis}",
                elapsed,
                [
                    ("P", column[f"proportional_wrench_{axis}"]),
                    ("D", column[f"derivative_wrench_{axis}"]),
                    ("clamped P+D", column[f"clamped_wrench_{axis}"]),
                ],
            )
        )
    cartesian_plot_count = len(plots)
    for joint in range(1, 7):
        plots.append(
            _svg_plot(
                f"Joint {joint} position",
                elapsed,
                [
                    ("command", column[f"joint_position_command_{joint}"]),
                    ("measured", column[f"joint_position_{joint}"]),
                ],
            )
        )
        plots.append(
            _svg_plot(
                f"Joint {joint} velocity",
                elapsed,
                [
                    ("measured", column[f"joint_velocity_{joint}"]),
                    ("filtered", column[f"filtered_joint_velocity_{joint}"]),
                ],
            )
        )
        plots.append(
            _svg_plot(
                f"Joint {joint} torque",
                elapsed,
                [
                    (
                        "task requested",
                        column[f"requested_cartesian_task_torque_{joint}"],
                    ),
                    (
                        "task after slew",
                        column[f"slewed_cartesian_task_torque_{joint}"],
                    ),
                    ("final sent", column[f"joint_torque_command_{joint}"]),
                    ("measured", column[f"joint_torque_measured_{joint}"]),
                ],
            )
        )
    joint_plot_end = len(plots)
    plots.append(
        _svg_plot(
            "Recorder timing",
            elapsed,
            [
                ("sample dt (ms)", 1000.0 * column["sample_dt_s"]),
                ("lateness (ms)", 1000.0 * column["schedule_lateness_s"]),
                (
                    "telemetry read (ms)",
                    1000.0 * column["telemetry_read_duration_s"],
                ),
                (
                    "command call (ms)",
                    1000.0 * column["command_call_duration_s"],
                ),
            ],
        )
    )
    send_durations = np.vstack(
        [
            1000.0
            * (
                column[f"joint_send_call_end_s_{joint}"]
                - column[f"joint_send_call_start_s_{joint}"]
            )
            for joint in range(1, 7)
        ]
    )
    plots.append(
        _svg_plot(
            "500 Hz controller timing",
            elapsed,
            [
                (
                    "compute (ms)",
                    1000.0
                    * (
                        column["control_compute_end_s"]
                        - column["control_compute_start_s"]
                    ),
                ),
                (
                    "cycle (ms)",
                    1000.0
                    * (
                        column["control_cycle_end_s"]
                        - column["control_cycle_start_s"]
                    ),
                ),
                (
                    "feedback age (ms)",
                    1000.0
                    * (
                        column["control_compute_start_s"]
                        - column["feedback_used_timestamp_s"]
                    ),
                ),
                (
                    "snapshot read (ms)",
                    1000.0
                    * (
                        column["motor_snapshot_read_end_s"]
                        - column["motor_snapshot_read_start_s"]
                    ),
                ),
            ],
        )
    )
    plots.append(
        _svg_plot(
            "Per-joint CAN call timing",
            elapsed,
            [
                ("max call duration (ms)", np.max(send_durations, axis=0)),
                *[
                    (
                        f"joint {joint} start offset (ms)",
                        1000.0
                        * (
                            column[f"joint_send_call_start_s_{joint}"]
                            - column["control_cycle_start_s"]
                        ),
                    )
                    for joint in range(1, 7)
                ],
            ],
        )
    )
    summary = metadata["summary"]
    mode_label = metadata.get("control_mode", "host-cartesian")
    metadata_json = html.escape(json.dumps(metadata, indent=2))
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cartesian motion recording</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
h1, h2 {{ margin-bottom: 8px; }} .summary {{ display:flex; gap:24px; flex-wrap:wrap; }}
.card {{ border:1px solid #d7dce5; border-radius:8px; padding:10px 14px; }}
svg {{ width:100%; max-width:1100px; border:1px solid #e2e5eb; margin:8px 0; }}
.plot-bg {{ fill:#fff; }} .grid {{ stroke:#e5e7eb; stroke-width:1; }}
.title {{ font-size:14px; font-weight:600; }} .tick,.legend {{ font-size:10px; fill:#475569; }}
pre {{ background:#f6f8fa; padding:12px; overflow:auto; }}
</style></head><body>
<h1>Cartesian motion recording — {html.escape(mode_label)}</h1>
<div class="summary">
<div class="card"><b>Samples</b><br>{summary['sample_count']}</div>
<div class="card"><b>Observed rate</b><br>{summary['observed_rate_hz']:.1f} Hz</div>
<div class="card"><b>Missed 2 ms deadlines</b><br>{summary['missed_sample_deadlines']}</div>
<div class="card"><b>Max sample dt</b><br>{1000.0 * summary['max_sample_dt_s']:.3f} ms</div>
<div class="card"><b>Max command call</b><br>{1000.0 * summary['max_command_call_duration_s']:.3f} ms</div>
</div>
<h2>Cartesian signals</h2>{''.join(plots[:cartesian_plot_count])}
<h2>Joint signals</h2>{''.join(plots[cartesian_plot_count:joint_plot_end])}
<h2>Timing</h2>{''.join(plots[joint_plot_end:])}
<h2>Metadata</h2><pre>{metadata_json}</pre>
</body></html>"""
    path.write_text(document)


def build_metadata(
    model,
    interface,
    axis,
    kp,
    kd,
    filter_cutoff_hz,
    sample_dt_s,
    command_dt_s,
    cycles,
    rows,
    failure,
    return_home=None,
    control_mode="host-cartesian",
    motor_kp=None,
    motor_kd=None,
    anchor_joint_position=None,
    anchor_pose=None,
):
    data = np.asarray(rows, dtype=np.float64)
    columns = recording_columns()
    column_index = {name: index for index, name in enumerate(columns)}
    sample_intervals = data[:, 2] if data.size else np.zeros(0)
    command_durations = data[:, 6] if data.size else np.zeros(0)
    sequences = (
        data[:, column_index["control_cycle_sequence"]]
        if data.size
        else np.zeros(0)
    )
    sequence_steps = np.diff(sequences)
    compute_durations = (
        data[:, column_index["control_compute_end_s"]]
        - data[:, column_index["control_compute_start_s"]]
        if data.size
        else np.zeros(0)
    )
    cycle_durations = (
        data[:, column_index["control_cycle_end_s"]]
        - data[:, column_index["control_cycle_start_s"]]
        if data.size
        else np.zeros(0)
    )
    duration = float(data[-1, 0] - data[0, 0]) if len(rows) > 1 else 0.0
    summary = {
        "sample_count": len(rows),
        "duration_s": duration,
        "observed_rate_hz": (len(rows) - 1) / duration if duration > 0 else 0.0,
        "missed_sample_deadlines": int(
            np.count_nonzero(sample_intervals > 1.5 * sample_dt_s)
        ),
        "max_sample_dt_s": float(np.max(sample_intervals)) if len(rows) else 0.0,
        "max_command_call_duration_s": (
            float(np.max(command_durations)) if len(rows) else 0.0
        ),
        "unique_control_cycles": int(np.unique(sequences).size),
        "duplicate_control_cycle_rows": int(np.count_nonzero(sequence_steps == 0)),
        "control_cycles_skipped_by_recorder": int(
            np.sum(np.maximum(sequence_steps - 1, 0))
        ),
        "max_control_compute_duration_s": (
            float(np.max(compute_durations)) if len(rows) else 0.0
        ),
        "max_control_cycle_duration_s": (
            float(np.max(cycle_durations)) if len(rows) else 0.0
        ),
    }
    return {
        "model": model,
        "interface": interface,
        "control_mode": control_mode,
        "axis": axis,
        "translation_amplitude_m": MOTION_TRANSLATION_AMPLITUDE_M,
        "rotation_amplitude_rad": MOTION_ROTATION_AMPLITUDE_RAD,
        "motion_frequency_hz": MOTION_FREQUENCY_HZ,
        "motion_ramp_duration_s": MOTION_RAMP_DURATION_S,
        "cycles_before_return": cycles,
        "requested_sample_dt_s": sample_dt_s,
        "requested_command_dt_s": command_dt_s,
        "kp": np.asarray(kp).tolist(),
        "kd": np.asarray(kd).tolist(),
        "motor_kp": None if motor_kp is None else np.asarray(motor_kp).tolist(),
        "motor_kd": None if motor_kd is None else np.asarray(motor_kd).tolist(),
        "anchor_joint_position": None if anchor_joint_position is None else np.asarray(anchor_joint_position).tolist(),
        "anchor_pose": None if anchor_pose is None else np.asarray(anchor_pose).tolist(),
        "filter_cutoff_hz": filter_cutoff_hz,
        "failure": failure,
        "return_home": return_home,
        "timestamp_semantics": {
            "clock": "controller-relative std::chrono::steady_clock seconds",
            "joint_send_call_start_end": (
                "brackets the call into the prebuilt CAN library; it is not a hardware bus timestamp"
            ),
            "feedback_publish_timestamp": (
                "time the latest aggregate motor snapshot was copied into joint_state; per-frame receive timestamps are unavailable"
            ),
            "motor_snapshot_read_start_end": (
                "brackets get_motor_msg() on the prebuilt CAN library's latest aggregate snapshot"
            ),
            "motor_local_torque": (
                "motor-local Kp/Kd torque is evaluated inside each drive and is not included in joint_torque_command; compare returned joint_torque_measured and motion instead"
            ),
        },
        "summary": summary,
        "condition_metrics": condition_metrics(columns, rows, axis) if len(rows) > 1 else None,
    }


def record_one_condition(
    controller,
    kinematics,
    solver,
    robot_config,
    controller_config,
    control_mode,
    axis_index,
    anchor_pose,
    measured_joint_state,
    cycles,
    sample_dt,
    command_dt,
    settle_duration,
    amplitude_text,
):
    """Run one sinusoidal condition and finish back at its test anchor."""
    rows = []
    local_command_generator = None
    if control_mode == "motor-local":
        local_command_generator = LocalJointCommandGenerator(
            solver,
            kinematics,
            np.asarray(measured_joint_state.pos(), dtype=np.float64),
            np.asarray(robot_config.joint_vel_max, dtype=np.float64),
        )

    motion = SinusoidalMotion()
    host_start = time.monotonic()
    motion.start(axis_index, anchor_pose, measured_joint_state.gripper_pos, host_start)
    return_time = host_start + cycles / MOTION_FREQUENCY_HZ
    return_started = False
    settle_end = None
    next_sample = host_start
    next_command = host_start
    previous_sample = host_start
    axis = AXES[axis_index]
    click.echo(
        f"Recording {axis} {amplitude_text} in {control_mode} mode at "
        f"{1.0 / sample_dt:.0f} Hz; Ctrl-C stops the test."
    )

    while True:
        now = time.monotonic()
        if now < next_sample:
            time.sleep(next_sample - now)
            now = time.monotonic()
        lateness = max(0.0, now - next_sample)
        sample_interval = now - previous_sample
        previous_sample = now
        skipped_slots = max(0, int((now - next_sample) // sample_dt))
        next_sample += (skipped_slots + 1) * sample_dt

        if not return_started and now >= return_time:
            motion.begin_return(now)
            return_started = True
        source_pose, source_twist = motion.command_reference(now)
        if return_started and not motion.active and settle_end is None:
            settle_end = now + settle_duration

        command_update = now >= next_command
        command_duration = 0.0
        if command_update:
            command_timestamp = controller.get_timestamp() + max(
                command_dt, 2.0 * controller_config.controller_dt
            )
            command_start = time.monotonic()
            if control_mode in ("host-cartesian", "hybrid-cartesian"):
                command = arx5.EEFState(source_pose, motion.gripper_position)
                command.timestamp = command_timestamp
                controller.set_eef_cmd_with_twist(command, source_twist)
            else:
                command = local_command_generator.make_command(
                    source_pose,
                    source_twist,
                    motion.gripper_position,
                    command_timestamp,
                )
                controller.set_joint_cmd(command)
            command_duration = time.monotonic() - command_start
            skipped_commands = max(0, int((now - next_command) // command_dt))
            next_command += (skipped_commands + 1) * command_dt

        rows.append(
            capture_row(
                controller,
                kinematics,
                source_pose,
                source_twist,
                now - host_start,
                sample_interval,
                lateness,
                command_update,
                command_duration,
            )
        )
        if control_mode in ("host-cartesian", "hybrid-cartesian") and not controller.is_cartesian_pd_enabled():
            return rows, "Cartesian PD disabled itself during recording"
        if settle_end is not None and now >= settle_end:
            return rows, None


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--axis", type=click.Choice(AXES), required=True)
@click.option(
    "--control-mode",
    type=click.Choice(CONTROL_MODES),
    default="host-cartesian",
    show_default=True,
    help=(
        "Use host Cartesian P+D, hybrid host stiffness with motor-local "
        "damping, or motor-local joint MIT-mode PD."
    ),
)
@click.option(
    "--output-prefix",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output prefix; .csv, .json, and .html are added automatically.",
)
@click.option(
    "--paired-comparison",
    is_flag=True,
    help="Run the Trossen_L5/can2 hybrid-vs-motor-local experiment on the selected axis.",
)
@click.option(
    "--required-anchor-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    hidden=True,
)
@click.option(
    "--gain-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Gain JSON from tune_cartesian_pd.py; defaults to the model file.",
)
@click.option("--kp", type=float, nargs=6, metavar="X Y Z RX RY RZ")
@click.option("--kd", type=float, nargs=6, metavar="X Y Z RX RY RZ")
@click.option(
    "--motor-kp",
    type=float,
    nargs=6,
    metavar="J1 J2 J3 J4 J5 J6",
    help=(
        "Motor-local joint Kp; defaults to diag(J.T Cartesian-Kp J) at "
        "the anchored pose."
    ),
)
@click.option(
    "--motor-kd",
    type=float,
    nargs=6,
    metavar="J1 J2 J3 J4 J5 J6",
    help=(
        "Motor-local joint Kd; defaults to the comparison vector "
        "[5,5,5,1,1,1]."
    ),
)
@click.option("--filter-cutoff-hz", type=click.FloatRange(min=0.0), default=None)
@click.option("--cycles", type=click.FloatRange(min=0.25), default=2.0, show_default=True)
@click.option(
    "--sample-dt", type=click.FloatRange(min=0.001), default=0.002, show_default=True
)
@click.option(
    "--command-dt", type=click.FloatRange(min=0.002), default=0.02, show_default=True
)
@click.option(
    "--settle-duration", type=click.FloatRange(min=0.0), default=1.0, show_default=True
)
@click.option("--yes", "assume_yes", is_flag=True, help="Skip the swept-envelope confirmation.")
@click.option(
    "--skip-positioning",
    is_flag=True,
    help="Skip the gravity-compensated manual start-position phase.",
)
def main(
    model,
    interface,
    axis,
    control_mode,
    output_prefix,
    paired_comparison,
    required_anchor_json,
    gain_config,
    kp,
    kd,
    motor_kp,
    motor_kd,
    filter_cutoff_hz,
    cycles,
    sample_dt,
    command_dt,
    settle_duration,
    assume_yes,
    skip_positioning,
):
    """Record one Cartesian sinusoid on MODEL using CAN INTERFACE."""
    paired_output_prefix = None
    if paired_comparison:
        if (model, interface) != ("Trossen_L5", "can2"):
            raise click.ClickException(
                "The paired experiment is fixed to MODEL=Trossen_L5 and INTERFACE=can2; "
                "all Cartesian --axis choices are supported."
            )
        if assume_yes or skip_positioning:
            raise click.ClickException(
                "The paired experiment requires manual positioning and operator envelope confirmation."
            )
        if output_prefix is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = Path.cwd() / f"cartesian_stiffness_comparison_{axis}_{stamp}"
        paired_output_prefix = output_prefix.resolve()
        output_prefix = Path(f"{paired_output_prefix}_hybrid_cartesian")
        control_mode = "hybrid-cartesian"
        kp = (200.0, 200.0, 200.0, 16.0, 16.0, 16.0)
        kd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        motor_kd = tuple(COMPARISON_MOTOR_KD)
        cycles = 2.0
        sample_dt = 0.002
        command_dt = 0.02
        settle_duration = 1.0

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if robot_config.joint_dof != 6:
        raise click.ClickException("Cartesian recording requires a six-joint model.")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )
    if sample_dt < controller_config.controller_dt - 1e-12:
        raise click.ClickException(
            f"--sample-dt cannot be below the {controller_config.controller_dt:g} s controller period"
        )
    if command_dt < sample_dt:
        raise click.ClickException("--command-dt must be at least --sample-dt")

    if gain_config is None:
        gain_config = default_config_path(model)
    loaded_kp = loaded_kd = None
    loaded_filter = controller_config.cartesian_joint_velocity_filter_cutoff_hz
    if gain_config.exists():
        loaded_kp, loaded_kd, loaded_filter = load_gain_config(
            gain_config,
            model,
            loaded_filter,
            controller_config.controller_dt,
        )
    if kp is None:
        kp = loaded_kp
    if kd is None:
        kd = loaded_kd
    if kp is None or kd is None:
        raise click.ClickException(
            "No complete gains are available; pass --kp and --kd or provide --gain-config."
        )
    kp = validate_gain_vector(kp, "kp")
    kd = validate_gain_vector(kd, "kd")
    if np.allclose(kp, 0.0) and np.allclose(kd, 0.0):
        if control_mode in ("host-cartesian", "hybrid-cartesian"):
            raise click.ClickException(
                "Refusing to record Cartesian motion with all gains zero."
            )
    if control_mode == "hybrid-cartesian" and np.allclose(kp, 0.0):
        raise click.ClickException("Hybrid Cartesian motion requires nonzero Cartesian Kp.")
    if motor_kp is not None:
        motor_kp = validate_motor_gain_vector(
            motor_kp, "motor Kp", MOTOR_KP_MAX
        )
    if motor_kd is not None:
        motor_kd = validate_motor_gain_vector(
            motor_kd, "motor Kd", MOTOR_KD_MAX
        )
    filter_cutoff_hz = validate_filter_cutoff(
        loaded_filter if filter_cutoff_hz is None else filter_cutoff_hz,
        controller_config.controller_dt,
    )
    controller_config.cartesian_joint_velocity_filter_cutoff_hz = filter_cutoff_hz
    controller_config.background_send_recv = True
    controller_config.gravity_compensation = True
    controller_config.friction_compensation = False
    controller_config.joint_friction_compensation = False
    controller_config.gripper_friction_compensation = False
    required_anchor = None
    if required_anchor_json is not None:
        required_anchor = json.loads(required_anchor_json.read_text())
        if required_anchor.get("anchor_joint_position") is None or required_anchor.get("anchor_pose") is None:
            raise click.ClickException("required anchor metadata is missing from the first condition")

    axis_index = AXES.index(axis)
    amplitude_text = (
        f"±{MOTION_TRANSLATION_AMPLITUDE_M * 1000:g} mm"
        if axis_index < 3
        else f"±{np.rad2deg(MOTION_ROTATION_AMPLITUDE_RAD):g}°"
    )
    if output_prefix is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_prefix = "cartesian_pd" if control_mode == "host-cartesian" else control_mode.replace("-", "_")
        output_prefix = Path.cwd() / f"{mode_prefix}_{model}_{axis}_{stamp}"
    output_prefix = output_prefix.resolve()
    columns = recording_columns(robot_config.joint_dof)
    rows = []
    paired_hybrid_rows = None
    failure = None
    cancelled = False
    controller = None
    controller_cleanup_complete = False
    phase = "setup"
    return_home_status = {
        "requested_after_clean_recording": True,
        "attempted": False,
        "succeeded": False,
    }

    try:
        controller_class = (
            arx5.Arx5CartesianController
            if control_mode != "motor-local"
            else arx5.Arx5JointController
        )
        controller = controller_class(robot_config, controller_config, interface)
        controller.set_log_level(arx5.LogLevel.WARNING)
        zero_joint_gain = arx5.Gain(robot_config.joint_dof)
        zero_joint_gain.kp()[:] = 0.0
        zero_joint_gain.kd()[:] = 0.0
        zero_joint_gain.gripper_kp = controller_config.default_gripper_kp
        zero_joint_gain.gripper_kd = controller_config.default_gripper_kd
        controller.set_gain(zero_joint_gain)
        controller.set_gravity_compensation(True)
        controller.set_friction_compensation(False)

        if not skip_positioning:
            if required_anchor is not None:
                click.echo(
                    "Reposition to the recorded common joint anchor: "
                    + np.array2string(
                        np.asarray(required_anchor["anchor_joint_position"], dtype=np.float64),
                        precision=4,
                    )
                )
            while True:
                click.prompt(
                    "Arm is gravity-compensated with arm PD off. Move it to the recording start pose, release it, then press Enter",
                    default="",
                    show_default=False,
                )
                maximum_speed = 0.0
                stationary_start = time.monotonic()
                while time.monotonic() - stationary_start < 0.25:
                    joint_velocity = np.asarray(
                        controller.get_joint_state().vel(), dtype=np.float64
                    )
                    maximum_speed = max(
                        maximum_speed, float(np.max(np.abs(joint_velocity)))
                    )
                    time.sleep(0.01)
                if maximum_speed <= 0.10:
                    break
                click.echo(
                    f"Arm is still moving (peak {maximum_speed:.3f} rad/s); hold it still and press Enter again.",
                    err=True,
                )

        kinematics = arx5._CartesianKinematics(
            robot_config.urdf_path,
            robot_config.joint_dof,
            robot_config.base_link_name,
            robot_config.eef_link_name,
        )
        solver = arx5.Arx5Solver(
            robot_config.urdf_path,
            robot_config.joint_dof,
            np.asarray(robot_config.joint_pos_min, dtype=np.float64),
            np.asarray(robot_config.joint_pos_max, dtype=np.float64),
            robot_config.base_link_name,
            robot_config.eef_link_name,
            np.asarray(robot_config.gravity_vector, dtype=np.float64),
        )

        # Establish a measured hold before preflight and operator confirmation.
        # In motor-local mode the interpolator may still contain the pose from
        # process startup after gravity-compensated manual positioning, so it
        # must be replaced before any nonzero local stiffness is applied.
        if control_mode in ("host-cartesian", "hybrid-cartesian"):
            controller.set_cartesian_pd_gain(arx5.CartesianPDGain(kp, kd))
            controller.set_cartesian_joint_velocity_filter_cutoff(filter_cutoff_hz)
            if control_mode == "hybrid-cartesian":
                if motor_kd is None:
                    motor_kd = COMPARISON_MOTOR_KD.copy()
                motor_kd = validate_motor_gain_vector(motor_kd, "motor Kd", MOTOR_KD_MAX)
                controller.set_cartesian_damping_mode(
                    arx5.CartesianDampingMode.MotorLocalJoint
                )
                controller.set_cartesian_motor_kd(motor_kd)
            controller.enable_cartesian_pd()
        else:
            measured_for_hold = controller.get_joint_state()
            hold_command = arx5.JointState(
                np.asarray(measured_for_hold.pos(), dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                measured_for_hold.gripper_pos,
            )
            hold_command.timestamp = controller.get_timestamp()
            controller.set_joint_cmd(hold_command)
            time.sleep(2.0 * controller_config.controller_dt)

            held_joint_position = np.asarray(
                controller.get_joint_state().pos(), dtype=np.float64
            )
            if required_anchor is not None:
                validate_paired_anchor(
                    required_anchor["anchor_joint_position"],
                    required_anchor["anchor_pose"],
                    held_joint_position,
                    kinematics.pose(held_joint_position),
                )
            projected_motor_kp, projected_motor_kd = (
                project_cartesian_gains_to_joint_diagonal(
                    kinematics.jacobian(
                        held_joint_position
                        if required_anchor is None
                        else np.asarray(required_anchor["anchor_joint_position"], dtype=np.float64)
                    ),
                    kp,
                    kd,
                )
            )
            if motor_kp is None:
                motor_kp = projected_motor_kp
            if motor_kd is None:
                motor_kd = COMPARISON_MOTOR_KD.copy()
            motor_kp = validate_motor_gain_vector(
                motor_kp, "motor Kp", MOTOR_KP_MAX
            )
            motor_kd = validate_motor_gain_vector(
                motor_kd, "motor Kd", MOTOR_KD_MAX
            )
            if np.allclose(motor_kp, 0.0):
                raise click.ClickException(
                    "Motor-local motion requires nonzero motor Kp to track the trajectory."
                )
            click.echo(
                "Motor-local diagonal gains: "
                f"Kp={np.array2string(motor_kp, precision=3)}, "
                f"Kd={np.array2string(motor_kd, precision=3)}"
            )
            local_gain = arx5.Gain(
                motor_kp,
                motor_kd,
                controller_config.default_gripper_kp,
                controller_config.default_gripper_kd,
            )
            controller.set_gain(local_gain)

        time.sleep(2.0 * controller_config.controller_dt)
        if control_mode == "hybrid-cartesian":
            packed = controller._get_control_cycle_telemetry()
            if not np.allclose(np.asarray(packed.final_motor_kp), 0.0, atol=1e-12):
                raise RuntimeError("hybrid stationary hold packed nonzero arm motor Kp")
            if not np.allclose(np.asarray(packed.final_motor_kd), motor_kd, atol=1e-12):
                raise RuntimeError("hybrid stationary hold did not pack the configured motor Kd")
        measured_joint_state = controller.get_joint_state()
        measured_joint_position = np.asarray(
            measured_joint_state.pos(), dtype=np.float64
        )
        anchor_pose = np.asarray(
            kinematics.pose(measured_joint_position), dtype=np.float64
        )
        if required_anchor is not None:
            validate_paired_anchor(
                required_anchor["anchor_joint_position"],
                required_anchor["anchor_pose"],
                measured_joint_position,
                anchor_pose,
            )
        preflight_sinusoidal_motion(
            solver,
            anchor_pose,
            measured_joint_position,
            np.asarray(robot_config.joint_pos_min, dtype=np.float64),
            np.asarray(robot_config.joint_pos_max, dtype=np.float64),
            axis_index,
        )
        if not assume_yes and not click.confirm(
            f"Start pose anchored and IK preflight passed. Clear the full {amplitude_text} {axis} envelope; IK cannot check collisions. Start {control_mode} recording?"
        ):
            raise click.Abort()

        phase = "recording"
        rows, failure = record_one_condition(
            controller,
            kinematics,
            solver,
            robot_config,
            controller_config,
            control_mode,
            axis_index,
            anchor_pose,
            measured_joint_state,
            cycles,
            sample_dt,
            command_dt,
            settle_duration,
            amplitude_text,
        )

        if paired_comparison and failure is None:
            paired_hybrid_rows = rows
            phase = "transitioning to motor-local"
            returned_state = controller.get_joint_state()
            validate_paired_anchor(
                measured_joint_position,
                anchor_pose,
                returned_state.pos(),
                kinematics.pose(returned_state.pos()),
            )
            controller.disable_cartesian_pd()

            returned_state = controller.get_joint_state()
            hold_command = arx5.JointState(
                np.asarray(returned_state.pos(), dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                returned_state.gripper_pos,
            )
            hold_command.timestamp = controller.get_timestamp()
            controller.set_joint_cmd(hold_command)
            time.sleep(2.0 * controller_config.controller_dt)
            returned_state = controller.get_joint_state()
            validate_paired_anchor(
                measured_joint_position,
                anchor_pose,
                returned_state.pos(),
                kinematics.pose(returned_state.pos()),
            )

            projected_motor_kp, _ = project_cartesian_gains_to_joint_diagonal(
                kinematics.jacobian(measured_joint_position), kp, kd
            )
            motor_kp = validate_motor_gain_vector(projected_motor_kp, "motor Kp", MOTOR_KP_MAX)
            motor_kd = validate_motor_gain_vector(COMPARISON_MOTOR_KD, "motor Kd", MOTOR_KD_MAX)
            controller.set_gain(
                arx5.Gain(
                    motor_kp,
                    motor_kd,
                    controller_config.default_gripper_kp,
                    controller_config.default_gripper_kd,
                )
            )
            time.sleep(2.0 * controller_config.controller_dt)
            packed = controller._get_control_cycle_telemetry()
            if not np.allclose(np.asarray(packed.final_motor_kp), motor_kp, atol=1e-12):
                raise RuntimeError("motor-local transition did not pack the projected motor Kp")
            if not np.allclose(np.asarray(packed.final_motor_kd), motor_kd, atol=1e-12):
                raise RuntimeError("motor-local transition did not pack the configured motor Kd")
            control_mode = "motor-local"
            rows = []
            if not click.confirm(
                "Hybrid returned to the original test position and motor-local gains are armed. Start motor-local recording?"
            ):
                raise click.Abort()

            phase = "recording motor-local"
            motor_start_state = controller.get_joint_state()
            rows, failure = record_one_condition(
                controller,
                kinematics,
                solver,
                robot_config,
                controller_config,
                control_mode,
                axis_index,
                anchor_pose,
                motor_start_state,
                cycles,
                sample_dt,
                command_dt,
                settle_duration,
                amplitude_text,
            )

        if failure is None:
            phase = "returning home"
            return_home_status["attempted"] = True
            click.echo("All requested recording conditions are complete; returning to joint-space home.")
            return_home_after_recording(controller, control_mode=control_mode)
            return_home_status["succeeded"] = True
            controller_cleanup_complete = True
            phase = "complete"
    except KeyboardInterrupt:
        failure = f"operator interrupted {phase}"
        click.echo(f"Interrupted during {phase}; stopping commanded motion.", err=True)
    except click.Abort:
        cancelled = True
        failure = "operator canceled before recording"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        click.echo(f"Recording stopped: {failure}", err=True)
    finally:
        if controller is not None and not controller_cleanup_complete:
            try:
                if control_mode in ("host-cartesian", "hybrid-cartesian"):
                    controller.disable_cartesian_pd()
                else:
                    controller.set_to_damping()
            except Exception as exc:
                disable_failure = f"failed to disable Cartesian PD: {exc}"
                failure = (
                    disable_failure
                    if failure is None
                    else f"{failure}; {disable_failure}"
                )
                click.echo(disable_failure, err=True)

    if not rows:
        if cancelled:
            raise click.Abort()
        raise click.ClickException(failure or "No samples were recorded")
    metadata = build_metadata(
        model,
        interface,
        axis,
        kp,
        kd,
        filter_cutoff_hz,
        sample_dt,
        command_dt,
        cycles,
        rows,
        failure,
        return_home_status,
        control_mode,
        motor_kp,
        motor_kd,
        measured_joint_position,
        anchor_pose,
    )
    if paired_comparison and paired_hybrid_rows is not None:
        hybrid_metadata = build_metadata(
            model,
            interface,
            axis,
            kp,
            kd,
            filter_cutoff_hz,
            sample_dt,
            command_dt,
            cycles,
            paired_hybrid_rows,
            None,
            {"returned_to_test_anchor": True, "home_return_deferred_until_after_motor_local": True},
            "hybrid-cartesian",
            np.zeros(6, dtype=np.float64),
            COMPARISON_MOTOR_KD,
            measured_joint_position,
            anchor_pose,
        )
        local_prefix = Path(f"{paired_output_prefix}_motor_local")
        write_recording_artifacts(output_prefix, columns, paired_hybrid_rows, hybrid_metadata)
        write_recording_artifacts(local_prefix, columns, rows, metadata)
        comparison_prefix = Path(f"{paired_output_prefix}_comparison")
        write_paired_comparison_report(
            comparison_prefix,
            hybrid_metadata["condition_metrics"],
            metadata["condition_metrics"],
        )
        click.echo(f"Comparison: {comparison_prefix.with_suffix('.html')}")
    else:
        write_recording_artifacts(output_prefix, columns, rows, metadata)
    summary = metadata["summary"]
    click.echo(
        f"Recorded {summary['sample_count']} samples at {summary['observed_rate_hz']:.1f} Hz; "
        f"{summary['missed_sample_deadlines']} sample gaps exceeded 3 ms."
    )
    if failure and not failure.startswith("operator interrupted"):
        raise click.ClickException(failure)


if __name__ == "__main__":
    main()
