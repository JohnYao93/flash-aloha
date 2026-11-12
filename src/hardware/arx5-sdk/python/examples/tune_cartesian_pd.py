"""Interactively tune six-axis Cartesian PD gains on a six-joint arm.

The tuner starts with arm joint PD disabled, enables Cartesian PD at the
measured end-effector pose, and applies gains live. Tuned gains can be saved to
JSON and loaded by impedance_hold_test.py.

Example:
    python examples/tune_cartesian_pd.py L5 can0
"""

import curses
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))
import arx5_interface as arx5


AXES = ("x", "y", "z", "rx", "ry", "rz")
GAIN_FIELDS = ("kp", "kd")
KP_BASE_STEPS = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
KD_BASE_STEPS = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
KP_MAX = np.array([1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0])
KD_MAX = np.array([200.0, 200.0, 200.0, 20.0, 20.0, 20.0])
FILTER_CUTOFFS_HZ = (0.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
MOTION_TRANSLATION_AMPLITUDE_M = 0.040
MOTION_ROTATION_AMPLITUDE_RAD = np.deg2rad(20.0)
MOTION_FREQUENCY_HZ = 0.25
MOTION_RAMP_DURATION_S = 1.0
MOTION_JOINT_LIMIT_MARGIN_RAD = 0.05
MOTION_PREFLIGHT_SAMPLE_COUNT = 81
MOTION_OFF = "OFF"
MOTION_STARTING = "STARTING"
MOTION_RUNNING = "RUNNING"
MOTION_RETURNING = "RETURNING"


def motion_amplitude(axis_index):
    """Return the conservative excitation amplitude for a Cartesian axis."""
    if axis_index < 0 or axis_index >= len(AXES):
        raise ValueError(f"axis_index must be between 0 and {len(AXES) - 1}")
    if axis_index < 3:
        return MOTION_TRANSLATION_AMPLITUDE_M
    return MOTION_ROTATION_AMPLITUDE_RAD


def smoothstep(value):
    """Cubic zero-slope transition from zero to one."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def smoothstep_derivative(value, duration_s):
    """Time derivative of smoothstep(value) for value=t/duration_s."""
    value = float(value)
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return 6.0 * value * (1.0 - value) / float(duration_s)


def sinusoidal_motion_offset(elapsed_s, axis_index, ramp_envelope=1.0):
    """Evaluate the signed excitation offset for one axis."""
    elapsed_s = max(0.0, float(elapsed_s))
    return (
        motion_amplitude(axis_index)
        * float(np.clip(ramp_envelope, 0.0, 1.0))
        * np.sin(2.0 * np.pi * MOTION_FREQUENCY_HZ * elapsed_s)
    )


def sinusoidal_motion_velocity(
    elapsed_s, axis_index, ramp_envelope=1.0, ramp_envelope_derivative=0.0
):
    """Evaluate the analytic derivative of the excitation offset."""
    elapsed_s = max(0.0, float(elapsed_s))
    phase = 2.0 * np.pi * MOTION_FREQUENCY_HZ * elapsed_s
    amplitude = motion_amplitude(axis_index)
    return amplitude * (
        float(ramp_envelope_derivative) * np.sin(phase)
        + float(ramp_envelope)
        * 2.0
        * np.pi
        * MOTION_FREQUENCY_HZ
        * np.cos(phase)
    )


def _rpy_matrix(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _matrix_rpy(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = np.arctan2(
        -rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])
    )
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def apply_motion_offset(anchor_pose, axis_index, offset):
    """Apply translation or a base-frame axis-angle rotation to an anchor."""
    pose = np.asarray(anchor_pose, dtype=np.float64).copy()
    if axis_index < 3:
        pose[axis_index] += offset
        return pose
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index - 3] = 1.0
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    delta_rotation = (
        np.eye(3)
        + np.sin(offset) * cross
        + (1.0 - np.cos(offset)) * (cross @ cross)
    )
    pose[3:] = _matrix_rpy(delta_rotation @ _rpy_matrix(pose[3:]))
    return pose


@dataclass
class SinusoidalMotion:
    """Stateful, bounded sinusoidal command with smooth start and return."""

    state: str = MOTION_OFF
    axis_index: int | None = None
    anchor_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    gripper_position: float = 0.0
    start_time: float = 0.0
    return_start_time: float = 0.0
    return_start_envelope: float = 0.0

    @property
    def active(self):
        return self.state != MOTION_OFF

    def start(self, axis_index, anchor_pose, gripper_position, now):
        anchor_pose = np.asarray(anchor_pose, dtype=np.float64)
        if anchor_pose.shape != (6,) or not np.all(np.isfinite(anchor_pose)):
            raise ValueError("anchor_pose must contain six finite values")
        motion_amplitude(axis_index)
        self.axis_index = axis_index
        self.anchor_pose = anchor_pose.copy()
        self.gripper_position = float(gripper_position)
        self.start_time = float(now)
        self.return_start_time = 0.0
        self.return_start_envelope = 0.0
        self.state = MOTION_STARTING

    def begin_return(self, now):
        if self.state in (MOTION_STARTING, MOTION_RUNNING):
            self.return_start_time = float(now)
            elapsed_s = max(0.0, self.return_start_time - self.start_time)
            self.return_start_envelope = smoothstep(
                elapsed_s / MOTION_RAMP_DURATION_S
            )
            self.state = MOTION_RETURNING

    def cancel(self):
        self.state = MOTION_OFF
        self.axis_index = None
        self.return_start_time = 0.0
        self.return_start_envelope = 0.0

    def command_pose(self, now):
        """Return the current pose, changing to OFF after an exact return."""
        return self.command_reference(now)[0]

    def command_reference(self, now):
        """Return pose and analytic base-frame twist for the current profile."""
        if not self.active or self.axis_index is None:
            return self.anchor_pose.copy(), np.zeros(6, dtype=np.float64)

        now = float(now)
        elapsed_s = max(0.0, now - self.start_time)
        if self.state == MOTION_RETURNING:
            return_elapsed_s = max(0.0, now - self.return_start_time)
            if return_elapsed_s >= MOTION_RAMP_DURATION_S:
                pose = self.anchor_pose.copy()
                self.cancel()
                return pose, np.zeros(6, dtype=np.float64)
            return_progress = return_elapsed_s / MOTION_RAMP_DURATION_S
            envelope = self.return_start_envelope * (
                1.0 - smoothstep(return_progress)
            )
            envelope_derivative = -self.return_start_envelope * (
                smoothstep_derivative(
                    return_progress, MOTION_RAMP_DURATION_S
                )
            )
        else:
            ramp_progress = elapsed_s / MOTION_RAMP_DURATION_S
            envelope = smoothstep(ramp_progress)
            envelope_derivative = smoothstep_derivative(
                ramp_progress, MOTION_RAMP_DURATION_S
            )
            self.state = (
                MOTION_STARTING
                if ramp_progress < 1.0
                else MOTION_RUNNING
            )

        offset = sinusoidal_motion_offset(elapsed_s, self.axis_index, envelope)
        offset_velocity = sinusoidal_motion_velocity(
            elapsed_s,
            self.axis_index,
            envelope,
            envelope_derivative,
        )
        pose = apply_motion_offset(self.anchor_pose, self.axis_index, offset)
        twist = np.zeros(6, dtype=np.float64)
        twist[self.axis_index] = offset_velocity
        return pose, twist


def preflight_sinusoidal_motion(
    solver,
    anchor_pose,
    measured_joint_position,
    joint_pos_min,
    joint_pos_max,
    axis_index,
    sample_count=MOTION_PREFLIGHT_SAMPLE_COUNT,
):
    """Check one full-amplitude cycle for IK and joint-limit clearance."""
    anchor_pose = np.asarray(anchor_pose, dtype=np.float64)
    measured_joint_position = np.asarray(
        measured_joint_position, dtype=np.float64
    )
    joint_pos_min = np.asarray(joint_pos_min, dtype=np.float64)
    joint_pos_max = np.asarray(joint_pos_max, dtype=np.float64)
    if anchor_pose.shape != (6,):
        raise ValueError("anchor_pose must contain six values")
    if measured_joint_position.shape != joint_pos_min.shape or (
        measured_joint_position.shape != joint_pos_max.shape
    ):
        raise ValueError("joint position and limit vectors must have equal shapes")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    cycle_duration_s = 1.0 / MOTION_FREQUENCY_HZ
    for sample_index, elapsed_s in enumerate(
        np.linspace(0.0, cycle_duration_s, sample_count)
    ):
        sample_pose = apply_motion_offset(
            anchor_pose,
            axis_index,
            sinusoidal_motion_offset(elapsed_s, axis_index),
        )
        ik_status, sample_joint_position = solver.inverse_kinematics(
            sample_pose, measured_joint_position
        )
        if ik_status != 0:
            try:
                status_name = solver.get_ik_status_name(ik_status)
            except (AttributeError, TypeError):
                status_name = "unknown"
            raise ValueError(
                f"IK failed at preflight sample {sample_index + 1}/"
                f"{sample_count}: {status_name} ({ik_status})"
            )

        sample_joint_position = np.asarray(
            sample_joint_position, dtype=np.float64
        )
        if sample_joint_position.shape != measured_joint_position.shape or (
            not np.all(np.isfinite(sample_joint_position))
        ):
            raise ValueError(
                f"IK returned invalid joints at preflight sample "
                f"{sample_index + 1}/{sample_count}"
            )
        joint_clearance = np.minimum(
            sample_joint_position - joint_pos_min,
            joint_pos_max - sample_joint_position,
        )
        closest_joint = int(np.argmin(joint_clearance))
        if joint_clearance[closest_joint] <= (
            MOTION_JOINT_LIMIT_MARGIN_RAD + 1e-12
        ):
            raise ValueError(
                f"joint {closest_joint + 1} has only "
                f"{joint_clearance[closest_joint]:.3f} rad clearance at "
                f"preflight sample {sample_index + 1}/{sample_count}; "
                f"requires more than {MOTION_JOINT_LIMIT_MARGIN_RAD:.2f} rad"
            )


def validate_gain_vector(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} must contain exactly six values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return array.copy()


def validate_filter_cutoff(cutoff_hz, controller_dt, name="filter cutoff"):
    cutoff_hz = float(cutoff_hz)
    controller_dt = float(controller_dt)
    if not np.isfinite(controller_dt) or controller_dt <= 0.0:
        raise ValueError("controller_dt must be finite and positive")
    nyquist_hz = 1.0 / (2.0 * controller_dt)
    if (
        not np.isfinite(cutoff_hz)
        or cutoff_hz < 0.0
        or cutoff_hz >= nyquist_hz
    ):
        raise ValueError(
            f"{name} must be finite, nonnegative, and below {nyquist_hz:g} Hz"
        )
    return cutoff_hz


def model_default_filter_cutoff(model):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )
    return (
        float(controller_config.cartesian_joint_velocity_filter_cutoff_hz),
        float(controller_config.controller_dt),
    )


def default_config_path(model):
    return REPO_DIR / "models" / f"{model}_cartesian_pd.json"


def load_gain_config(
    path,
    model,
    default_filter_cutoff_hz=None,
    controller_dt=None,
):
    with Path(path).open() as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Cartesian PD config must be a JSON object")
    if config.get("model") != model:
        raise ValueError(
            f"Cartesian PD config is for model {config.get('model')!r}, "
            f"not {model!r}"
        )
    if default_filter_cutoff_hz is None or controller_dt is None:
        model_cutoff_hz, model_controller_dt = model_default_filter_cutoff(model)
        if default_filter_cutoff_hz is None:
            default_filter_cutoff_hz = model_cutoff_hz
        if controller_dt is None:
            controller_dt = model_controller_dt
    filter_cutoff_hz = config.get(
        "cartesian_joint_velocity_filter_cutoff_hz",
        default_filter_cutoff_hz,
    )
    return (
        validate_gain_vector(config.get("kp"), "kp"),
        validate_gain_vector(config.get("kd"), "kd"),
        validate_filter_cutoff(
            filter_cutoff_hz,
            controller_dt,
            "cartesian_joint_velocity_filter_cutoff_hz",
        ),
    )


def save_gain_config(
    path,
    model,
    kp,
    kd,
    filter_cutoff_hz=None,
    controller_dt=None,
):
    kp = validate_gain_vector(kp, "kp")
    kd = validate_gain_vector(kd, "kd")
    if filter_cutoff_hz is None or controller_dt is None:
        model_cutoff_hz, model_controller_dt = model_default_filter_cutoff(model)
        if filter_cutoff_hz is None:
            filter_cutoff_hz = model_cutoff_hz
        if controller_dt is None:
            controller_dt = model_controller_dt
    filter_cutoff_hz = validate_filter_cutoff(
        filter_cutoff_hz,
        controller_dt,
        "cartesian_joint_velocity_filter_cutoff_hz",
    )
    config = {
        "model": model,
        "axes": list(AXES),
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "cartesian_joint_velocity_filter_cutoff_hz": filter_cutoff_hz,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            json.dump(config, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def safe_addstr(stdscr, row, column, text):
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height or column < 0 or column >= width:
        return
    available = width - column
    if row == height - 1:
        available -= 1
    if available <= 0:
        return
    try:
        stdscr.addstr(row, column, str(text)[:available])
    except curses.error:
        pass


def apply_gain(controller, kp, kd):
    controller.set_cartesian_pd_gain(arx5.CartesianPDGain(kp, kd))


def draw(
    stdscr,
    model,
    interface,
    config_path,
    status,
    axis_index,
    field_index,
    step_scale,
    kp,
    kd,
    enabled,
    gravity_enabled,
    friction_enabled,
    measured_pose,
    desired_pose,
    pose_error,
    raw_twist,
    filtered_twist,
    desired_twist,
    filter_cutoff_hz,
    wrench,
    wrench_limit,
    motion,
):
    selected_field = GAIN_FIELDS[field_index]
    base_step = (
        KP_BASE_STEPS[axis_index]
        if selected_field == "kp"
        else KD_BASE_STEPS[axis_index]
    )

    stdscr.erase()
    safe_addstr(
        stdscr,
        0,
        0,
        f"Cartesian PD tuner  model={model}  interface={interface}",
    )
    safe_addstr(
        stdscr,
        1,
        0,
        "q quit | [/ ] axis | left/right Kp/Kd | up/down tune | "
        "SPACE enable | h re-anchor | m motion | 0 zero | r/R reset",
    )
    safe_addstr(
        stdscr,
        2,
        0,
        "s save | l load | v cycle velocity filter | g gravity | f friction | +/- step",
    )
    safe_addstr(
        stdscr,
        4,
        0,
        f"Cartesian PD: {'ENABLED' if enabled else 'DISABLED'}   "
        f"gravity: {'ON' if gravity_enabled else 'OFF'}   "
        f"friction: {'ON' if friction_enabled else 'OFF'}   "
        f"motion: {motion.state}",
    )
    safe_addstr(
        stdscr,
        5,
        0,
        f"selected: {AXES[axis_index]} {selected_field.upper()}   "
        f"step: {base_step * step_scale:g}   velocity filter: {filter_cutoff_hz:g} Hz   {status}",
    )
    safe_addstr(stdscr, 6, 0, f"config: {config_path}")
    shown_motion_axis = (
        motion.axis_index if motion.axis_index is not None else axis_index
    )
    shown_amplitude = (
        f"{MOTION_TRANSLATION_AMPLITUDE_M * 1000:g} mm"
        if shown_motion_axis < 3
        else f"{np.rad2deg(MOTION_ROTATION_AMPLITUDE_RAD):g} deg"
    )
    safe_addstr(
        stdscr,
        7,
        0,
        f"motion axis: {AXES[shown_motion_axis]}   sine +/-{shown_amplitude} "
        f"at {MOTION_FREQUENCY_HZ:g} Hz   {MOTION_RAMP_DURATION_S:g} s "
        "smooth start/return; pose/twist interpolated at 500 Hz",
    )

    safe_addstr(stdscr, 8, 0, "axis        x        y        z       rx       ry       rz")
    safe_addstr(
        stdscr,
        9,
        0,
        "Kp    " + " ".join(f"{value:8.3f}" for value in kp),
    )
    safe_addstr(
        stdscr,
        10,
        0,
        "Kd    " + " ".join(f"{value:8.3f}" for value in kd),
    )
    safe_addstr(
        stdscr,
        12,
        0,
        "desired pose: " + np.array2string(desired_pose, precision=4),
    )
    safe_addstr(
        stdscr,
        13,
        0,
        "measured pose: " + np.array2string(measured_pose, precision=4),
    )
    safe_addstr(
        stdscr,
        14,
        0,
        "pose error:    " + np.array2string(pose_error, precision=4),
    )
    safe_addstr(
        stdscr,
        15,
        0,
        "raw twist:     " + np.array2string(raw_twist, precision=4),
    )
    safe_addstr(
        stdscr,
        16,
        0,
        "filtered twist:" + np.array2string(filtered_twist, precision=4),
    )
    safe_addstr(
        stdscr,
        17,
        0,
        "desired twist: " + np.array2string(desired_twist, precision=4),
    )
    safe_addstr(
        stdscr,
        18,
        0,
        "PD wrench:     " + np.array2string(wrench, precision=3),
    )
    safe_addstr(
        stdscr,
        19,
        0,
        "wrench limit:  " + np.array2string(wrench_limit, precision=3),
    )
    safe_addstr(
        stdscr,
        21,
        0,
        f"Clear the full +/-{MOTION_TRANSLATION_AMPLITUDE_M * 1000:g} mm "
        f"translation and +/-{np.rad2deg(MOTION_ROTATION_AMPLITUDE_RAD):g} "
        "deg rotation swept envelope before using m.",
    )
    safe_addstr(
        stdscr,
        22,
        0,
        "IK preflight cannot check self-collision or external obstacles. "
        "SPACE immediately disables Cartesian PD.",
    )
    stdscr.refresh()


def run_tuner(
    stdscr,
    controller,
    kinematics,
    solver,
    model,
    interface,
    loop_dt,
    config_path,
    initial_kp,
    initial_kd,
    initial_filter_cutoff_hz,
):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    controller_config = controller.get_controller_config()

    if config_path.exists():
        try:
            loaded_kp, loaded_kd, loaded_filter_cutoff_hz = load_gain_config(
                config_path,
                model,
                controller.get_cartesian_joint_velocity_filter_cutoff(),
                controller_config.controller_dt,
            )
            status = "loaded saved gains"
        except Exception as exc:
            loaded_kp = np.zeros(6, dtype=np.float64)
            loaded_kd = np.zeros(6, dtype=np.float64)
            loaded_filter_cutoff_hz = (
                controller.get_cartesian_joint_velocity_filter_cutoff()
            )
            status = f"load failed; using zero gains: {exc}"
    else:
        loaded_kp = np.zeros(6, dtype=np.float64)
        loaded_kd = np.zeros(6, dtype=np.float64)
        loaded_filter_cutoff_hz = (
            controller.get_cartesian_joint_velocity_filter_cutoff()
        )
        status = "no saved config; using zero gains"

    kp = loaded_kp.copy() if initial_kp is None else initial_kp.copy()
    kd = loaded_kd.copy() if initial_kd is None else initial_kd.copy()
    filter_cutoff_hz = (
        loaded_filter_cutoff_hz
        if initial_filter_cutoff_hz is None
        else initial_filter_cutoff_hz
    )
    apply_gain(controller, kp, kd)
    controller.set_cartesian_joint_velocity_filter_cutoff(filter_cutoff_hz)

    axis_index = 0
    field_index = 0
    step_scale = 1.0
    enabled = controller.is_cartesian_pd_enabled()
    gravity_enabled = bool(controller_config.gravity_compensation)
    friction_enabled = bool(controller_config.friction_compensation)
    robot_config = controller.get_robot_config()
    joint_pos_min = np.asarray(robot_config.joint_pos_min, dtype=np.float64)
    joint_pos_max = np.asarray(robot_config.joint_pos_max, dtype=np.float64)
    motion = SinusoidalMotion()
    wrench_limit = np.asarray(
        controller_config.cartesian_wrench_limit, dtype=np.float64
    )
    next_loop_time = time.monotonic()

    while True:
        key = stdscr.getch()
        gain_changed = False

        while key != -1:
            if key in (ord("q"), 27):
                return
            if key == ord("["):
                if motion.active:
                    status = "axis is locked until motion returns to the anchor"
                else:
                    axis_index = (axis_index - 1) % len(AXES)
            elif key == ord("]"):
                if motion.active:
                    status = "axis is locked until motion returns to the anchor"
                else:
                    axis_index = (axis_index + 1) % len(AXES)
            elif key == curses.KEY_LEFT:
                field_index = (field_index - 1) % len(GAIN_FIELDS)
            elif key == curses.KEY_RIGHT:
                field_index = (field_index + 1) % len(GAIN_FIELDS)
            elif key in (curses.KEY_UP, curses.KEY_DOWN):
                direction = 1.0 if key == curses.KEY_UP else -1.0
                if GAIN_FIELDS[field_index] == "kp":
                    kp[axis_index] = np.clip(
                        kp[axis_index]
                        + direction * KP_BASE_STEPS[axis_index] * step_scale,
                        0.0,
                        KP_MAX[axis_index],
                    )
                else:
                    kd[axis_index] = np.clip(
                        kd[axis_index]
                        + direction * KD_BASE_STEPS[axis_index] * step_scale,
                        0.0,
                        KD_MAX[axis_index],
                    )
                gain_changed = True
            elif key == ord(" "):
                motion.cancel()
                if enabled:
                    controller.disable_cartesian_pd()
                    enabled = False
                    status = "motion cancelled; Cartesian PD disabled"
                else:
                    controller.enable_cartesian_pd()
                    enabled = True
                    status = "Cartesian PD enabled and anchored; motion off"
            elif key == ord("h"):
                motion.cancel()
                controller.disable_cartesian_pd()
                controller.enable_cartesian_pd()
                enabled = True
                status = "motion cancelled; hold anchored at measured pose"
            elif key == ord("m"):
                if not enabled:
                    status = "motion refused: Cartesian PD is disabled"
                elif motion.state == MOTION_RETURNING:
                    status = "motion return already in progress; m ignored"
                elif motion.active:
                    motion.begin_return(time.monotonic())
                    status = "returning smoothly to the motion anchor"
                else:
                    # Calling enable while already enabled safely freezes the
                    # desired command at the currently measured joint pose.
                    controller.enable_cartesian_pd()
                    anchor_joint_state = controller.get_joint_state()
                    measured_joint_position = np.asarray(
                        anchor_joint_state.pos(), dtype=np.float64
                    )
                    anchor_pose = np.asarray(
                        kinematics.pose(measured_joint_position),
                        dtype=np.float64,
                    )
                    try:
                        preflight_sinusoidal_motion(
                            solver,
                            anchor_pose,
                            measured_joint_position,
                            joint_pos_min,
                            joint_pos_max,
                            axis_index,
                        )
                    except (RuntimeError, ValueError) as exc:
                        status = f"motion preflight refused: {exc}"
                    else:
                        motion.start(
                            axis_index,
                            anchor_pose,
                            anchor_joint_state.gripper_pos,
                            time.monotonic(),
                        )
                        status = (
                            f"motion starting on {AXES[axis_index]}; ensure "
                            "the displayed sweep is clear"
                        )
            elif key == ord("0"):
                if GAIN_FIELDS[field_index] == "kp":
                    kp[axis_index] = 0.0
                else:
                    kd[axis_index] = 0.0
                gain_changed = True
            elif key == ord("r"):
                if GAIN_FIELDS[field_index] == "kp":
                    kp[axis_index] = loaded_kp[axis_index]
                else:
                    kd[axis_index] = loaded_kd[axis_index]
                gain_changed = True
                status = "selected gain reset"
            elif key == ord("R"):
                kp = loaded_kp.copy()
                kd = loaded_kd.copy()
                gain_changed = True
                status = "all gains reset"
            elif key in (ord("+"), ord("=")):
                step_scale = min(step_scale * 10.0, 100.0)
            elif key in (ord("-"), ord("_")):
                step_scale = max(step_scale / 10.0, 0.01)
            elif key == ord("g"):
                gravity_enabled = not gravity_enabled
                controller.set_gravity_compensation(gravity_enabled)
                status = (
                    "gravity compensation enabled"
                    if gravity_enabled
                    else "gravity compensation disabled"
                )
            elif key == ord("f"):
                friction_enabled = not friction_enabled
                controller.set_friction_compensation(friction_enabled)
                status = (
                    "friction compensation enabled"
                    if friction_enabled
                    else "friction compensation disabled"
                )
            elif key == ord("v"):
                filter_cutoff_hz = next(
                    (
                        candidate
                        for candidate in FILTER_CUTOFFS_HZ
                        if candidate > filter_cutoff_hz + 1e-12
                    ),
                    FILTER_CUTOFFS_HZ[0],
                )
                controller.set_cartesian_joint_velocity_filter_cutoff(
                    filter_cutoff_hz
                )
                status = f"velocity filter set to {filter_cutoff_hz:g} Hz"
            elif key == ord("s"):
                try:
                    save_gain_config(
                        config_path,
                        model,
                        kp,
                        kd,
                        filter_cutoff_hz,
                        controller_config.controller_dt,
                    )
                    loaded_kp = kp.copy()
                    loaded_kd = kd.copy()
                    loaded_filter_cutoff_hz = filter_cutoff_hz
                    status = "saved gains"
                except Exception as exc:
                    status = f"save failed: {exc}"
            elif key == ord("l"):
                try:
                    (
                        loaded_kp,
                        loaded_kd,
                        loaded_filter_cutoff_hz,
                    ) = load_gain_config(
                        config_path,
                        model,
                        controller_config.cartesian_joint_velocity_filter_cutoff_hz,
                        controller_config.controller_dt,
                    )
                    kp = loaded_kp.copy()
                    kd = loaded_kd.copy()
                    filter_cutoff_hz = loaded_filter_cutoff_hz
                    controller.set_cartesian_joint_velocity_filter_cutoff(
                        filter_cutoff_hz
                    )
                    gain_changed = True
                    status = "loaded saved gains"
                except Exception as exc:
                    status = f"load failed: {exc}"
            key = stdscr.getch()

        if gain_changed:
            apply_gain(controller, kp, kd)

        if motion.active:
            previous_motion_state = motion.state
            motion_pose, motion_twist = motion.command_reference(
                time.monotonic()
            )
            motion_command = arx5.EEFState(
                motion_pose, motion.gripper_position
            )
            motion_command.timestamp = controller.get_timestamp() + max(
                loop_dt, 2.0 * controller_config.controller_dt
            )
            controller.set_eef_cmd_with_twist(motion_command, motion_twist)
            if previous_motion_state == MOTION_RETURNING and not motion.active:
                status = "motion returned exactly to the anchor"

        joint_state = controller.get_joint_state()
        measured_state = controller.get_eef_state()
        desired_state = controller.get_eef_cmd()
        measured_pose = np.asarray(measured_state.pose_6d(), dtype=np.float64)
        desired_pose = np.asarray(desired_state.pose_6d(), dtype=np.float64)
        pose_error = np.asarray(
            arx5._cartesian_pose_error(desired_pose, measured_pose),
            dtype=np.float64,
        )
        jacobian = np.asarray(
            kinematics.jacobian(
                np.asarray(joint_state.pos(), dtype=np.float64)
            ),
            dtype=np.float64,
        )
        raw_twist = jacobian @ np.asarray(
            joint_state.vel(), dtype=np.float64
        )
        filtered_twist = jacobian @ np.asarray(
            controller.get_filtered_joint_velocity(), dtype=np.float64
        )
        desired_twist = np.asarray(
            controller.get_eef_cmd_twist(), dtype=np.float64
        )
        wrench = np.clip(
            kp * pose_error
            + kd * (desired_twist - filtered_twist),
            -wrench_limit,
            wrench_limit,
        )

        draw(
            stdscr,
            model,
            interface,
            config_path,
            status,
            axis_index,
            field_index,
            step_scale,
            kp,
            kd,
            enabled,
            gravity_enabled,
            friction_enabled,
            measured_pose,
            desired_pose,
            pose_error,
            raw_twist,
            filtered_twist,
            desired_twist,
            filter_cutoff_hz,
            wrench,
            wrench_limit,
            motion,
        )
        next_loop_time += loop_dt
        sleep_duration = next_loop_time - time.monotonic()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)
        else:
            # Do not accumulate missed deadlines after a slow key handler or
            # preflight; resume a stable period from the current iteration.
            next_loop_time = time.monotonic()


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option(
    "--config-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSON file used to load and save gains. Defaults to "
        "models/MODEL_cartesian_pd.json."
    ),
)
@click.option(
    "--initial-kp",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Override the loaded starting Kp values.",
)
@click.option(
    "--initial-kd",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Override the loaded starting Kd values.",
)
@click.option(
    "--filter-cutoff-hz",
    type=click.FloatRange(min=0.0),
    default=None,
    help=(
        "Override the loaded/model Cartesian joint-velocity filter cutoff. "
        "Zero bypasses filtering."
    ),
)
@click.option(
    "--loop-dt",
    default=0.02,
    show_default=True,
    type=click.FloatRange(min=0.001),
    help="UI refresh period in seconds; the controller itself remains at 500 Hz.",
)
def main(
    model,
    interface,
    config_path,
    initial_kp,
    initial_kd,
    filter_cutoff_hz,
    loop_dt,
):
    """Tune Cartesian PD gains on MODEL using CAN INTERFACE."""
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if robot_config.joint_dof != 6:
        raise click.ClickException(
            f"Cartesian PD tuning requires six joints; {model} has "
            f"{robot_config.joint_dof}."
        )
    if config_path is None:
        config_path = default_config_path(model)
    else:
        config_path = config_path.resolve()

    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )
    try:
        initial_kp = (
            None
            if initial_kp is None
            else validate_gain_vector(initial_kp, "--initial-kp")
        )
        initial_kd = (
            None
            if initial_kd is None
            else validate_gain_vector(initial_kd, "--initial-kd")
        )
        filter_cutoff_hz = (
            None
            if filter_cutoff_hz is None
            else validate_filter_cutoff(
                filter_cutoff_hz,
                controller_config.controller_dt,
                "--filter-cutoff-hz",
            )
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if filter_cutoff_hz is not None:
        controller_config.cartesian_joint_velocity_filter_cutoff_hz = (
            filter_cutoff_hz
        )
    elif config_path.exists():
        try:
            _, _, saved_filter_cutoff_hz = load_gain_config(
                config_path,
                model,
                controller_config.cartesian_joint_velocity_filter_cutoff_hz,
                controller_config.controller_dt,
            )
            controller_config.cartesian_joint_velocity_filter_cutoff_hz = (
                saved_filter_cutoff_hz
            )
        except (OSError, ValueError):
            # run_tuner reports the full load error and safely starts with zero gains.
            pass
    controller_config.background_send_recv = True
    controller = arx5.Arx5CartesianController(
        robot_config, controller_config, interface
    )
    controller.set_log_level(arx5.LogLevel.WARNING)

    # Cartesian mode suppresses arm joint Kp/Kd. Setting them to zero here also
    # makes SPACE-to-disable a gravity-compensated, zero-stiffness state instead
    # of unexpectedly restoring joint-space stiffness.
    zero_joint_gain = arx5.Gain(robot_config.joint_dof)
    zero_joint_gain.kp()[:] = 0.0
    zero_joint_gain.kd()[:] = 0.0
    zero_joint_gain.gripper_kp = controller_config.default_gripper_kp
    zero_joint_gain.gripper_kd = controller_config.default_gripper_kd
    controller.set_gain(zero_joint_gain)
    controller.set_cartesian_pd_gain(
        arx5.CartesianPDGain(
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )
    )
    controller.enable_cartesian_pd()

    try:
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
        curses.wrapper(
            run_tuner,
            controller,
            kinematics,
            solver,
            model,
            interface,
            loop_dt,
            config_path,
            initial_kp,
            initial_kd,
            filter_cutoff_hz,
        )
    finally:
        controller.disable_cartesian_pd()
        controller.set_to_damping()


if __name__ == "__main__":
    main()
