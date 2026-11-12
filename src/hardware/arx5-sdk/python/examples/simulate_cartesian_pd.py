"""Run the SDK Cartesian PD kernel against the supplied L5 MuJoCo model.

This is a hardware-free controller test. The first six MuJoCo position
actuators are converted to torque actuators at runtime. The desired pose and
analytic twist are generated directly in Cartesian space at 500 Hz.

Install the optional simulator dependency with:

    python -m pip install mujoco

Example:

    PYTHONPATH=python python python/examples/simulate_cartesian_pd.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


SDK_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = SDK_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# ROS 2-flavored kdl_parser builds use the ament index to locate the URDF
# parser plugin. Conda activation normally sets this; direct interpreter
# invocations do not.
if "AMENT_PREFIX_PATH" not in os.environ:
    ament_index = Path(sys.prefix) / "share/ament_index"
    if ament_index.exists():
        os.environ["AMENT_PREFIX_PATH"] = sys.prefix

import arx5_interface as arx5


ARM_DOF = 6
DEFAULT_INITIAL_Q = np.array(
    [0.0, 0.8, 1.0, -0.4, 0.3, 0.0], dtype=np.float64
)
DEFAULT_TARGET_OFFSET = np.array(
    [0.03, 0.00, 0.02, 0.00, 0.10, 0.00], dtype=np.float64
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the SDK's operational-space Cartesian PD controller "
            "on the supplied L5 MuJoCo model."
        )
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        nargs=6,
        default=DEFAULT_TARGET_OFFSET,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help=(
            "Target offset from the initial EEF pose in meters/radians "
            "(default: 0.03 0 0.02 0 0.10 0)."
        ),
    )
    parser.add_argument(
        "--target-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Absolute target EEF pose; overrides --target-offset.",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=2.0,
        help="Seconds for the Cartesian reference transition (default: 2).",
    )
    parser.add_argument(
        "--settle-duration",
        type=float,
        default=3.0,
        help="Seconds to hold the target after the move (default: 3).",
    )
    parser.add_argument(
        "--joint-armature",
        type=float,
        default=0.02,
        help=(
            "Reflected arm-joint inertia added to the zero-armature MuJoCo "
            "model, in kg m^2 (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=4,
        help="MuJoCo integration steps per 500 Hz control update (default: 4).",
    )
    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_Q,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Initial six arm joint positions in radians.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        nargs=6,
        default=(100.0, 100.0, 100.0, 10.0, 10.0, 10.0),
        metavar=("KPX", "KPY", "KPZ", "KPRX", "KPRY", "KPRZ"),
        help="Cartesian proportional gains.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        nargs=6,
        default=(20.0, 20.0, 20.0, 2.0, 2.0, 2.0),
        metavar=("KDX", "KDY", "KDZ", "KDRX", "KDRY", "KDRZ"),
        help="Cartesian derivative gains.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for the complete simulation log.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the MuJoCo passive viewer and run in real time.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.015,
        help="Final position-error threshold in meters (default: 0.015).",
    )
    parser.add_argument(
        "--rotation-tolerance",
        type=float,
        default=0.08,
        help="Final SO(3) error threshold in radians (default: 0.08).",
    )
    args = parser.parse_args(argv)
    if args.move_duration <= 0.0:
        parser.error("--move-duration must be positive")
    if args.settle_duration < 0.0:
        parser.error("--settle-duration must be nonnegative")
    if args.joint_armature < 0.0:
        parser.error("--joint-armature must be nonnegative")
    if args.physics_substeps < 1:
        parser.error("--physics-substeps must be at least 1")
    if args.position_tolerance < 0.0 or args.rotation_tolerance < 0.0:
        parser.error("error tolerances must be nonnegative")
    return args


def import_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise SystemExit(
            "MuJoCo is required for this example. Install it with "
            "`python -m pip install mujoco`."
        ) from error
    return mujoco


def configure_torque_actuators(model, mujoco, joint_torque_max: np.ndarray) -> None:
    """Convert the six arm position actuators into direct torque actuators."""
    if model.nu < ARM_DOF:
        raise RuntimeError(f"MuJoCo model has {model.nu} actuators; expected at least 6.")

    for index in range(ARM_DOF):
        joint_name = f"joint{index + 1}"
        actuator_name = f"joint{index + 1}_servo"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(
                f"Missing {joint_name!r} or {actuator_name!r} in MuJoCo model."
            )
        if actuator_id != index:
            raise RuntimeError(
                "The simulation expects arm actuators 0-5 to match joints 1-6."
            )

        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = 1.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id, :] = (
            -joint_torque_max[index],
            joint_torque_max[index],
        )


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(
        -rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])
    )
    return np.array(
        [
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            pitch,
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ]
    )


def rotation_exp(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-14:
        return np.eye(3)
    axis = rotation_vector / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (
        cross @ cross
    )


def write_csv(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["time"]
        + [f"pose_error_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]
        + [f"wrench_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]
        + [f"task_torque_{index}" for index in range(1, 7)]
        + [f"motor_torque_{index}" for index in range(1, 7)]
        + [f"joint_{index}" for index in range(1, 7)]
    )
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def run_simulation(args: argparse.Namespace) -> int:
    mujoco = import_mujoco()
    model_path = SDK_ROOT / "models/mujoco/ARX_L5/ARX_L5.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    gravity_data = mujoco.MjData(model)

    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )
    joint_torque_max = np.asarray(
        robot_config.joint_torque_max, dtype=np.float64
    )
    wrench_limit = np.asarray(
        controller_config.cartesian_wrench_limit, dtype=np.float64
    )
    gain = arx5.CartesianPDGain(
        np.asarray(args.kp, dtype=np.float64),
        np.asarray(args.kd, dtype=np.float64),
    )
    controller_dt = float(controller_config.controller_dt)
    model.opt.timestep = controller_dt / args.physics_substeps
    model.dof_armature[:ARM_DOF] += args.joint_armature
    kinematics = arx5._CartesianKinematics(
        robot_config.urdf_path,
        robot_config.joint_dof,
        robot_config.base_link_name,
        robot_config.eef_link_name,
    )
    configure_torque_actuators(model, mujoco, joint_torque_max)
    initial_q = np.asarray(args.initial_q, dtype=np.float64)
    data.qpos[:ARM_DOF] = initial_q
    if model.nq >= 8:
        data.qpos[6:8] = (0.03, -0.03)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    initial_pose = np.asarray(kinematics.pose(initial_q), dtype=np.float64)
    if args.target_pose is None:
        target_pose = initial_pose + np.asarray(
            args.target_offset, dtype=np.float64
        )
    else:
        target_pose = np.asarray(args.target_pose, dtype=np.float64)
    reference_delta = np.asarray(
        arx5._cartesian_pose_error(target_pose, initial_pose),
        dtype=np.float64,
    )
    initial_rotation = rpy_matrix(initial_pose[3:])

    total_duration = args.move_duration + args.settle_duration
    step_count = int(np.ceil(total_duration / controller_dt))
    previous_task_torque = np.zeros(ARM_DOF, dtype=np.float64)
    rows: list[list[float]] = []
    max_abs_wrench = np.zeros(6, dtype=np.float64)
    max_abs_motor_torque = np.zeros(6, dtype=np.float64)
    viewer = None
    if args.view:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    wall_start = time.monotonic()
    try:
        for _ in range(step_count):
            sim_time = float(data.time)
            progress = min(sim_time / args.move_duration, 1.0)
            alpha = progress * progress * (3.0 - 2.0 * progress)
            alpha_dot = (
                6.0 * progress * (1.0 - progress) / args.move_duration
                if progress < 1.0
                else 0.0
            )
            desired_pose = initial_pose.copy()
            desired_pose[:3] += alpha * reference_delta[:3]
            desired_pose[3:] = matrix_rpy(
                rotation_exp(alpha * reference_delta[3:]) @ initial_rotation
            )
            desired_twist = alpha_dot * reference_delta

            measured_q = np.asarray(data.qpos[:ARM_DOF], dtype=np.float64).copy()
            measured_qvel = np.asarray(
                data.qvel[:ARM_DOF], dtype=np.float64
            ).copy()
            measured_jacobian = np.asarray(
                kinematics.jacobian(measured_q), dtype=np.float64
            )
            measured_pose = np.asarray(
                kinematics.pose(measured_q), dtype=np.float64
            )
            pose_error = np.asarray(
                arx5._cartesian_pose_error(desired_pose, measured_pose),
                dtype=np.float64,
            )
            actual_twist = measured_jacobian @ measured_qvel
            task_torque, wrench = arx5._compute_cartesian_pd_torque(
                measured_jacobian,
                pose_error,
                desired_twist,
                actual_twist,
                gain,
                wrench_limit,
            )
            task_torque = np.asarray(task_torque, dtype=np.float64)
            wrench = np.asarray(wrench, dtype=np.float64)
            maximum_torque_delta = (
                joint_torque_max
                * controller_dt
                / controller_config.cartesian_task_torque_ramp_time_s
            )
            task_torque = previous_task_torque + np.clip(
                task_torque - previous_task_torque,
                -maximum_torque_delta,
                maximum_torque_delta,
            )

            # Match production: compensate gravity only, not Coriolis. A
            # separate zero-velocity data object avoids perturbing the plant.
            gravity_data.qpos[:] = data.qpos
            gravity_data.qvel[:] = 0.0
            mujoco.mj_forward(model, gravity_data)
            dynamics_compensation = np.asarray(
                gravity_data.qfrc_bias[:ARM_DOF], dtype=np.float64
            ).copy()
            motor_torque = np.clip(
                task_torque + dynamics_compensation,
                -joint_torque_max,
                joint_torque_max,
            )
            data.ctrl[:ARM_DOF] = motor_torque
            if model.nu >= 8:
                data.ctrl[6:8] = (0.03, -0.03)

            max_abs_wrench = np.maximum(max_abs_wrench, np.abs(wrench))
            max_abs_motor_torque = np.maximum(
                max_abs_motor_torque, np.abs(motor_torque)
            )
            rows.append(
                [sim_time]
                + pose_error.tolist()
                + wrench.tolist()
                + task_torque.tolist()
                + motor_torque.tolist()
                + measured_q.tolist()
            )
            previous_task_torque = task_torque
            for _ in range(args.physics_substeps):
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                deadline = wall_start + float(data.time)
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    final_q = np.asarray(data.qpos[:ARM_DOF], dtype=np.float64)
    final_pose = np.asarray(kinematics.pose(final_q), dtype=np.float64)
    final_error = np.asarray(
        arx5._cartesian_pose_error(
            target_pose,
            final_pose,
        ),
        dtype=np.float64,
    )
    position_error = float(np.linalg.norm(final_error[:3]))
    rotation_error = float(np.linalg.norm(final_error[3:]))
    wrench_within_limits = bool(
        np.all(max_abs_wrench <= wrench_limit + 1e-10)
    )
    torque_within_limits = bool(
        np.all(max_abs_motor_torque <= joint_torque_max + 1e-10)
    )

    print(f"Initial pose:          {initial_pose}")
    print(f"Requested target pose: {target_pose}")
    print(f"Final pose:            {final_pose}")
    print(f"Final position error:  {position_error:.6f} m")
    print(f"Final rotation error:  {rotation_error:.6f} rad")
    print(f"Max |wrench|:          {max_abs_wrench}")
    print(f"Wrench limits:         {wrench_limit}")
    print(f"Max |motor torque|:    {max_abs_motor_torque}")
    print(f"Motor torque limits:   {joint_torque_max}")

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"Wrote simulation log:  {args.csv}")

    passed = (
        position_error <= args.position_tolerance
        and rotation_error <= args.rotation_tolerance
        and wrench_within_limits
        and torque_within_limits
    )
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_simulation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
