"""Move to a Cartesian pose under operational-space PD, lower stiffness, and hold.

Example:
    python examples/impedance_hold_test.py X5 can0 \
        --target-pose 0.30 0.00 0.25 0.00 0.00 0.00 \
        --kp 20 20 20 5 5 5 --kd 5 5 5 0.5 0.5 0.5
"""

import os
import sys
import time
from typing import Optional, Tuple

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
from tune_cartesian_pd import (
    default_config_path,
    load_gain_config,
    validate_filter_cutoff,
    validate_gain_vector,
)


def make_default_gain(controller_config: arx5.ControllerConfig) -> arx5.Gain:
    """Build a gain object from the controller's configured defaults."""
    return arx5.Gain(
        np.asarray(controller_config.default_kp, dtype=np.float64).copy(),
        np.asarray(controller_config.default_kd, dtype=np.float64).copy(),
        controller_config.default_gripper_kp,
        controller_config.default_gripper_kd,
    )


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option(
    "--target-pose",
    type=float,
    nargs=6,
    required=True,
    metavar="X Y Z ROLL PITCH YAW",
    help="Target end-effector pose in meters and radians.",
)
@click.option(
    "--gain-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Cartesian gain JSON written by tune_cartesian_pd.py. If omitted, "
        "the tuner's default model-specific path is loaded when it exists."
    ),
)
@click.option(
    "--kp",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Movement Cartesian Kp; overrides the loaded config.",
)
@click.option(
    "--kd",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Movement Cartesian Kd; overrides the loaded config.",
)
@click.option(
    "--hold-kp",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Hold Cartesian Kp. Defaults to the movement Kp.",
)
@click.option(
    "--hold-kd",
    type=float,
    nargs=6,
    metavar="X Y Z RX RY RZ",
    help="Hold Cartesian Kd. Defaults to the movement Kd.",
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
    "--move-duration",
    type=click.FloatRange(min=0.1, min_open=True),
    default=3.0,
    show_default=True,
    help="Time in seconds to move from the current pose to the target.",
)
@click.option(
    "--settle-duration",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Time at the target under movement gains before applying hold gains.",
)
@click.option(
    "--hold-duration",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Hold time in seconds. By default, hold until Ctrl-C.",
)
def main(
    model: str,
    interface: str,
    target_pose: Tuple[float, ...],
    gain_config: Optional[str],
    kp: Optional[Tuple[float, ...]],
    kd: Optional[Tuple[float, ...]],
    hold_kp: Optional[Tuple[float, ...]],
    hold_kd: Optional[Tuple[float, ...]],
    filter_cutoff_hz: Optional[float],
    move_duration: float,
    settle_duration: float,
    hold_duration: Optional[float],
) -> None:
    """Run the impedance hold test on MODEL using CAN INTERFACE."""
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config
    )

    movement_kp = np.asarray(
        controller_config.default_cartesian_kp, dtype=np.float64
    )
    movement_kd = np.asarray(
        controller_config.default_cartesian_kd, dtype=np.float64
    )
    configured_filter_cutoff_hz = float(
        controller_config.cartesian_joint_velocity_filter_cutoff_hz
    )
    if robot_config.joint_dof != 6:
        raise click.ClickException(
            "impedance_hold_test expects a six-joint arm, but "
            f"{model} has {robot_config.joint_dof} joints."
        )

    loaded_config_path = None
    if gain_config is not None:
        loaded_config_path = gain_config
    else:
        candidate_path = default_config_path(model)
        if candidate_path.exists():
            loaded_config_path = candidate_path

    try:
        if loaded_config_path is not None:
            (
                movement_kp,
                movement_kd,
                configured_filter_cutoff_hz,
            ) = load_gain_config(
                loaded_config_path,
                model,
                controller_config.cartesian_joint_velocity_filter_cutoff_hz,
                controller_config.controller_dt,
            )
        if kp is not None:
            movement_kp = validate_gain_vector(kp, "--kp")
        if kd is not None:
            movement_kd = validate_gain_vector(kd, "--kd")
        hold_cartesian_kp = (
            movement_kp.copy()
            if hold_kp is None
            else validate_gain_vector(hold_kp, "--hold-kp")
        )
        hold_cartesian_kd = (
            movement_kd.copy()
            if hold_kd is None
            else validate_gain_vector(hold_kd, "--hold-kd")
        )
        if filter_cutoff_hz is not None:
            configured_filter_cutoff_hz = validate_filter_cutoff(
                filter_cutoff_hz,
                controller_config.controller_dt,
                "--filter-cutoff-hz",
            )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"invalid Cartesian gain config: {exc}") from exc

    if not np.any(movement_kp > 0.0):
        raise click.ClickException(
            "movement Cartesian Kp is all zero. Tune and save gains with "
            "tune_cartesian_pd.py, pass --gain-config, or provide --kp/--kd."
        )

    target_pose_array = np.asarray(target_pose, dtype=np.float64)
    controller_config.cartesian_joint_velocity_filter_cutoff_hz = (
        configured_filter_cutoff_hz
    )
    controller = arx5.Arx5CartesianController(
        robot_config, controller_config, interface
    )
    controller.set_log_level(arx5.LogLevel.INFO)

    try:
        print("Resetting to home with default gains...")
        controller.reset_to_home()
        controller.set_gain(make_default_gain(controller_config))

        current_state = controller.get_eef_state()
        movement_cartesian_gain = arx5.CartesianPDGain(
            movement_kp.copy(), movement_kd.copy()
        )
        controller.set_cartesian_pd_gain(movement_cartesian_gain)
        controller.enable_cartesian_pd()
        print(
            "Cartesian PD enabled at the current pose "
            f"{np.asarray(current_state.pose_6d())}."
        )

        gripper_position = current_state.gripper_pos
        target_command = arx5.EEFState(target_pose_array, gripper_position)
        target_command.timestamp = controller.get_timestamp() + move_duration

        print(
            f"Moving to target pose {target_pose_array} over "
            f"{move_duration:.2f} s..."
        )
        controller.set_eef_cmd(target_command)
        while controller.get_timestamp() < target_command.timestamp:
            time.sleep(0.02)

        if settle_duration > 0.0:
            print(
                f"Holding with movement gains for {settle_duration:.2f} s "
                "to settle..."
            )
            time.sleep(settle_duration)

        controller.set_cartesian_pd_gain(
            arx5.CartesianPDGain(
                hold_cartesian_kp, hold_cartesian_kd
            )
        )

        if loaded_config_path is not None:
            print(f"Loaded gain config:    {loaded_config_path}")
        print(f"Movement Cartesian kp: {movement_kp}")
        print(f"Movement Cartesian kd: {movement_kd}")
        print(f"Hold Cartesian kp:    {hold_cartesian_kp}")
        print(f"Hold Cartesian kd:    {hold_cartesian_kd}")
        print(
            "Joint velocity filter: "
            f"{controller.get_cartesian_joint_velocity_filter_cutoff():g} Hz"
        )
        if hold_duration is None:
            print("Holding the target pose. Press Ctrl-C to stop.")
            while True:
                time.sleep(1.0)
        else:
            print(f"Holding the target pose for {hold_duration:.2f} s...")
            time.sleep(hold_duration)
    except KeyboardInterrupt:
        print("\nStopping impedance hold test.")
    finally:
        controller.set_to_damping()


if __name__ == "__main__":
    main()
