"""Replay helpers shared by standalone visualization and recording review.

Importing this module opens no hardware. Callers retain their existing start,
stop, gain, and homing policies; replay targets retain their nominal intervals.
"""
import contextlib
from contextlib import contextmanager
import math
from pathlib import Path
import signal
import threading
import time

import numpy as np

from config.constants import resolve_arx_single_arm_can
from hardware.safe_exit import ARXDamping, damping_exit
from utils.hardware_utils import (
    ARX_COMMAND_HZ,
    configure_controller,
    configure_active_gains,
    replay_timestep,
    send_joint_target,
)

DEFAULT_REPLAY_FPS = 50  # Internal legacy positioning default; CLI reads episode metadata.
DT = 1 / DEFAULT_REPLAY_FPS


def send_replay_command(controller, command, interval_s):
    """Send a prepared target, preserving immediate holds and timed waypoints."""
    if interval_s is None:
        controller.set_joint_cmd(command)
    else:
        send_joint_target(controller, command, interval_s)


def resolve_playback_fps(fps_arg):
    if fps_arg <= 0:
        raise ValueError(f"--fps must be > 0, got {fps_arg}")
    return float(fps_arg)


def select_replay_stream(action_source, qpos, qvel, action, action_timestamps, action_raw, action_raw_timestamps):
    if action_source == "raw":
        if action_raw is None:
            raise ValueError(
                "Requested raw replay but /action_raw is missing. "
                "Re-zip with --include-raw-actions."
            )
        return action_raw, None, action_raw_timestamps, "raw"

    if action is not None:
        action_v = qvel if (qvel is not None and len(qvel) == len(action)) else None
        if qvel is not None and action_v is None:
            print(
                "[WARN] qvel length mismatch for processed action replay "
                f"(len(action)={len(action)} len(qvel)={len(qvel)}). Ignoring velocities."
            )
        return action, action_v, action_timestamps, "processed"

    print("[WARN] /action missing, falling back to /observations/qpos for processed replay.")
    action_v = qvel if (qvel is not None and len(qvel) == len(qpos)) else None
    return qpos, action_v, None, "processed(qpos-fallback)"


def interpolate_fill_in(actions):
    actions = np.asarray(actions)
    if actions.ndim > 2:
        actions = actions.squeeze()
    new_action_list = np.zeros((actions.shape[0] * 2, actions.shape[1]), dtype=np.float32)
    for i, action in enumerate(actions):
        new_action_list[i * 2] = action
        if i < len(actions) - 1:
            new_action_list[i * 2 + 1] = 0.5 * action + 0.5 * actions[i + 1]
        else:
            new_action_list[-1] = action
    return new_action_list


def robot_interfaces(robot_type, task_config, is_dual_arm):
    if robot_type == "arx":
        left = (
            task_config.get("can_interface_left", "can0")
            if is_dual_arm
            else resolve_arx_single_arm_can(task_config)
        )
        right = task_config.get("can_interface_right", "can1") if is_dual_arm else None
    return left, right


def configure_arx_gain(controller, kp=None, kd=None):
    if kp is None and kd is None:
        return  # reset_to_home ramps to the shared controller-config defaults.

    gain = controller.get_gain()
    if kp is not None:
        kp_arr = np.asarray(kp, dtype=float)
        gain.kp()[:] = kp_arr[:6]
        gain.gripper_kp = float(kp_arr[-1])
    if kd is not None:
        kd_arr = np.asarray(kd, dtype=float)
        gain.kd()[:] = kd_arr[:6]
        gain.gripper_kd = float(kd_arr[-1])
    controller.set_gain(gain)


def format_gain_value_for_filename(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_gain_vector_for_filename(values):
    return "-".join(format_gain_value_for_filename(value) for value in values)


def format_arx_gain_filename_suffix(gain):
    kp = np.concatenate((gain.kp().copy(), [gain.gripper_kp]))
    kd = np.concatenate((gain.kd().copy(), [gain.gripper_kd]))
    return f"_kp{format_gain_vector_for_filename(kp)}_kd{format_gain_vector_for_filename(kd)}"


def make_arx_solver(arx5, robot_config):
    return arx5.Arx5Solver(
        robot_config.urdf_path,
        robot_config.joint_dof,
        robot_config.joint_pos_min,
        robot_config.joint_pos_max,
        robot_config.base_link_name,
        robot_config.eef_link_name,
        robot_config.gravity_vector,
    )


def compute_arx_torque_minus_gravity(solver, joint_pos, measured_torque, joint_dof):
    zero = np.zeros(joint_dof, dtype=np.float64)
    gravity_torque = solver.inverse_dynamics(
        np.asarray(joint_pos[:joint_dof], dtype=np.float64),
        zero,
        zero,
    )
    return np.asarray(measured_torque[:joint_dof], dtype=np.float64) - np.asarray(gravity_torque, dtype=np.float64)


def make_arx_joint_command(arx5, robot_config):
    return arx5.JointState(robot_config.joint_dof)


def set_arx_action(controller, command, action, interval_s=None):
    command.pos()[:] = np.asarray(action[:6], dtype=float)
    command.gripper_pos = float(action[-1])
    send_replay_command(controller, command, DT if interval_s is None else interval_s)


def read_arx_state(controller, include_torque=True):
    joint_state = controller.get_joint_state()
    pos = np.concatenate((joint_state.pos().copy(), [joint_state.gripper_pos])).astype(np.float32)
    vel = np.concatenate((joint_state.vel().copy(), [joint_state.gripper_vel])).astype(np.float32)
    torque = None
    if include_torque:
        torque = np.concatenate((joint_state.torque().copy(), [joint_state.gripper_torque])).astype(np.float32)
    return pos, vel, torque


def prepare_action_arrays(action_list, action_v_list, use_fill_in):
    action_list = np.asarray(action_list).squeeze()
    if action_list.ndim == 1:
        action_list = action_list[None, :]
    if action_v_list is not None:
        action_v_list = np.asarray(action_v_list).squeeze()
        if action_v_list.ndim == 1:
            action_v_list = action_v_list[None, :]

        if len(action_v_list) != len(action_list):
            print(
                "[WARN] Action velocity timestep count does not match the action stream "
                f"({len(action_v_list)} vs {len(action_list)}). Using zero target velocities."
            )
            action_v_list = None
        elif action_v_list.shape[1] != action_list.shape[1]:
            arm_count = action_list.shape[1] // 7
            expected_arm_velocity_dim = arm_count * 6
            if action_list.shape[1] in (7, 14) and action_v_list.shape[1] == expected_arm_velocity_dim:
                # Some ARX recordings store velocity for the six arm joints but
                # omit the gripper. Insert one zero gripper velocity per arm so
                # the layout matches [joint0..joint5, gripper] action blocks.
                action_v_list = action_v_list.reshape(len(action_v_list), arm_count, 6)
                gripper_velocity = np.zeros(
                    (len(action_v_list), arm_count, 1),
                    dtype=action_v_list.dtype,
                )
                action_v_list = np.concatenate(
                    (action_v_list, gripper_velocity), axis=2
                ).reshape(len(action_v_list), action_list.shape[1])
                print(
                    "[WARN] Action velocity stream omits gripper velocity; "
                    f"inserted {arm_count} zero gripper velocity column(s)."
                )
            else:
                print(
                    "[WARN] Action velocity dimension does not match the action stream "
                    f"({action_v_list.shape[1]} vs {action_list.shape[1]}). "
                    "Using zero target velocities."
                )
                action_v_list = None

    if use_fill_in:
        action_list = interpolate_fill_in(action_list)
        action_v_list = interpolate_fill_in(action_v_list) if action_v_list is not None else None
        print(f"    Fill-in enabled: replaying {len(action_list)} interpolated actions")
    else:
        print(f"    Fill-in disabled: replaying {len(action_list)} original actions")

    return action_list, action_v_list


def init_tracking_buffers(plot_tracking):
    if not plot_tracking:
        return None
    return {
        "commanded_left_joints": [[] for _ in range(7)],
        "commanded_left_joint_velocities": [[] for _ in range(7)],
        "achieved_left_joints": [[] for _ in range(7)],
        "achieved_left_joint_velocities": [[] for _ in range(7)],
        "achieved_left_joint_torques": [[] for _ in range(7)],
        "achieved_left_joint_torques_minus_gravity": [[] for _ in range(6)],
        "commanded_right_joints": [[] for _ in range(7)],
        "commanded_right_joint_velocities": [[] for _ in range(7)],
        "achieved_right_joints": [[] for _ in range(7)],
        "achieved_right_joint_velocities": [[] for _ in range(7)],
        "achieved_right_joint_torques": [[] for _ in range(7)],
        "achieved_right_joint_torques_minus_gravity": [[] for _ in range(6)],
    }


def record_tracking(tracking, is_dual_arm, left_action, left_action_v, left_actual, left_actual_vel,
                    right_action=None, right_action_v=None, right_actual=None, right_actual_vel=None,
                    left_actual_torque=None, right_actual_torque=None, left_torque_minus_gravity=None,
                    right_torque_minus_gravity=None):
    if tracking is None:
        return
    for j in range(7):
        tracking["commanded_left_joints"][j].append(left_action[j])
        tracking["commanded_left_joint_velocities"][j].append(left_action_v[j])
        tracking["achieved_left_joints"][j].append(left_actual[j])
        tracking["achieved_left_joint_velocities"][j].append(left_actual_vel[j])
        if left_actual_torque is not None:
            tracking["achieved_left_joint_torques"][j].append(left_actual_torque[j])
        if is_dual_arm:
            tracking["commanded_right_joints"][j].append(right_action[j])
            tracking["commanded_right_joint_velocities"][j].append(right_action_v[j])
            tracking["achieved_right_joints"][j].append(right_actual[j])
            tracking["achieved_right_joint_velocities"][j].append(right_actual_vel[j])
            if right_actual_torque is not None:
                tracking["achieved_right_joint_torques"][j].append(right_actual_torque[j])
    if left_torque_minus_gravity is not None:
        for j in range(6):
            tracking["achieved_left_joint_torques_minus_gravity"][j].append(left_torque_minus_gravity[j])
    if is_dual_arm and right_torque_minus_gravity is not None:
        for j in range(6):
            tracking["achieved_right_joint_torques_minus_gravity"][j].append(right_torque_minus_gravity[j])


def save_tracking_plots(task_name, episode_idx, tick_delay, is_dual_arm, tracking, replay_gain_suffix=""):
    import matplotlib.pyplot as plt

    if tracking is None:
        print("[->] Tracking plots disabled, skipping replay data recording and plot generation.")
        return
    if len(tracking["commanded_left_joints"][0]) <= tick_delay:
        print("[WARN] Not enough replay samples to create tracking plots.")
        return

    prefix = f"{task_name}_ep{episode_idx}" if task_name and episode_idx is not None else "replay"

    def create_tracking_plot(commanded_data, achieved_data, plot_type, arm_label, filename):
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()

        unit = "rad" if plot_type == "position" else "rad/s"
        data_label = "Position" if plot_type == "position" else "Velocity"

        for joint_num in range(7):
            commanded_plot = np.array(commanded_data[joint_num][:-tick_delay])
            achieved_plot = np.array(achieved_data[joint_num][tick_delay:])
            time_steps = np.arange(len(commanded_plot))

            ax = axes[joint_num]
            ax.plot(time_steps, commanded_plot, label=f"Commanded {data_label}[t]", linewidth=2, alpha=0.8)
            ax.plot(time_steps, achieved_plot, label=f"Achieved {data_label}[t+{tick_delay}]", linewidth=2, alpha=0.8)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel(f"Joint {joint_num} {data_label} ({unit})", fontsize=10)
            ax.set_title(
                f"{arm_label}Joint {joint_num}: {data_label} Tracking ({tick_delay}-tick delay)",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[7].axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"[OK] Saved: {filename}")
        plt.close(fig)

    def create_torque_plot(torque_data, arm_label, filename):
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()

        for joint_num in range(7):
            torque_plot = np.array(torque_data[joint_num])
            time_steps = np.arange(len(torque_plot))
            joint_label = "Gripper" if joint_num == 6 else f"Joint {joint_num}"

            ax = axes[joint_num]
            ax.plot(time_steps, torque_plot, label="Recorded Torque", linewidth=2, alpha=0.85)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel(f"{joint_label} Torque (Nm)", fontsize=10)
            ax.set_title(
                f"{arm_label}{joint_label}: Recorded Torque",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[7].axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"[OK] Saved: {filename}")
        plt.close(fig)

    def create_torque_minus_gravity_plot(torque_data, arm_label, filename):
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        axes = axes.flatten()

        for joint_num in range(6):
            torque_plot = np.array(torque_data[joint_num])
            time_steps = np.arange(len(torque_plot))

            ax = axes[joint_num]
            ax.plot(time_steps, torque_plot, label="Measured - Gravity Compensation", linewidth=2, alpha=0.85)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel(f"Joint {joint_num} Torque (Nm)", fontsize=10)
            ax.set_title(
                f"{arm_label}Joint {joint_num}: Torque Minus Gravity",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"[OK] Saved: {filename}")
        plt.close(fig)

    def replay_plot_filename(filename):
        if not replay_gain_suffix:
            return filename
        path = Path(filename)
        return str(path.with_name(f"{path.stem}{replay_gain_suffix}{path.suffix}"))

    print(f"\n[->] Creating tracking plots (tick_delay={tick_delay})...")
    create_tracking_plot(
        tracking["commanded_left_joints"],
        tracking["achieved_left_joints"],
        "position",
        "Left Arm " if is_dual_arm else "",
        replay_plot_filename(f"{prefix}_left_position.png" if is_dual_arm else f"{prefix}_position.png"),
    )
    create_tracking_plot(
        tracking["commanded_left_joint_velocities"],
        tracking["achieved_left_joint_velocities"],
        "velocity",
        "Left Arm " if is_dual_arm else "",
        replay_plot_filename(f"{prefix}_left_velocity.png" if is_dual_arm else f"{prefix}_velocity.png"),
    )
    if any(tracking["achieved_left_joint_torques"]):
        create_torque_plot(
            tracking["achieved_left_joint_torques"],
            "Left Arm " if is_dual_arm else "",
            replay_plot_filename(f"{prefix}_left_torque.png" if is_dual_arm else f"{prefix}_torque.png"),
        )
    if any(tracking["achieved_left_joint_torques_minus_gravity"]):
        create_torque_minus_gravity_plot(
            tracking["achieved_left_joint_torques_minus_gravity"],
            "Left Arm " if is_dual_arm else "",
            replay_plot_filename(
                f"{prefix}_left_torque_minus_gravity.png" if is_dual_arm else f"{prefix}_torque_minus_gravity.png"
            ),
        )
    if is_dual_arm:
        create_tracking_plot(
            tracking["commanded_right_joints"],
            tracking["achieved_right_joints"],
            "position",
            "Right Arm ",
            replay_plot_filename(f"{prefix}_right_position.png"),
        )
        create_tracking_plot(
            tracking["commanded_right_joint_velocities"],
            tracking["achieved_right_joint_velocities"],
            "velocity",
            "Right Arm ",
            replay_plot_filename(f"{prefix}_right_velocity.png"),
        )
        if any(tracking["achieved_right_joint_torques"]):
            create_torque_plot(
                tracking["achieved_right_joint_torques"],
                "Right Arm ",
                replay_plot_filename(f"{prefix}_right_torque.png"),
            )
        if any(tracking["achieved_right_joint_torques_minus_gravity"]):
            create_torque_minus_gravity_plot(
                tracking["achieved_right_joint_torques_minus_gravity"],
                "Right Arm ",
                replay_plot_filename(f"{prefix}_right_torque_minus_gravity.png"),
            )


def replay_arx_actions(action_list, action_v_list=None, is_dual_arm=False, task_name=None, episode_idx=None,
                       dt=DT, replay_source="processed", use_fill_in=False, plot_tracking=True,
                       left_can="can0", right_can="can1", kp=None, kd=None,
                       single_arm_side=None, plot_torque_minus_gravity=False):
    import arx5_interface as arx5

    if use_fill_in:
        raise ValueError("ARX replay uses native 500 Hz interpolation; omit --fill_in")
    action_list, action_v_list = prepare_action_arrays(action_list, action_v_list, False)
    dt = replay_timestep(1.0 / dt)
    plot_torque_minus_gravity = bool(plot_torque_minus_gravity and plot_tracking)

    print("\n" + "=" * 70)
    print("  ARX Arm Action Replay")
    print("=" * 70)
    print(f"  Mode: {'Double Arm' if is_dual_arm else 'Single Arm'}")
    print(f"  Source: {replay_source}")
    print(f"  Total actions: {len(action_list)}")
    print(f"  Replay rate: {1 / dt:.1f} Hz")
    if is_dual_arm:
        print(f"  Left CAN: {left_can}")
    else:
        print(f"  Single-arm side: {single_arm_side or 'legacy/unspecified'}")
        print(f"  Active CAN: {left_can}")
    if is_dual_arm:
        print(f"  Right CAN: {right_can}")
    print(f"  Tracking plots: {'enabled' if plot_tracking else 'disabled'}")
    print(f"  Torque minus gravity plot: {'enabled' if plot_torque_minus_gravity else 'disabled'}")
    print("=" * 70)

    np.set_printoptions(precision=3, suppress=True)
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config
    )
    configure_active_gains(controller_config)
    controller_config.gravity_compensation = True
    # Friction feedforward is for hand guiding only; it stays off during replay.
    controller_config.friction_compensation = False
    controller_config.background_send_recv = True
    configure_controller(controller_config)

    print("\n[->] Initializing ARX arm(s)...")
    active_label = 'left' if is_dual_arm else (single_arm_side or 'left')
    print(f"[->] Active {active_label} controller on {left_can}")
    controller_left = arx5.Arx5JointController(robot_config, controller_config, left_can)
    controller_right = arx5.Arx5JointController(robot_config, controller_config, right_can) if is_dual_arm else None
    configure_arx_gain(controller_left, kp=kp, kd=kd)
    if controller_right is not None:
        configure_arx_gain(controller_right, kp=kp, kd=kd)
    gravity_solver = make_arx_solver(arx5, robot_config) if plot_torque_minus_gravity and plot_tracking else None

    cmd_left = make_arx_joint_command(arx5, robot_config)
    cmd_right = make_arx_joint_command(arx5, robot_config) if is_dual_arm else None
    tracking = init_tracking_buffers(plot_tracking)
    completed = False
    interrupted = False
    tick_delay = 3
    replay_plot_gain_suffix = format_arx_gain_filename_suffix(controller_left.get_gain())

    failed = False
    try:
        # Any failure, including Ctrl+C: native damping until 'h', then home.
        with damping_exit(lambda: ARXDamping([controller_left, controller_right])):
            print("\n[->] Moving to home position...")
            controller_left.reset_to_home()
            if controller_right is not None:
                controller_right.reset_to_home()
            replay_plot_gain_suffix = format_arx_gain_filename_suffix(controller_left.get_gain())

            print("[->] Going to the first pose...")
            first_left_action = action_list[0][:7]
            first_right_action = action_list[0][7:14] if is_dual_arm else None
            init_time = time.time()
            while time.time() - init_time < 2.0:
                ratio = (time.time() - init_time) / 2.0
                set_arx_action(controller_left, cmd_left, first_left_action * ratio, dt)
                if is_dual_arm:
                    set_arx_action(controller_right, cmd_right, first_right_action * ratio, dt)
                time.sleep(dt)

            print("\n[->] Starting replay...")

            start_time = time.time()
            for i, action in enumerate(action_list):
                loop_start = time.monotonic()
                action_v = action_v_list[i] if action_v_list is not None else None
                left_action = action[:7]
                left_action_v = action_v[:7] if action_v is not None else np.zeros(7)
                set_arx_action(controller_left, cmd_left, left_action, dt)

                if is_dual_arm:
                    right_action = action[7:14]
                    right_action_v = action_v[7:14] if action_v is not None else np.zeros(7)
                    set_arx_action(controller_right, cmd_right, right_action, dt)
                else:
                    right_action = right_action_v = None

                if tracking is not None:
                    include_torque = tracking is not None
                    left_actual, left_actual_vel, left_actual_torque = read_arx_state(
                        controller_left,
                        include_torque=include_torque,
                    )
                    if is_dual_arm:
                        right_actual, right_actual_vel, right_actual_torque = read_arx_state(
                            controller_right,
                            include_torque=include_torque,
                        )
                    else:
                        right_actual = right_actual_vel = right_actual_torque = None
                    left_torque_minus_gravity = None
                    right_torque_minus_gravity = None
                    if gravity_solver is not None:
                        left_torque_minus_gravity = compute_arx_torque_minus_gravity(
                            gravity_solver,
                            left_actual,
                            left_actual_torque,
                            robot_config.joint_dof,
                        )
                        if is_dual_arm:
                            right_torque_minus_gravity = compute_arx_torque_minus_gravity(
                                gravity_solver,
                                right_actual,
                                right_actual_torque,
                                robot_config.joint_dof,
                            )


                    record_tracking(
                        tracking,
                        is_dual_arm,
                        left_action,
                        left_action_v,
                        left_actual,
                        left_actual_vel,
                        right_action,
                        right_action_v,
                        right_actual,
                        right_actual_vel,
                        left_actual_torque=left_actual_torque,
                        right_actual_torque=right_actual_torque,
                        left_torque_minus_gravity=left_torque_minus_gravity,
                        right_torque_minus_gravity=right_torque_minus_gravity,
                    )

                # Fixed steps, in row order, just like policy action execution.
                time.sleep(max(0.0, dt - (time.monotonic() - loop_start)))
                if (i + 1) % 50 == 0:
                    print(f"[->] Progress: {i + 1}/{len(action_list)} actions")

            print("\n[OK] Replay completed!")
            print(f"[->] Total replay time: {time.time() - start_time:.2f} seconds")
            completed = True
    except KeyboardInterrupt:
        interrupted = failed = True
        print("\n[WARN] Replay interrupted by Ctrl+C; the failure exit damped, then homed on 'h'.")
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            print("\n[->] Returning to home position...")
            controller_left.reset_to_home()
            if controller_right is not None:
                controller_right.reset_to_home()
            time.sleep(1.0)


        save_tracking_plots(
            task_name,
            episode_idx,
            tick_delay,
            is_dual_arm,
            tracking,
            replay_gain_suffix=replay_plot_gain_suffix,
        )
