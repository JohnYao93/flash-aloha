#!/usr/bin/env python3
"""Practice with ARX arms and save a confirmed recording start pose."""

import argparse
from contextlib import nullcontext
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ACT_DIR = Path(__file__).resolve().parents[2] / "src"
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))
from utils.hardware_utils import configure_collection_gains, configure_controller, send_joint_target
from hardware.safe_exit import damping_exit


DEFAULT_CAN_INTERFACES = {
    "left": "can0",
    "right": "can1",
}
DEFAULT_CAMERA_NAMES = {
    "left": ["left", "overhead"],
    "right": ["right", "overhead"],
    "both": ["left", "overhead", "right"],
}
ARM_SAVE_ORDER = ["left", "right"]
CAMERA_SAVE_ORDER = ["left", "overhead", "right", "torso", "test"]
SHORT_CAMERA_NAMES = {"left", "overhead", "right", "torso", "test"}


def load_arx5_interface():
    """Import the local ARX5 Python binding."""
    sdk_python_dir = (
        Path(__file__).resolve().parents[1] / "src" / "hardware" / "arx5-sdk" / "python"
    )
    if str(sdk_python_dir) not in sys.path:
        sys.path.insert(0, str(sdk_python_dir))

    import arx5_interface as arx5

    return arx5


def parse_camera_names(camera_arg, arm):
    if camera_arg is None:
        return DEFAULT_CAMERA_NAMES[arm]
    if camera_arg.strip().lower() == "none":
        return []
    camera_names = [name.strip() for name in camera_arg.split(",") if name.strip()]
    if not camera_names:
        raise ValueError("--cameras must contain at least one camera name")
    return camera_names


def resolve_camera_device(camera_name):
    if camera_name.startswith("/dev/"):
        return camera_name
    if camera_name.isdigit():
        return f"/dev/video{int(camera_name)}"
    if camera_name in SHORT_CAMERA_NAMES:
        return f"/dev/video_{camera_name}"
    if camera_name.startswith("video_"):
        return f"/dev/{camera_name}"
    return f"/dev/{camera_name}"


def resolve_arm_can_interfaces(args):
    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    can_interfaces = {
        "left": args.left_can_interface or DEFAULT_CAN_INTERFACES["left"],
        "right": args.right_can_interface or DEFAULT_CAN_INTERFACES["right"],
    }

    if args.can_interface:
        if len(arms) == 1:
            can_interfaces[arms[0]] = args.can_interface
        else:
            parts = [part.strip() for part in args.can_interface.split(",")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    "--can-interface with --arm both must be a comma-separated pair "
                    "like 'can0,can1'. You can also use --left-can-interface and "
                    "--right-can-interface."
                )
            can_interfaces["left"], can_interfaces["right"] = parts

    selected = {arm: can_interfaces[arm] for arm in arms}
    if len(set(selected.values())) != len(selected):
        raise ValueError(f"Each arm needs a unique CAN interface, got {selected}")
    return selected


def build_arx_controllers(arx5, arm_can_interfaces, controller_fps):
    np.set_printoptions(precision=4, suppress=True)
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller",
        robot_config,
    )
    configure_collection_gains(controller_config)
    controller_config.gravity_compensation = True
    controller_config.background_send_recv = True
    configure_controller(controller_config)

    controllers = {}
    for arm, can_interface in arm_can_interfaces.items():
        print(f"Initializing ARX {arm} arm on {can_interface}...")
        controllers[arm] = arx5.Arx5JointController(
            robot_config,
            controller_config,
            can_interface,
        )
    return robot_config, controller_config, controllers


def configured_controller_gain_snapshot(controller, controller_config):
    """Build active position gains from SDK config, not damping-only startup state."""
    gain = controller.get_gain()
    kp = np.zeros_like(np.asarray(gain.kp(), dtype=np.float64))
    kd = np.zeros_like(np.asarray(gain.kd(), dtype=np.float64))
    default_kp = np.asarray(controller_config.default_kp, dtype=np.float64).reshape(-1)
    default_kd = np.asarray(controller_config.default_kd, dtype=np.float64).reshape(-1)
    kp_count = min(len(kp), len(default_kp))
    kd_count = min(len(kd), len(default_kd))
    kp[:kp_count] = default_kp[:kp_count]
    kd[:kd_count] = default_kd[:kd_count]
    return {
        "kp": kp,
        "kd": kd,
        "gripper_kp": float(controller_config.default_gripper_kp),
        "gripper_kd": float(controller_config.default_gripper_kd),
    }


def apply_controller_gain_snapshot(controller, snapshot):
    gain = controller.get_gain()
    gain.kp()[:] = snapshot["kp"]
    gain.kd()[:] = snapshot["kd"]
    gain.gripper_kp = snapshot["gripper_kp"]
    gain.gripper_kd = snapshot["gripper_kd"]
    controller.set_gain(gain)


def apply_zero_controller_gain(controller):
    gain = controller.get_gain()
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)


def apply_passive_gripper_gain(controller):
    """Keep the arm position-controlled while allowing manual gripper motion."""
    gain = controller.get_gain()
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)


def enter_practice_mode(controllers):
    for controller in controllers.values():
        apply_zero_controller_gain(controller)
        # Hand guiding, as in the recorder: gravity and friction feedforward only.
        controller.set_friction_compensation(True)
    print("Practice mode active. Move the arm(s), then press Space in a camera window.")


def enter_hold_mode(controllers, gain_snapshots, *, gripper_only=False):
    for arm, controller in controllers.items():
        controller.set_friction_compensation(False)  # hand-guiding support only
        apply_controller_gain_snapshot(controller, gain_snapshots[arm])
    if gripper_only:
        print(
            "Arms and grippers locked. Press 'y' to save, 'n' to adjust the "
            "grippers again, or 'q' to quit."
        )
    else:
        print("Arms locked. Press 'y' to save, 'n' to return to practice, or 'q' to quit.")


def read_locked_states(controllers):
    return {arm: controller.get_joint_state() for arm, controller in controllers.items()}


def build_hold_commands(arx5, robot_config, locked_states):
    commands = {}
    for arm, joint_state in locked_states.items():
        command = arx5.JointState(robot_config.joint_dof)
        command.pos()[:] = joint_state.pos().copy()
        command.gripper_pos = float(joint_state.gripper_pos)
        commands[arm] = command
    return commands


def build_qpos_commands(arx5, robot_config, qpos_by_arm, gripper_states=None):
    commands = {}
    for arm, qpos in qpos_by_arm.items():
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != 7:
            raise ValueError(f"{arm}_qpos must contain 7 values, got {qpos.size}")
        command = arx5.JointState(robot_config.joint_dof)
        command.pos()[:] = qpos[:6]
        command.gripper_pos = (
            float(gripper_states[arm].gripper_pos)
            if gripper_states is not None
            else float(qpos[6])
        )
        commands[arm] = command
    return commands


def send_hold_commands(controllers, commands, interval_s=None):
    for arm, controller in controllers.items():
        if interval_s is None:
            controller.set_joint_cmd(commands[arm])
        else:
            send_joint_target(controller, commands[arm], interval_s)


def move_to_saved_start_position(
    arx5,
    robot_config,
    controllers,
    saved_qpos_by_arm,
    gain_snapshots,
    *,
    duration,
    controller_fps,
    smooth=False,
):
    """Smoothly command selected arms and grippers to their existing saved pose."""
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("--move-duration must be positive and finite")

    current_by_arm = {
        arm: joint_state_to_qpos(controller.get_joint_state())
        for arm, controller in controllers.items()
    }
    # Seed each controller with its measured pose before starting interpolation.
    send_hold_commands(
        controllers,
        build_qpos_commands(arx5, robot_config, current_by_arm),
    )
    # ARX controllers start in damping mode with kp=0. Enable configured
    # position gains only after the command buffer matches the measured pose;
    # this ordering also satisfies the SDK's anti-jump safety check.
    for arm, controller in controllers.items():
        controller.set_friction_compensation(False)  # hand-guiding support only
        apply_controller_gain_snapshot(controller, gain_snapshots[arm])
    steps = max(1, int(round(duration * controller_fps)))
    dt = duration / steps
    started = time.monotonic()
    worst_lateness, worst_step = 0.0, 0
    for step in range(1, steps + 1):
        alpha = step / steps
        if smooth:
            # Match the SDK home trajectory's zero endpoint velocities.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        qpos_by_arm = {
            arm: current_by_arm[arm]
            + (np.asarray(saved_qpos_by_arm[arm]) - current_by_arm[arm]) * alpha
            for arm in controllers
        }
        send_hold_commands(
            controllers,
            build_qpos_commands(arx5, robot_config, qpos_by_arm),
            interval_s=dt,
        )
        remaining = started + step * dt - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            # Never catch up after a stall: a burst of missed steps would make
            # the SDK traverse them within milliseconds. Pause the plan instead.
            started -= remaining
            if -remaining > worst_lateness:
                worst_lateness, worst_step = -remaining, step
    if worst_lateness > 2.0 * dt:
        print(f"[Move] Trajectory loop stalled up to {worst_lateness * 1e3:.0f} ms "
              f"at step {worst_step}/{steps}; the move was slowed, not rushed")


def enter_gripper_adjust_mode(
    arx5,
    robot_config,
    controllers,
    gain_snapshots,
    saved_qpos_by_arm,
):
    """Hold saved arm joints and make only the grippers manually adjustable."""
    live_states = read_locked_states(controllers)
    commands = build_qpos_commands(
        arx5,
        robot_config,
        saved_qpos_by_arm,
        gripper_states=live_states,
    )
    for arm, controller in controllers.items():
        apply_controller_gain_snapshot(controller, gain_snapshots[arm])
        apply_passive_gripper_gain(controller)
        # The gripper is moved by hand here, so friction feedforward is on.
        controller.set_friction_compensation(True)
    send_hold_commands(controllers, commands)
    print(
        "Saved arm pose held. Adjust only the grippers, then press Space "
        "in a camera window."
    )
    return commands


def return_arms_home(controllers):
    if not controllers:
        return

    print("Returning arm(s) to home before exit...")
    for controller in controllers.values():
        # reset_to_home() safely seeds the current command and restores default
        # gains itself when the controller is in zero-kp/passive mode.
        controller.reset_to_home()


def joint_state_to_qpos(joint_state):
    """Return six arm joints followed by that arm's measured gripper position."""
    return np.concatenate((joint_state.pos().copy(), [float(joint_state.gripper_pos)]))


def _string_list(value):
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _ordered_unique(names, preferred_order):
    seen = set()
    unique = []
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    ordered = [name for name in preferred_order if name in seen]
    ordered.extend(name for name in unique if name not in preferred_order)
    return ordered


def load_existing_start_position(output_path):
    output_path = Path(output_path).expanduser()
    per_arm_qpos = {}
    can_by_arm = {}
    camera_names = []
    if not output_path.exists():
        return per_arm_qpos, can_by_arm, camera_names

    with np.load(output_path, allow_pickle=True) as data:
        for arm in ARM_SAVE_ORDER:
            key = f"{arm}_qpos"
            if key in data:
                qpos = np.asarray(data[key], dtype=np.float64).reshape(-1)
                if qpos.size == 7:
                    per_arm_qpos[arm] = qpos

        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64).reshape(-1)
            if qpos.size >= 14:
                per_arm_qpos.setdefault("left", qpos[:7])
                per_arm_qpos.setdefault("right", qpos[7:14])

        if "arms" in data and "can_interfaces" in data:
            for arm, can_interface in zip(_string_list(data["arms"]), _string_list(data["can_interfaces"])):
                can_by_arm[arm] = can_interface
        if "camera_names" in data:
            camera_names = _string_list(data["camera_names"])

    return per_arm_qpos, can_by_arm, camera_names


def save_start_position(
    output_path,
    locked_states,
    arm_can_interfaces,
    camera_names,
    *,
    gripper_only=False,
    source_position_path=None,
):
    output_path = Path(output_path).expanduser()
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = (
        Path(source_position_path).expanduser()
        if source_position_path is not None
        else output_path
    )
    per_arm_qpos, can_by_arm, previous_camera_names = load_existing_start_position(
        source_path
    )
    for arm, joint_state in locked_states.items():
        measured_qpos = joint_state_to_qpos(joint_state)
        if gripper_only:
            if arm not in per_arm_qpos:
                raise ValueError(
                    f"Cannot update only the {arm} gripper: {source_path} has no "
                    f"saved {arm} arm pose"
                )
            updated_qpos = per_arm_qpos[arm].copy()
            updated_qpos[6] = measured_qpos[6]
            per_arm_qpos[arm] = updated_qpos
        else:
            per_arm_qpos[arm] = measured_qpos
    for arm, can_interface in arm_can_interfaces.items():
        can_by_arm[arm] = can_interface

    arm_names = _ordered_unique(per_arm_qpos.keys(), ARM_SAVE_ORDER)
    qpos = np.concatenate([per_arm_qpos[arm] for arm in arm_names])
    merged_camera_names = _ordered_unique([*previous_camera_names, *camera_names], CAMERA_SAVE_ORDER)

    payload = {
        "arms": np.asarray(arm_names),
        "qpos": qpos,
        "can_interfaces": np.asarray([can_by_arm.get(arm, DEFAULT_CAN_INTERFACES[arm]) for arm in arm_names]),
        "camera_names": np.asarray(merged_camera_names),
        "created_unix_time": np.asarray(time.time(), dtype=np.float64),
    }
    for arm, arm_qpos in per_arm_qpos.items():
        payload[f"{arm}_qpos"] = arm_qpos

    np.savez(output_path, **payload)
    return output_path


class LiveCamera:
    def __init__(
        self,
        name,
        width,
        height,
        fps,
        autofocus,
        focus,
        gain,
        scale,
        show_fps,
    ):
        self.name = name
        self.device_path = resolve_camera_device(name)
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(fps)
        self.autofocus = int(autofocus)
        self.focus = int(focus)
        self.gain = int(gain)
        self.scale = float(scale)
        self.show_fps = bool(show_fps)
        self.cap = None
        self.window_name = f"Recording setup - {name}"
        self.actual_width = self.width
        self.actual_height = self.height
        self.window_width = max(1, int(self.width * self.scale))
        self.window_height = max(1, int(self.height * self.scale))
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.display_fps = 0.0
        self.last_warn_time = 0.0

    def open(self):
        if not os.path.exists(self.device_path):
            raise RuntimeError(f"{self.device_path} does not exist")

        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {self.device_path}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 3615)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, self.autofocus)
        cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        cap.set(cv2.CAP_PROP_GAIN, self.gain)

        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    self.device_path,
                    "--set-ctrl=exposure_dynamic_framerate=0",
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pass

        self.cap = cap
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.window_width = max(1, int(self.actual_width * self.scale))
        self.window_height = max(1, int(self.actual_height * self.scale))

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(
            f"Camera {self.name}: {self.device_path} "
            f"{self.actual_width}x{self.actual_height} @ {actual_fps:.1f} FPS"
        )

    def update(self, status_text):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            now = time.monotonic()
            if now - self.last_warn_time >= 1.0:
                print(f"[WARN] Failed to grab frame from {self.name}")
                self.last_warn_time = now
            return

        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.display_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

        if self.scale != 1.0:
            display_frame = cv2.resize(
                frame,
                (self.window_width, self.window_height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            display_frame = frame.copy()

        if self.show_fps:
            cv2.putText(
                display_frame,
                f"FPS: {self.display_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"{self.actual_width}x{self.actual_height}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display_frame,
            status_text,
            (10, max(90, self.window_height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(self.window_name, display_frame)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_cameras(args, camera_names):
    cameras = []
    for camera_name in camera_names:
        camera = LiveCamera(
            name=camera_name,
            width=args.width,
            height=args.height,
            fps=args.fps,
            autofocus=args.autofocus,
            focus=args.focus,
            gain=args.gain,
            scale=args.scale,
            show_fps=not args.no_fps,
        )
        camera.open()
        cameras.append(camera)
    return cameras


class TerminalKeys:
    """Keys typed in this terminal, for a run with no camera windows."""

    def __init__(self):
        self.keys = queue.Queue()
        threading.Thread(target=self._run, name="setup-terminal-keys", daemon=True).start()

    def _run(self):
        from getkey import getkey
        while True:
            try:
                key = getkey()
            except (EOFError, KeyboardInterrupt):
                return  # Ctrl+C also reaches the signal handler.
            except Exception:
                time.sleep(0.1)
                continue
            if key:
                self.keys.put(str(key))

    def poll(self):
        """One waitKey-style code (255 when nothing was typed)."""
        try:
            key = self.keys.get_nowait()
        except queue.Empty:
            return 255
        return ord(key.lower()) if len(key) == 1 else 255

    def drain(self):
        keys = []
        while (key := self.poll()) != 255:
            keys.append(key)
        return keys


def update_cameras(cameras, status_text):
    for camera in cameras:
        camera.update(status_text)
    return cv2.waitKey(1) & 0xFF


def close_cameras(cameras):
    for camera in cameras:
        camera.close()
    cv2.destroyAllWindows()


def _require_task_flag(args, field, expected):
    actual = getattr(args, field, None)
    if actual is not None and actual != expected:
        flag = '--' + field.replace('_', '-')
        raise ValueError(f'{flag}={actual} conflicts with the task YAML value {expected}; edit the task YAML')


def apply_task_config(args):
    """Fill robot, arms, CAN, cameras and output from --task_name/--task-config."""
    task_name, task_path = getattr(args, 'task_name', None), getattr(args, 'task_config', None)
    if task_name is None and task_path is None:
        return None
    from config.task_config import (REPO_ROOT, default_start_position_path,
                                    select_task_config, task_start_position_path)
    task = select_task_config(task_name, task_path)
    if task is None:
        raise ValueError(f'Unknown task {task_name!r}: no src/config/tasks/{task_name}.yaml')
    doc = task.as_dict()
    robot, arms = doc['robot']['type'], doc['robot']['arms']
    arm = 'both' if len(arms) == 2 else next(iter(arms))
    _require_task_flag(args, 'robot', robot)
    _require_task_flag(args, 'arm', arm)
    if args.can_interface is not None:
        expected = ','.join(arms[side]['can_interface'] for side in ARM_SAVE_ORDER if side in arms)
        _require_task_flag(args, 'can_interface', expected)
    for side in ARM_SAVE_ORDER:
        if side in arms:
            _require_task_flag(args, f'{side}_can_interface', arms[side]['can_interface'])
            setattr(args, f'{side}_can_interface', arms[side]['can_interface'])
    args.robot, args.arm, args.can_interface = robot, arm, None
    if args.cameras is None:
        args.cameras = ','.join(name.removeprefix('video_') for name in doc['cameras']['names']) or 'none'
    if args.output is None:
        args.output = str(REPO_ROOT / task_start_position_path(task.name))
    if args.gripper_only and args.source_position is None and not Path(args.output).exists():
        # First per-task gripper adjustment starts from the shared pose.
        args.source_position = str(REPO_ROOT / default_start_position_path(None, robot))
    ports = ', '.join(f"{side}={spec['can_interface']}" for side, spec in arms.items())
    print(f'Task: {task.name} ({robot}, {ports})')
    return task


def run_setup(args):
    apply_task_config(args)
    if getattr(args, 'robot', None) is None:
        args.robot = 'arx'
    if getattr(args, 'arm', None) is None:
        args.arm = 'left'
    return _run_setup(args)


def _run_setup(args):
    if args.controller_fps is None:
        args.controller_fps = 50.0
    if args.controller_fps <= 0:
        raise ValueError("--controller-fps must be > 0")

    arm_can_interfaces = resolve_arm_can_interfaces(args)
    if args.output is None:
        args.output = './calibration/start_position.npz'
    camera_names = parse_camera_names(args.cameras, args.arm)
    saved_qpos_by_arm = None
    source_position_path = None
    if args.gripper_only:
        source_position_path = args.source_position or args.output
        existing_qpos, _, _ = load_existing_start_position(source_position_path)
        missing = [arm for arm in arm_can_interfaces if arm not in existing_qpos]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "--gripper-only requires an existing saved arm pose for "
                f"{missing_text} in {Path(source_position_path).expanduser()}"
            )
        saved_qpos_by_arm = {
            arm: existing_qpos[arm].copy() for arm in arm_can_interfaces
        }

    print("Connecting to arms:")
    for arm, can_interface in arm_can_interfaces.items():
        print(f"  {arm}: {can_interface}")
    print(f"Cameras: {', '.join(camera_names) or 'none'}")
    terminal_keys = None
    if not camera_names:
        terminal_keys = TerminalKeys()
        print("No camera windows: type the keys in this terminal "
              "(Space lock, y save, n retry, q quit).")
    print(f"Output: {Path(args.output).expanduser()}")
    if args.gripper_only:
        print(
            "Mode: adjust grippers only; existing six-joint arm poses will be "
            "preserved"
        )
        print(f"Source pose: {Path(source_position_path).expanduser()}")

    arx5 = load_arx5_interface()
    robot_config, controller_config, controllers = build_arx_controllers(
        arx5,
        arm_can_interfaces=arm_can_interfaces,
        controller_fps=args.controller_fps,
    )

    if args.reset_home:
        print("Resetting arm(s) to home...")
        for controller in controllers.values():
            controller.reset_to_home()

    gain_snapshots = {
        arm: configured_controller_gain_snapshot(controller, controller_config)
        for arm, controller in controllers.items()
    }
    cameras = []
    synchronize = nullcontext
    failed = False
    home_reader = None

    def damping_target():
        from hardware.safe_exit import ARXDamping
        return ARXDamping(controllers.values())

    def home_requested():
        # 'h' in a camera window or on the terminal; keep the windows live.
        nonlocal home_reader
        if terminal_keys is not None:
            # The terminal reader owns stdin; HomeKeyReader would compete for it.
            return ord("h") in terminal_keys.drain()
        if home_reader is None:
            from hardware.safe_exit import HomeKeyReader
            home_reader = HomeKeyReader()
        key = update_cameras(cameras, "DAMPING: press h to return home") if cameras else 255
        return key == ord("h") or home_reader.home.is_set()

    try:
        # Any failure, including Ctrl+C: damping until 'h', then home, then exit.
        with damping_exit(damping_target, home_requested=home_requested):
            cameras = open_cameras(args, camera_names)

            state = "gripper_adjust" if args.gripper_only else "practice"
            locked_states = None
            hold_commands = None
            last_hold_time = 0.0
            hold_dt = 1.0 / args.controller_fps
            with synchronize():
                if args.gripper_only:
                    print(
                        f"Moving to the existing saved arm and gripper pose over "
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
                    hold_commands = enter_gripper_adjust_mode(
                        arx5,
                        robot_config,
                        controllers,
                        gain_snapshots,
                        saved_qpos_by_arm,
                    )
                    last_hold_time = time.monotonic()
                else:
                    enter_practice_mode(controllers)

            while True:
                if state == "practice":
                    status_text = "Practice: Space lock | q quit"
                elif state == "gripper_adjust":
                    status_text = "Arms held: adjust grippers | Space lock | q quit"
                else:
                    status_text = "Locked: y save | n retry | q quit"

                if terminal_keys is not None:
                    key = terminal_keys.poll()
                    time.sleep(0.005)  # The camera windows' frame wait paces the loop otherwise.
                else:
                    key = update_cameras(cameras, status_text)
                with synchronize():
                    if key in (ord("q"), 27):
                        print("Quit requested.")
                        return 1

                    now = time.monotonic()
                    if (state in {"gripper_adjust", "locked"}
                            and now - last_hold_time >= hold_dt):
                        if state == "gripper_adjust":
                            hold_commands = build_qpos_commands(
                                arx5,
                                robot_config,
                                saved_qpos_by_arm,
                                gripper_states=read_locked_states(controllers),
                            )
                        send_hold_commands(controllers, hold_commands)
                        last_hold_time = now

                    if state in {"practice", "gripper_adjust"} and key == ord(" "):
                        locked_states = read_locked_states(controllers)
                        if args.gripper_only:
                            hold_commands = build_qpos_commands(
                                arx5,
                                robot_config,
                                saved_qpos_by_arm,
                                gripper_states=locked_states,
                            )
                        else:
                            hold_commands = build_hold_commands(arx5, robot_config, locked_states)
                        send_hold_commands(controllers, hold_commands)
                        enter_hold_mode(
                            controllers,
                            gain_snapshots,
                            gripper_only=args.gripper_only,
                        )
                        send_hold_commands(controllers, hold_commands)
                        last_hold_time = time.monotonic()
                        state = "locked"
                    elif state == "locked" and key == ord("n"):
                        locked_states = None
                        if args.gripper_only:
                            hold_commands = enter_gripper_adjust_mode(
                                arx5,
                                robot_config,
                                controllers,
                                gain_snapshots,
                                saved_qpos_by_arm,
                            )
                            last_hold_time = time.monotonic()
                            state = "gripper_adjust"
                        else:
                            enter_practice_mode(controllers)
                            hold_commands = None
                            state = "practice"
                    elif state == "locked" and key == ord("y"):
                        output_path = save_start_position(
                            args.output,
                            locked_states=locked_states,
                            arm_can_interfaces=arm_can_interfaces,
                            camera_names=camera_names,
                            gripper_only=args.gripper_only,
                            source_position_path=source_position_path,
                        )
                        if args.gripper_only:
                            print(f"Saved gripper positions to {output_path}; arm poses unchanged")
                        else:
                            print(f"Saved start position to {output_path}")
                        return 0
    except KeyboardInterrupt:
        failed = True
        print("\nInterrupted by user.")
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
        close_cameras(cameras)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Set up ARX arm(s), livestream cameras, lock a manually positioned "
            "start pose, and save it to a .npz file."
        )
    )
    parser.add_argument('--task_name', '--task-name', dest='task_name', default=None,
                        help=('Task whose YAML supplies robot, arm(s), CAN interfaces and '
                              'cameras; saves calibration/start_positions/<task>.npz'))
    parser.add_argument('--task-config', dest='task_config', default=None,
                        help='Task YAML path (alternative to --task_name)')
    parser.add_argument('--robot', choices=['arx'], default=None, help='Arm backend (ARX)')
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default=None,
        help="Which arm(s) to connect to (default: left, or the task YAML); both defaults to left=can0, right=can1",
    )
    parser.add_argument(
        "--can-interface",
        default=None,
        help=(
            "Optional CAN interface override. With --arm both, pass "
            "a comma-separated pair like 'can0,can1'."
        ),
    )
    parser.add_argument(
        "--left-can-interface",
        default=None,
        help="CAN interface override for the left arm (default: can0)",
    )
    parser.add_argument(
        "--right-can-interface",
        default=None,
        help="CAN interface override for the right arm (default: can1)",
    )
    parser.add_argument(
        "--controller-fps",
        type=float,
        default=None,
        help="Practice/hold command rate (default: 50)",
    )
    parser.add_argument(
        "--reset-home",
        action="store_true",
        help="Reset the selected arm(s) to SDK home before calibration",
    )
    parser.add_argument(
        "--gripper-only",
        action="store_true",
        help=(
            "Move to and hold the existing saved arm pose, allow only gripper "
            "adjustment, and preserve the six saved arm joints when saving"
        ),
    )
    parser.add_argument(
        "--source-position",
        default=None,
        help=(
            "Existing .npz pose to load in --gripper-only mode. Defaults to "
            "--output; set this separately to preserve the source file."
        ),
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=3.0,
        help=(
            "Seconds to move to the existing saved pose in --gripper-only mode "
            "(default: 3.0)"
        ),
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help=(
            "Comma-separated cameras to livestream. Defaults to left,overhead for "
            "single left; right,overhead for single right; left,overhead,right for both. "
            "'none' opens no cameras; keys are then typed in the terminal."
        ),
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=320,
        help="Camera frame width (default: 320)",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=180,
        help="Camera frame height (default: 180)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target camera FPS (default: 60)",
    )
    parser.add_argument(
        "--autofocus",
        type=int,
        choices=[0, 1],
        default=0,
        help="Auto-focus mode: 0=manual, 1=auto (default: 0)",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=20,
        help="Manual focus value (default: 20)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=30,
        help="Camera gain value (default: 30)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=3.0,
        help="Display window scale factor (default: 3.0)",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS/resolution overlay",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Output .npz path (default: calibration/start_positions/<task>.npz with --task_name; "
              "otherwise calibration/start_position.npz)"),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_setup(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
