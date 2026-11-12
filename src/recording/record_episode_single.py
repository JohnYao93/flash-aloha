#!/usr/bin/env python3
"""Hand-guided episode recorder for ARX arms."""

from __future__ import annotations

# Support direct script execution and package imports from any working directory.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

from recording.recording_utils import get_auto_index, print_dt_diagnosis
import camera_utils.capture as camera_capture

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from getkey import getkey
from tqdm import tqdm

from contracts import standard_capture
from utils.hardware_utils import configure_collection_gains
from recording.recording_utils import save_recording_rate
from utils.hardware_utils import (
    ARX_CONTROLLER_HZ,
    ARX_COMMAND_HZ,
    configure_controller,
    replay_timestep,
    send_joint_target,
)
from recording.episode_buffer import EpisodeBuffer, recording_scope
from recording.pedal_keys import FAILURE_STAMP_KEY, GO_KEY, REJECT_KEYS
from hardware.safe_exit import QuitRequested, damping_exit
from config.constants import FPS, TASK_CONFIGS, resolve_arx_single_arm_can, resolve_arx_single_arm_side
from camera_utils.capture import capture_image, capture_image_from_shared_memory

DEFAULT_START_POSITION_PATH = "calibration/start_position.npz"


@dataclass
class RunContext:
    task_name: str
    dataset_dir: str
    max_timesteps: int
    camera_names: list[str]
    robot_type: str
    single_arm: bool
    double_arm: bool
    can_interface_left: str | None = None
    can_interface_right: str | None = None
    arx_single_arm_side: str | None = None
    fps: float = FPS
    use_start_position: bool = True
    start_position_path: str | None = DEFAULT_START_POSITION_PATH
    start_position_move_duration: float = 3.0
    deprecated_single_arm_override: bool = False
    robot_contract: dict | None = None
    hardware_config: dict | None = None

    @property
    def camera_short_names(self) -> list[str]:
        return [name.replace("video_", "") for name in self.camera_names]

    @property
    def camera_side(self) -> str:
        if self.arx_single_arm_side:
            return self.arx_single_arm_side
        if "video_right" in self.camera_names and "video_left" not in self.camera_names:
            return "right"
        return "left"

    @property
    def qpos_dim(self) -> int:
        return 7 if self.single_arm else 14

    @property
    def qvel_dim(self) -> int:
        return 7 if self.single_arm else 14


@dataclass
class EpisodeEvents:
    failure_event_timestamps: list[float] = field(default_factory=list)
    subgoal_switch_event_timestamps: list[float] = field(default_factory=list)
    recording_stop_event_timestamp: float | None = None
    partial_subgoal_start: bool = False
    partial_subgoal_start_timestamp: float | None = None


@dataclass
class EpisodeSample:
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    qvel: np.ndarray
    timestamp: float
    capture_interval: float | None = None
    robot_timestamp: float | None = field(default=None, kw_only=True)
    robot_read_started_at: float | None = field(default=None, kw_only=True)
    camera_observation: dict | None = field(default=None, kw_only=True)
    capture_contract_version: int | None = field(default=None, kw_only=True)

    @classmethod
    def from_capture(cls, observation, timestamp, **extra):
        return cls(images=observation.images, qpos=observation.qpos, qvel=observation.qvel,
                   timestamp=timestamp, robot_timestamp=observation.robot_timestamp,
                   robot_read_started_at=observation.robot_read_started_at,
                   camera_observation=observation.camera_observation,
                   capture_contract_version=standard_capture.CAPTURE_CONTRACT_VERSION, **extra)


def capture_episode_sample(backend, camera_names, timestamp, *, include_images=False):
    observation = standard_capture.capture_observation(
        lambda: read_robot_state(backend), camera_names, include_images=include_images)
    return EpisodeSample.from_capture(observation, timestamp)


def read_robot_state(backend):
    """Read position and velocity from the same snapshot per arm.

    The fallback supports older injected/simulated backends. Both physical
    backends below implement read_state; no recording uses two SDK reads.
    """
    if callable(getattr(backend, 'read_state', None)):
        return backend.read_state()
    return backend.read_qpos(), backend.read_qvel()


class CaptureTiming:
    """Report actual sample cadence in one-second active-capture windows."""

    def __init__(self, fps):
        self.fps = fps
        self.reset()

    def reset(self):
        self.elapsed = 0.
        self.intervals = self.missed = 0
        self.max_gap = 0.
        self.latest = self.warning = None

    def observe(self, interval):
        self.elapsed += interval
        self.intervals += 1
        # Estimated missing sampling intervals, not camera hardware drops.
        self.missed += max(0, math.floor(interval * self.fps + .5) - 1)
        self.max_gap = max(self.max_gap, interval)
        if self.elapsed < 1. - 1e-9:
            return
        hz = self.intervals / self.elapsed
        self.latest = dict(measured_hz=hz, target_hz=self.fps,
                           estimated_missed_samples=self.missed, max_gap_ms=self.max_gap * 1000)
        self.warning = (f'Capture timing: {hz:.1f}/{self.fps:g} Hz; '
                        f'{self.missed} estimated missed samples; '
                        f'longest gap {self.max_gap * 1000:.1f} ms in the last window.'
                        if hz < .96 * self.fps or self.missed else None)
        if self.warning:
            print(f'WARNING: {self.warning}', flush=True)
        self.elapsed = 0.
        self.intervals = self.missed = 0
        self.max_gap = 0.


class EpisodeCapture:
    """Owner-thread sampling shared by standalone and task-specific collectors.

    Uses standard_capture for state/image-reference pairing. Task callbacks may
    gate release but cannot customize camera selection or encode images.
    """

    def __init__(self, backend, camera_names, *, fps,
                 before_read=None, max_seconds=None, clock=None,
                 include_images=False, snapshot_provider=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Capture fps must be finite and positive')
        self.backend, self.camera_names = backend, camera_names
        self.fps, self.max_seconds = fps, max_seconds
        self.before_read = before_read
        self.include_images, self.snapshot_provider = include_images, snapshot_provider
        self.clock = clock or (lambda: time.monotonic())
        self.started = self.next_tick = self.clock()
        self.paused = False
        self.timing = CaptureTiming(fps)
        self.intervals = []
        self.previous_capture = None

    def due(self, force=False):
        return not self.paused and (force or self.clock() >= self.next_tick)

    def sample(self, *, force=False, pose=None):
        if not self.due(force):
            return None
        loop_start = self.clock()
        self.last_sample_started = loop_start
        if self.max_seconds is not None and loop_start - self.started >= self.max_seconds:
            raise OSError('Recording time limit reached; retry this arrangement')
        if self.before_read is not None:
            self.before_read()
        self.backend.passive_step()
        observation = standard_capture.capture_observation(
            lambda: read_robot_state(self.backend), self.camera_names, clock=self.clock,
            include_images=self.include_images, snapshot_provider=self.snapshot_provider)
        timestamp = observation.robot_timestamp - self.started
        sample = EpisodeSample.from_capture(observation, timestamp)
        # A checkpoint pose is a task hint, never an override of measured state.
        if self.previous_capture is not None and not force:
            interval = timestamp - self.previous_capture
            sample.capture_interval = interval
            self.intervals.append(interval)
            self.timing.observe(interval)
        self.previous_capture = timestamp
        self.next_tick = (loop_start + 1 / self.fps if force
                          else max(self.next_tick + 1 / self.fps, loop_start))
        return sample

    def wait_and_sample(self):
        if self.paused:
            raise ValueError('Cannot capture while paused')
        while not self.due():
            time.sleep(max(0., self.next_tick - self.clock()))
        return self.sample()

    def reset_timing(self):
        self.intervals.clear()
        self.previous_capture = None
        self.timing.reset()

    def pause(self):
        self.paused = True

    def resume(self, last_timestamp):
        now = self.clock()
        self.started = now - last_timestamp - 1 / self.fps
        self.next_tick = now
        self.reset_timing()
        self.paused = False


class ProprioJitterMonitor:
    def __init__(self, enabled: bool, window_sec: float, print_interval: float) -> None:
        self.enabled = bool(enabled)
        self.window_sec = float(window_sec)
        self.print_interval = float(print_interval)
        self.samples: deque[tuple[float, np.ndarray]] = deque()
        self.last_print = 0.0

    def update(self, timestamp: float, qpos: np.ndarray) -> None:
        if not self.enabled:
            return
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self.samples.append((timestamp, qpos))
        while self.samples and timestamp - self.samples[0][0] > self.window_sec:
            self.samples.popleft()
        if timestamp - self.last_print < self.print_interval or len(self.samples) < 2:
            return
        stacked = np.stack([sample for _, sample in self.samples], axis=0)
        peak_to_peak = np.ptp(stacked, axis=0)
        print(
            f"[Jitter] window={self.window_sec:.2f}s "
            f"max_qpos_ptp={float(np.max(peak_to_peak)):.5f} "
            f"mean_qpos_ptp={float(np.mean(peak_to_peak)):.5f}"
        )
        self.last_print = timestamp


def normalize_robot_type(robot_type: str | None) -> str:
    robot = (robot_type or "arx").lower()
    if robot == "arx5":
        robot = "arx"
    if robot != "arx":
        raise ValueError(f"Unsupported robot type {robot_type!r}; expected 'arx'")
    return robot


def resolve_run_context(args: argparse.Namespace | dict[str, Any]) -> RunContext:
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    from config.task_config import select_task_config, require_task_value
    task_config = TASK_CONFIGS.get(args.task_name, {})
    selected = select_task_config(args.task_name, getattr(args, 'task_config', None)) if (
        getattr(args, 'task_config', None) or task_config.get('_task_config_path')) else None
    if selected:
        task_config = selected.legacy()
        from camera_utils.camera_settings_contract import check_inference_camera_settings
        check_inference_camera_settings(task_config, context='collection')
        args.task_name = selected.name
        require_task_value(getattr(args, 'fps', None), selected.fps, 'fps')
        require_task_value(getattr(args, 'single_arm', None), task_config['single_arm'], 'single_arm')
        requested_robot = getattr(args, 'robot', None)
        if requested_robot is not None:
            require_task_value(normalize_robot_type(requested_robot), task_config['robot_type'], 'robot')
    if not task_config:
        raise ValueError('Choose a known --task_name or provide --task-config YAML')
    robot_type = normalize_robot_type(getattr(args, "robot", None) or task_config.get("robot_type", "arx"))

    single_arm = bool(task_config.get("single_arm", False))
    deprecated_override = getattr(args, "single_arm", None) is not None
    if deprecated_override:
        single_arm = bool(args.single_arm)
        print("[WARN] --single_arm is deprecated; set TASK_CONFIGS[task_name]['single_arm'] or use --robot instead.")

    can_interface_left = task_config.get("can_interface_left")
    can_interface_right = task_config.get("can_interface_right")
    arx_single_arm_side = None
    if robot_type == "arx" and single_arm:
        arx_single_arm_side = resolve_arx_single_arm_side(task_config)
        can_interface_left = resolve_arx_single_arm_can(task_config)

    from config.task_config import default_start_position_path
    start_position_path = getattr(args, 'start_position_path',
        default_start_position_path(args.task_name, robot_type))

    return RunContext(
        task_name=args.task_name,
        dataset_dir=task_config["dataset_dir"],
        max_timesteps=int(task_config["episode_len"]),
        camera_names=list(task_config["camera_names"]),
        robot_type=robot_type,
        single_arm=single_arm,
        double_arm=not single_arm,
        can_interface_left=can_interface_left,
        can_interface_right=can_interface_right,
        arx_single_arm_side=arx_single_arm_side,
        fps=float(selected.fps if selected else (getattr(args, "fps", None) or task_config.get('fps', FPS))),
        use_start_position=start_position_path is not None,
        start_position_path=start_position_path,
        start_position_move_duration=float(getattr(args, "start_position_move_duration", 3.0)),
        deprecated_single_arm_override=deprecated_override,
        robot_contract=task_config.get('robot_contract') or {'robot_contract_version': 1, 'arms': {
            role: {'model': 'L5'} for role in ([arx_single_arm_side or 'left'] if single_arm else ['left', 'right'])}},
        hardware_config=task_config.get('hardware_config'),
    )


def start_camera_capture(ctx: RunContext, use_camera_server: bool) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    if not ctx.camera_names:
        # Camera-less task: no readers to open; the thread only waits for stop.
        print("[Camera] Task has no cameras")
        thread = threading.Thread(target=stop_event.wait, daemon=True)
        thread.start()
        return stop_event, thread
    if use_camera_server:
        print(f"[Camera] Using camera-server shared memory: {ctx.camera_short_names}")
        target = capture_image_from_shared_memory
        kwargs = {"camera_names": ctx.camera_short_names}
        args = (stop_event,)
    else:
        print(f"[Camera] Opening cameras directly: {ctx.camera_names}")
        target = capture_image
        kwargs = {"camera_names": ctx.camera_names}
        args = (stop_event,)
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return stop_event, thread


def _camera_global(camera_name: str) -> np.ndarray | None:
    if camera_name == "video_left":
        return camera_capture.captured_img_l
    if camera_name == "video_overhead":
        return camera_capture.captured_img_h
    if camera_name == "video_right":
        return camera_capture.captured_img_r
    if camera_name == "video_torso":
        return camera_capture.captured_img_t
    if camera_name in ('video_extra', 'video_extra_overhead'):
        packet = camera_capture.captured_camera_packets.get(camera_name)
        return None if packet is None else packet[0]
    raise ValueError(f"Unsupported camera name {camera_name!r}")


def get_image_dict(camera_names: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for camera_name in camera_names:
        frame = _camera_global(camera_name)
        if frame is None or len(frame) == 0:
            raise RuntimeError(f"Camera {camera_name} has not produced a frame yet")
        images[camera_name] = frame.copy()
    return images


def wait_for_cameras(camera_names: list[str], timeout_sec: float | None = 15.0) -> None:
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    last_status = 0.0
    while deadline is None or time.monotonic() < deadline:
        try:
            get_image_dict(camera_names)
            return
        except RuntimeError:
            now = time.monotonic()
            if deadline is None and now - last_status >= 5.0:
                missing = [name for name in camera_names if _camera_global(name) is None]
                print(f"[Camera] Waiting for camera frames: {missing}")
                last_status = now
            time.sleep(0.02)
    missing = [name for name in camera_names if _camera_global(name) is None]
    raise TimeoutError(f"Timed out waiting for camera frames. Missing: {missing}")


class KeyWatcher:
    """Read terminal/pedal keys without blocking robot hold updates."""

    def __init__(self) -> None:
        self.keys: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="episode-key-watcher", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                key = getkey()
            except Exception:
                time.sleep(0.1)
                continue
            if key:
                self.keys.put(str(key).lower())

    def poll(self) -> str | None:
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def wait(self) -> str:
        return self.keys.get()


_key_watcher: KeyWatcher | None = None
_key_watcher_lock = threading.Lock()


def get_key_watcher() -> KeyWatcher:
    global _key_watcher
    with _key_watcher_lock:
        if _key_watcher is None:
            _key_watcher = KeyWatcher()
        return _key_watcher


def read_key_nonblocking() -> str | None:
    return get_key_watcher().poll()


def read_key_blocking() -> str:
    return get_key_watcher().wait()


def read_line_from_keys(prompt: str) -> str | None:
    """Read one typed line through the key watcher (it owns the terminal); Esc cancels."""
    print(prompt, end="", flush=True)
    text = ""
    while True:
        key = read_key_blocking()
        if key in ("\n", "\r", "enter"):
            print()
            return text
        if key in ("\x1b", "escape", "esc"):
            print()
            return None
        if key in ("\x7f", "\x08", "backspace"):
            if text:
                text = text[:-1]
                print("\b \b", end="", flush=True)
        elif len(key) == 1 and key.isprintable():
            text += key
            print(key, end="", flush=True)


# Replay gain fractions for this session; Enter at the prompt keeps them.
_replay_gain_scales = [1.0, 1.0]


def prompt_replay_gain_scales() -> tuple[float, float] | None:
    """Ask for replay kp/kd fractions before the arm moves; None cancels the replay."""
    print("[Replay] Tracking gain fractions for this replay (Enter keeps, Esc cancels).")
    values = list(_replay_gain_scales)
    for index, name in enumerate(("kp", "kd")):
        while True:
            text = read_line_from_keys(f"  {name} fraction [{values[index]:g}]: ")
            if text is None:
                print("[Replay] Cancelled")
                return None
            try:
                candidate = list(values)
                if text.strip():
                    candidate[index] = float(text)
                validate_replay_gain_scales(*candidate)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            values = candidate
            break
    _replay_gain_scales[:] = values
    return values[0], values[1]


def home_key_pressed() -> bool:
    """The failure exit's 'h', read through the collector's own key watcher."""
    key = read_key_nonblocking()
    return key is not None and key.lower() == "h"


def resolve_repo_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return Path(__file__).resolve().parents[2] / raw


def load_start_qpos(path: str | os.PathLike[str] | None, single_arm: bool, side: str | None) -> np.ndarray | None:
    resolved = resolve_repo_path(path)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(f"Saved start position not found: {resolved}")
    with np.load(resolved, allow_pickle=True) as data:
        if single_arm:
            arm = side or "left"
            key = f"{arm}_qpos"
            if key in data:
                qpos = np.asarray(data[key], dtype=np.float64).reshape(-1)[:7]
            elif "qpos" in data and np.asarray(data["qpos"]).size == 7:
                qpos = np.asarray(data["qpos"], dtype=np.float64).reshape(-1)
            elif "qpos" in data and np.asarray(data["qpos"]).size >= 14:
                full = np.asarray(data["qpos"], dtype=np.float64).reshape(-1)
                qpos = full[7:14] if arm == "right" else full[:7]
            else:
                raise ValueError(f"{resolved} does not contain a 7D start qpos for {arm!r}")
        elif "left_qpos" in data and "right_qpos" in data:
            left = np.asarray(data["left_qpos"], dtype=np.float64).reshape(-1)[:7]
            right = np.asarray(data["right_qpos"], dtype=np.float64).reshape(-1)[:7]
            qpos = np.concatenate([left, right])
        elif "qpos" in data and np.asarray(data["qpos"]).size >= 14:
            qpos = np.asarray(data["qpos"], dtype=np.float64).reshape(-1)[:14]
        else:
            raise ValueError(f"{resolved} must contain left_qpos/right_qpos or a 14D qpos")
    expected = 7 if single_arm else 14
    if qpos.size != expected:
        raise ValueError(f"Expected {expected} start qpos values, got {qpos.size}")
    return qpos.astype(np.float64, copy=False)


def ensure_arx_sdk_on_path() -> None:
    sdk_path = Path(__file__).resolve().parents[1] / "hardware" / "arx5-sdk" / "python"
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))


def set_arx_passive_gain(controller: Any) -> None:
    gain = controller.get_gain()
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)


def enable_arx_passive_compensation(controller: Any) -> None:
    """Leave only gravity and URDF friction feedforward active for manual motion."""
    set_arx_passive_gain(controller)
    controller.set_gravity_compensation(True)
    controller.set_friction_compensation(True)


def set_arx_default_gain(controller: Any, controller_config: Any, label: str) -> None:
    gain = controller.get_gain()
    default_kp = np.asarray(controller_config.default_kp, dtype=np.float64).reshape(-1)
    default_kd = np.asarray(controller_config.default_kd, dtype=np.float64).reshape(-1)
    kp_count = min(len(default_kp), len(gain.kp()))
    kd_count = min(len(default_kd), len(gain.kd()))
    gain.kp()[:kp_count] = default_kp[:kp_count]
    gain.kd()[:kd_count] = default_kd[:kd_count]
    gain.gripper_kp = float(controller_config.default_gripper_kp)
    gain.gripper_kd = float(controller_config.default_gripper_kd)
    controller.set_gain(gain)
    print(f"[ARX] Restored default gains for {label}")


def arx_qpos(controller: Any) -> np.ndarray:
    state = controller.get_joint_state()
    return np.concatenate([state.pos().copy(), [float(state.gripper_pos)]]).astype(np.float32, copy=False)


def arx_qvel(controller: Any) -> np.ndarray:
    state = controller.get_joint_state()
    return np.concatenate([state.vel().copy(), [float(state.gripper_vel)]]).astype(
        np.float32,
        copy=False,
    )


def make_arx_command(arx5: Any, robot_config: Any, qpos: np.ndarray) -> Any:
    command = arx5.JointState(robot_config.joint_dof)
    command.pos()[:] = np.asarray(qpos[:6], dtype=np.float64)
    command.vel()[:] = 0.0
    command.torque()[:] = 0.0
    command.gripper_pos = float(qpos[6])
    command.gripper_vel = 0.0
    command.gripper_torque = 0.0
    return command


MAX_REPLAY_GAIN_SCALE = 2.0


def validate_replay_gain_scales(kp_scale, kd_scale):
    """kp keeps some stiffness (0 would release the arm); both are capped."""
    kp_scale, kd_scale = float(kp_scale), float(kd_scale)
    if not 0 < kp_scale <= MAX_REPLAY_GAIN_SCALE:
        raise ValueError(f'kp fraction must be in (0, {MAX_REPLAY_GAIN_SCALE:g}]')
    if not 0 <= kd_scale <= MAX_REPLAY_GAIN_SCALE:
        raise ValueError(f'kd fraction must be in [0, {MAX_REPLAY_GAIN_SCALE:g}]')
    return kp_scale, kd_scale


def scaled_gain_snapshot(snapshot, kp_scale, kd_scale):
    """Tracking gains with kp and kd multiplied; the snapshot itself is unchanged."""
    return {'kp': np.asarray(snapshot['kp'], dtype=float) * kp_scale,
            'kd': np.asarray(snapshot['kd'], dtype=float) * kd_scale,
            'gripper_kp': float(snapshot['gripper_kp']) * kp_scale,
            'gripper_kd': float(snapshot['gripper_kd']) * kd_scale}


def arx_tracking_gain_snapshot(controller_config: Any) -> dict[str, Any]:
    """The configured collection gains (not the damping-only startup state)."""
    return {'kp': np.asarray(controller_config.default_kp, dtype=np.float64).reshape(-1)[:6].copy(),
            'kd': np.asarray(controller_config.default_kd, dtype=np.float64).reshape(-1)[:6].copy(),
            'gripper_kp': float(controller_config.default_gripper_kp),
            'gripper_kd': float(controller_config.default_gripper_kd)}


def apply_arx_gain_snapshot(controller: Any, snapshot: dict[str, Any]) -> None:
    gain = controller.get_gain()
    gain.kp()[:] = snapshot['kp']
    gain.kd()[:] = snapshot['kd']
    gain.gripper_kp = snapshot['gripper_kp']
    gain.gripper_kd = snapshot['gripper_kd']
    controller.set_gain(gain)


def arx_measured_pose(controller: Any) -> np.ndarray:
    """Six joints then the gripper from one SDK state, at full precision."""
    state = controller.get_joint_state()
    return np.r_[np.asarray(state.pos(), dtype=np.float64), float(state.gripper_pos)]


def move_arx_smoothly(arx5: Any, robot_config: Any, controllers: dict[str, Any], targets: dict[str, np.ndarray],
                      gains: dict[str, dict[str, Any]], *, duration: float, smooth: bool = False,
                      command_hz: float = ARX_COMMAND_HZ) -> None:
    """Move every selected arm and gripper to its 7-D target over a fixed duration.

    The measured pose is seeded before tracking gains are enabled (the SDK
    rejects active gains on a far command). A stalled loop slows the move and
    never sends a catch-up burst. ``smooth`` uses zero endpoint velocities.
    Friction feedforward is switched off: it supports hand guiding only.
    """
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError('Move duration must be positive and finite')
    current = {role: arx_measured_pose(controller) for role, controller in controllers.items()}
    for role, controller in controllers.items():
        controller.set_joint_cmd(make_arx_command(arx5, robot_config, current[role]))
    for role, controller in controllers.items():
        controller.set_friction_compensation(False)  # hand-guiding support only
        apply_arx_gain_snapshot(controller, gains[role])
    steps = max(1, int(round(duration * command_hz)))
    dt = duration / steps
    started = time.monotonic()
    for step in range(1, steps + 1):
        alpha = step / steps
        if smooth:
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        for role, controller in controllers.items():
            qpos = current[role] + (np.asarray(targets[role], dtype=np.float64) - current[role]) * alpha
            send_joint_target(controller, make_arx_command(arx5, robot_config, qpos), dt)
        remaining = started + step * dt - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            started -= remaining  # Pause the plan after a stall; never burst.


class ARXBackend:
    """Hand-guided ARX L5 recording, started and stopped by the pedal/keyboard.

    Modes: ``motion`` holds the arms and grippers with the collection tracking
    gains and friction feedforward off (also for moves and replays); ``recording`` and ``guided`` are passive (kp=kd=0 with the SDK's
    gravity and friction feedforward) for hand guiding; ``safe_stop`` is the
    SDK's native damping. Coordinates are the SDK's joint radians and gripper
    width. The SDK clips commanded positions to its own limits.
    """
    keyboard_start = True

    def __init__(self, ctx: RunContext) -> None:
        if ctx.robot_contract is not None and any(arm['model'] != 'L5' for arm in ctx.robot_contract['arms'].values()):
            raise ValueError('This recorder uses L5; select the shared recorder for other models')
        if not np.isfinite(ctx.fps) or not 0 < ctx.fps <= ARX_CONTROLLER_HZ:
            raise ValueError(f'ARX recording rate must be in (0, {ARX_CONTROLLER_HZ:g}] Hz')
        ensure_arx_sdk_on_path()
        import arx5_interface as arx5

        self.ctx = ctx
        self.arx5 = arx5
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config
        )
        configure_collection_gains(self.controller_config)
        self.controller_config.gravity_compensation = True
        self.controller_config.background_send_recv = True
        configure_controller(self.controller_config)
        if ctx.double_arm:
            interfaces = {"left": ctx.can_interface_left or "can0", "right": ctx.can_interface_right or "can1"}
        else:
            # In single-arm mode `can_interface_left` is the active arm's interface.
            interfaces = {ctx.arx_single_arm_side or "left": ctx.can_interface_left or "can0"}
        self.start_position_applied = False
        self.before_home = None  # Collector hook, e.g. stop cameras before a home move.
        self._closed = False
        self.mode: str | None = None
        self._by_role: dict[str, Any] = {}
        for role, interface in interfaces.items():
            print(f"[ARX] Initializing {role} arm on {interface}")
            self._by_role[role] = arx5.Arx5JointController(self.robot_config, self.controller_config, interface)
        self.controllers = list(self._by_role.values())
        self.left = self.controllers[0]
        self.right = self.controllers[1] if len(self.controllers) > 1 else None
        self._gains = {role: arx_tracking_gain_snapshot(self.controller_config) for role in self._by_role}
        print("[ARX] Recording in native SDK joint radians and gripper width")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError('ARX recorder is closed')

    def _split(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        return {role: np.asarray(qpos[index * 7:(index + 1) * 7], dtype=np.float64).copy()
                for index, role in enumerate(self._by_role)}

    def _require_pose(self, qpos: Any, label: str) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (7 * len(self._by_role),) or not np.isfinite(qpos).all():
            raise ValueError(f'{label} must contain seven finite positions per selected arm')
        return qpos

    def _load_start_qpos(self) -> np.ndarray:
        # Re-read setup's file for every take, including its seventh (gripper)
        # coordinate. A new setup save must not leave an old gripper cached.
        qpos = load_start_qpos(self.ctx.start_position_path, self.ctx.single_arm,
                               next(iter(self._by_role)) if self.ctx.single_arm else None)
        return self._require_pose(qpos, 'ARX start pose')

    def _move_to(self, targets: dict[str, np.ndarray], *, duration: float | None = None,
                 smooth: bool = False) -> None:
        """The saved-pose move with the tracking gains; ends holding in motion mode."""
        move_arx_smoothly(self.arx5, self.robot_config, self._by_role, targets, self._gains,
                          duration=self.ctx.start_position_move_duration if duration is None else duration,
                          smooth=smooth)
        self.mode = 'motion'

    def setup_for_recording(self) -> None:
        """Move to the saved start pose (gripper included) and hold it until the take starts."""
        self._require_open()
        self.start_position_applied = False
        if not self.ctx.use_start_position:
            self.enter_motion_mode()
        else:
            # Errors reach the collector's failure exit (damping until 'h').
            target = self._load_start_qpos()
            targets = self._split(target)
            print(f"[ARX] Moving to saved start pose: {resolve_repo_path(self.ctx.start_position_path)} "
                  f"(grippers={np.array2string(target[6::7], precision=4)})")
            self._move_to(targets)
            self.start_position_applied = True
        print("[ARX] Motion mode: holding arms and grippers until recording starts")

    def move_to_pose(self, qpos: Any) -> None:
        """Move to an explicit native pose with the saved-pose mover, then hold it.

        Errors propagate without shutting down so a caller (episode replay) can
        enter safe stop first.
        """
        self._require_open()
        self._move_to(self._split(self._require_pose(qpos, 'ARX pose')))

    def replay_rows(self, actions: Any, *, fps: float, should_stop: Callable[[], bool],
                    kp_scale: float = 1.0, kd_scale: float = 1.0, clock=time.monotonic,
                    sleep=time.sleep, log: Any = None, on_tick: Callable[[int, np.ndarray], None] | None = None) -> bool:
        """Replay native rows exactly like policy execution; return False if stopped.

        One timestamped target per arm and tick at ``fps``; the SDK interpolates
        natively over that interval with tracking gains and gravity. Rows are
        never skipped, re-timed or sent in a catch-up burst.

        With kp_scale/kd_scale (the operator's fractions) the tracking gains are
        scaled for these rows only. The stored gains stay unscaled, and every exit
        reapplies them: completion holds the measured pose (enter_motion_mode),
        stops and failures damp first, then the saved-pose mover.

        ``log`` (a utils.run_log.RunLog) records each sent row with the measured
        positions read right after it, for tracking-delay diagnosis only.

        ``on_tick(index, row)`` (a replay recording) runs at the start of each
        tick, before that row is sent, inside the same tick budget.
        """
        self._require_open()
        if self.mode != 'motion':
            raise RuntimeError('Replay starts from motion mode at the recorded start pose')
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 * len(self._by_role) or not np.isfinite(actions).all():
            raise ValueError('Replay rows must contain seven finite positions per selected arm')
        dt = replay_timestep(fps)
        kp_scale, kd_scale = validate_replay_gain_scales(kp_scale, kd_scale)
        if (kp_scale, kd_scale) != (1.0, 1.0):
            print(f'[ARX] Replay gains: kp x{kp_scale:g}, kd x{kd_scale:g}')
            for role, controller in self._by_role.items():
                apply_arx_gain_snapshot(controller, scaled_gain_snapshot(self._gains[role], kp_scale, kd_scale))
        for index, row in enumerate(actions):
            if should_stop():
                return False
            started = clock()
            if on_tick is not None:
                on_tick(index, row)
            for role, target in self._split(row).items():
                send_joint_target(self._by_role[role], make_arx_command(self.arx5, self.robot_config, target), dt)
            if log is not None:
                log.step(row=index, target=row.copy(),
                         qpos=np.concatenate([arx_qpos(controller) for controller in self.controllers]))
            # Same tick timing as shared policy execution: no catch-up after a stall.
            remaining = dt - (clock() - started)
            if remaining > 0:
                sleep(remaining)
        return True

    def enter_safe_stop(self) -> None:
        """Operator stop during replay: the same damping as the failure exit."""
        self._require_open()
        self.damping_target().damp()
        self.mode = 'safe_stop'
        print('[ARX] Safe stop: native damping (kp=0), gravity support; grippers held')

    def enter_motion_mode(self) -> None:
        """Hold the measured arm AND gripper pose before restoring tracking gains."""
        self._require_open()
        # Errors reach the collector's failure exit (damping until 'h').
        current = self._require_pose(np.concatenate([arx_measured_pose(controller) for controller in self.controllers]),
                                     'Measured ARX pose')
        for role, qpos in self._split(current).items():
            self._by_role[role].set_joint_cmd(make_arx_command(self.arx5, self.robot_config, qpos))
        for role, controller in self._by_role.items():
            controller.set_friction_compensation(False)  # hand-guiding support only
            apply_arx_gain_snapshot(controller, self._gains[role])
        self.mode = 'motion'

    def enable_passive_compensation(self) -> None:
        """Enable manual motion without commanding a pose or retaining damping gains."""
        for controller in self.controllers:
            enable_arx_passive_compensation(controller)
        self.mode = 'passive'
        print("[ARX] Passive mode active: kp=0, kd=0, gravity=on, friction=on")

    def enter_recording_mode(self) -> None:
        """Passive arms and grippers for hand guiding: only gravity and friction feedforward."""
        self._require_open()
        self.enable_passive_compensation()
        self.mode = 'recording'

    def enter_guided_mode(self) -> None:
        """--guided-start: the recording-mode passive support, before any take starts.

        Replaces the saved-pose move, so the take starts wherever the operator
        freezes the arm. Nothing is captured in this mode.
        """
        self._require_open()
        self.start_position_applied = False
        self.enable_passive_compensation()
        self.mode = 'guided'
        print('[ARX] Guided start: passive arms and grippers; not recording')

    def freeze_guided_pose(self) -> None:
        """Hold the guided arm and gripper pose (the same hold as after a take)."""
        self.enter_motion_mode()
        print('[ARX] Motion mode: holding the guided arm and gripper pose until recording starts')

    def start_recording(self) -> None:
        self.enter_recording_mode()

    def abort_recording(self) -> None:
        self.hold_after_recording()

    def read_qpos(self) -> np.ndarray:
        arrays = [arx_qpos(controller) for controller in self.controllers]
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions, velocities = [], []
        for controller in self.controllers:
            state = controller.get_joint_state()
            # Copy each state before the next controller call; SDK objects may
            # reference a live receive buffer.
            positions.append(np.r_[state.pos().copy(), float(state.gripper_pos)])
            velocities.append(np.r_[state.vel().copy(), float(state.gripper_vel)])
        return (np.concatenate(positions).astype(np.float32),
                np.concatenate(velocities).astype(np.float32))

    def read_qvel(self) -> np.ndarray:
        arrays = [arx_qvel(controller) for controller in self.controllers]
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def passive_step(self) -> None:
        return None  # The SDK's background loop keeps holds and gravity support fresh.

    def hold_after_recording(self) -> None:
        self.enter_motion_mode()
        print('[ARX] Motion mode: holding the measured arm and gripper pose for review')

    def damping_target(self):
        """safe_exit target: native ARX damping until 'h', then reset_to_home."""
        from hardware.safe_exit import ARXDamping
        return ARXDamping(self.controllers, before_home=getattr(self, "before_home", None))

    def shutdown(self, *, home: bool = True) -> None:
        """Normal exit homes; after the failure exit the arm is already home or disabled."""
        if self._closed:
            return
        self._closed = True
        if not home:
            return  # The failure exit already homed (or disabled in place).
        print("[ARX] Returning arm(s) home")
        for label, controller in zip(("left", "right"), self.controllers):
            try:
                set_arx_default_gain(controller, self.controller_config, label)
                controller.reset_to_home()
            except Exception as exc:
                print(f"[ARX] Warning during shutdown for {label}: {exc}")


def build_backend(ctx: RunContext) -> Any:
    if ctx.robot_type == "arx":
        return ARXBackend(ctx)
    raise ValueError(f"Unsupported robot type: {ctx.robot_type}")


def guide_to_start_pose(backend: Any, dt: float) -> None:
    """--guided-start: gravity support until GO_KEY freezes the arm where it is."""
    backend.enter_guided_mode()
    print(f"\nGuided start: move the arm by hand, then press '{GO_KEY}' to freeze it there, "
          "or 'x' to quit.")
    while True:
        backend.passive_step()
        key = read_key_nonblocking()
        if key in ('x', GO_KEY):
            # Quitting also holds first, so the normal home move starts from motion mode.
            backend.freeze_guided_pose()
            if key == 'x':
                raise QuitRequested
            return
        time.sleep(min(max(dt, 0.001), 0.02))


def prompt_for_episode_start(args: argparse.Namespace, events: EpisodeEvents, backend: Any, dt: float) -> bool:
    """Wait for GO_KEY (pedal) while the arm holds; 'x' quits through the normal home move."""
    guided = bool(getattr(args, 'guided_start', False))
    if guided:
        guide_to_start_pose(backend, dt)
        prompt = (f"\nPose frozen: press '{GO_KEY}' to record from here with gravity compensation, "
                  f"'{REJECT_KEYS[0]}' to release and guide again, or 'x' to quit.")
    else:
        prompt = f"\nMotion mode: press '{GO_KEY}' for gravity-compensated recording, or 'x' to quit."
    if args.partial_subgoal_mode:
        prompt += "\nPartial-subgoal mode: '2' marks a partial-subgoal start and begins recording."
    print(prompt)
    while True:
        backend.passive_step()
        key = read_key_nonblocking()
        if key == 'x':
            raise QuitRequested
        if guided and key in REJECT_KEYS:
            guide_to_start_pose(backend, dt)
            print(prompt)
            continue
        if key == GO_KEY or (args.partial_subgoal_mode and key == '2'):
            if key == '2':
                events.partial_subgoal_start = True
                events.partial_subgoal_start_timestamp = 0.0
                print("[Event] Partial-subgoal start selected")
            return True
        time.sleep(min(max(dt, 0.001), 0.02))


def prompt_next_episode(ctx: RunContext, backend: Any, last_saved: str | None,
                        record_replay: Callable[[], bool] | None = None) -> bool:
    """Enter prepares another episode; r replays the last saved one; x quits.

    With ``record_replay`` (--record-replay, after a save) '1' first records a
    replay of the saved take; it returns True on quit. Afterwards '1' prepares
    another episode, like any key other than 'r' and 'x'.
    """
    can_replay = hasattr(backend, "replay_rows")
    while True:
        if record_replay is not None:
            print(f"\nPress '{GO_KEY}' to replay {Path(last_saved).name} while recording it, "
                  "Enter to prepare another episode, 'r' to replay without recording, or 'x' to quit.")
        else:
            print("\nPress Enter to prepare another episode, "
                  + ("'r' to replay the last saved episode, " if can_replay else "") + "or 'x' to quit.")
        key = read_key_blocking()
        if key == "x":
            return False
        if key == GO_KEY and record_replay is not None:
            pending, record_replay = record_replay, None
            if pending():
                return False
            continue
        if key.lower() != "r" or not can_replay:
            return True
        from recording.recording_replay import latest_saved_episode, load_replay_episode, replay_take
        path = last_saved or latest_saved_episode(ctx.dataset_dir)
        if path is None:
            print("[Replay] No saved episode in this dataset yet")
            continue
        gains = prompt_replay_gain_scales()
        if gains is None:
            continue
        wait_for_cameras(ctx.camera_names, timeout_sec=15.0)
        try:
            actions, camera, reference = load_replay_episode(
                path, camera_names=ctx.camera_names, robot_contract=ctx.robot_contract, fps=ctx.fps)
        except Exception as exc:  # Nothing has moved.
            print(f"[Replay] {path} cannot be replayed: {exc}")
            continue
        result = replay_take(
            backend, actions, camera, reference, label=str(path), gain_scales=gains,
            poll_keys=lambda: [key for key in (read_key_nonblocking(),) if key],
            live_frame=lambda name: _camera_global(name))
        if result == "quit":
            return False


def replay_unsaved_take(backend, samples) -> str:
    """Review: replay the stopped take from its raw capture before save/discard."""
    from recording.recording_replay import replay_take, take_replay_source
    camera_names = backend.ctx.camera_names
    gains = prompt_replay_gain_scales()
    if gains is None:
        return 'cancelled'
    wait_for_cameras(camera_names, timeout_sec=15.0)
    try:
        actions, camera, reference = take_replay_source(samples, camera_names)
    except Exception as exc:  # Nothing has moved; keep the take for review.
        print(f"[Replay] Take cannot be replayed: {exc}")
        return 'unavailable'
    return replay_take(
        backend, actions, camera, reference, label='unsaved take', gain_scales=gains,
        poll_keys=lambda: [key for key in (read_key_nonblocking(),) if key],
        live_frame=lambda name: _camera_global(name))


class ReplayRecording:
    """Standard capture of a replay: one observation per replayed row.

    Each tick captures before its row is sent, so a sample pairs the observation
    with the target commanded right after it, as in policy execution.
    """

    def __init__(self, ctx: RunContext, backend: Any, args: argparse.Namespace) -> None:
        self.ctx, self.backend, self.args = ctx, backend, args
        self.samples: EpisodeBuffer | None = None

    def start(self) -> None:
        self.samples = EpisodeBuffer(camera_names=self.ctx.camera_names, fps=self.ctx.fps,
                                     jpeg_quality=int(self.args.jpeg_quality),
                                     directory=self.ctx.dataset_dir, is_double_arm=self.ctx.double_arm)
        self.timing = CaptureTiming(self.ctx.fps)
        self.started = time.monotonic()
        self.previous: float | None = None

    def sample(self, index: int, row: np.ndarray) -> None:
        observation = standard_capture.capture_observation(
            lambda: read_robot_state(self.backend), self.ctx.camera_names)
        sample = EpisodeSample.from_capture(observation, observation.robot_timestamp - self.started)
        if self.previous is not None:
            sample.capture_interval = sample.timestamp - self.previous
            self.timing.observe(sample.capture_interval)
        self.previous = sample.timestamp
        sample.action = np.array(row, dtype=np.float32)
        self.samples.append(sample)

    def stop(self) -> None:
        self.samples.finish()


@recording_scope
def record_replay_episode(ctx: RunContext, backend: Any, source: str, args: argparse.Namespace) -> bool:
    """--record-replay: replay a saved take and save that replay as a new episode.

    The review replay (recorded start pose, overlay, '1' starts, '1' stops) runs
    at this session's gain fractions without prompting. The new episode's
    actions are the replayed rows; its metadata names the source episode.
    Returns True when the operator quits.
    """
    from recording.recording_replay import load_replay_episode, replay_take
    wait_for_cameras(ctx.camera_names, timeout_sec=15.0)
    try:
        actions, camera, reference = load_replay_episode(
            source, camera_names=ctx.camera_names, robot_contract=ctx.robot_contract, fps=ctx.fps)
    except Exception as exc:  # Nothing has moved.
        print(f"[Replay] {source} cannot be replayed: {exc}")
        return False
    kp_scale, kd_scale = _replay_gain_scales
    print(f"[Replay] Recording a replay of {Path(source).name} at kp x{kp_scale:g}, kd x{kd_scale:g} "
          "('r' at the next-episode prompt changes these)")
    recording = ReplayRecording(ctx, backend, args)
    result = replay_take(
        backend, actions, camera, reference, label=str(source), gain_scales=(kp_scale, kd_scale),
        poll_keys=lambda: [key for key in (read_key_nonblocking(),) if key],
        live_frame=lambda name: _camera_global(name), record=recording)
    if result == "quit":
        return True
    if result != "completed":
        if recording.samples is not None:
            print("[Replay] Replay stopped; its recording is not saved")
        return False
    samples = recording.samples
    if not prompt_accept_episode(backend=backend, samples=samples):
        print("[Discarded] Replay recording not saved")
        return False
    if len(samples) < 2:
        print("[WARN] Replay recording has fewer than two samples; discarding")
        return False
    dataset_path = os.path.join(ctx.dataset_dir, f"episode_{get_auto_index(ctx.dataset_dir):05d}.zarr")
    events = EpisodeEvents()
    extra_metadata = {
        "error_recovery": False,
        "start_position_applied": False,  # Replays start from the source's first row.
        "image_compress": bool(args.compress_images),
        "image_jpeg_quality": int(args.jpeg_quality) if args.compress_images else -1,
        "replay_recording": True,
        "replay_source_episode": Path(source).name,
        "replay_kp_scale": float(kp_scale),
        "replay_kd_scale": float(kd_scale),
    }
    write_episode(
        dataset_path,
        samples,
        ctx.camera_names,
        qpos_dim=ctx.qpos_dim,
        qvel_dim=ctx.qvel_dim,
        metadata=build_episode_metadata(ctx, args, events, extra_metadata),
        events=events,
        compress=bool(args.compress_images),
        jpeg_quality=int(args.jpeg_quality),
    )
    print(f"[Saved] {dataset_path} (replay of {Path(source).name})")
    return False


def prompt_accept_episode(*, backend=None, samples=None) -> bool:
    can_replay = hasattr(backend, 'replay_rows') and samples is not None
    reject = '/'.join(REJECT_KEYS)
    print(f"\nPress '{GO_KEY}' to save/accept, '{reject}' to reject, "
          + ("'r' to replay before deciding, " if can_replay else '') + "or 'q' to quit.")
    while True:
        key = read_key_blocking()
        if key == GO_KEY:
            return True
        if key in REJECT_KEYS:
            return False
        if key in {"q", "x", "\x1b"}:
            raise QuitRequested
        if key.lower() == "r" and can_replay:
            if replay_unsaved_take(backend, samples) == 'quit':
                raise QuitRequested
            print(f"Review (take not saved yet): {GO_KEY} saves, {reject} rejects, r replays again, q quits.")


@recording_scope
def capture_one_episode(
    ctx: RunContext,
    backend: Any,
    dataset_name: str,
    overwrite: bool,
    args: argparse.Namespace,
    episode_metadata: dict[str, Any] | None = None,
) -> bool:
    os.makedirs(ctx.dataset_dir, exist_ok=True)
    dataset_path = os.path.join(ctx.dataset_dir, dataset_name + ".zarr")
    if os.path.exists(dataset_path) and not overwrite:
        print(f"Dataset already exists at {dataset_path}; pass overwrite=True to replace it")
        return False

    wait_for_cameras(ctx.camera_names, timeout_sec=None if args.use_camera_server else 15.0)
    events = EpisodeEvents()
    dt = 1.0 / ctx.fps
    if not prompt_for_episode_start(args, events, backend, dt):
        return False

    samples = EpisodeBuffer(camera_names=ctx.camera_names, fps=ctx.fps,
                               jpeg_quality=int(args.jpeg_quality), directory=ctx.dataset_dir,
                               is_double_arm=ctx.double_arm)
    start_recording = getattr(backend, 'start_recording', None)
    if start_recording is not None:
        start_recording()
    actual_dt_history: list[tuple[float, float]] = []
    jitter = ProprioJitterMonitor(
        args.check_proprio_jitter,
        args.proprio_jitter_window_sec,
        args.proprio_jitter_print_interval,
    )
    capture = EpisodeCapture(backend, ctx.camera_names, fps=ctx.fps)
    recording_start = capture.started
    print("\n**************\nStart recording...\n**************\n")

    for _ in tqdm(range(ctx.max_timesteps)):
        sample = capture.wait_and_sample()
        timestamp = sample.timestamp
        samples.append(sample)
        jitter.update(timestamp, sample.qpos)
        sample_done = time.monotonic()
        actual_dt_history.append((capture.last_sample_started, sample_done))

        key = read_key_nonblocking()
        if key == FAILURE_STAMP_KEY:
            events.failure_event_timestamps.append(timestamp)
            print(f"[Event] Failure stamp at {timestamp:.3f}s")
        elif key == "2":
            events.subgoal_switch_event_timestamps.append(timestamp)
            print(f"[Event] Subgoal switch at {timestamp:.3f}s")
        elif key == GO_KEY:
            events.recording_stop_event_timestamp = timestamp
            print(f"[Event] Recording stop at {timestamp:.3f}s")
            break
        elif key == "x":
            print("[Event] Recording aborted")
            abort_recording = getattr(backend, 'abort_recording', None)
            if abort_recording is not None:
                abort_recording()
            return False

    elapsed = time.monotonic() - recording_start
    if elapsed > 0:
        print(f"Avg fps: {(len(samples) - 1) / elapsed:.2f}")
    print_dt_diagnosis(actual_dt_history)
    freq_mean = len(capture.intervals) / sum(capture.intervals) if capture.intervals else 0.
    if freq_mean and freq_mean < ctx.fps * 0.96:
        print(f"[WARN] Sample frequency {freq_mean:.2f} Hz is below target {ctx.fps:.2f} Hz")

    backend.hold_after_recording()
    samples.finish()
    if not prompt_accept_episode(backend=backend, samples=samples):
        return False
    if len(samples) < 2:
        print("[WARN] Episode has fewer than two samples; discarding")
        return False

    extra_metadata = dict(episode_metadata or {})
    extra_metadata["start_position_applied"] = bool(getattr(backend, "start_position_applied", False))
    extra_metadata["guided_start"] = bool(getattr(args, "guided_start", False))
    extra_metadata["image_compress"] = bool(args.compress_images)
    extra_metadata["image_jpeg_quality"] = int(args.jpeg_quality) if args.compress_images else -1
    write_episode(
        dataset_path,
        samples,
        ctx.camera_names,
        qpos_dim=ctx.qpos_dim,
        qvel_dim=ctx.qvel_dim,
        metadata=build_episode_metadata(ctx, args, events, extra_metadata),
        events=events,
        compress=bool(args.compress_images),
        jpeg_quality=int(args.jpeg_quality),
    )
    print(f"[Saved] {dataset_path}")
    return True


def build_episode_metadata(
    ctx: RunContext,
    args: argparse.Namespace,
    events: EpisodeEvents,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "task_name": ctx.task_name,
        "robot_type": ctx.robot_type,
        "robot_contract": ctx.robot_contract or {'robot_contract_version': 1, 'arms': {
            role: {'model': 'L5'} for role in ([ctx.arx_single_arm_side or 'left'] if ctx.single_arm else ['left', 'right'])}},
        "single_arm": ctx.single_arm,
        "double_arm": ctx.double_arm,
        "camera_names": ctx.camera_names,
        "fps": ctx.fps,
        "nominal_recording_hz": ctx.fps,
        "action_rate_hz": ctx.fps,
        "sdk_controller_hz": ARX_CONTROLLER_HZ,
        "can_interface_left": ctx.can_interface_left or "",
        "can_interface_right": ctx.can_interface_right or "",
        "arx_single_arm_side": ctx.arx_single_arm_side or "",
        "use_camera_server": bool(args.use_camera_server),
        "start_position_path": ctx.start_position_path or "",
        "start_position_enabled": bool(ctx.use_start_position),
        "start_position_move_duration": float(ctx.start_position_move_duration),
        "partial_subgoal_mode": bool(args.partial_subgoal_mode),
        "partial_subgoal_start": bool(events.partial_subgoal_start),
        "partial_subgoal_start_timestamp": -1.0
        if events.partial_subgoal_start_timestamp is None
        else float(events.partial_subgoal_start_timestamp),
        "failure_event_count": len(events.failure_event_timestamps),
        "subgoal_switch_event_count": len(events.subgoal_switch_event_timestamps),
        "recording_stop_event_present": events.recording_stop_event_timestamp is not None,
        "error_recovery": bool(extra.get("error_recovery", False)),
        "arm_recording_key": GO_KEY,
        "start_gripper": "keyboard",
        "stop_accept_key": GO_KEY,
        "failure_key": FAILURE_STAMP_KEY,
        "subgoal_switch_key": "2",
        "reject_key": "/".join(REJECT_KEYS),
    }
    metadata.update(extra)
    return metadata


def write_episode(dataset_path, samples, camera_names, qpos_dim, qvel_dim,
                  metadata=None, events=None, **options):
    """Publish recorded sample references and labels without decoding video."""
    from data_processing.episode_store import publish_samples
    if not samples:
        raise ValueError('Need at least one sample to save an episode')
    return publish_samples(dataset_path, samples, metadata=metadata, events=events)


def main(raw_args: argparse.Namespace | dict[str, Any]) -> None:
    args = argparse.Namespace(**raw_args) if isinstance(raw_args, dict) else raw_args
    ctx = resolve_run_context(args)
    from hardware.safe_exit import raising_signals
    # SIGTERM, like Ctrl+C, reaches the failure exit (damping until 'h').
    with raising_signals():
        return run_recording(args, ctx)


def run_recording(args, ctx):
    if not args.use_camera_server:
        raise ValueError('Independent raw recording requires the camera server and its V4L2 timestamps')
    record_replay = bool(getattr(args, 'record_replay', False))
    if record_replay and args.episode_idx is not None:
        raise ValueError('--record-replay requires auto-incremented episodes')
    guided_start = bool(getattr(args, 'guided_start', False))
    print(
        f"[Config] task={ctx.task_name} robot={ctx.robot_type} "
        f"mode={'single' if ctx.single_arm else 'double'} dataset_dir={ctx.dataset_dir}"
    )
    if args.masking_strategy is not None and not args.rerun:
        print("[WARN] masking_strategy is only supported with --rerun; ignoring")
        args.masking_strategy = None
    if args.rerun:
        print("[WARN] --rerun is accepted for CLI compatibility but direct recorder visualization is not enabled here")

    stop_event, camera_thread = start_camera_capture(ctx, args.use_camera_server)
    backend = None
    failed = False

    def stop_cameras():
        # Close the camera cache before homing: its native teardown holds the
        # GIL and would stall the home trajectory.
        stop_event.set()
        camera_thread.join(timeout=2.0)

    def damping_target():
        if backend is None:
            return None
        backend.before_home = stop_cameras
        return backend.damping_target()

    try:
        # Any failure, including Ctrl+C, damps until 'h', homes, then exits.
        with damping_exit(damping_target, home_requested=home_key_pressed):
            backend = build_backend(ctx)
            retained_episode_parameters: dict[str, Any] = {"error_recovery": False}
            last_saved: str | None = None
            while True:
                pending_replay = None
                episode_idx = args.episode_idx if args.episode_idx is not None else get_auto_index(ctx.dataset_dir)
                dataset_name = f"episode_{episode_idx:05d}"
                print(f"\nDataset name: {dataset_name}")
                if not guided_start:  # Guided starts skip the saved-pose move.
                    backend.setup_for_recording()
                saved = capture_one_episode(
                    ctx,
                    backend,
                    dataset_name=dataset_name,
                    overwrite=True,
                    args=args,
                    episode_metadata=retained_episode_parameters,
                )
                if saved:
                    print(f"[OK] Episode {episode_idx} saved")
                    last_saved = os.path.join(ctx.dataset_dir, dataset_name + ".zarr")
                    retained_episode_parameters = {"error_recovery": False}
                    if args.episode_idx is not None:
                        break
                    if record_replay:
                        pending_replay = lambda source=last_saved: record_replay_episode(ctx, backend, source, args)
                else:
                    print(f"[Discarded] Episode {episode_idx} not saved; retry will reuse current episode parameters")

                if not prompt_next_episode(ctx, backend, last_saved, record_replay=pending_replay):
                    break
    except QuitRequested:
        pass  # Intentional quit: the normal exit below homes, then disables.
    except BaseException:
        failed = True  # The failure exit already homed or disabled in place.
        raise
    finally:
        stop_event.set()
        # The arm keeps its current hold while the cameras close, and
        # shutdown still runs if the join fails.
        try:
            camera_thread.join(timeout=2.0)
        finally:
            if backend is not None:
                backend.shutdown(home=not failed)
        print("Shutdown complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default=None, help="Task name in the YAML catalog.")
    parser.add_argument("--task-config", dest="task_config", help="Explicit task YAML.")
    parser.add_argument("--robot", choices=["arx"], default=None, help="Robot backend (ARX).")
    parser.add_argument("--rerun", action="store_true", default=False, help="Accepted for compatibility.")
    parser.add_argument("--episode_idx", type=int, default=None, help="Episode index; auto-increments when omitted.")
    parser.add_argument("--single_arm", action="store_true", default=None, help="Deprecated compatibility override.")
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default=None,
        choices=[
            "moving_block",
            "end_effector",
            "prioproceptive",
            "prioproceptive_and_moving_block",
            "overhead_left_90px",
            "left_right_overhead",
            "none",
        ],
        help="Debug-only compatibility option; requires --rerun.",
    )
    parser.add_argument("--use-camera-server", action="store_true", dest="use_camera_server", default=True)
    parser.add_argument("--no-camera-server", action="store_false", dest="use_camera_server")
    parser.add_argument("--partial-subgoal-mode", action="store_true", default=False)
    parser.add_argument("--check-proprio-jitter", action="store_true", default=False)
    parser.add_argument("--proprio-jitter-print-interval", type=float, default=1.0)
    parser.add_argument("--proprio-jitter-window-sec", type=float, default=1.0)
    parser.add_argument("--start-position", dest="start_position_path", type=str, default=argparse.SUPPRESS,
                        help="Saved start pose (default: calibration/start_positions/<task>.npz if saved, "
                             "else calibration/start_position.npz)")
    parser.add_argument("--no-start-position", "--no_start_position", dest="start_position_path",
                        action="store_const", const=None, default=argparse.SUPPRESS)
    parser.add_argument("--start-position-move-duration", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=None, help="Must match the task YAML FPS.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for compressed Zarr images (0-100).")
    parser.add_argument("--compress-images", action="store_true", dest="compress_images", default=True)
    parser.add_argument("--no-compress-images", action="store_false", dest="compress_images")
    parser.add_argument("--record-replay", action="store_true", default=False,
                        help="After each saved take, '1' replays it while recording the replay as "
                             "another episode (metadata replay_source_episode names the take).")
    parser.add_argument("--guided-start", action="store_true", default=False,
                        help=f"Start each take wherever you place the arm. Instead of the saved-pose "
                             f"move, the arm is under gravity compensation; '{GO_KEY}' freezes it, "
                             f"'{GO_KEY}' again records from there ('{REJECT_KEYS[0]}' re-guides).")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
