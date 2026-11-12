"""Direct, in-process policy inference runner."""

# Support direct execution as well as package imports.
if __package__ in (None, ''):
    import sys as _path_sys
    from pathlib import Path as _Path
    _path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


from contracts.action_contract import require_absolute_actions
from utils.hardware_utils import (
    configure_controller,
    send_joint_target,
    ACTIVE_KP,
    ACTIVE_KD,
    configure_active_gains,
)
from contracts.action_rate_contract import resolve_policy_rate
from camera_utils.camera_settings_contract import check_inference_camera_settings
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import os
from pathlib import Path
import pickle
import argparse
from datetime import date
from queue import Empty, Queue
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import time as time_V2
from torchvision import transforms

from config.paths import DAGGER_DIR, ROLLOUT_DIR
from config.task_config import default_start_position_path, require_chunk_length
from config.constants import (
    EXECUTION_DT as DEFAULT_EXECUTION_DT,
    FPS as DEFAULT_FPS,
    TASK_CONFIGS as RUNTIME_TASK_CONFIGS,
    resolve_arx_single_arm_can,
    resolve_arx_single_arm_side,
)
from camera_utils.capture import init_camera, get_camera_info, capture_image, capture_image_from_shared_memory
from utils.random_utils import set_seed
from config.supported_policy import require_supported_policy
from utils.masking import Masker
from config.masking_config import require_masking_contract
from inference.inference_utils import fill_in
from inference import lookahead_tuning

import camera_utils.capture as camera_capture
from contracts import capture_gc, standard_capture
from contracts import standard_recording
from contracts import standard_image_processing
from hardware.safe_exit import ARXDamping, damping_exit
from recording.pedal_keys import GO_KEY as PEDAL_GO_KEY, REJECT_KEY as PEDAL_RESET_KEY

import cv2, socket, threading, multiprocessing
from getkey import getkey  # for keystroke capture without stdout echo
from policy.model_util import make_policy

import sys
from contextlib import contextmanager, nullcontext
try:
    import arx5_interface as arx5
except ImportError:  # Offline machines; ARX execution reports the missing SDK.
    arx5 = None

import subprocess

# Rerun visualization imports
import rerun as rr
from visualization.rerun_utils import load_dual_arm_calibration, log_robot_model, log_robot_model_with_transform, parse_urdf

####===================CONFIGURATION===================####
# Deployment and debugging configuration
DEPLOY = True #if subprocess.getoutput('whoami')=="unitree" else False
DEBUG = not DEPLOY

# Performance and optimization settings
SAVE_EPISODE = False  # Save episode videos for analysis
ACTION_INTERP = False  # Enable action interpolation for smoother robot motion
FACTOR = 100  # Episode length multiplier for real-world tasks
NN_CHUNK_SEARCH_RADIUS = 5
DAGGER_CORRECTION_KEY = '1'
DAGGER_RECORD_KEY = '2'
DAGGER_EXIT_KEY = '3'
DAGGER_STATE_POLICY = 'policy'
DAGGER_STATE_GRAVITY_COMP = 'gravity_comp'
DAGGER_STATE_RECORDING = 'recording'
DAGGER_JPEG_WORKERS = 8
dagger_state = DAGGER_STATE_POLICY
dagger_command_enabled = False
dagger_prompt_active = False
dagger_prompt_key_queue = Queue()
continue_flag = True
pause_flag = False
exit_requested = False
reset_ready_requested = False
# Set for a direct rollout without RTC: the operator may change the lookaheads while paused.
lookahead_tuner = None

# Configure PyTorch for optimal performance
torch.backends.cudnn.benchmark = True

EXECUTION_DT = DEFAULT_EXECUTION_DT
urdf_path = "src/hardware/arx5-sdk/models/L5.urdf"
DEFAULT_START_POSITION_PATH = "calibration/start_position.npz"
GRIPPER_CLOSED_POS = 0.0


class RuntimeArms:
    """Shared-hardware view of a HardwareRuntime: one native target row per tick.

    The runtime owns limits, the command watchdog and damping. Targets are only
    resubmitted inside idle_hold() while the main loop blocks on prompts or loading.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.last_target = None
        self.failure_handled = False
        self._lock = threading.Lock()

    def connect_and_hold(self):
        self.runtime.connect()
        self.submit(self.runtime.read_state()['qpos'])

    def read_runtime_state(self):
        return self.runtime.read_state()

    def submit(self, target):
        target = np.asarray(target, dtype=float).reshape(-1).copy()
        with self._lock:
            self.runtime.submit_target(target)
            self.last_target = target

    @contextmanager
    def holding(self):
        stop, errors = threading.Event(), []

        def run():
            while not stop.wait(self.runtime.interval_s):
                try:
                    self.submit(self.last_target)
                except BaseException as exc:  # The runtime has already started damping.
                    errors.append(exc)
                    return
        worker = threading.Thread(target=run, daemon=True, name='shared-idle-hold')
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
        if errors:
            raise errors[0]


def idle_hold(controllers):
    """Keep shared-runtime targets fresh while the main loop blocks; SDK controllers hold natively."""
    return controllers.holding() if isinstance(controllers, RuntimeArms) else nullcontext()


def damping_target(controllers):
    """safe_exit target for the connected arms: shared runtime damping or native ARX."""
    if isinstance(controllers, RuntimeArms):
        from hardware.runtime import RuntimeDamping
        return RuntimeDamping(controllers.runtime) if controllers.runtime.live_arms() else None
    return ARXDamping(controllers.values())


def save_videos(video, EXECUTION_DT, video_path=None):
    if isinstance(video, list):
        video = np.array(video)
        print(f"video shape: {video.shape}")
        h, w, _ = video[0].shape
        for i,img in enumerate(video):
            if i % 10 == 0:
                cv2.imwrite(os.path.join(video_path, f'img_{i}.png'), img)

        fps = 50
        out = cv2.VideoWriter(os.path.join(video_path, f'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v') , fps, (w, h))
        out.write(np.array(video))
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []

        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / EXECUTION_DT)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v') , fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            # image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')




















def count_rollout_label_episodes(root_dir, label):
    label_dir = os.path.join(root_dir, label)
    if not os.path.isdir(label_dir):
        return 0
    return sum(
        1
        for name in os.listdir(label_dir)
        if name.startswith("episode_") and name.endswith(".zarr")
    )


def print_success_rate_banner(rollout_root_dir):
    success_count = count_rollout_label_episodes(rollout_root_dir, "success")
    failure_count = count_rollout_label_episodes(rollout_root_dir, "failure")
    total_count = success_count + failure_count
    rate = 0.0 if total_count == 0 else (100.0 * success_count / total_count)
    print(
        f"[Rollout Recording] Success rate: {success_count}/{total_count} "
        f"recorded success/failure episodes ({rate:.1f}% success, "
        f"{failure_count} failure)"
    )


def rollout_date_string():
    return date.today().isoformat()


def resolve_rollout_recording_root(rollout_dir, task_name, eval_mode=False, date_string=None):
    save_parts = [rollout_dir]
    if eval_mode:
        save_parts.append(date_string or rollout_date_string())
    save_parts.append(task_name)
    return os.path.join(*save_parts)


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def get_stored_images(height, width, camera_names=None, double_arm=False):
    """Get resized images from robot cameras."""
    if camera_names is not None:
        result = {}
        for cam_name in camera_names:
            result[cam_name] = get_resized_image(cam_name, height, width)
        return result

    img_r = None
    img_l = get_resized_image('video_left', height, width)
    img_h = get_resized_image('video_overhead', height, width)
    if double_arm:
        img_r = get_resized_image('video_right', height, width)
    return img_l, img_h, img_r

def get_resized_image(camera_name, height=None, width=None):
    cam_to_global = {
        'video_left': camera_capture.captured_img_l,
        'video_overhead': camera_capture.captured_img_h,
        'video_right': camera_capture.captured_img_r,
        'video_torso': camera_capture.captured_img_t,
    }
    image = cam_to_global.get(camera_name)
    if image is None:
        raise ValueError(f"Unknown camera name: {camera_name}")
    img_resized = standard_image_processing.resize_image(image, width, height)
    return img_resized.copy()

def capture_fake_image():
    """Load fake images for debugging."""
    camera_capture.captured_img_l = cv2.imread('DEBUG/im1.png')
    camera_capture.captured_img_h = cv2.imread('DEBUG/im2.png')

def get_fake_robot_image(height=None, width=None):
    """Get resized fake images for debugging."""

    img_l = camera_capture.captured_img_l.copy()
    img_h = camera_capture.captured_img_h.copy()

    if height is not None and width is not None:
        img_l = standard_image_processing.resize_image(img_l, width, height)
        img_h = standard_image_processing.resize_image(img_h, width, height)

    return img_l, img_h






def resolve_base_camera_names(config, task_config):
    policy_config = config.get('policy_config', {})
    camera_names = policy_config.get('base_camera_names', task_config.get('camera_names', ['video_left', 'video_overhead']))
    return list(camera_names)


def resolve_model_camera_names(config, task_config):
    standard_image_processing.require_single_frame(config, task_config)
    return resolve_base_camera_names(config, task_config)


def resolve_single_arm_runtime_side_and_can(config, task_config):
    """Resolve ARX single-arm side from constants.py, falling back to checkpoint metadata."""
    if task_config.get('_task_config_path'):
        return (resolve_arx_single_arm_side(task_config),
                resolve_arx_single_arm_can(task_config), 'task YAML')
    task_name = config.get('task_name') or task_config.get('task_name')
    runtime_task_config = RUNTIME_TASK_CONFIGS.get(task_name, {}) if task_name is not None else {}
    policy_config = config.get('policy_config', {})

    side = None
    source = 'legacy/unspecified'
    if 'single_arm_side' in runtime_task_config:
        side = runtime_task_config.get('single_arm_side')
        source = 'constants.py'
    elif policy_config.get('single_arm_side') is not None:
        side = policy_config.get('single_arm_side')
        source = 'policy_config'
    elif config.get('single_arm_side') is not None:
        side = config.get('single_arm_side')
        source = 'config'
    elif task_config.get('single_arm_side') is not None:
        side = task_config.get('single_arm_side')
        source = 'task_config'

    if side is not None:
        side = resolve_arx_single_arm_side({'single_arm_side': side})

    can_task_config = dict(task_config)
    can_task_config.update(runtime_task_config)
    if side is not None:
        can_task_config['single_arm_side'] = side
    single_arm_can = resolve_arx_single_arm_can(can_task_config)
    return side, single_arm_can, source


def jpeg_compress_image_dict(image_dict, jpeg_quality=50):
    """Apply in-memory JPEG encode/decode to current camera frames."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    compressed_dict = {}
    for cam_name, image in image_dict.items():
        if image is None:
            compressed_dict[cam_name] = image
            continue
        ok, encoded_image = cv2.imencode('.jpg', image, encode_param)
        if not ok:
            compressed_dict[cam_name] = image
            continue
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        compressed_dict[cam_name] = decoded_image if decoded_image is not None else image
    return compressed_dict


def process_image(image_dict, normalize_resnet=False, grayscale=False, is_double_arm=False, aloha=False, camera_names=None):
    return standard_image_processing.inference_images(
        image_dict, normalize_resnet=normalize_resnet, grayscale=grayscale,
        is_double_arm=is_double_arm, aloha=aloha, camera_names=camera_names)






def fake_get_observations(state_dim=40):
    return {"qpos": np.zeros(state_dim), "qvel": np.zeros(state_dim)}

def get_observations(controllers):
    """
    Get observations from robots: qpos, qvel
    Args:
        controllers: dict with keys 'left' and optionally 'right', or single controller for backward compatibility
    """
    read_runtime_state = getattr(controllers, 'read_runtime_state', None)
    if read_runtime_state is not None:  # RuntimeArms: one SDK state result per arm.
        return read_runtime_state()
    # Handle backward compatibility - if single controller passed, treat as left
    if not isinstance(controllers, dict):
        controllers = {'left': controllers}
    
    # Get left arm state
    l_JointState = controllers['left'].get_joint_state()
    left_action = np.concatenate((l_JointState.pos().copy(), [l_JointState.gripper_pos]))
    left_vel = np.concatenate((l_JointState.vel().copy(), [l_JointState.gripper_vel]))

    if 'right' in controllers and controllers['right'] is not None:
        # Double arm configuration
        r_JointState = controllers['right'].get_joint_state()
        right_action = np.concatenate((r_JointState.pos().copy(), [r_JointState.gripper_pos]))
        right_vel = np.concatenate((r_JointState.vel().copy(), [r_JointState.gripper_vel]))
        
        # Concatenate left and right arm observations
        action = np.concatenate([left_action, right_action])
        vel = np.concatenate([left_vel, right_vel])
    else:
        # Single arm configuration
        action = left_action
        vel = left_vel
    
    obs = {
        "qpos": action,
        "qvel": vel,
    }

    return obs

def get_observations_array(controllers, observation_name=[]):
    """Get observations as a concatenated array."""
    obs_dict = get_observations(controllers)
    observation_data = []
    
    for name in observation_name:
        observation_data.append(obs_dict[name])

    obs = np.concatenate(observation_data, axis=-1)
    return {'qpos': obs}

def set_action(action, controllers, commands, is_double_arm=False, *, interpolate=True):
    """Send action command to robot(s).
    Args:
        action: action array
        controllers: dict with 'left' and optionally 'right' controllers
        commands: dict with 'left' and optionally 'right' commands
        is_double_arm: whether this is a double arm setup
        interpolate: False for immediate measured-pose anchoring and pause/stop holds
    """
    if isinstance(controllers, RuntimeArms):
        # The adapter's SDK interpolates natively over the runtime's policy interval.
        controllers.submit(np.asarray(action)[0])
        return

    def send(side):
        if interpolate:
            send_joint_target(controllers[side], commands[side], EXECUTION_DT)
        else:
            commands[side].timestamp = 0.0
            controllers[side].set_joint_cmd(commands[side])

    if is_double_arm:
        # Double arm: split action into left and right components
        action_dim_per_arm = action.shape[1] // 2
        left_action = action[0, :action_dim_per_arm]
        right_action = action[0, action_dim_per_arm:]
        
        # Set left arm
        commands['left'].pos()[:] = np.array(left_action[:6])
        commands['left'].gripper_pos = left_action[-1]
        send('left')
        
        # Set right arm
        commands['right'].pos()[:] = np.array(right_action[:6])
        commands['right'].gripper_pos = right_action[-1]
        send('right')
    else:
        # Single arm
        commands['left'].pos()[:] = np.array(action[0, :6])
        commands['left'].gripper_pos = action[0, -1]
        send('left')


def motion_joint_indices(is_double_arm=False):
    if is_double_arm:
        return list(range(6)) + list(range(7, 13))
    return list(range(6))


def gripper_indices(is_double_arm=False):
    return [6, 13] if is_double_arm else [6]


def load_start_action_from_npz(start_position_path, is_double_arm, action_dim,
                               closed_gripper_pos=GRIPPER_CLOSED_POS,
                               single_arm_side=None,
                               use_saved_gripper_positions=False):
    """Load saved calibration joint positions, optionally including grippers."""
    if start_position_path is None:
        return None

    path = os.path.abspath(os.path.expanduser(start_position_path))
    if not os.path.exists(path):
        print(f"[Start Position] {path} not found; keeping ARX home start.")
        return None

    with np.load(path, allow_pickle=True) as data:
        if is_double_arm:
            if action_dim != 14:
                raise ValueError(f"Double-arm start position expects action_dim=14, got {action_dim}")
            if 'left_qpos' in data and 'right_qpos' in data:
                left_qpos = np.asarray(data['left_qpos'], dtype=float).reshape(-1)
                right_qpos = np.asarray(data['right_qpos'], dtype=float).reshape(-1)
            elif 'qpos' in data and np.asarray(data['qpos']).size >= 14:
                qpos = np.asarray(data['qpos'], dtype=float).reshape(-1)
                left_qpos = qpos[:7]
                right_qpos = qpos[7:14]
            else:
                raise ValueError(
                    f"{path} must contain left_qpos/right_qpos or a 14D qpos for double-arm inference"
                )
            required_size = 7 if use_saved_gripper_positions else 6
            if left_qpos.size < required_size or right_qpos.size < required_size:
                raise ValueError(f"{path} contains incomplete double-arm qpos arrays")
            start_action = np.zeros((1, action_dim), dtype=np.float32)
            start_action[0, :6] = left_qpos[:6]
            start_action[0, 6] = (
                left_qpos[6] if use_saved_gripper_positions else closed_gripper_pos
            )
            start_action[0, 7:13] = right_qpos[:6]
            start_action[0, 13] = (
                right_qpos[6] if use_saved_gripper_positions else closed_gripper_pos
            )
        else:
            if action_dim != 7:
                raise ValueError(f"Single-arm start position expects action_dim=7, got {action_dim}")
            qpos_key = f"{single_arm_side}_qpos" if single_arm_side is not None else 'left_qpos'
            if qpos_key in data:
                qpos = np.asarray(data[qpos_key], dtype=float).reshape(-1)
            elif single_arm_side is None and 'left_qpos' in data:
                qpos = np.asarray(data['left_qpos'], dtype=float).reshape(-1)
            elif 'qpos' in data and np.asarray(data['qpos']).size >= 7:
                all_qpos = np.asarray(data['qpos'], dtype=float).reshape(-1)
                if single_arm_side == 'right' and all_qpos.size >= 14:
                    qpos = all_qpos[7:14]
                else:
                    qpos = all_qpos[:7]
            else:
                raise ValueError(
                    f"{path} must contain {qpos_key} or a compatible qpos for single-arm inference"
                )
            required_size = 7 if use_saved_gripper_positions else 6
            if qpos.size < required_size:
                raise ValueError(f"{path} contains incomplete single-arm qpos array")
            start_action = np.zeros((1, action_dim), dtype=np.float32)
            start_action[0, :6] = qpos[:6]
            start_action[0, 6] = (
                qpos[6] if use_saved_gripper_positions else closed_gripper_pos
            )

    if not np.isfinite(start_action).all():
        raise ValueError(f"{path} contains NaN or infinite start-position values")

    gripper_source = "saved pose" if use_saved_gripper_positions else "closed default"
    print(f"[Start Position] Loaded {path} (grippers: {gripper_source})")
    print(f"[Start Position] Target action={np.array2string(start_action[0], precision=3, suppress_small=True)}")
    return start_action


def move_to_start_action(controllers, commands, start_action, execution_dt,
                         is_double_arm=False, max_joint_step=0.01,
                         max_gripper_step=0.002, hold_seconds=0.25):
    if start_action is None:
        return

    current_qpos = np.asarray(get_observations(controllers)['qpos'], dtype=float).reshape(-1)
    current_action = start_action.astype(float).copy()
    current_action[0, :min(current_qpos.size, start_action.shape[1])] = current_qpos[:start_action.shape[1]]

    joint_delta = np.max(np.abs(start_action[0, motion_joint_indices(is_double_arm)] -
                                current_action[0, motion_joint_indices(is_double_arm)]))
    gripper_delta = np.max(np.abs(start_action[0, gripper_indices(is_double_arm)] -
                                  current_action[0, gripper_indices(is_double_arm)]))
    joint_steps = int(np.ceil(joint_delta / max_joint_step)) if max_joint_step > 0 else 1
    gripper_steps = int(np.ceil(gripper_delta / max_gripper_step)) if max_gripper_step > 0 else 1
    n_steps = max(1, joint_steps, gripper_steps)

    print(f"[Start Position] Moving to saved arm pose over {n_steps} control steps.")
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        action = current_action + (start_action - current_action) * alpha
        set_action(action, controllers, commands, is_double_arm)
        time_V2.sleep(execution_dt)

    hold_steps = max(1, int(round(hold_seconds / execution_dt))) if execution_dt > 0 else 1
    for _ in range(hold_steps):
        set_action(start_action, controllers, commands, is_double_arm)
        time_V2.sleep(execution_dt)
    print("[Start Position] Arm pose reached.")


def open_gripper_positions(controllers, is_double_arm=False):
    """Fully open gripper target per arm, in action order; None where none is known."""
    if isinstance(controllers, RuntimeArms):
        return [arm.open_gripper_target() for arm in controllers.runtime.arms.values()]
    sides = ('left', 'right') if is_double_arm else ('left',)
    return [float(controllers[side].get_robot_config().gripper_width) for side in sides]


def return_to_start_with_open_grippers(controllers, commands, start_action, execution_dt,
                                       is_double_arm=False, max_joint_step=0.01,
                                       max_gripper_step=0.002, hold_seconds=0.25,
                                       gripper_seconds=1.0):
    """Reset from a pause: open the grippers on the way back, then set the saved gripper.

    The arm joints and grippers ramp together to the start joints with open grippers.
    Only once the arm has settled there do the grippers ramp to the saved start value
    over ``gripper_seconds``, with the joints held. An arm with no known open position keeps its saved gripper
    target for the whole move.
    """
    if start_action is None:
        return
    indices = gripper_indices(is_double_arm)
    open_positions = open_gripper_positions(controllers, is_double_arm)
    open_action = start_action.astype(float).copy()
    for index, position in zip(indices, open_positions):
        if position is not None:
            open_action[0, index] = position
    move_to_start_action(controllers, commands, open_action, execution_dt=execution_dt,
                         is_double_arm=is_double_arm, max_joint_step=max_joint_step,
                         max_gripper_step=max_gripper_step, hold_seconds=hold_seconds)
    gripper_delta = np.max(np.abs(start_action[0, indices] - open_action[0, indices]))
    if gripper_delta == 0:
        return
    # From the commanded pose, so the joints do not move at all here.
    n_steps = max(1, int(round(gripper_seconds / execution_dt))) if execution_dt > 0 else 1
    print(f"[Start Position] Arm back; moving grippers to the saved position over {n_steps} control steps.")
    for step in range(1, n_steps + 1):
        action = open_action + (start_action - open_action) * (step / n_steps)
        set_action(action, controllers, commands, is_double_arm)
        time_V2.sleep(execution_dt)
    hold_steps = max(1, int(round(hold_seconds / execution_dt))) if execution_dt > 0 else 1
    for _ in range(hold_steps):
        set_action(start_action, controllers, commands, is_double_arm)
        time_V2.sleep(execution_dt)


def hold_action_until_unpaused(action, controllers, commands, execution_dt,
                               is_double_arm=False):
    try:
        current_qpos = np.asarray(get_observations(controllers)['qpos'], dtype=float).reshape(-1)
        hold_action = np.asarray(action, dtype=float).copy()
        hold_action[0, :min(current_qpos.size, hold_action.shape[1])] = current_qpos[:hold_action.shape[1]]
    except Exception as exc:
        print(f"[Warning] Failed to read current state for pause hold; holding last command instead: {exc}")
        hold_action = np.asarray(action, dtype=float).copy()

    print("Paused - holding current position. Press c/p/1 to continue, r/h to reset ready, or q to home and quit.")
    while pause_flag and not reset_ready_requested and not exit_requested:
        loop_start = time_V2.time()
        set_action(hold_action, controllers, commands, is_double_arm, interpolate=False)
        remaining = execution_dt - (time_V2.time() - loop_start)
        if remaining > 0:
            time_V2.sleep(remaining)
    return hold_action


def reset_controllers_to_home(controllers, is_double_arm=False):
    """Return initialized controller(s) to ARX home during shutdown."""
    if not controllers:
        return
    if isinstance(controllers, RuntimeArms):
        if controllers.runtime.live_arms():
            from hardware.runtime import RuntimeDamping
            print("Returning arm(s) home...")
            RuntimeDamping(controllers.runtime).go_home()
        return

    print("Returning arm(s) to home position...")
    for label in ('left', 'right'):
        if label == 'right' and not is_double_arm:
            continue
        controller = controllers.get(label)
        if controller is None:
            continue
        try:
            controller.reset_to_home()
            print(f"{label.capitalize()} arm reset to home.")
        except Exception as e:
            print(f"[Warning] Failed to reset {label} arm to home: {e}")


def validate_action(action, stats):
    action_orig = action.copy()
    action_v = action[:,1:]-action[:,:-1]
    action_v_max = np.max(np.abs(action_v), axis=1)
    data_action_v_max = stats['all_action_v_max']
    data_action_v_max[10] = 0.1
    if (action_v_max > data_action_v_max).any():
        print(f'Warning: action velocity too high')
    if (action_v_max > data_action_v_max*1.4).any():
        print(f'Warning: action velocity too high: 1.4x, filter')
        action_v = np.clip(action_v, -1000, data_action_v_max)
        action_v = np.concatenate([action_orig[:,0:1], action_v], axis=1)
        action = np.cumsum(action_v, axis=1)
    return action


def init_arm(is_double_arm=False, kp=None, kd=None, single_arm_can=None, single_arm_side=None,
             can_interface_left=None, can_interface_right=None):
    np.set_printoptions(precision=3, suppress=True)
    if arx5 is None:
        raise ImportError('arx5_interface is not installed; ARX direct inference needs the ARX SDK')
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_config)
    configure_active_gains(controller_config)
    controller_config.gravity_compensation = True
    # Friction feedforward is for hand guiding only; it stays off while the policy moves the arm.
    controller_config.friction_compensation = False
    controller_config.background_send_recv = True
    configure_controller(controller_config)

    left_can = (can_interface_left or "can0") if is_double_arm else (single_arm_can or can_interface_left or "can0")
    active_label = 'left' if is_double_arm else (single_arm_side or 'left')
    print(f"[ARX] Initializing {active_label} arm on {left_can}")
    arx5_joint_controller_left = arx5.Arx5JointController(robot_config, controller_config, left_can)
    arx5_joint_controller_right = None
    
    if is_double_arm:
        right_can = can_interface_right or "can1"
        print(f"[ARX] Initializing right arm on {right_can}")
        arx5_joint_controller_right = arx5.Arx5JointController(robot_config, controller_config, right_can)

    arx5_joint_controller_left.reset_to_home()
    if arx5_joint_controller_right is not None:
        arx5_joint_controller_right.reset_to_home()

    configure_arx_runtime_gains(
        arx5_joint_controller_left,
        kp=kp,
        kd=kd,
        label=active_label,
    )
    if arx5_joint_controller_right is not None:
        configure_arx_runtime_gains(
            arx5_joint_controller_right,
            kp=kp,
            kd=kd,
            label='right',
        )

    return robot_config, arx5_joint_controller_left, arx5_joint_controller_right

def setup_cameras_and_robot(is_double_arm=False, camera_side='left', use_camera_server=True, camera_names=None):
    if not use_camera_server:
        raise ValueError('Live inference requires camera_server and standard_capture packet readers')
    """Setup camera capture thread and initialize robot communication."""
    if DEBUG:
        capture_fake_image()
        return None
    else:
        stop_event = threading.Event()
        camera_short_names = None
        if camera_names is not None:
            camera_short_names = [cn.replace('video_', '') for cn in camera_names]

        if use_camera_server:
            if camera_short_names is not None:
                capture_thread = threading.Thread(
                    target=capture_image_from_shared_memory,
                    args=(stop_event,),
                    kwargs={'camera_names': camera_short_names},
                )
            else:
                capture_thread = threading.Thread(
                    target=capture_image_from_shared_memory,
                    args=(stop_event, is_double_arm, camera_side),
                )
        else:
            if camera_names is not None:
                capture_thread = threading.Thread(
                    target=capture_image,
                    args=(stop_event,),
                    kwargs={'camera_names': camera_names},
                )
            else:
                capture_thread = threading.Thread(
                    target=capture_image,
                    args=(stop_event, is_double_arm, camera_side),
                )
        capture_thread.daemon = True
        capture_thread.start()
        return stop_event

def setup_rerun_visualization(is_double_arm=False):
    """Initialize rerun visualization if requested."""
    rr.init("Robot Control", spawn=True)
    # Parse URDF and log robot model
    if os.path.exists(urdf_path):
        links, joints = parse_urdf(urdf_path)
        left_arm_transform = None
        right_arm_transform = None

        if is_double_arm:
            # Display placement of the two arm models, side by side.
            left_arm_transform = np.eye(4)
            left_arm_transform[:3, 3] = [-0.25, 0, 0]
            right_arm_transform = np.eye(4)
            right_arm_transform[:3, 3] = [0.25, 0, 0]

        # Log initial robot pose(s)
        print("Setting initial robot pose(s)...")
        if is_double_arm:
            from visualization.rerun_utils import log_robot_model_with_transform
            log_robot_model_with_transform(links, joints, gripper_pos=0.0, base_transform=left_arm_transform, robot_name="left_arm")
            log_robot_model_with_transform(links, joints, gripper_pos=0.0, base_transform=right_arm_transform, robot_name="right_arm")
        else:
            log_robot_model(links, joints, gripper_pos=0.0)
    else:
        print(f"Warning: URDF file not found at {urdf_path}")
        return None, None

def load_policy_and_stats(config, ckpt_name, fast_inference=False):
    """Load policy model and statistics from checkpoint."""
    require_absolute_actions(config)
    ckpt_dir = config['ckpt_dir']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    
    # Load policy
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path, map_location="cuda:0"))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if fast_inference:
        from policy.model_util import accelerate_policy
        accelerate_policy(policy)
    
    # Load statistics
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    from contracts.robot_contract import metadata_robot_contract, require_robot_match
    robot = metadata_robot_contract(config)
    if robot is not None:
        require_robot_match(robot, stats, context='direct policy normalization statistics')
    elif metadata_robot_contract(stats) is not None:
        raise ValueError('Tagged statistics require a checkpoint robot_contract')
    
    print(f'=======================MODEL Loaded=========================:')
    print(f'{ckpt_path}')
    print(f'{stats_path}')
    
    return policy, stats, policy_config

def setup_temporal_aggregation(num_queries, max_timesteps, action_dim):
    """Initialize temporal aggregation data structures."""
    # Use circular buffer approach to prevent memory issues
    all_time_actions = torch.zeros((num_queries, max_timesteps, action_dim)).cuda()
    action_timestamps = torch.full((num_queries,), -1, dtype=torch.long).cuda()
    print(f"Temporal aggregation initialized with circular buffer: {num_queries} slots")
    return all_time_actions, action_timestamps

def get_current_arm_state(task_config, controllers, state_dim=None):
    """Get current robot arm state."""
    if DEBUG:
        obs = fake_get_observations(state_dim)
    else:
        observation_name = task_config['observation_name'] if 'observation_name' in task_config.keys() else ['qpos']
        if len(observation_name) > 1:
            obs = get_observations_array(controllers, observation_name)
        else:
            obs = get_observations(controllers)
    return obs

_last_observation_timestamps = {}


def capture_current_observation(config, controllers, state_dim, camera_names):
    global _last_observation_timestamps
    observation = standard_capture.capture_observation(
        lambda: fake_get_observations(state_dim) if DEBUG else get_observations(controllers),
        camera_names, include_images=True)
    _last_observation_timestamps = {name: reference['capture_timestamp']
        for name, reference in observation.camera_observation['cameras'].items()}
    return observation


def get_current_images(config, is_double_arm=False, camera_names=None):
    """Get current camera images."""
    global _last_observation_timestamps
    if not DEBUG:
        names = camera_names or (['video_left', 'video_overhead', 'video_right'] if is_double_arm
                                 else ['video_left', 'video_overhead'])
        frames, timestamps = camera_capture.get_camera_bundle(names)
        _last_observation_timestamps = timestamps
        return {name: standard_image_processing.resize_image(frame, config['width'], config['height'])
                for name, frame in frames.items()}
    if camera_names is not None:
        return get_stored_images(config['height'], config['width'], camera_names=camera_names)

    img_l, img_h, img_r = get_stored_images(config['height'], config['width'], double_arm=is_double_arm)
    image_dict = {
        'video_left': img_l,
        'video_overhead': img_h
    }
    if is_double_arm:
        image_dict['video_right'] = img_r
    return image_dict


def get_current_image_timestamps(camera_names):
    """Return the timestamps belonging to the images supplied to the policy."""
    if not DEBUG:
        return {name: _last_observation_timestamps[name] for name in camera_names}
    timestamps = {}
    for cam_name in camera_names:
        timestamps[cam_name] = float(camera_capture.get_latest_camera_timestamp(cam_name))
    return timestamps


def get_timestamp_reference_camera(camera_names):
    """Select reference camera used for aligned action/query timestamps."""
    if 'video_overhead' in camera_names:
        return 'video_overhead'
    return camera_names[0] if len(camera_names) > 0 else None


def have_required_camera_frames(camera_names):
    camera_globals = {
        'video_left': camera_capture.captured_img_l,
        'video_overhead': camera_capture.captured_img_h,
        'video_right': camera_capture.captured_img_r,
        'video_torso': camera_capture.captured_img_t,
    }
    return all(camera_globals.get(cam_name) is not None for cam_name in camera_names)

def direct_warmup_query(policy, qpos, curr_image, rtc_prefix_steps=0):
    """One direct policy query for policy_warmup.warm_up, like the loop's own queries.

    Query 0 has no prefix, like an episode's first; later ones, with direct RTC,
    carry a prefix of the trained length so the prefixed sampler is warm too.
    """
    def query(index):
        started = time_V2.perf_counter()
        prefixed = bool(rtc_prefix_steps and index > 0)
        if prefixed:
            prefix = torch.zeros((qpos.shape[0], rtc_prefix_steps, policy.action_dim), device=qpos.device)
            actions = policy.forward_inf_training_time_rtc(qpos, curr_image, prefix)
        else:
            actions = policy.forward_inf(qpos, curr_image)
        actions.cpu()  # waits for the GPU, as the loop's .cpu().numpy() does
        return {'query_total': time_V2.perf_counter() - started, 'rtc_prefixed': prefixed}
    return query


def warm_up_policy(policy, qpos, curr_image, rtc_prefix_steps=0, execution_dt=None):
    """Discarded startup queries so the first real query is not cold (CUDA graphs come from --fast_inference).

    With RTC, prefixed queries run asynchronously during the pinned prefix; as in remote
    mode, a warm query slower than that overlap fails before execution starts.
    """
    from inference.policy_warmup import warm_up
    from inference.policy_chunks import validate_overlap_budget
    check = None
    if rtc_prefix_steps:
        if execution_dt is None:
            raise ValueError('RTC warm-up needs the execution interval to check the overlap budget')

        def check(timing):
            if timing['rtc_prefixed']:
                validate_overlap_budget(1.0 / execution_dt, rtc_prefix_steps, timing['query_total'] * 1000)
    warm_up(direct_warmup_query(policy, qpos, curr_image, rtc_prefix_steps), check=check)
    return policy

def blend_actions(prev_action_chunk, action_chunk, mixing_steps, query_frequency,
                  prev_chunk_start_index, new_chunk_start_index, interp_type='linear'):
    """Blend previous and new action chunks aligned to executed indices."""
    new_chunk = action_chunk.copy()
    prev_blend_start = query_frequency + prev_chunk_start_index
    new_blend_start = new_chunk_start_index

    chunk_size = prev_action_chunk.shape[1]
    prev_blend_end = min(prev_blend_start + mixing_steps, chunk_size)
    new_blend_end = min(new_blend_start + mixing_steps, action_chunk.shape[1])
    actual_mix_len = min(prev_blend_end - prev_blend_start, new_blend_end - new_blend_start)

    if actual_mix_len <= 0:
        return action_chunk

    if interp_type == 'linear':
        start = prev_action_chunk[:, prev_blend_start]
        end = action_chunk[:, new_blend_start + actual_mix_len - 1]
        new_chunk[:, new_blend_start:new_blend_start + actual_mix_len] = np.linspace(start, end, num=actual_mix_len, axis=1)
    elif interp_type == 'exponential':
        for i in range(actual_mix_len):
            weight = 1 - (i / actual_mix_len) ** 3
            new_chunk[:, new_blend_start + i] = (
                weight * prev_action_chunk[:, prev_blend_start + i] +
                (1 - weight) * action_chunk[:, new_blend_start + i]
            )
    elif interp_type == 'sigmoid':
        for i in range(actual_mix_len):
            weight = 1 / (1 + np.exp(3 * (i / actual_mix_len - 0.5)))
            new_chunk[:, new_blend_start + i] = (
                weight * prev_action_chunk[:, prev_blend_start + i] +
                (1 - weight) * action_chunk[:, new_blend_start + i]
            )

    return new_chunk


def blend_rtc_boundary(prev_plan, new_chunk, prev_next_index, boundary, steps, interp_type='sigmoid'):
    """Crossfade the executed actions from the previous plan into a new RTC chunk.

    Only adjusts actions after a prediction; the policy and its prefixes are untouched.
    ``boundary`` is the handoff: the RTC prefix length, where the new chunk's own
    predictions begin. Before it the new chunk is the pinned prefix, equal to the
    previous plan, and is left as is; the ``steps`` new-chunk actions from the handoff
    on fade from the previous plan into the new chunk. New-chunk index ``i`` and
    previous-plan index ``prev_next_index + i`` are the same instant (the prefix
    aligns them), so the previous plan must still hold ``steps`` actions after the
    handoff.
    """
    if steps <= 0:
        raise ValueError(f'Blending window {steps} must be positive')
    if prev_next_index + boundary + steps > prev_plan.shape[1] or boundary + steps > new_chunk.shape[1]:
        raise ValueError(f'Blending {steps} actions after the handoff needs actions up to previous-plan index '
                         f'{prev_next_index + boundary + steps}; the previous plan has only {prev_plan.shape[1]}')
    blended = new_chunk.copy()
    for k in range(steps):
        index, old_index = boundary + k, prev_next_index + boundary + k
        x = (k + 0.5) / steps  # position across the window, 0 -> 1
        if interp_type == 'linear':
            weight = 1 - x
        elif interp_type == 'exponential':
            weight = 1 - x ** 3
        elif interp_type == 'sigmoid':
            weight = 1 / (1 + np.exp(6 * (x - 0.5)))
        else:
            raise ValueError(f'Unknown blending method {interp_type!r}')
        blended[:, index] = weight * prev_plan[:, old_index] + (1 - weight) * new_chunk[:, index]
    return blended


# A fresh start's first target is sent as one step; beyond this (rad, any arm joint) it is refused.
# Normal starts stay well inside it: a first chunk runs from its first action.
FRESH_START_MAX_TARGET_GAP = 0.15


def check_fresh_start_target(action, qpos, is_double_arm=False, limit=FRESH_START_MAX_TARGET_GAP):
    """Fail closed before sending a fresh start's first target far from the measured arm pose.

    Nothing blends a fresh start's first target with the arm's state, so the arm is commanded to
    jump to it in one control step.
    """
    joints = motion_joint_indices(is_double_arm)
    gap = np.abs(np.asarray(action, dtype=float).reshape(-1)[joints] - np.asarray(qpos, dtype=float).reshape(-1)[joints])
    if gap.max() > limit:
        index = joints[int(np.argmax(gap))]
        raise RuntimeError(f'First policy target is {gap.max():.3f} rad from the measured pose at action index {index} '
                           f'(limit {limit} rad); refusing to send it')


def chunk_handoff_metrics(old_plan, new_raw, executed, last_sent, before_last, old_offset, boundary,
                          query_lag, step, is_double_arm, window=10, image_age_ms=None, mode='rtc'):
    """Numbers for one direct chunk handoff, to diagnose back-and-forth at chunk boundaries (arm joints, rad).

    New-chunk index ``i`` and old-plan index ``old_offset + i`` are the same instant; ``boundary``
    is the first new index executed. jump: the first new target minus the last sent one, next to
    the step before it. new vs old: the new prediction minus the old plan at the same instants
    after the handoff; 'along' projects their mean difference on the old plan's motion over that
    window (negative: the new plan is behind and pulls the arm back). trailed: last target minus
    measured qpos at the query (with action_lookahead it includes the intended lead). image age:
    the oldest query image's capture time before the proprio read (training data: median ~18 ms
    on one task).
    """
    arm = np.r_[0:6, 7:13] if is_double_arm else np.arange(6)
    last, before = np.asarray(last_sent).reshape(-1)[arm], np.asarray(before_last).reshape(-1)[arm]
    old_index = old_offset + boundary
    n = max(1, min(window, old_plan.shape[1] - old_index, new_raw.shape[1] - boundary))
    diff = new_raw[0, boundary:boundary + n][:, arm] - old_plan[0, old_index:old_index + n][:, arm]
    motion = old_plan[0, old_index + n - 1, arm] - old_plan[0, old_index, arm]
    speed = float(np.linalg.norm(motion))
    lag = np.asarray(query_lag, dtype=float).reshape(-1)[arm]
    j = int(np.argmax(np.abs(lag)))
    return dict(mode=mode, step=int(step), boundary=int(boundary), window=int(n),
                jump=float(np.abs(executed[0, boundary, arm] - last).max()),
                previous_step=float(np.abs(last - before).max()),
                new_vs_old_max=float(np.abs(diff).max()),
                new_vs_old_along=float(diff.mean(0) @ motion / speed) if speed > 1e-9 else 0.0,
                old_motion=speed, trailed_max=float(abs(lag[j])), trailed_joint=j,
                trailed=lag.tolist(), image_age_ms=None if image_age_ms is None else float(image_age_ms))


def format_chunk_handoff(m):
    along = m['new_vs_old_along']
    return (f"[Direct {'RTC' if m['mode'] == 'rtc' else 'lookahead'}] handoff step={m['step']}: "
            f"target jump {m['jump']:.4f} (previous step {m['previous_step']:.4f}); "
            f"new vs old plan over next {m['window']}: max {m['new_vs_old_max']:.4f}, {along:+.4f} along motion "
            f"{'(behind)' if along < 0 else '(ahead)'} of {m['old_motion']:.4f}; "
            f"arm trailed its target at query by {m['trailed_max']:.4f} (joint {m['trailed_joint']})"
            + ('' if m['image_age_ms'] is None else f"; image age {m['image_age_ms']:.1f} ms"))


def rtc_planned_prefix(plan, chunk_start_index, executed_steps, prefix_steps):
    """The previous plan's next ``prefix_steps`` actions, from the first one not yet executed.

    ``plan`` is the (1, T, D) chunk being executed from ``chunk_start_index``, of
    which ``executed_steps`` actions have been sent. Returns None if nothing is left.
    """
    next_index = chunk_start_index + executed_steps
    planned = plan[:, next_index:next_index + prefix_steps]
    return planned if planned.shape[1] else None


class DirectRTCQueries:
    """Direct RTC runs asynchronously, like remote mode, on the same query worker.

    Each query after a fresh start is submitted ``prefix_steps`` before its chunk
    boundary, with the plan's next ``prefix_steps`` actions pinned as the prefix.
    The arm keeps executing those actions while the policy runs, and at the boundary
    the result takes over from index ``prefix_steps``. A result that is not ready by
    then, or took the whole overlap, fails closed. Fresh starts (the first query, or
    after a reset or hand correction) query synchronously without a prefix.
    """

    LOG_PREFIX = '[Direct RTC]'

    def __init__(self, policy, prefix_steps, execution_dt, first_chunk_size, query_frequency,
                 timing_stats=None):
        from inference.policy_chunks import AsyncOpenPIQueryWorker
        from inference.policy_transport import TimingStats
        if not 0 < prefix_steps < min(first_chunk_size, query_frequency):
            raise ValueError('Direct RTC prefix must be shorter than the executed chunks')
        self.policy, self.prefix_steps = policy, int(prefix_steps)
        self.first_chunk_size, self.query_frequency = first_chunk_size, query_frequency
        self.pending, self.result, self.queries = False, None, 0
        self.worker = AsyncOpenPIQueryWorker(policy_client=None, query_actions=self._query,
                                             timing_stats=timing_stats or TimingStats(enabled=False, interval=1),
                                             max_delay_s=self.prefix_steps * execution_dt)
        self.worker.start()

    def _query(self, _client, image_dict, qpos, _prompt, *, query_index, rtc_metadata):
        start = time_V2.time()
        image, prefix = image_dict['image'], rtc_metadata['action_prefix']['actions']
        with torch.inference_mode():
            chunk = self.policy.forward_inf_training_time_rtc(
                torch.from_numpy(qpos).to(image.device), image, prefix)
            chunk = chunk.float().cpu().numpy()
        return chunk, {'query_total': time_V2.time() - start}, chunk.shape

    def warm_up(self, qpos, image, queries=3):
        """Discarded prefixed queries on the worker thread before execution starts.

        warm_up_policy warms the main thread only; the worker thread's first query also
        pays one-time CUDA setup (about 40 ms for ACT on the laptop), close to the overlap.
        A cold query may exceed the overlap, but the last one must not: a policy too slow
        for the overlap fails here, before the arm moves.
        """
        from inference.policy_chunks import InferenceDelayExceeded
        prefix = torch.zeros((qpos.shape[0], self.prefix_steps, self.policy.action_dim), device=qpos.device)
        latency = None
        for _ in range(queries):
            self.discard()
            if not self.worker.submit(image_dict={'image': image}, qpos=qpos.detach().cpu().numpy(), prompt='',
                                      query_index=-1, submit_chunk_step=0, copy_inputs=False,
                                      rtc_metadata={'action_prefix': {'prefix_length': self.prefix_steps,
                                                                      'actions': prefix}}):
                raise RuntimeError('Direct RTC warm-up could not submit to an idle worker')
            latency, result = None, None
            try:
                while result is None:
                    self.worker.raise_if_failed()
                    result = self.worker.poll_result()
                    time_V2.sleep(0.001)
                latency = result['latency_s']
            except InferenceDelayExceeded:
                pass
        self.discard()
        if latency is None:
            raise InferenceDelayExceeded(f'Warm direct RTC query still exceeds the {self.prefix_steps}-step overlap')
        print(f'{self.LOG_PREFIX} worker warm-up: {queries} queries, last {latency * 1000:.1f} ms')

    def schedule(self, t, plan_valid):
        """(observe, submit, merge) for step ``t``; observe means a policy observation is taken."""
        boundary = is_query_point(t, self.first_chunk_size, self.query_frequency)
        if boundary and t > 0 and self.pending:
            if plan_valid:
                return False, False, True  # its observation was taken at submission
            self.discard()  # the arm left the plan (e.g. a hand correction): query fresh
        if (not boundary and plan_valid and not self.pending
                and is_query_point(t + self.prefix_steps, self.first_chunk_size, self.query_frequency)):
            return True, True, False
        return boundary, False, False

    def submit(self, qpos, image, prefix, t):
        if prefix.shape[1] != self.prefix_steps:
            raise ValueError(f'RTC prefix has {prefix.shape[1]} actions; the plan must cover {self.prefix_steps}')
        if not self.worker.submit(image_dict={'image': image}, qpos=qpos.detach().cpu().numpy(), prompt='',
                                  query_index=self.queries, submit_chunk_step=t, copy_inputs=False,
                                  rtc_metadata={'action_prefix': {'prefix_length': self.prefix_steps,
                                                                  'actions': prefix}}):
            raise RuntimeError('Direct RTC query submitted while the previous one is still running')
        self.pending, self.result = True, None
        self.queries += 1

    def poll(self, t):
        """Collect a finished query each step, printing remote mode's 'received' line."""
        from inference.policy_chunks import poll_async_result
        if self.pending and self.result is None:
            self.result = poll_async_result(self.worker, None, current_chunk_step=t,
                                            log_prefix=self.LOG_PREFIX, step=t)

    def take(self, t):
        """The submitted query's normalized chunk, at its boundary; late results fail closed."""
        from inference.policy_chunks import InferenceDelayExceeded
        self.poll(t)
        result = self.result
        if result is None:
            raise InferenceDelayExceeded(f'Policy response not ready {self.prefix_steps} steps after its query; '
                                         'the arm has executed the whole pinned prefix')
        self.pending, self.result = False, None
        print(f'{self.LOG_PREFIX} merged query={result["query_index"]} step={t} '
              f'configured_overlap={self.prefix_steps} observed_steps={t - int(result["submit_chunk_step"])} '
              f'latency={result["latency_s"] * 1000:.1f}ms')
        return result['action_chunk'], int(result['rtc_prefix_length'])

    def discard(self):
        """Drop any submitted query and wait until the policy is free for a synchronous query."""
        self.worker.reset()
        self.worker.wait_until_idle()
        self.pending, self.result = False, None

    def stop(self):
        self.worker.stop()


def get_chunk_step_offset(t, first_chunk_size, query_frequency):
    """Calculate the timestep offset within the current chunk."""
    if t < first_chunk_size:
        return t
    return (t - first_chunk_size) % query_frequency


def get_action_index(t, first_chunk_size, query_frequency, chunk_start_index, chunk_length):
    """Calculate the executed action index within the current chunk."""
    action_ind = get_chunk_step_offset(t, first_chunk_size, query_frequency)
    return min(action_ind + chunk_start_index, chunk_length - 1)


def is_query_point(t, first_chunk_size, query_frequency):
    """Check whether the policy should be queried at the current timestep."""
    return t == 0 or (t >= first_chunk_size and (t - first_chunk_size) % query_frequency == 0)


def select_action_from_chunk(t, first_chunk_size, query_frequency, chunk_start_index,
                             gripper_lookahead, all_actions, is_double_arm):
    """Select an action from the current chunk with joint/gripper lookahead."""
    chunk_length = all_actions.shape[1]
    action_ind = get_action_index(t, first_chunk_size, query_frequency, chunk_start_index, chunk_length)

    action = all_actions[:, action_ind].copy()
    gripper_action_ind = min(action_ind + gripper_lookahead, chunk_length - 1)
    action[0, 6] = all_actions[0, gripper_action_ind, 6]
    if is_double_arm:
        action[0, 13] = all_actions[0, gripper_action_ind, 13]
    return action


def find_nn_chunk_start_index(prev_action_chunk, next_action_chunk, prev_chunk_size,
                              prev_chunk_start_index, action_lookahead,
                              search_radius=NN_CHUNK_SEARCH_RADIUS):
    """
    Choose the next chunk start index by matching a target future timestep.

    `action_lookahead` defines the center of the search window in the next chunk.
    The matched nearest-neighbor index becomes the immediate start index of the
    next chunk.
    """
    prev_target_index = min(
        prev_chunk_start_index + prev_chunk_size + action_lookahead,
        prev_action_chunk.shape[1] - 1,
    )
    next_target_index = min(action_lookahead, next_action_chunk.shape[1] - 1)
    search_start = max(next_target_index - search_radius, 0)
    search_end = min(next_target_index + search_radius, next_action_chunk.shape[1] - 1)

    target_action = prev_action_chunk[0, prev_target_index]
    candidate_actions = next_action_chunk[0, search_start:search_end + 1]
    distances = np.linalg.norm(candidate_actions - target_action[None, :], axis=1)
    best_match_index = search_start + int(np.argmin(distances))

    return best_match_index, best_match_index, prev_target_index


def get_rollout_auto_index(dataset_dir):
    """Get auto-incrementing index for rollout recording episodes."""
    max_idx = 10000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.exists(os.path.join(dataset_dir, f'episode_{i:05d}.zarr')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def save_rollout_episode(dataset_dir, episode_idx, image_lists, position_list,
                         commanded_action_list, action_dim, is_double_arm,
                         camera_names, velocity_list=None, action_timestamps=None,
                         image_timestamps=None, inference_query_indices=None,
                         inference_query_timestamps=None,
                         inference_query_overhead_frame_numbers=None,
                         inference_action_chunks=None,
                         inference_chunk_query_indices=None,
                         inference_chunk_merge_indices=None,
                         inference_chunk_start_indices=None,
                         inference_chunk_query_ids=None,
                         inference_chunk_submit_steps=None,
                         inference_chunk_measured_start_indices=None,
                         inference_chunk_query_qpos=None,
                         timestamp_reference_camera=None):
    """Publish rollout streams and numeric inference diagnostics."""
    from data_processing.episode_store import publish_samples
    buffers = {id(images.buffer): images.buffer for images in image_lists.values() if hasattr(images, 'buffer')}
    if len(buffers) != 1:
        raise ValueError('Rollout saving requires its original recorded observation buffer')
    buffer = next(iter(buffers.values()))
    path = os.path.join(dataset_dir, f'episode_{episode_idx:05d}.zarr')
    metadata = dict(sim=False, rollout=True, robot_type='arx', robot_model='L5',
        single_arm=not is_double_arm, double_arm=is_double_arm,
        timestamp_reference_camera=timestamp_reference_camera)
    diagnostics = {
        'inference/query_indices': inference_query_indices,
        'inference/query_timestamps': inference_query_timestamps,
        'inference/query_overhead_frame_numbers': inference_query_overhead_frame_numbers,
        'inference/chunks/actions': inference_action_chunks,
        'inference/chunks/query_indices': inference_chunk_query_indices,
        'inference/chunks/merge_indices': inference_chunk_merge_indices,
        'inference/chunks/start_indices': inference_chunk_start_indices,
        'inference/chunks/query_ids': inference_chunk_query_ids,
        'inference/chunks/submit_chunk_steps': inference_chunk_submit_steps,
        'inference/chunks/measured_start_indices': inference_chunk_measured_start_indices,
        'inference/chunks/query_qpos': inference_chunk_query_qpos,
    }
    def annotate(root):
        for name, values in diagnostics.items():
            if values is not None and len(values):
                root.create_dataset(name, data=np.asarray(values))
        if inference_query_indices is not None and len(inference_query_indices):
            mask = np.zeros(len(buffer), dtype=np.uint8)
            indices = np.asarray(inference_query_indices, dtype=np.int64)
            mask[indices[(indices >= 0) & (indices < len(buffer))]] = 1
            root.create_dataset('inference_query_mask', data=mask)
    publish_samples(path, buffer, metadata=metadata, annotate=annotate)
    print(f'[Rollout Recording] Saved streams and diagnostics: {path}')
    return True


def prompt_success_failure():
    """Prompt user to mark episode as success or failure."""
    print("\n" + "="*50)
    print("Episode complete. Mark as:")
    print("  [s] Success")
    print("  [f] Failure")
    print("  [d] Discard (don't save)")
    print("="*50)

    while True:
        try:
            key = getkey()
            if key.lower() == 's':
                print("Marked as SUCCESS")
                return 'success'
            if key.lower() == 'f':
                print("Marked as FAILURE")
                return 'failure'
            if key.lower() == 'd':
                print("DISCARDED")
                return 'discard'
            print(f"Unknown key '{key}'. Press s/f/d")
        except (KeyboardInterrupt, EOFError):
            print("Interrupted - discarding episode")
            return 'discard'


def prompt_keep_discard():
    """Prompt user to keep or discard a non-eval rollout."""
    print("\n" + "="*50)
    print("Episode complete. Save rollout?")
    print("  [k] Keep")
    print("  [d] Discard (don't save)")
    print("="*50)

    while True:
        try:
            key = getkey()
            if key.lower() == 'k':
                print("Keeping rollout")
                return 'keep'
            if key.lower() == 'd':
                print("DISCARDED")
                return 'discard'
            print(f"Unknown key '{key}'. Press k/d")
        except (KeyboardInterrupt, EOFError):
            print("Interrupted - discarding episode")
            return 'discard'


def prompt_rollout_label(eval_mode=False):
    if eval_mode:
        return prompt_success_failure()
    return prompt_keep_discard()


def create_rollout_lists(camera_names):
    """Create empty rollout data collection lists including per-camera image lists."""
    return {
        'images': {cam_name: [] for cam_name in camera_names},
        'image_timestamps': {cam_name: [] for cam_name in camera_names},
        'position': [],
        'commanded_action': [],
        'velocity': [],
        'action_timestamps': [],
        'inference_query_indices': [],
        'inference_query_timestamps': [],
        'inference_query_overhead_frame_numbers': [],
        'inference_action_chunks': [],
        'inference_chunk_query_indices': [],
        'inference_chunk_merge_indices': [],
        'inference_chunk_start_indices': [],
        'inference_chunk_query_ids': [],
        'inference_chunk_submit_steps': [],
        'inference_chunk_measured_start_indices': [],
        'inference_chunk_query_qpos': [],
    }


def clear_rollout_lists(rollout_lists):
    standard_recording.finish_rollout(rollout_lists)
    for cam_name in rollout_lists['images']:
        rollout_lists['images'][cam_name] = []
    for cam_name in rollout_lists['image_timestamps']:
        rollout_lists['image_timestamps'][cam_name] = []
    rollout_lists['position'] = []
    rollout_lists['commanded_action'] = []
    rollout_lists['velocity'] = []
    rollout_lists['action_timestamps'] = []
    rollout_lists['inference_query_indices'] = []
    rollout_lists['inference_query_timestamps'] = []
    rollout_lists['inference_query_overhead_frame_numbers'] = []
    rollout_lists['inference_action_chunks'] = []
    rollout_lists['inference_chunk_query_indices'] = []
    rollout_lists['inference_chunk_merge_indices'] = []
    rollout_lists['inference_chunk_start_indices'] = []
    rollout_lists['inference_chunk_query_ids'] = []
    rollout_lists['inference_chunk_submit_steps'] = []
    rollout_lists['inference_chunk_measured_start_indices'] = []
    rollout_lists['inference_chunk_query_qpos'] = []


def save_and_clear_rollout_data(rollout_dir, task_name, action_dim, is_double_arm,
                                camera_names, rollout_lists, label, subdir_name,
                                eval_mode=False, date_string=None):
    """Save rollout episode to disk and clear lists."""
    episode_buffer = standard_recording.finish_rollout(rollout_lists)
    if label == 'discard':
        print("[Rollout Recording] Episode discarded")
        clear_rollout_lists(rollout_lists)
        return False

    if len(rollout_lists['position']) == 0:
        print("[Rollout Recording] No data to save")
        return False

    save_parts = [resolve_rollout_recording_root(rollout_dir, task_name, eval_mode, date_string)]
    if subdir_name:
        save_parts.append(subdir_name)
    if eval_mode:
        if label not in ('success', 'failure'):
            raise ValueError(f"Eval rollout label must be success/failure/discard, got {label!r}")
        save_parts.append(label)
    save_dir = os.path.join(*save_parts)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episode_idx = get_rollout_auto_index(save_dir)
    print(f"[Rollout Recording] Saving {len(rollout_lists['position'])} timesteps to {save_dir}/...")

    save_rollout_episode(
        save_dir,
        episode_idx,
        rollout_lists['images'],
        rollout_lists['position'],
        rollout_lists['commanded_action'],
        action_dim,
        is_double_arm,
        camera_names,
        velocity_list=rollout_lists['velocity'],
        action_timestamps=rollout_lists['action_timestamps'],
        image_timestamps=rollout_lists['image_timestamps'],
        inference_query_indices=rollout_lists['inference_query_indices'],
        inference_query_timestamps=rollout_lists['inference_query_timestamps'],
        inference_query_overhead_frame_numbers=rollout_lists['inference_query_overhead_frame_numbers'],
        inference_action_chunks=rollout_lists.get('inference_action_chunks'),
        inference_chunk_query_indices=rollout_lists.get('inference_chunk_query_indices'),
        inference_chunk_merge_indices=rollout_lists.get('inference_chunk_merge_indices'),
        inference_chunk_start_indices=rollout_lists.get('inference_chunk_start_indices'),
        inference_chunk_query_ids=rollout_lists.get('inference_chunk_query_ids'),
        inference_chunk_submit_steps=rollout_lists.get('inference_chunk_submit_steps'),
        inference_chunk_measured_start_indices=rollout_lists.get('inference_chunk_measured_start_indices'),
        inference_chunk_query_qpos=rollout_lists.get('inference_chunk_query_qpos'),
        timestamp_reference_camera=get_timestamp_reference_camera(camera_names),
    )
    if eval_mode:
        print_success_rate_banner(os.path.dirname(save_dir))

    clear_rollout_lists(rollout_lists)
    return True


CAMERA_RERUN_NAMES = {
    'video_left': 'left',
    'video_overhead': 'overhead',
    'video_right': 'right',
    'video_torso': 'torso',
    'video_test': 'test',
}


def iter_available_camera_images(image_dict, camera_names):
    for cam_name in camera_names:
        image = image_dict.get(cam_name)
        if image is not None:
            yield cam_name, image


def log_rerun_camera_images(image_dict, camera_names):
    for cam_name, image in iter_available_camera_images(image_dict, camera_names):
        rerun_name = CAMERA_RERUN_NAMES.get(cam_name, cam_name.replace('video_', ''))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rr.log(f"cameras/{rerun_name}", rr.Image(rgb_image))


def make_visualization_frame(image_dict, camera_names):
    images = [image for _, image in iter_available_camera_images(image_dict, camera_names)]
    if not images:
        return None
    return np.concatenate(images, axis=1)


def snapshot_controller_gain(controller):
    """Capture numeric gain values without holding a mutable SDK gain object."""
    gain = controller.get_gain()
    return {
        'kp': gain.kp().copy(),
        'kd': gain.kd().copy(),
        'gripper_kp': float(gain.gripper_kp),
        'gripper_kd': float(gain.gripper_kd),
    }


def gain_snapshot_to_cli_arrays(snapshot):
    """Return gains as 6 arm joints + gripper arrays."""
    kp = np.concatenate((np.asarray(snapshot['kp'], dtype=float), [snapshot['gripper_kp']]))
    kd = np.concatenate((np.asarray(snapshot['kd'], dtype=float), [snapshot['gripper_kd']]))
    return kp, kd


def format_gain_array(values):
    return np.array2string(np.asarray(values, dtype=float), precision=3, suppress_small=True)


def set_controller_gain_arrays(controller, kp, kd):
    gain = controller.get_gain()
    expected_len = len(gain.kp()) + 1
    kp_arr = np.asarray(kp, dtype=float).reshape(-1)
    kd_arr = np.asarray(kd, dtype=float).reshape(-1)
    if kp_arr.size != expected_len:
        raise ValueError(f"kp must have {expected_len} values (joints + gripper), got {kp_arr.size}")
    if kd_arr.size != expected_len:
        raise ValueError(f"kd must have {expected_len} values (joints + gripper), got {kd_arr.size}")

    gain.kp()[:] = kp_arr[:-1]
    gain.kd()[:] = kd_arr[:-1]
    gain.gripper_kp = float(kp_arr[-1])
    gain.gripper_kd = float(kd_arr[-1])
    controller.set_gain(gain)


def configure_arx_runtime_gains(controller, kp=None, kd=None, label='left'):
    default_snapshot = snapshot_controller_gain(controller)
    default_kp, default_kd = gain_snapshot_to_cli_arrays(default_snapshot)
    print(
        f"[ARX Gains] {label} default/readback kp={format_gain_array(default_kp)}, "
        f"kd={format_gain_array(default_kd)}"
    )

    target_kp = ACTIVE_KP if kp is None else np.asarray(kp, dtype=float)
    target_kd = ACTIVE_KD if kd is None else np.asarray(kd, dtype=float)
    set_controller_gain_arrays(controller, target_kp, target_kd)

    active_kp, active_kd = gain_snapshot_to_cli_arrays(snapshot_controller_gain(controller))
    print(
        f"[ARX Gains] {label} active kp={format_gain_array(active_kp)}, "
        f"kd={format_gain_array(active_kd)}"
    )


def apply_controller_gain_snapshot(controller, snapshot):
    gain = controller.get_gain()
    gain.kp()[:] = snapshot['kp']
    gain.kd()[:] = snapshot['kd']
    gain.gripper_kp = snapshot['gripper_kp']
    gain.gripper_kd = snapshot['gripper_kd']
    controller.set_gain(gain)


def apply_zero_controller_gain(controller):
    gain = controller.get_gain()
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)


def apply_gain_mode(controllers, gain_snapshots, is_double_arm, zero_gains):
    sides = ['left'] + (['right'] if is_double_arm and controllers.get('right') is not None else [])
    if zero_gains:
        for side in sides:
            apply_zero_controller_gain(controllers[side])
            # Hand guiding, as in the recorder: friction feedforward supports the operator.
            controllers[side].set_friction_compensation(True)
        print("[DAGGER] ARX gains set to zero, friction feedforward on; correction mode active.")
        return

    for side in sides:
        # Policy control runs without friction feedforward.
        controllers[side].set_friction_compensation(False)
        apply_controller_gain_snapshot(controllers[side], gain_snapshots[side])
    print("[DAGGER] ARX gains restored, friction feedforward off; policy control active.")


def get_dagger_auto_index(dataset_dir):
    max_idx = 10000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.exists(os.path.join(dataset_dir, f'episode_{i:05d}.zarr')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} correction episodes")


def resolve_dagger_recording_root(root_dir, task_name):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    return os.path.join(root_dir, task_name)


class DaggerCorrectionRecorder:
    """Record shared raw streams, then export task-specific DAgger labels."""

    def __init__(
        self,
        root_dir,
        task_name,
        is_double_arm,
        camera_names,
        fps=None,
        jpeg_quality=95,
        recording_scope='correction_segment',
        jpeg_workers=None,
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.task_name = task_name
        self.dataset_name = task_name
        self.is_double_arm = is_double_arm
        self.camera_names = list(camera_names)
        if not self.camera_names:
            raise ValueError("DAGGER direct recording requires at least one camera")
        self.fps = float(fps if fps is not None else 1.0 / EXECUTION_DT)
        self.jpeg_quality = int(jpeg_quality)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [0, 100], got {jpeg_quality}")
        self.jpeg_workers = (
            DAGGER_JPEG_WORKERS if jpeg_workers is None else int(jpeg_workers)
        )
        if self.jpeg_workers < 1:
            raise ValueError(
                f"jpeg_workers must be at least 1, got {jpeg_workers}"
            )
        self.recording_scope = str(recording_scope)
        self.run_dir = resolve_dagger_recording_root(self.root_dir, self.task_name)
        self.active = False
        self.current_episode_path = None
        self.pending_episode_paths = []
        self.qpos = []
        self.qvel = []
        self.target_action = []
        self.timestamps = []
        self.current_subgoals = []
        self.raw_images = {}
        self.segment_metadata = {}
        self.episode_metadata = {}
        self.capture_enabled = False
        self.capture_segments = []
        self.sample_capture_segment_ids = []
        self.sample_capture_continuity_ids = []
        self._capture_continuity_id = 0
        self._open_capture_segment = None
        self._next_capture_reason = None

    def set_segment_metadata(self, **metadata):
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (str, int, float, bool, list, tuple)):
                cleaned_metadata[key] = value
            else:
                cleaned_metadata[key] = str(value)
        if cleaned_metadata == self.segment_metadata:
            return
        if self.active and self.capture_enabled:
            self._close_capture_segment()
        self.segment_metadata = cleaned_metadata

    def _episode_metadata(self):
        metadata = {
            'source': 'hg_dagger_correction',
            'robot_type': 'arx',
            'robot_model': 'L5',
            'single_arm': not self.is_double_arm,
            'double_arm': self.is_double_arm,
            'camera_names': self.camera_names,
            'fps': self.fps,
            'recording_scope': self.recording_scope,
        }
        metadata.update(self.episode_metadata or self.segment_metadata)
        return metadata

    def _next_episode_path(self):
        episode_idx = get_dagger_auto_index(self.run_dir)
        return os.path.join(self.run_dir, f'episode_{episode_idx:05d}.zarr')

    def start_segment(self, *, capture_reason='demo_start'):
        if self.active:
            return

        os.makedirs(self.run_dir, exist_ok=True)
        self.qpos = []
        self.qvel = []
        self.target_action = []
        self.timestamps = []
        self.current_subgoals = []
        self.raw_images = {camera_name: [] for camera_name in self.camera_names}
        self.episode_metadata = dict(self.segment_metadata)
        self.capture_segments = []
        self.sample_capture_segment_ids = []
        self.sample_capture_continuity_ids = []
        self._capture_continuity_id = 0
        self._open_capture_segment = None
        self._next_capture_reason = str(capture_reason)
        self.current_episode_path = self._next_episode_path()
        self.episode_buffer = standard_recording.EpisodeBuffer(
            camera_names=self.camera_names, directory=self.run_dir, fps=self.fps,
            is_double_arm=self.is_double_arm)
        self.raw_images = {name: standard_recording.RecordedImages(self.episode_buffer, name)
                           for name in self.camera_names}
        self.active = True
        self.capture_enabled = True
        label = (
            "stitched demo episode"
            if self.recording_scope == 'full_demo_stitched'
            else "correction segment"
        )
        print(f"[DAGGER] Started {label}: {self.current_episode_path}")

    def _close_capture_segment(self):
        if self._open_capture_segment is None:
            return
        self._open_capture_segment['raw_end_index_exclusive'] = len(self.qpos)
        if (
            self._open_capture_segment['raw_end_index_exclusive']
            > self._open_capture_segment['raw_start_index']
        ):
            self.capture_segments.append(self._open_capture_segment)
        self._open_capture_segment = None

    def pause_capture(self, *, reason='paused'):
        if not self.active or not self.capture_enabled:
            return
        self._close_capture_segment()
        self.capture_enabled = False
        self._next_capture_reason = None
        print(f"[DAGGER] Demo capture paused ({reason}); episode remains open.")

    def resume_capture(self, *, reason='resumed'):
        if not self.active or self.capture_enabled:
            return
        self._capture_continuity_id += 1
        self.capture_enabled = True
        self._next_capture_reason = str(reason)
        print(f"[DAGGER] Demo capture resumed ({reason}).")

    def _ensure_capture_segment(self):
        if self._open_capture_segment is not None:
            return
        segment_id = len(self.capture_segments)
        self._open_capture_segment = {
            'segment_index': segment_id,
            'raw_start_index': len(self.qpos),
            'raw_end_index_exclusive': None,
            'reason': self._next_capture_reason or 'metadata_change',
            'metadata': dict(self.segment_metadata),
        }
        self._next_capture_reason = None

    def _copy_sample_images(self, images):
        if images is None:
            raise ValueError("DAGGER direct recording requires synchronized camera images")
        missing = [camera_name for camera_name in self.camera_names if camera_name not in images]
        if missing:
            raise KeyError(f"DAGGER image sample is missing cameras: {missing}")

        raw_sample = {}
        for camera_name in self.camera_names:
            image = np.asarray(images[camera_name])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"DAGGER {camera_name} must have HWC three-channel shape, got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(f"DAGGER {camera_name} must use uint8, got {image.dtype}")
            raw_sample[camera_name] = np.array(
                image,
                dtype=np.uint8,
                copy=True,
                order='C',
            )
        return raw_sample

    def append_observation(self, observation, target_action):
        if not self.active or not self.capture_enabled:
            return
        self._ensure_capture_segment()
        standard_recording.append_observation(self.episode_buffer, observation, target_action)
        qpos, qvel, timestamp = observation.qpos, observation.qvel, observation.robot_timestamp
        self.qpos.append(np.asarray(qpos, dtype=np.float64).copy())
        if qvel is not None:
            self.qvel.append(np.asarray(qvel, dtype=np.float64).copy())
        self.target_action.append(np.asarray(target_action, dtype=np.float64).copy())
        self.timestamps.append(float(timestamp))
        self.current_subgoals.append(int(self.segment_metadata.get('current_subgoal', -1)))
        self.sample_capture_segment_ids.append(
            int(self._open_capture_segment['segment_index'])
        )
        self.sample_capture_continuity_ids.append(self._capture_continuity_id)
    def append_sample(self, *args, **kwargs):
        raise ValueError('Use append_observation with standard_capture.capture_observation')

    def append_controller_sample(self, *args, **kwargs):
        raise ValueError('Do not reread proprio after image selection; use append_observation')

    @staticmethod
    def _write_metadata_attr(root, key, value):
        if value is None:
            return
        if isinstance(value, (str, int, float, bool, np.generic)):
            root.attrs[key] = value
            return
        array = np.asarray(value)
        if array.dtype.kind in {'b', 'i', 'u', 'f'}:
            root.attrs[key] = array
        else:
            root.attrs[key] = json.dumps(value)

    def _write_episode(self):
        from data_processing.episode_store import publish_episode
        if self.current_episode_path is None:
            return False
        self._close_capture_segment()
        if len(self.qpos) < 2:
            return False
        qpos = np.asarray(self.qpos, dtype=np.float32)
        stamps = np.asarray(self.timestamps, dtype=np.float64)
        continuity = np.asarray(self.sample_capture_continuity_ids, dtype=np.int64)
        segments = np.asarray(self.sample_capture_segment_ids, dtype=np.int64)
        if continuity.shape != (len(qpos),) or segments.shape != (len(qpos),):
            raise ValueError('DAgger sample/continuity/segment counts differ')
        indices = np.flatnonzero(continuity[:-1] == continuity[1:])
        if not len(indices):
            return False
        arrays = {
            'action': qpos[indices + 1],
            'action_timestamps': stamps[indices + 1],
            'policy_target_action': np.asarray(self.target_action, dtype=np.float32)[indices],
            'continuity_id': continuity[indices],
            'observations/capture_segment_index': segments[indices],
            'observations/capture_continuity_index': continuity[indices],
            'action_raw': qpos,
            'action_raw_timestamps': stamps,
            'action_raw_capture_segment_index': segments,
            'action_raw_capture_continuity_index': continuity,
        }
        metadata = dict(self._episode_metadata(), sim=False, rollout=False, dagger=True,
                        has_raw_actions=True, image_color_space='BGR',
                        action_label_source='next_measured_qpos_within_continuity')
        # Subtask labels are optional; generic DAgger does not declare subtasks.
        if any(value >= 0 for value in self.current_subgoals):
            arrays['observations/current_subgoal'] = np.asarray(self.current_subgoals, dtype=np.int64)[indices]
        offset = 0
        stitched = []
        for segment in self.capture_segments:
            count = int(np.count_nonzero(segments[indices] == segment['segment_index']))
            stitched.append(dict(segment, start_index=offset, end_index_exclusive=offset+count))
            offset += count
        metadata['dagger_segment_count'] = len(stitched)
        for name in ('start_index', 'end_index_exclusive', 'raw_start_index', 'raw_end_index_exclusive', 'reason'):
            arrays[f'dagger_segments/{name}'] = np.asarray([segment[name] for segment in stitched])
        arrays['dagger_segments/metadata_json'] = np.asarray([json.dumps(segment['metadata']) for segment in stitched])
        if 'openpi_prompt' in metadata:
            arrays['openpi_prompt'] = np.asarray(str(metadata['openpi_prompt']))
        if os.path.exists(self.current_episode_path):
            self.current_episode_path = self._next_episode_path()
        publish_episode(self.current_episode_path,
            [(self.episode_buffer.data_dir, [self.episode_buffer[int(i)].robot_index for i in indices])],
            metadata=metadata, arrays=arrays)
        print(f'[DAGGER] Saved {len(indices)} transitions with original stream references: {self.current_episode_path}')
        return True

    def stop_segment(self):
        if not self.active:
            return None

        self.capture_enabled = False
        self.episode_buffer.finish()
        wrote_data = self._write_episode()
        episode_path = self.current_episode_path
        if wrote_data and episode_path is not None:
            self.pending_episode_paths.append(episode_path)
            label = (
                "stitched demo episode"
                if self.recording_scope == 'full_demo_stitched'
                else "correction segment"
            )
            print(f"[DAGGER] Finished {label}: {episode_path}")

        self.active = False
        self.current_episode_path = None
        self.qpos = []
        self.qvel = []
        self.target_action = []
        self.timestamps = []
        self.current_subgoals = []
        self.raw_images = {}
        self.episode_metadata = {}
        self.capture_segments = []
        self.sample_capture_segment_ids = []
        self.sample_capture_continuity_ids = []
        self._capture_continuity_id = 0
        self._open_capture_segment = None
        self._next_capture_reason = None
        return episode_path if wrote_data else None

    def has_pending_data(self):
        return self.active or len(self.pending_episode_paths) > 0

    def keep_pending(self):
        count = len(self.pending_episode_paths)
        self.pending_episode_paths = []
        print(f"[DAGGER] Kept {count} episode(s).")

    def discard_pending(self):
        count = len(self.pending_episode_paths)
        for episode_path in self.pending_episode_paths:
            if os.path.isdir(episode_path):
                from recording.recording_utils import atomic_write_json
                path = Path(episode_path) / 'manifest.json'
                manifest = json.loads(path.read_text())
                manifest.update(retained=False, decision='discard')
                atomic_write_json(path, manifest)
        self.pending_episode_paths = []
        print(f"[DAGGER] Discarded {count} episode(s).")


def prompt_save_dagger_corrections(dagger_recorder):
    global dagger_prompt_active, dagger_prompt_key_queue
    if dagger_recorder is None or not dagger_recorder.has_pending_data():
        return
    if dagger_recorder.active:
        dagger_recorder.stop_segment()
    if len(dagger_recorder.pending_episode_paths) == 0:
        return

    def clear_prompt_keys():
        while True:
            try:
                dagger_prompt_key_queue.get_nowait()
            except Empty:
                break

    save_keys = {DAGGER_RECORD_KEY, 'b'}
    discard_keys = {DAGGER_EXIT_KEY, 'c'}
    print("\n[DAGGER] Keep recorded DAgger episode? Press 2 to save/keep, 3 to discard.")
    print("[DAGGER] Keyboard aliases: b saves, c discards.")
    clear_prompt_keys()
    dagger_prompt_active = True
    try:
        while True:
            try:
                key = dagger_prompt_key_queue.get().lower()
                if key in save_keys:
                    dagger_recorder.keep_pending()
                    return
                if key in discard_keys:
                    dagger_recorder.discard_pending()
                    return
                print("[DAGGER] Unknown key. Press 2 to save/keep, 3 to discard.")
            except (KeyboardInterrupt, EOFError):
                print("[DAGGER] Interrupted; discarding pending correction data.")
                dagger_recorder.discard_pending()
                return
    finally:
        dagger_prompt_active = False
        clear_prompt_keys()


@standard_recording.recording_scope
def eval_bc(config, ckpt_name, save_episode=False, num_rollouts=50, task_config=None, use_rerun=False,
            query_frequency=None, first_chunk_size=None, temporal_agg=False, masking_strategy=None,
            history_len=0, use_camera_server=True, action_lookahead=None, gripper_lookahead=None,
            execution_dt=None, use_fill_in=False, blend_actions_enabled=True, mixing_steps=5,
            inference_on_refresh=False, compress_inference_images=False,
            inference_jpeg_quality=50, record_rollout=False, rollout_dir=None,
            rollout_task_name=None, nn_chunk_search=False,
            record_dagger_corrections=False, dagger_dir=None,
            dagger_task_name=None,
            arm_kp=None, arm_kd=None, start_position_path=DEFAULT_START_POSITION_PATH,
            action_command_style_override=None, eval_mode=False, blending_method='sigmoid',
            rtc_prefix_steps=0, fast_inference=False, timing=False, timing_interval=500):
    global continue_flag, pause_flag, exit_requested, reset_ready_requested, dagger_state, EXECUTION_DT
    global lookahead_tuner
    check_inference_camera_settings(config)
    if record_rollout and record_dagger_corrections:
        raise ValueError('Use inference_remote for simultaneous rollout and DAgger raw recording')
    # Get ckpt_dir from config
    ckpt_dir = config['ckpt_dir']

    from contracts.robot_contract import metadata_robot_contract, require_robot_match
    if metadata_robot_contract(config) is not None:
        roles = ['left', 'right'] if not task_config.get('single_arm', True) else [task_config.get('single_arm_side') or 'left']
        require_robot_match({'robot_contract_version': 1, 'arms': {role: {'model': 'L5'} for role in roles}},
                            config, context='direct ARX execution')
    aloha = task_config.get('aloha', False)
    camera_names = resolve_base_camera_names(config, task_config)
    model_camera_names = resolve_model_camera_names(config, task_config)
    standard_image_processing.require_single_frame(config, task_config)
    
    # Determine if this is a double arm policy based on action dimension
    action_dim = task_config['action_dim']
    is_double_arm = action_dim > 7  # Single arm has 7 DOF (6 joints + 1 gripper)
    if is_double_arm:
        single_arm_side = None
        single_arm_can = None
        single_arm_side_source = None
    else:
        single_arm_side, single_arm_can, single_arm_side_source = resolve_single_arm_runtime_side_and_can(
            config,
            task_config,
        )
    print(f"{'Double' if is_double_arm else 'Single'} arm configuration detected (action_dim: {action_dim})")
    action_command_style = require_absolute_actions(
        config, task_config, {'action_command_style': action_command_style_override})
    print(f"[Action Command] style={action_command_style}")
    if not is_double_arm:
        print(
            f"[ARX] Single-arm side={single_arm_side or 'legacy/unspecified'} "
            f"(source={single_arm_side_source}), CAN={single_arm_can}"
        )
    action_fps = resolve_policy_rate(config,
        None if execution_dt is None else 1.0 / execution_dt,
        legacy_rate=config.get('legacy_policy_fps'), context='checkpoint')
    execution_dt = EXECUTION_DT = 1.0 / action_fps
    if use_fill_in:
        raise ValueError('Use native SDK interpolation; disable use_fill_in to preserve the policy action rate')
    if query_frequency is not None and query_frequency <= 0:
        raise ValueError(f"query_frequency must be > 0, got {query_frequency}")
    if first_chunk_size is not None and first_chunk_size <= 0:
        raise ValueError(f"first_chunk_size must be > 0, got {first_chunk_size}")
    # if action_lookahead < 10:
    #     raise ValueError(f"action_lookahead must be >= 10, got {action_lookahead}")
    if gripper_lookahead is not None and gripper_lookahead < 0:
        raise ValueError(f"gripper_lookahead must be >= 0, got {gripper_lookahead}")
    if inference_on_refresh:
        if not use_camera_server:
            raise ValueError(
                "--inference_on_refresh requires --use-camera-server because frame numbers "
                "come from shared-memory camera_server metadata."
            )
        if 'video_overhead' not in camera_names:
            raise ValueError(
                "--inference_on_refresh requires video_overhead in task camera_names."
            )
    if compress_inference_images and not (0 <= inference_jpeg_quality <= 100):
        raise ValueError(
            f"--inference_jpeg_quality must be in [0, 100], got {inference_jpeg_quality}"
        )
    rollout_camera_names = list(camera_names)
    rollout_timestamp_reference_camera = get_timestamp_reference_camera(rollout_camera_names)
    record_rollout = bool(record_rollout or eval_mode)
    rollout_date = rollout_date_string() if eval_mode else None
    if record_rollout:
        if rollout_dir is None:
            rollout_dir = str(ROLLOUT_DIR)
        if rollout_task_name is None:
            rollout_task_name_actual = os.path.basename(ckpt_dir).replace('base_', '').split('_ACT')[0]
        else:
            rollout_task_name_actual = rollout_task_name
        rollout_subdir_name = ''
        rollout_root = resolve_rollout_recording_root(
            rollout_dir,
            rollout_task_name_actual,
            eval_mode=eval_mode,
            date_string=rollout_date,
        )
        print(
            f"[Rollout Recording] Enabled - saving to "
            f"{rollout_root}/"
        )
    else:
        rollout_task_name_actual = None
        rollout_subdir_name = None

    if record_dagger_corrections:
        if dagger_dir is None:
            dagger_dir = str(DAGGER_DIR)
        if dagger_task_name is None:
            dagger_task_name_actual = (
                rollout_task_name
                if rollout_task_name is not None
                else os.path.basename(ckpt_dir).replace('base_', '').split('_ACT')[0]
            )
        else:
            dagger_task_name_actual = dagger_task_name
        print(
            f"[DAGGER] Correction recording enabled - saving to "
            f"{resolve_dagger_recording_root(dagger_dir, dagger_task_name_actual)}/"
        )
    else:
        dagger_task_name_actual = None

    # Setup cameras and robot
    stop_event = setup_cameras_and_robot(
        is_double_arm=is_double_arm,
        use_camera_server=use_camera_server,
        camera_names=camera_names,
    )
    # Initialize robot(s)
    print(f'========================ROBOT START========================:')
    robot_config, arx_L, arx_R = init_arm(
        is_double_arm=is_double_arm,
        kp=arm_kp,
        kd=arm_kd,
        single_arm_can=single_arm_can,
        single_arm_side=single_arm_side,
        can_interface_left=task_config.get('can_interface_left'),
        can_interface_right=task_config.get('can_interface_right'),
    )

    # Create controller and command dictionaries
    controllers = {'left': arx_L}
    commands = {'left': arx5.JointState(robot_config.joint_dof)}

    if is_double_arm:
        controllers['right'] = arx_R
        commands['right'] = arx5.JointState(robot_config.joint_dof)

    start_action = load_start_action_from_npz(
        start_position_path,
        is_double_arm=is_double_arm,
        action_dim=action_dim,
        closed_gripper_pos=GRIPPER_CLOSED_POS,
        single_arm_side=single_arm_side,
        # Takes start from the saved start pose, gripper included (pedal start).
        use_saved_gripper_positions=True,
    )

    runtime_gain_snapshots = None
    dagger_recorder = None
    dagger_gains_zeroed = False
    if record_dagger_corrections:
        runtime_gain_snapshots = {'left': snapshot_controller_gain(controllers['left'])}
        if is_double_arm and controllers.get('right') is not None:
            runtime_gain_snapshots['right'] = snapshot_controller_gain(controllers['right'])
        dagger_recorder = DaggerCorrectionRecorder(
            dagger_dir,
            dagger_task_name_actual,
            is_double_arm,
            camera_names,
            fps=1.0 / execution_dt,
        )

    move_to_start_action(
        controllers,
        commands,
        start_action,
        execution_dt=execution_dt,
        is_double_arm=is_double_arm,
    )
    
    with idle_hold(controllers):
        # Keep sending hold-position commands while paused.
        while not have_required_camera_frames(camera_names):
            time_V2.sleep(0.001)
        
        # Initialize rerun if requested
        if use_rerun:
            setup_rerun_visualization(is_double_arm)
        
        # Load policy and stats
        policy, stats, policy_config = load_policy_and_stats(config, ckpt_name, fast_inference)
    
    if stats['qpos_mean'].shape[0] > task_config['state_dim']:
        # Trim qpos_mean and qpos_std to match state_dim
        stats['qpos_mean'] = stats['qpos_mean'][:task_config['state_dim']]
        stats['qpos_std'] = stats['qpos_std'][:task_config['state_dim']]
    
    # Setup preprocessing and postprocessing functions
    pre_process = lambda s_qpos: (s_qpos - torch.from_numpy(stats['qpos_mean'])) / torch.from_numpy(stats['qpos_std'])
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    if rtc_prefix_steps:
        # Training-time RTC, always asynchronous as in remote mode: each query after the
        # first is sent rtc_prefix_steps (the checkpoint's trained maximum delay) before
        # its chunk boundary with the plan's next rtc_prefix_steps actions pinned as a
        # clean prefix; the arm executes them meanwhile and the new chunk takes over
        # after the prefix (DirectRTCQueries). It replaces action_lookahead, so the two
        # are mutually exclusive.
        if action_lookahead is not None:
            raise ValueError('action_lookahead and RTC are mutually exclusive; RTC executes its prefix first')
        # The pinned prefix is whole plan rows, so the gripper must execute the same row as the arm:
        # a gripper lookahead would send other gripper actions than the ones the policy was told.
        if gripper_lookahead not in (None, 0):
            raise ValueError('gripper_lookahead and RTC are mutually exclusive; the pinned prefix is what is sent')
        gripper_lookahead = 0
        if not getattr(policy, 'training_time_rtc', False) or not hasattr(policy, 'forward_inf_training_time_rtc'):
            raise ValueError('Direct RTC requires a Flow or ACT policy trained with training-time RTC')
        if rtc_prefix_steps > policy.training_time_rtc_max_delay:
            raise ValueError(f'RTC prefix {rtc_prefix_steps} exceeds the trained maximum delay {policy.training_time_rtc_max_delay}')
        if nn_chunk_search or temporal_agg:
            raise ValueError('Direct RTC pins the previous plan; disable nn_chunk_search and temporal_agg')
        if blend_actions_enabled and mixing_steps <= 0:
            raise ValueError(f'Blending window {mixing_steps} must be positive')
        print(f'[RTC] Conditioning each chunk on the previous plan\'s next {rtc_prefix_steps} actions')
    else:
        action_lookahead = 10 if action_lookahead is None else action_lookahead
        gripper_lookahead = 5 if gripper_lookahead is None else gripper_lookahead
    rtc_normalize = lambda a: (a - stats['action_mean']) / stats['action_std']

    # Add after loading stats (around line 563)
    print("=== Dataset Statistics Check ===")
    print(f"qpos_mean: {stats['qpos_mean']}")
    print(f"qpos_std: {stats['qpos_std']}")
    print(f"action_mean: {stats['action_mean']}")
    print(f"action_std: {stats['action_std']}")
        
    if query_frequency is None:
        query_frequency = 100
    if first_chunk_size is None:
        first_chunk_size = query_frequency
    
    num_queries = policy_config['num_queries']  # Number of actions predicted by policy
    # RTC executes its pinned prefix instead of a lookahead, so there is nothing to tune.
    lookahead_tuner = None if rtc_prefix_steps else lookahead_tuning.LookaheadTuner(
        action_lookahead, gripper_lookahead, chunk_steps=max(query_frequency, first_chunk_size),
        horizon=num_queries, fps=1.0 / execution_dt)
    from inference.policy_transport import TimingStats
    timing_stats = TimingStats(enabled=timing, interval=timing_interval)  # remote mode's --timing report
    query_count = 0
    rtc_queries = None

    def recapture_query_observation(t):
        """Capture and preprocess step ``t``'s query observation again, as the query step does, so the
        query never uses inputs that aged meanwhile (step 0's capture precedes warm-up). Everything
        downstream (query, recording, logs) then uses the new observation."""
        nonlocal captured_observation, obs, qpos_numpy, current_qvel, raw_image_dict, image_dict, qpos, curr_image
        nonlocal query_overhead_frame_number, image_timestamp_dict
        captured_observation = capture_current_observation(config, controllers, state_dim, camera_names)
        obs, current_qvel = captured_observation.state, captured_observation.qvel
        qpos_numpy = standard_capture.policy_state(captured_observation, task_config, state_dim)[:state_dim]
        if current_qvel is not None:
            current_qvel = current_qvel[:state_dim]
        image_dict = standard_capture.resized_images(captured_observation, config)
        if not DEBUG and 'video_overhead' in camera_names:
            query_overhead_frame_number = int(captured_observation.camera_observation[
                'cameras']['video_overhead']['frame_number'])
        raw_image_dict = {name: image.copy() for name, image in image_dict.items()}
        if compress_inference_images:
            image_dict = jpeg_compress_image_dict(image_dict, jpeg_quality=inference_jpeg_quality)
        if record_rollout:
            image_timestamp_dict = get_current_image_timestamps(rollout_camera_names)
        qpos_history_raw[t] = qpos_numpy
        qpos = torch.from_numpy(qpos_numpy).float()
        image_dict, qpos = masker.mask(image_dict, qpos)
        if not masker.masks_proprioception:
            qpos = pre_process(qpos)
        if history_len > 0:
            history = np.zeros((history_len, state_dim))
            valid = min(t, history_len)
            history[:valid] = qpos_history_raw[t - valid:t][::-1]
            history = torch.from_numpy(history).float()
            for i in range(valid):
                history[i] = pre_process(history[i])
            qpos = torch.cat([qpos.unsqueeze(0), history], dim=0).view(1, -1)
        else:
            qpos = qpos.unsqueeze(0)
        qpos = qpos.cuda()
        curr_image = process_image(image_dict, grayscale=grayscale, is_double_arm=is_double_arm, aloha=aloha,
                                   camera_names=camera_names)

    if rtc_prefix_steps:
        # Chunks after the first run from index rtc_prefix_steps; blending reads the previous
        # plan after its next prefix.
        covered = max(first_chunk_size, query_frequency + rtc_prefix_steps)
        if blend_actions_enabled and covered + mixing_steps > num_queries:
            raise ValueError(f'Previous plan of {num_queries} actions cannot cover {covered} executed/prefix '
                             f'actions and {mixing_steps} blended actions')
        rtc_queries = DirectRTCQueries(policy, rtc_prefix_steps, execution_dt, first_chunk_size, query_frequency,
                                       timing_stats=timing_stats)
    if temporal_agg:
        query_frequency = 1
        first_chunk_size = 1
        print(f"Temporal aggregation enabled. Query frequency set to 1 frame, num_queries: {num_queries}")
    else:
        print(f"Using query frequency: {query_frequency} frames, first chunk size: {first_chunk_size} frames")

    max_timesteps = int(config['episode_len'] * 1)*FACTOR # may increase for real-world tasks

    # Initialize action based on configuration
    if is_double_arm:
        action = [0]*action_dim  # Full action dimension for double arm
    else:
        action = [0]*7  # Single arm: 6 joints + 1 gripper
    action = np.array(action)[None]
    
    state_dim = task_config['state_dim']
    action_idx = policy_config['action_idx'] if 'action_idx' in policy_config.keys() else np.arange(action_dim).tolist()
    state_idx = policy_config['state_idx'] if 'state_idx' in policy_config.keys() else np.arange(state_dim).tolist()
    grayscale = config['grayscale'] if 'grayscale' in config.keys() else False
    print('======================================================')
    print("grayscale",grayscale)
    print('======================================================')
    
    # The checkpoint's training mask, applied to the policy's images and qpos exactly as in training.
    print(f'Initializing Masker: {masking_strategy}')
    masker = Masker(masking_strategy=masking_strategy)
    # Same strategy definition and calibration as the checkpoint was trained with, or refuse.
    require_masking_contract(config.get('task_config', {}).get('masking_contract'), masking_strategy)
    
    # Initialize commandline input thread flag
    commandline_thread_started = False

    links, joints = parse_urdf(urdf_path)
    
    # Initialize temporal aggregation variables
    if temporal_agg:
        all_time_actions, action_timestamps = setup_temporal_aggregation(num_queries, max_timesteps, action_dim)
    else:
        all_time_actions, action_timestamps = None, None
    episode_metadata = lambda rollout_id, demo_id: {"task_name": dagger_task_name_actual}
    
    failed = False
    home_reader = None

    def damping_home_requested():
        # Keys already belong to the command-line thread: route them to its
        # prompt queue while damping; before it starts, read the terminal.
        global dagger_prompt_active
        nonlocal home_reader
        if not commandline_thread_started:
            if home_reader is None:
                from hardware.safe_exit import HomeKeyReader
                home_reader = HomeKeyReader()
            return home_reader.home.is_set()
        dagger_prompt_active = True
        while True:
            try:
                if dagger_prompt_key_queue.get_nowait() == 'h':
                    dagger_prompt_active = False
                    return True
            except Empty:
                return False

    # Console output and per-step targets/measured state for offline diagnosis (logs/inference/).
    import atexit
    from utils.run_log import RunLog as DirectRunLog
    run_log = DirectRunLog(os.path.basename(os.path.normpath(ckpt_dir)))  # e.g. single_arm_flow
    run_log.write_settings(task=config.get("task_name"), ckpt_name=ckpt_name, rtc_prefix_steps=rtc_prefix_steps,
                           chunk_steps=query_frequency, first_chunk_steps=first_chunk_size,
                           action_lookahead=action_lookahead, gripper_lookahead=gripper_lookahead,
                           blending=blending_method if blend_actions_enabled else 'none', mixing_steps=mixing_steps,
                           num_inference_timesteps=policy_config.get('num_inference_timesteps'),
                           fast_inference=fast_inference, execution_dt=execution_dt)
    atexit.register(run_log.close)  # also on Ctrl+C or a failure that ends the process
    try:
        # Any failure, including Ctrl+C: native ARX damping until 'h', then home.
        with damping_exit(lambda: damping_target(controllers),
                          home_requested=damping_home_requested):
            for rollout_id in range(num_rollouts):
                print('======================================================')
                print(f'Rollout {rollout_id} start')
                print('======================================================')
                if dagger_recorder is not None:
                    dagger_recorder.set_segment_metadata(
                        **episode_metadata(rollout_id, 0),
                        rollout_id=rollout_id,
                        demo_id=0,
                    )
                print('Press x to exit, press any key to continue')
                if exit_requested:
                    break
                try:
                    with idle_hold(controllers):
                        var = getkey()
                    if var.lower() == 'x':
                        print("Exit requested before rollout start.")
                        exit_requested = True
                        break
                except EOFError:
                    print("\nInput closed before rollout start.")
                    break
            
                # Start commandline input thread after first getkey interaction
                if not commandline_thread_started:
                    threading.Thread(target=commandline_input, daemon=True).start()
                    commandline_thread_started = True
        
                rollout_id += 0
            

                for _ in range(3):
                    demo_id = 0
                    qpos_history_raw = np.zeros((max_timesteps, state_dim))
                    image_list = [] # for visualization
                    qpos_list = []
                    target_qpos_list = []
                    action = start_action.copy() if start_action is not None else np.zeros((1, action_dim))
                    prev_action = action.copy()
                    prev_prev_action = action.copy()
                    prev_chunk_size = first_chunk_size
                    # No chunk boundary precedes a first chunk: it runs from its first action.
                    current_chunk_start_index = 0
                    rtc_plan_valid = False  # no previous plan yet: the first query has no prefix
                    current_qvel = None
                    if record_rollout:
                        rollout_lists = create_rollout_lists(rollout_camera_names)
                    if record_dagger_corrections:
                        dagger_state = DAGGER_STATE_POLICY
                        dagger_gains_zeroed = False
                
                    t = 0
                    # Chunk schedule origin: direct RTC restarts it, as remote mode resets its session,
                    # when policy control resumes after a pause or hand correction.
                    chunk_origin, rtc_fresh_start = 0, False
                    culmulated_delay = 0

                    while not exit_requested:
                        if continue_flag and (not pause_flag):
                            if record_rollout:
                                standard_recording.ensure_rollout_started(rollout_lists,
                                    os.path.join(rollout_dir, task_name), rollout_camera_names,
                                    1.0 / execution_dt, is_double_arm)
                            if (record_dagger_corrections and dagger_state == DAGGER_STATE_RECORDING
                                    and dagger_recorder is not None and not dagger_recorder.active):
                                dagger_recorder.start_segment()
                            with torch.inference_mode():
                                if rtc_queries is not None and rtc_fresh_start and not (
                                        record_dagger_corrections
                                        and dagger_state in {DAGGER_STATE_GRAVITY_COMP, DAGGER_STATE_RECORDING}):
                                    # As in remote mode, policy control resuming after a pause or hand
                                    # correction starts a fresh chunk here: an immediate query without
                                    # an RTC prefix (any query still running is discarded).
                                    chunk_origin, rtc_fresh_start, rtc_plan_valid = t, False, False
                                chunk_t = t - chunk_origin
                                if rtc_queries is not None:
                                    # should_query: a policy observation is taken this step (at
                                    # submission for asynchronous RTC queries, merged later).
                                    should_query, rtc_submit, rtc_merge = rtc_queries.schedule(chunk_t, rtc_plan_valid)
                                    if not rtc_merge:
                                        rtc_queries.poll(chunk_t)  # remote mode's 'received' line
                                else:
                                    should_query = is_query_point(chunk_t, first_chunk_size, query_frequency)
                                    rtc_submit = rtc_merge = False
                                image_timestamp_dict = {}
                                query_overhead_frame_number = -1

                                if should_query:
                                    if inference_on_refresh:
                                        current_overhead_frame_number = camera_capture.get_latest_camera_frame_number(
                                            'video_overhead'
                                        )
                                        while True:
                                            fresh_overhead_frame_number = camera_capture.get_latest_camera_frame_number(
                                                'video_overhead'
                                            )
                                            if fresh_overhead_frame_number != current_overhead_frame_number:
                                                break
                                            time_V2.sleep(0.001)
                                        query_overhead_frame_number = int(fresh_overhead_frame_number)
                                    elif use_camera_server and 'video_overhead' in camera_names:
                                        query_overhead_frame_number = int(
                                            camera_capture.get_latest_camera_frame_number('video_overhead')
                                        )

                                    # Get observations
                                    captured_observation = capture_current_observation(config, controllers, state_dim, camera_names)
                                    obs = captured_observation.state
                                    qpos_numpy = standard_capture.policy_state(captured_observation, task_config, state_dim)
                                    current_qvel = captured_observation.qvel

                                    if qpos_numpy.shape[0] > state_dim:
                                        qpos_numpy = qpos_numpy[:state_dim]
                                    if current_qvel.shape[0] > state_dim:
                                        current_qvel = current_qvel[:state_dim]
                                    # Get images
                                    image_dict = standard_capture.resized_images(captured_observation, config)
                                    if not DEBUG and 'video_overhead' in camera_names:
                                        query_overhead_frame_number = int(captured_observation.camera_observation[
                                            'cameras']['video_overhead']['frame_number'])
                                    raw_image_dict = {
                                        cam_name: image.copy()
                                        for cam_name, image in image_dict.items()
                                    }
                                    if compress_inference_images:
                                        image_dict = jpeg_compress_image_dict(
                                            image_dict,
                                            jpeg_quality=inference_jpeg_quality,
                                        )
                                    if record_rollout:
                                        image_timestamp_dict = get_current_image_timestamps(rollout_camera_names)

                                    qpos_history_raw[t] = qpos_numpy
                                    qpos = torch.from_numpy(qpos_numpy).float()

                                    image_dict, qpos = masker.mask(image_dict, qpos)

                                    if not masker.masks_proprioception:
                                        qpos = pre_process(qpos)
                                
                                    if history_len > 0:
                                        # Add history to qpos
                                        qpos_history = np.zeros((history_len, state_dim))
                                        valid_history_len = min(t, history_len)
                                        qpos_history[:valid_history_len] = qpos_history_raw[t-valid_history_len:t][::-1]
                                        qpos_history = torch.from_numpy(qpos_history).float()
                                        # Only pre-process the valid history values (up to min(t, history_len))
                                        for i in range(valid_history_len):
                                            qpos_history[i] = pre_process(qpos_history[i])
                                    
                                        qpos = torch.cat([qpos.unsqueeze(0), qpos_history], dim=0)  # Concatenate history with current state
                                        # flatten qpos to match policy input
                                        qpos = qpos.view(1, -1)  # Shape: (1, history_len * state_dim + state_dim)
                                    else:
                                        qpos = qpos.unsqueeze(0)


                                    qpos = qpos.cuda()

                                    curr_image = process_image(
                                        image_dict,
                                        grayscale=grayscale,
                                        is_double_arm=is_double_arm,
                                        aloha=aloha,
                                        camera_names=camera_names,
                                    )
                                    expected_num_views = len(model_camera_names)
                                    assert curr_image.shape == (1, expected_num_views, 3, config['height'], config['width']), \
                                        f"Expected shape (1, {expected_num_views}, 3, {config['height']}, {config['width']}), got {curr_image.shape}"
                                elif record_rollout or (
                                    record_dagger_corrections and (
                                        dagger_state != DAGGER_STATE_POLICY
                                        or (dagger_recorder is not None and dagger_recorder.active)
                                    )
                                ):
                                    captured_observation = capture_current_observation(config, controllers, state_dim, camera_names)
                                    obs = captured_observation.state
                                    qpos_numpy = standard_capture.policy_state(captured_observation, task_config, state_dim)
                                    current_qvel = captured_observation.qvel
                                    if qpos_numpy.shape[0] > state_dim:
                                        qpos_numpy = qpos_numpy[:state_dim]
                                    if current_qvel is not None and current_qvel.shape[0] > state_dim:
                                        current_qvel = current_qvel[:state_dim]
                                    image_dict = standard_capture.resized_images(captured_observation, config)
                                    raw_image_dict = {
                                        cam_name: image.copy()
                                        for cam_name, image in image_dict.items()
                                    }
                                    if compress_inference_images:
                                        image_dict = jpeg_compress_image_dict(
                                            image_dict,
                                            jpeg_quality=inference_jpeg_quality,
                                        )
                                    if record_rollout:
                                        image_timestamp_dict = get_current_image_timestamps(rollout_camera_names)

                                if t == 0:
                                    # Warm up the policy
                                    with idle_hold(controllers):
                                        warm_up_policy(policy, qpos, curr_image, rtc_prefix_steps, execution_dt)
                                        if rtc_queries is not None:
                                            rtc_queries.warm_up(qpos, curr_image)
                                    if should_query:
                                        # Warm-up took a while since this step's capture: query fresh inputs.
                                        recapture_query_observation(t)

                                time1 = time_V2.time() 

                                ### query policy
                                # Flow and ACT share chunked execution; rtc_prefix_steps is nonzero only for
                                # training-time RTC checkpoints (checked against the checkpoint).
                                if config['policy_class'] in ('Flow', 'ACT'):
                                    current_chunk_size = first_chunk_size if chunk_t == 0 else query_frequency
                                    if rtc_submit:
                                        # The plan's next actions from now, from the unblended prediction:
                                        # blending never reaches the policy. The arm keeps executing them.
                                        planned = rtc_planned_prefix(
                                            rtc_plan, current_chunk_start_index,
                                            get_chunk_step_offset(chunk_t, first_chunk_size, query_frequency),
                                            rtc_prefix_steps)
                                        rtc_queries.submit(
                                            qpos, curr_image,
                                            torch.from_numpy(rtc_normalize(planned)).float().to(curr_image.device),
                                            chunk_t)
                                        query_count += 1
                                    if should_query:
                                        # For the handoff report: how far the arm trails its last target,
                                        # and how old the query's images are.
                                        query_lag = np.asarray(prev_action, dtype=float).reshape(-1)[:len(qpos_numpy)] - qpos_numpy
                                        cameras = captured_observation.camera_observation.get('cameras', {})
                                        query_image_age = (1000 * (captured_observation.robot_timestamp - min(
                                            camera['capture_timestamp'] for camera in cameras.values()))
                                            if cameras else None)
                                    if (should_query and not rtc_submit) or rtc_merge:
                                        start_time = time_V2.time()

                                        rtc_prefix_length = 0
                                        if rtc_merge:
                                            new_action_chunk, rtc_prefix_length = rtc_queries.take(chunk_t)
                                        else:
                                            if rtc_queries is not None:
                                                # A fresh start: no prefix, and nothing else may use the policy.
                                                rtc_queries.discard()
                                            new_action_chunk = policy.forward_inf(qpos, curr_image).cpu().numpy()
                                            query_count += 1
                                            timing_stats.record('query_total', time_V2.time() - start_time)
                                        rtc_plan_valid = True
                                        # Fail closed on a chunk of any other length than the checkpoint's chunk_size.
                                        new_action_chunk = require_chunk_length(new_action_chunk, num_queries)
                                        new_action_chunk = post_process(new_action_chunk)
                                        if use_fill_in:
                                            new_action_chunk = fill_in(new_action_chunk)

                                        prev_chunk_start_index = current_chunk_start_index
                                        if rtc_prefix_steps:
                                            # The arm executed the pinned prefix while the query ran (as during
                                            # remote mode's overlap); a fresh start runs from the first action.
                                            next_chunk_start_index = rtc_prefix_length
                                        else:
                                            next_chunk_start_index = action_lookahead
                                        if t > 0 and nn_chunk_search:
                                            next_chunk_start_index, best_match_index, prev_target_index = find_nn_chunk_start_index(
                                                all_actions,
                                                new_action_chunk,
                                                prev_chunk_size,
                                                prev_chunk_start_index,
                                                action_lookahead,
                                            )
                                            print(
                                                "[Chunk NN Alignment] "
                                                f"prev_target_idx={prev_target_index}, "
                                                f"matched_next_idx={best_match_index}, "
                                                f"next_chunk_start_idx={next_chunk_start_index}"
                                            )

                                        # The plan this chunk replaces, for the handoff report (none at a fresh start).
                                        handoff_old_plan = rtc_plan if (rtc_merge or (not rtc_prefix_steps and chunk_t > 0)) else None
                                        rtc_plan = new_action_chunk  # the unblended prediction
                                        if rtc_prefix_steps:
                                            if blend_actions_enabled and rtc_prefix_length:
                                                # Executed actions only, after the switch to the new predictions.
                                                # The query was submitted rtc_prefix_length steps before the end
                                                # of the previous chunk's window.
                                                all_actions = blend_rtc_boundary(
                                                    all_actions, new_action_chunk,
                                                    prev_chunk_start_index + prev_chunk_size - rtc_prefix_length,
                                                    rtc_prefix_length, mixing_steps,
                                                    interp_type=blending_method)
                                            else:
                                                all_actions = new_action_chunk
                                        elif t > 0 and 'all_actions' in locals() and blend_actions_enabled:
                                            all_actions = blend_actions(
                                                all_actions,
                                                new_action_chunk,
                                                mixing_steps,
                                                prev_chunk_size,
                                                prev_chunk_start_index,
                                                next_chunk_start_index,
                                                interp_type=blending_method,
                                            )
                                        else:
                                            all_actions = new_action_chunk
                                        if handoff_old_plan is not None:
                                            # New-chunk index 0 is the query's instant: RTC submitted the prefix length
                                            # before the end of the previous window; synchronous chunks query at it.
                                            handoff = chunk_handoff_metrics(
                                                handoff_old_plan, new_action_chunk, all_actions, prev_action, prev_prev_action,
                                                (prev_chunk_start_index + prev_chunk_size - rtc_prefix_length) if rtc_merge
                                                else prev_chunk_size,
                                                next_chunk_start_index, query_lag, t, is_double_arm,
                                                image_age_ms=query_image_age, mode='rtc' if rtc_merge else 'lookahead')
                                            print(format_chunk_handoff(handoff))
                                            run_log.handoff(handoff)
                                        current_chunk_start_index = next_chunk_start_index
                                        prev_chunk_size = current_chunk_size
                                    
                                        if temporal_agg:
                                            all_actions_tensor = torch.from_numpy(all_actions).cuda()
                                            slot_idx = t % num_queries
                                            end_idx = min(num_queries, all_actions.shape[1])
                                            for i in range(end_idx):
                                                if t + i < max_timesteps:
                                                    all_time_actions[slot_idx, t + i] = all_actions_tensor[0, i]
                                            action_timestamps[slot_idx] = t
                                        else:
                                            required_chunk_len = current_chunk_size + current_chunk_start_index
                                            if all_actions.shape[1] < required_chunk_len:
                                                last_action = all_actions[:, -1:]
                                                needed_actions = required_chunk_len - all_actions.shape[1]
                                                repeated_actions = np.repeat(last_action, needed_actions, axis=1)
                                                all_actions = np.concatenate([all_actions, repeated_actions], axis=1)
                                    
                                        if ACTION_INTERP and not temporal_agg:
                                            cur_position = get_observations(controllers)['qpos']
                                            cur_position = np.array(cur_position)[None]
                                            print(cur_position)
                                            new_action = all_actions[:, current_chunk_start_index]
                                            #if the action is too far from the last action, interpolate, measure by 
                                            # For double arm, compare only the first 6 joints for distance calculation
                                            joints_to_compare = 6 if not is_double_arm else 12  # 6 for single, 12 for double (6 per arm)
                                            dis = np.linalg.norm(new_action[:,:joints_to_compare] - cur_position[:,:joints_to_compare])
                                            dis *= (dis>0.03).astype(int)
                                            print(f"Distance: {dis:.3f}")
                                            n_steps =  2*np.ceil(dis / 0.05).astype(int) # 3*np.ceil(dis / 0.01).astype(int)
                                            print(f"prev prev action:{prev_prev_action}")
                                            print(f"prev action:{prev_action}")
                                            print(f"current position:{cur_position}")
                                        
                                            for i_ in range(0, n_steps):
                                                # print("interpolating")
                                                alpha = i_ / n_steps
                                                virtual_action = 2 * prev_action - prev_prev_action
                                                wp1 = alpha * new_action + (1 - alpha) * virtual_action
                                                wp2 = (1 - alpha) * prev_action + alpha * virtual_action
                                                action = wp2 + (wp1 - wp2) * alpha
                                                # action = last_action + (new_action - last_action) * (i_ + 1) / n_steps
                                                cur_position = get_observations(controllers)['qpos']
                                                # time_V2.sleep(0.2)
                                                print(f"action:{action}")
                                                print(f"cur_position:{cur_position}")
                                                set_action(action, controllers, commands, is_double_arm)
                                                while execution_dt - (time_V2.time() - time1) > 0:
                                                    pass
                                                time1 = time_V2.time()
                                        
                                            print(f"new action:{new_action}")
                                
                                    # Select action based on temporal aggregation or normal mode
                                    if temporal_agg:
                                        # Temporal aggregation: exponentially weighted average of overlapping actions
                                        if t < max_timesteps:
                                            # Collect all valid actions for current timestep from overlapping chunks
                                            actions_for_curr_step = []
                                            timestamps_for_curr_step = []
                                        
                                            for slot_idx in range(num_queries):
                                                if action_timestamps[slot_idx] >= 0:  # Valid slot
                                                    # Check if this chunk's prediction covers the current timestep
                                                    chunk_start_time = action_timestamps[slot_idx]
                                                    if chunk_start_time <= t < chunk_start_time + num_queries:
                                                        action_val = all_time_actions[slot_idx, t]
                                                        # Check if this action exists and is non-zero (valid)
                                                        if torch.any(action_val != 0):
                                                            actions_for_curr_step.append(action_val)
                                                            timestamps_for_curr_step.append(chunk_start_time)
                                        
                                            if len(actions_for_curr_step) > 0:
                                                actions_for_curr_step = torch.stack(actions_for_curr_step)
                                                # Compute exponential weights based on recency of chunk start times
                                                timestamps_for_curr_step = torch.tensor(timestamps_for_curr_step).cuda()
                                                time_diffs = t - timestamps_for_curr_step
                                                k = 0.005
                                                exp_weights = torch.exp(-k * time_diffs)
                                                exp_weights = exp_weights / exp_weights.sum()
                                                exp_weights = exp_weights.unsqueeze(dim=1)
                                                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                                                action = raw_action.cpu().numpy()
                                            else:
                                                # Fallback if no actions available
                                                action = prev_action.copy()
                                        else:
                                                action = prev_action.copy()
                                    else:
                                        action = select_action_from_chunk(
                                            chunk_t,
                                            first_chunk_size,
                                            query_frequency,
                                            current_chunk_start_index,
                                            gripper_lookahead,
                                            all_actions,
                                            is_double_arm,
                                        )
                                        if should_query and chunk_t == 0 and dagger_state == DAGGER_STATE_POLICY:
                                            # A fresh start (first query, or after a reset) sends this target
                                            # as one step from the measured pose; a far one fails closed.
                                            check_fresh_start_target(action, qpos_numpy, is_double_arm)

                                    prev_prev_action = prev_action.copy()
                                    prev_action = action.copy()

                                # Log to rerun if enabled
                                if use_rerun:
                                    rr.set_time("frame", sequence=t)
                                
                                    # Log camera images
                                    if should_query:
                                        log_rerun_camera_images(image_dict, camera_names)
                                
                                    # Log joint positions and gripper
                                    obs = get_observations(controllers)
                                    current_qpos = obs['qpos']
                                    current_qvel = obs['qvel']

                                    if is_double_arm:
                                        # Convert to numpy for logging
                                        qpos_numpy = np.array(qpos.cpu().squeeze(0))
                                        # Log the input qpos to the policy
                                        seen_left_joint_positions = qpos_numpy[:6]
                                        seen_left_gripper_position = qpos_numpy[6]
                                        seen_right_joint_positions = qpos_numpy[7:13]
                                        seen_right_gripper_position = qpos_numpy[13]

                                        for i, joint_val in enumerate(seen_left_joint_positions):
                                            rr.log(f"seen_robot_state/left_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("seen_robot_state/left_arm/gripper", rr.Scalars(seen_left_gripper_position))
                                        for i, joint_val in enumerate(seen_right_joint_positions):
                                            rr.log(f"seen_robot_state/right_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("seen_robot_state/right_arm/gripper", rr.Scalars(seen_right_gripper_position))

                                        # Double arm: log both arms separately
                                        left_joint_positions = current_qpos[:6]
                                        left_gripper_position = current_qpos[6]
                                        right_joint_positions = current_qpos[7:13]
                                        right_gripper_position = current_qpos[13]

                                        # Log velocities
                                        left_joint_velocities = current_qvel[:6]
                                        left_gripper_velocity = current_qvel[6]
                                        right_joint_velocities = current_qvel[7:13]
                                        right_gripper_velocity = current_qvel[13]

                                        # Log left arm

                                        for i, joint_val in enumerate(left_joint_positions):
                                            rr.log(f"robot_state/left_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("robot_state/left_arm/gripper", rr.Scalars(left_gripper_position))
                                    
                                        # Log right arm
                                        for i, joint_val in enumerate(right_joint_positions):
                                            rr.log(f"robot_state/right_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("robot_state/right_arm/gripper", rr.Scalars(right_gripper_position))

                                        # Log left arm velocities
                                        for i, vel_val in enumerate(left_joint_velocities):
                                            rr.log(f"robot_state/left_arm/velocity/joint_{i}", rr.Scalars(vel_val))
                                        rr.log("robot_state/left_arm/velocity/gripper", rr.Scalars(left_gripper_velocity))
                                        # Log right arm velocities
                                        for i, vel_val in enumerate(right_joint_velocities):
                                            rr.log(f"robot_state/right_arm/velocity/joint_{i}", rr.Scalars(vel_val))
                                        rr.log("robot_state/right_arm/velocity/gripper", rr.Scalars(right_gripper_velocity))

                                        # Log target actions for both arms
                                        target_left_joints = action[0][:6]
                                        target_left_gripper = action[0][6]
                                        target_right_joints = action[0][7:13]
                                        target_right_gripper = action[0][13]
                                    
                                        for i, joint_val in enumerate(target_left_joints):
                                            rr.log(f"target_actions/left_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("target_actions/left_arm/gripper", rr.Scalars(target_left_gripper))
                                    
                                        for i, joint_val in enumerate(target_right_joints):
                                            rr.log(f"target_actions/right_arm/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("target_actions/right_arm/gripper", rr.Scalars(target_right_gripper))

                                        # left_arm_transform, right_arm_transform = load_dual_arm_calibration()

                                    
                                        # # Update robot model visualization for both arms
                                        # if links is not None and joints is not None:
                                        #     log_robot_model_with_transform(links, joints, left_joint_positions, left_gripper_position, base_transform=left_arm_transform, robot_name="left_arm")
                                        #     log_robot_model_with_transform(links, joints, right_joint_positions, right_gripper_position, base_transform=right_arm_transform, robot_name="right_arm")
                                    else:
                                        # Single arm: original behavior
                                        joint_positions = current_qpos[:6]
                                        gripper_position = current_qpos[-1]
                                    
                                        for i, joint_val in enumerate(joint_positions):
                                            rr.log(f"robot_state/joint_positions/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("robot_state/gripper_position", rr.Scalars(gripper_position))
                                    
                                        # Log target actions
                                        target_joint_positions = action[0][:6]
                                        target_gripper_position = action[0][-1]
                                        for i, joint_val in enumerate(target_joint_positions):
                                            rr.log(f"target_actions/joint_positions/joint_{i}", rr.Scalars(joint_val))
                                        rr.log("target_actions/gripper_position", rr.Scalars(target_gripper_position))
                                    
                                        # Update robot model visualization
                                        if links is not None and joints is not None:
                                            log_robot_model(links, joints, joint_positions, gripper_position)

                                # l_js = controllers['left'].get_joint_state()
                                # r_js = controllers['right'].get_joint_state() if is_double_arm else None
                            
                                # l_vel = l_js.gripper_vel
                                # r_vel = r_js.gripper_vel if is_double_arm else None
                                # if l_vel < 0:
                                #     action[0][6] -= 0.005
                                # if r_vel < 0:
                                #     action[0][13] -= 0.005
                                # send action to robot and perform step
                                if record_dagger_corrections:
                                    dagger_correction_requested = dagger_state in {
                                        DAGGER_STATE_GRAVITY_COMP,
                                        DAGGER_STATE_RECORDING,
                                    }
                                    if dagger_correction_requested:
                                        # A hand correction moves the arm off the plan: the next
                                        # policy query starts fresh instead of pinning a stale prefix,
                                        # and with RTC right when policy control resumes.
                                        rtc_plan_valid = False
                                        rtc_fresh_start = True
                                    dagger_recording_requested = dagger_state == DAGGER_STATE_RECORDING
                                    if dagger_correction_requested and not dagger_gains_zeroed:
                                        apply_gain_mode(
                                            controllers,
                                            runtime_gain_snapshots,
                                            is_double_arm,
                                            zero_gains=True,
                                        )
                                        dagger_gains_zeroed = True
                                    if (
                                        not dagger_recording_requested
                                        and dagger_recorder is not None
                                        and dagger_recorder.active
                                    ):
                                        dagger_recorder.stop_segment()
                                    if (
                                        not dagger_correction_requested
                                        and dagger_gains_zeroed
                                        and should_query
                                    ):
                                        apply_gain_mode(
                                            controllers,
                                            runtime_gain_snapshots,
                                            is_double_arm,
                                            zero_gains=False,
                                        )
                                        dagger_gains_zeroed = False

                                set_action(action, controllers, commands, is_double_arm)
                                # Diagnosis only: the measured pose right after sending this target.
                                run_log.step(t=t, chunk_t=chunk_t, rollout=rollout_id, demo=demo_id,
                                             target=np.asarray(action, dtype=float).reshape(-1),
                                             qpos=np.asarray(get_observations(controllers)['qpos'], dtype=float).reshape(-1),
                                             observed=bool(should_query), submitted=bool(rtc_submit),
                                             merged=bool(rtc_merge))

                                if (
                                    record_dagger_corrections
                                    and dagger_recorder is not None
                                    and dagger_recorder.active
                                ):
                                    action_sample = action[0] if len(action.shape) > 1 else action
                                    dagger_recorder.append_observation(captured_observation, action_sample)

                                if record_rollout:
                                    current_rollout_idx = len(rollout_lists['position'])
                                    standard_recording.append_rollout(rollout_lists, captured_observation, action)
                                    if should_query:
                                        rollout_lists['inference_query_indices'].append(current_rollout_idx)
                                        rollout_lists['inference_query_timestamps'].append(captured_observation.robot_timestamp)
                                        rollout_lists['inference_query_overhead_frame_numbers'].append(query_overhead_frame_number)
                                                    
                                ### for visualization
                                if save_episode:
                                    visualization_frame = make_visualization_frame(image_dict, camera_names)
                                    if visualization_frame is not None:
                                        image_list.append(visualization_frame)
                                qpos_list.append(qpos_numpy)
                                target_qpos_list.append(action)
                            
                                duration = time_V2.time() - time1
                                timing_stats.record('loop_total', duration)
                                timing_stats.report_if_due(step=t, query_count=query_count)
                                while (time_V2.time() - time1) < execution_dt:
                                    if rtc_queries is not None:
                                        # Yield the GIL: a pure spin starves the RTC query thread
                                        # (~1.5 s instead of ~15 ms per query); step timing is unchanged.
                                        time_V2.sleep(0)
                            
                                if duration >= execution_dt:
                                    culmulated_delay += (duration - execution_dt)
                                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than EXECUTION_DT: {execution_dt} s, culmulated delay: {culmulated_delay:.3f} s')
                                t+=1 
                            
                        elif pause_flag:
                            action = hold_action_until_unpaused(
                                action,
                                controllers,
                                commands,
                                execution_dt,
                                is_double_arm=is_double_arm,
                            )
                            prev_prev_action = prev_action.copy()
                            prev_action = action.copy()
                            # The arm held still off the plan's timeline: with RTC, continue with a fresh chunk.
                            rtc_fresh_start = True
                            if lookahead_tuner is not None and (
                                    (lookahead_tuner.action, lookahead_tuner.gripper)
                                    != (action_lookahead, gripper_lookahead)):
                                # Tuned while paused: the gripper offset applies from the next step,
                                # the action lookahead from the next chunk boundary.
                                action_lookahead, gripper_lookahead = lookahead_tuner.action, lookahead_tuner.gripper
                                print(f'{lookahead_tuner}: in use from step {t}')
                        elif reset_ready_requested or (not continue_flag):
                            print('Paused, returning arm(s) to saved start position')
                            if record_dagger_corrections and dagger_recorder is not None:
                                dagger_state = DAGGER_STATE_POLICY
                                if dagger_recorder.active:
                                    dagger_recorder.stop_segment()
                                if dagger_gains_zeroed:
                                    apply_gain_mode(
                                        controllers,
                                        runtime_gain_snapshots,
                                        is_double_arm,
                                        zero_gains=False,
                                    )
                                    dagger_gains_zeroed = False
                                prompt_save_dagger_corrections(dagger_recorder)
                            if record_rollout and len(rollout_lists['position']) > 0:
                                label = prompt_rollout_label(eval_mode=eval_mode)
                                save_and_clear_rollout_data(
                                    rollout_dir,
                                    rollout_task_name_actual,
                                    action_dim,
                                    is_double_arm,
                                    rollout_camera_names,
                                    rollout_lists,
                                    label,
                                    rollout_subdir_name,
                                    eval_mode=eval_mode,
                                    date_string=rollout_date,
                                )
                            return_to_start_with_open_grippers(
                                controllers,
                                commands,
                                start_action,
                                execution_dt=execution_dt,
                                is_double_arm=is_double_arm,
                                **start_move_options,
                            )
                            action = start_action.copy() if start_action is not None else np.zeros((1, action_dim))
                            prev_action = action.copy()
                            prev_prev_action = action.copy()
                            reset_ready_requested = False
                            print("Ready position reached. Press c/p/1 to continue, r/h to reset again, or q to home and quit.")
                            with idle_hold(controllers):
                                while not continue_flag and not reset_ready_requested and not exit_requested:
                                    time_V2.sleep(0.001)
                                if not (reset_ready_requested or exit_requested):
                                    run_log.flush(rollout=rollout_id, demo=demo_id)
                            if reset_ready_requested or exit_requested:
                                continue
                            t = 0
                            chunk_origin, rtc_fresh_start = 0, False
                            demo_id += 1
                            if dagger_recorder is not None:
                                dagger_recorder.set_segment_metadata(
                                    **episode_metadata(rollout_id, demo_id),
                                    rollout_id=rollout_id,
                                    demo_id=demo_id,
                                )
                            prev_chunk_size = first_chunk_size
                            # No chunk boundary precedes a first chunk: it runs from its first action.
                            current_chunk_start_index = 0
                            rtc_plan_valid = False
                        
                print('======================================================')
                print(f'demo {demo_id} done')
                print('======================================================')
                # Saving (~35 MB) blocks for longer than a command watchdog; keep the hold fresh.
                with idle_hold(controllers):
                    capture_gc.collect_idle()  # the rollout has ended; the arm holds natively
                    run_log.flush(rollout=rollout_id, demo=demo_id)
                    log_id = get_auto_index(ckpt_dir)
                    np.save(os.path.join(ckpt_dir, f'qpos_{log_id}_{demo_id}.npy'), qpos_history_raw)
                    np.save(os.path.join(ckpt_dir, f'target_qpos_{log_id}_{demo_id}.npy'), target_qpos_list)
                if record_rollout and len(rollout_lists['position']) > 0:
                    label = prompt_rollout_label(eval_mode=eval_mode)
                    save_and_clear_rollout_data(
                        rollout_dir,
                        rollout_task_name_actual,
                        action_dim,
                        is_double_arm,
                        rollout_camera_names,
                        rollout_lists,
                        label,
                        rollout_subdir_name,
                        eval_mode=eval_mode,
                        date_string=rollout_date,
                    )
                if record_dagger_corrections and dagger_recorder is not None:
                    dagger_state = DAGGER_STATE_POLICY
                    if dagger_recorder.active:
                        if dagger_gains_zeroed:
                            apply_gain_mode(
                                controllers,
                                runtime_gain_snapshots,
                                is_double_arm,
                                zero_gains=False,
                            )
                            dagger_gains_zeroed = False
                        dagger_recorder.stop_segment()
                    prompt_save_dagger_corrections(dagger_recorder)
    except BaseException:
        failed = True  # The failure exit already homed (or disabled in place).
        if isinstance(controllers, RuntimeArms):
            controllers.failure_handled = True
        raise
    finally:
        if record_dagger_corrections and dagger_recorder is not None:
            dagger_state = DAGGER_STATE_POLICY
            if dagger_recorder.active:
                dagger_recorder.stop_segment()
            if dagger_gains_zeroed:
                apply_gain_mode(
                    controllers,
                    runtime_gain_snapshots,
                    is_double_arm,
                    zero_gains=False,
                )
            prompt_save_dagger_corrections(dagger_recorder)
        if stop_event is not None:
            stop_event.set()
        if rtc_queries is not None:
            rtc_queries.stop()
        run_log.flush()  # before homing, which can itself fail
        if not failed:
            reset_controllers_to_home(controllers, is_double_arm)
        run_log.close()
        
        if save_episode:
            print("saving video")
            save_videos(image_list, execution_dt, video_path=os.path.join(ckpt_dir, 'figs'))
    return

def main_eval(args):
    global continue_flag, pause_flag, exit_requested, reset_ready_requested, EXECUTION_DT
    continue_flag = True
    pause_flag = False
    exit_requested = False
    reset_ready_requested = False
    set_seed(1)
    from config.task_config import load_inference_checkpoint_config
    config = load_inference_checkpoint_config(args)
    require_absolute_actions(config)
    from config.task_config import configure_inference_task
    yaml_task = configure_inference_task(config, args)
    if yaml_task and config['task_config']['robot_type'] != 'arx':
        raise ValueError('This inference runner requires robot.type arx')
    task_config = config['task_config']
    ckpt_dir = config['ckpt_dir']
    # ckpt_dir = config['ckpt_dir'].split('/')[-1]
    require_supported_policy(config)
    masking_strategy = standard_image_processing.masking_strategy(config, args.get('masking_strategy'))
    
    # ckpt_dir = os.path.join(os.getcwd(),"ckpts", ckpt_dir)
    config['ckpt_dir'] = ckpt_dir
    
    # Get rerun option from command line args
    use_rerun = args.get('rerun')
    
    # Get query frequency from command line args
    query_frequency = args.get('query_frequency')
    
    # Get temporal aggregation from command line args
    temporal_agg = args.get('temporal_agg')
    first_chunk_size = args.get('first_chunk_size')

    diffusion_inference_steps = args.get('diffusion_inference_steps')
    if diffusion_inference_steps is not None:
        if diffusion_inference_steps <= 0:
            raise ValueError(
                f"--diffusion_inference_steps must be > 0, got {diffusion_inference_steps}"
            )
        if config.get('policy_class') == 'Flow':
            config['policy_config']['num_inference_timesteps'] = int(diffusion_inference_steps)
            print(
                "[Config Override] "
                f"policy_config.num_inference_timesteps={diffusion_inference_steps}"
            )
        else:
            print(
                "[Config Override] Ignoring --diffusion_inference_steps because "
                f"policy_class={config.get('policy_class')} is not Diffusion/Flow."
            )

    use_camera_server = args.get('use_camera_server', True)
    action_lookahead = args.get('action_lookahead')  # None: 10 without RTC; RTC excludes it
    gripper_lookahead = args.get('gripper_lookahead')  # None: 5, or 0 with RTC
    inference_on_refresh = args.get('inference_on_refresh', False)
    compress_inference_images = args.get('compress_inference_images', False)
    inference_jpeg_quality = args.get('inference_jpeg_quality', 50)
    eval_mode = args.get('eval', False)
    record_rollout = args.get('record_rollout', False) or eval_mode
    rollout_dir = args.get('rollout_dir', str(ROLLOUT_DIR))
    rollout_task_name = args.get('rollout_task_name', None)
    record_dagger_corrections = args.get('record_dagger_corrections', False)
    dagger_dir = args.get('dagger_dir', str(DAGGER_DIR))
    dagger_task_name = args.get('dagger_task_name', None)
    nn_chunk_search = args.get('nn_chunk_search', False)
    arm_kp = args.get('arm_kp', None)
    arm_kd = args.get('arm_kd', None)
    start_position_path = args.get('start_position_path', DEFAULT_START_POSITION_PATH)
    if start_position_path == DEFAULT_START_POSITION_PATH:
        # Same per-task/shared pose selection as the recorder.
        start_position_path = default_start_position_path(
            task_config.get('task_name'), task_config.get('robot_type'))
    action_command_style = args.get('action_command_style', None)
    fps = resolve_policy_rate(config, args.get('fps'), legacy_rate=args.get('legacy_policy_fps'), context='checkpoint')
    config['control_freq'] = fps
    execution_dt = 1.0 / fps
    EXECUTION_DT = execution_dt
    use_fill_in = args.get('use_fill_in', False)
    if use_fill_in:
        raise ValueError('Use native SDK interpolation; disable --use_fill_in to preserve the policy action rate')
    print(f"[Config] FPS={fps}, EXECUTION_DT={execution_dt:.6f}s, use_fill_in={use_fill_in}")
    if start_position_path is not None:
        print(f"[Config] Start position path={start_position_path}, gripper_closed_pos={GRIPPER_CLOSED_POS:.3f}")
    
    # Get policy step from command line args, default to -1 (best)
    policy_step = args.get('policy_step', -1)
    if policy_step == -1:
        ckpt_names = [f'policy_best.ckpt']
    else:
        ckpt_names = [f'policy_step_{policy_step}_seed_0.ckpt']

    # if classifier_free_guidance:
    #     config['cfg_ckpt_dir'] = os.path.join(config['cfg_ckpt_dir'], )
    print('checkpoint names:', ckpt_dir)
    for ckpt_name in ckpt_names:
        if not os.path.isfile(os.path.join(ckpt_dir, ckpt_name)):
            print(f'Warning: {ckpt_name} not found at {os.path.join(ckpt_dir, ckpt_name)}')
        else:
            eval_bc(
                config,
                ckpt_name,
                save_episode=args.get('save_episode', SAVE_EPISODE),
                num_rollouts=10,
                task_config=task_config,
                use_rerun=use_rerun,
                query_frequency=query_frequency,
                first_chunk_size=first_chunk_size,
                temporal_agg=temporal_agg,
                masking_strategy=masking_strategy,
                history_len=args.get('history_len', 0),
                use_camera_server=use_camera_server,
                action_lookahead=action_lookahead,
                gripper_lookahead=gripper_lookahead,
                execution_dt=execution_dt,
                use_fill_in=use_fill_in,
                blend_actions_enabled=args.get('blend_actions', True),
                blending_method=args.get('blending_method', 'sigmoid'),
                mixing_steps=args.get('mixing_steps', 5),
                inference_on_refresh=inference_on_refresh,
                compress_inference_images=compress_inference_images,
                inference_jpeg_quality=inference_jpeg_quality,
                record_rollout=record_rollout,
                rollout_dir=rollout_dir,
                rollout_task_name=rollout_task_name,
                nn_chunk_search=nn_chunk_search,
                record_dagger_corrections=record_dagger_corrections,
                dagger_dir=dagger_dir,
                dagger_task_name=dagger_task_name,
                rtc_prefix_steps=args.get('rtc_prefix_steps', 0) if args.get('rtc') else 0,
                fast_inference=bool(args.get('fast_inference', False)),
                timing=bool(args.get('timing', False)),
                timing_interval=int(args.get('timing_interval', 500)),
                arm_kp=arm_kp,
                arm_kd=arm_kd,
                start_position_path=start_position_path,
                action_command_style_override=action_command_style,
                eval_mode=eval_mode,
            )

def commandline_input():
    global continue_flag, pause_flag, exit_requested, reset_ready_requested, dagger_state
    global dagger_command_enabled, dagger_prompt_active, dagger_prompt_key_queue
    # Wait a moment to allow initial getkey to complete
    import time
    time.sleep(0.5)
    print(
        '\n[Input] c/p freeze or resume, r resets to ready, q homes and quits. '
        'Foot pedal: 1 acts as c, h resets to ready. '
        'DAGGER footpedal: 1 enters gravity comp while running, 2 starts recording, 3 exits correction.'
    )
    
    while not exit_requested:
        try:
            key = getkey().lower()
            if dagger_prompt_active:
                dagger_prompt_key_queue.put(key)
                continue
            # Two-pedal foot switch: right pedal resets to ready; left pedal is
            # c, except that DAgger keeps it for gravity comp while running.
            if key == PEDAL_RESET_KEY:
                key = 'r'
            elif key == PEDAL_GO_KEY and (not dagger_command_enabled or pause_flag
                                          or reset_ready_requested or not continue_flag):
                key = 'c'
            if pause_flag and lookahead_tuner is not None and lookahead_tuner.adjust(key):
                continue
            if key in ('c', 'p'):
                if reset_ready_requested or not continue_flag:
                    continue_flag = True
                    pause_flag = False
                    reset_ready_requested = False
                    print("Continuing from current position")
                else:
                    pause_flag = not pause_flag
                    state = "frozen in place" if pause_flag else "continuing from current position"
                    print(f'pause_flag: {pause_flag} ({state})')
                    if pause_flag and lookahead_tuner is not None:
                        print(f'{lookahead_tuning.HELP} {lookahead_tuner}')
                continue
            if key == 'r':
                reset_ready_requested = True
                continue_flag = False
                pause_flag = False
                print('reset_ready_requested: True')
                continue
            if key == 'q':
                exit_requested = True
                continue_flag = False
                pause_flag = False
                print('exit_requested: True (homing before quit)')
                continue
            if dagger_command_enabled and key == DAGGER_CORRECTION_KEY:
                if dagger_state == DAGGER_STATE_POLICY:
                    dagger_state = DAGGER_STATE_GRAVITY_COMP
                    print('[DAGGER] gravity_comp: True, recording: False')
                else:
                    print(f'[DAGGER] state: {dagger_state}')
                continue
            if dagger_command_enabled and key == DAGGER_RECORD_KEY:
                if dagger_state == DAGGER_STATE_GRAVITY_COMP:
                    dagger_state = DAGGER_STATE_RECORDING
                    print('[DAGGER] recording: True')
                elif dagger_state == DAGGER_STATE_RECORDING:
                    print('[DAGGER] recording already active')
                continue
            if key == DAGGER_EXIT_KEY:
                if dagger_command_enabled and dagger_state != DAGGER_STATE_POLICY:
                    dagger_state = DAGGER_STATE_POLICY
                    print('[DAGGER] gravity_comp: False, recording: False')
                else:
                    reset_ready_requested = True
                    continue_flag = False
                    pause_flag = False
                    print('reset_ready_requested: True')
        except EOFError:
            # Handle case where input is not available
            time.sleep(1)
            continue
  

def build_parser():
    from config.inference_config import InferenceArgumentParser
    parser = InferenceArgumentParser(backend='direct')
    parser.add_argument('--blending-method', choices=['linear', 'exponential', 'sigmoid'], default='sigmoid')
    parser.add_argument('--config', '-c', action='store', type=str, help='Checkpoint all_configs.json; may be selected by --inference-config.', default=None)
    parser.add_argument('--rerun', '-r', action='store_true', default=False, help='Use rerun visualizer for policy replay.')
    parser.add_argument('--policy_step', '-p', action='store', type=int, default=-1, help='Policy step to load from checkpoint.')
    parser.add_argument('--masking_strategy', '-m', action='store', type=str, default=None,
                        help='Confirms the checkpoint\'s training mask; a different value is refused. Default: the checkpoint\'s.')
    parser.add_argument('--diffusion_inference_steps', action='store', type=int, default=5,
                        help='Override diffusion/flow sampling steps at inference time.')
    parser.add_argument('--fast_inference', action=argparse.BooleanOptionalAction, default=False,
                        help='Flow only: fp16 channels-last U-Net convolutions and a CUDA-graphed U-Net step.')
    # Same timing report as the remote client.
    parser.add_argument('--timing', action='store_true',
                        help='Print per-stage timing (avg/max) every --timing_interval control steps.')
    parser.add_argument('--timing_interval', type=int, default=500)
    parser.add_argument('--query_frequency', '-q', action='store', type=int, default=None, help='How often to query the policy (in frames).')
    parser.add_argument('--task-config', dest='task_config', help='Task YAML; defaults to checkpoint task name.')
    parser.add_argument('--save_episode', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--first_chunk_size', '-f', action='store', type=int, default=None,
                        help='Size of the first action chunk (in frames). If omitted, uses query_frequency.')
    parser.add_argument('--temporal_agg', '-t', action='store_true', help='Enable temporal aggregation. When enabled, queries policy every frame and uses exponentially weighted average.')
    parser.add_argument('--history_len', '-hl', action='store', type=int, default=0, help='Length of history for classifier free guidance approximation')
    parser.add_argument('--use-camera-server', action='store_true', dest='use_camera_server',
                        help='Use camera server for shared memory access.', default=True)
    parser.add_argument('--no-camera-server', action='store_false', dest='use_camera_server',
                        help='Disable camera server and open cameras directly.')
    parser.add_argument('--inference_on_refresh', action='store_true', default=False,
                        help='Wait for a fresh overhead frame before running inference at query points.')
    parser.add_argument('--compress_inference_images', action='store_true', default=False,
                        help='JPEG-compress/decompress incoming frames before inference.')
    parser.add_argument('--inference_jpeg_quality', action='store', type=int, default=50,
                        help='JPEG quality used when --compress_inference_images is enabled (0-100).')
    parser.add_argument('--action_lookahead', action='store', type=int, default=None,
                        help='Target timestep offset used for chunk execution alignment (default 10). '
                             'Not used with RTC, which executes its prefix first.')
    parser.add_argument('--gripper_lookahead', action='store', type=int, default=None,
                        help='Additional lookahead for gripper actions beyond action_lookahead. Default 5; '
                             'RTC runs the gripper on the arm\'s action (0, and refuses others).')
    parser.add_argument('--nn_chunk_search', action='store_true', default=False,
                        help='At chunk boundaries, search +/-10 timesteps around action_lookahead in the next chunk and choose the start index whose target-time action best matches the previous chunk continuation.')
    parser.add_argument('--legacy-policy-fps', type=float, default=None, help='Explicit training FPS for legacy checkpoints without rate metadata.')
    parser.add_argument('--control-freq','--fps',dest='fps', action='store', type=float, default=None,
                        help='Action execution FPS; defaults to checkpoint metadata and must match it. SDK remains 500 Hz.')
    parser.add_argument('--use_fill_in', action='store_true', dest='use_fill_in', default=False,
                        help='Use fill_in function to interpolate action chunks (unsupported; SDK interpolation preserves timing)')
    parser.add_argument('--no_fill_in', action='store_false', dest='use_fill_in',
                        help='Disable fill_in function for action chunks')
    parser.add_argument('--blend_actions', action='store_true', dest='blend_actions', default=True,
                        help='Blend consecutive action chunks for smooth transitions (default: enabled)')
    parser.add_argument('--no_blend_actions', action='store_false', dest='blend_actions',
                        help='Disable action blending and use new chunks directly')
    parser.add_argument('--mixing_steps', action='store', type=int, default=10,
                        help='Number of steps to blend over when blending actions (default: 5)')
    parser.add_argument('--kp', '--arm_kp', action='store', type=float, nargs=7, default=None, dest='arm_kp',
                        help=f'ARX proportional gains for 6 joints + gripper. Default: {ACTIVE_KP}.')
    parser.add_argument('--kd', '--arm_kd', action='store', type=float, nargs=7, default=None, dest='arm_kd',
                        help=f'ARX derivative gains for 6 joints + gripper. Default: {ACTIVE_KD}.')
    parser.add_argument('--start_position', action='store', type=str, default=DEFAULT_START_POSITION_PATH,
                        dest='start_position_path',
                        help='Start inference from this calibration .npz. Default: calibration/start_positions/<task>.npz '
                             'if saved, else the shared pose for the robot type.')
    parser.add_argument('--no_start_position', action='store_const', const=None, dest='start_position_path',
                        help='Do not move to a saved calibration start position before inference.')
    parser.add_argument('--record_rollout', action=argparse.BooleanOptionalAction, default=None,
                        help='Record full rollout episodes (images + proprio) to Zarr with keep/discard prompting.')
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=None,
                        help='Evaluation rollout mode: records rollouts under today date, prompts success/failure/discard, and prints eval banners.')
    parser.add_argument('--rollout_dir', action='store', type=str, default=str(ROLLOUT_DIR),
                        help='Directory for rollout recordings. Default is data/rollouts/ in this repository.')
    parser.add_argument('--rollout_task_name', action='store', type=str, default=None,
                        help='Task name for rollout recordings. Default is derived from checkpoint name.')
    parser.add_argument('--record_dagger_corrections', action='store_true', default=False,
                        help='Enable HG-DAGGER correction recording. Footpedal mapping: a enters gravity comp, b starts recording, c exits correction.')
    parser.add_argument('--dagger_dir', action='store', type=str, default=str(DAGGER_DIR),
                        help='Directory for direct HG-DAGGER Zarr datasets. Episodes go under <task_name>/.')
    parser.add_argument('--dagger_task_name', action='store', type=str, default=None,
                        help='Task name for DAGGER recordings. Default is derived from checkpoint name.')
    parser.add_argument('--dagger_toggle_command', action='store', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--action_command_style', action='store', type=str, default=None,
                        choices=['absolute'],
                        help='Only absolute actions are supported; legacy relative checkpoints are rejected.')
    return parser


def main(argv=None):
    global dagger_command_enabled
    args = build_parser().parse_args(argv)
    dagger_command_enabled = args.record_dagger_corrections
    main_eval(vars(args))


if __name__ == "__main__":
    main()
