"""
Unified visualization and physical replay script for ARX arms.

Supports:
1. Rerun visualization of recorded episodes.
2. Physical replay on ARX arms with --action_replay.

"""

# Support direct script execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

from contracts.action_contract import require_absolute_actions
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import cv2
from data_processing import episode_store
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr

from config.constants import DATA_DIR, TASK_CONFIGS, resolve_arx_single_arm_can, resolve_arx_single_arm_side
from visualization.rerun_utils import (
    build_failure_frame_set,
    depisode_data_camera_overlay,
    parse_urdf,
    visualize_episode_rerun,
)
from visualization.visualization_utils.memory_guard import MemoryGuard, open_rrd


from utils import replay_utils
from contracts.action_rate_contract import resolve_rate


OPENPI_IMAGE_RESOLUTION = (224, 224)
VIEW_JPEG_QUALITY = 90


CAMERA_ENDPOINT_OFFSETS = {
    "left": 0,
    "overhead": 1,
    "right": 2,
    "torso": 3,
    "test": 4,
}


def default_arx_urdf_path():
    return Path(__file__).resolve().parents[2] / "src" / "hardware" / "arx5-sdk" / "models" / "L5.urdf"


def normalize_robot_type(robot_type):
    if robot_type in {"arx", "arx5"}:
        return "arx"
    raise ValueError(f"Unsupported robot type: {robot_type}")


def openpi_resize_with_pad(image, height=224, width=224):
    """Mirror openpi.shared.image_tools.resize_with_pad for uint8 HWC images."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image for OpenPI resize, got shape {image.shape}")
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(
        image.astype(np.float32),
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    if image.dtype == np.uint8:
        resized = np.rint(resized).clip(0, 255).astype(np.uint8)
    else:
        resized = resized.clip(-1.0, 1.0).astype(image.dtype, copy=False)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    pad_value = 0 if image.dtype == np.uint8 else -1.0
    return np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def prepare_visualization_image(image, *, pimode=False):
    if pimode:
        return openpi_resize_with_pad(image, *OPENPI_IMAGE_RESOLUTION)
    return np.asarray(image)


class LazyFrames:
    """Decode or transform one frame per access instead of holding the episode in RAM."""
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.transform(self.frames[index if index >= 0 else index + len(self)])

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def read_optional_dataset(root, candidate_paths):
    for path in candidate_paths:
        if path in root:
            return root[path][()]
    return None


def load_episode_annotations(root):
    return {
        "subgoal_switch": read_optional_dataset(
            root,
            (
                "/observations/subgoal_switch",
                "/subgoal_switch",
                "/annotations/subgoal_switch",
                "/metadata/subgoal_switch",
            ),
        ),
        "subgoal_switch_frames": read_optional_dataset(
            root,
            (
                "/observations/subgoal_switch_frames",
                "/observations/subgoal_switch_frame",
                "/subgoal_switch_frames",
                "/subgoal_switch_frame",
                "/subgoal_switch_frame_indices",
                "/annotations/subgoal_switch_frames",
                "/annotations/subgoal_switch_frame",
                "/annotations/subgoal_switch_frame_indices",
            ),
        ),
        "subgoal_switch_timestamps": read_optional_dataset(
            root,
            (
                "/subgoal_switch_timestamps",
                "/annotations/subgoal_switch_timestamps",
                "/metadata/subgoal_switch_timestamps",
            ),
        ),
        "subgoal_switch_key": root.attrs.get("subgoal_switch_key"),
        "subgoal_switch_count": root.attrs.get("subgoal_switch_count"),
        "failure": read_optional_dataset(
            root,
            (
                "/observations/failure",
                "/failure",
                "/annotations/failure",
                "/metadata/failure",
            ),
        ),
        "failure_frames": read_optional_dataset(
            root,
            (
                "/observations/failure_frames",
                "/observations/failure_frame",
                "/failure_frames",
                "/failure_frame",
                "/failure_frame_indices",
                "/annotations/failure_frames",
                "/annotations/failure_frame",
                "/annotations/failure_frame_indices",
            ),
        ),
        "failure_key": root.attrs.get("failure_key"),
        "failure_count": root.attrs.get("failure_count"),
    }


def nearest_frame_indices_for_timestamps(reference_timestamps, event_timestamps):
    if reference_timestamps is None or event_timestamps is None:
        return None

    reference = np.asarray(reference_timestamps, dtype=np.float64).reshape(-1)
    events = np.asarray(event_timestamps, dtype=np.float64).reshape(-1)
    if reference.size == 0 or events.size == 0:
        return None

    indices = []
    for event_ts in events:
        if not np.isfinite(event_ts):
            continue
        idx = int(np.argmin(np.abs(reference - event_ts)))
        indices.append(idx)
    if not indices:
        return None
    return np.asarray(indices, dtype=np.int64)


def load_episode(dataset_dir, dataset_name, pimode=False):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".zarr")
    if not os.path.isdir(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        raise SystemExit(1)

    with episode_store.Store(dataset_path, "r") as root:
        require_absolute_actions(root.attrs)
        compressed = root.attrs.get("compress", False)
        is_rollout = root.attrs.get("rollout", False)

        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()] if "/observations/qvel" in root else None
        action = root["/action"][()] if "/action" in root else None
        action_timestamps = root["/action_timestamps"][()] if "/action_timestamps" in root else None
        action_raw = root["/action_raw"][()] if "/action_raw" in root else None
        action_raw_timestamps = root["/action_raw_timestamps"][()] if "/action_raw_timestamps" in root else None

        inference_query_indices = None
        inference_query_timestamps = None
        inference_query_overhead_frame_numbers = None
        inference_chunks = None
        if "/inference/query_indices" in root:
            inference_query_indices = root["/inference/query_indices"][()]
            if "/inference/query_timestamps" in root:
                inference_query_timestamps = root["/inference/query_timestamps"][()]
            if "/inference/query_overhead_frame_numbers" in root:
                inference_query_overhead_frame_numbers = root["/inference/query_overhead_frame_numbers"][()]
        elif "/inference_query_mask" in root:
            query_mask = root["/inference_query_mask"][()]
            if query_mask.ndim == 1:
                inference_query_indices = np.flatnonzero(query_mask).astype(np.int64)
        if "/inference/chunks/actions" in root:
            chunks_group = root["/inference/chunks"]
            inference_chunks = {
                "actions": chunks_group["actions"][()],
                "query_indices": chunks_group["query_indices"][()] if "query_indices" in chunks_group else None,
                "merge_indices": chunks_group["merge_indices"][()] if "merge_indices" in chunks_group else None,
                "start_indices": chunks_group["start_indices"][()] if "start_indices" in chunks_group else None,
                "query_ids": chunks_group["query_ids"][()] if "query_ids" in chunks_group else None,
                "submit_chunk_steps": (
                    chunks_group["submit_chunk_steps"][()] if "submit_chunk_steps" in chunks_group else None
                ),
                "measured_start_indices": (
                    chunks_group["measured_start_indices"][()] if "measured_start_indices" in chunks_group else None
                ),
                "query_qpos": chunks_group["query_qpos"][()] if "query_qpos" in chunks_group else None,
            }

        processed_commanded = None
        if action is not None and len(action) == len(qpos):
            processed_commanded = action
        elif action is not None:
            print(
                "[WARN] /action length does not match /observations/qpos for visualization "
                f"(len(action)={len(action)} len(qpos)={len(qpos)})."
            )
        if is_rollout and processed_commanded is not None:
            print(f"Loaded rollout data: {len(qpos)} actual positions, {len(processed_commanded)} commanded actions")

        image_dict = {}
        images_group = root.get("/observations/images")
        if images_group is not None:
            # Lazy readers: decoding a whole episode up front exhausts RAM on the robot host.
            for cam_name in images_group.keys():
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"]
        annotations = load_episode_annotations(root)
        if (
            annotations.get("subgoal_switch") is None
            and annotations.get("subgoal_switch_frames") is None
            and annotations.get("subgoal_switch_timestamps") is not None
        ):
            annotations["subgoal_switch_frames"] = nearest_frame_indices_for_timestamps(
                action_timestamps,
                annotations["subgoal_switch_timestamps"],
            )

    if image_dict:
        if compressed:
            if pimode:
                print("[PiMode] Rerun images use OpenPI resize_with_pad to 224x224 before display.")
            for cam_name in image_dict.keys():
                image_dict[cam_name] = LazyFrames(
                    image_dict[cam_name],
                    lambda data: prepare_visualization_image(cv2.imdecode(data, 1), pimode=pimode))
        elif pimode:
            print("[PiMode] Rerun images use OpenPI resize_with_pad to 224x224 before display.")
            for cam_name in image_dict.keys():
                image_dict[cam_name] = LazyFrames(
                    image_dict[cam_name], lambda image: openpi_resize_with_pad(image, *OPENPI_IMAGE_RESOLUTION))

    return (
        qpos,
        image_dict,
        qvel,
        processed_commanded,
        action,
        action_timestamps,
        action_raw,
        action_raw_timestamps,
        inference_query_indices,
        inference_query_timestamps,
        inference_query_overhead_frame_numbers,
        inference_chunks,
        annotations,
    )


def resolve_dataset_location(args):
    dataset_dir = args.get("dataset_dir") or DATA_DIR
    episode_idx = args["episode_idx"]
    task_name = args["task_name"]

    if args.get("ismirror"):
        dataset_name = f"mirror_episode_{episode_idx}"
    elif args.get("aloha"):
        dataset_name = f"episode_{episode_idx}"
        dataset_dir = os.path.join(dataset_dir, "aloha")
    else:
        dataset_name = f"episode_{episode_idx:05d}"

    from config.task_config import select_task_config
    task = select_task_config(task_name, args.get('task_config'))
    if task and not args.get('dataset_dir') and not args.get('aloha'):
        return task.as_dict()['data']['directory'], dataset_name
    return os.path.join(dataset_dir, task_name), dataset_name


def save_videos(video, dt, video_path=None):
    cam_names = list(video.keys())
    all_cam_videos = [video[cam_name] for cam_name in cam_names]
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)

    _n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in all_cam_videos:
        out.write(frame)
    out.release()
    print(f"Saved video to: {video_path}")


def ordered_image_dict(image_dict, task_config):
    configured = list((task_config or {}).get("camera_names", []))
    ordered_names = [cam for cam in configured if cam in image_dict]
    ordered_names.extend(cam for cam in image_dict.keys() if cam not in ordered_names)
    return {cam: image_dict[cam] for cam in ordered_names}


def build_inference_lookup(inference_data, num_frames):
    query_idx_set = set()
    query_timestamp_by_idx = {}
    query_overhead_frame_by_idx = {}

    query_indices = inference_data.get("query_indices")
    if query_indices is None:
        return query_idx_set, query_timestamp_by_idx, query_overhead_frame_by_idx

    query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    query_idx_set = set(int(i) for i in query_indices if 0 <= int(i) < num_frames)

    query_timestamps = inference_data.get("query_timestamps")
    if query_timestamps is not None:
        query_timestamps = np.asarray(query_timestamps, dtype=np.float64).reshape(-1)
        for i in range(min(len(query_indices), len(query_timestamps))):
            frame_idx = int(query_indices[i])
            if 0 <= frame_idx < num_frames:
                query_timestamp_by_idx[frame_idx] = float(query_timestamps[i])

    query_overhead = inference_data.get("query_overhead_frame_numbers")
    if query_overhead is not None:
        query_overhead = np.asarray(query_overhead, dtype=np.int64).reshape(-1)
        for i in range(min(len(query_indices), len(query_overhead))):
            frame_idx = int(query_indices[i])
            if 0 <= frame_idx < num_frames:
                query_overhead_frame_by_idx[frame_idx] = int(query_overhead[i])

    return query_idx_set, query_timestamp_by_idx, query_overhead_frame_by_idx


def draw_inference_marker(image, frame_idx, overhead_frame=None):
    marked = image.copy()
    h, w = marked.shape[:2]
    cv2.rectangle(marked, (0, 0), (w - 1, h - 1), (0, 255, 255), 3)
    label = f"INFERENCE t={frame_idx}"
    if overhead_frame is not None and overhead_frame >= 0:
        label += f" overhead={overhead_frame}"
    cv2.putText(
        marked,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return marked


def log_arm_series(prefix, values, suffix=""):
    for i, value in enumerate(values[:6]):
        rr.log(f"{prefix}/joint_{i}{suffix}", rr.Scalars(float(value)))
    if len(values) > 6:
        rr.log(f"{prefix}/gripper{suffix}", rr.Scalars(float(values[6])))


def log_arx_proprio_frame(qpos, qvel, processed_action, is_dual_arm):
    if is_dual_arm:
        log_arm_series("proprio/left_arm/qpos", qpos[:7])
        log_arm_series("proprio/right_arm/qpos", qpos[7:14])
        if qvel is not None:
            log_arm_series("proprio/left_arm/qvel", qvel[:7])
            log_arm_series("proprio/right_arm/qvel", qvel[7:14])
        if processed_action is not None:
            log_arm_series("processed_action/left_arm", processed_action[:7])
            log_arm_series("processed_action/right_arm", processed_action[7:14])
    else:
        log_arm_series("proprio/left_arm/qpos", qpos)
        if qvel is not None:
            log_arm_series("proprio/left_arm/qvel", qvel)
        if processed_action is not None:
            log_arm_series("processed_action/left_arm", processed_action)


def run_arx_data_rerun_visualization(
    args,
    qpos,
    image_dict,
    qvel,
    processed_actions,
    inference_data,
    is_dual_arm,
    annotations=None,
    observation_timestamps=None,
    raw_indices=None,
    frame_provider=None,
    guard=None,
):
    cam_names = list(image_dict.keys())
    num_camera_frames = max((len(image_dict[cam]) for cam in cam_names), default=0)
    num_qpos_frames = len(qpos)
    num_action_frames = len(processed_actions) if processed_actions is not None else 0
    num_frames = max(num_camera_frames, num_qpos_frames, num_action_frames)

    query_idx_set, query_timestamp_by_idx, query_overhead_frame_by_idx = build_inference_lookup(
        inference_data,
        num_frames,
    )
    annotations = annotations or {}
    subgoal_switch_frame_set = build_failure_frame_set(
        annotations.get("subgoal_switch"),
        annotations.get("subgoal_switch_frames"),
        num_frames,
    )
    failure_frame_set = build_failure_frame_set(
        annotations.get("failure"),
        annotations.get("failure_frames"),
        num_frames,
    )

    print(
        "Starting ARX data visualization: "
        f"{len(cam_names)} cameras, qpos={num_qpos_frames}, qvel={len(qvel) if qvel is not None else 0}, "
        f"processed_action={num_action_frames}, frames={num_frames}"
    )
    if cam_names:
        print(f"Cameras: {cam_names}")
    if processed_actions is None:
        print("[WARN] /action is unavailable or not aligned with qpos; processed actions will not be plotted.")
    if query_idx_set:
        print(f"Inference query timesteps: {len(query_idx_set)}")
    if subgoal_switch_frame_set:
        print(f"Subgoal switch overlay enabled: {len(subgoal_switch_frame_set)} frame(s)")
    if failure_frame_set:
        print(f"Failure frame overlay enabled: {len(failure_frame_set)} frame(s)")

    guard = guard or MemoryGuard()
    for frame_idx in range(num_frames):
        guard.check(f"Rerun export at frame {frame_idx}/{num_frames}")
        rr.set_time("frame", sequence=frame_idx)
        if observation_timestamps is not None:
            timestamp = float(observation_timestamps[frame_idx])
            rr.set_time("episode_time", duration=timestamp)
            rr.log("timing/elapsed_seconds", rr.Scalars(timestamp))
            if frame_idx:
                interval = timestamp - float(observation_timestamps[frame_idx - 1])
                rr.log("timing/sample_interval_ms", rr.Scalars(interval * 1000))
        if raw_indices is not None:
            rr.log("timing/robot_index", rr.Scalars(int(raw_indices[frame_idx])))
        rr.log("timeline/timestep", rr.Scalars(float(frame_idx)))

        is_query_frame = frame_idx in query_idx_set
        is_failure_frame = frame_idx in failure_frame_set
        is_subgoal_switch_frame = frame_idx in subgoal_switch_frame_set
        rr.log("inference/is_query_timestep", rr.Scalars(1.0 if is_query_frame else 0.0))
        rr.log("annotations/is_failure_frame", rr.Scalars(1.0 if is_failure_frame else 0.0))
        rr.log("annotations/is_subgoal_switch_frame", rr.Scalars(1.0 if is_subgoal_switch_frame else 0.0))
        if is_query_frame:
            rr.log("inference/query_timestep", rr.Scalars(float(frame_idx)))
        if frame_idx in query_timestamp_by_idx:
            rr.log("inference/query_timestamp", rr.Scalars(query_timestamp_by_idx[frame_idx]))
        if frame_idx in query_overhead_frame_by_idx:
            rr.log(
                "inference/query_overhead_frame_number",
                rr.Scalars(float(query_overhead_frame_by_idx[frame_idx])),
            )

        images = {}
        bundle = frame_provider(frame_idx) if frame_provider is not None else None
        for cam_name in cam_names:
            if bundle is not None:
                image = bundle[cam_name]
            elif frame_idx < len(image_dict[cam_name]):
                image = image_dict[cam_name][frame_idx]
            elif len(image_dict[cam_name]) > 0:
                image = image_dict[cam_name][-1]
            else:
                continue

            images[cam_name] = image if args.get("aloha", False) else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        qpos_frame = qpos[frame_idx] if frame_idx < len(qpos) else qpos[-1]

        overhead_frame = query_overhead_frame_by_idx.get(frame_idx)
        for cam_name, image in images.items():
            camera_image = np.asarray(image)
            if is_query_frame:
                camera_image = draw_inference_marker(camera_image, frame_idx, overhead_frame)
            camera_image = depisode_data_camera_overlay(
                camera_image,
                frame_idx,
                is_subgoal_switch_frame=is_subgoal_switch_frame,
                is_failure_frame=is_failure_frame,
            )
            # JPEG keeps the saved file and the viewer small; raw 720p frames need GBs per camera.
            rr.log(f"cameras/{cam_name}", rr.Image(camera_image).compress(jpeg_quality=VIEW_JPEG_QUALITY))

        qvel_frame = None
        if qvel is not None and len(qvel) > 0:
            qvel_frame = qvel[frame_idx] if frame_idx < len(qvel) else qvel[-1]

        processed_action_frame = None
        if processed_actions is not None and len(processed_actions) > 0:
            processed_action_frame = (
                processed_actions[frame_idx] if frame_idx < len(processed_actions) else processed_actions[-1]
            )

        log_arx_proprio_frame(qpos_frame, qvel_frame, processed_action_frame, is_dual_arm)


def run_rerun_visualization(
    robot_type,
    args,
    qpos,
    image_dict,
    qvel,
    commanded,
    inference_data,
    is_dual_arm,
    annotations=None,
    observation_timestamps=None,
    raw_indices=None,
    frame_provider=None,
    rrd_name="episode",
):
    """Write the whole episode to an .rrd file, then open that file (never a live stream).

    A live viewer buffers every logged frame while logging runs; a finished
    file opens in a viewer whose memory limit leaves the guard's floor free.
    """
    guard = MemoryGuard()
    guard.check("Rerun export")
    output = rrd_output_path(args, rrd_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{os.getpid()}.rrd")
    rr.init(f"{robot_type}_episode_visualization")
    recording = rr.get_global_data_recording()
    rr.save(str(temporary))
    try:
        if observation_timestamps is not None:
            import rerun.blueprint as rrb
            rr.send_blueprint(rrb.Blueprint(rrb.TimePanel(timeline='episode_time'), auto_views=True))

        # Recorded-data plots and camera streams; no robot-specific model is drawn.
        if normalize_robot_type(robot_type) == "arx":
            run_arx_data_rerun_visualization(
                args,
                qpos,
                image_dict,
                qvel,
                commanded,
                inference_data,
                is_dual_arm,
                annotations=annotations,
                observation_timestamps=observation_timestamps,
                raw_indices=raw_indices,
                frame_provider=frame_provider,
                guard=guard,
            )
        recording.flush()
        rr.disconnect(recording=recording)
        recording = None
        os.replace(temporary, output)
    finally:
        if recording is not None:
            rr.disconnect(recording=recording)
        temporary.unlink(missing_ok=True)
    print(f"[Rerun] Saved: {output}", flush=True)
    if not args.get("save_rrd"):
        open_rrd(output, guard)
    return output


def rrd_output_path(args, name):
    if args.get("save_rrd"):
        return Path(args["save_rrd"]).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache / "flash-aloha" / "episode-views" / f"{name}.rrd"


def resolve_chunk_plot_dims(action_dim, arm, explicit_dims=None):
    if explicit_dims:
        dims = [int(dim) for dim in explicit_dims]
        return [dim for dim in dims if 0 <= dim < action_dim]
    if arm == "left":
        return list(range(0, min(7, action_dim)))
    if arm == "right":
        if action_dim >= 14:
            return list(range(7, 14))
        return list(range(0, min(7, action_dim)))
    return list(range(action_dim))


def annotate_chunk_boundary_timestep(ax, x, color):
    ax.plot(
        [x],
        [0.0],
        marker="o",
        color=color,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=6,
    )
    ax.annotate(
        str(int(x)),
        xy=(x, -0.04),
        xycoords=ax.get_xaxis_transform(),
        textcoords=ax.get_xaxis_transform(),
        color=color,
        fontsize=8,
        rotation=90,
        ha="center",
        va="top",
        alpha=0.95,
        annotation_clip=False,
    )


def plot_inference_chunks(
    dataset_dir,
    dataset_name,
    commanded_actions,
    inference_chunks,
    *,
    qpos=None,
    arm="right",
    explicit_dims=None,
    max_chunks=None,
    output_path=None,
):
    if commanded_actions is None:
        print("[Chunk Plot] Skipping: /action is not available.")
        return
    if inference_chunks is None or inference_chunks.get("actions") is None:
        print("[Chunk Plot] Skipping: /inference/chunks/actions is not available.")
        return

    executed = np.asarray(commanded_actions)
    qpos = None if qpos is None else np.asarray(qpos)
    chunks = np.asarray(inference_chunks["actions"])
    if chunks.ndim != 3:
        print(f"[Chunk Plot] Skipping: expected chunk actions shape (N, H, D), got {chunks.shape}.")
        return

    num_chunks, horizon, action_dim = chunks.shape
    dims = resolve_chunk_plot_dims(action_dim, arm, explicit_dims)
    if not dims:
        print(f"[Chunk Plot] Skipping: no valid action dims selected for action_dim={action_dim}.")
        return

    query_indices = inference_chunks.get("query_indices")
    query_indices = (
        np.zeros((num_chunks,), dtype=np.int64)
        if query_indices is None
        else np.asarray(query_indices, dtype=np.int64)
    )
    start_indices = inference_chunks.get("start_indices")
    start_indices = (
        np.zeros((num_chunks,), dtype=np.int64)
        if start_indices is None
        else np.asarray(start_indices, dtype=np.int64)
    )
    merge_indices = inference_chunks.get("merge_indices")
    merge_indices = (
        query_indices + start_indices
        if merge_indices is None
        else np.asarray(merge_indices, dtype=np.int64)
    )
    query_qpos = inference_chunks.get("query_qpos")
    query_qpos = None if query_qpos is None else np.asarray(query_qpos, dtype=np.float32)

    plot_chunk_count = num_chunks if max_chunks is None or max_chunks <= 0 else min(num_chunks, int(max_chunks))
    rows = len(dims)
    fig_height = max(2.2 * rows, 4.0)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(16, fig_height), squeeze=False)
    axes = axes[:, 0]
    frame_x = np.arange(executed.shape[0])
    cmap = plt.get_cmap("tab20")

    for ax, dim in zip(axes, dims):
        if dim < executed.shape[1]:
            ax.plot(frame_x, executed[:, dim], color="black", linewidth=1.6, label="/action executed")
        if qpos is not None and dim < qpos.shape[1]:
            ax.plot(frame_x, qpos[:, dim], color="0.45", linestyle="--", linewidth=1.1, alpha=0.85, label="/qpos actual")
        for chunk_idx in range(plot_chunk_count):
            color = cmap(chunk_idx % 20)
            query_idx = int(query_indices[chunk_idx]) if chunk_idx < len(query_indices) else 0
            start_idx = int(start_indices[chunk_idx]) if chunk_idx < len(start_indices) else 0
            start_idx = min(max(start_idx, 0), horizon - 1)
            merge_idx = int(merge_indices[chunk_idx]) if chunk_idx < len(merge_indices) else query_idx + start_idx
            next_merge_idx = (
                int(merge_indices[chunk_idx + 1])
                if chunk_idx + 1 < min(plot_chunk_count, len(merge_indices))
                else None
            )
            exec_end_idx = min(query_idx + horizon - 1, next_merge_idx - 1) if next_merge_idx is not None else query_idx + horizon - 1
            full_end_idx = query_idx + horizon - 1

            full_x = query_idx + np.arange(horizon)
            ax.plot(full_x, chunks[chunk_idx, :, dim], color=color, alpha=0.22, linewidth=1.0)
            if query_qpos is not None and chunk_idx < len(query_qpos) and dim < query_qpos.shape[1]:
                query_qpos_value = query_qpos[chunk_idx, dim]
                if np.isfinite(query_qpos_value):
                    ax.scatter(
                        [query_idx],
                        [query_qpos_value],
                        color=color,
                        marker="x",
                        s=34,
                        linewidths=1.5,
                        zorder=6,
                    )

            if start_idx > 0:
                prefix_x = query_idx + np.arange(start_idx)
                ax.plot(
                    prefix_x,
                    chunks[chunk_idx, :start_idx, dim],
                    color=color,
                    alpha=0.72,
                    linewidth=1.25,
                    linestyle=":",
                )

            active_x = merge_idx + np.arange(horizon - start_idx)
            ax.plot(
                active_x,
                chunks[chunk_idx, start_idx:, dim],
                color=color,
                alpha=0.78,
                linewidth=1.3,
            )
            ax.scatter([merge_idx], [chunks[chunk_idx, start_idx, dim]], color=color, s=18, zorder=4)
            ax.axvline(query_idx, color=color, alpha=0.10, linewidth=0.8)
            ax.axvline(merge_idx, color=color, alpha=0.18, linestyle="--", linewidth=0.8)
            if ax is axes[-1]:
                annotate_chunk_boundary_timestep(ax, merge_idx, color)
            if exec_end_idx < full_end_idx:
                ax.axvspan(exec_end_idx + 1, full_end_idx, color=color, alpha=0.035)

        ax.set_ylabel(f"a[{dim}]")
        ax.grid(True, alpha=0.2)

    axes[0].set_title(f"{dataset_name}: recorded inference chunks ({plot_chunk_count}/{num_chunks}) over /action")
    axes[-1].set_xlabel("rollout frame (numbers under axis mark executed chunk handoff timesteps)")
    axes[0].legend(loc="upper right")
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if output_path is None:
        suffix = arm if not explicit_dims else "dims_" + "_".join(str(dim) for dim in dims)
        output_path = os.path.join(dataset_dir, f"{dataset_name}_inference_chunks_{suffix}.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[Chunk Plot] Saved {output_path}")


def visualize_episode_data(args):
    """Raw support is inspection-only; never route this input into robot replay."""
    if (args.get('action_replay')
            or args.get('save_video')
            or args.get('plot_inference_chunks') or args.get('aloha')):
        raise ValueError('--data-dir supports Rerun visualization only; use --rerun or --save-rrd.')
    from visualization.episode_visualization import VisualizationEpisode
    transform = lambda image: prepare_visualization_image(image, pimode=args.get('pimode', False))
    with VisualizationEpisode(args['data_dir'], transform=transform) as episode:
        print(f'Raw streams: {episode.directory}')
        print(f'{len(episode.qpos)} retained observations; {episode.timestamps[-1]:.3f} seconds. '
              'Rerun episode_time uses saved timestamps; frame uses retained observation indices.')
        if not episode.manifest.get('complete'):
            print('[WARN] Unfinished task: visualizing the saved portion only.')
        inference_data = dict(query_indices=None, query_timestamps=None,
                              query_overhead_frame_numbers=None, chunks=None)
        run_rerun_visualization(args.get('robot', 'arx'), args, episode.qpos, episode.images, episode.qvel,
            episode.actions, inference_data, episode.qpos.shape[1] == 14,
            observation_timestamps=episode.timestamps, raw_indices=episode.retained_indices,
            frame_provider=episode.frame_bundle,
            rrd_name='_'.join(part.removesuffix('.zarr') for part in Path(episode.directory).resolve().parts[-2:]))


def main(args):
    if args.get('prepare_only') and not args.get('training_config'):
        raise ValueError('--prepare-only requires --training-config; use --save-rrd for a recorded-resolution export')
    if args.get('training_config'):
        from visualization.training_preview import visualize_training_episode
        return visualize_training_episode(args)
    if args.get('data_dir'):
        return visualize_episode_data(args)
    if args.get('review_replay'):
        return run_review_replay(args)
    robot_type = normalize_robot_type(args["robot"])
    task_name = args["task_name"]
    task_config = TASK_CONFIGS.get(task_name, {})
    from config.task_config import select_task_config, require_task_value
    yaml_task = select_task_config(task_name, args.get('task_config'))
    if yaml_task:
        task_config = yaml_task.legacy(task_config)
        if args.get('action_replay'):
            require_task_value(robot_type, task_config['robot_type'], 'robot')
    if task_config.get("robot_type") and normalize_robot_type(task_config["robot_type"]) != robot_type:
        print(f"[WARN] Task config robot_type={task_config['robot_type']} but --robot {robot_type} was requested.")

    dataset_dir, dataset_name = resolve_dataset_location(args)

    (
        qpos,
        image_dict,
        qvel,
        commanded,
        action,
        action_timestamps,
        action_raw,
        action_raw_timestamps,
        inference_query_indices,
        inference_query_timestamps,
        inference_query_overhead_frame_numbers,
        inference_chunks,
        annotations,
    ) = load_episode(dataset_dir, dataset_name, pimode=args.get("pimode", False))
    episode_attrs = {}
    if args.get("action_replay") or args.get("save_video"):
        with episode_store.Store(os.path.join(dataset_dir, dataset_name + ".zarr"), "r") as root:
            episode_attrs = dict(root.attrs)
            requested_fps = args.get("fps")
            raw_replay = args.get("action_replay") and args.get("action_source") == "raw"
            args["fps"] = resolve_rate(root.attrs, None if raw_replay else requested_fps,
                legacy_rate=args.get("legacy_episode_fps"), context="replay episode")
            args["replay_fps"] = (resolve_rate(
                {"action_rate_hz": root.attrs.get("raw_action_rate_hz")}, requested_fps,
                legacy_rate=args.get("legacy_raw_action_fps"), context="raw replay actions")
                if raw_replay else args["fps"])
            if yaml_task and args.get('action_replay'):
                # Tagged episodes must also match the task's robot models and roles.
                tagged = ({'robot_contract': episode_attrs['robot_contract']}
                          if 'robot_contract' in episode_attrs else {})
                yaml_task.require_recording({'fps': args['fps'], **tagged})
                yaml_task.require_recording({'fps': args['replay_fps'], **tagged})
            source_timestamps = root["timestamps"][()] if "timestamps" in root else None
            if action_timestamps is None and action is not None and len(action) == len(qpos):
                action_timestamps = source_timestamps
    else:
        source_timestamps = None
    image_dict = ordered_image_dict(image_dict, task_config)

    print("episode loaded!!")
    if not image_dict:
        print("No images found in episode. Rerun will visualize robot state only.")
    if commanded is not None:
        print(f"Processed action visualization enabled: {len(qpos)} qpos, {len(commanded)} processed actions")
    print(f"Processed action stream available: {action is not None}")
    print(f"Raw action stream available: {action_raw is not None}")

    action_dim = len(qpos[0]) if len(qpos) > 0 else 7
    is_dual_arm = action_dim == 14
    print(f"Action dimension: {action_dim}, Dual arm: {is_dual_arm}")
    if args.get("single_arm", False) and is_dual_arm:
        print("Warning: Data appears to be dual arm but --single_arm flag is set. Using data dimension.")

    inference_data = {
        "query_indices": inference_query_indices,
        "query_timestamps": inference_query_timestamps,
        "query_overhead_frame_numbers": inference_query_overhead_frame_numbers,
        "chunks": inference_chunks,
    }

    if args.get("plot_inference_chunks", False):
        plot_inference_chunks(
            dataset_dir,
            dataset_name,
            commanded,
            inference_chunks,
            qpos=qpos,
            arm=args.get("chunk_plot_arm", "right"),
            explicit_dims=args.get("chunk_plot_dims"),
            max_chunks=args.get("chunk_plot_max_chunks"),
            output_path=args.get("chunk_plot_output"),
        )

    if args.get("rerun", False) or args.get("save_rrd"):
        run_rerun_visualization(
            robot_type,
            args,
            qpos,
            image_dict,
            qvel,
            commanded,
            inference_data,
            is_dual_arm,
            annotations=annotations,
            rrd_name=f"{task_name}_{dataset_name}",
        )

    if args.get("save_video", False):
        if not image_dict:
            print("[WARN] Skipping --save_video because this episode has no images.")
        else:
            video_fps = replay_utils.resolve_playback_fps(args.get("fps"))
            save_videos(image_dict, 1.0 / video_fps, video_path=os.path.join(dataset_dir, dataset_name + "_video.mp4"))

    if args.get("action_replay", False):
        try:
            replay_actions, replay_action_v, _replay_timestamps, replay_source_label = replay_utils.select_replay_stream(
                args.get("action_source", "processed"),
                qpos,
                qvel,
                action,
                action_timestamps,
                action_raw,
                action_raw_timestamps,
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        if args.get("no_track_vel", False):
            replay_action_v = np.zeros_like(replay_actions, dtype=np.float32)
            print("[Replay] --no-track-vel enabled: target velocities set to zero.")

        if _replay_timestamps is None and len(replay_actions) == len(qpos):
            _replay_timestamps = source_timestamps
        replay_fps = replay_utils.resolve_playback_fps(args["replay_fps"])
        replay_dt = 1.0 / replay_fps

        replay_utils.DT = replay_dt


        print(f"[Replay] Source={replay_source_label}, FPS={replay_fps:.2f}, Timesteps={len(replay_actions)}")
        if robot_type == "arx":
            left_can, right_can = replay_utils.robot_interfaces(robot_type, task_config, is_dual_arm)
            single_arm_side = None if is_dual_arm else resolve_arx_single_arm_side(task_config)
            replay_utils.replay_arx_actions(
                replay_actions,
                action_v_list=replay_action_v,
                is_dual_arm=is_dual_arm,
                task_name=task_name,
                episode_idx=args["episode_idx"],
                dt=replay_dt,
                replay_source=replay_source_label,
                use_fill_in=args.get("fill_in", False),
                plot_tracking=args.get("plot_replay_graph", True),
                left_can=left_can,
                right_can=right_can or "can1",
                kp=args.get("arm_kp", None),
                kd=args.get("arm_kd", None),
                single_arm_side=single_arm_side,
                plot_torque_minus_gravity=args.get("plot_torque_minus_gravity", False),
            )


def review_episode_paths(args):
    """Published episode paths for --review_replay, in the order given."""
    paths = []
    for episode_idx in args["review_replay"]:
        dataset_dir, dataset_name = resolve_dataset_location({**args, "episode_idx": episode_idx})
        paths.append(os.path.join(dataset_dir, dataset_name + ".zarr"))
    return paths


def run_review_replay(args):
    """The collector's review replay for saved episodes, in the order given.

    Each episode returns to its recorded start pose, shows its first frame over
    the live camera, asks for the kp/kd fractions, and replays on '1'. The
    collector's backend, camera cache, key watcher, gain prompt, replay_take
    and exits run unchanged; this only selects the episodes. Every episode is
    loaded and checked against the task before anything moves.
    """
    from recording import record_episode_single as collector
    from recording.recording_replay import load_replay_episode
    from hardware.safe_exit import damping_exit, raising_signals

    ctx = collector.resolve_run_context(
        {"task_name": args["task_name"], "task_config": args.get("task_config"),
         "robot": normalize_robot_type(args["robot"])})
    paths = review_episode_paths(args)
    episodes = [load_replay_episode(path, camera_names=ctx.camera_names,
                                    robot_contract=ctx.robot_contract, fps=ctx.fps)
                for path in paths]

    with raising_signals():
        stop_event, camera_thread = collector.start_camera_capture(ctx, use_camera_server=True)
        backend = None
        failed = False

        def stop_cameras():
            # As in the collector: close the camera cache before homing.
            stop_event.set()
            camera_thread.join(timeout=2.0)

        def damping_target():
            if backend is None:
                return None
            backend.before_home = stop_cameras
            return backend.damping_target()

        try:
            # Any failure, including Ctrl+C, damps until 'h', homes, then exits.
            with damping_exit(damping_target, home_requested=collector.home_key_pressed):
                backend = collector.build_backend(ctx)
                backend.setup_for_recording()
                review_episodes(collector, backend, ctx, args["review_replay"], episodes)
        except BaseException:
            failed = True  # The failure exit already homed or disabled in place.
            raise
        finally:
            stop_event.set()
            try:
                camera_thread.join(timeout=2.0)
            finally:
                if backend is not None:
                    backend.shutdown(home=not failed)
            print("Shutdown complete.")


def review_episodes(collector, backend, ctx, episode_indices, episodes):
    """Replay each loaded episode; r repeats, Enter moves on (cycling), x finishes."""
    from recording.recording_replay import replay_take

    position = 0
    while True:
        actions, camera, reference = episodes[position]
        label = f"episode {episode_indices[position]}"
        print(f"\n[Review] {label} ({position + 1}/{len(episodes)})")
        result = "cancelled"
        gains = collector.prompt_replay_gain_scales()
        if gains is not None:
            collector.wait_for_cameras(ctx.camera_names, timeout_sec=15.0)
            result = replay_take(
                backend, actions, camera, reference, label=label, gain_scales=gains,
                poll_keys=lambda: [key for key in (collector.read_key_nonblocking(),) if key],
                live_frame=collector._camera_global)
        if result == "quit":
            return
        following = episode_indices[(position + 1) % len(episodes)]
        print(f"\nPress 'r' to replay {label} again, Enter for episode {following}, or 'x' to finish.")
        while True:
            key = collector.read_key_blocking()
            if key == "x":
                return
            if key.lower() == "r":
                break
            if key in ("\n", "\r", "enter"):
                position = (position + 1) % len(episodes)
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, choices=["arx"], default="arx", help="Robot backend (ARX).")
    parser.add_argument("--task_name", action="store", type=str, help="Task name (required for Zarr).")
    parser.add_argument("--task-config", dest="task_config", help="Task YAML for replay hardware and FPS validation.")
    parser.add_argument("--dataset_dir", action="store", type=str, help="Dataset root override; otherwise use the task YAML directory.", default=None)
    parser.add_argument("--episode_idx", action="store", type=int, help="Episode index (required for Zarr).")
    parser.add_argument('--data-dir', type=Path,
                        help='Raw episode, segment, or raw stream directory; visualize without Zarr export.')
    parser.add_argument('--save-rrd', type=Path,
                        help='Save Rerun visualization to this .rrd file instead of opening the viewer.')
    parser.add_argument('--training-config', help='Use only camera names/resolution from a training YAML or catalog name.')
    parser.add_argument('--preview-frame', type=int,
                        help='Preview only this recorded observation index (default: all recorded samples).')
    parser.add_argument('--prepare-only', action='store_true',
                        help='With --training-config, prepare an offline .rrd without opening Rerun.')
    parser.add_argument("--ismirror", action="store_true", help="Use mirror_episode_<idx> naming.")
    parser.add_argument("--action_replay", action="store_true", help="Replay actions on physical robot.", default=False)
    parser.add_argument(
        "--review_replay",
        type=int,
        nargs="+",
        metavar="EPISODE",
        default=None,
        help="Replay these saved episodes in order with the collector's start-pose overlay "
             "and kp/kd fraction prompt.",
    )
    parser.add_argument("--save_video", action="store_true", default=False, help="Save the video from episode.")
    parser.add_argument("--rerun", action="store_true", default=False, help="Use rerun visualizer.")
    parser.add_argument(
        "--pimode",
        action="store_true",
        default=False,
        help="Render episode images with OpenPI's 224x224 resize_with_pad preprocessing.",
    )
    parser.add_argument(
        "--plot_inference_chunks",
        action="store_true",
        default=False,
        help="Save a PNG overlay of recorded full inference chunks against /action.",
    )
    parser.add_argument(
        "--chunk_plot_arm",
        type=str,
        choices=["left", "right", "both"],
        default="right",
        help="Arm dimensions to plot for --plot_inference_chunks. Default: right.",
    )
    parser.add_argument(
        "--chunk_plot_dims",
        type=int,
        nargs="+",
        default=None,
        help="Explicit action dimensions to plot for --plot_inference_chunks.",
    )
    parser.add_argument(
        "--chunk_plot_max_chunks",
        type=int,
        default=0,
        help="Maximum number of chunks to overlay; 0 plots all recorded chunks.",
    )
    parser.add_argument(
        "--chunk_plot_output",
        type=str,
        default=None,
        help="Output PNG path for --plot_inference_chunks.",
    )
    parser.add_argument("--single_arm", action="store_true", help="Single arm mode hint.", default=False)
    parser.add_argument("--aloha", action="store_true", default=False, help="Use ALOHA dataset naming/layout.")
    parser.add_argument("--urdf_path", action="store", type=str, default=None, help="Override ARX URDF path.")
    parser.add_argument(
        "--arm_kp",
        action="store",
        type=float,
        nargs=7,
        default=None,
        help="Proportional gains for 6 joints + gripper.",
    )
    parser.add_argument(
        "--arm_kd",
        action="store",
        type=float,
        nargs=7,
        default=None,
        help="Derivative gains for 6 joints + gripper.",
    )
    parser.add_argument(
        "--fps",
        action="store",
        type=float,
        default=None,
        help="Replay FPS; defaults to episode metadata and must match it.",
    )
    parser.add_argument(
        "--action_source",
        action="store",
        type=str,
        default="processed",
        choices=["processed", "raw"],
        help="Action stream for physical replay: processed=/action, raw=/action_raw.",
    )
    parser.add_argument("--legacy-episode-fps", type=float, default=None, help="Explicit FPS for a legacy episode without rate metadata.")
    parser.add_argument("--legacy-raw-action-fps", type=float, default=None, help="Explicit FPS for action_raw without raw_action_rate_hz metadata.")
    parser.add_argument("--fill_in", action="store_true", help="Enable interpolation fill-in for action replay.")
    parser.add_argument(
        "--no_replay_graph",
        action="store_true",
        help="Disable end-of-replay tracking graph generation and skip recording replay states.",
    )
    parser.add_argument(
        "--plot_torque_minus_gravity",
        action="store_true",
        help="For ARX replay graphs, also plot measured joint torque minus ARX inverse-dynamics gravity torque.",
    )
    parser.add_argument(
        "--no-track-vel",
        action="store_true",
        dest="no_track_vel",
        help="Set replay target velocities to zero instead of using dataset qvel.",
    )

    parsed_args = vars(parser.parse_args())
    if parsed_args['review_replay']:
        if parsed_args['task_name'] is None:
            parser.error('--review_replay requires --task_name')
        conflicts = [flag for flag, key in (('--action_replay', 'action_replay'), ('--data-dir', 'data_dir'),
                                            ('--training-config', 'training_config'), ('--arm_kp', 'arm_kp'),
                                            ('--arm_kd', 'arm_kd'), ('--fill_in', 'fill_in'))
                     if parsed_args[key]]
        if conflicts:
            parser.error(f"--review_replay uses the collector's replay and gain prompt; omit {', '.join(conflicts)}")
    elif not parsed_args['data_dir'] and (parsed_args['task_name'] is None or parsed_args['episode_idx'] is None):
        parser.error('--task_name and --episode_idx are required unless --data-dir is used')
    parsed_args["plot_replay_graph"] = not parsed_args.pop("no_replay_graph")
    return parsed_args


if __name__ == "__main__":
    main(parse_args())
