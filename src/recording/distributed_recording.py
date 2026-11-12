"""Distributed raw recording for live policy rollouts.

The live control process only places small numeric samples on a bounded
``multiprocessing.Queue``.  A coordinator process owns camera lifecycle
barriers and the append-only Zarr control stream.  Episode publication selects saved stream rows and adds Zarr annotations after
robot operation has finished, without decoding or duplicating camera video.

This module deliberately has no dependency on the robot or policy client so
that lifecycle and conversion behavior can be tested without hardware.
"""

from __future__ import annotations

# Support direct script execution and package imports from any working directory.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

from recording.recording_utils import _json_value, atomic_write_json

from contracts import capture_gc
from contracts.action_contract import require_absolute_actions
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
import uuid
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import zarr
from tqdm.auto import tqdm
from contracts.standard_capture import CAPTURE_CONTRACT_VERSION, validate_capture_payload
from camera_utils.camera_timing import TimingLimits, check_camera_timestamps, read_shared_camera_timestamps, CameraTimingError, require_camera_timestamps


CONTROL_QUEUE_CAPACITY = 512
CAMERA_ENDPOINT_OFFSETS = {
    "left": 0,
    "overhead": 1,
    "right": 2,
    "torso": 3,
    "test": 4,
    "extra": 5,
    "extra_overhead": 6,
}
TERMINAL_CAMERA_STATES = {"paused", "ready", "stopped", "error"}
DAGGER_SCHEMA_VERSION = 2
DAGGER_CAMERA_NAMES = ("video_overhead", "video_left", "video_right")
DAGGER_JPEG_QUALITY = 95
DEFAULT_CONVERSION_WORKERS = 8


class RecorderError(RuntimeError):
    """A fatal error from the coordinator or one of its cameras."""


def camera_video_encoding_from_statuses(
    statuses: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract stable source-video provenance from camera start acknowledgements."""
    result = {}
    for camera_name, status in statuses.items():
        encoder = status.get("video_encoder")
        if isinstance(encoder, dict):
            result[str(camera_name)] = _json_value(encoder)
    return result


def update_manifest_decision(
    manifest_path: str | Path,
    *,
    decision: str,
    retained: bool,
) -> dict[str, Any]:
    """Atomically update a stopped raw DAgger manifest for web finalization.

    A completed web-DAgger subtask is intentionally stopped with a pending
    decision before scripted reset motion. Browser Save later promotes that
    recoverable manifest to retained/keep; Abort promotes it to discard while
    leaving the raw data available for audit or recovery.
    """
    if decision not in {"pending", "keep", "failure", "discard"}:
        raise ValueError(f"unsupported DAgger manifest decision {decision!r}")
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("complete"):
        raise RecorderError(f"Cannot finalize incomplete raw manifest: {path}")
    output = manifest.setdefault("outputs", {}).setdefault(
        "dagger", {"requested": True, "decision": None}
    )
    if not output.get("requested"):
        raise RecorderError(f"Manifest did not request DAgger output: {path}")
    previous = output.get("decision")
    converted_output = (
        (manifest.get("publication") or {}).get("outputs", {}).get("dagger")
    )
    may_discard_unconverted_keep = bool(
        decision == "discard"
        and previous in {"keep", "failure"}
        and not (converted_output and Path(converted_output).is_dir())
    )
    if (
        previous in {"keep", "failure", "discard"}
        and previous != decision
        and not may_discard_unconverted_keep
    ):
        raise RecorderError(
            f"Manifest decision is already {previous!r}, cannot change to {decision!r}: {path}"
        )
    output["decision"] = decision
    manifest["retained"] = bool(retained)
    manifest["web_finalized_at"] = time.time()
    atomic_write_json(path, manifest)
    return manifest


def dagger_intervention_ranges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open contiguous true ranges for an aligned intervention mask."""
    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 1:
        raise ValueError("intervention mask must be one-dimensional")
    active = values != 0
    padded = np.pad(active.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1).astype(np.int64)
    ends = np.flatnonzero(edges == -1).astype(np.int64)
    return starts, ends


def camera_endpoint(camera_name: str, endpoint_base: int) -> str:
    short_name = camera_name.removeprefix("video_")
    if short_name not in CAMERA_ENDPOINT_OFFSETS:
        raise ValueError(f"Unknown camera name: {camera_name}")
    return f"tcp://127.0.0.1:{int(endpoint_base) + CAMERA_ENDPOINT_OFFSETS[short_name]}"


class CameraBarrierClient:
    """Command/ack client for the camera-server recording protocol."""

    def __init__(
        self,
        camera_names: Iterable[str],
        endpoint_base: int,
        *,
        client_factory: Callable[..., Any] | None = None,
        status_timeout: float = 2.0,
    ):
        if client_factory is None:
            from robotmq import RMQClient

            client_factory = RMQClient
        self.camera_names = list(camera_names)
        self.status_timeout = float(status_timeout)
        self.clients = {
            name: client_factory(
                client_name=f"distributed_recorder_{name}_{os.getpid()}",
                server_endpoint=camera_endpoint(name, endpoint_base),
            )
            for name in self.camera_names
        }
        self.last_status: dict[str, dict[str, Any]] = {}
        self.last_heartbeat_monotonic: dict[str, float] = {}

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> bytes:
        from robotmq import serialize

        return serialize(payload)

    @staticmethod
    def _deserialize(payload: bytes) -> Any:
        from robotmq import deserialize

        return deserialize(payload)

    def wait_until_connected(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        pending = set(self.camera_names)
        while pending and time.monotonic() < deadline:
            for name in tuple(pending):
                if self.clients[name].get_topic_status("info", timeout_s=0.05) >= 0:
                    pending.remove(name)
            if pending:
                time.sleep(0.01)
        if pending:
            raise RecorderError(f"Camera logger connection timeout: {sorted(pending)}")

    def _read_statuses(self, camera_name: str) -> list[dict[str, Any]]:
        client = self.clients[camera_name]
        raw, _ = client.pop_data("info", n=0, timeout_s=0.02)
        statuses = []
        for item in raw:
            decoded = self._deserialize(item)
            if not isinstance(decoded, dict) or "state" not in decoded:
                continue
            statuses.append(decoded)
            self.last_status[camera_name] = decoded
            self.last_heartbeat_monotonic[camera_name] = time.monotonic()
        return statuses

    def poll(self) -> dict[str, dict[str, Any]]:
        for name in self.camera_names:
            self._read_statuses(name)
        return dict(self.last_status)

    def assert_healthy(self) -> None:
        now = time.monotonic()
        self.poll()
        for name in self.camera_names:
            status = self.last_status.get(name)
            if status is not None and status.get("fatal_error"):
                raise RecorderError(f"Camera {name} failed: {status['fatal_error']}")
            last = self.last_heartbeat_monotonic.get(name)
            if last is not None and now - last > self.status_timeout:
                raise RecorderError(
                    f"Camera {name} heartbeat lost for {now - last:.3f}s"
                )

    def command(
        self,
        command_type: str,
        *,
        session_id: str,
        episode_dir: str | None = None,
        timeout: float,
        command_id: str | None = None,
        ready_state: str | None = None,
        payload_extra: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        command_id = command_id or uuid.uuid4().hex
        payload = {
            "type": str(command_type),
            "session_id": str(session_id),
            "command_id": command_id,
        }
        if episode_dir is not None:
            payload["episode_dir"] = str(episode_dir)
        if payload_extra:
            payload.update(payload_extra)
        message = self._serialize(payload)
        for client in self.clients.values():
            client.put_data("command", message, timeout_s=0.1)

        expected = ready_state or {
            "start": "ready",
            "pause": "paused",
            "resume": "ready",
            "stop": "stopped",
        }[command_type]
        pending = set(self.camera_names)
        acknowledgements: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + float(timeout)
        while pending and time.monotonic() < deadline:
            for name in tuple(pending):
                for status in self._read_statuses(name):
                    if status.get("fatal_error"):
                        raise RecorderError(
                            f"Camera {name} failed during {command_type}: "
                            f"{status['fatal_error']}"
                        )
                    if (
                        status.get("session_id") == session_id
                        and status.get("command_id") == command_id
                        and status.get("state") == expected
                    ):
                        acknowledgements[name] = status
                        pending.remove(name)
                        break
            if pending:
                time.sleep(0.005)
        if pending:
            details = {name: self.last_status.get(name) for name in sorted(pending)}
            raise RecorderError(
                f"Timed out waiting for camera {command_type}/{expected} barrier: {details}"
            )
        return acknowledgements


@dataclass
class EpisodeRequest:
    session_id: str
    data_dir: str
    camera_names: list[str]
    record_rollout: bool
    record_dagger: bool
    rollout_dir: str
    rollout_task_name: str
    dagger_dir: str
    dagger_task_name: str
    dagger_camera_names: list[str]
    is_double_arm: bool
    fps: float
    eval_mode: bool = False
    rollout_date: str | None = None
    metadata: dict[str, Any] | None = None


class ControlWriter:
    """Append a typed, numeric 100 Hz control stream to Zarr."""

    def __init__(self, path: str | Path, *, capture_limits=None):
        self.path = Path(path)
        self.group = zarr.open_group(str(self.path), mode="w")
        self.datasets: dict[str, Any] = {}
        self.count = 0
        self.capture_limits = capture_limits

    def _dataset(self, name: str, value: np.ndarray, dtype: Any) -> Any:
        if name not in self.datasets:
            shape = (0, *value.shape)
            chunks = (min(512, CONTROL_QUEUE_CAPACITY), *(max(1, size) for size in value.shape))
            self.datasets[name] = self.group.create_dataset(
                name, shape=shape, chunks=chunks, dtype=dtype
            )
        return self.datasets[name]

    def append(self, sample: dict[str, Any]) -> None:
        require_absolute_actions(sample)
        if not self.count:
            self.group.attrs["action_command_style"] = "absolute"
        if self.count and sample.get('capture_contract_version') != self.group.attrs.get('capture_contract_version'):
            raise RecorderError('Capture contract version cannot change within a control stream')
        if sample.get('capture_contract_version') is not None:
            validate_capture_payload(sample, sample['camera_observation']['cameras'], limits=self.capture_limits)
            self.group.attrs['capture_contract_version'] = CAPTURE_CONTRACT_VERSION
        required = ("timestamp", "qpos", "qvel", "commanded_action")
        missing = [name for name in required if name not in sample]
        if missing:
            raise RecorderError(f"Control sample missing required fields: {missing}")
        values = {
            "timestamps": (np.asarray(sample["timestamp"]), np.float64),
            "qpos": (np.asarray(sample["qpos"]), np.float32),
            "qvel": (np.asarray(sample["qvel"]), np.float32),
            "commanded_action": (
                np.asarray(sample["commanded_action"]),
                np.float32,
            ),
            "policy_action": (
                np.asarray(sample.get("policy_action", sample["commanded_action"])),
                np.float32,
            ),
            "continuity_id": (np.asarray(sample.get("continuity_id", 0)), np.int64),
            "segment_id": (np.asarray(sample.get("segment_id", 0)), np.int64),
            "metadata_id": (np.asarray(sample.get("metadata_id", 0)), np.int64),
            "intervention_active": (
                np.asarray(bool(sample.get("intervention_active", False))),
                np.uint8,
            ),
            "inference_query": (
                np.asarray(bool(sample.get("inference_query", False))),
                np.uint8,
            ),
            "inference_query_id": (
                np.asarray(sample.get("inference_query_id", -1)),
                np.int64,
            ),
            "inference_query_timestamp": (
                np.asarray(sample.get("inference_query_timestamp", np.nan)),
                np.float64,
            ),
            "inference_query_overhead_frame_number": (
                np.asarray(sample.get("inference_query_overhead_frame_number", -1)),
                np.int64,
            ),
        }
        # Optional collection timing is separate from the camera clocks and
        # the edited episode timeline. Existing inference schemas are unchanged.
        for name in ("elapsed_seconds", "sampled_at", "capture_interval", "robot_read_started_at"):
            if name in sample:
                values[name] = (np.asarray(sample[name]), np.float64)
        observation = sample.get('camera_observation')
        has_references = 'camera_observation' in self.group
        if self.count and bool(observation) != has_references:
            raise RecorderError('Camera references must be present for every control sample, or none in a legacy stream')
        if observation:
            names = list(observation['cameras'])
            group = self.group.require_group('camera_observation')
            if self.count and set(group.attrs['camera_names']) != set(names):
                raise RecorderError('Camera reference names changed during recording')
            group.attrs.update(camera_names=names, alignment_rule='selected_consumer_frame_ids')
            if sample.get('capture_contract_version') is not None:
                group.attrs['capture_contract_version'] = CAPTURE_CONTRACT_VERSION
            values['camera_observation/snapshot_timestamp'] = (np.asarray(observation['snapshot_timestamp']), np.float64)
            for name, reference in observation['cameras'].items():
                for field, dtype in (('frame_number', np.int64), ('capture_timestamp', np.float64),
                                     ('consumer_received_timestamp', np.float64)):
                    values[f'camera_observation/{name}/{field}'] = (np.asarray(reference[field]), dtype)
        for name, (value, dtype) in values.items():
            value = np.asarray(value, dtype=dtype)
            dataset = self._dataset(name, value, dtype)
            dataset.resize((self.count + 1, *value.shape))
            dataset[self.count] = value
        self.count += 1

    def append_action_chunk(self, payload: dict[str, Any]) -> None:
        group = self.group.require_group("inference_chunks")
        chunk = np.asarray(payload["actions"], dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise RecorderError(
                f"Inference chunk must have (horizon, action_dim), got {chunk.shape}"
            )
        index = int(group.attrs.get("count", 0))
        item = group.require_group(f"{index:06d}")
        item.create_dataset("actions", data=chunk, dtype=np.float32)
        for key, value in payload.items():
            if key != "actions":
                item.attrs[key] = _json_value(value)
        group.attrs["count"] = index + 1


def _coordinator_main(
    config: dict[str, Any],
    control_queue: Any,
    command_queue: Any,
    event_queue: Any,
) -> None:
    camera_client = None
    writer = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    metadata_stream = None
    collection_stream = health_stream = None
    camera_progress = {}
    control_queue_high_water = 0
    max_control_write_delay = 0.

    def control_backlog():
        nonlocal control_queue_high_water
        try:
            depth = control_queue.qsize()
        except (NotImplementedError, AttributeError):
            depth = None
        if depth is not None:
            control_queue_high_water = max(control_queue_high_water, depth)
        return dict(queue_depth=depth, queue_capacity=CONTROL_QUEUE_CAPACITY,
                    queue_high_water=control_queue_high_water,
                    max_write_delay_seconds=max_control_write_delay)

    def drain_control_samples(
        *,
        max_items: int | None = 128,
        expected_count: int | None = None,
        timeout: float = 0.0,
    ) -> None:
        """Commit queued controls, optionally waiting for an exact producer barrier."""
        nonlocal max_control_write_delay
        if collection_stream is not None:
            control_backlog()
        if writer is None or manifest is None:
            if expected_count not in {None, 0}:
                raise RecorderError(
                    f"Cannot commit {expected_count} samples without an active episode"
                )
            return
        deadline = time.monotonic() + max(0.0, float(timeout))
        drained = 0
        while max_items is None or drained < max_items:
            if expected_count is not None and writer.count >= expected_count:
                break
            try:
                if expected_count is not None and time.monotonic() < deadline:
                    sample = control_queue.get(
                        timeout=min(0.01, max(0.0, deadline - time.monotonic()))
                    )
                else:
                    sample = control_queue.get_nowait()
            except queue.Empty:
                if expected_count is not None and writer.count < expected_count:
                    if time.monotonic() < deadline:
                        continue
                    raise RecorderError(
                        "Timed out committing control samples before recorder boundary: "
                        f"committed={writer.count}, expected={expected_count}"
                    )
                break
            if writer.count == 0:
                manifest["first_active_control_timestamp"] = float(sample["timestamp"])
            if config.get('capture_contract_version'):
                validate_capture_payload(sample, config['camera_names'],
                                         limits=TimingLimits(**config['timing_limits']))
            writer.append(sample)
            if collection_stream is not None:
                max_control_write_delay = max(max_control_write_delay, time.monotonic() - sample["timestamp"])
                collection_stream.write(json.dumps(_json_value(dict(
                    raw_index=writer.count - 1, label=sample.get("collection_label", {})))) + "\n")
                if writer.count % 100 == 0:
                    collection_stream.flush()
            drained += 1
        manifest["sample_count"] = writer.count

    try:
        # This process carries the control stream for the whole session.
        capture_gc.pause_for_stream()
        camera_client = CameraBarrierClient(
            config["camera_names"],
            config["camera_logger_endpoint_base"],
            status_timeout=config["status_timeout"],
        )
        camera_client.wait_until_connected(config["warmup_timeout"])
        event_queue.put({"type": "health", "state": "idle", "sample_count": 0})
        running = True
        last_health_event = 0.0
        while running:
            try:
                command = command_queue.get(timeout=0.01)
            except queue.Empty:
                command = None

            if command is not None:
                kind = command["type"]
                if kind == "shutdown":
                    if manifest is not None:
                        camera_client.command("stop", session_id=manifest["session_id"],
                                              timeout=max(config["status_timeout"], 2.0))
                        drain_control_samples()
                        manifest.update(complete=False, state="incomplete", stopped_at=time.time(),
                                        fatal_error="Coordinator closed before episode finalization")
                        atomic_write_json(manifest_path, manifest)
                    running = False
                    continue
                if kind == "start":
                    request = EpisodeRequest(**command["request"])
                    episode_dir = Path(request.data_dir).expanduser() / request.session_id
                    episode_dir.mkdir(parents=True, exist_ok=False)
                    manifest_path = episode_dir / "manifest.json"
                    manifest = {
                        **asdict(request),
                        "schema_version": 1,
                        "state": "warming",
                        "complete": False,
                        "retained": False,
                        "created_at": time.time(),
                        "first_active_control_timestamp": None,
                        "sample_count": 0,
                        "metadata_segments": [],
                        "observation_alignment": ('selected_consumer_frame_ids'
                            if config.get('record_camera_references') else 'legacy_timestamp_matching'),
                        "capture_contract_version": config.get('capture_contract_version'),
                        "outputs": {
                            "rollout": {"requested": request.record_rollout, "decision": None},
                            "dagger": {"requested": request.record_dagger, "decision": None},
                        },
                    }
                    atomic_write_json(manifest_path, manifest)
                    writer = ControlWriter(episode_dir / "control.zarr",
                                              capture_limits=TimingLimits(**config.get('timing_limits', {})))
                    metadata_stream = (episode_dir / "metadata.jsonl").open(
                        "a", encoding="utf-8"
                    )
                    acknowledgements = camera_client.command(
                        "start",
                        session_id=request.session_id,
                        episode_dir=str(episode_dir),
                        timeout=config["warmup_timeout"],
                        payload_extra={"warmup_frames": config["warmup_frames"],
                            **({"source_color_order": "bgr"} if config.get("require_camera_timing") else {})},
                    )
                    if config.get("require_camera_timing"):
                        outdated = [name for name, status in acknowledgements.items()
                                    if status.get("timing_schema_version", 0) < 2]
                        if outdated:
                            raise RecorderError("Restart camera servers to enable V4L2 capture-clock recording (timing schema 2): " + ", ".join(outdated))
                        collection_stream = (episode_dir / "collection_labels.jsonl").open("w")
                        health_stream = (episode_dir / "writer_health.jsonl").open("w")
                        camera_progress.clear()
                        control_queue_high_water = 0
                        max_control_write_delay = 0.
                    manifest["camera_start_status"] = acknowledgements
                    manifest["camera_video_encoding"] = (
                        camera_video_encoding_from_statuses(acknowledgements)
                    )
                    manifest["state"] = "ready"
                    if config.get('check_alignment'):
                        manifest['initial_alignment_check'] = require_camera_timestamps(
                            read_shared_camera_timestamps(config['camera_names']),
                            limits=TimingLimits(**config['timing_limits']))
                    atomic_write_json(manifest_path, manifest)
                    event_queue.put(
                        {
                            "type": "ack",
                            "command_id": command["command_id"],
                            "state": "ready",
                            "sample_count": writer.count,
                        }
                    )
                elif kind in {"pause", "resume"}:
                    if manifest is None:
                        raise RecorderError(f"Cannot {kind}: no active raw episode")
                    if kind == "pause":
                        expected_sample_count = command.get("expected_sample_count")
                        drain_control_samples(
                            max_items=None,
                            expected_count=(
                                None
                                if expected_sample_count is None
                                else int(expected_sample_count)
                            ),
                            timeout=config["status_timeout"],
                        )
                    acknowledgements = camera_client.command(
                        kind,
                        session_id=manifest["session_id"],
                        timeout=config["status_timeout"],
                    )
                    manifest["state"] = "paused" if kind == "pause" else "ready"
                    if kind == 'resume' and config.get('check_alignment'):
                        manifest['resume_alignment_check'] = require_camera_timestamps(
                            read_shared_camera_timestamps(config['camera_names']),
                            limits=TimingLimits(**config['timing_limits']))
                    manifest[f"camera_{kind}_status"] = acknowledgements
                    atomic_write_json(manifest_path, manifest)
                    event_queue.put(
                        {
                            "type": "ack",
                            "command_id": command["command_id"],
                            "state": manifest["state"],
                            "sample_count": writer.count,
                        }
                    )
                elif kind == "metadata":
                    if manifest is None or metadata_stream is None:
                        raise RecorderError("Cannot write metadata: no active episode")
                    record = dict(command["metadata"])
                    record["metadata_id"] = len(manifest["metadata_segments"])
                    metadata_stream.write(json.dumps(_json_value(record), sort_keys=True) + "\n")
                    metadata_stream.flush()
                    manifest["metadata_segments"].append(record)
                    atomic_write_json(manifest_path, manifest)
                    event_queue.put(
                        {"type": "ack", "command_id": command["command_id"], "state": "metadata"}
                    )
                elif kind == "chunk":
                    if writer is None:
                        raise RecorderError("Cannot write inference chunk: no active episode")
                    writer.append_action_chunk(command["payload"])
                elif kind == "stop":
                    if manifest is None:
                        raise RecorderError("Cannot stop: no active episode")
                    acknowledgements = camera_client.command(
                        "stop",
                        session_id=manifest["session_id"],
                        timeout=max(config["status_timeout"], 10.0),
                    )
                    expected_sample_count = command.get("expected_sample_count")
                    drain_control_samples(
                        max_items=None,
                        expected_count=(
                            None
                            if expected_sample_count is None
                            else int(expected_sample_count)
                        ),
                        timeout=config["status_timeout"],
                    )
                    if metadata_stream is not None:
                        metadata_stream.close()
                        metadata_stream = None
                    if collection_stream is not None:
                        collection_stream.close()
                        collection_stream = None
                    if health_stream is not None:
                        health_stream.write(json.dumps(dict(monotonic=time.monotonic(),
                            sample_count=writer.count, cameras=acknowledgements, robot=control_backlog(), state="stopped")) + "\n")
                        health_stream.close()
                        health_stream = None
                    manifest["camera_stop_status"] = acknowledgements
                    manifest["sample_count"] = writer.count
                    if config.get("require_camera_timing"):
                        manifest["robot_writer_backlog"] = control_backlog()
                    manifest["complete"] = not bool(command.get("incomplete", False))
                    manifest["retained"] = bool(command.get("retained", False))
                    manifest["outputs"] = command.get("outputs", manifest["outputs"])
                    manifest["state"] = "complete" if manifest["complete"] else "incomplete"
                    manifest["stopped_at"] = time.time()
                    atomic_write_json(manifest_path, manifest)
                    # Audit every contract stream, camera-less ones too: publishing
                    # requires the report even when only robot timing is checked.
                    if config.get('require_camera_timing'):
                        from recording.recording_alignment import audit_episode_data
                        report = audit_episode_data(episode_dir, limits=TimingLimits(**config['timing_limits']))
                        atomic_write_json(episode_dir / 'alignment_report.json', report)
                        manifest['alignment_report'] = 'alignment_report.json'
                        manifest['alignable'] = report['alignable']
                        # Preserve every raw stream even when it fails validation.
                        if not report['alignable']:
                            manifest['complete'] = False
                            manifest['state'] = 'incomplete'
                        atomic_write_json(manifest_path, manifest)
                    event_queue.put(
                        {
                            "type": "ack",
                            "command_id": command["command_id"],
                            "state": manifest["state"],
                            "manifest_path": str(manifest_path),
                            "sample_count": writer.count,
                            "cameras": acknowledgements,
                            "robot_writer": control_backlog(),
                        }
                    )
                    writer = None
                    manifest = None
                    manifest_path = None
                    capture_gc.collect_idle()  # the episode is closed and acknowledged
                else:
                    raise RecorderError(f"Unknown coordinator command: {kind}")

            if writer is not None:
                drain_control_samples()
                camera_client.assert_healthy()
            now = time.monotonic()
            if now - last_health_event >= 0.25:
                statuses = dict(getattr(camera_client, "last_status", {}))
                alignment = None
                if writer is not None and config.get('check_alignment') and statuses and all(
                        status.get('state') == 'ready' for status in statuses.values()):
                    # Read a contemporaneous bundle; asynchronous heartbeats
                    # cannot be compared as if they were sampled simultaneously.
                    stamps = read_shared_camera_timestamps(config['camera_names'])
                    alignment = check_camera_timestamps(stamps, limits=TimingLimits(**config['timing_limits']))
                    manifest['latest_alignment_check'] = alignment
                if health_stream is not None:
                    health_stream.write(json.dumps(dict(monotonic=now, sample_count=writer.count,
                                                        cameras=statuses, robot=control_backlog(),
                                                        alignment=alignment)) + "\n")
                    health_stream.flush()
                    # Heartbeats alone do not prove cameras still deliver images.
                    for name, status in statuses.items():
                        accepted = status.get("accepted_frames", 0)
                        previous, changed = camera_progress.get(name, (accepted, now))
                        if status.get("state") != "ready" or accepted != previous:
                            changed = now
                        camera_progress[name] = (accepted, changed)
                        if now - changed > config["status_timeout"]:
                            raise RecorderError(f"Camera {name} stopped delivering frames")
                if alignment is not None and not alignment['alignable']:
                    raise CameraTimingError('; '.join(alignment['errors']))
                try:
                    event_queue.put_nowait(
                        {
                            "type": "health",
                            "state": "recording" if writer else "idle",
                            "cameras": statuses,
                            "robot_writer": control_backlog(),
                            **({"sample_count": writer.count} if writer is not None else {}),
                        }
                    )
                except queue.Full:
                    pass
                last_health_event = now
    except Exception as exc:
        if manifest is not None and manifest_path is not None:
            try:
                camera_client.command(
                    "stop",
                    session_id=manifest["session_id"],
                    timeout=max(float(config["status_timeout"]), 2.0),
                )
            except Exception:
                pass
            manifest["complete"] = False
            manifest["state"] = "incomplete"
            manifest["fatal_error"] = str(exc)
            manifest["traceback"] = traceback.format_exc()
            atomic_write_json(manifest_path, manifest)
        try:
            event_queue.put(
                {"type": "fatal", "error": str(exc), "traceback": traceback.format_exc()}
            )
        except Exception:
            pass
    finally:
        if metadata_stream is not None:
            metadata_stream.close()
        if collection_stream is not None:
            collection_stream.close()
        if health_stream is not None:
            health_stream.close()


class DistributedRecorder:
    """Non-blocking control-loop facade for the coordinator process."""

    def __init__(
        self,
        *,
        camera_names: Iterable[str],
        data_dir: str,
        camera_logger_endpoint_base: int = 55551,
        warmup_frames: int = 5,
        warmup_timeout: float = 10.0,
        status_timeout: float = 2.0,
        mp_context: str = "spawn",
        require_camera_timing: bool = True,
        timing_limits: TimingLimits | None = None,
    ):
        camera_names = list(camera_names)
        context = mp.get_context(mp_context)
        self.control_queue = context.Queue(maxsize=CONTROL_QUEUE_CAPACITY)
        self.command_queue = context.Queue(maxsize=64)
        self.event_queue = context.Queue(maxsize=256)
        self.config = {
            "camera_names": list(camera_names),
            "data_dir": str(Path(data_dir).expanduser()),
            "camera_logger_endpoint_base": int(camera_logger_endpoint_base),
            "warmup_timeout": float(warmup_timeout),
            "warmup_frames": int(warmup_frames),
            "status_timeout": float(status_timeout),
            "require_camera_timing": bool(require_camera_timing),
            "check_alignment": bool(require_camera_timing and camera_names),
            "record_camera_references": bool(require_camera_timing),
            "capture_contract_version": CAPTURE_CONTRACT_VERSION if require_camera_timing else None,
            "timing_limits": asdict(timing_limits or TimingLimits.from_env()),
        }
        self.process = context.Process(
            target=_coordinator_main,
            args=(self.config, self.control_queue, self.command_queue, self.event_queue),
            name="inference-recorder-coordinator",
            daemon=False,
        )
        self.process.start()
        self.fatal_error: str | None = None
        self.events: dict[str, dict[str, Any]] = {}
        self.active_session_id: str | None = None
        self.continuity_id = 0
        self.segment_id = 0
        self.metadata_id = 0
        self.manifests: list[str] = []
        self.enqueued_sample_count = 0
        self.committed_sample_count = 0
        self.camera_statuses = {}
        self.robot_writer_status = {}
        self.camera_reader_lease = None
        self.observation_camera_names = None

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "fatal":
                self.fatal_error = event.get("error", "unknown recorder error")
            if "sample_count" in event:
                self.committed_sample_count = int(event["sample_count"])
            if "cameras" in event:
                self.camera_statuses = event["cameras"]
            if "robot_writer" in event:
                self.robot_writer_status = event["robot_writer"]
            command_id = event.get("command_id")
            if command_id:
                self.events[command_id] = event
            if event.get("manifest_path"):
                self.manifests.append(event["manifest_path"])

    def check_health(self) -> None:
        self._poll_events()
        if self.fatal_error:
            raise RecorderError(self.fatal_error)
        if not self.process.is_alive():
            raise RecorderError(f"Recorder coordinator exited with code {self.process.exitcode}")

    def _send_and_wait(
        self,
        command: dict[str, Any],
        timeout: float,
        *,
        allow_known_fatal: bool = False,
    ) -> dict[str, Any]:
        command_id = command.setdefault("command_id", uuid.uuid4().hex)
        self.command_queue.put(command, timeout=min(float(timeout), 0.5))
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if allow_known_fatal:
                self._poll_events()
                if not self.process.is_alive():
                    raise RecorderError(
                        f"Recorder coordinator exited with code {self.process.exitcode}"
                    )
            else:
                self.check_health()
            if command_id in self.events:
                return self.events.pop(command_id)
            time.sleep(0.005)
        raise RecorderError(f"Coordinator command timed out: {command['type']}")

    def start_episode(self, request: EpisodeRequest, timeout: float) -> None:
        from camera_utils.camera_settings_contract import recording_camera_metadata
        request = replace(request, metadata=recording_camera_metadata(request.metadata, request.camera_names))
        require_absolute_actions(request.metadata)
        self.enqueued_sample_count = 0
        self.committed_sample_count = 0
        if self.config['require_camera_timing'] and self.config['camera_names'] and self.camera_reader_lease is None:
            from camera_utils.capture import acquire_camera_readers
            self.camera_reader_lease = acquire_camera_readers(self.config['camera_names'], timeout=timeout)
        self.observation_camera_names = list(request.camera_names)
        self._send_and_wait({"type": "start", "request": asdict(request)}, timeout)
        self.active_session_id = request.session_id
        self.continuity_id = 0
        self.segment_id = -1
        self.metadata_id = -1

    def append_sample(self, sample: dict[str, Any]) -> dict | None:
        self.check_health()
        require_absolute_actions(sample)
        payload = dict(sample)
        if getattr(self, 'camera_reader_lease', None) is not None:
            self.camera_reader_lease.check_health()
        if (getattr(self, 'config', {}).get('require_camera_timing')
                or getattr(self, 'camera_reader_lease', None) is not None):
            validate_capture_payload(payload, self.observation_camera_names,
                                     limits=TimingLimits(**self.config['timing_limits']))
        payload.setdefault("continuity_id", self.continuity_id)
        payload.setdefault("segment_id", self.segment_id)
        payload.setdefault("metadata_id", self.metadata_id)
        try:
            self.control_queue.put_nowait(payload)
        except queue.Full as exc:
            self.fatal_error = "Control recording queue overflow (capacity=512)"
            raise RecorderError(self.fatal_error) from exc
        self.enqueued_sample_count += 1
        return payload.get('camera_observation')

    def append_action_chunk(self, **payload: Any) -> None:
        try:
            self.command_queue.put_nowait({"type": "chunk", "payload": payload})
        except queue.Full as exc:
            self.fatal_error = "Recorder command queue overflow"
            raise RecorderError(self.fatal_error) from exc

    def set_metadata(
        self,
        metadata: dict[str, Any],
        *,
        timeout: float = 2.0,
    ) -> None:
        self.segment_id += 1
        self.metadata_id += 1
        self._send_and_wait(
            {"type": "metadata", "metadata": metadata},
            timeout,
        )

    def pause(self, timeout: float) -> None:
        self._send_and_wait(
            {"type": "pause", "expected_sample_count": self.enqueued_sample_count},
            timeout,
        )

    def resume(self, timeout: float) -> None:
        self._send_and_wait({"type": "resume"}, timeout)
        self.continuity_id += 1
        self.segment_id += 1

    def stop_episode(
        self,
        *,
        retained: bool,
        outputs: dict[str, Any],
        incomplete: bool = False,
        timeout: float = 15.0,
    ) -> str | None:
        command_id = self.begin_stop_episode(
            retained=retained,
            outputs=outputs,
            incomplete=incomplete,
            timeout=timeout,
        )
        return self.finish_stop_episode(command_id, timeout=timeout)

    def begin_stop_episode(
        self,
        *,
        retained: bool,
        outputs: dict[str, Any],
        incomplete: bool = False,
        timeout: float = 15.0,
    ) -> str:
        command_id = uuid.uuid4().hex
        self.command_queue.put(
            {
                "type": "stop",
                "command_id": command_id,
                "retained": retained,
                "outputs": outputs,
                "incomplete": incomplete,
                "expected_sample_count": self.enqueued_sample_count,
            },
            timeout=min(float(timeout), 0.5),
        )
        return command_id

    def finish_stop_episode(self, command_id: str, *, timeout: float = 15.0) -> str | None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self._poll_events()
            if command_id in self.events:
                event = self.events.pop(command_id)
                self.active_session_id = None
                capture_gc.collect_idle()  # between episodes, never during capture
                return event.get("manifest_path")
            if not self.process.is_alive():
                raise RecorderError(
                    f"Recorder coordinator exited with code {self.process.exitcode}"
                )
            time.sleep(0.005)
        raise RecorderError("Coordinator command timed out: stop")

    def close(self) -> None:
        if getattr(self, 'camera_reader_lease', None) is not None:
            self.camera_reader_lease.close()
            self.camera_reader_lease = None
        if self.process.is_alive():
            try:
                self.command_queue.put({"type": "shutdown"}, timeout=0.2)
            except queue.Full:
                pass
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)


def _write_inference_chunks(root: Any, control: Any) -> None:
    if "inference_chunks" not in control:
        return
    raw_group = control["inference_chunks"]
    count = int(raw_group.attrs.get("count", 0))
    if count <= 0:
        return
    records = [raw_group[f"{index:06d}"] for index in range(count)]
    actions = [np.asarray(record["actions"], dtype=np.float32) for record in records]
    if len({action.shape for action in actions}) != 1:
        raise RecorderError("Inference action chunks have inconsistent shapes")
    inference = root.require_group("inference")
    chunks = inference.create_group("chunks")
    chunks.create_dataset("actions", data=np.stack(actions))
    attr_mapping = {
        "query_index": "query_indices",
        "merge_index": "merge_indices",
        "start_index": "start_indices",
        "query_id": "query_ids",
        "submit_chunk_step": "submit_chunk_steps",
        "measured_start_index": "measured_start_indices",
    }
    for attr_name, dataset_name in attr_mapping.items():
        chunks.create_dataset(
            dataset_name,
            data=np.asarray(
                [record.attrs.get(attr_name, -1) for record in records],
                dtype=np.int64,
            ),
        )
    query_qpos_values = [record.attrs.get("query_qpos") for record in records]
    if all(value is not None for value in query_qpos_values):
        chunks.create_dataset(
            "query_qpos",
            data=np.asarray(query_qpos_values, dtype=np.float32),
        )


def publish_recording(manifest_path: str | Path) -> list[str]:
    """Publish requested rollout and DAgger views over the original streams."""
    from data_processing.episode_store import publish_episode, next_episode_path, open_store
    from data_processing.dagger_review import retained_indices
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('complete') or not manifest.get('retained'):
        return []
    publication = manifest.setdefault('publication', {'outputs': {}})
    if publication.get('complete'):
        return []
    directory = manifest_path.parent
    control = zarr.open_group(str(directory / 'control.zarr'), mode='r')
    if not int(manifest.get('sample_count', 0)):
        publication.update(complete=True, paths=[], reason='no_control_samples')
        atomic_write_json(manifest_path, manifest)
        return []
    require_absolute_actions(manifest, control.attrs)
    stamps = np.asarray(control['timestamps'])
    continuity = np.asarray(control['continuity_id'])
    qpos = np.asarray(control['qpos'])
    review_path = directory / 'review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    if review.get('discarded'):
        return []
    kept = retained_indices(control, review) if review else np.arange(len(stamps))
    edits_path = directory / 'edits.json'
    if edits_path.exists():
        kept = np.intersect1d(kept, json.loads(edits_path.read_text())['retained_robot_indices'])
    transitions = kept[:-1][(np.diff(kept) == 1) & (continuity[kept[:-1]] == continuity[kept[1:]])]
    outputs = []
    for kind in ('rollout', 'dagger'):
        request = manifest.get('outputs', {}).get(kind, {})
        decision = request.get('decision')
        if not request.get('requested') or decision in (None, 'pending', 'discard'):
            continue
        existing = publication['outputs'].get(kind)
        if existing and Path(existing).is_dir():
            outputs.append(existing)
            continue
        indices = transitions if kind == 'dagger' else kept
        if not len(indices):
            continue
        target = Path(manifest[f'{kind}_dir']).expanduser()
        task = manifest.get(f'{kind}_task_name')
        if task:
            target /= task
        if kind == 'rollout' and manifest.get('eval_mode'):
            target /= str(decision)
        metadata = dict(manifest.get('metadata') or {}, sim=False,
            rollout=kind == 'rollout', dagger=kind == 'dagger', recording_decision=decision,
            failure=decision == 'failure', robot_type='arx', robot_model='L5',
            single_arm=not manifest['is_double_arm'], double_arm=manifest['is_double_arm'])
        arrays = {}
        if kind == 'dagger':
            segments = np.asarray(control['segment_id'])
            active = np.asarray(control['intervention_active'], dtype=np.uint8)
            starts, ends = dagger_intervention_ranges(active[indices])
            arrays.update({
                'action': qpos[indices + 1], 'action_timestamps': stamps[indices + 1],
                'policy_target_action': np.asarray(control['policy_action'])[indices],
                'observations/capture_segment_index': segments[indices],
                'observations/capture_continuity_index': continuity[indices],
                'observations/intervention_active': active[indices],
                'action_raw': qpos, 'action_raw_timestamps': stamps,
                'action_raw_capture_segment_index': segments,
                'action_raw_capture_continuity_index': continuity,
                'intervention_active_raw': active,
                'dagger_interventions/start_index': starts,
                'dagger_interventions/end_index_exclusive': ends,
            })
            metadata.update(has_raw_actions=True, intervention_count=len(starts),
                            action_label_source='next_measured_qpos_within_continuity')
        def annotate(root):
            _write_inference_chunks(root, control)
            mask = np.asarray(control['inference_query'])[indices]
            queries = np.flatnonzero(mask)
            root.create_dataset('inference_query_mask', data=mask)
            if len(queries):
                group = root.require_group('inference')
                group.create_dataset('query_indices', data=queries)
                group.create_dataset('query_timestamps', data=np.asarray(control['inference_query_timestamp'])[indices[queries]])
                group.create_dataset('query_overhead_frame_numbers', data=np.asarray(control['inference_query_overhead_frame_number'])[indices[queries]])
        output = publish_episode(next_episode_path(target), [(directory, indices)],
                                 metadata=metadata, arrays=arrays, annotate=annotate)
        publication['outputs'][kind] = output
        outputs.append(output)
        atomic_write_json(manifest_path, manifest)
    # Pending decisions may be accepted later; do not seal them prematurely.
    pending = any(request.get('requested') and request.get('decision') in (None, 'pending')
                  for request in manifest.get('outputs', {}).values() if isinstance(request, dict))
    publication.update(complete=not pending, paths=outputs, published_at=time.time())
    atomic_write_json(manifest_path, manifest)
    return outputs


def publish_pending_recordings(data_dir: str | Path) -> list[str]:
    outputs = []
    for manifest in sorted(Path(data_dir).expanduser().glob('*/manifest.json')):
        outputs.extend(publish_recording(manifest))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description='Publish completed Zarr/video inference episodes')
    parser.add_argument('--data-dir', required=True, help='recording manifest root')
    args = parser.parse_args()
    for output in publish_pending_recordings(args.data_dir):
        print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
