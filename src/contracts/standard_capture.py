"""Mandatory observation boundary for recording and live policy inference.

Task code supplies a single state-reader callback, not its own image selection.
This module owns read ordering, timing, packet selection, and provenance.
It never commands hardware, encodes images, or writes files.
"""
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from camera_utils.camera_timing import CameraTimingError, TimingLimits, require_camera_timestamps
from contracts import capture_gc

CAPTURE_CONTRACT_VERSION = 1


def _snapshot_cameras(names, *, copy_images):
    # Lazy import keeps storage/auditing and simulated tests independent of SDKs.
    from camera_utils.capture import get_camera_observation
    return get_camera_observation(names, copy_images=copy_images)


@dataclass(frozen=True)
class CapturedObservation:
    state: dict
    images: dict
    robot_read_started_at: float
    robot_timestamp: float
    camera_observation: dict

    @property
    def qpos(self):
        return self.state['qpos']

    @property
    def qvel(self):
        return self.state['qvel']

    def recording_fields(self):
        """Small numeric payload only; image pixels never enter the control queue."""
        return dict(capture_contract_version=CAPTURE_CONTRACT_VERSION,
                    timestamp=self.robot_timestamp, robot_read_started_at=self.robot_read_started_at,
                    qpos=self.qpos, qvel=self.qvel, camera_observation=self.camera_observation)


def capture_observation(read_state: Callable, camera_names, *, include_images=False,
                        clock=None, snapshot_provider=None):
    """Read one state snapshot, then select one immutable camera packet bundle.

    ``read_state`` returns (qpos, qvel), or a dict containing both plus optional
    auxiliary state. Physical adapters must obtain position/velocity from the
    same SDK result per arm. ``snapshot_provider``/``clock`` are test injection
    points, not customization hooks for live capture.
    """
    # Before the read: automatic garbage collection never runs in a capturing process.
    capture_gc.pause_for_stream()
    clock = clock or time.monotonic
    names = list(camera_names)
    started = clock()
    result = read_state()
    finished = clock()
    # Freeze state before another SDK poll can mutate a reusable result buffer.
    state = dict(result) if isinstance(result, dict) else dict(zip(('qpos', 'qvel'), result))
    for key in ('qpos', 'qvel'):
        state[key] = np.asarray(state[key]).copy()
    if names:
        images, metadata = (snapshot_provider or _snapshot_cameras)(names, copy_images=include_images)
    else:
        images, metadata = {}, dict(snapshot_timestamp=clock(), cameras={})
    metadata = dict(metadata, capture_contract_version=CAPTURE_CONTRACT_VERSION,
                    robot_timestamp=finished, robot_read_started_at=started)
    observation = CapturedObservation(state, images, started, finished, metadata)
    validate_capture_payload(observation.recording_fields(), names)
    return observation


def validate_capture_payload(sample, camera_names, *, limits=None):
    """Reject independent/resampled image metadata at the recording boundary."""
    limits = limits or TimingLimits.from_env()
    observation = sample.get('camera_observation')
    if (sample.get('capture_contract_version') != CAPTURE_CONTRACT_VERSION
            or not observation
            or observation.get('capture_contract_version') != CAPTURE_CONTRACT_VERSION):
        raise CameraTimingError('Use standard_capture.capture_observation; an unbound image/proprio sample is not recordable')
    started = float(sample['robot_read_started_at'])
    finished = float(sample['timestamp'])
    snapshot = float(observation['snapshot_timestamp'])
    if (not np.isfinite([started, finished, snapshot]).all() or started > finished
            or finished - started > limits.max_age_seconds
            or finished > snapshot or snapshot - finished > limits.max_age_seconds):
        raise CameraTimingError('Invalid capture ordering or excessive proprio-to-camera snapshot delay')
    if (observation.get('robot_timestamp') != finished
            or observation.get('robot_read_started_at') != started):
        raise CameraTimingError('Camera references belong to a different proprio observation')
    names = list(camera_names)
    if set(observation['cameras']) != set(names):
        raise CameraTimingError('Capture camera set differs from the recording camera set')
    for key in ('qpos', 'qvel'):
        value = np.asarray(sample[key])
        if value.ndim != 1 or not np.isfinite(value).all():
            raise CameraTimingError(f'Invalid {key} in captured observation')
    if np.shape(sample['qpos']) != np.shape(sample['qvel']):
        raise CameraTimingError('Position and velocity must describe the same state dimensions')
    if names:
        require_camera_timestamps({name: item['capture_timestamp']
            for name, item in observation['cameras'].items()}, now=snapshot, limits=limits)
        for name, item in observation['cameras'].items():
            number = item['frame_number']
            received = float(item['consumer_received_timestamp'])
            if (not isinstance(number, (int, np.integer)) or number < 0
                    or not np.isfinite(received) or received > snapshot
                    or received < item['capture_timestamp'] - limits.max_future_seconds):
                raise CameraTimingError(f'{name}: invalid published frame reference')


def empty_robot_state():
    """Human/camera-only collection still uses the same observation boundary."""
    return np.empty(0, np.float32), np.empty(0, np.float32)


def policy_state(observation, task_config, state_dim=None):
    """Build model features after capture without changing recorded proprio."""
    names = task_config.get('observation_name', ['qpos'])
    result = np.concatenate([np.asarray(observation.state[name]).reshape(-1) for name in names])
    return result if state_dim is None else result[:state_dim]


def resized_images(observation, config):
    """Pure post-capture preprocessing; never fetches a newer camera frame."""
    from contracts.standard_image_processing import policy_image_spec, resize_image
    spec = policy_image_spec(config)
    return {name: resize_image(frame, spec.width, spec.height)
            for name, frame in observation.images.items()}
