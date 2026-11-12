"""Common independent-stream recording for task-specific robot collectors.

The list interface preserves task labels/review workflows. Camera pixels are
owned by camera_server; list items refer to video and decode only for inspection.
Saving an episode publishes these references without duplicating images.
"""
from collections.abc import MutableMapping
from contextvars import ContextVar
import copy
from functools import wraps
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import zarr

from camera_utils.camera_timing import TimingLimits, load_camera_timestamps, require_camera_timestamps, selected_frame_indices
from recording.distributed_recording import DistributedRecorder, EpisodeRequest
from recording.recording_utils import atomic_write_json

_buffers = ContextVar('episode_recording_buffers', default=None)


def recording_scope(function):
    """Close/drain all episode streams on return, cancellation and exceptions."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        buffers = []
        token = _buffers.set(buffers)
        failed = True
        try:
            result = function(*args, **kwargs)
            failed = False
            return result
        finally:
            try:
                for buffer in buffers:
                    buffer.finish(incomplete=failed)
            finally:
                _buffers.reset(token)
    return wrapped


def recording_images(camera_names, fallback):
    """Only sampling calls use this; previews continue reading live images."""
    buffers = _buffers.get()
    if buffers and any(not buffer.finished and buffer.names == list(camera_names) for buffer in buffers):
        return {}
    return fallback(camera_names)


class _RecordedImages(MutableMapping):
    def __init__(self, buffer, timestamp, overrides, raw_index):
        self.buffer, self.timestamp = buffer, timestamp
        self.raw_index = raw_index
        self.overrides = {name: np.array(value, copy=True) for name, value in overrides.items()
                          if name not in buffer.names}

    def __getitem__(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return self.buffer.image(name, self.timestamp, self.raw_index)

    def __setitem__(self, name, value):
        self.overrides[name] = value

    def __delitem__(self, name):
        raise TypeError('Raw cameras cannot be removed from a recorded sample')

    def __iter__(self):
        return iter(dict.fromkeys([*self.buffer.names, *self.overrides]))

    def __len__(self):
        return len(set(self.buffer.names) | set(self.overrides))


class EpisodeBuffer(list):
    def __init__(self, *, camera_names, directory, fps=100, jpeg_quality=80,
                 stream_factory=None, is_double_arm=True, **compatibility_options):
        super().__init__()
        if directory is None:
            raise ValueError('Raw recording requires a persistent dataset directory')
        self.names = list(camera_names)
        self.jpeg_quality = jpeg_quality
        self.session_id = 'capture_' + uuid.uuid4().hex
        parent = Path(directory).expanduser().resolve() / 'data'
        self.data_dir = parent / self.session_id
        self.finished = False
        self.alignment_ok = False
        self.sample_count = 0
        self.readers = {}
        from camera_utils.camera_settings_contract import snapshot_camera_settings
        self.camera_settings = snapshot_camera_settings(self.names)
        self.stream = (stream_factory or DistributedRecorder)(camera_names=self.names, data_dir=str(parent), require_camera_timing=True)
        try:
            self.stream.start_episode(EpisodeRequest(self.session_id, str(parent), self.names,
                False, False, '', '', '', '', [], is_double_arm, fps,
                metadata=dict(collection='shared_raw_recording', timestamp_source='v4l2_monotonic',
                              camera_settings=self.camera_settings)),
                timeout=15.)
        except BaseException:
            self.stream.close()
            raise
        if _buffers.get() is not None:
            _buffers.get().append(self)

    def append(self, sample):
        if self.finished:
            raise RuntimeError('Cannot append after raw recording has stopped')
        self.stream.check_health()
        stamp = sample.robot_timestamp
        if stamp is None:
            raise ValueError('Raw robot samples require an absolute monotonic read timestamp')
        copied = copy.copy(sample)
        copied.qpos, copied.qvel = sample.qpos.copy(), sample.qvel.copy()
        copied.robot_index = self.sample_count
        copied.images = _RecordedImages(self, stamp, sample.images, self.sample_count)
        self.stream.append_sample(dict(timestamp=stamp, qpos=copied.qpos, qvel=copied.qvel,
            camera_observation=sample.camera_observation, capture_contract_version=sample.capture_contract_version,
            commanded_action=getattr(sample, 'action', copied.qpos),
            robot_read_started_at=sample.robot_read_started_at if sample.robot_read_started_at is not None else stamp,
            elapsed_seconds=sample.timestamp, sampled_at=time.time(),
            capture_interval=sample.capture_interval if sample.capture_interval is not None else np.nan,
            collection_label=dict(getattr(sample, 'task_label', {}), **{key: getattr(sample, key) for key in
                              ('target_timestamp', 'effort') if hasattr(sample, key)})))
        self.sample_count += 1
        super().append(copied)

    def finish(self, *, incomplete=False):
        if self.finished:
            return
        try:
            atomic_write_json(self.data_dir / 'edits.json', dict(schema_version=1,
                retained_robot_indices=[sample.robot_index for sample in self],
                camera_rule='selected consumer frame IDs; legacy fallback by capture timestamp'))
            self.stream.stop_episode(retained=False, outputs={}, incomplete=incomplete, timeout=30.)
            manifest = json.loads((self.data_dir / 'manifest.json').read_text())
            self.alignment_ok = manifest.get('alignable') is not False
        finally:
            self.finished = True
            self.stream.close()

    def drain(self):
        self.finish()

    def image(self, name, timestamp, raw_index=None):
        self.finish()
        if not self.alignment_ok:
            raise RuntimeError(f'Raw stream timing failed validation: {self.data_dir / "alignment_report.json"}')
        if name not in self.names:
            raise KeyError(name)
        if name not in self.readers:
            group = zarr.open_group(str(self.data_dir / f'{name}.zarr'), mode='r')
            stamps = load_camera_timestamps(self.data_dir, name, group)
            decoder = cv2.VideoCapture(str(self.data_dir / f'{name}.zarr/main.mp4'))
            if not decoder.isOpened():
                raise RuntimeError(f'{name}: raw video cannot be opened')
            control = zarr.open_group(str(self.data_dir / 'control.zarr'), mode='r')
            retained = [sample.robot_index for sample in self]
            selected = selected_frame_indices(self.data_dir, name, control, sample_indices=retained)
            references = None if selected is None else dict(zip(retained, selected))
            snapshots = (np.asarray(control['camera_observation/snapshot_timestamp'])
                         if references is not None else None)
            self.readers[name] = [stamps, decoder, -1, None, references, snapshots]
        reader = self.readers[name]
        index = (reader[4][raw_index] if reader[4] is not None
                 else int(np.searchsorted(reader[0], timestamp, side='right')) - 1)
        if index < 0:
            raise RuntimeError(f'{name}: no frame at robot sample time')
        require_camera_timestamps({name: float(reader[0][index])},
                                  now=reader[5][raw_index] if reader[5] is not None else timestamp)
        if index < reader[2]:
            reader[1].release()
            reader[1] = cv2.VideoCapture(str(self.data_dir / f'{name}.zarr/main.mp4'))
            reader[2] = -1
        while reader[2] < index:
            ok, frame = reader[1].read()
            if not ok:
                raise RuntimeError(f'{name}: truncated raw video at frame {reader[2]+1}')
            reader[2] += 1
            reader[3] = frame
        return reader[3].copy()

    def __del__(self):
        # Lifecycle belongs to recording_scope; this only releases offline decoders.
        for reader in getattr(self, 'readers', {}).values():
            reader[1].release()


class ProprioLogger:
    """Legacy RoboLogger CLI adapter to the same coordinator/storage contract."""
    def __init__(self, *, camera_names, fps, attr, endpoint_base=55551):
        self.names, self.fps, self.attr = list(camera_names), fps, attr
        self._explicit_camera_settings = 'camera_settings' in attr
        self.endpoint_base = endpoint_base
        self.zarr_group = self.stream = None
        self.stopped = False

    def start_recording(self, episode_dir):
        self.directory = Path(episode_dir)
        self.data_dir = self.directory / 'streams/episode'
        from camera_utils.camera_settings_contract import recording_camera_metadata
        if not self._explicit_camera_settings:
            self.attr = {key: value for key, value in self.attr.items() if key != 'camera_settings'}
        self.attr = recording_camera_metadata(self.attr, self.names)
        self.stream = DistributedRecorder(camera_names=self.names, data_dir=str(self.data_dir.parent),
                                           camera_logger_endpoint_base=self.endpoint_base, require_camera_timing=True)
        try:
            self.stream.start_episode(EpisodeRequest('episode', str(self.data_dir.parent), self.names,
                False, False, '', '', '', '', [], bool(self.attr['double_arm']), self.fps,
                metadata=self.attr), timeout=15.)
        except BaseException:
            self.stream.close()
            raise

    def log_observation(self, observation, *, commanded_action=None):
        self.stream.append_sample(dict(observation.recording_fields(),
            commanded_action=observation.qpos if commanded_action is None else commanded_action,
            collection_label={}))

    def log_data(self, **kwargs):
        raise ValueError('Use standard_capture.capture_observation and log_observation; split state/image logging is prohibited')

    def stop_recording(self, *, incomplete=False):
        if self.stopped or self.stream is None:
            return
        try:
            self.stream.stop_episode(retained=True, outputs={}, incomplete=incomplete, timeout=30.)
            self.zarr_group = zarr.open_group(str(self.data_dir / 'control.zarr'), mode='a')
            self.zarr_group.attrs.update(self.attr)
        finally:
            self.stopped = True
            self.stream.close()

    def close(self):
        self.stop_recording(incomplete=True)
