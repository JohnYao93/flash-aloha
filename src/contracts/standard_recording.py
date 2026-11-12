"""Stream recording shared by rollout and DAgger recorders.

Start before capturing the first observation. Only numeric data and exact frame
references are appended in the control loop; saving never decodes or re-encodes pixels.
"""
from collections.abc import Sequence
from threading import Lock
from types import SimpleNamespace

import numpy as np

from recording.episode_buffer import EpisodeBuffer, recording_scope


def append_observation(buffer, observation, action, *, task_label=None):
    fields = observation.recording_fields()
    buffer.append(SimpleNamespace(
        images={}, qpos=observation.qpos, qvel=observation.qvel,
        timestamp=observation.robot_timestamp, robot_timestamp=observation.robot_timestamp,
        robot_read_started_at=observation.robot_read_started_at, capture_interval=None,
        camera_observation=fields['camera_observation'],
        capture_contract_version=fields['capture_contract_version'],
        action=np.asarray(action).reshape(-1).copy(), task_label=task_label or {}))


class RecordedImages(Sequence):
    """Lazy, serialized decoder access, including threaded legacy exporters."""
    def __init__(self, buffer, camera):
        self.buffer, self.camera = buffer, camera
        self.lock = Lock()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.lock:
            return self.buffer[index].images[self.camera]


def ensure_rollout_started(lists, directory, names, fps, is_double_arm):
    buffer = lists.get('_episode_buffer')
    if buffer is None or buffer.finished:
        buffer = EpisodeBuffer(camera_names=names, directory=directory, fps=fps,
                                  is_double_arm=is_double_arm)
        lists['_episode_buffer'] = buffer
        lists['images'] = {name: RecordedImages(buffer, name) for name in names}
    return buffer


def append_rollout(lists, observation, action):
    append_observation(lists['_episode_buffer'], observation, action)
    lists['position'].append(observation.qpos.copy())
    lists['velocity'].append(observation.qvel.copy())
    lists['commanded_action'].append(np.asarray(action).reshape(-1).copy())
    lists['action_timestamps'].append(observation.robot_timestamp)
    for name, reference in observation.camera_observation['cameras'].items():
        lists['image_timestamps'][name].append(reference['capture_timestamp'])


def finish_rollout(lists):
    buffer = lists.pop('_episode_buffer', None)
    if buffer is not None:
        buffer.finish()
    return buffer
