"""Shared recorder contract tests; all cameras, clocks, and arms are synthetic."""
from pathlib import Path
import sys
import threading
from unittest import mock

import cv2
from data_processing import episode_store
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from recording import record_episode_single as single


class Clock:
    def __init__(self):
        self.now = 10.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def __init__(self):
        self.pose = np.arange(14, dtype=np.float32)
        self.velocity = np.zeros(14, dtype=np.float32)
        self.held = False
        self.steps = 0

    def passive_step(self):
        self.steps += 1

    def read_qpos(self):
        return self.pose

    def read_qvel(self):
        return self.velocity

    def hold_after_recording(self):
        self.held = True


def samples():
    rng = np.random.default_rng(123)
    return [single.EpisodeSample({'video_overhead': rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)},
                                 np.full(14, i), np.full(14, i / 10), i / 100) for i in range(4)]


def test_nonblocking_capture_owns_arrays_and_excludes_paused_time():
    clock, backend = Clock(), Backend()
    frame = np.full((24, 32, 3), 47, np.uint8)
    def snapshot(names, *, copy_images):
        return {'video_overhead': frame.copy()}, dict(snapshot_timestamp=clock.now, cameras={
            'video_overhead': dict(frame_number=1, capture_timestamp=clock.now-.001,
                                   consumer_received_timestamp=clock.now)})
    capture = single.EpisodeCapture(backend, ['video_overhead'], fps=100,
                                    snapshot_provider=snapshot, include_images=True, clock=clock.monotonic)
    first = capture.sample()
    assert capture.sample() is None and backend.steps == 1
    frame[:] = 99
    backend.pose[:] = -1
    assert np.all(first.images['video_overhead'] == 47)
    np.testing.assert_array_equal(first.qpos, np.arange(14))
    clock.sleep(.01)
    second = capture.sample()
    assert second.timestamp == pytest.approx(.01)
    capture.pause()
    clock.sleep(300)
    assert capture.sample(force=True) is None
    capture.resume(second.timestamp)
    third = capture.sample()
    assert third.timestamp == pytest.approx(.02)
    assert capture.intervals == [] and capture.timing.latest is None
    clock.sleep(.01)
    fourth = capture.sample()
    assert fourth.timestamp == pytest.approx(.03)
    assert capture.intervals == pytest.approx([.01])


def test_capture_limit_and_nonfinite_state_do_not_emit_samples():
    clock, backend = Clock(), Backend()
    capture = single.EpisodeCapture(backend, [], fps=100,
                                    clock=clock.monotonic, max_seconds=1)
    backend.pose[0] = np.nan
    with pytest.raises(RuntimeError, match='Invalid qpos'):
        capture.sample()
    assert capture.previous_capture is None
    clock.sleep(1)
    with pytest.raises(OSError, match='time limit'):
        capture.sample()


def test_standalone_uses_shared_capture_and_writer_and_preserves_events(tmp_path, monkeypatch):
    clock, backend = Clock(), Backend()
    monkeypatch.setattr(single, 'time', clock)
    monkeypatch.setattr(single, 'wait_for_cameras', lambda *a, **kw: None)
    monkeypatch.setattr(single, 'prompt_for_episode_start', lambda *a: True)
    monkeypatch.setattr(single, 'prompt_accept_episode', lambda **kwargs: True)
    monkeypatch.setattr(single, 'get_image_dict', lambda names: {n: np.zeros((24, 32, 3), np.uint8) for n in names})
    keys = iter([None, '2', 'f', '1'])
    monkeypatch.setattr(single, 'read_key_nonblocking', lambda: next(keys))
    ctx = single.RunContext('test', str(tmp_path), 10, ['video_overhead'], 'arx', False, True)
    args = single.build_arg_parser().parse_args(['--task_name', 'test'])
    original = single.write_episode
    def save(*args, **kwargs):
        assert backend.held
        return original(*args, **kwargs)
    with mock.patch.object(single.EpisodeCapture, 'sample', autospec=True,
                           side_effect=single.EpisodeCapture.sample) as capture_call, \
         mock.patch.object(single, 'write_episode', side_effect=save) as write_call:
        assert single.capture_one_episode(ctx, backend, 'episode_00000', False, args)
    assert capture_call.call_count == 4 and write_call.call_count == 1
    with episode_store.Store(tmp_path / 'episode_00000.zarr') as root:
        assert root['observations/qpos'].shape == (4, 14)
        np.testing.assert_allclose(root['recording_events/subgoal_switch_event_timestamps'], [.01])
        np.testing.assert_allclose(root['recording_events/failure_event_timestamps'], [.02])
        assert root['recording_events/recording_stop_event_timestamp'][()] == pytest.approx(.03)
        assert 'action' in root and 'timestamps' in root






def test_image_free_recordings_still_supported(tmp_path):
    from recording.episode_buffer import EpisodeBuffer
    buffer = EpisodeBuffer(camera_names=[], directory=tmp_path, fps=100)
    capture = single.EpisodeCapture(Backend(), [], fps=100, clock=lambda: 100.)
    buffer.append(capture.sample())
    single.write_episode(tmp_path / 'episode.zarr', buffer, [], 14, 14)
    with episode_store.Store(tmp_path / 'episode.zarr') as root:
        assert root['observations/qpos'].shape == (1, 14)
        assert 'observations/images' not in root
