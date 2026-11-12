"""Parallel consumers and inference-faithful frame references, without hardware."""
import json
from pathlib import Path
import queue
import sys
import threading
import time
from unittest import mock

import numpy as np
import pytest
import zarr

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src'),
                str(Path(__file__).resolve().parents[2] / 'scripts')]
import camera_utils.capture as cameras
from camera_utils.camera_timing import CameraTimingError, selected_frame_indices
from recording.distributed_recording import DistributedRecorder, ControlWriter, RecorderError
from recording.recording_alignment import audit_episode_data


def test_metadata_snapshot_never_copies_pixels_and_matches_inference(monkeypatch):
    now = time.monotonic()
    class Image:
        copies = 0
        def copy(self):
            self.copies += 1
            return np.zeros((2, 2, 3), np.uint8)
    frame = Image()
    monkeypatch.setattr(cameras, 'captured_camera_packets', {
        'video_left': (frame, 11, now - .01, now - .005),
        'video_right': (frame, 22, now - .012, now - .004)})
    images, recording = cameras.get_camera_observation(['video_left', 'video_right'])
    assert not images and frame.copies == 0
    images, inference = cameras.get_camera_observation(['video_left', 'video_right'], copy_images=True)
    assert frame.copies == 2
    assert recording['cameras'] == inference['cameras']
    assert cameras.get_last_camera_observation() is inference


def test_snapshot_keeps_ids_and_pixels_paired_during_new_publication(monkeypatch):
    now = time.monotonic()
    original = np.zeros((2, 2, 3), np.uint8)
    class RacingImage:
        def copy(self):
            with cameras._camera_packet_lock:
                cameras.captured_camera_packets['video_left'] = (np.ones_like(original), 2, now, now)
            return original.copy()
    monkeypatch.setattr(cameras, 'captured_camera_packets', {'video_left': (RacingImage(), 1, now-.01, now-.005)})
    images, observation = cameras.get_camera_observation(['video_left'], copy_images=True)
    assert not images['video_left'].any()
    assert observation['cameras']['video_left']['frame_number'] == 1
    assert cameras.captured_camera_packets['video_left'][1] == 2


def test_camera_readers_run_in_parallel_and_leases_share_workers(monkeypatch):
    monkeypatch.setenv('FLASH_ALOHA_CAMERA_CACHE', 'python')
    slow_entered, release_slow, fast_read = threading.Event(), threading.Event(), threading.Event()
    created, closed = [], []
    class Reader:
        def __init__(self, name):
            self.name = name
            created.append(name)
        def read_frame(self):
            if self.name == 'left':
                slow_entered.set()
                assert release_slow.wait(2.)
            else:
                fast_read.set()
            return None, -1, 0.
        def close(self): closed.append(self.name)
    monkeypatch.setattr(cameras, 'SharedFrameReader', Reader)
    first = cameras.CameraReaderLease(['video_left', 'video_right'])
    second = None
    try:
        assert slow_entered.wait(1.)
        assert fast_read.wait(1.), 'A blocked left camera must not delay the right reader'
        second = cameras.CameraReaderLease(['video_right'])
        assert created.count('right') == 1
        release_slow.set()
        first.close()
        assert 'left' in closed and 'right' not in closed
    finally:
        release_slow.set()
        first.close()
        if second: second.close()
    assert sorted(closed) == ['left', 'right']
    assert not cameras._camera_readers


def referenced_episode(path):
    from test_camera_capture_clock import episode_data
    episode_data(path)
    control = zarr.open_group(str(path / 'control.zarr'), mode='a')
    # Deliberately choose older frames than capture-time matching would choose.
    indices = np.arange(2, 6)
    control.create_dataset('camera_observation/snapshot_timestamp', data=control['timestamps'][:]+.001)
    for name in ('video_left', 'video_right'):
        control.create_dataset(f'camera_observation/{name}/frame_number', data=indices)
        control.create_dataset(f'camera_observation/{name}/capture_timestamp', data=100.+indices*.01)
        control.create_dataset(f'camera_observation/{name}/consumer_received_timestamp', data=100.+indices*.01+.003)
    return control


def test_alignment_resolves_selected_ids_not_newest_capture(tmp_path):
    control = referenced_episode(tmp_path)
    np.testing.assert_array_equal(selected_frame_indices(tmp_path, 'video_left', control), [2, 3, 4, 5])
    report = audit_episode_data(tmp_path)
    assert report['alignable'], report
    assert report['alignment_rule'] == 'selected_consumer_frame_ids'
    assert report['cameras']['video_left']['consumer']['age']['max_ms'] == pytest.approx(21.)


def test_audit_rejects_dropped_selected_frame_and_late_consumer_publication(tmp_path):
    control = referenced_episode(tmp_path)
    control['camera_observation/video_left/consumer_received_timestamp'][0] = 101.
    assert not audit_episode_data(tmp_path)['alignable']
    control['camera_observation/video_left/frame_number'][0] = 9999
    with pytest.raises(CameraTimingError, match='not recorded'):
        selected_frame_indices(tmp_path, 'video_left', control)
    assert not audit_episode_data(tmp_path)['alignable']


def test_capture_after_robot_read_is_valid_if_visible_at_snapshot(tmp_path):
    control = referenced_episode(tmp_path)
    control['camera_observation/snapshot_timestamp'][:] = control['timestamps'][:]+.015
    for name in ('video_left', 'video_right'):
        indices = np.arange(5, 9)
        control[f'camera_observation/{name}/frame_number'][:] = indices
        control[f'camera_observation/{name}/capture_timestamp'][:] = 100.+indices*.01
        control[f'camera_observation/{name}/consumer_received_timestamp'][:] = 100.+indices*.01+.003
    report = audit_episode_data(tmp_path)
    assert report['alignable'], report


def test_recorder_preserves_query_snapshot_instead_of_resampling(monkeypatch):
    now = time.monotonic()
    observation = dict(snapshot_timestamp=now, cameras={'video_left': dict(
        frame_number=7, capture_timestamp=now-.01, consumer_received_timestamp=now-.005)})
    recorder = DistributedRecorder.__new__(DistributedRecorder)
    recorder.check_health = lambda: None
    recorder.camera_reader_lease = mock.Mock()
    recorder.config = dict(timing_limits={})
    recorder.observation_camera_names = ['video_left']
    recorder.continuity_id = recorder.segment_id = recorder.metadata_id = recorder.enqueued_sample_count = 0
    recorder.control_queue = queue.Queue()
    monkeypatch.setattr(cameras, 'get_camera_observation', lambda *a, **kw: pytest.fail('resampled query'))
    observation.update(capture_contract_version=1, robot_timestamp=now, robot_read_started_at=now)
    recorder.append_sample(dict(timestamp=now, robot_read_started_at=now, capture_contract_version=1,
                               qpos=[0.], qvel=[0.], commanded_action=[0.], camera_observation=observation))
    assert recorder.control_queue.get()['camera_observation'] == observation


def test_control_writer_refuses_partially_referenced_stream(tmp_path):
    writer = ControlWriter(tmp_path / 'control.zarr')
    sample = dict(timestamp=100., qpos=[0.], qvel=[0.], commanded_action=[0.])
    sample['camera_observation'] = dict(snapshot_timestamp=100., cameras={'video_left': dict(
        frame_number=1, capture_timestamp=99.99, consumer_received_timestamp=99.995)})
    writer.append(sample)
    sample.pop('camera_observation')
    with pytest.raises(RecorderError, match='every control sample'):
        writer.append(sample)


def test_publication_uses_saved_reference_even_when_newer_capture_exists(tmp_path):
    from test_raw_training import make_raw
    from data_processing.episode_store import publish_episode
    from data_processing.episode_data import Episode
    source = make_raw(tmp_path/'source')
    path = publish_episode(tmp_path/'episode.zarr', [(source, [0, 1, 2])])
    with Episode(source) as original, Episode(path) as published:
        for name in original.images:
            np.testing.assert_array_equal(published.images[name][2], original.images[name][1])
            np.testing.assert_array_equal(published[f'camera_observation/{name}/frame_number'], [1000, 1001, 1001])


