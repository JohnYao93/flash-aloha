"""Driver clock, shared health policy and raw audit; all hardware is simulated."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src'),
                str(Path(__file__).resolve().parents[2] / 'scripts')]
from camera_utils.camera_timing import CameraTimingError, TimingLimits, check_camera_timestamps, driver_timestamp, load_camera_timestamps, require_camera_timestamps
from recording.recording_alignment import audit_episode_data
from recording.distributed_recording import ControlWriter
from recording.record_episode_single import ARXBackend, EpisodeCapture, EpisodeSample


def test_same_delay_on_all_cameras_is_detected_even_with_zero_skew():
    report = check_camera_timestamps({'left': 98.88, 'right': 98.88}, now=100.)
    assert report['skew_seconds'] == 0
    assert not report['alignable']
    assert len(report['errors']) == 2
    assert check_camera_timestamps({'left': 99.99, 'right': 99.98}, now=100.)['alignable']


def test_skew_future_unknown_clock_and_missing_data_are_rejected():
    report = check_camera_timestamps({'left': 99.99, 'right': 99.90}, now=100.,
                                     limits=TimingLimits(max_age_seconds=1.))
    assert not report['alignable'] and 'spread' in report['errors'][0]
    for stamps in ({'left': 100.02}, {'left': None}, {'left': float('nan')}, {}):
        with pytest.raises(CameraTimingError):
            require_camera_timestamps(stamps, now=100.)
    assert driver_timestamp(100., 0x2000) == 100.
    assert driver_timestamp(100., 0x12000) == 100.  # start-of-exposure + monotonic
    for stamp, flags in ((0., 0x2000), (100., 0), (100., 0x4000)):
        with pytest.raises(CameraTimingError):
            driver_timestamp(stamp, flags)


def test_position_and_velocity_use_one_snapshot_per_arm():
    class Controller:
        calls = 0
        def get_joint_state(self):
            self.calls += 1
            n = self.calls
            class State:
                gripper_pos, gripper_vel = n, -n
                def pos(self): return np.full(6, n)
                def vel(self): return np.full(6, -n)
            return State()
    backend = ARXBackend.__new__(ARXBackend)
    backend.controllers = [Controller(), Controller()]
    backend._start_hold_qpos = None
    # A split SDK read would make velocities come from a different generation.
    backend.read_qpos = backend.read_qvel = lambda: pytest.fail('split read')
    sample = EpisodeCapture(backend, [], fps=100).sample(force=True)
    np.testing.assert_array_equal(sample.qpos, -sample.qvel)
    assert [c.calls for c in backend.controllers] == [1, 1]
    assert sample.robot_read_started_at <= sample.robot_timestamp


def episode_data(path, *, right_delay=.002, continuity=False):
    names = ['video_left', 'video_right']
    (path / 'manifest.json').write_text(json.dumps(dict(camera_names=names)))
    times = np.array([100.04, 100.05, 100.06, 100.07])
    if continuity:
        times[2:] += 30.
    writer = ControlWriter(path / 'control.zarr')
    for i, stamp in enumerate(times):
        writer.append(dict(timestamp=stamp, qpos=np.zeros(7), qvel=np.zeros(7), commanded_action=np.zeros(7),
                           continuity_id=int(continuity and i >= 2)))
    for name in names:
        capture = 100. + np.arange(10) * .01
        ids = np.zeros(10, dtype=int)
        if continuity:
            capture[6:] += 30.
            ids[6:] = 1
        rows = [dict(video_frame=i, frame_number=i, capture_timestamp=float(stamp), timestamp=float(stamp+1.12),
                     frame_ready_timestamp=float(stamp + (right_delay if name == 'video_right' else .002)),
                     continuity_id=int(ids[i]), buffer_flags=0x2000, driver_sequence=i)
                for i, stamp in enumerate(capture)]
        (path / f'{name}.timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        group = zarr.open_group(str(path / f'{name}.zarr'), mode='w')
        group.create_dataset('main_timestamps', data=capture + 1.12)
    return path


def test_audit_and_converter_use_capture_clock_despite_synthetic_offset(tmp_path):
    episode_data(tmp_path)
    report = audit_episode_data(tmp_path)
    assert report['alignable'], report
    assert report['cross_camera']['capture']['spread']['max_ms'] == 0
    group = zarr.open_group(str(tmp_path / 'video_left.zarr'), mode='r')
    np.testing.assert_allclose(load_camera_timestamps(tmp_path, 'video_left', group), 100.+np.arange(10)*.01)


def test_audit_finds_transport_delay_even_when_physical_matching_works(tmp_path):
    episode_data(tmp_path, right_delay=.12)
    report = audit_episode_data(tmp_path)
    assert not report['alignable']
    assert report['cameras']['video_right']['capture']['bad_samples'] == 0
    assert report['cameras']['video_right']['available']['bad_samples'] == 4


def test_audit_never_bridges_rollback_continuity(tmp_path):
    episode_data(tmp_path, continuity=True)
    # First resumed row has a physical image but it has not arrived yet. Keep
    # the next row, for which a new-continuity frame has actually arrived.
    (tmp_path / 'edits.json').write_text(json.dumps(dict(retained_robot_indices=[0, 1, 3])))
    assert audit_episode_data(tmp_path)['alignable']
    path = tmp_path / 'video_right.timing.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row['continuity_id'] = 0
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = audit_episode_data(tmp_path)
    assert not report['alignable']
    assert report['cameras']['video_right']['capture']['bad_samples'] == 1




def test_shared_episode_buffer_closes_on_exception_and_keeps_raw_data(tmp_path, monkeypatch):
    from recording import episode_buffer as module
    from fake_streams import FakeStreams
    monkeypatch.setattr(module, 'DistributedRecorder', FakeStreams)
    buffers = []
    @module.recording_scope
    def capture():
        buffer = module.EpisodeBuffer(camera_names=['video_overhead'], directory=tmp_path)
        buffers.append(buffer)
        buffer.append(EpisodeSample({}, np.zeros(7), np.zeros(7), 0., robot_timestamp=100.))
        assert module.recording_images(['video_overhead'], lambda _: pytest.fail('camera read on robot loop')) == {}
        raise RuntimeError('operator abort')
    with pytest.raises(RuntimeError, match='operator abort'):
        capture()
    assert buffers[0].finished
    manifest = json.loads((buffers[0].data_dir / 'manifest.json').read_text())
    assert not manifest['complete'] and (buffers[0].data_dir / 'control.zarr').exists()




def test_inference_bundle_pairs_pixels_with_the_checked_capture_time(monkeypatch):
    import camera_utils.capture as camera_capture
    import camera_utils.capture as camera_capture
    now = 100.
    monkeypatch.setattr('camera_utils.camera_timing.time.monotonic', lambda: now)
    old = np.zeros((2, 2, 3), np.uint8)
    new = np.ones((2, 2, 3), np.uint8)
    monkeypatch.setattr(camera_capture, 'captured_camera_packets', {'video_left': (old, 1, 99.99, 99.995)})
    images, stamps = camera_capture.get_camera_bundle(['video_left'])
    camera_capture.captured_camera_packets['video_left'] = (new, 2, 100., 100.)
    assert stamps['video_left'] == 99.99
    assert not images['video_left'].any()
    camera_capture.captured_camera_packets['video_left'] = (old, 1, 98.88, 98.89)
    with pytest.raises(CameraTimingError, match='age'):
        camera_capture.get_camera_bundle(['video_left'])


def test_no_future_frame_is_substituted_for_missing_camera_coverage(tmp_path):
    from test_camera_observation import referenced_episode
    from camera_utils.camera_timing import selected_frame_indices
    control = referenced_episode(tmp_path)
    control['camera_observation/video_left/frame_number'][0] = 9999
    with pytest.raises(CameraTimingError, match='not recorded'):
        selected_frame_indices(tmp_path, 'video_left', control)
