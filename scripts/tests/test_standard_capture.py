"""Hardware-free tests of the mandatory collection/inference boundary."""
import ast
import copy
from pathlib import Path
import queue
import sys
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from contracts import standard_capture as capture
from camera_utils.camera_timing import CameraTimingError, TimingLimits
from recording.distributed_recording import DistributedRecorder, ControlWriter


def observation(*, include_images=False, stamp=100., frame=7):
    return capture.capture_observation(lambda: (np.arange(7), -np.arange(7)), ['video_left'],
        clock=lambda: stamp, include_images=include_images,
        snapshot_provider=lambda names, copy_images: (
            {'video_left': np.full((2, 3, 3), frame, np.uint8)} if copy_images else {},
            dict(snapshot_timestamp=stamp, cameras={'video_left': dict(frame_number=frame,
                capture_timestamp=stamp-.01, consumer_received_timestamp=stamp-.005)})))


def test_state_is_read_once_before_packet_selection_and_owned():
    order = []
    state = np.arange(7.)
    def read():
        order.append('state')
        return state, state + 1
    def snapshot(names, *, copy_images):
        order.append('snapshot')
        state[:] = -99
        return {}, observation().camera_observation
    result = capture.capture_observation(read, ['video_left'], clock=lambda: 100., snapshot_provider=snapshot)
    assert order == ['state', 'snapshot']
    np.testing.assert_array_equal(result.qpos, np.arange(7.))
    np.testing.assert_array_equal(result.qvel, np.arange(7.) + 1)


def test_collection_and_inference_choose_identical_references():
    collected, inferred = observation(), observation(include_images=True)
    assert collected.images == {}
    assert collected.camera_observation == inferred.camera_observation
    assert np.all(inferred.images['video_left'] == 7)
    assert 'images' not in inferred.recording_fields()


def test_concurrent_new_frame_does_not_replace_selected_pixels(monkeypatch):
    import camera_utils.capture as cameras
    old = np.full((2, 3, 3), 7, np.uint8)
    monkeypatch.setattr(cameras, 'captured_camera_packets', {'video_left': (old, 7, 99.99, 99.995)})
    monkeypatch.setattr(cameras.time_module, 'monotonic', lambda: 100.)
    result = capture.capture_observation(lambda: (np.zeros(7), np.zeros(7)), ['video_left'], include_images=True)
    cameras.captured_camera_packets['video_left'] = (np.zeros_like(old), 8, 100., 100.)
    assert result.camera_observation['cameras']['video_left']['frame_number'] == 7
    assert np.all(result.images['video_left'] == 7)


def test_missing_publication_timestamp_is_not_fabricated(monkeypatch):
    import camera_utils.capture as cameras
    monkeypatch.setattr(cameras, 'captured_camera_packets', {'video_left': (np.zeros((2, 3, 3)), 7, 99.99)})
    with pytest.raises(RuntimeError, match='publication timestamps'):
        cameras.get_camera_observation(['video_left'])


@pytest.mark.parametrize('change,match', [
    (lambda p: p.pop('capture_contract_version'), 'standard_capture'),
    (lambda p: p.update(timestamp=100.001), 'ordering|different proprio'),
    (lambda p: p['camera_observation'].update(snapshot_timestamp=100.1), 'delay'),
    (lambda p: p['camera_observation']['cameras']['video_left'].update(capture_timestamp=99.), 'age'),
    (lambda p: p['camera_observation']['cameras']['video_left'].update(capture_timestamp=101.), 'future'),
    (lambda p: p['camera_observation']['cameras']['video_left'].update(consumer_received_timestamp=100.01), 'published'),
    (lambda p: p['camera_observation']['cameras']['video_left'].update(frame_number=-1), 'published'),
    (lambda p: p.update(qvel=[float('nan')]*7), 'qvel'),
    (lambda p: p.update(qvel=[0.]), 'dimensions'),
])
def test_invalid_pairings_fail_closed(change, match):
    payload = observation().recording_fields()
    change(payload)
    with pytest.raises(CameraTimingError, match=match):
        capture.validate_capture_payload(payload, ['video_left'])


def test_cross_camera_skew_is_checked():
    payload = observation().recording_fields()
    payload['camera_observation']['cameras']['video_right'] = dict(
        frame_number=9, capture_timestamp=99.955, consumer_received_timestamp=99.96)
    with pytest.raises(CameraTimingError, match='spread'):
        capture.validate_capture_payload(payload, ['video_left', 'video_right'],
            limits=TimingLimits(max_skew_seconds=.02))


def test_writer_validates_at_capture_time_not_after_policy_compute(tmp_path, monkeypatch):
    result = observation()
    recorder = DistributedRecorder.__new__(DistributedRecorder)
    recorder.check_health = lambda: None
    recorder.camera_reader_lease = mock.Mock()
    recorder.config = dict(timing_limits={})
    recorder.observation_camera_names = ['video_left']
    recorder.continuity_id = recorder.segment_id = recorder.metadata_id = recorder.enqueued_sample_count = 0
    recorder.control_queue = queue.Queue()
    monkeypatch.setattr(capture.time, 'monotonic', lambda: 110.)
    recorder.append_sample(dict(result.recording_fields(), commanded_action=[0.]*7))
    writer = ControlWriter(tmp_path / 'control.zarr')
    writer.append(recorder.control_queue.get())
    assert writer.group.attrs['capture_contract_version'] == 1
    assert writer.group['camera_observation/video_left/frame_number'][0] == 7
    assert writer.group['timestamps'][0] == 100.
    with pytest.raises(Exception, match='contract version cannot change'):
        writer.append(dict(timestamp=100.01, qpos=[0.]*7, qvel=[0.]*7, commanded_action=[0.]*7))
    with pytest.raises(CameraTimingError, match='standard_capture'):
        recorder.append_sample(dict(timestamp=110., qpos=[0.], qvel=[0.], commanded_action=[0.]))


def test_image_preprocessing_does_not_resnapshot(monkeypatch):
    result = observation(include_images=True)
    monkeypatch.setattr(capture, '_snapshot_cameras', lambda *a, **k: pytest.fail('reread'))
    assert capture.resized_images(result, dict(width=6, height=4))['video_left'].shape == (4, 6, 3)
    np.testing.assert_array_equal(capture.policy_state(result, {'observation_name': ['qpos', 'qvel']}),
                                  np.r_[np.arange(7), -np.arange(7)])


@pytest.mark.parametrize('path,required', [
    ('src/recording/record_episode_single.py', 'standard_capture.capture_observation'),
    ('src/inference/inference_remote.py', 'standard_capture.capture_observation'),
    ('src/inference/inference_direct.py', 'standard_capture.capture_observation'),
])
def test_live_entrypoints_reuse_standard_capture(path, required):
    tree = ast.parse((ROOT / path).read_text())
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert required in calls, f'{path} must reuse {required}; see AGENTS.md'




def test_clock_contract_rejects_legacy_and_replaced_segments(tmp_path):
    from types import SimpleNamespace
    from camera_utils.camera_timing import publish_shared_clock_contract, validate_shared_clock_contract
    segment = tmp_path / 'fake_segment'
    segment.write_bytes(b'0' * 32)
    with segment.open('rb') as file:
        shm = SimpleNamespace(name=segment.name, _fd=file.fileno())
        with pytest.raises(CameraTimingError, match='restart camera_server'):
            validate_shared_clock_contract(shm, directory=tmp_path)
        publish_shared_clock_contract(shm, directory=tmp_path)
        validate_shared_clock_contract(shm, directory=tmp_path)
        segment.unlink()
        segment.write_bytes(b'0' * 32)
        with segment.open('rb') as replacement:
            newer = SimpleNamespace(name=segment.name, _fd=replacement.fileno())
            with pytest.raises(CameraTimingError, match='stale'):
                validate_shared_clock_contract(newer, directory=tmp_path)
            publish_shared_clock_contract(newer, directory=tmp_path)
            validate_shared_clock_contract(newer, directory=tmp_path)


@pytest.mark.parametrize('module_name,sdk_method', [
    ('inference/inference_direct', 'get_joint_state'),
])
def test_live_inference_adapters_read_position_velocity_from_one_sdk_result(module_name, sdk_method, monkeypatch):
    from types import SimpleNamespace
    # Execute the actual pure adapter against synthetic SDK objects without
    tree = ast.parse((ROOT / f'src/{module_name}.py').read_text())
    adapter = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_observations')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), module_name, 'exec'), namespace)
    if sdk_method == 'get_joint_state':
        state = SimpleNamespace(pos=lambda: np.arange(6.), vel=lambda: np.arange(6.)+10,
                                gripper_pos=6., gripper_vel=16.)
    else:
        state = [SimpleNamespace(position=float(i), velocity=float(i+10)) for i in range(7)]
    controllers = {name: mock.Mock(**{sdk_method+'.return_value': state}) for name in ['left', 'right']}
    monkeypatch.setattr(capture, '_snapshot_cameras', lambda *a, **k: ({}, observation().camera_observation))
    monkeypatch.setattr(capture.time, 'monotonic', lambda: 100.)
    result = capture.capture_observation(lambda: namespace['get_observations'](controllers), ['video_left'])
    np.testing.assert_array_equal(result.qvel, result.qpos+10)
    for controller in controllers.values():
        getattr(controller, sdk_method).assert_called_once_with()


def test_capture_disables_automatic_garbage_collection():
    import gc
    from contracts import capture_gc
    capture_gc._reset_for_tests()
    assert gc.isenabled()
    observation()
    assert not gc.isenabled() and capture_gc.is_paused()
    gc.enable()  # a library re-enabling collection is overridden at the next capture
    observation()
    assert not gc.isenabled()


def test_idle_collection_reclaims_cycles_and_stays_disabled():
    import gc
    import weakref
    from contracts import capture_gc
    observation()
    class Node:
        pass
    node = Node()
    node.self = node
    alive = weakref.ref(node)
    del node
    assert alive() is not None  # no automatic collection while streaming
    assert capture_gc.collect_idle() > 0
    assert alive() is None and not gc.isenabled()


@pytest.mark.parametrize('path,required', [
    ('src/recording/distributed_recording.py', ['capture_gc.pause_for_stream', 'capture_gc.collect_idle']),
    ('src/camera_utils/camera_server.py', ['capture_gc.pause_for_stream', 'capture_gc.collect_idle']),
    ('src/camera_utils/robotmq_camera_cache.py', ['capture_gc.pause_for_stream']),
    ('src/inference/inference_direct.py', ['capture_gc.collect_idle']),
    ('src/inference/inference_remote.py', ['capture_gc.collect_idle']),
])
def test_streaming_processes_pause_garbage_collection(path, required):
    tree = ast.parse((ROOT / path).read_text())
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    missing = [name for name in required if name not in calls]
    assert not missing, f'{path} must call {missing}; see AGENTS.md'
    assert 'gc.disable' not in calls and 'gc.enable' not in calls, f'{path} must use contracts.capture_gc'
