from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest
import yaml

from config.task_config import load_task_config
from contracts.robot_contract import require_robot_match
from test_robot_contract import contract
from test_shared_hardware import FakeArm


def arx_task(tmp_path, role='right'):
    path = tmp_path / 'task.yaml'
    path.write_text(yaml.safe_dump(dict(schema_version=1, task={'name': 'arx_test'},
        control_freq=50, data={'directory': str(tmp_path/'data')},
        robot={'type': 'arx', 'arms': {role: {'model': 'Trossen_L5', 'can_interface': 'can1'}}},
        cameras={'names': []}, collection={'max_duration_seconds': .04},
        training={'observations': ['qpos'], 'outputs': ['action'], 'state_dim': 7, 'action_dim': 7})))
    return load_task_config(path)


def test_arx_task_carries_actual_single_arm_role_and_requires_checkpoint_identity(tmp_path):
    task = arx_task(tmp_path)
    assert task.robot_contract == contract(role='right')
    assert task.legacy()['single_arm_side'] == 'right'
    assert task.legacy()['hardware_config']['type'] == 'arx'
    metadata = {'control_freq': 50, 'prediction_horizon': 50}
    task.require_checkpoint(metadata)  # Untagged ARX checkpoints keep their legacy path.
    with pytest.raises(ValueError, match='mismatch'):
        task.require_checkpoint({**metadata, 'robot_contract': contract()})
    task.require_checkpoint({**metadata, 'robot_contract': contract(role='right')})


def test_tagged_normalization_stats_cannot_mix_models_or_roles(tmp_path):
    from test_action_contract import write_episode
    from data_processing import episode_store
    from policy.training_utils import get_norm_stats
    paths = [tmp_path/'a.zarr', tmp_path/'b.zarr']
    for path in paths:
        write_episode(path)
        with episode_store.Store(path, 'a') as root:
            root.attrs['robot_contract'] = contract(role='right')
    stats, _ = get_norm_stats([str(path) for path in paths])
    assert stats['robot_contract'] == contract(role='right')
    with episode_store.Store(paths[1], 'a') as root:
        root.attrs['robot_contract'] = contract('L5', 'right')
    with pytest.raises(ValueError, match='Mixed'):
        get_norm_stats([str(path) for path in paths])


def test_shared_recorder_starts_before_capture_and_keeps_exact_snapshot(tmp_path, monkeypatch):
    from recording import record_hardware, record_episode_single
    task = arx_task(tmp_path)
    arm = FakeArm('right')
    events = []
    original = arm._read_state
    def read():
        events.append('capture')
        return original()
    arm._read_state = read
    monkeypatch.setattr(record_hardware, 'build_arms', lambda _: {'right': arm})
    monkeypatch.setattr(record_episode_single, 'wait_for_cameras', lambda *a, **k: None)
    provider = Mock(return_value=np.arange(7))
    provider.robot_contract = contract(role='right')
    class Recorder:
        active_session_id = None
        def start_episode(self, request, timeout):
            events.append('start')
            self.active_session_id = request.session_id
            require_robot_match(contract(role='right'), request.metadata)
        def append_sample(self, sample):
            events.append('append')
            assert sample['robot_read_started_at'] <= sample['timestamp']
            np.testing.assert_array_equal(sample['qvel'], sample['qpos'] * 2)
        def stop_episode(self, **options):
            events.append('stop'); self.active_session_id = None
        def close(self): events.append('close')
    def cameras(stop, **kwargs): stop.wait()
    record_hardware.record_targets(task, provider, recorder_factory=lambda **kw: Recorder(), camera_target=cameras)
    assert events[0] == 'start'
    assert events.count('capture') == events.count('append') == 2
    assert events[-2:] == ['stop', 'close']
    assert arm.closed_count == 1
    provider.close.assert_called_once()


def test_recorder_fault_closes_robot_and_marks_episode_incomplete(tmp_path, monkeypatch):
    from recording import record_hardware, record_episode_single
    task = arx_task(tmp_path)
    arm = FakeArm('right')
    monkeypatch.setattr(record_hardware, 'build_arms', lambda _: {'right': arm})
    monkeypatch.setattr(record_episode_single, 'wait_for_cameras', lambda *a, **k: None)
    provider = Mock(side_effect=RuntimeError('input disconnected'))
    provider.robot_contract = contract(role='right')
    recorder = Mock(active_session_id='test')
    def cameras(stop, **kwargs): stop.wait()
    with pytest.raises(RuntimeError, match='input disconnected'):
        record_hardware.record_targets(task, provider, recorder_factory=lambda **kw: recorder, camera_target=cameras)
    assert recorder.stop_episode.call_args.kwargs['incomplete'] is True
    assert arm.closed_count == 1
    recorder.close.assert_called_once()


class HomingArm(FakeArm):
    """Simulated arm: a native home move, and a log of submissions in order."""
    def __init__(self, role='right', **options):
        super().__init__(role, **options)
        self.events = []
    def _submit_target(self, target, *, interval_s):
        super()._submit_target(target, interval_s=interval_s)
        self.events.append(('target', np.asarray(target, dtype=float).copy()))
    def _close(self):
        super()._close()
        self.events.append(('close', None))
    def _home(self):
        self.events.append(('home', None))


START = np.array([0.03, 1.02, 2.04, 2.97, 4.01, 5.0, 6.05])
OPEN_GRIPPER = 5.9


def shared_setup(tmp_path, monkeypatch, keys, *, start=True, num_rollouts=1, tick=None):
    """Run the shared rollout loop on simulated hardware; ``keys`` yields (wait_event, key)."""
    from inference import shared_hardware, inference_remote, policy_transport
    from recording import record_episode_single
    from camera_utils import capture
    task = arx_task(tmp_path)
    arm = HomingArm('right')
    monkeypatch.setattr(shared_hardware, 'build_arms', lambda _: {'right': arm})
    monkeypatch.setattr(record_episode_single, 'wait_for_cameras', lambda *a, **k: None)
    def cameras(stop, **kwargs): stop.wait()
    monkeypatch.setattr(capture, 'capture_image_from_shared_memory', cameras)
    client = Mock(server_metadata={'robot_contract': contract(role='right'), 'control_freq': 50, 'action_dim': 7})
    monkeypatch.setattr(policy_transport, 'WebsocketOpenPIClient', lambda *a: client)
    warmups = []
    def make_runtime(args, supplied_client):
        args._yaml_task = task; args.fps = 50
        def query(*a, **k):
            warmups.append(len(arm.sent))
            return np.zeros((1, 50, 7)), {'query_total': 0.01}, (50, 7)
        return SimpleNamespace(task=task.legacy(), cameras=[], action_dim=7, query=query,
                               rtc_metadata=lambda prefix=None: {})
    session = Mock()
    session.tick.side_effect = tick or (lambda observe, prompt: np.arange(7)[None] + 0.001)
    monkeypatch.setattr(inference_remote, 'PolicyRuntime', make_runtime)
    monkeypatch.setattr(inference_remote, 'PolicySession', lambda _: session)
    path = tmp_path / 'start.npz'
    np.savez(path, right_qpos=START)
    script = iter(keys)
    def reader():
        wait, key = next(script, (None, 'q'))
        if wait is not None:
            assert wait(), 'simulated operator waited too long'
        __import__('time').sleep(.02)
        return key
    args = SimpleNamespace(start_position_path=str(path) if start else None, record_dagger_corrections=False,
        dagger_footpedal_device=None, save_episode=False, remote_host='localhost', remote_port=0,
        api_key=None, record_rollout=False, eval=False, num_rollouts=num_rollouts, max_timesteps=1000,
        prompt='', async_inference=True, async_prefetch_steps=30, live_startup_warmup_steps=2)
    return arm, session, warmups, lambda: shared_hardware.run_shared_rollouts(args, key_reader=reader)


def reached(arm, target):
    return [i for i, (kind, value) in enumerate(arm.events)
            if kind == 'target' and np.allclose(value, target)]


def test_shared_holds_then_ramps_to_saved_start_warms_up_and_homes_on_quit(tmp_path, monkeypatch):
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [(None, 'q')])
    run()
    targets = [value for kind, value in arm.events if kind == 'target']
    # Connecting holds the measured pose before any other command.
    np.testing.assert_array_equal(targets[0], np.arange(7))
    # Then a bounded ramp (direct inference's move_to_start_action) reaches the saved pose.
    steps = np.abs(np.diff(np.array(targets[:reached(arm, START)[0] + 1]), axis=0))
    assert steps.max() <= 0.01 + 1e-9
    # Warm-up queries run only after the start pose is held.
    assert len(warmups) == 2 and min(warmups) > reached(arm, START)[0]
    # 'q' homes natively before the single close.
    assert arm.events[-1] == ('close', None)
    assert arm.events[-2][0] == 'home'
    assert arm.closed_count == 1 and not arm.faulted
    session.tick.assert_not_called()  # it never left the paused start pose


def test_shared_episode_end_holds_and_next_episode_starts_from_saved_pose(tmp_path, monkeypatch):
    executed = threading.Event()
    def tick(observe, prompt):
        return np.arange(7)[None] + 0.001
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [
        (None, '1'),                       # left pedal: run
        (lambda: executed.wait(2), 's'),   # finish episode 1 as a success
        (None, 'h'),                       # right pedal: return to the start pose (not quit)
        (None, 'q')], num_rollouts=2, tick=tick)
    session.committed.side_effect = executed.set
    run()
    assert session.committed.call_count >= 1
    # The episode end did not disable the arm: one close, after the final home move.
    assert arm.closed_count == 1 and arm.events[-1] == ('close', None)
    # The saved start pose was held at startup and again for episode 2.
    starts = reached(arm, START)
    policy = [i for i, (kind, value) in enumerate(arm.events)
              if kind == 'target' and np.allclose(value, np.arange(7) + 0.001)]
    assert policy and starts[0] < policy[0] < starts[-1]
    assert arm.events[-2][0] == 'home'


def reset_targets(arm):
    """Every target, the startup arrival at START, and where the reset ramp begins.

    The simulated arm always measures np.arange(7), which 'r' anchors to before moving.
    """
    targets = [value for kind, value in arm.events if kind == 'target']
    arrival = reached(arm, START)[0]
    anchor = next(i for i, t in enumerate(targets) if i > arrival and np.allclose(t, np.arange(7)))
    return targets, arrival, anchor


def test_shared_reset_opens_gripper_on_the_way_back_then_sets_saved_gripper(tmp_path, monkeypatch):
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [(None, 'r'), (None, 'q')])
    arm.open_gripper_target = lambda: OPEN_GRIPPER
    run()
    targets, arrival, anchor = reset_targets(arm)
    # The startup ramp is unchanged: straight to the saved pose, never opening.
    assert all(t[6] > OPEN_GRIPPER + 1e-6 for t in targets[:arrival + 1])
    open_at_start = np.r_[START[:6], OPEN_GRIPPER]
    arrived = [i for i, t in enumerate(targets) if np.allclose(t, open_at_start)]
    assert arrived and arrived[0] > anchor
    # On the way back the joints and gripper move together, bounded per tick.
    way_back = np.array(targets[anchor:arrived[0] + 1])
    assert np.abs(np.diff(way_back, axis=0)).max() <= 0.01 + 1e-6
    assert np.all(np.diff(way_back[:, 6]) <= 0) and way_back[-2, 6] > way_back[-1, 6]
    assert np.all(np.abs(way_back[1:-1, :6] - START[:6]).max(axis=1) > 1e-6)
    # The gripper only moves once the arm has arrived, and the joints stay put meanwhile.
    final = next(i for i, t in enumerate(targets) if i > arrived[-1] and np.allclose(t, START))
    closing = np.array(targets[arrived[-1]:final + 1])
    np.testing.assert_allclose(closing[:, :6], np.tile(START[:6], (len(closing), 1)), atol=1e-6)
    assert np.all(np.diff(closing[:, 6]) > 0)
    assert len(closing) - 1 == 50  # 1 s at the simulated 50 Hz, whatever the distance
    # 'q' homing is unchanged: the native home move still runs.
    assert arm.events[-2][0] == 'home'


def test_shared_reset_keeps_saved_gripper_when_open_position_is_unknown(tmp_path, monkeypatch):
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [(None, 'r'), (None, 'q')])
    run()  # the simulated arm has no calibrated gripper
    targets, arrival, anchor = reset_targets(arm)
    grippers = np.array([t[6] for t in targets[anchor:reached(arm, START)[-1] + 1]])
    assert np.all(np.diff(grippers) >= -1e-9)  # straight from measured to saved, never opening
    np.testing.assert_allclose(grippers[-1], START[6])


def test_open_gripper_targets_come_from_each_backends_calibration():
    from hardware.backends.arx import ARXAdapter
    from inference import inference_direct as direct
    arx = ARXAdapter.__new__(ARXAdapter)
    arx._robot_config = None
    assert arx.open_gripper_target() is None
    arx._robot_config = Mock(gripper_width=0.088)
    assert arx.open_gripper_target() == 0.088
    # Legacy ARX controllers read the same value from their own robot config.
    controller = Mock()
    controller.get_robot_config.return_value = Mock(gripper_width=0.088)
    assert direct.open_gripper_positions({'left': controller, 'right': controller}, True) == [0.088, 0.088]


def test_shared_refuses_to_run_without_a_start_pose(tmp_path, monkeypatch):
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [], start=False)
    with pytest.raises(ValueError, match='start pose'):
        run()
    assert arm.sent == [] and not warmups  # rejected before connecting or querying


def test_shared_keeps_targets_fresh_while_a_synchronous_query_blocks(tmp_path, monkeypatch):
    # A 0.1 s command watchdog: a blocking query without the background hold would expire it.
    executed = threading.Event()
    def slow_tick(observe, prompt):
        __import__('time').sleep(0.3)
        return np.arange(7)[None] + 0.001
    arm, session, warmups, run = shared_setup(tmp_path, monkeypatch, [
        (None, 'c'), (lambda: executed.wait(3), 'q')], tick=slow_tick)
    arm._timeout = 0.1
    session.worker = None  # synchronous inference
    session.committed.side_effect = executed.set
    run()
    assert not arm.faulted and arm.closed_count == 1
    assert arm.events[-2][0] == 'home'
