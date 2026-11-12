"""Pedal-started ARX episodes with a simulated SDK and synthetic camera streams only."""
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr

from contracts import standard_capture
from data_processing import episode_store
from recording import record_episode_single as recorder, episode_buffer
from fake_streams import FakeStreams
from utils.hardware_utils import ACTIVE_KD, ACTIVE_KP

TRACKING_KP, TRACKING_KD = np.array(ACTIVE_KP[:6]), np.array(ACTIVE_KD[:6])
GRIPPER_WIDTH = 0.088


class Gain:
    def __init__(self, dof=6):
        self._kp, self._kd = np.zeros(6), np.zeros(6)
        self.gripper_kp = self.gripper_kd = 0.0

    def kp(self):
        return self._kp

    def kd(self):
        return self._kd

    def copy(self):
        gain = Gain()
        gain._kp, gain._kd = self._kp.copy(), self._kd.copy()
        gain.gripper_kp, gain.gripper_kd = self.gripper_kp, self.gripper_kd
        return gain


class JointState:
    def __init__(self, dof=6):
        self._pos, self._vel, self._torque = np.zeros(dof), np.zeros(dof), np.zeros(dof)
        self.gripper_pos = self.gripper_vel = self.gripper_torque = 0.0
        self.timestamp = 0.0

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel

    def torque(self):
        return self._torque


class Controller:
    """ARX joint controller stand-in: tracks perfectly whenever position gains are on."""

    def __init__(self, robot_config, controller_config, interface):
        self.config, self.interface = controller_config, interface
        self.positions, self.gripper = np.zeros(6), 0.04
        self.gain = Gain()
        self.commands, self.events = [], []
        self.gravity = self.friction = None

    def get_gain(self):
        return self.gain.copy()

    def set_gain(self, gain):
        self.gain = gain.copy()
        self.events.append(('gain', gain.kp().copy(), gain.kd().copy(), gain.gripper_kp, gain.gripper_kd))

    def get_joint_state(self):
        state = JointState()
        state.pos()[:] = self.positions
        state.gripper_pos = self.gripper
        state.timestamp = time.monotonic()
        return state

    def get_timestamp(self):
        return time.monotonic()

    def set_joint_cmd(self, command):
        self.commands.append((command.pos().copy(), float(command.gripper_pos), self.gain.copy()))
        if np.any(self.gain.kp() > 0):
            self.positions[:] = command.pos()
        if self.gain.gripper_kp > 0:
            self.gripper = float(command.gripper_pos)

    def set_gravity_compensation(self, enabled):
        self.gravity = enabled

    def set_friction_compensation(self, enabled):
        self.friction = enabled

    def get_controller_config(self):
        return self.config

    def set_to_damping(self):
        self.gain = Gain()
        self.events.append('damping')

    def reset_to_home(self):
        self.positions[:] = 0.0
        self.gripper = GRIPPER_WIDTH
        self.events.append('home')

    def passive(self):
        return (not np.any(self.gain.kp()) and not np.any(self.gain.kd())
                and self.gain.gripper_kp == 0 and self.gain.gripper_kd == 0)

    def tracking(self):
        return (np.array_equal(self.gain.kp(), TRACKING_KP) and np.array_equal(self.gain.kd(), TRACKING_KD)
                and self.gain.gripper_kp == ACTIVE_KP[6] and self.gain.gripper_kd == ACTIVE_KD[6])


@pytest.fixture
def sdk(monkeypatch):
    """A simulated arx5_interface; ``controllers`` lists the arms in connection order."""
    robot_config = SimpleNamespace(joint_dof=6, gripper_width=GRIPPER_WIDTH)
    controller_config = SimpleNamespace(default_kp=np.ones(6), default_kd=np.ones(6),
                                        default_gripper_kp=1.0, default_gripper_kd=1.0)
    controllers = []

    def connect(robot, config, interface):
        controllers.append(Controller(robot, config, interface))
        return controllers[-1]

    module = SimpleNamespace(
        RobotConfigFactory=SimpleNamespace(get_instance=lambda: SimpleNamespace(get_config=lambda model: robot_config)),
        ControllerConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=lambda kind, robot: controller_config)),
        Arx5JointController=connect, JointState=JointState, Gain=Gain, controllers=controllers)
    monkeypatch.setitem(sys.modules, 'arx5_interface', module)
    return module


def arguments(*flags, task='arx_left_wrist_only'):
    return recorder.build_arg_parser().parse_args(['--task_name', task, *flags])


@pytest.fixture
def manual(sdk, tmp_path):
    ctx = replace(recorder.resolve_run_context(arguments('--no-start-position')),
                  dataset_dir=str(tmp_path), max_timesteps=3, start_position_move_duration=.01)
    backend = recorder.ARXBackend(ctx)
    yield SimpleNamespace(backend=backend, ctx=ctx, arm=sdk.controllers[0])
    backend.shutdown()


def start_pose_backend(tmp_path, pose, *flags):
    path = tmp_path / 'start.npz'
    np.savez(path, left_qpos=pose)
    ctx = replace(recorder.resolve_run_context(arguments('--start-position', str(path), *flags)),
                  dataset_dir=str(tmp_path / 'data'), max_timesteps=3, start_position_move_duration=.01)
    return path, ctx, recorder.ARXBackend(ctx)


def test_saved_per_task_start_pose_replaces_the_shared_default(tmp_path, monkeypatch):
    from config import task_config
    monkeypatch.setattr(task_config, 'REPO_ROOT', tmp_path)
    assert recorder.resolve_run_context(arguments()).start_position_path == 'calibration/start_position.npz'
    (tmp_path / 'calibration/start_positions').mkdir(parents=True)
    (tmp_path / 'calibration/start_positions/arx_left_wrist_only.npz').touch()
    ctx = recorder.resolve_run_context(arguments())
    assert ctx.start_position_path == 'calibration/start_positions/arx_left_wrist_only.npz'
    assert recorder.resolve_run_context(arguments('--no-start-position')).start_position_path is None
    assert task_config.default_start_position_path('arx_right', 'arx') == 'calibration/start_position.npz'


def test_task_resolves_role_camera_rate_and_start_pose():
    ctx = recorder.resolve_run_context(arguments())
    assert ctx.robot_type == 'arx'
    assert ctx.single_arm and not ctx.double_arm
    assert ctx.camera_names == ['video_left']
    assert ctx.fps == 100 and ctx.max_timesteps == 1500  # 15 s task limit
    assert ctx.start_position_path == recorder.DEFAULT_START_POSITION_PATH
    assert ctx.hardware_config['arms']['left']['can_interface'] == 'can0'
    assert ctx.robot_contract['arms']['left']['model'] == 'L5'
    assert recorder.resolve_run_context(arguments('--start-position', '/tmp/custom.npz')).start_position_path == '/tmp/custom.npz'
    with pytest.raises(SystemExit):
        arguments('--robot', 'other')


def test_connects_each_role_to_its_interface_in_native_order(sdk):
    ctx = recorder.resolve_run_context(arguments('--no-start-position', task='arx_bimanual'))
    backend = recorder.ARXBackend(ctx)
    assert [arm.interface for arm in sdk.controllers] == ['can0', 'can1']
    sdk.controllers[0].positions[:] = .1
    sdk.controllers[1].positions[:] = .2
    qpos, qvel = backend.read_state()
    np.testing.assert_allclose(qpos, np.r_[np.full(6, .1), .04, np.full(6, .2), .04])
    assert qvel.shape == (14,)
    right = recorder.ARXBackend(recorder.resolve_run_context(arguments('--no-start-position', task='arx_right')))
    assert sdk.controllers[-1].interface == 'can1' and list(right._by_role) == ['right']
    assert sdk.controllers[0].config.gravity_compensation and sdk.controllers[0].config.controller_dt == 0.002


def test_recording_mode_is_passive_with_gravity_and_friction_support(manual):
    manual.backend.setup_for_recording()
    assert manual.backend.mode == 'motion' and manual.arm.tracking()
    assert manual.arm.friction is False  # friction feedforward is for hand guiding only
    manual.backend.start_recording()
    assert manual.backend.mode == 'recording' and manual.arm.passive()
    assert manual.arm.gravity is True and manual.arm.friction is True


def test_saved_pose_including_the_gripper_is_held_until_recording_releases_it(sdk, tmp_path, capsys):
    pose = np.array([.4, 1.2, 1.1, .2, .1, .1, .061])
    _, _, backend = start_pose_backend(tmp_path, pose)
    arm = sdk.controllers[0]
    backend.setup_for_recording()
    assert backend.start_position_applied and backend.mode == 'motion' and arm.tracking()
    np.testing.assert_allclose(np.r_[arm.positions, arm.gripper], pose)
    # The measured pose was seeded before any position gain was enabled.
    seeded, gripper, gain = arm.commands[0]
    np.testing.assert_array_equal(seeded, np.zeros(6))
    assert gripper == .04 and not np.any(gain.kp())
    assert 'holding arms and grippers until recording starts' in capsys.readouterr().out
    backend.start_recording()
    assert arm.passive()
    backend.shutdown()


def test_two_arms_move_to_both_saved_grippers_together(sdk, tmp_path):
    left, right = np.r_[np.full(6, .1), .031], np.r_[np.full(6, .2), .047]
    path = tmp_path / 'start.npz'
    np.savez(path, left_qpos=left, right_qpos=right)
    ctx = replace(recorder.resolve_run_context(arguments('--start-position', str(path), task='arx_bimanual')),
                  start_position_move_duration=.01)
    backend = recorder.ARXBackend(ctx)
    backend.setup_for_recording()
    for arm, pose in zip(sdk.controllers, (left, right)):
        np.testing.assert_allclose(np.r_[arm.positions, arm.gripper], pose)
        assert arm.tracking()
    assert backend.start_position_applied
    backend.shutdown()


def test_recorder_reloads_a_new_gripper_save_for_the_next_take(sdk, tmp_path):
    pose = np.array([.4, 1.2, 1.1, .2, .1, .1, .061])
    path, _, backend = start_pose_backend(tmp_path, pose)
    backend.setup_for_recording()
    np.savez(path, left_qpos=np.r_[pose[:6], .02])
    backend.start_recording()
    backend.setup_for_recording()
    np.testing.assert_allclose(np.r_[sdk.controllers[0].positions, sdk.controllers[0].gripper], np.r_[pose[:6], .02])
    backend.shutdown()


def test_missing_or_malformed_start_pose_never_enables_gains(sdk, tmp_path):
    _, ctx, backend = start_pose_backend(tmp_path, np.r_[np.zeros(6), np.nan])
    with pytest.raises(ValueError, match='seven finite positions'):
        backend.setup_for_recording()
    assert sdk.controllers[0].passive()
    missing = recorder.ARXBackend(replace(ctx, start_position_path=str(tmp_path / 'absent.npz')))
    with pytest.raises(FileNotFoundError):
        missing.setup_for_recording()
    assert sdk.controllers[-1].passive()


def test_review_holds_the_new_measured_pose_after_manual_motion(manual):
    backend, arm = manual.backend, manual.arm
    backend.setup_for_recording()
    backend.start_recording()
    arm.positions[:] = [.3, .8, 1.2, -.2, .1, .2]
    arm.gripper = .01
    backend.hold_after_recording()
    assert backend.mode == 'motion' and arm.tracking() and arm.friction is False
    target, gripper, _ = arm.commands[-1]
    np.testing.assert_array_equal(target, [.3, .8, 1.2, -.2, .1, .2])
    assert gripper == .01


def test_invalid_measured_hold_pose_never_enables_gains(manual):
    manual.arm.positions[0] = np.nan
    with pytest.raises(ValueError, match='seven finite positions'):
        manual.backend.enter_motion_mode()
    assert manual.arm.passive()


def test_start_key_does_not_poll_or_require_the_gripper(manual, monkeypatch):
    manual.backend.setup_for_recording()
    manual.arm.gripper = GRIPPER_WIDTH  # fully open: the old close-to-start trigger is gone
    monkeypatch.setattr(recorder, 'read_key_nonblocking', Mock(side_effect=[None, '1']))
    assert recorder.prompt_for_episode_start(arguments(), recorder.EpisodeEvents(), manual.backend, .001)
    assert manual.backend.mode == 'motion'  # still holding; recording mode starts with the take
    monkeypatch.setattr(recorder, 'read_key_nonblocking', lambda: 'x')
    with pytest.raises(recorder.QuitRequested):
        recorder.prompt_for_episode_start(arguments(), recorder.EpisodeEvents(), manual.backend, .001)


def test_replay_rows_send_one_timestamped_target_per_tick_with_scaled_gains(manual):
    backend, arm = manual.backend, manual.arm
    rows = np.linspace(0, .2, 5)[:, None] * np.ones(7)
    with pytest.raises(RuntimeError, match='motion mode'):
        backend.replay_rows(rows, fps=100, should_stop=lambda: False)
    backend.setup_for_recording()
    before = len(arm.commands)
    ticks = []
    assert backend.replay_rows(rows, fps=100, should_stop=lambda: False, kp_scale=.5, kd_scale=.25,
                               sleep=lambda seconds: None, on_tick=lambda index, row: ticks.append(index))
    sent = arm.commands[before:]
    assert ticks == [0, 1, 2, 3, 4] and len(sent) == 5
    for (target, gripper, gain), row in zip(sent, rows):
        np.testing.assert_allclose(np.r_[target, gripper], row)
        np.testing.assert_allclose(gain.kp(), TRACKING_KP * .5)
        np.testing.assert_allclose(gain.kd(), TRACKING_KD * .25)
        assert arm.friction is False
        assert gain.gripper_kp == ACTIVE_KP[6] * .5
    backend.enter_motion_mode()  # every exit restores the unscaled tracking gains
    assert arm.tracking()
    for bad in (rows[:, :6], np.full((2, 7), np.nan)):
        with pytest.raises(ValueError, match='seven finite positions'):
            backend.replay_rows(bad, fps=100, should_stop=lambda: False)
    with pytest.raises(ValueError, match='kp fraction'):
        backend.replay_rows(rows, fps=100, should_stop=lambda: False, kp_scale=0)
    with pytest.raises(ValueError, match='500 Hz'):
        backend.replay_rows(rows, fps=600, should_stop=lambda: False)


def test_stopped_replay_damps_natively_and_the_start_pose_move_restores_gains(manual):
    backend, arm = manual.backend, manual.arm
    backend.setup_for_recording()
    rows = np.full((4, 7), .01)
    stops = iter([False, True])
    assert not backend.replay_rows(rows, fps=100, should_stop=lambda: next(stops), sleep=lambda seconds: None)
    backend.enter_safe_stop()
    assert backend.mode == 'safe_stop' and 'damping' in arm.events
    assert not np.any(arm.gain.kp()) and arm.gain.gripper_kp == ACTIVE_KP[6]  # gripper held in place
    backend.setup_for_recording()
    assert backend.mode == 'motion' and arm.tracking()


def test_shutdown_homes_once_and_the_failure_exit_skips_the_home_move(sdk, manual):
    manual.backend.setup_for_recording()
    manual.backend.shutdown()
    manual.backend.shutdown()
    assert manual.arm.events.count('home') == 1
    with pytest.raises(RuntimeError, match='closed'):
        manual.backend.setup_for_recording()
    failed = recorder.ARXBackend(manual.ctx)
    failed.shutdown(home=False)
    assert 'home' not in sdk.controllers[-1].events


@pytest.fixture
def streams(monkeypatch):
    FakeStreams.instances = []
    monkeypatch.setattr(episode_buffer, 'DistributedRecorder', FakeStreams)

    def snapshot(names, *, copy_images):
        stream = FakeStreams.instances[-1]
        now = time.monotonic()
        return {}, dict(snapshot_timestamp=now, cameras={name: dict(
            frame_number=1001 + len(stream.timing[name]), capture_timestamp=now - .00101,
            consumer_received_timestamp=now - .000009) for name in names})

    monkeypatch.setattr(standard_capture, '_snapshot_cameras', snapshot)
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    stop, thread = threading.Event(), Mock()
    monkeypatch.setattr(recorder, 'start_camera_capture', lambda *args: (stop, thread))
    yield SimpleNamespace(stop=stop, thread=thread)


def collect(manual, monkeypatch):
    monkeypatch.setattr(recorder, 'resolve_run_context', lambda args: manual.ctx)
    monkeypatch.setattr(recorder, 'build_backend', lambda ctx: manual.backend)


def test_episode_saves_exact_frame_references_native_state_and_identity(manual, streams, monkeypatch):
    original_start = manual.backend.start_recording

    def start_recording():
        assert FakeStreams.instances and not FakeStreams.instances[-1].writer.count
        assert manual.backend.mode == 'motion'
        original_start()

    monkeypatch.setattr(manual.backend, 'start_recording', start_recording)
    original_snapshot = standard_capture._snapshot_cameras

    def snapshot(*args, **kwargs):
        assert manual.backend.mode == 'recording' and manual.arm.passive()
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(standard_capture, '_snapshot_cameras', snapshot)
    collect(manual, monkeypatch)
    monkeypatch.setattr(recorder, 'read_key_nonblocking', Mock(side_effect=['1', None, None, '1']))

    def accept():
        assert manual.backend.mode == 'motion' and manual.arm.tracking()
        return '1'

    monkeypatch.setattr(recorder, 'read_key_blocking', accept)
    recorder.main(arguments('--no-start-position', '--episode_idx', '0'))
    with episode_store.Store(Path(manual.ctx.dataset_dir) / 'episode_00000.zarr') as data:
        assert data['observations/qpos'].shape == (3, 7)
        assert data['observations/qvel'].shape == (3, 7)
        assert data.attrs['robot_type'] == 'arx'
        assert data.attrs['sdk_controller_hz'] == 500
        assert data.attrs['fps'] == 100
        assert data.attrs['start_gripper'] == 'keyboard'
        assert data.attrs['robot_contract']['arms']['left']['model'] == 'L5'
        assert not data.attrs['start_position_applied'] and not data.attrs['guided_start']
    raw = zarr.open_group(str(FakeStreams.instances[0].directory / 'control.zarr'), mode='r')
    np.testing.assert_array_equal(raw['camera_observation/video_left/frame_number'], [1001, 1003, 1005])
    np.testing.assert_allclose(np.asarray(raw['qvel']), np.zeros((3, 7)))
    assert manual.arm.events[-1] == 'home'  # the normal exit homes
    assert streams.stop.is_set()
    streams.thread.join.assert_called_once()


def test_record_replay_saves_the_take_then_its_recorded_replay(manual, streams, monkeypatch):
    from data_processing.episode_data import Episode
    import cv2
    monkeypatch.setattr(cv2, 'waitKey', lambda *_: -1)
    monkeypatch.setattr(cv2, 'destroyWindow', lambda *_: None)
    collect(manual, monkeypatch)
    original_sample = recorder.ReplayRecording.sample

    def sample(self, index, row):
        # Streams opened before the first replay capture, from the held start pose.
        assert self.samples.sample_count == index and self.samples.stream is FakeStreams.instances[-1]
        assert manual.backend.mode == 'motion'
        original_sample(self, index, row)

    monkeypatch.setattr(recorder.ReplayRecording, 'sample', sample)
    # Start, two samples, stop; then the replay overlay's go-ahead.
    nonblocking = iter(['1', None, None, '1', '1'])
    monkeypatch.setattr(recorder, 'read_key_nonblocking', lambda: next(nonblocking, None))
    # Save the take, record its replay, save the replay, quit.
    blocking = iter(['1', '1', '1', 'x'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(blocking))
    recorder.main(arguments('--no-start-position', '--record-replay'))
    assert len(FakeStreams.instances) == 2
    with Episode(Path(manual.ctx.dataset_dir) / 'episode_00000.zarr', inspection=True) as take:
        take_qpos = np.array(take.qpos)
        assert 'replay_source_episode' not in take.attrs
    with Episode(Path(manual.ctx.dataset_dir) / 'episode_00001.zarr', inspection=True) as replay:
        assert replay.attrs['replay_recording'] is True
        assert replay.attrs['replay_source_episode'] == 'episode_00000.zarr'
        assert (replay.attrs['replay_kp_scale'], replay.attrs['replay_kd_scale']) == (1.0, 1.0)
        np.testing.assert_allclose(replay.actions, take_qpos)  # commanded rows = the take
        assert np.array(replay.qpos).shape == (3, 7)
    raw = zarr.open_group(str(FakeStreams.instances[1].directory / 'control.zarr'), mode='r')
    assert len(raw['camera_observation/video_left/frame_number']) == 3
    assert manual.arm.events[-1] == 'home'


def test_record_replay_requires_auto_indices(manual, streams, monkeypatch):
    monkeypatch.setattr(recorder, 'resolve_run_context', lambda args: manual.ctx)
    with pytest.raises(ValueError, match='--record-replay'):
        recorder.main(arguments('--no-start-position', '--record-replay', '--episode_idx', '0'))


def test_abort_returns_to_the_motion_hold_and_closes_streams(manual, streams, monkeypatch):
    collect(manual, monkeypatch)
    monkeypatch.setattr(recorder, 'read_key_nonblocking', Mock(side_effect=['1', 'x']))
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: 'x')
    recorder.main(arguments('--no-start-position', '--episode_idx', '0'))
    assert not (Path(manual.ctx.dataset_dir) / 'episode_00000.zarr').exists()
    assert manual.arm.events[-1] == 'home'
    assert streams.stop.is_set()


def test_camera_consumer_stops_on_connection_failure(streams, monkeypatch):
    monkeypatch.setattr(recorder, 'build_backend', Mock(side_effect=RuntimeError('CAN unavailable')))
    with pytest.raises(RuntimeError, match='CAN unavailable'):
        recorder.main(arguments('--no-start-position'))
    assert streams.stop.is_set()
    streams.thread.join.assert_called_once()


def test_quit_before_recording_homes_normally_without_a_stream(manual, streams, monkeypatch):
    collect(manual, monkeypatch)
    monkeypatch.setattr(recorder, 'read_key_nonblocking', lambda: 'x')
    recorder.main(arguments('--no-start-position'))
    assert FakeStreams.instances == [] and manual.arm.events[-1] == 'home'


def test_failure_during_a_take_damps_until_h_then_homes_without_a_second_home(manual, streams, monkeypatch):
    collect(manual, monkeypatch)
    monkeypatch.setattr(recorder, 'read_key_nonblocking', Mock(side_effect=['1', None]))
    monkeypatch.setattr(manual.backend, 'read_state', Mock(side_effect=RuntimeError('feedback failed')))
    monkeypatch.setattr(recorder, 'home_key_pressed', lambda: True)
    with pytest.raises(RuntimeError, match='feedback failed'):
        recorder.main(arguments('--no-start-position', '--episode_idx', '0'))
    assert 'damping' in manual.arm.events and manual.arm.events.count('home') == 1
    assert streams.stop.is_set()


def test_guided_start_freezes_the_hand_placed_pose_and_can_reguide(manual, monkeypatch):
    backend, arm = manual.backend, manual.arm
    first, second = np.array([.3, .8, 1.2, -.2, .1, .2]), np.array([.5, .6, 1.3, -.1, .2, -.2])

    def place(pose, gripper):
        assert backend.mode == 'guided' and arm.passive() and arm.friction is True
        arm.positions[:] = pose
        arm.gripper = gripper

    def frozen_at(pose, gripper):
        assert backend.mode == 'motion' and arm.tracking()
        target, held_gripper, _ = arm.commands[-1]
        np.testing.assert_array_equal(target, pose)
        assert held_gripper == gripper

    steps = iter([lambda: place(first, .03), '1', lambda: frozen_at(first, .03), 'h',
                  lambda: place(second, .05), '1', lambda: frozen_at(second, .05), '1'])

    def keys():
        step = next(steps)
        if callable(step):
            step()
            return None
        return step

    monkeypatch.setattr(recorder, 'read_key_nonblocking', keys)
    assert recorder.prompt_for_episode_start(arguments('--guided-start'), recorder.EpisodeEvents(),
                                             backend, .001)
    assert backend.mode == 'motion' and not backend.start_position_applied
    assert next(steps, None) is None


def test_guided_start_skips_the_saved_pose_and_records_from_the_frozen_pose(sdk, streams, tmp_path, monkeypatch):
    saved = np.array([.4, 1.2, 1.1, .2, .1, .1, .061])
    path, ctx, backend = start_pose_backend(tmp_path, saved, '--guided-start')
    arm = sdk.controllers[0]
    placed = np.array([.3, .8, 1.2, -.2, .1, .2])
    original_start = backend.start_recording

    def start_recording():
        assert backend.mode == 'motion'  # frozen; streams already open
        assert FakeStreams.instances and not FakeStreams.instances[-1].writer.count
        original_start()

    def place():
        arm.positions[:] = placed
        return None

    monkeypatch.setattr(backend, 'start_recording', start_recording)
    monkeypatch.setattr(recorder, 'resolve_run_context', lambda args: ctx)
    monkeypatch.setattr(recorder, 'build_backend', lambda ctx: backend)
    keys = iter([place, '1', '1', None, None, '1'])  # place, freeze, record, stop
    monkeypatch.setattr(recorder, 'read_key_nonblocking',
                        lambda: (lambda key: key() if callable(key) else key)(next(keys, None)))
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: '1')
    recorder.main(arguments('--start-position', str(path), '--guided-start', '--episode_idx', '0'))
    # The saved pose was never commanded; the holds were at the placed pose.
    targets = [target for target, _, _ in arm.commands]
    assert targets and not any(np.allclose(target, saved[:6]) for target in targets)
    assert any(np.allclose(target, placed) for target in targets)
    with episode_store.Store(Path(ctx.dataset_dir) / 'episode_00000.zarr') as data:
        assert data.attrs['guided_start'] is True
        assert not data.attrs['start_position_applied']
        np.testing.assert_allclose(data['observations/qpos'][0][:6], placed)
    assert arm.events[-1] == 'home'


def test_guided_start_quit_holds_before_the_normal_home_move(manual, streams, monkeypatch):
    collect(manual, monkeypatch)
    monkeypatch.setattr(recorder, 'read_key_nonblocking', lambda: 'x')
    modes = []
    original = manual.backend.shutdown
    monkeypatch.setattr(manual.backend, 'shutdown',
                        lambda **kwargs: (modes.append(manual.backend.mode), original(**kwargs)))
    recorder.main(arguments('--no-start-position', '--guided-start'))
    assert modes == ['motion'] and FakeStreams.instances == []
    assert manual.arm.events[-1] == 'home'


def test_inference_and_action_replay_controllers_start_with_friction_feedforward_off(sdk, monkeypatch):
    from hardware.backends.arx import ARXAdapter
    from inference import inference_direct as direct
    monkeypatch.setattr(direct, 'arx5', sdk)
    direct.init_arm(is_double_arm=True)
    assert [arm.config.friction_compensation for arm in sdk.controllers] == [False, False]
    sdk.controllers.clear()
    adapter = ARXAdapter('left', 'L5', 'can0', sdk=sdk)
    adapter.connect()
    try:
        assert sdk.controllers[0].config.friction_compensation is False
    finally:
        adapter.close()


def test_setup_and_dagger_use_friction_feedforward_only_while_the_arm_is_hand_guided(sdk):
    from inference import inference_direct as direct
    setup_dir = str(Path(__file__).resolve().parents[1] / 'setup')
    if setup_dir not in sys.path:
        sys.path.insert(0, setup_dir)
    import set_up_recording as setup
    arm = Controller(None, SimpleNamespace(), 'can0')
    snapshot = recorder.arx_tracking_gain_snapshot(SimpleNamespace(
        default_kp=TRACKING_KP, default_kd=TRACKING_KD, default_gripper_kp=ACTIVE_KP[6], default_gripper_kd=ACTIVE_KD[6]))
    direct.apply_gain_mode({'left': arm}, {'left': snapshot}, False, zero_gains=True)
    assert arm.passive() and arm.friction is True  # DAgger correction: hand guided
    direct.apply_gain_mode({'left': arm}, {'left': snapshot}, False, zero_gains=False)
    assert arm.tracking() and arm.friction is False  # policy control
    setup.enter_practice_mode({'left': arm})
    assert arm.passive() and arm.friction is True
    setup.enter_hold_mode({'left': arm}, {'left': snapshot})
    assert arm.tracking() and arm.friction is False
    setup.enter_gripper_adjust_mode(sdk, SimpleNamespace(joint_dof=6), {'left': arm}, {'left': snapshot},
                                    {'left': np.zeros(7)})
    assert arm.gain.gripper_kp == 0 and np.array_equal(arm.gain.kp(), TRACKING_KP) and arm.friction is True
