"""Physical replay scheduling exercised with a fake SDK and clock only."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize('fps', [50, 100, 200])
@pytest.mark.parametrize('arm_count', [1, 2])
def test_standalone_arx_replay_sends_current_rows_with_fixed_intervals(monkeypatch, fps, arm_count):
    from utils import replay_utils as replay
    now, sent = [0.], [[], []]
    sdk = Mock()
    sdk.ControllerConfigFactory.get_instance.return_value.get_config.return_value = SimpleNamespace(default_kp=np.zeros(6))
    sdk.RobotConfigFactory.get_instance.return_value.get_config.return_value = SimpleNamespace(joint_dof=6)
    controllers = [Mock() for _ in range(arm_count)]
    sdk.Arx5JointController.side_effect = controllers
    for arm, controller in enumerate(controllers):
        controller.get_timestamp.side_effect = lambda: now[0]
        def send(command, arm=arm):
            qpos = np.r_[command.pos().copy(), command.gripper_pos]
            sent[arm].append((now[0], command.timestamp, qpos))
        controller.set_joint_cmd.side_effect = send
    def command(*args):
        values = np.zeros(6)
        return SimpleNamespace(pos=lambda: values, gripper_pos=0., timestamp=0.)
    monkeypatch.setitem(sys.modules, 'arx5_interface', sdk)
    monkeypatch.setattr(replay, 'make_arx_joint_command', command)
    monkeypatch.setattr(replay, 'configure_arx_gain', lambda *a, **kw: None)
    monkeypatch.setattr(replay, 'format_arx_gain_filename_suffix', lambda *a: '')
    monkeypatch.setattr(replay, 'save_tracking_plots', lambda *a, **kw: None)
    monkeypatch.setattr(replay.time, 'time', lambda: now[0])
    monkeypatch.setattr(replay.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(replay.time, 'sleep', lambda dt: now.__setitem__(0, now[0] + dt))
    actions = np.array([np.zeros(7 * arm_count), np.arange(7 * arm_count), np.full(7 * arm_count, 20)])
    replay.replay_arx_actions(actions, dt=1 / fps, is_dual_arm=arm_count == 2, plot_tracking=False)
    for arm in range(arm_count):
        rows = sent[arm][-len(actions):]
        assert np.diff([row[0] for row in rows]) == pytest.approx([1 / fps, 1 / fps])
        for index, (issued, deadline, qpos) in enumerate(rows):
            assert deadline - issued == pytest.approx(1 / fps)
            np.testing.assert_array_equal(qpos, actions[index, arm*7:(arm+1)*7])
        assert controllers[arm].reset_to_home.call_count == 2


@pytest.mark.parametrize('interval', [None, .01, .02])
def test_shared_dispatch_preserves_immediate_holds_and_target_deadlines(interval):
    from utils.replay_utils import send_replay_command
    controller = Mock()
    controller.get_timestamp.return_value = 12.5
    command = SimpleNamespace(timestamp=-1., gripper_pos=.03)
    send_replay_command(controller, command, interval)
    controller.set_joint_cmd.assert_called_once_with(command)
    assert command.timestamp == (-1. if interval is None else 12.5 + interval)
    assert command.gripper_pos == .03
    assert controller.get_timestamp.call_count == (0 if interval is None else 1)


def test_viewer_routes_replay_to_shared_utilities(monkeypatch, tmp_path):
    from visualization import visualize_episode as viewer
    from utils import replay_utils
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    timestamps = np.arange(3, dtype=np.float64) / 100
    root = SimpleNamespace(attrs={'fps': 100})
    store = Mock()
    store.__enter__ = Mock(return_value=type('Root', (dict,), {})(timestamps=timestamps))
    store.__enter__.return_value.attrs = root.attrs
    store.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(viewer.episode_store, 'Store', lambda *args, **kwargs: store)
    monkeypatch.setattr(viewer, 'load_episode', lambda *args, **kwargs: (
        actions.copy(), {}, np.zeros_like(actions), None, actions, timestamps,
        None, None, None, None, None, None, {}))
    replay = Mock()
    monkeypatch.setattr(replay_utils, 'replay_arx_actions', replay)
    monkeypatch.setattr(replay_utils, 'DT', .02)
    viewer.main(dict(robot='arx', task_name='synthetic_replay', episode_idx=0,
                     dataset_dir=str(tmp_path), action_replay=True))
    replay.assert_called_once()
    np.testing.assert_array_equal(replay.call_args.args[0], actions)
    assert replay.call_args.kwargs['dt'] == .01
    assert replay.call_args.kwargs['is_dual_arm'] is False
