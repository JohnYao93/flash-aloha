"""Camera-free manual practice with simulated SDK controllers only."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import practice


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(practice.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(practice.time, 'sleep', lambda dt: now.__setitem__(0, now[0] + dt))
    return now


@pytest.mark.parametrize('arm,interfaces', [
    ('left', {'left': 'can0'}), ('right', {'right': 'can1'}),
    ('both', {'left': 'can0', 'right': 'can1'})])
def test_selected_arms_run_gravity_until_duration_then_close_without_homing(monkeypatch, clock, arm, interfaces):
    controllers = {role: Mock() for role in interfaces}
    arx = Mock(return_value=controllers)
    monkeypatch.setattr(practice, 'build_arx_controllers', arx)
    args = practice.build_parser().parse_args(['--arm', arm, '--duration', '0.04'])
    assert practice.run(args) == 0
    assert arx.call_args.args == (interfaces, 'L5')
    for controller in controllers.values():
        assert controller.practice_step.call_count >= 2
        controller.close.assert_called_once()
        controller.reset_to_home.assert_not_called()


@pytest.mark.parametrize('error,result', [(KeyboardInterrupt(), 0), (RuntimeError('feedback failed'), 1)])
def test_interrupt_or_failure_damps_until_h_then_homes_then_closes(monkeypatch, clock, error, result):
    from hardware import safe_exit
    controllers = {'left': Mock(), 'right': Mock()}
    controllers['left'].practice_step.side_effect = [None, error]
    events = []

    class Damping:
        def __init__(self, *args, **kwargs):
            events.append('target')
        def damp(self): events.append('damp')
        def refresh(self): events.append('refresh')
        def go_home(self): events.append('home')
    monkeypatch.setattr(practice, 'ARXDamping', Damping)
    monkeypatch.setattr(safe_exit, 'HomeKeyReader', lambda: SimpleNamespace(
        home=SimpleNamespace(is_set=lambda: events.count('refresh') >= 2)))
    monkeypatch.setattr(practice, 'build_arx_controllers', Mock(return_value=controllers))
    args = practice.build_parser().parse_args(['--arm', 'both'])
    assert practice.run(args) == result
    assert events == ['target', 'damp', 'refresh', 'refresh', 'home']
    for controller in controllers.values():
        controller.close.assert_called_once()  # after the operator-confirmed home move
        controller.reset_to_home.assert_not_called()


def test_one_shutdown_failure_still_closes_the_other_arm(monkeypatch, clock):
    controllers = {'left': Mock(), 'right': Mock()}
    controllers['left'].close.side_effect = RuntimeError('disable failed')
    monkeypatch.setattr(practice, 'build_arx_controllers', Mock(return_value=controllers))
    assert practice.run(practice.build_parser().parse_args(['--arm', 'both', '--duration', '0.02'])) == 1
    controllers['right'].close.assert_called_once()


@pytest.mark.parametrize('flags', [
    ['--controller-fps', 'nan'], ['--controller-fps', '501'], ['--controller-fps', '0'],
    ['--duration', '-1'], ['--duration', 'inf'], ['--arm', 'both', '--can-interface', 'can0'],
    ['--arm', 'both', '--can-interface', 'can0,can0'], ['--left-can-interface', '']])
def test_invalid_arguments_do_not_construct_hardware(monkeypatch, flags):
    builder = Mock()
    monkeypatch.setattr(practice, 'build_arx_controllers', builder)
    with pytest.raises(ValueError):
        practice.run(practice.build_parser().parse_args(flags))
    builder.assert_not_called()


def test_custom_interface_and_model_reach_the_arx_builder(monkeypatch, clock):
    builder = Mock(return_value={'right': Mock()})
    monkeypatch.setattr(practice, 'build_arx_controllers', builder)
    args = practice.build_parser().parse_args([
        '--arm', 'right', '--can-interface', 'can3', '--arx-model', 'Trossen_L5', '--duration', '0.02'])
    assert practice.run(args) == 0
    builder.assert_called_once_with({'right': 'can3'}, 'Trossen_L5')


def fake_arx(monkeypatch, second_failure=False):
    class Gain:
        gripper_kp = 5.0
        gripper_kd = 1.0
        def __init__(self):
            self.p = np.ones(6)
            self.d = np.ones(6)
        def kp(self): return self.p
        def kd(self): return self.d
    first, second = Mock(), Mock()
    first.get_gain.return_value = Gain()
    second.get_gain.return_value = Gain()
    config = SimpleNamespace()
    robot = object()
    factory = Mock(side_effect=[first, RuntimeError('second connection failed')] if second_failure else [first, second])
    sdk = SimpleNamespace(
        RobotConfigFactory=Mock(), ControllerConfigFactory=Mock(), Arx5JointController=factory)
    sdk.RobotConfigFactory.get_instance.return_value.get_config.return_value = robot
    sdk.ControllerConfigFactory.get_instance.return_value.get_config.return_value = config
    importer = Mock(return_value=sdk)
    monkeypatch.setattr(practice.importlib, 'import_module', importer)
    return first, second, config, importer


def test_arx_uses_sdk_gravity_with_zero_gains_and_native_shutdown(monkeypatch):
    first, second, config, importer = fake_arx(monkeypatch)
    controllers = practice.build_arx_controllers({'left': 'can0', 'right': 'can1'}, 'L5')
    importer.assert_called_once_with('arx5_interface')
    assert config.gravity_compensation and config.background_send_recv
    assert not config.friction_compensation
    assert config.controller_dt == 0.002
    for native in (first, second):
        gain = native.set_gain.call_args.args[0]
        np.testing.assert_array_equal(gain.kp(), np.zeros(6))
        np.testing.assert_array_equal(gain.kd(), np.zeros(6))
        assert gain.gripper_kp == gain.gripper_kd == 0
    assert practice.close_controllers(controllers)
    for native in (first, second):
        native.set_to_damping.assert_called_once()
        native.reset_to_home.assert_not_called()


def test_arx_partial_connection_failure_closes_first_controller(monkeypatch):
    first, _, _, _ = fake_arx(monkeypatch, second_failure=True)
    with pytest.raises(RuntimeError, match='second connection failed'):
        practice.build_arx_controllers({'left': 'can0', 'right': 'can1'}, 'L5')
    first.set_to_damping.assert_called_once()
