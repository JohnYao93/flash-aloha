from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from hardware.arm_interface import ArmDescription, ArmState, ManagedArm
from hardware.runtime import HardwareRuntime
from hardware.arm_factory import build_arms, normalize_hardware_config
from hardware.backends.arx import ARXAdapter
from test_robot_contract import contract

class FakeArm(ManagedArm):
    def __init__(self, role='left', model='Trossen_L5', hz=200, **options):
        self.sent = []
        self.closed_count = 0
        self.fail_connect = False
        self.reads = 0
        self.state = ArmState(np.arange(7), np.arange(7) * 2, np.zeros(7), np.ones(7))
        super().__init__(ArmDescription(role, model, 'fake', 7, hz), **options)
    def _connect(self):
        if self.fail_connect:
            raise RuntimeError('connect failure')
    def _read_state(self):
        self.reads += 1
        return self.state
    def _submit_target(self, target, *, interval_s):
        self.sent.append((target, interval_s))
    def _close(self):
        self.closed_count += 1
    def _validate_limits(self, target):
        if np.any(np.abs(target) > 20):
            raise ValueError('limits')


def test_imports_and_factory_are_inert():
    code = "import hardware.arm_factory; import hardware.runtime; import sys; assert 'arx5_interface' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)
    arms = build_arms({'type': 'arx', 'arms': {'right': {'model': 'L5', 'can_interface': 'can1'}}})
    assert arms['right'].describe().role == 'right'
    assert arms['right']._controller is None
    arms['right'].close()


@pytest.mark.parametrize('config', [
    {'type': 'other', 'arms': {'left': {'model': 'L5', 'can_interface': 'can0'}}},
    {'type': 'arx', 'arms': {'left': {'model': 'L5', 'can_interface': 'can0', 'config_path': 'arm.toml'}}},
    {'type': 'arx', 'arms': {'left': {'model': 'L5', 'can_interface': 'can0'}, 'right': {'model': 'L5', 'can_interface': 'can0'}}},
    {'type': 'arx', 'arms': {'left': {'model': 'X5', 'can_interface': 'can0'}}},
    {'type': 'arx', 'arms': {'active': {'model': 'L5', 'can_interface': 'can0'}}}])
def test_invalid_hardware_config(config):
    with pytest.raises(ValueError):
        normalize_hardware_config(config)


def test_state_is_frozen_and_qpos_qvel_share_one_read():
    arm = FakeArm()
    runtime = HardwareRuntime({'left': arm})
    runtime.connect()
    try:
        value = runtime.read_state()
        assert arm.reads == 1
        np.testing.assert_array_equal(value['qvel'], value['qpos'] * 2)
        with pytest.raises(ValueError):
            arm.state.qpos[0] = 99
        observation = runtime.capture([])
        assert observation.recording_fields()['qpos'].shape == (7,)
    finally:
        runtime.close()


@pytest.mark.parametrize('metadata', [{}, {'robot_contract': contract('L5'), 'control_freq': 50},
    {'robot_contract': contract(role='right'), 'control_freq': 50},
    {'robot_contract': contract(), 'control_freq': 250}])
def test_policy_rejected_before_connection(metadata):
    arm = FakeArm()
    runtime = HardwareRuntime({'left': arm})
    with pytest.raises(ValueError):
        runtime.require_policy(metadata)
    assert not arm._connected
    runtime.close()


def test_stats_contract_required_and_action_rate_stays_separate():
    arm = FakeArm()
    runtime = HardwareRuntime({'left': arm})
    metadata = {'robot_contract': contract(), 'control_freq': 50, 'action_dim': 7}
    with pytest.raises(ValueError, match='Missing'):
        runtime.require_policy(metadata, stats={})
    runtime.require_policy(metadata, stats={'robot_contract': contract()})
    runtime.connect()
    try:
        target = np.arange(7, dtype=float)
        runtime.submit_target(target)
        target[:] = 99
        assert arm.sent[0][1] == 0.02
        np.testing.assert_array_equal(arm.sent[0][0], np.arange(7))
        assert runtime.recording_metadata(50)['sdk_controller_hz_by_role'] == {'left': 200}
    finally:
        runtime.close()


def test_dual_target_validated_before_first_send():
    arms = {'right': FakeArm('right'), 'left': FakeArm()}
    runtime = HardwareRuntime(arms)
    runtime.connect()
    try:
        with pytest.raises(ValueError):
            runtime.submit_target([0] * 7 + [21] * 7, interval_s=.02)
        assert all(not arm.sent for arm in arms.values())
        runtime.submit_target(np.arange(14), interval_s=.02)
        np.testing.assert_array_equal(arms['left'].sent[0][0], np.arange(7))
        np.testing.assert_array_equal(arms['right'].sent[0][0], np.arange(7, 14))
    finally:
        runtime.close()


@pytest.mark.parametrize('failure', ['age', 'fault', 'skew'])
def test_feedback_failure_disables_arms_that_cannot_damp(failure):
    left, right = FakeArm(), FakeArm('right')
    if failure == 'age':
        left.state = ArmState(np.zeros(7), np.zeros(7), np.ones(7), np.ones(7))
    elif failure == 'fault':
        left.state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7), ('motor fault',))
    else:
        right.state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7) + .03)
    runtime = HardwareRuntime({'left': left, 'right': right})
    runtime.connect()
    with pytest.raises(RuntimeError):
        runtime.read_state()
    assert left.closed_count == right.closed_count == 1


def test_connection_failure_closes_successful_peer_and_is_idempotent():
    left, right = FakeArm(), FakeArm('right')
    right.fail_connect = True
    runtime = HardwareRuntime({'left': left, 'right': right})
    with pytest.raises(RuntimeError, match='connect failure'):
        runtime.connect()
    runtime.close()
    assert left.closed_count == right.closed_count == 1


def test_command_expiry_disables_an_arm_that_cannot_damp():
    now = [10.]
    arm = FakeArm(clock=lambda: now[0])
    arm.connect()
    arm.submit_target(np.zeros(7), interval_s=.02)
    now[0] += .6
    with pytest.raises(RuntimeError, match='expired'):
        arm.check_command_expiry()
    assert arm.closed_count == 1
    arm.close()


def test_watchdog_disables_an_undampable_arm_without_another_call():
    arm = FakeArm(command_timeout_s=.04)
    arm.connect()
    arm.submit_target(np.zeros(7), interval_s=.005)
    deadline = time.monotonic() + 1
    while arm.closed_count == 0 and time.monotonic() < deadline:
        time.sleep(.01)
    arm.close()
    assert arm.closed_count == 1
    assert 'expired' in arm._error


def test_arx_stamps_target_in_native_clock_and_never_homes():
    class Joint:
        def __init__(self, dof):
            self._pos = np.zeros(dof); self._vel = np.zeros(dof)
            self.gripper_pos = .02; self.gripper_vel = .01; self.timestamp = 10.
        def pos(self): return self._pos
        def vel(self): return self._vel
    class Gain:
        def __init__(self, dof): self._kp = np.zeros(dof); self._kd = np.zeros(dof)
        def kp(self): return self._kp
        def kd(self): return self._kd
    robot = SimpleNamespace(joint_dof=6, joint_pos_min=np.full(6, -10),
        joint_pos_max=np.full(6, 10), gripper_width=.1)
    config = SimpleNamespace(default_kp=np.zeros(6))
    controller = Mock()
    controller.get_joint_state.return_value = Joint(6)
    controller.get_timestamp.return_value = 10.001
    sdk = SimpleNamespace(RobotConfigFactory=Mock(), ControllerConfigFactory=Mock(),
        Arx5JointController=Mock(return_value=controller), JointState=Joint, Gain=Gain)
    sdk.RobotConfigFactory.get_instance().get_config.return_value = robot
    sdk.ControllerConfigFactory.get_instance().get_config.return_value = config
    arm = ARXAdapter('left', 'L5', 'can0', sdk=sdk)
    arm.connect()
    controller.set_gain.assert_not_called()
    controller.set_to_damping.reset_mock()
    try:
        arm.read_state()
        arm.submit_target([1] * 6 + [.03], interval_s=.02)
        command = controller.set_joint_cmd.call_args.args[0]
        assert command.timestamp == pytest.approx(10.021)
        assert config.controller_dt == .002
        controller.reset_to_home.assert_not_called()
        with pytest.raises(ValueError):
            arm.submit_target([1] * 6 + [.2], interval_s=.02)
    finally:
        arm.close()
    controller.set_to_damping.assert_called_once()


def test_connection_barriers_do_not_start_command_expiry_before_activation():
    now = [0.]
    arm = FakeArm(clock=lambda: now[0])
    arm.connect()
    now[0] = 10.
    arm.check_command_expiry()
    assert arm.closed_count == 0
    arm.submit_target(np.zeros(7), interval_s=.02)
    now[0] = 10.6
    with pytest.raises(RuntimeError, match='expired'):
        arm.check_command_expiry()
    arm.close()


class DampingArm(FakeArm):
    """Fake arm with a damping mode, like the ARX adapter."""
    def __init__(self, *args, fail_damp=False, **kwargs):
        self.damped = 0
        self.homed = 0
        self.fail_damp = fail_damp
        super().__init__(*args, **kwargs)
    def _submit_safe_stop(self, *, interval_s):
        if self.fail_damp:
            raise RuntimeError('200 Hz controller failed')
        self.damped += 1
    def _home(self):
        self.homed += 1


@pytest.mark.parametrize('failure', ['age', 'fault', 'skew'])
def test_feedback_failure_damps_every_arm_and_blocks_targets_until_acknowledged(failure):
    left, right = DampingArm(), DampingArm('right')
    if failure == 'age':
        left.state = ArmState(np.zeros(7), np.zeros(7), np.ones(7), np.ones(7))
    elif failure == 'fault':
        left.state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7), ('motor fault',))
    else:
        right.state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7) + .03)
    runtime = HardwareRuntime({'left': left, 'right': right})
    runtime.connect()
    try:
        with pytest.raises(RuntimeError):
            runtime.read_state()
        assert left.closed_count == right.closed_count == 0
        assert left.damped >= 1 and right.damped >= 1 and left.faulted and right.faulted
        with pytest.raises(RuntimeError):
            runtime.submit_target(np.zeros(14), interval_s=.02)
        assert not left.sent and not right.sent
        runtime.safe_stop(interval_s=.01)  # damping can be refreshed while faulted
        left.state = right.state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7))
        runtime.acknowledge()
        runtime.submit_target(np.zeros(14), interval_s=.02)
        assert len(left.sent) == len(right.sent) == 1
    finally:
        runtime.close()


def test_rejected_target_does_not_fault_or_move_any_arm():
    left = DampingArm()
    runtime = HardwareRuntime({'left': left})
    runtime.connect()
    with pytest.raises(ValueError, match='limits'):
        runtime.submit_target([30] + [0] * 6, interval_s=.02)
    assert not left.sent and not left.faulted  # caller's damping_exit decides what happens next
    runtime.close()


def test_watchdog_expiry_damps_natively_instead_of_disabling():
    arm = DampingArm(command_timeout_s=.04)
    arm.connect()
    try:
        arm.submit_target(np.zeros(7), interval_s=.005)
        deadline = time.monotonic() + 1
        while arm.damped == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        assert arm.damped >= 1 and arm.faulted and arm.closed_count == 0
        assert 'expired' in arm._error
    finally:
        arm.close()


def test_stopped_damping_refresh_is_held_natively_by_the_watchdog():
    now = [0.]
    arm = DampingArm(clock=lambda: now[0])
    arm.connect()
    arm.submit_target(np.zeros(7), interval_s=.02)
    now[0] = 1.
    with pytest.raises(RuntimeError, match='expired'):
        arm.check_command_expiry()
    arm.submit_safe_stop(interval_s=.01)  # refresh renews the deadline
    before = arm.damped
    now[0] = 2.
    arm.check_command_expiry()  # refresh stopped: re-damp, never disable
    assert arm.damped == before + 1 and arm.closed_count == 0
    arm.close()


def test_one_undampable_arm_is_disabled_while_the_other_keeps_damping():
    left, right = DampingArm(fail_damp=True), DampingArm('right')
    runtime = HardwareRuntime({'left': left, 'right': right})
    runtime.connect()
    left.state = ArmState(np.zeros(7), np.zeros(7), np.ones(7), np.ones(7))
    with pytest.raises(RuntimeError):
        runtime.read_state()
    assert left.closed_count == 1 and right.closed_count == 0 and right.damped >= 1
    runtime.safe_stop(interval_s=.01)
    assert list(runtime.live_arms()) == ['right']
    right.fail_damp = True
    with pytest.raises(RuntimeError):
        runtime.safe_stop(interval_s=.01)
    assert not runtime.live_arms()
    runtime.close()


def test_arx_damping_is_native_and_the_next_target_restores_gains():
    controller = Mock()
    joint = SimpleNamespace(pos=lambda: np.zeros(6), vel=lambda: np.zeros(6), gripper_pos=.02,
                            gripper_vel=0., timestamp=10.)
    controller.get_joint_state.return_value = joint
    controller.get_timestamp.return_value = 10.0
    robot = SimpleNamespace(joint_dof=6, joint_pos_min=np.full(6, -10), joint_pos_max=np.full(6, 10), gripper_width=.1)
    class Joint:
        def __init__(self, dof):
            self._pos = np.zeros(dof); self.gripper_pos = 0.
        def pos(self): return self._pos
    class Gain:
        def __init__(self, dof): self._kp = np.zeros(dof); self._kd = np.zeros(dof)
        def kp(self): return self._kp
        def kd(self): return self._kd
    sdk = SimpleNamespace(RobotConfigFactory=Mock(), ControllerConfigFactory=Mock(),
                          Arx5JointController=Mock(return_value=controller), JointState=Joint, Gain=Gain)
    sdk.RobotConfigFactory.get_instance().get_config.return_value = robot
    sdk.ControllerConfigFactory.get_instance().get_config.return_value = SimpleNamespace(
        default_kp=np.ones(6), default_kd=np.ones(6), default_gripper_kp=1., default_gripper_kd=1.)
    arm = ARXAdapter('left', 'L5', 'can0', sdk=sdk)
    arm.connect()
    try:
        arm.submit_target([0] * 6 + [.03], interval_s=.02)
        assert controller.set_gain.call_count == 1
        controller.set_to_damping.reset_mock()
        controller.get_controller_config.return_value = SimpleNamespace(
            default_gripper_kp=5.0, default_gripper_kd=0.2)
        arm.fail('camera')
        arm.submit_safe_stop(interval_s=.01)
        arm.submit_safe_stop(interval_s=.01)
        controller.set_to_damping.assert_called_once()  # native hold; refresh only reads
        held = controller.set_gain.call_args.args[0]
        assert (held.gripper_kp, held.gripper_kd) == (5.0, 0.2)  # gripper held in place
        arm.acknowledge()
        arm.home()
        controller.reset_to_home.assert_called_once()
        arm.submit_target([0] * 6 + [.03], interval_s=.02)
        assert controller.set_gain.call_count == 3  # activation, gripper hold, re-activation
    finally:
        arm.close()



def test_runtime_damping_acknowledges_then_homes_every_live_arm_natively():
    from hardware.runtime import RuntimeDamping
    left, right = DampingArm(), DampingArm('right')
    runtime = HardwareRuntime({'left': left, 'right': right})
    runtime.connect()
    target = RuntimeDamping(runtime, interval_s=.01)
    left.fail('test'); right.fail('test')
    target.damp(); target.refresh()
    target.go_home()
    assert not left.faulted and not right.faulted  # acknowledged by the operator's 'h'
    assert left.homed == right.homed == 1 and not left.sent and not right.sent
    runtime.close()
