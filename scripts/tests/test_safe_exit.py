"""The shared failure exit: damping until 'h', then home."""
import os
import signal
import threading
from types import SimpleNamespace

import pytest

from hardware import safe_exit
from hardware.safe_exit import ARXDamping, QuitRequested, damp_until_home, damping_exit


class Target:
    def __init__(self, *, home_failures=0, damp_error=None):
        self.events = []
        self.home_failures = home_failures
        self.damp_error = damp_error

    def damp(self):
        if self.damp_error:
            raise self.damp_error
        self.events.append('damp')

    def refresh(self):
        self.events.append('refresh')

    def go_home(self):
        self.events.append('home')
        if self.home_failures:
            self.home_failures -= 1
            raise RuntimeError('home move failed')


def after(target, refreshes):
    return lambda: target.events.count('refresh') >= refreshes


def run(target, *, refreshes=3, **kwargs):
    return damp_until_home(target, RuntimeError('boom'), home_requested=after(target, refreshes),
                           notify=lambda *_: None, install_signals=False,
                           clock=lambda: 0.0, sleep=lambda _: None, **kwargs)


def test_damps_and_refreshes_until_h_then_homes():
    target = Target()
    run(target)
    assert target.events == ['damp', 'refresh', 'refresh', 'refresh', 'home']


def test_failed_home_move_returns_to_damping_then_homes_again():
    target = Target(home_failures=1)
    run(target)
    assert target.events.count('damp') == 2 and target.events[-1] == 'home'
    assert target.events.count('home') == 2


def test_ctrl_c_while_damped_requests_disable_in_place():
    target = Target()
    with safe_exit.operator_stop_signals(threading.Event()) as _:
        pass
    calls = {'n': 0}

    def requested():
        calls['n'] += 1
        if calls['n'] == 3:
            os.kill(os.getpid(), signal.SIGINT)
        return False
    with pytest.raises(KeyboardInterrupt):
        damp_until_home(target, 'test', home_requested=requested, notify=lambda *_: None,
                        clock=lambda: 0.0, sleep=lambda _: None)
    assert 'home' not in target.events
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_damping_exit_homes_then_reraises_and_reports_state():
    target = Target()
    with pytest.raises(ValueError, match='joint limits'):
        with damping_exit(lambda: target, home_requested=after(target, 1), notify=lambda *_: None,
                          install_signals=False, clock=lambda: 0.0, sleep=lambda _: None) as state:
            raise ValueError('target exceeds native joint limits')
    assert state.homed and isinstance(state.failure, ValueError)
    assert target.events[0] == 'damp' and target.events[-1] == 'home'


def test_ctrl_c_in_the_block_also_damps():
    target = Target()
    with pytest.raises(KeyboardInterrupt):
        with damping_exit(lambda: target, home_requested=after(target, 1), notify=lambda *_: None,
                          install_signals=False, clock=lambda: 0.0, sleep=lambda _: None) as state:
            raise KeyboardInterrupt
    assert state.homed


def test_intentional_quit_skips_damping():
    target = Target()
    with pytest.raises(QuitRequested):
        with damping_exit(lambda: target) as state:
            raise QuitRequested
    assert target.events == [] and state.failure is None


def test_nothing_connected_just_reraises():
    with pytest.raises(RuntimeError):
        with damping_exit(lambda: None) as state:
            raise RuntimeError('before connecting')
    assert state.failure is not None and not state.homed


def test_undampable_arm_is_reported_and_original_error_kept():
    target = Target(damp_error=RuntimeError('200 Hz controller failed'))
    messages = []
    with pytest.raises(OSError):
        with damping_exit(lambda: target, notify=messages.append, install_signals=False) as state:
            raise OSError('CAN down')
    assert not state.homed and 'Damping unavailable' in messages[-1]


def test_home_key_reader_waits_for_h():
    keys = iter(['x', 'H'])
    assert safe_exit.HomeKeyReader(read_key=lambda: next(keys)).home.wait(1)


def test_arx_damping_holds_the_gripper_in_place_then_reset_to_home_opens_it():
    events = []
    gain = SimpleNamespace(gripper_kp=0.0, gripper_kd=0.0)
    controller = SimpleNamespace(
        set_to_damping=lambda: events.append('damping'),
        get_controller_config=lambda: SimpleNamespace(default_gripper_kp=5.0, default_gripper_kd=0.2),
        get_gain=lambda: gain,
        set_gain=lambda value: events.append(('gain', value.gripper_kp, value.gripper_kd)),
        get_joint_state=lambda: events.append('read'),
        reset_to_home=lambda: events.append('home'))
    target = ARXDamping([controller, None], before_home=lambda: events.append('before home'))
    target.damp(); target.refresh(); target.go_home()
    # set_to_damping pins the current state; the gripper keeps its default gains.
    assert events == ['damping', ('gain', 5.0, 0.2), 'read', 'before home', 'home']

