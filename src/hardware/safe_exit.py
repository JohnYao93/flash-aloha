"""One failure exit for every path that moves a real arm.

Any catchable failure while an arm is under control -- an exception (including
joint-limit and target validation, camera, policy or SDK errors), Ctrl+C or
SIGTERM, or the command watchdog expiring -- enters **damping mode**: zero
position gain, the arm's usual damping gains toward zero velocity, gravity
compensation, grippers held. The arm stays there until the operator presses
'h', which moves it home with its normal gains. A failure while homing returns
to damping. Ctrl+C during damping disables the arm in place (hold it).

Intentional quits raise :class:`QuitRequested` and keep each tool's normal
exit. An arm whose own controller has already failed (native motor fault, CAN
failure, native stale feedback) cannot be commanded and stays disabled.

Each backend supplies a target with ``damp()``, ``refresh()`` and ``go_home()``.
"""
import contextlib
from contextlib import contextmanager
import signal
import threading
import time
from types import SimpleNamespace


class QuitRequested(KeyboardInterrupt):
    """An intentional operator quit: use the tool's normal exit, not damping."""


class HomeKeyReader:
    """Wait in the background for the operator's 'h' on the terminal."""

    def __init__(self, read_key=None):
        self.home = threading.Event()
        self._read = read_key
        threading.Thread(target=self._run, name='safe-exit-home-key', daemon=True).start()

    def _run(self):
        read = self._read
        if read is None:
            from getkey import getkey as read
        while True:
            try:
                key = read()
            except (EOFError, KeyboardInterrupt):
                return  # Ctrl+C also reaches the signal handler (cbreak keeps ISIG).
            if str(key).lower() == 'h':
                self.home.set()
                return


def _main_thread():
    return threading.current_thread() is threading.main_thread()


@contextmanager
def operator_stop_signals(stop):
    """SIGINT/SIGTERM set ``stop`` instead of raising mid-command; main thread only."""
    if not _main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, lambda signum, frame: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def raising_signals():
    """SIGINT/SIGTERM raise KeyboardInterrupt (used while a home move runs)."""
    if not _main_thread():
        yield
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def describe(reason):
    if isinstance(reason, KeyboardInterrupt) and not str(reason):
        return 'Ctrl+C/SIGTERM'
    return f'{type(reason).__name__}: {reason}' if isinstance(reason, BaseException) else str(reason)


def damp_until_home(target, reason, *, home_requested=None, notify=print, interval=0.01,
                    install_signals=True, stop=None, clock=time.monotonic, sleep=time.sleep):
    """Damping until the operator presses 'h', then home; repeat if homing fails.

    Raises KeyboardInterrupt when Ctrl+C/SIGTERM arrives during damping (the
    caller disables in place), and any error from ``damp``/``refresh`` when the
    arm can no longer be commanded.
    """
    stop = stop or threading.Event()  # Set by Ctrl+C/SIGTERM (or the caller) to disable.
    signals = operator_stop_signals(stop) if install_signals else contextlib.nullcontext()
    with signals:
        while True:
            target.damp()
            stop.clear()
            requested = home_requested or HomeKeyReader().home.is_set
            notify(f'[Safe exit] {describe(reason)} -> DAMPING (Kp=0, usual damping, gravity on). '
                   "Press 'h' to return home; Ctrl+C disables the arm in place.")
            while not requested():
                if stop.is_set():
                    raise KeyboardInterrupt('disable requested during damping')
                started = clock()
                target.refresh()
                remaining = interval - (clock() - started)
                if remaining > 0:
                    sleep(remaining)
            notify('[Safe exit] Returning home...')
            try:
                with raising_signals() if install_signals else contextlib.nullcontext():
                    target.go_home()
            except (Exception, KeyboardInterrupt) as exc:
                reason = exc
                continue
            notify('[Safe exit] Home reached')
            return


@contextmanager
def damping_exit(make_target, *, notify=print, **options):
    """Run a block; on any non-quit failure, damp until 'h', home, then re-raise.

    ``make_target()`` returns the damping target for whatever is connected at
    failure time, or None when nothing is under control. The yielded state
    reports ``failure`` and ``homed`` so the caller's cleanup skips its own home
    move and only disables.
    """
    state = SimpleNamespace(failure=None, homed=False, disabled_in_place=False)
    try:
        yield state
    except (QuitRequested, SystemExit, GeneratorExit):
        raise
    except BaseException as exc:
        state.failure = exc
        target = make_target()
        if target is None:
            raise
        try:
            damp_until_home(target, exc, notify=notify, **options)
            state.homed = True
        except KeyboardInterrupt:
            state.disabled_in_place = True
            notify('[Safe exit] Disabling the arm in place')
        except Exception as damp_error:
            notify(f'[Safe exit] Damping unavailable ({damp_error}); the controller has disabled the arm')
        raise


class ARXDamping:
    """ARX SDK controllers: native damping (default kd, native gravity) and home."""

    def __init__(self, controllers, *, before_home=None):
        self.controllers = [controller for controller in controllers if controller is not None]
        self._before_home = before_home

    def damp(self):
        for controller in self.controllers:
            hold_arx_gripper_in_damping(controller)

    def refresh(self):
        # Native gravity runs in the SDK; reading surfaces SDK faults.
        for controller in self.controllers:
            controller.get_joint_state()

    def go_home(self):
        if self._before_home is not None:
            self._before_home()
        # reset_to_home ramps from zero Kp to the default gains itself.
        for controller in self.controllers:
            controller.reset_to_home()


def hold_arx_gripper_in_damping(controller):
    """ARX set_to_damping, then hold the gripper where it is with its default gains.

    set_to_damping pins the interpolator at the current state but zeroes the
    gripper gains; restoring them keeps the gripper in place instead of slack.
    """
    controller.set_to_damping()
    config = controller.get_controller_config()
    gain = controller.get_gain()
    gain.gripper_kp = config.default_gripper_kp
    gain.gripper_kd = config.default_gripper_kd
    controller.set_gain(gain)
