"""Shared lifecycle, snapshots and timing; all action values remain backend native."""
from dataclasses import dataclass
from typing import Protocol
import math
import threading
import time

import numpy as np


@dataclass(frozen=True)
class ArmDescription:
    role: str
    model: str
    backend: str
    action_dim: int
    motor_hz: float


@dataclass(frozen=True)
class ArmState:
    qpos: np.ndarray
    qvel: np.ndarray
    feedback_ages: np.ndarray
    # Receipt/publication time, not a claim about motor sensor sampling time.
    feedback_times: np.ndarray
    faults: tuple = ()

    def __post_init__(self):
        size = None
        for name in ('qpos', 'qvel', 'feedback_ages', 'feedback_times'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not np.isfinite(value).all() or (size is not None and len(value) != size):
                raise ValueError('State fields must be equally sized finite vectors')
            size = len(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.feedback_ages < -0.001):
            raise ValueError('Feedback timestamps are in the future')
        object.__setattr__(self, 'faults', tuple(self.faults))


class Arm(Protocol):
    def describe(self) -> ArmDescription: ...
    def connect(self) -> None: ...
    def read_state(self) -> ArmState: ...
    def validate_target(self, target, *, interval_s: float) -> np.ndarray: ...
    def submit_target(self, target, *, interval_s: float) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def positive_seconds(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


class ManagedArm:
    """Serialized SDK ownership plus an independent command-expiry watchdog.

    Failures (feedback, commands, watchdog expiry) put the arm into damping;
    see hardware.safe_exit. Only an arm that cannot be damped is disabled.
    """
    def __init__(self, description, *, command_timeout_s=0.5, max_feedback_age_s=0.1,
                 clock=time.monotonic):
        self._description = description
        self._timeout = positive_seconds(command_timeout_s, 'command_timeout_s')
        self._max_age = positive_seconds(max_feedback_age_s, 'max_feedback_age_s')
        self._clock = clock
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._connected = self._closed = False
        self._deadline = None
        self._error = None
        self._worker = None

    def describe(self):
        return self._description

    def connect(self):
        with self._lock:
            if self._closed or self._connected:
                raise RuntimeError('Arm is already connected or closed')
            try:
                self._connect()
                self._connected = True
                # The watchdog starts on the first submitted command, after startup barriers.
                self._deadline = None
                self._worker = threading.Thread(target=self._watchdog, daemon=True,
                    name=f'{self._description.role}-{self._description.backend}-watchdog')
                self._worker.start()
            except BaseException:
                self._closed = True
                self._wake.set()
                self._close()
                raise

    @property
    def closed(self):
        return self._closed

    @property
    def faulted(self):
        """Failed and damping (not disabled); targets wait for acknowledge()."""
        return self._error is not None and not self._closed

    def _check(self, *, allow_fault=False):
        if self._closed or not self._connected:
            raise RuntimeError(self._error or 'Arm is not connected or has been closed')
        if self._error and not allow_fault:
            raise RuntimeError(self._error)
        if not self._error:
            self.check_command_expiry()

    def validate_target(self, target, *, interval_s):
        interval_s = positive_seconds(interval_s, 'interval_s')
        if interval_s < 1 / self._description.motor_hz:
            raise ValueError(f'Target interval is faster than the {self._description.motor_hz:g} Hz motor loop')
        if interval_s >= self._timeout:
            raise ValueError('Target interval must be shorter than command_timeout_s')
        value = np.array(target, dtype=float, copy=True)
        if value.shape != (self._description.action_dim,) or not np.isfinite(value).all():
            raise ValueError('Target must be a complete finite action vector')
        self._validate_limits(value)
        return value

    def read_state(self):
        with self._lock:
            self._check()
            try:
                result = self._read_state()
                if len(result.qpos) != self._description.action_dim:
                    raise RuntimeError('SDK state dimension differs from arm description')
                if result.faults or np.max(result.feedback_ages) > self._max_age:
                    raise RuntimeError('Motor feedback is stale or faulted')
                return result
            except BaseException as exc:
                self._fail(str(exc) or type(exc).__name__)
                raise

    def submit_target(self, target, *, interval_s):
        value = self.validate_target(target, interval_s=interval_s)
        with self._lock:
            self._check()
            try:
                self._submit_target(value, interval_s=float(interval_s))
                self._deadline = self._clock() + self._timeout
            except BaseException as exc:
                self._fail(str(exc) or type(exc).__name__)
                raise

    def submit_safe_stop(self, *, interval_s):
        """Damping: zero position gain, usual damping gains, backend gravity support.

        Allowed while faulted. Refresh it every tick; it renews the command
        watchdog. An arm that cannot be damped is disabled.
        """
        interval_s = positive_seconds(interval_s, 'interval_s')
        if interval_s < 1 / self._description.motor_hz or interval_s >= self._timeout:
            raise ValueError('Safe-stop interval must fit the motor loop and command timeout')
        with self._lock:
            self._check(allow_fault=True)
            try:
                self._submit_safe_stop(interval_s=interval_s)
                self._deadline = self._clock() + self._timeout
            except BaseException as exc:
                self._error = self._error or str(exc) or type(exc).__name__
                self._disable()
                raise

    def _submit_safe_stop(self, *, interval_s):
        raise NotImplementedError(f'{self._description.backend} has no safe-stop mode')

    def acknowledge(self):
        """Operator accepted the failure ('h'): allow targets again, e.g. to go home."""
        with self._lock:
            if self._closed or not self._connected:
                raise RuntimeError(self._error or 'Arm is not connected or has been closed')
            self._error = None
            self._deadline = None

    def open_gripper_target(self):
        """Fully open gripper position, or None when the backend does not know it."""
        return None

    def home(self):
        """Native SDK home move."""
        with self._lock:
            self._check()
            try:
                self._home()
            except BaseException as exc:
                self._fail(str(exc) or type(exc).__name__)
                raise

    def _home(self):
        raise NotImplementedError(f'{self._description.backend} has no native home move')

    def check_command_expiry(self):
        with self._lock:
            if self._closed or self._deadline is None or self._clock() < self._deadline:
                return
            if self._error:
                # Damping refresh stopped (e.g. Python stalled): hold damping natively.
                self._deadline = None
                self._damp_natively()
                return
            self._fail('Policy command expired')
            raise RuntimeError(self._error)

    def _watchdog(self):
        while not self._wake.wait(min(self._timeout / 4, 0.01)):
            try:
                self.check_command_expiry()
            except Exception:
                continue  # The arm is damping now; keep watching until it closes.

    def fail(self, message):
        """Externally detected failure (e.g. cross-arm skew): damp and await acknowledge()."""
        with self._lock:
            if not self._closed:
                self._fail(message)

    def _fail(self, message):
        """Every failure damps instead of disabling; only an undampable arm is disabled."""
        self._error = message
        self._deadline = None
        self._damp_natively()

    def _damp_natively(self):
        # The SDK loop keeps sending this command even if Python stops refreshing it.
        try:
            self._submit_safe_stop(interval_s=max(1 / self._description.motor_hz, 0.01))
        except BaseException:
            self._disable()

    def _disable(self):
        if not self._closed:
            self._closed = True
            self._wake.set()
            self._close()

    def stop(self):
        self.close()

    def close(self):
        with self._lock:
            self._disable()
        if self._worker is not None and threading.current_thread() is not self._worker:
            self._worker.join()

    def _validate_limits(self, target):
        pass
