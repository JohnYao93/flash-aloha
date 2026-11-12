"""Common capture/action boundary for recorders and policy execution.

Failures damp the arms instead of disabling them (see hardware.safe_exit);
RuntimeDamping is the matching damping target for safe_exit.damping_exit.
"""
import time

import numpy as np

from contracts.robot_contract import require_robot_match, normalize_robot_contract
from contracts.action_contract import require_absolute_actions
from contracts.action_rate_contract import resolve_policy_rate
from contracts import standard_capture
from hardware.arm_interface import positive_seconds


class HardwareRuntime:
    def __init__(self, arms, *, max_feedback_skew_s=0.02):
        if not arms or set(arms) - {'left', 'right'}:
            raise ValueError('Runtime requires named left and/or right arms')
        self.arms = {role: arms[role] for role in ('left', 'right') if role in arms}
        for role, arm in self.arms.items():
            if arm.describe().role != role:
                raise ValueError('Adapter role differs from its runtime assignment')
        self.contract = normalize_robot_contract({'robot_contract_version': 1,
            'arms': {role: {'model': arm.describe().model} for role, arm in self.arms.items()}})
        self.action_dim = sum(arm.describe().action_dim for arm in self.arms.values())
        self.max_skew = positive_seconds(max_feedback_skew_s, 'max_feedback_skew_s')
        self.interval_s = None
        self._connected = False

    def require_policy(self, metadata, *, stats=None):
        if self._connected:
            raise RuntimeError('Validate policy before connecting hardware')
        require_robot_match(self.contract, metadata, context='checkpoint')
        if stats is not None:
            require_robot_match(self.contract, stats, context='normalization statistics')
        require_absolute_actions(metadata)
        interval = 1 / resolve_policy_rate(metadata)
        for arm in self.arms.values():
            arm.validate_target(np.zeros(arm.describe().action_dim), interval_s=interval)
        for part in (metadata, metadata.get('policy_config', {}), metadata.get('task_config', {})):
            if part.get('action_dim') is not None and part['action_dim'] != self.action_dim:
                raise ValueError('Checkpoint action dimension differs from selected hardware')
        self.interval_s = interval

    def connect(self):
        try:
            for arm in self.arms.values():
                arm.connect()
            self._connected = True
        except BaseException:
            self.close()
            raise

    def read_state(self):
        if not self._connected:
            raise RuntimeError('Runtime is not connected')
        try:
            snapshots = [arm.read_state() for arm in self.arms.values()]
            times = np.concatenate([s.feedback_times for s in snapshots])
            if np.ptp(times) > self.max_skew:
                raise RuntimeError('Motor feedback timestamps are too skewed')
            return {'qpos': np.concatenate([s.qpos for s in snapshots]),
                    'qvel': np.concatenate([s.qvel for s in snapshots]),
                    'feedback_ages': np.concatenate([s.feedback_ages for s in snapshots])}
        except BaseException as exc:
            self._damp_all(exc)
            raise

    def capture(self, camera_names, *, include_images=False):
        return standard_capture.capture_observation(self.read_state, camera_names,
                                                    include_images=include_images)

    def submit_target(self, target, *, interval_s=None):
        if not self._connected:
            raise RuntimeError('Runtime is not connected')
        interval_s = self.interval_s if interval_s is None else interval_s
        if interval_s is None:
            raise ValueError('Declare an action interval or validate a policy first')
        target = np.array(target, dtype=float, copy=True)
        if target.shape != (self.action_dim,) or not np.isfinite(target).all():
            raise ValueError('Target must describe every selected arm')
        prepared, offset = [], 0
        # All vectors and limits pass before the first arm receives a command.
        for arm in self.arms.values():
            end = offset + arm.describe().action_dim
            prepared.append((arm, arm.validate_target(target[offset:end], interval_s=interval_s)))
            offset = end
        try:
            for arm, values in prepared:
                arm.submit_target(values, interval_s=interval_s)
        except BaseException as exc:
            self._damp_all(exc)
            raise

    def safe_stop(self, *, interval_s):
        """Damp every arm that can still be commanded; refresh each tick.

        An arm that cannot be damped is disabled by its adapter; this raises only
        when no arm is left under control.
        """
        if not self._connected:
            raise RuntimeError('Runtime is not connected')
        errors = []
        for arm in self.live_arms().values():
            try:
                arm.submit_safe_stop(interval_s=interval_s)
            except Exception as exc:
                errors.append(exc)
        if not self.live_arms():
            raise RuntimeError('No arm can be damped; all are disabled') from (errors[0] if errors else None)

    def live_arms(self):
        return {role: arm for role, arm in self.arms.items() if not getattr(arm, 'closed', False)}

    def _damp_all(self, reason):
        # Every live arm damps and rejects targets until acknowledge().
        for arm in self.live_arms().values():
            if not getattr(arm, 'faulted', False):
                arm.fail(str(reason) or type(reason).__name__)

    def acknowledge(self):
        """Operator accepted the failure: live arms accept targets again."""
        for arm in self.live_arms().values():
            arm.acknowledge()

    def recording_metadata(self, control_freq):
        return {'robot_contract': self.contract, 'control_freq': float(control_freq),
                'action_command_style': 'absolute',
                'sdk_controller_hz_by_role': {role: arm.describe().motor_hz for role, arm in self.arms.items()}}

    def stop(self):
        self.close()

    def close(self):
        errors = []
        for arm in self.arms.values():
            try:
                arm.close()
            except Exception as exc:
                errors.append(exc)
        self._connected = False
        if errors:
            raise RuntimeError('Failed to close one or more arms') from errors[0]


class RuntimeDamping:
    """safe_exit target for a HardwareRuntime: damp, refresh, then home each live arm.

    Arms use their SDK's native home move. Disabled arms are left disabled.
    """

    def __init__(self, runtime, *, interval_s=0.01):
        self.runtime, self.interval = runtime, float(interval_s)

    def damp(self):
        self.runtime.safe_stop(interval_s=self.interval)

    refresh = damp

    def go_home(self):
        self.runtime.acknowledge()
        for arm in self.runtime.live_arms().values():
            arm.home()
