"""Shared ARX gains, controller timing, target submission, and model naming.

Importing this module does not load the robot SDK or construct controllers.
Callers supply their existing controller, command, and configuration objects.
"""
import math

import numpy as np

ACTIVE_KP = (200.0, 200.0, 200.0, 100.0, 100.0, 100.0, 10.0)
ACTIVE_KD = (7.0, 7.0, 7.0, 1.5, 1.5, 1.5, 1.0)


def configure_active_gains(controller_config):
    """Set the shared defaults before constructing an L5/Trossen_L5 controller."""
    if np.asarray(controller_config.default_kp).size != 6:
        raise ValueError('Active gains require a six-joint L5 arm plus gripper')
    controller_config.default_kp = np.asarray(ACTIVE_KP[:6], dtype=np.float64)
    controller_config.default_kd = np.asarray(ACTIVE_KD[:6], dtype=np.float64)
    controller_config.default_gripper_kp = ACTIVE_KP[6]
    controller_config.default_gripper_kd = ACTIVE_KD[6]


configure_collection_gains = configure_active_gains


DEFAULT_ARM_KP = ACTIVE_KP
DEFAULT_ARM_KD = ACTIVE_KD


ARX_CONTROLLER_HZ = 500.0
ARX_CONTROLLER_DT = 1.0 / ARX_CONTROLLER_HZ
ARX_COMMAND_HZ = 100.0  # Legacy positioning-command cadence, not the SDK loop.


def configure_controller(config):
    config.controller_dt = ARX_CONTROLLER_DT
    config.interpolation_method = 'linear'
    # Keep immediate holds/cancellation and native homing semantics available.
    config.default_preview_time = 0.0
    return config


def require_controller_hz(hz):
    if not math.isfinite(float(hz)) or float(hz) != ARX_CONTROLLER_HZ:
        raise ValueError('ARX SDK control is fixed at 500 Hz; set recording/action FPS separately')


def send_joint_target(controller, command, interval_s):
    """Replace a streaming segment through native SDK interpolation.

    The interval is the intended time to the target, never the motor-loop
    period. Explicit irregular intervals are accepted. Restamp reused commands
    on every send; interpolation starts from the current SDK reference.
    """
    interval_s = float(interval_s)
    if not math.isfinite(interval_s) or interval_s < ARX_CONTROLLER_DT:
        raise ValueError('Target interval must be finite and at least 0.002 seconds')
    command.timestamp = controller.get_timestamp() + interval_s
    controller.set_joint_cmd(command)


def replay_timestep(fps):
    """Fixed action interval for physical replay; saved timestamps are diagnostic."""
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0 or fps > ARX_CONTROLLER_HZ:
        raise ValueError('Replay FPS must be finite, positive and at most 500 Hz')
    return 1.0 / fps


def resolve_arx_model(model: str) -> str:
    """Normalize Trossen spelling while preserving its distinct SDK profile.

    Other SDK model names pass through unchanged, including their validation
    by RobotConfigFactory. This does not select or remap a CAN interface.
    """
    return "Trossen_L5" if model.casefold() == "trossen_l5" else model
