"""ARX adapter: retain native SDK coordinates, gains and 500 Hz interpolation."""
import importlib
from pathlib import Path
import sys

import numpy as np

from hardware.arm_interface import ArmDescription, ArmState, ManagedArm
from utils.hardware_utils import configure_controller, configure_active_gains, resolve_arx_model


class ARXAdapter(ManagedArm):
    def __init__(self, role, model, can_interface, *, sdk=None, **options):
        model = resolve_arx_model(model)
        if model not in ('L5', 'Trossen_L5'):
            raise ValueError(f'Unsupported ARX robot model {model!r}')
        self._sdk = sdk
        self._can = can_interface
        self._controller = None
        self._robot_config = None
        self._active_gain = None
        self._activated = False
        self._damping = False
        super().__init__(ArmDescription(role, model, 'arx', 7, 500.0), **options)

    def _connect(self):
        sdk = self._sdk
        if sdk is None:
            directory = Path(__file__).resolve().parents[1] / 'arx5-sdk' / 'python'
            sys.path.insert(0, str(directory))
            sdk = self._sdk = importlib.import_module('arx5_interface')
        robot = sdk.RobotConfigFactory.get_instance().get_config(self.describe().model)
        config = sdk.ControllerConfigFactory.get_instance().get_config('joint_controller', robot)
        configure_active_gains(config)
        configure_controller(config)
        config.gravity_compensation = True
        # Friction feedforward is for hand guiding only; it stays off under policy control.
        config.friction_compensation = False
        config.background_send_recv = True
        self._robot_config = robot
        self._controller = sdk.Arx5JointController(robot, config, self._can)
        # Anchor the measured pose before enabling active gains, without homing.
        measured = self._controller.get_joint_state()
        command = sdk.JointState(robot.joint_dof)
        command.pos()[:] = measured.pos()
        command.gripper_pos = measured.gripper_pos
        self._controller.set_joint_cmd(command)
        gain = sdk.Gain(robot.joint_dof)
        gain.kp()[:] = config.default_kp
        gain.kd()[:] = config.default_kd
        gain.gripper_kp = config.default_gripper_kp
        gain.gripper_kd = config.default_gripper_kd
        self._active_gain = gain
        self._controller.set_to_damping()

    def _validate_limits(self, target):
        robot = self._robot_config
        if robot is not None and (np.any(target[:6] < robot.joint_pos_min) or
                np.any(target[:6] > robot.joint_pos_max) or not 0 <= target[6] <= robot.gripper_width):
            raise ValueError('ARX target exceeds native position limits')

    def _read_state(self):
        state = self._controller.get_joint_state()
        now = self._controller.get_timestamp()
        # SDK publication time only; its CAN library does not expose per-frame ages.
        age = now - state.timestamp
        host_time = self._clock() - age
        return ArmState(np.r_[state.pos(), state.gripper_pos], np.r_[state.vel(), state.gripper_vel],
                        np.full(7, age), np.full(7, host_time))

    def _submit_target(self, target, *, interval_s):
        if not self._activated:
            # Startup can take seconds. Anchor a fresh measured pose at activation.
            measured = self._controller.get_joint_state()
            seed = self._sdk.JointState(6)
            seed.pos()[:] = measured.pos()
            seed.gripper_pos = measured.gripper_pos
            self._controller.set_joint_cmd(seed)
            self._controller.set_gain(self._active_gain)
            self._activated = True
            self._damping = False
        command = self._sdk.JointState(6)
        command.pos()[:] = target[:6]
        command.gripper_pos = float(target[6])
        command.timestamp = self._controller.get_timestamp() + interval_s
        self._controller.set_joint_cmd(command)

    def _submit_safe_stop(self, *, interval_s):
        # Native damping: Kp=0, the controller's default Kd, native gravity.
        # The SDK holds it without refresh; later refreshes only surface faults.
        if not self._damping:
            from hardware.safe_exit import hold_arx_gripper_in_damping
            hold_arx_gripper_in_damping(self._controller)  # gripper held in place
            self._damping = True
            self._activated = False  # Re-anchor and restore gains on the next target.
        else:
            self._controller.get_joint_state()

    def open_gripper_target(self):
        return None if self._robot_config is None else float(self._robot_config.gripper_width)

    def _home(self):
        # reset_to_home ramps from zero Kp to the default gains itself.
        self._controller.reset_to_home()
        self._damping = False
        self._activated = False

    def _close(self):
        if self._controller is not None:
            controller, self._controller = self._controller, None
            try:
                controller.set_to_damping()
            finally:
                # Releasing the sole SDK owner invokes its thread-joining destructor.
                del controller
