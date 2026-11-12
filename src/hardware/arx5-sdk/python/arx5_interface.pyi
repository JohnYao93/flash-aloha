from typing import Tuple, overload
import numpy as np
import numpy.typing as npt
from enum import Enum

class MotorType:
    EC_A4310: "MotorType"
    DM_J4310: "MotorType"
    DM_J4340: "MotorType"
    NONE: "MotorType"

class RobotConfig:
    """Does not have a constructor, use RobotConfigFactory.get_instance().get_config(...) instead."""

    robot_model: str
    joint_pos_min: np.ndarray
    joint_pos_max: np.ndarray
    joint_vel_max: np.ndarray
    joint_torque_max: np.ndarray
    ee_vel_max: np.ndarray
    gripper_vel_max: float
    gripper_torque_max: float
    gripper_width: float
    gripper_open_readout: float
    joint_dof: int
    motor_id: list[int]
    motor_type: list[MotorType]
    joint_velocity_feedback_scale: np.ndarray
    gripper_motor_id: int
    gripper_motor_type: MotorType
    gripper_velocity_feedback_scale: float
    gravity_vector: np.ndarray
    base_link_name: str
    eef_link_name: str
    urdf_path: str

class ControllerConfig:
    """Does not have a constructor, use ControllerConfigFactory.get_instance().get_config(...) instead."""

    controller_type: str
    default_kp: np.ndarray
    default_kd: np.ndarray
    default_gripper_kp: float
    default_gripper_kd: float
    over_current_cnt_max: int
    controller_dt: float
    gravity_compensation: bool
    default_cartesian_kp: npt.NDArray[np.float64]
    default_cartesian_kd: npt.NDArray[np.float64]
    cartesian_wrench_limit: npt.NDArray[np.float64]
    cartesian_joint_velocity_filter_cutoff_hz: float
    cartesian_tracking_error_translation_limit_m: float
    cartesian_tracking_error_rotation_limit_rad: float
    cartesian_tracking_error_timeout_s: float
    cartesian_task_torque_ramp_time_s: float
    cartesian_joint_limit_margin_rad: float
    friction_compensation: bool
    joint_friction_compensation: bool
    joint_friction_transition_velocity: np.ndarray
    joint_friction_constant_term: np.ndarray
    joint_friction_coulomb_coef: np.ndarray
    joint_friction_viscous_coef: np.ndarray
    gripper_friction_compensation: bool
    gripper_friction_transition_velocity: float
    gripper_friction_constant_term: float
    gripper_friction_coulomb_coef: float
    gripper_friction_viscous_coef: float
    background_send_recv: bool
    shutdown_to_passive: bool
    interpolation_method: str
    default_preview_time: float

class RobotConfigFactory:
    @classmethod
    def get_instance(cls) -> RobotConfigFactory: ...
    def get_config(self, robot_model: str) -> RobotConfig: ...

class ControllerConfigFactory:
    @classmethod
    def get_instance(cls) -> ControllerConfigFactory: ...
    @overload
    def get_config(self, controller_type: str, joint_dof: int) -> ControllerConfig: ...
    @overload
    def get_config(self, controller_type: str, robot_config: RobotConfig) -> ControllerConfig: ...

class LogLevel:
    TRACE: "LogLevel"
    DEBUG: "LogLevel"
    INFO: "LogLevel"
    WARNING: "LogLevel"
    ERROR: "LogLevel"
    CRITICAL: "LogLevel"
    OFF: "LogLevel"

class CartesianDampingMode:
    HostCartesian: "CartesianDampingMode"
    MotorLocalJoint: "CartesianDampingMode"

class Gain:
    def kp(self) -> npt.NDArray[np.float64]: ...
    def kd(self) -> npt.NDArray[np.float64]: ...
    gripper_kp: float
    gripper_kd: float
    @overload
    def __init__(self, dof: int) -> None: ...
    @overload
    def __init__(
        self,
        kp: npt.NDArray[np.float64],
        kd: npt.NDArray[np.float64],
        gripper_kp: float,
        gripper_kd: float,
    ) -> None: ...
    def __add__(self, other: Gain) -> Gain: ...
    def __mul__(self, scalar: float) -> Gain: ...

class CartesianPDGain:
    def kp(self) -> npt.NDArray[np.float64]: ...
    def kd(self) -> npt.NDArray[np.float64]: ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        kp: npt.NDArray[np.float64],
        kd: npt.NDArray[np.float64],
    ) -> None: ...

class _ControlCycleTelemetry:
    sequence: int
    cartesian_pd_enabled: bool
    has_cartesian_data: bool
    cartesian_damping_mode: CartesianDampingMode
    cartesian_pseudoinverse_damping: float
    cartesian_joint_velocity_scale: float
    cartesian_joint_velocity_scaled: bool
    cycle_start_s: float
    control_compute_start_s: float
    control_compute_end_s: float
    feedback_used_timestamp_s: float
    motor_snapshot_read_start_s: float
    motor_snapshot_read_end_s: float
    feedback_publish_timestamp_s: float
    cycle_end_s: float
    joint_send_call_start_s: npt.NDArray[np.float64]
    joint_send_call_end_s: npt.NDArray[np.float64]
    control_joint_position: npt.NDArray[np.float64]
    control_joint_velocity: npt.NDArray[np.float64]
    control_joint_torque: npt.NDArray[np.float64]
    filtered_joint_velocity: npt.NDArray[np.float64]
    desired_pose: npt.NDArray[np.float64]
    desired_twist: npt.NDArray[np.float64]
    measured_pose: npt.NDArray[np.float64]
    pose_error: npt.NDArray[np.float64]
    raw_twist: npt.NDArray[np.float64]
    filtered_twist: npt.NDArray[np.float64]
    proportional_wrench: npt.NDArray[np.float64]
    derivative_wrench: npt.NDArray[np.float64]
    clamped_wrench: npt.NDArray[np.float64]
    requested_cartesian_task_torque: npt.NDArray[np.float64]
    slewed_cartesian_task_torque: npt.NDArray[np.float64]
    host_cartesian_stiffness_torque: npt.NDArray[np.float64]
    host_cartesian_damping_torque: npt.NDArray[np.float64]
    gravity_torque: npt.NDArray[np.float64]
    desired_joint_velocity_unlimited: npt.NDArray[np.float64]
    desired_joint_velocity_limited: npt.NDArray[np.float64]
    final_motor_kp: npt.NDArray[np.float64]
    final_motor_kd: npt.NDArray[np.float64]
    final_motor_protocol_velocity: npt.NDArray[np.float64]
    final_joint_position_command: npt.NDArray[np.float64]
    final_joint_velocity_command: npt.NDArray[np.float64]
    final_joint_torque_command: npt.NDArray[np.float64]
    feedback_joint_position: npt.NDArray[np.float64]
    feedback_joint_velocity: npt.NDArray[np.float64]
    feedback_joint_torque: npt.NDArray[np.float64]

class _CartesianKinematics:
    def __init__(
        self,
        urdf_path: str,
        joint_dof: int,
        base_link: str,
        eef_link: str,
    ) -> None: ...
    def jacobian(
        self, joint_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def pose(
        self, joint_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

def _cartesian_pose_error(
    desired_pose: npt.NDArray[np.float64],
    measured_pose: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

def _compute_cartesian_pd_torque(
    jacobian: npt.NDArray[np.float64],
    pose_error: npt.NDArray[np.float64],
    desired_twist: npt.NDArray[np.float64],
    actual_twist: npt.NDArray[np.float64],
    gain: CartesianPDGain,
    wrench_limit: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def _compute_host_joint_damping_torque(
    joint_velocity: npt.NDArray[np.float64],
    joint_damping: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

def _compute_damped_cartesian_joint_velocity(
    jacobian: npt.NDArray[np.float64],
    twist: npt.NDArray[np.float64],
    limits: npt.NDArray[np.float64],
    damping: float = 0.01,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, bool]: ...

class JointState:
    timestamp: float
    gripper_pos: float
    gripper_vel: float
    gripper_torque: float
    @overload
    def __init__(self, dof: int) -> None: ...
    @overload
    def __init__(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
        gripper_pos: float,
    ) -> None: ...
    @overload
    def __init__(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
        gripper_pos: float,
        gripper_vel: float,
        gripper_torque: float,
    ) -> None: ...
    def __add__(self, other: JointState) -> JointState: ...
    def __mul__(self, scalar: float) -> JointState: ...
    def pos(self) -> npt.NDArray[np.float64]: ...
    def vel(self) -> npt.NDArray[np.float64]: ...
    def torque(self) -> npt.NDArray[np.float64]: ...

class Arx5JointController:
    @overload
    def __init__(
        self,
        model: str,
        interface_name: str,
    ) -> None: ...
    @overload
    def __init__(
        self,
        robot_config: RobotConfig,
        controller_config: ControllerConfig,
        interface_name: str,
    ) -> None: ...
    def send_recv_once(self) -> None: ...
    def recv_once(self) -> None: ...
    def set_joint_cmd(self, cmd: JointState) -> None: ...
    def set_joint_traj(self, traj: list[JointState]) -> None: ...
    def _set_host_joint_damping(
        self, joint_damping: npt.NDArray[np.float64]
    ) -> None: ...
    def _get_host_joint_damping(self) -> npt.NDArray[np.float64]: ...
    def get_joint_cmd(self) -> JointState: ...
    def get_timestamp(self) -> float: ...
    def get_joint_state(self) -> JointState: ...
    def _get_control_cycle_telemetry(self) -> _ControlCycleTelemetry: ...
    def get_eef_state(self) -> EEFState: ...
    def get_home_pose(self) -> np.ndarray: ...
    def set_gain(self, gain: Gain) -> None: ...
    def get_gain(self) -> Gain: ...
    def get_robot_config(self) -> RobotConfig: ...
    def get_controller_config(self) -> ControllerConfig: ...
    def set_gravity_compensation(self, enabled: bool) -> None: ...
    def set_friction_compensation(self, enabled: bool) -> None: ...
    def set_joint_friction_compensation(
        self,
        enabled: bool,
        transition_velocity: npt.NDArray[np.float64],
        constant_term: npt.NDArray[np.float64],
        coulomb_coef: npt.NDArray[np.float64],
        viscous_coef: npt.NDArray[np.float64],
    ) -> None: ...
    def set_gripper_friction_compensation(
        self,
        enabled: bool,
        transition_velocity: float,
        constant_term: float,
        coulomb_coef: float,
        viscous_coef: float,
    ) -> None: ...
    def reset_to_home(self) -> None: ...
    def set_to_damping(self) -> None: ...
    def calibrate_gripper(self) -> None: ...
    def calibrate_joint(self, joint_id: int) -> None: ...
    def set_log_level(self, level: LogLevel) -> None: ...

class EEFState:
    timestamp: float
    gripper_pos: float
    gripper_vel: float
    gripper_torque: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, pose_6d: npt.NDArray[np.float64], gripper_pos: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        pose_6d: npt.NDArray[np.float64],
        gripper_pos: float,
        gripper_vel: float,
        gripper_torque: float,
    ) -> None: ...
    def __add__(self, other: EEFState) -> EEFState: ...
    def __mul__(self, scalar: float) -> EEFState: ...
    def pose_6d(self) -> npt.NDArray[np.float64]: ...

class Arx5CartesianController:
    @overload
    def __init__(self, model: str, interface_name: str) -> None: ...
    @overload
    def __init__(
        self,
        robot_config: RobotConfig,
        controller_config: ControllerConfig,
        interface_name: str,
    ) -> None: ...
    def set_eef_cmd(self, cmd: EEFState) -> None: ...
    def set_joint_cmd(self, cmd: JointState) -> None: ...
    def set_joint_traj(self, traj: list[JointState]) -> None: ...
    def set_eef_traj(self, traj: list[EEFState]) -> None: ...
    def set_eef_cmd_with_twist(
        self, cmd: EEFState, terminal_twist: npt.NDArray[np.float64]
    ) -> None: ...
    def set_eef_traj_with_twists(
        self,
        traj: list[EEFState],
        waypoint_twists: list[npt.NDArray[np.float64]],
    ) -> None: ...
    def get_joint_cmd(self) -> JointState: ...
    def get_eef_cmd(self) -> EEFState: ...
    def get_eef_cmd_twist(self) -> npt.NDArray[np.float64]: ...
    def get_eef_state(self) -> EEFState: ...
    def get_joint_state(self) -> JointState: ...
    def _get_control_cycle_telemetry(self) -> _ControlCycleTelemetry: ...
    def get_timestamp(self) -> float: ...
    def set_gain(self, gain: Gain) -> None: ...
    def get_gain(self) -> Gain: ...
    def get_home_pose(self) -> np.ndarray: ...
    def set_log_level(self, level: LogLevel) -> None: ...
    def get_robot_config(self) -> RobotConfig: ...
    def get_controller_config(self) -> ControllerConfig: ...
    def set_cartesian_pd_gain(self, gain: CartesianPDGain) -> None: ...
    def get_cartesian_pd_gain(self) -> CartesianPDGain: ...
    def set_cartesian_damping_mode(self, mode: CartesianDampingMode) -> None: ...
    def get_cartesian_damping_mode(self) -> CartesianDampingMode: ...
    def set_cartesian_motor_kd(self, motor_kd: npt.NDArray[np.float64]) -> None: ...
    def get_cartesian_motor_kd(self) -> npt.NDArray[np.float64]: ...
    def set_cartesian_joint_velocity_filter_cutoff(self, cutoff_hz: float) -> None: ...
    def get_cartesian_joint_velocity_filter_cutoff(self) -> float: ...
    def get_filtered_joint_velocity(self) -> npt.NDArray[np.float64]: ...
    def enable_cartesian_pd(self) -> None: ...
    def disable_cartesian_pd(self) -> None: ...
    def is_cartesian_pd_enabled(self) -> bool: ...
    def set_gravity_compensation(self, enabled: bool) -> None: ...
    def set_friction_compensation(self, enabled: bool) -> None: ...
    def set_joint_friction_compensation(
        self,
        enabled: bool,
        transition_velocity: npt.NDArray[np.float64],
        constant_term: npt.NDArray[np.float64],
        coulomb_coef: npt.NDArray[np.float64],
        viscous_coef: npt.NDArray[np.float64],
    ) -> None: ...
    def set_gripper_friction_compensation(
        self,
        enabled: bool,
        transition_velocity: float,
        constant_term: float,
        coulomb_coef: float,
        viscous_coef: float,
    ) -> None: ...
    def reset_to_home(self) -> None: ...
    def set_to_damping(self) -> None: ...

class Arx5Solver:
    @overload
    def __init__(
        self,
        urdf_path: str,
        joint_dof: int,
        joint_pos_min: npt.NDArray[np.float64],
        joint_pos_max: npt.NDArray[np.float64],
    ) -> None: ...
    @overload
    def __init__(
        self,
        urdf_path: str,
        joint_dof: int,
        joint_pos_min: npt.NDArray[np.float64],
        joint_pos_max: npt.NDArray[np.float64],
        base_link: str,
        eef_link: str,
        gravity_vector: npt.NDArray[np.float64],
    ) -> None: ...
    def inverse_dynamics(
        self,
        joint_pos: npt.NDArray[np.float64],
        joint_vel: npt.NDArray[np.float64],
        joint_acc: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def inverse_kinematics(
        self,
        target_pose_6d: npt.NDArray[np.float64],
        current_joint_pos: npt.NDArray[np.float64],
    ) -> Tuple[int, npt.NDArray[np.float64]]: ...
    def multi_trial_ik(
        self,
        target_pose_6d: npt.NDArray[np.float64],
        current_joint_pos: npt.NDArray[np.float64],
        additional_trial_num: int,
    ) -> Tuple[int, npt.NDArray[np.float64]]: ...
    def get_ik_status_name(self, status: int) -> str: ...
    def forward_kinematics(
        self, joint_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
