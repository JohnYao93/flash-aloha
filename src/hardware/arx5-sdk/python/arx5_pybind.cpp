#include "app/cartesian_controller.h"
#include "app/common.h"
#include "app/config.h"
#include "app/controller_base.h"
#include "app/joint_controller.h"
#include "hardware/arx_can.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace arx;
using Pose6d = Eigen::Matrix<double, 6, 1>;
using VecDoF = Eigen::VectorXd;

namespace
{
void validate_positive_config_value(double value, const char *name)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and positive.");
}
} // namespace

PYBIND11_MODULE(arx5_interface, m)
{
    py::enum_<spdlog::level::level_enum>(m, "LogLevel", py::module_local())
        .value("TRACE", spdlog::level::level_enum::trace)
        .value("DEBUG", spdlog::level::level_enum::debug)
        .value("INFO", spdlog::level::level_enum::info)
        .value("WARNING", spdlog::level::level_enum::warn)
        .value("ERROR", spdlog::level::level_enum::err)
        .value("CRITICAL", spdlog::level::level_enum::critical)
        .value("OFF", spdlog::level::level_enum::off)
        .export_values();
    py::enum_<CartesianDampingMode>(m, "CartesianDampingMode")
        .value("HostCartesian", CartesianDampingMode::HostCartesian)
        .value("MotorLocalJoint", CartesianDampingMode::MotorLocalJoint);
    py::class_<JointState>(m, "JointState")
        .def(py::init<int>())
        .def(py::init<VecDoF, VecDoF, VecDoF, double>())
        .def(py::init<VecDoF, VecDoF, VecDoF, double, double, double>())
        .def_readwrite("timestamp", &JointState::timestamp)
        .def_readwrite("gripper_pos", &JointState::gripper_pos)
        .def_readwrite("gripper_vel", &JointState::gripper_vel)
        .def_readwrite("gripper_torque", &JointState::gripper_torque)
        .def("__add__", [](const JointState &self, const JointState &other) { return self + other; })
        .def("__mul__", [](const JointState &self, const float &scalar) { return self * scalar; })
        .def("pos", &JointState::get_pos_ref, py::return_value_policy::reference)
        .def("vel", &JointState::get_vel_ref, py::return_value_policy::reference)
        .def("torque", &JointState::get_torque_ref, py::return_value_policy::reference);
    py::class_<EEFState>(m, "EEFState")
        .def(py::init<>())
        .def(py::init<Pose6d, double>())
        .def(py::init<Pose6d, double, double, double>())
        .def_readwrite("timestamp", &EEFState::timestamp)
        .def_readwrite("gripper_pos", &EEFState::gripper_pos)
        .def_readwrite("gripper_vel", &EEFState::gripper_vel)
        .def_readwrite("gripper_torque", &EEFState::gripper_torque)
        .def("__add__", [](const EEFState &self, const EEFState &other) { return self + other; })
        .def("__mul__", [](const EEFState &self, const float &scalar) { return self * scalar; })
        .def("pose_6d", &EEFState::get_pose_6d_ref, py::return_value_policy::reference);
    py::class_<Gain>(m, "Gain")
        .def(py::init<int>())
        .def(py::init<VecDoF, VecDoF, double, double>())
        .def_readwrite("gripper_kp", &Gain::gripper_kp)
        .def_readwrite("gripper_kd", &Gain::gripper_kd)
        .def("__add__", [](const Gain &self, const Gain &other) { return self + other; })
        .def("__mul__", [](const Gain &self, const float &scalar) { return self * scalar; })
        .def("kp", &Gain::get_kp_ref, py::return_value_policy::reference)
        .def("kd", &Gain::get_kd_ref, py::return_value_policy::reference);
    py::class_<CartesianPDGain>(m, "CartesianPDGain")
        .def(py::init<>())
        .def(py::init<Pose6d, Pose6d>())
        .def("kp", &CartesianPDGain::get_kp_ref, py::return_value_policy::reference)
        .def("kd", &CartesianPDGain::get_kd_ref, py::return_value_policy::reference);
    py::class_<ControlCycleTelemetry>(m, "_ControlCycleTelemetry")
        .def_readonly("sequence", &ControlCycleTelemetry::sequence)
        .def_readonly("cartesian_pd_enabled", &ControlCycleTelemetry::cartesian_pd_enabled)
        .def_readonly("has_cartesian_data", &ControlCycleTelemetry::has_cartesian_data)
        .def_readonly("cartesian_damping_mode", &ControlCycleTelemetry::cartesian_damping_mode)
        .def_readonly("cartesian_pseudoinverse_damping", &ControlCycleTelemetry::cartesian_pseudoinverse_damping)
        .def_readonly("cartesian_joint_velocity_scale", &ControlCycleTelemetry::cartesian_joint_velocity_scale)
        .def_readonly("cartesian_joint_velocity_scaled", &ControlCycleTelemetry::cartesian_joint_velocity_scaled)
        .def_readonly("cycle_start_s", &ControlCycleTelemetry::cycle_start_s)
        .def_readonly("control_compute_start_s", &ControlCycleTelemetry::control_compute_start_s)
        .def_readonly("control_compute_end_s", &ControlCycleTelemetry::control_compute_end_s)
        .def_readonly("feedback_used_timestamp_s", &ControlCycleTelemetry::feedback_used_timestamp_s)
        .def_readonly("motor_snapshot_read_start_s", &ControlCycleTelemetry::motor_snapshot_read_start_s)
        .def_readonly("motor_snapshot_read_end_s", &ControlCycleTelemetry::motor_snapshot_read_end_s)
        .def_readonly("feedback_publish_timestamp_s", &ControlCycleTelemetry::feedback_publish_timestamp_s)
        .def_readonly("cycle_end_s", &ControlCycleTelemetry::cycle_end_s)
        .def_readonly("joint_send_call_start_s", &ControlCycleTelemetry::joint_send_call_start_s)
        .def_readonly("joint_send_call_end_s", &ControlCycleTelemetry::joint_send_call_end_s)
        .def_readonly("control_joint_position", &ControlCycleTelemetry::control_joint_position)
        .def_readonly("control_joint_velocity", &ControlCycleTelemetry::control_joint_velocity)
        .def_readonly("control_joint_torque", &ControlCycleTelemetry::control_joint_torque)
        .def_readonly("filtered_joint_velocity", &ControlCycleTelemetry::filtered_joint_velocity)
        .def_readonly("desired_pose", &ControlCycleTelemetry::desired_pose)
        .def_readonly("desired_twist", &ControlCycleTelemetry::desired_twist)
        .def_readonly("measured_pose", &ControlCycleTelemetry::measured_pose)
        .def_readonly("pose_error", &ControlCycleTelemetry::pose_error)
        .def_readonly("raw_twist", &ControlCycleTelemetry::raw_twist)
        .def_readonly("filtered_twist", &ControlCycleTelemetry::filtered_twist)
        .def_readonly("proportional_wrench", &ControlCycleTelemetry::proportional_wrench)
        .def_readonly("derivative_wrench", &ControlCycleTelemetry::derivative_wrench)
        .def_readonly("clamped_wrench", &ControlCycleTelemetry::clamped_wrench)
        .def_readonly("requested_cartesian_task_torque", &ControlCycleTelemetry::requested_cartesian_task_torque)
        .def_readonly("slewed_cartesian_task_torque", &ControlCycleTelemetry::slewed_cartesian_task_torque)
        .def_readonly("host_cartesian_stiffness_torque", &ControlCycleTelemetry::host_cartesian_stiffness_torque)
        .def_readonly("host_cartesian_damping_torque", &ControlCycleTelemetry::host_cartesian_damping_torque)
        .def_readonly("gravity_torque", &ControlCycleTelemetry::gravity_torque)
        .def_readonly("desired_joint_velocity_unlimited", &ControlCycleTelemetry::desired_joint_velocity_unlimited)
        .def_readonly("desired_joint_velocity_limited", &ControlCycleTelemetry::desired_joint_velocity_limited)
        .def_readonly("final_motor_kp", &ControlCycleTelemetry::final_motor_kp)
        .def_readonly("final_motor_kd", &ControlCycleTelemetry::final_motor_kd)
        .def_readonly("final_motor_protocol_velocity", &ControlCycleTelemetry::final_motor_protocol_velocity)
        .def_readonly("final_joint_position_command", &ControlCycleTelemetry::final_joint_position_command)
        .def_readonly("final_joint_velocity_command", &ControlCycleTelemetry::final_joint_velocity_command)
        .def_readonly("final_joint_torque_command", &ControlCycleTelemetry::final_joint_torque_command)
        .def_readonly("feedback_joint_position", &ControlCycleTelemetry::feedback_joint_position)
        .def_readonly("feedback_joint_velocity", &ControlCycleTelemetry::feedback_joint_velocity)
        .def_readonly("feedback_joint_torque", &ControlCycleTelemetry::feedback_joint_torque);
    py::class_<CartesianKinematics>(m, "_CartesianKinematics")
        .def(py::init<const std::string &, int, const std::string &, const std::string &>())
        .def("jacobian", &CartesianKinematics::jacobian)
        .def("pose", &CartesianKinematics::pose);
    m.def("_cartesian_pose_error",
          py::overload_cast<const Pose6d &, const Pose6d &>(&cartesian_pose_error));
    m.def("_compute_cartesian_pd_torque",
          [](const Eigen::MatrixXd &jacobian, const Pose6d &pose_error, const Pose6d &desired_twist,
             const Pose6d &actual_twist, const CartesianPDGain &gain, const Pose6d &wrench_limit) {
              Pose6d wrench;
              VecDoF torque = compute_cartesian_pd_torque(jacobian, pose_error, desired_twist, actual_twist, gain,
                                                          wrench_limit, &wrench);
              return py::make_tuple(torque, wrench);
          });
    m.def("_compute_host_joint_damping_torque", &compute_host_joint_damping_torque);
    m.def("_compute_damped_cartesian_joint_velocity",
          [](const Eigen::MatrixXd &jacobian, const Pose6d &twist, const VecDoF &limits, double damping) {
              const auto solution = compute_damped_cartesian_joint_velocity(jacobian, twist, limits, damping);
              return py::make_tuple(solution.unlimited, solution.limited, solution.scale, solution.valid);
          }, py::arg("jacobian"), py::arg("twist"), py::arg("limits"), py::arg("damping") = 0.01);
    py::class_<Arx5JointController>(m, "Arx5JointController")
        .def(py::init<const std::string &, const std::string &>())
        .def(py::init<RobotConfig, ControllerConfig, const std::string &>())
        .def("send_recv_once", &Arx5JointController::send_recv_once)
        .def("recv_once", &Arx5JointController::recv_once)
        .def("get_joint_state", &Arx5JointController::get_joint_state)
        .def("_get_control_cycle_telemetry", &Arx5JointController::get_control_cycle_telemetry)
        .def("get_timestamp", &Arx5JointController::get_timestamp)
        .def("set_joint_cmd", &Arx5JointController::set_joint_cmd)
        .def("set_joint_traj", &Arx5JointController::set_joint_traj)
        .def("_set_host_joint_damping", &Arx5JointController::set_host_joint_damping)
        .def("_get_host_joint_damping", &Arx5JointController::get_host_joint_damping)
        .def("get_home_pose", &Arx5JointController::get_home_pose)
        .def("get_eef_state", &Arx5JointController::get_eef_state)
        .def("get_joint_cmd", &Arx5JointController::get_joint_cmd)
        .def("set_gain", &Arx5JointController::set_gain)
        .def("get_gain", &Arx5JointController::get_gain)
        .def("get_robot_config", &Arx5JointController::get_robot_config)
        .def("get_controller_config", &Arx5JointController::get_controller_config)
        .def("reset_to_home", &Arx5JointController::reset_to_home)
        .def("set_to_damping", &Arx5JointController::set_to_damping)
        .def("set_log_level", &Arx5JointController::set_log_level)
        .def("set_gravity_compensation", &Arx5JointController::set_gravity_compensation)
        .def("set_friction_compensation", &Arx5JointController::set_friction_compensation)
        .def("set_joint_friction_compensation", &Arx5JointController::set_joint_friction_compensation)
        .def("set_gripper_friction_compensation", &Arx5JointController::set_gripper_friction_compensation)
        .def("calibrate_joint", &Arx5JointController::calibrate_joint)
        .def("calibrate_gripper", &Arx5JointController::calibrate_gripper);
    py::class_<Arx5CartesianController>(m, "Arx5CartesianController")
        .def(py::init<const std::string &, const std::string &>())
        .def(py::init<RobotConfig, ControllerConfig, const std::string &>())
        .def("set_eef_cmd", &Arx5CartesianController::set_eef_cmd)
        .def("set_joint_cmd", &Arx5CartesianController::set_joint_cmd)
        .def("set_joint_traj", &Arx5CartesianController::set_joint_traj)
        .def("set_eef_traj", &Arx5CartesianController::set_eef_traj)
        .def("set_eef_cmd_with_twist", &Arx5CartesianController::set_eef_cmd_with_twist)
        .def("set_eef_traj_with_twists", &Arx5CartesianController::set_eef_traj_with_twists)
        .def("get_joint_cmd", &Arx5CartesianController::get_joint_cmd)
        .def("get_eef_cmd", &Arx5CartesianController::get_eef_cmd)
        .def("get_eef_cmd_twist", &Arx5CartesianController::get_eef_cmd_twist)
        .def("get_eef_state", &Arx5CartesianController::get_eef_state)
        .def("get_joint_state", &Arx5CartesianController::get_joint_state)
        .def("_get_control_cycle_telemetry", &Arx5CartesianController::get_control_cycle_telemetry)
        .def("get_timestamp", &Arx5CartesianController::get_timestamp)
        .def("get_home_pose", &Arx5CartesianController::get_home_pose)
        .def("set_gain", &Arx5CartesianController::set_gain)
        .def("get_gain", &Arx5CartesianController::get_gain)
        .def("set_log_level", &Arx5CartesianController::set_log_level)
        .def("get_robot_config", &Arx5CartesianController::get_robot_config)
        .def("get_controller_config", &Arx5CartesianController::get_controller_config)
        .def("set_cartesian_pd_gain", &Arx5CartesianController::set_cartesian_pd_gain)
        .def("get_cartesian_pd_gain", &Arx5CartesianController::get_cartesian_pd_gain)
        .def("set_cartesian_damping_mode", &Arx5CartesianController::set_cartesian_damping_mode)
        .def("get_cartesian_damping_mode", &Arx5CartesianController::get_cartesian_damping_mode)
        .def("set_cartesian_motor_kd", &Arx5CartesianController::set_cartesian_motor_kd)
        .def("get_cartesian_motor_kd", &Arx5CartesianController::get_cartesian_motor_kd)
        .def("set_cartesian_joint_velocity_filter_cutoff",
             &Arx5CartesianController::set_cartesian_joint_velocity_filter_cutoff)
        .def("get_cartesian_joint_velocity_filter_cutoff",
             &Arx5CartesianController::get_cartesian_joint_velocity_filter_cutoff)
        .def("get_filtered_joint_velocity", &Arx5CartesianController::get_filtered_joint_velocity)
        .def("enable_cartesian_pd", &Arx5CartesianController::enable_cartesian_pd)
        .def("disable_cartesian_pd", &Arx5CartesianController::disable_cartesian_pd)
        .def("is_cartesian_pd_enabled", &Arx5CartesianController::is_cartesian_pd_enabled)
        .def("set_gravity_compensation", &Arx5CartesianController::set_gravity_compensation)
        .def("set_friction_compensation", &Arx5CartesianController::set_friction_compensation)
        .def("set_joint_friction_compensation", &Arx5CartesianController::set_joint_friction_compensation)
        .def("set_gripper_friction_compensation", &Arx5CartesianController::set_gripper_friction_compensation)
        .def("reset_to_home", &Arx5CartesianController::reset_to_home)
        .def("multi_trial_ik", &Arx5CartesianController::multi_trial_ik)
        .def("set_to_damping", &Arx5CartesianController::set_to_damping);
    py::class_<Arx5Solver>(m, "Arx5Solver")
        .def(py::init<const std::string &, int, Eigen::VectorXd, Eigen::VectorXd>())
        .def(py::init<const std::string &, int, Eigen::VectorXd, Eigen::VectorXd, const std::string &,
                      const std::string &, Eigen::Vector3d>())
        .def("inverse_dynamics", &Arx5Solver::inverse_dynamics)
        .def("forward_kinematics", &Arx5Solver::forward_kinematics)
        .def("inverse_kinematics", &Arx5Solver::inverse_kinematics)
        .def("get_ik_status_name", &Arx5Solver::get_ik_status_name)
        .def("multi_trial_ik", &Arx5Solver::multi_trial_ik);
    py::class_<RobotConfig>(m, "RobotConfig")
        .def_readwrite("robot_model", &RobotConfig::robot_model)
        .def_readwrite("joint_pos_min", &RobotConfig::joint_pos_min)
        .def_readwrite("joint_pos_max", &RobotConfig::joint_pos_max)
        .def_readwrite("joint_vel_max", &RobotConfig::joint_vel_max)
        .def_readwrite("joint_torque_max", &RobotConfig::joint_torque_max)
        .def_readwrite("ee_vel_max", &RobotConfig::ee_vel_max)
        .def_readwrite("gripper_vel_max", &RobotConfig::gripper_vel_max)
        .def_readwrite("gripper_torque_max", &RobotConfig::gripper_torque_max)
        .def_readwrite("gripper_width", &RobotConfig::gripper_width)
        .def_readwrite("gripper_open_readout", &RobotConfig::gripper_open_readout)
        .def_readwrite("joint_dof", &RobotConfig::joint_dof)
        .def_readwrite("motor_id", &RobotConfig::motor_id)
        .def_readwrite("motor_type", &RobotConfig::motor_type)
        .def_readwrite("joint_velocity_feedback_scale", &RobotConfig::joint_velocity_feedback_scale)
        .def_readwrite("gripper_motor_id", &RobotConfig::gripper_motor_id)
        .def_readwrite("gripper_motor_type", &RobotConfig::gripper_motor_type)
        .def_readwrite("gripper_velocity_feedback_scale", &RobotConfig::gripper_velocity_feedback_scale)
        .def_readwrite("gravity_vector", &RobotConfig::gravity_vector)
        .def_readwrite("base_link_name", &RobotConfig::base_link_name)
        .def_readwrite("eef_link_name", &RobotConfig::eef_link_name)
        .def_readwrite("urdf_path", &RobotConfig::urdf_path);

    py::class_<ControllerConfig>(m, "ControllerConfig")
        .def_readwrite("controller_type", &ControllerConfig::controller_type)
        .def_readwrite("default_kp", &ControllerConfig::default_kp)
        .def_readwrite("default_kd", &ControllerConfig::default_kd)
        .def_readwrite("default_gripper_kp", &ControllerConfig::default_gripper_kp)
        .def_readwrite("default_gripper_kd", &ControllerConfig::default_gripper_kd)
        .def_readwrite("over_current_cnt_max", &ControllerConfig::over_current_cnt_max)
        .def_readwrite("background_send_recv", &ControllerConfig::background_send_recv)
        .def_readwrite("gravity_compensation", &ControllerConfig::gravity_compensation)
        .def_property(
            "default_cartesian_kp", [](const ControllerConfig &config) { return config.default_cartesian_kp; },
            [](ControllerConfig &config, const Pose6d &value) {
                validate_cartesian_vector(value, "default_cartesian_kp");
                config.default_cartesian_kp = value;
            })
        .def_property(
            "default_cartesian_kd", [](const ControllerConfig &config) { return config.default_cartesian_kd; },
            [](ControllerConfig &config, const Pose6d &value) {
                validate_cartesian_vector(value, "default_cartesian_kd");
                config.default_cartesian_kd = value;
            })
        .def_property(
            "cartesian_wrench_limit", [](const ControllerConfig &config) { return config.cartesian_wrench_limit; },
            [](ControllerConfig &config, const Pose6d &value) {
                validate_cartesian_vector(value, "cartesian_wrench_limit");
                config.cartesian_wrench_limit = value;
            })
        .def_property(
            "cartesian_joint_velocity_filter_cutoff_hz",
            [](const ControllerConfig &config) { return config.cartesian_joint_velocity_filter_cutoff_hz; },
            [](ControllerConfig &config, double value) {
                validate_cartesian_joint_velocity_filter_cutoff(value, config.controller_dt);
                config.cartesian_joint_velocity_filter_cutoff_hz = value;
            })
        .def_property(
            "cartesian_tracking_error_translation_limit_m",
            [](const ControllerConfig &config) { return config.cartesian_tracking_error_translation_limit_m; },
            [](ControllerConfig &config, double value) {
                validate_positive_config_value(value, "cartesian_tracking_error_translation_limit_m");
                config.cartesian_tracking_error_translation_limit_m = value;
            })
        .def_property(
            "cartesian_tracking_error_rotation_limit_rad",
            [](const ControllerConfig &config) { return config.cartesian_tracking_error_rotation_limit_rad; },
            [](ControllerConfig &config, double value) {
                validate_positive_config_value(value, "cartesian_tracking_error_rotation_limit_rad");
                config.cartesian_tracking_error_rotation_limit_rad = value;
            })
        .def_property(
            "cartesian_tracking_error_timeout_s",
            [](const ControllerConfig &config) { return config.cartesian_tracking_error_timeout_s; },
            [](ControllerConfig &config, double value) {
                validate_positive_config_value(value, "cartesian_tracking_error_timeout_s");
                config.cartesian_tracking_error_timeout_s = value;
            })
        .def_property(
            "cartesian_task_torque_ramp_time_s",
            [](const ControllerConfig &config) { return config.cartesian_task_torque_ramp_time_s; },
            [](ControllerConfig &config, double value) {
                validate_positive_config_value(value, "cartesian_task_torque_ramp_time_s");
                config.cartesian_task_torque_ramp_time_s = value;
            })
        .def_property(
            "cartesian_joint_limit_margin_rad",
            [](const ControllerConfig &config) { return config.cartesian_joint_limit_margin_rad; },
            [](ControllerConfig &config, double value) {
                validate_positive_config_value(value, "cartesian_joint_limit_margin_rad");
                config.cartesian_joint_limit_margin_rad = value;
            })
        .def_readwrite("friction_compensation", &ControllerConfig::friction_compensation)
        .def_readwrite("joint_friction_compensation", &ControllerConfig::joint_friction_compensation)
        .def_readwrite("joint_friction_transition_velocity", &ControllerConfig::joint_friction_transition_velocity)
        .def_readwrite("joint_friction_constant_term", &ControllerConfig::joint_friction_constant_term)
        .def_readwrite("joint_friction_coulomb_coef", &ControllerConfig::joint_friction_coulomb_coef)
        .def_readwrite("joint_friction_viscous_coef", &ControllerConfig::joint_friction_viscous_coef)
        .def_readwrite("gripper_friction_compensation", &ControllerConfig::gripper_friction_compensation)
        .def_readwrite("gripper_friction_transition_velocity",
                       &ControllerConfig::gripper_friction_transition_velocity)
        .def_readwrite("gripper_friction_constant_term", &ControllerConfig::gripper_friction_constant_term)
        .def_readwrite("gripper_friction_coulomb_coef", &ControllerConfig::gripper_friction_coulomb_coef)
        .def_readwrite("gripper_friction_viscous_coef", &ControllerConfig::gripper_friction_viscous_coef)
        .def_readwrite("shutdown_to_passive", &ControllerConfig::shutdown_to_passive)
        .def_readwrite("interpolation_method", &ControllerConfig::interpolation_method)
        .def_readwrite("default_preview_time", &ControllerConfig::default_preview_time)
        .def_readwrite("controller_dt", &ControllerConfig::controller_dt);
    py::class_<RobotConfigFactory>(m, "RobotConfigFactory")
        .def_static("get_instance", &RobotConfigFactory::get_instance, py::return_value_policy::reference)
        .def("get_config", &RobotConfigFactory::get_config);
    py::class_<ControllerConfigFactory>(m, "ControllerConfigFactory")
        .def_static("get_instance", &ControllerConfigFactory::get_instance, py::return_value_policy::reference)
        .def("get_config", py::overload_cast<const std::string &, const int>(&ControllerConfigFactory::get_config))
        .def("get_config", py::overload_cast<const std::string &, const RobotConfig &>(&ControllerConfigFactory::get_config));
    py::enum_<MotorType>(m, "MotorType")
        .value("EC_A4310", MotorType::EC_A4310)
        .value("DM_J4310", MotorType::DM_J4310)
        .value("DM_J4340", MotorType::DM_J4340)
        .value("NONE", MotorType::NONE);
}
