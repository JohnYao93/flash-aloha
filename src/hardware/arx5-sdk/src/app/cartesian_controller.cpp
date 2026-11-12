#include "app/cartesian_controller.h"
#include "app/common.h"
#include "app/config.h"
#include "cartesian_ik.h"
#include "utils.h"
#include <stdexcept>
#include <sys/syscall.h>
#include <sys/types.h>

using namespace arx;

namespace
{
class ScopedAtomicFlag
{
  public:
    explicit ScopedAtomicFlag(std::atomic<bool> &flag) : flag_(flag)
    {
        flag_.store(true);
    }

    ~ScopedAtomicFlag()
    {
        flag_.store(false);
    }

    ScopedAtomicFlag(const ScopedAtomicFlag &) = delete;
    ScopedAtomicFlag &operator=(const ScopedAtomicFlag &) = delete;

  private:
    std::atomic<bool> &flag_;
};

ControllerConfig validate_cartesian_controller_config(ControllerConfig config)
{
    validate_cartesian_vector(config.default_cartesian_kp, "default_cartesian_kp");
    validate_cartesian_vector(config.default_cartesian_kd, "default_cartesian_kd");
    validate_cartesian_vector(config.cartesian_wrench_limit, "cartesian_wrench_limit");
    if (!std::isfinite(config.controller_dt) || config.controller_dt <= 0.0)
        throw std::invalid_argument("controller_dt must be finite and positive for Cartesian PD.");
    validate_cartesian_joint_velocity_filter_cutoff(config.cartesian_joint_velocity_filter_cutoff_hz,
                                                    config.controller_dt);
    const double positive_values[] = {config.cartesian_tracking_error_translation_limit_m,
                                      config.cartesian_tracking_error_rotation_limit_rad,
                                      config.cartesian_tracking_error_timeout_s,
                                      config.cartesian_task_torque_ramp_time_s,
                                      config.cartesian_joint_limit_margin_rad};
    for (double value : positive_values)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("Cartesian safety limits and durations must be finite and positive.");
    return config;
}
} // namespace

Arx5CartesianController::Arx5CartesianController(RobotConfig robot_config, ControllerConfig controller_config,
                                                 std::string interface_name)
    : Arx5ControllerBase(robot_config, validate_cartesian_controller_config(controller_config), interface_name),
      cartesian_joint_velocity_filter_(robot_config.joint_dof, controller_config.controller_dt,
                                       controller_config.cartesian_joint_velocity_filter_cutoff_hz),
      cartesian_pd_gain_(controller_config.default_cartesian_kp, controller_config.default_cartesian_kd),
      cartesian_motor_kd_(VecDoF::Zero(robot_config.joint_dof)),
      cartesian_reference_(controller_config.controller_dt, robot_config.ee_vel_max),
      previous_cartesian_task_torque_(VecDoF::Zero(robot_config.joint_dof))
{
    if (!controller_config.background_send_recv)
        throw std::runtime_error(
            "controller_config.background_send_recv should be set to true when running cartesian controller.");
    if (robot_config.joint_dof == 6)
    {
        cartesian_kinematics_.reset(new CartesianKinematics(robot_config.urdf_path, robot_config.joint_dof,
                                                            robot_config.base_link_name, robot_config.eef_link_name));
    }
    start_background_send_recv_();
}

Arx5CartesianController::Arx5CartesianController(std::string model, std::string interface_name)
    : Arx5CartesianController::Arx5CartesianController(
          RobotConfigFactory::get_instance().get_config(model),
          ControllerConfigFactory::get_instance().get_config(
              "cartesian_controller", RobotConfigFactory::get_instance().get_config(model)),
          interface_name)
{
}

Arx5CartesianController::~Arx5CartesianController()
{
    disable_cartesian_pd();
    shutdown_();
}

void Arx5CartesianController::set_cartesian_pd_gain(CartesianPDGain new_gain)
{
    validate_cartesian_vector(new_gain.kp, "Cartesian kp");
    validate_cartesian_vector(new_gain.kd, "Cartesian kd");
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    cartesian_pd_gain_ = new_gain;
}

CartesianPDGain Arx5CartesianController::get_cartesian_pd_gain()
{
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    return CartesianPDGain(cartesian_pd_gain_.kp, cartesian_pd_gain_.kd);
}

void Arx5CartesianController::set_cartesian_damping_mode(CartesianDampingMode mode)
{
    if (cartesian_pd_enabled_.load() || cartesian_mode_transitioning_.load())
        throw std::runtime_error("Cartesian damping mode cannot be changed while Cartesian PD is enabled.");
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    cartesian_damping_mode_ = mode;
}

CartesianDampingMode Arx5CartesianController::get_cartesian_damping_mode() const
{
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    return cartesian_damping_mode_;
}

void Arx5CartesianController::set_cartesian_motor_kd(VecDoF motor_kd)
{
    if (cartesian_pd_enabled_.load() || cartesian_mode_transitioning_.load())
        throw std::runtime_error("Cartesian motor Kd cannot be changed while Cartesian PD is enabled.");
    if (motor_kd.size() != robot_config_.joint_dof || !motor_kd.allFinite() || (motor_kd.array() < 0.0).any())
        throw std::invalid_argument("Cartesian motor Kd must contain one finite nonnegative value per joint.");
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    cartesian_motor_kd_ = motor_kd;
}

VecDoF Arx5CartesianController::get_cartesian_motor_kd() const
{
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    return cartesian_motor_kd_;
}

void Arx5CartesianController::set_cartesian_joint_velocity_filter_cutoff(double cutoff_hz)
{
    std::lock_guard<std::mutex> guard(cartesian_joint_velocity_filter_mutex_);
    cartesian_joint_velocity_filter_.set_cutoff(cutoff_hz);
    controller_config_.cartesian_joint_velocity_filter_cutoff_hz = cutoff_hz;
}

double Arx5CartesianController::get_cartesian_joint_velocity_filter_cutoff()
{
    std::lock_guard<std::mutex> guard(cartesian_joint_velocity_filter_mutex_);
    return cartesian_joint_velocity_filter_.get_cutoff();
}

VecDoF Arx5CartesianController::get_filtered_joint_velocity()
{
    std::lock_guard<std::mutex> guard(cartesian_joint_velocity_filter_mutex_);
    return cartesian_joint_velocity_filter_.get_filtered_joint_velocity();
}

void Arx5CartesianController::freeze_desired_at_measured_()
{
    JointState measured = get_joint_state();
    measured.vel.setZero();
    measured.torque.setZero();
    measured.gripper_vel = 0.0;
    measured.gripper_torque = 0.0;
    measured.timestamp = get_timestamp();
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.init_fixed(measured);
}

void Arx5CartesianController::anchor_cartesian_reference_(const JointState &measured, double timestamp)
{
    const KDL::Frame measured_frame = cartesian_kinematics_->frame(measured.pos);
    std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
    cartesian_reference_.init_fixed(measured_frame, timestamp);
    last_safe_reference_ = {timestamp, measured_frame, Pose6d::Zero()};
    previous_cartesian_task_torque_.setZero();
    tracking_error_active_ = false;
    tracking_error_start_time_ = 0.0;
}

void Arx5CartesianController::enable_cartesian_pd()
{
    validate_cartesian_pd_model(robot_config_.joint_dof);
    if (!cartesian_kinematics_)
        throw std::runtime_error("Cartesian PD kinematics are unavailable for this model.");
    validate_cartesian_vector(controller_config_.cartesian_wrench_limit, "cartesian_wrench_limit");
    // Re-enabling is also the re-anchor operation. Remove task torque before
    // touching either command state so no control tick can use the old anchor.
    ScopedAtomicFlag transition(cartesian_mode_transitioning_);
    cartesian_pd_enabled_.store(false);
    JointState measured = get_joint_state();
    const double timestamp = get_timestamp();
    measured.vel.setZero();
    measured.torque.setZero();
    measured.gripper_vel = 0.0;
    measured.gripper_torque = 0.0;
    measured.timestamp = timestamp;
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        interpolator_.init_fixed(measured);
    }
    anchor_cartesian_reference_(measured, timestamp);
    cartesian_pd_enabled_.store(true);
}

void Arx5CartesianController::disable_cartesian_pd()
{
    const bool motor_local_damping = get_cartesian_damping_mode() == CartesianDampingMode::MotorLocalJoint;
    ScopedAtomicFlag transition(cartesian_mode_transitioning_);
    cartesian_pd_enabled_.store(false);
    JointState measured = get_joint_state();
    const double timestamp = get_timestamp();
    measured.vel.setZero();
    measured.torque.setZero();
    measured.gripper_vel = 0.0;
    measured.gripper_torque = 0.0;
    measured.timestamp = timestamp;
    if (cartesian_kinematics_)
        anchor_cartesian_reference_(measured, timestamp);
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        interpolator_.init_fixed(measured);
        if (motor_local_damping)
        {
            gain_.kp.setZero();
            gain_.kd = controller_config_.default_kd;
        }
    }
    // Let any control tick that sampled the old joint command finish while
    // motor-level joint PD is still suppressed. Task torque is already off.
    sleep_us(static_cast<int>(controller_config_.controller_dt * 1e6));
}

bool Arx5CartesianController::is_cartesian_pd_enabled() const
{
    return cartesian_pd_enabled_.load();
}

void Arx5CartesianController::set_joint_cmd(JointState new_cmd)
{
    if (cartesian_pd_enabled_.load() || cartesian_mode_transitioning_.load())
        throw std::runtime_error("Joint commands require Cartesian PD to be disabled.");
    const double current_time = get_timestamp();
    if (new_cmd.timestamp == 0.0)
        new_cmd.timestamp = current_time + controller_config_.default_preview_time;
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    if (std::abs(new_cmd.timestamp - current_time) < 1e-3)
        interpolator_.init_fixed(new_cmd);
    else
        interpolator_.override_waypoint(current_time, new_cmd);
}

void Arx5CartesianController::set_joint_traj(std::vector<JointState> new_traj)
{
    if (cartesian_pd_enabled_.load() || cartesian_mode_transitioning_.load())
        throw std::runtime_error("Joint trajectories require Cartesian PD to be disabled.");

    const double start_time = get_timestamp();
    const double avg_window_s = 0.05;
    std::vector<JointState> joint_traj;
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        joint_traj.push_back(interpolator_.interpolate(start_time - 2 * avg_window_s));
        joint_traj.push_back(interpolator_.interpolate(start_time - avg_window_s));
        joint_traj.push_back(interpolator_.interpolate(start_time));
    }

    double previous_timestamp = 0.0;
    for (const JointState &joint_state : new_traj)
    {
        if (joint_state.timestamp <= start_time)
            continue;
        if (joint_state.timestamp <= previous_timestamp)
            throw std::invalid_argument("JointState timestamps must be in ascending order");
        joint_traj.push_back(joint_state);
        previous_timestamp = joint_state.timestamp;
    }
    calc_joint_vel(joint_traj, avg_window_s);

    if (cartesian_pd_enabled_.load() || cartesian_mode_transitioning_.load())
        throw std::runtime_error("Cartesian PD changed state while preparing a joint trajectory.");
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_traj(get_timestamp(), joint_traj);
}

VecDoF Arx5CartesianController::compute_additional_torque_(const JointState &desired_joint_state,
                                                           const JointState &previous_desired_joint_state,
                                                           const JointState &measured_joint_state)
{
    (void)desired_joint_state;
    (void)previous_desired_joint_state;
    VecDoF filtered_joint_velocity;
    {
        std::lock_guard<std::mutex> guard(cartesian_joint_velocity_filter_mutex_);
        filtered_joint_velocity =
            cartesian_joint_velocity_filter_.update(measured_joint_state.vel, measured_joint_state.timestamp);
    }

    pending_control_telemetry_.filtered_joint_velocity = filtered_joint_velocity;
    pending_control_telemetry_.cartesian_pd_enabled = cartesian_pd_enabled_.load();

    CartesianDampingMode damping_mode;
    {
        std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
        damping_mode = cartesian_damping_mode_;
    }
    pending_control_telemetry_.cartesian_damping_mode = damping_mode;

    if (!cartesian_pd_enabled_.load())
        return VecDoF::Zero(robot_config_.joint_dof);

    constexpr double outward_velocity_threshold_rad_s = 0.02;
    for (int index = 0; index < robot_config_.joint_dof; ++index)
    {
        if (cartesian_joint_limit_guard_triggered(
                measured_joint_state.pos[index], filtered_joint_velocity[index], robot_config_.joint_pos_min[index],
                robot_config_.joint_pos_max[index], controller_config_.cartesian_joint_limit_margin_rad,
                outward_velocity_threshold_rad_s))
        {
            disable_from_control_loop_(measured_joint_state,
                                       "joint " + std::to_string(index + 1) + " approached its position limit");
            return VecDoF::Zero(robot_config_.joint_dof);
        }
    }

    const Eigen::MatrixXd measured_jacobian = cartesian_kinematics_->jacobian(measured_joint_state.pos);
    if (!measured_jacobian.allFinite())
    {
        disable_from_control_loop_(measured_joint_state, "Jacobian contained non-finite values");
        return VecDoF::Zero(robot_config_.joint_dof);
    }
    const Pose6d raw_twist = measured_jacobian * measured_joint_state.vel;
    const Pose6d actual_twist = measured_jacobian * filtered_joint_velocity;
    const KDL::Frame measured_frame = cartesian_kinematics_->frame(measured_joint_state.pos);
    CartesianReferenceSample desired_reference;
    Pose6d raw_pose_error;
    bool tracking_timeout = false;
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        desired_reference = cartesian_reference_.interpolate(measured_joint_state.timestamp);
        raw_pose_error = cartesian_pose_error(desired_reference.frame, measured_frame);
        const bool over_limit = cartesian_tracking_error_exceeds(
            raw_pose_error, controller_config_.cartesian_tracking_error_translation_limit_m,
            controller_config_.cartesian_tracking_error_rotation_limit_rad);
        if (!tracking_error_active_ && over_limit)
        {
            tracking_error_active_ = true;
            tracking_error_start_time_ = measured_joint_state.timestamp;
            cartesian_reference_.init_fixed(last_safe_reference_.frame, measured_joint_state.timestamp);
            desired_reference = {measured_joint_state.timestamp, last_safe_reference_.frame, Pose6d::Zero()};
            raw_pose_error = cartesian_pose_error(desired_reference.frame, measured_frame);
            logger_->warn("Cartesian tracking guard froze and canceled reference progression");
        }
        else if (tracking_error_active_)
        {
            const bool recovered = cartesian_tracking_error_recovered(
                raw_pose_error, controller_config_.cartesian_tracking_error_translation_limit_m,
                controller_config_.cartesian_tracking_error_rotation_limit_rad);
            if (recovered)
            {
                tracking_error_active_ = false;
                tracking_error_start_time_ = 0.0;
                logger_->info("Cartesian tracking recovered; holding the frozen reference");
            }
            else if (measured_joint_state.timestamp - tracking_error_start_time_ >=
                     controller_config_.cartesian_tracking_error_timeout_s)
            {
                tracking_timeout = true;
            }
        }
        else
        {
            last_safe_reference_ = desired_reference;
        }
    }
    if (tracking_timeout)
    {
        disable_from_control_loop_(measured_joint_state, "Cartesian tracking error persisted past the safety timeout");
        return VecDoF::Zero(robot_config_.joint_dof);
    }

    const Pose6d pose_error =
        clamp_cartesian_pose_error(raw_pose_error, controller_config_.cartesian_tracking_error_translation_limit_m,
                                   controller_config_.cartesian_tracking_error_rotation_limit_rad);

    pending_control_telemetry_.has_cartesian_data = true;
    pending_control_telemetry_.desired_pose = CartesianReferenceInterpolator::pose_from_frame(desired_reference.frame);
    pending_control_telemetry_.desired_twist = desired_reference.twist;
    pending_control_telemetry_.measured_pose = CartesianReferenceInterpolator::pose_from_frame(measured_frame);
    pending_control_telemetry_.pose_error = pose_error;
    pending_control_telemetry_.raw_twist = raw_twist;
    pending_control_telemetry_.filtered_twist = actual_twist;

    CartesianPDGain gain;
    {
        std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
        gain = cartesian_pd_gain_;
    }
    if (damping_mode == CartesianDampingMode::MotorLocalJoint)
    {
        const CartesianJointVelocitySolution velocity_solution = compute_damped_cartesian_joint_velocity(
            measured_jacobian, desired_reference.twist, robot_config_.joint_vel_max,
            cartesian_pseudoinverse_damping_);
        pending_control_telemetry_.cartesian_pseudoinverse_damping = cartesian_pseudoinverse_damping_;
        pending_control_telemetry_.desired_joint_velocity_unlimited = velocity_solution.unlimited;
        pending_control_telemetry_.desired_joint_velocity_limited = velocity_solution.limited;
        pending_control_telemetry_.cartesian_joint_velocity_scale = velocity_solution.scale;
        pending_control_telemetry_.cartesian_joint_velocity_scaled = velocity_solution.scale < 1.0 - 1e-12;
        // A scale below ten percent indicates that the requested Cartesian twist
        // is not a useful or safe joint-space reference at this configuration.
        if (!velocity_solution.valid || velocity_solution.scale < 0.1)
        {
            disable_from_control_loop_(measured_joint_state,
                                       velocity_solution.valid ? "Cartesian velocity scaling exceeded 10:1"
                                                               : "Cartesian velocity solution was invalid");
            return VecDoF::Zero(robot_config_.joint_dof);
        }
        output_joint_cmd_.pos = measured_joint_state.pos;
        output_joint_cmd_.vel = velocity_solution.limited;
        gain.kd.setZero();
    }
    pending_control_telemetry_.proportional_wrench = gain.kp.cwiseProduct(pose_error);
    pending_control_telemetry_.derivative_wrench = gain.kd.cwiseProduct(desired_reference.twist - actual_twist);
    Pose6d clamped_wrench;
    VecDoF requested_torque = compute_cartesian_pd_torque(measured_jacobian, pose_error, desired_reference.twist,
                                                          actual_twist, gain,
                                                          controller_config_.cartesian_wrench_limit,
                                                          &clamped_wrench);
    pending_control_telemetry_.clamped_wrench = clamped_wrench;
    pending_control_telemetry_.requested_cartesian_task_torque = requested_torque;
    pending_control_telemetry_.host_cartesian_stiffness_torque =
        damping_mode == CartesianDampingMode::MotorLocalJoint
            ? requested_torque
            : measured_jacobian.transpose() * pending_control_telemetry_.proportional_wrench;
    pending_control_telemetry_.host_cartesian_damping_torque =
        measured_jacobian.transpose() * pending_control_telemetry_.derivative_wrench;
    std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
    if (!cartesian_pd_enabled_.load())
        return VecDoF::Zero(robot_config_.joint_dof);
    previous_cartesian_task_torque_ =
        slew_limit_cartesian_task_torque(requested_torque, previous_cartesian_task_torque_,
                                         robot_config_.joint_torque_max, controller_config_.controller_dt,
                                         controller_config_.cartesian_task_torque_ramp_time_s);
    pending_control_telemetry_.slewed_cartesian_task_torque = previous_cartesian_task_torque_;
    return previous_cartesian_task_torque_;
}

bool Arx5CartesianController::arm_motor_joint_pd_enabled_() const
{
    return !cartesian_pd_enabled_.load() && !cartesian_mode_transitioning_.load();
}

double Arx5CartesianController::arm_motor_kp_(int joint_index) const
{
    if (!cartesian_pd_enabled_.load() && !cartesian_mode_transitioning_.load())
        return gain_.kp[joint_index];
    return 0.0;
}

double Arx5CartesianController::arm_motor_kd_(int joint_index) const
{
    if (!cartesian_pd_enabled_.load() && !cartesian_mode_transitioning_.load())
        return gain_.kd[joint_index];
    std::lock_guard<std::mutex> guard(cartesian_pd_mutex_);
    return cartesian_damping_mode_ == CartesianDampingMode::MotorLocalJoint ? cartesian_motor_kd_[joint_index] : 0.0;
}

void Arx5CartesianController::disable_cartesian_pd_for_safety_(bool freeze_desired)
{
    ScopedAtomicFlag transition(cartesian_mode_transitioning_);
    cartesian_pd_enabled_.store(false);
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        previous_cartesian_task_torque_.setZero();
        tracking_error_active_ = false;
    }
    if (freeze_desired)
    {
        JointState measured = get_joint_state();
        const double timestamp = get_timestamp();
        measured.timestamp = timestamp;
        anchor_cartesian_reference_(measured, timestamp);
        freeze_desired_at_measured_();
    }
}

void Arx5CartesianController::disable_from_control_loop_(const JointState &measured, const std::string &reason)
{
    const bool motor_local_damping = get_cartesian_damping_mode() == CartesianDampingMode::MotorLocalJoint;
    ScopedAtomicFlag transition(cartesian_mode_transitioning_);
    cartesian_pd_enabled_.store(false);
    const KDL::Frame measured_frame = cartesian_kinematics_->frame(measured.pos);
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        cartesian_reference_.init_fixed(measured_frame, measured.timestamp);
        last_safe_reference_ = {measured.timestamp, measured_frame, Pose6d::Zero()};
        previous_cartesian_task_torque_.setZero();
        tracking_error_active_ = false;
        tracking_error_start_time_ = 0.0;
    }
    JointState hold = measured;
    hold.vel.setZero();
    hold.torque.setZero();
    hold.gripper_vel = 0.0;
    hold.gripper_torque = 0.0;
    {
        std::lock_guard<std::mutex> guard(cmd_mutex_);
        interpolator_.init_fixed(hold);
        if (motor_local_damping)
        {
            gain_.kp.setZero();
            gain_.kd = controller_config_.default_kd;
        }
    }
    // This helper runs from update_output_cmd_ immediately before the motor
    // send. Replace that already-interpolated command as well as the future
    // interpolator state before motor-level joint PD is restored.
    output_joint_cmd_.pos = hold.pos;
    output_joint_cmd_.vel.setZero();
    logger_->error("Cartesian PD disabled for safety: {}", reason);
}

void Arx5CartesianController::set_gripper_waypoint_(const EEFState &command)
{
    JointState target = get_joint_cmd();
    target.vel.setZero();
    target.torque.setZero();
    target.gripper_pos = command.gripper_pos;
    target.gripper_vel = command.gripper_vel;
    target.gripper_torque = command.gripper_torque;
    target.timestamp = command.timestamp;
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_waypoint(get_timestamp(), target);
}

void Arx5CartesianController::set_gripper_trajectory_(const std::vector<EEFState> &trajectory)
{
    JointState current = get_joint_cmd();
    std::vector<JointState> gripper_trajectory;
    gripper_trajectory.reserve(trajectory.size());
    for (const EEFState &command : trajectory)
    {
        JointState target = current;
        target.vel.setZero();
        target.torque.setZero();
        target.gripper_pos = command.gripper_pos;
        target.gripper_vel = command.gripper_vel;
        target.gripper_torque = command.gripper_torque;
        target.timestamp = command.timestamp;
        gripper_trajectory.push_back(target);
    }
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_traj(get_timestamp(), gripper_trajectory);
}

void Arx5CartesianController::set_eef_cmd(EEFState new_cmd)
{
    if (!cartesian_pd_enabled_.load())
    {
        set_eef_cmd_legacy_ik_(new_cmd);
        return;
    }
    set_eef_cmd_with_twist(new_cmd, Pose6d::Zero());
}

void Arx5CartesianController::set_eef_cmd_with_twist(EEFState new_cmd, Pose6d terminal_twist)
{
    if (!cartesian_pd_enabled_.load())
        throw std::runtime_error("Explicit Cartesian twist commands require Cartesian PD to be enabled.");
    const double current_time = get_timestamp();
    if (new_cmd.timestamp == 0.0)
        new_cmd.timestamp = current_time + controller_config_.default_preview_time;
    const KDL::Frame target = CartesianReferenceInterpolator::frame_from_pose(new_cmd.pose_6d);
    CartesianReferenceSample current_reference;
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        if (!cartesian_pd_enabled_.load())
            throw std::runtime_error("Cartesian PD was disabled before the command could be prepared.");
        current_reference = cartesian_reference_.interpolate(current_time);
    }
    CartesianReferenceInterpolator next_reference(
        controller_config_.controller_dt, robot_config_.ee_vel_max);
    next_reference.init(current_reference);
    // Segment construction and velocity-limit retiming can be more expensive
    // than a 500 Hz evaluation, so do it without holding the control-loop lock.
    next_reference.override_waypoint(current_time, target, terminal_twist, new_cmd.timestamp);
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        if (!cartesian_pd_enabled_.load())
            throw std::runtime_error("Cartesian PD was disabled before the command could be installed.");
        cartesian_reference_ = std::move(next_reference);
        tracking_error_active_ = false;
        tracking_error_start_time_ = 0.0;
    }
    set_gripper_waypoint_(new_cmd);
}

void Arx5CartesianController::set_eef_cmd_legacy_ik_(EEFState new_cmd)
{
    JointState current_joint_state = get_joint_state();

    // The following line only works under c++17
    // auto [success, target_joint_pos] = solver_->inverse_kinematics(new_cmd.pose_6d, current_joint_state.pos);

    std::tuple<int, VecDoF> ik_results;
    ik_results = multi_trial_ik(new_cmd.pose_6d, current_joint_state.pos);
    int ik_status = std::get<0>(ik_results);

    if (new_cmd.timestamp == 0)
        new_cmd.timestamp = get_timestamp() + controller_config_.default_preview_time;

    JointState target_joint_state{robot_config_.joint_dof};
    target_joint_state.pos = std::get<1>(ik_results);
    target_joint_state.gripper_pos = new_cmd.gripper_pos;
    target_joint_state.gripper_vel = new_cmd.gripper_vel;
    target_joint_state.gripper_torque = new_cmd.gripper_torque;
    target_joint_state.timestamp = new_cmd.timestamp;

    // TODO: include velocity
    std::lock_guard<std::mutex> guard(cmd_mutex_);
    interpolator_.override_waypoint(get_timestamp(), target_joint_state);

    if (ik_status != 0)
    {
        logger_->warn("Inverse kinematics failed: {} ({})", solver_->get_ik_status_name(ik_status), ik_status);
    }
}

void Arx5CartesianController::set_eef_traj(std::vector<EEFState> new_traj)
{
    if (!cartesian_pd_enabled_.load())
    {
        set_eef_traj_legacy_ik_(new_traj);
        return;
    }
    if (new_traj.empty())
        throw std::invalid_argument("Cartesian trajectory must not be empty.");

    std::vector<Pose6d> twists(new_traj.size(), Pose6d::Zero());
    if (new_traj.size() > 1)
    {
        std::vector<KDL::Frame> frames;
        frames.reserve(new_traj.size());
        for (const EEFState &state : new_traj)
            frames.push_back(CartesianReferenceInterpolator::frame_from_pose(state.pose_6d));
        for (size_t index = 0; index + 1 < new_traj.size(); ++index)
        {
            const size_t before = index == 0 ? 0 : index - 1;
            const size_t after = index + 1;
            const double duration = new_traj[after].timestamp - new_traj[before].timestamp;
            if (!(duration > 0.0))
                throw std::invalid_argument("EEFState timestamps must be strictly increasing.");
            twists[index].head<3>() =
                (new_traj[after].pose_6d.head<3>() - new_traj[before].pose_6d.head<3>()) / duration;
            const KDL::Vector rotation = (frames[after].M * frames[before].M.Inverse()).GetRot();
            twists[index].tail<3>() << rotation.x() / duration, rotation.y() / duration, rotation.z() / duration;
            // These velocities are inferred, not caller requirements. Keep
            // them admissible and let the Cartesian interpolator retime the
            // surrounding segments for the remaining peak-speed constraint.
            twists[index] = twists[index].cwiseMin(robot_config_.ee_vel_max).cwiseMax(-robot_config_.ee_vel_max);
        }
    }
    set_eef_traj_with_twists(new_traj, twists);
}

void Arx5CartesianController::set_eef_traj_with_twists(std::vector<EEFState> new_traj,
                                                       std::vector<Pose6d> waypoint_twists)
{
    if (!cartesian_pd_enabled_.load())
        throw std::runtime_error("Explicit Cartesian twist trajectories require Cartesian PD to be enabled.");
    if (new_traj.empty() || new_traj.size() != waypoint_twists.size())
        throw std::invalid_argument("Cartesian pose and twist trajectories must be nonempty and equal length.");
    const double current_time = get_timestamp();
    std::vector<KDL::Frame> frames;
    std::vector<double> timestamps;
    frames.reserve(new_traj.size());
    timestamps.reserve(new_traj.size());
    double previous_timestamp = current_time;
    for (const EEFState &state : new_traj)
    {
        if (!(state.timestamp > previous_timestamp))
            throw std::invalid_argument("EEFState timestamps must be strictly increasing and in the future.");
        frames.push_back(CartesianReferenceInterpolator::frame_from_pose(state.pose_6d));
        timestamps.push_back(state.timestamp);
        previous_timestamp = state.timestamp;
    }
    CartesianReferenceSample current_reference;
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        if (!cartesian_pd_enabled_.load())
            throw std::runtime_error("Cartesian PD was disabled before the trajectory could be prepared.");
        current_reference = cartesian_reference_.interpolate(current_time);
    }
    CartesianReferenceInterpolator next_reference(
        controller_config_.controller_dt, robot_config_.ee_vel_max);
    next_reference.init(current_reference);
    next_reference.override_trajectory(current_time, frames, waypoint_twists, timestamps);
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        if (!cartesian_pd_enabled_.load())
            throw std::runtime_error("Cartesian PD was disabled before the trajectory could be installed.");
        cartesian_reference_ = std::move(next_reference);
        tracking_error_active_ = false;
        tracking_error_start_time_ = 0.0;
    }
    set_gripper_trajectory_(new_traj);
}

void Arx5CartesianController::set_eef_traj_legacy_ik_(std::vector<EEFState> new_traj)
{
    double start_time = get_timestamp();
    std::vector<JointState> joint_traj;
    double avg_window_s = 0.05;
    joint_traj.push_back(interpolator_.interpolate(start_time - 2 * avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time - avg_window_s));
    joint_traj.push_back(interpolator_.interpolate(start_time));

    double prev_timestamp = 0;
    for (auto eef_state : new_traj)
    {
        if (eef_state.timestamp <= start_time)
            continue;
        if (eef_state.timestamp == 0)
            throw std::invalid_argument("EEFState timestamp must be set for all waypoints");
        if (eef_state.timestamp <= prev_timestamp)
            throw std::invalid_argument("EEFState timestamps must be in ascending order");
        JointState current_joint_state = get_joint_state();
        std::tuple<int, VecDoF> ik_results;
        ik_results = multi_trial_ik(eef_state.pose_6d, current_joint_state.pos);
        int ik_status = std::get<0>(ik_results);

        JointState target_joint_state{robot_config_.joint_dof};
        target_joint_state.pos = std::get<1>(ik_results);
        target_joint_state.gripper_pos = eef_state.gripper_pos;
        target_joint_state.gripper_vel = eef_state.gripper_vel;
        target_joint_state.gripper_torque = eef_state.gripper_torque;
        target_joint_state.timestamp = eef_state.timestamp;

        joint_traj.push_back(target_joint_state);
        prev_timestamp = eef_state.timestamp;

        if (ik_status != 0)
        {
            logger_->warn("Inverse kinematics failed: {} ({})", solver_->get_ik_status_name(ik_status), ik_status);
        }
    }

    // Include velocity: first and last point based on current state, others based on neighboring points
    calc_joint_vel(joint_traj, avg_window_s);

    double current_time = get_timestamp();
    std::lock_guard<std::mutex> guard(cmd_mutex_);

    interpolator_.override_traj(current_time, joint_traj);
}

EEFState Arx5CartesianController::get_eef_cmd()
{
    JointState joint_cmd = get_joint_cmd();
    EEFState eef_cmd;
    if (cartesian_pd_enabled_.load())
    {
        std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
        const CartesianReferenceSample reference = cartesian_reference_.interpolate(get_timestamp());
        eef_cmd.pose_6d = CartesianReferenceInterpolator::pose_from_frame(reference.frame);
        eef_cmd.timestamp = reference.timestamp;
    }
    else
    {
        eef_cmd.pose_6d = solver_->forward_kinematics(joint_cmd.pos);
        eef_cmd.timestamp = joint_cmd.timestamp;
    }
    eef_cmd.gripper_pos = joint_cmd.gripper_pos;
    eef_cmd.gripper_vel = joint_cmd.gripper_vel;
    eef_cmd.gripper_torque = joint_cmd.gripper_torque;
    return eef_cmd;
}

Pose6d Arx5CartesianController::get_eef_cmd_twist()
{
    if (!cartesian_pd_enabled_.load())
        return Pose6d::Zero();
    std::lock_guard<std::mutex> guard(cartesian_reference_mutex_);
    return cartesian_reference_.interpolate(get_timestamp()).twist;
}

std::tuple<int, Eigen::VectorXd> Arx5CartesianController::multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                                         Eigen::VectorXd current_joint_pos,
                                                                         int additional_trial_num)
{
    if (additional_trial_num < 0)
        throw std::invalid_argument("Number of additional trials must be non-negative");
    if (current_joint_pos.size() != robot_config_.joint_dof || target_pose_6d.size() != 6)
        throw std::invalid_argument(
            "Inverse kinematics input expected size 6, " + std::to_string(robot_config_.joint_dof) + " but got " +
            std::to_string(target_pose_6d.size()) + ", " + std::to_string(current_joint_pos.size()));
    const auto solve_candidate = [this, &target_pose_6d](const VecDoF &initial_joint_position) {
        const std::tuple<int, VecDoF> result = solver_->inverse_kinematics(target_pose_6d, initial_joint_position);
        detail::IKCandidate candidate{std::get<0>(result), std::get<1>(result)};
        if (candidate.status == 0 &&
            !detail::is_valid_ik_candidate(candidate, robot_config_.joint_pos_min, robot_config_.joint_pos_max))
        {
            candidate.status = -9; // E_EXCEED_JOINT_LIMIT
        }
        return candidate;
    };

    // Small real-time Cartesian updates should converge from the measured
    // configuration. Avoid random restarts, branch changes, and extra work on
    // this normal path.
    const detail::IKCandidate current_candidate = solve_candidate(current_joint_pos);
    if (detail::is_valid_ik_candidate(current_candidate, robot_config_.joint_pos_min,
                                      robot_config_.joint_pos_max))
    {
        return std::make_tuple(current_candidate.status, current_candidate.joint_position);
    }

    std::vector<detail::IKCandidate> candidates;
    candidates.reserve(additional_trial_num + 2);
    candidates.push_back(current_candidate);
    candidates.push_back(solve_candidate(VecDoF::Zero(robot_config_.joint_dof)));
    for (int i = 0; i < additional_trial_num; i++)
    {
        VecDoF initial_joint_position = VecDoF::Random(robot_config_.joint_dof);
        for (int j = 0; j < robot_config_.joint_dof; j++)
        {
            initial_joint_position[j] =
                robot_config_.joint_pos_min[j] + (initial_joint_position[j] + 1.0) / 2.0 *
                                                     (robot_config_.joint_pos_max[j] - robot_config_.joint_pos_min[j]);
        }
        candidates.push_back(solve_candidate(initial_joint_position));
    }

    return detail::select_closest_valid_ik_candidate(candidates, current_joint_pos, robot_config_.joint_pos_min,
                                                      robot_config_.joint_pos_max);
}
