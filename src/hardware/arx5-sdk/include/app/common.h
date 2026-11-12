#ifndef COMMON_H
#define COMMON_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace arx
{
using Pose6d = Eigen::Matrix<double, 6, 1>;
using VecDoF = Eigen::VectorXd; // mutable size according to the robot motor number
enum class CartesianDampingMode
{
    HostCartesian,
    MotorLocalJoint,
};
enum class MotorType
{
    EC_A4310,
    DM_J4310,
    DM_J4340,
    DM_J8009,
    NONE, // If any motor is disabled (e.g. gripper motor)
};

inline double motor_protocol_velocity_command(MotorType motor_type, double joint_velocity,
                                              double velocity_feedback_scale)
{
    if (!std::isfinite(joint_velocity) || !std::isfinite(velocity_feedback_scale) || velocity_feedback_scale <= 0.0)
        throw std::invalid_argument("Motor command velocity and scale must be finite, with a positive scale.");
    if (motor_type == MotorType::EC_A4310)
        return joint_velocity;
    if (motor_type == MotorType::DM_J4310 || motor_type == MotorType::DM_J4340 || motor_type == MotorType::DM_J8009)
        return joint_velocity / velocity_feedback_scale;
    throw std::invalid_argument("Cannot scale command velocity for an unsupported motor type.");
}

struct JointState
{
    double timestamp = 0.0f;
    VecDoF pos;
    VecDoF vel;
    VecDoF torque;
    double gripper_pos = 0.0f;    // m; 0 for close, GRIPPER_WIDTH for fully open
    double gripper_vel = 0.0f;    // s^{-1}; currently not supported
    double gripper_torque = 0.0f; // Nm; currently not supported
    JointState(int dof) : pos(VecDoF::Zero(dof)), vel(VecDoF::Zero(dof)), torque(VecDoF::Zero(dof))
    {
    }
    JointState(VecDoF pos, VecDoF vel, VecDoF torque, double gripper_pos)
        : pos(pos), vel(vel), torque(torque), gripper_pos(gripper_pos)
    {
    }
    JointState(VecDoF pos, VecDoF vel, VecDoF torque, double gripper_pos, double gripper_vel, double gripper_torque)
        : pos(pos), vel(vel), torque(torque), gripper_pos(gripper_pos), gripper_vel(gripper_vel),
          gripper_torque(gripper_torque)
    {
    }

    JointState operator+(const JointState &other) const
    {
        return JointState(pos + other.pos, vel + other.vel, torque + other.torque, gripper_pos + other.gripper_pos,
                          gripper_vel + other.gripper_vel, gripper_torque + other.gripper_torque);
    }
    JointState operator-(const JointState &other) const
    {
        return JointState(pos - other.pos, vel - other.vel, torque - other.torque, gripper_pos - other.gripper_pos,
                          gripper_vel - other.gripper_vel, gripper_torque - other.gripper_torque);
    }
    JointState operator*(const double &scalar) const
    {
        return JointState(pos * scalar, vel * scalar, torque * scalar, gripper_pos * scalar, gripper_vel * scalar,
                          gripper_torque * scalar);
    }
    JointState operator/(const double &scalar) const
    {
        return JointState(pos / scalar, vel / scalar, torque / scalar, gripper_pos / scalar, gripper_vel / scalar,
                          gripper_torque / scalar);
    }
    // For pybind11 to update values
    VecDoF &get_pos_ref()
    {
        return pos;
    }
    VecDoF &get_vel_ref()
    {
        return vel;
    }
    VecDoF &get_torque_ref()
    {
        return torque;
    }
};

struct Gain
{
    VecDoF kp;
    VecDoF kd;
    float gripper_kp = 0.0f;
    float gripper_kd = 0.0f;
    Gain(int dof) : kp(VecDoF::Zero(dof)), kd(VecDoF::Zero(dof))
    {
    }
    Gain(VecDoF kp, VecDoF kd, float gripper_kp, float gripper_kd)
        : kp(kp), kd(kd), gripper_kp(gripper_kp), gripper_kd(gripper_kd)
    {
        if (kp.size() != kd.size())
            throw std::invalid_argument("Length of kp is not equal to kd.");
    }
    Gain operator+(const Gain &other) const
    {
        return Gain(kp + other.kp, kd + other.kd, gripper_kp + other.gripper_kp, gripper_kd + other.gripper_kd);
    }
    Gain operator*(const double &scalar) const
    {
        return Gain(kp * scalar, kd * scalar, gripper_kp * scalar, gripper_kd * scalar);
    }
    VecDoF &get_kp_ref()
    {
        return kp;
    }
    VecDoF &get_kd_ref()
    {
        return kd;
    }
};

struct CartesianPDGain
{
    Pose6d kp;
    Pose6d kd;

    CartesianPDGain() : kp(Pose6d::Zero()), kd(Pose6d::Zero())
    {
    }

    CartesianPDGain(Pose6d kp, Pose6d kd) : kp(kp), kd(kd)
    {
        if (!kp.array().isFinite().all() || !kd.array().isFinite().all())
            throw std::invalid_argument("Cartesian PD gains must be finite.");
        if ((kp.array() < 0.0).any() || (kd.array() < 0.0).any())
            throw std::invalid_argument("Cartesian PD gains must be nonnegative.");
    }

    Pose6d &get_kp_ref()
    {
        return kp;
    }

    Pose6d &get_kd_ref()
    {
        return kd;
    }
};

// One coherently published snapshot of the 500 Hz controller cycle. All
// timestamps use the controller's steady-clock epoch (the same epoch returned
// by Arx5ControllerBase::get_timestamp()). The CAN implementation is supplied
// by a prebuilt library, so joint_send_call_{start,end}_s bracket the SDK call
// into that library and feedback_publish_timestamp_s marks when its latest
// aggregate motor snapshot was copied into joint_state_. They are deliberately
// not described as hardware or per-frame timestamps.
struct ControlCycleTelemetry
{
    std::uint64_t sequence = 0;
    bool cartesian_pd_enabled = false;
    bool has_cartesian_data = false;
    CartesianDampingMode cartesian_damping_mode = CartesianDampingMode::HostCartesian;
    double cartesian_pseudoinverse_damping = 0.0;
    double cartesian_joint_velocity_scale = 1.0;
    bool cartesian_joint_velocity_scaled = false;

    double cycle_start_s = 0.0;
    double control_compute_start_s = 0.0;
    double control_compute_end_s = 0.0;
    double feedback_used_timestamp_s = 0.0;
    double motor_snapshot_read_start_s = 0.0;
    double motor_snapshot_read_end_s = 0.0;
    double feedback_publish_timestamp_s = 0.0;
    double cycle_end_s = 0.0;

    VecDoF joint_send_call_start_s;
    VecDoF joint_send_call_end_s;

    VecDoF control_joint_position;
    VecDoF control_joint_velocity;
    VecDoF control_joint_torque;
    VecDoF filtered_joint_velocity;

    Pose6d desired_pose = Pose6d::Zero();
    Pose6d desired_twist = Pose6d::Zero();
    Pose6d measured_pose = Pose6d::Zero();
    Pose6d pose_error = Pose6d::Zero();
    Pose6d raw_twist = Pose6d::Zero();
    Pose6d filtered_twist = Pose6d::Zero();
    Pose6d proportional_wrench = Pose6d::Zero();
    Pose6d derivative_wrench = Pose6d::Zero();
    Pose6d clamped_wrench = Pose6d::Zero();

    VecDoF requested_cartesian_task_torque;
    VecDoF slewed_cartesian_task_torque;
    VecDoF host_cartesian_stiffness_torque;
    VecDoF host_cartesian_damping_torque;
    VecDoF gravity_torque;
    VecDoF desired_joint_velocity_unlimited;
    VecDoF desired_joint_velocity_limited;
    VecDoF final_motor_kp;
    VecDoF final_motor_kd;
    VecDoF final_motor_protocol_velocity;
    VecDoF final_joint_position_command;
    VecDoF final_joint_velocity_command;
    VecDoF final_joint_torque_command;

    VecDoF feedback_joint_position;
    VecDoF feedback_joint_velocity;
    VecDoF feedback_joint_torque;

    explicit ControlCycleTelemetry(int dof = 0)
        : joint_send_call_start_s(VecDoF::Zero(dof)), joint_send_call_end_s(VecDoF::Zero(dof)),
          control_joint_position(VecDoF::Zero(dof)), control_joint_velocity(VecDoF::Zero(dof)),
          control_joint_torque(VecDoF::Zero(dof)), filtered_joint_velocity(VecDoF::Zero(dof)),
          requested_cartesian_task_torque(VecDoF::Zero(dof)), slewed_cartesian_task_torque(VecDoF::Zero(dof)),
          host_cartesian_stiffness_torque(VecDoF::Zero(dof)), host_cartesian_damping_torque(VecDoF::Zero(dof)),
          gravity_torque(VecDoF::Zero(dof)), desired_joint_velocity_unlimited(VecDoF::Zero(dof)),
          desired_joint_velocity_limited(VecDoF::Zero(dof)), final_motor_kp(VecDoF::Zero(dof)),
          final_motor_kd(VecDoF::Zero(dof)), final_motor_protocol_velocity(VecDoF::Zero(dof)),
          final_joint_position_command(VecDoF::Zero(dof)), final_joint_velocity_command(VecDoF::Zero(dof)),
          final_joint_torque_command(VecDoF::Zero(dof)), feedback_joint_position(VecDoF::Zero(dof)),
          feedback_joint_velocity(VecDoF::Zero(dof)), feedback_joint_torque(VecDoF::Zero(dof))
    {
    }

    void reset_for_cycle(std::uint64_t new_sequence)
    {
        sequence = new_sequence;
        cartesian_pd_enabled = false;
        has_cartesian_data = false;
        cartesian_damping_mode = CartesianDampingMode::HostCartesian;
        cartesian_pseudoinverse_damping = 0.0;
        cartesian_joint_velocity_scale = 1.0;
        cartesian_joint_velocity_scaled = false;
        cycle_start_s = 0.0;
        control_compute_start_s = 0.0;
        control_compute_end_s = 0.0;
        feedback_used_timestamp_s = 0.0;
        motor_snapshot_read_start_s = 0.0;
        motor_snapshot_read_end_s = 0.0;
        feedback_publish_timestamp_s = 0.0;
        cycle_end_s = 0.0;
        joint_send_call_start_s.setZero();
        joint_send_call_end_s.setZero();
        control_joint_position.setZero();
        control_joint_velocity.setZero();
        control_joint_torque.setZero();
        filtered_joint_velocity.setZero();
        desired_pose.setZero();
        desired_twist.setZero();
        measured_pose.setZero();
        pose_error.setZero();
        raw_twist.setZero();
        filtered_twist.setZero();
        proportional_wrench.setZero();
        derivative_wrench.setZero();
        clamped_wrench.setZero();
        requested_cartesian_task_torque.setZero();
        slewed_cartesian_task_torque.setZero();
        host_cartesian_stiffness_torque.setZero();
        host_cartesian_damping_torque.setZero();
        gravity_torque.setZero();
        desired_joint_velocity_unlimited.setZero();
        desired_joint_velocity_limited.setZero();
        final_motor_kp.setZero();
        final_motor_kd.setZero();
        final_motor_protocol_velocity.setZero();
        final_joint_position_command.setZero();
        final_joint_velocity_command.setZero();
        final_joint_torque_command.setZero();
        feedback_joint_position.setZero();
        feedback_joint_velocity.setZero();
        feedback_joint_torque.setZero();
    }
};

struct EEFState
{
    double timestamp = 0.0f;
    Pose6d pose_6d;               // x, y, z, roll, pitch, yaw
    double gripper_pos = 0.0f;    // m; 0 for close, GRIPPER_WIDTH for fully open
    double gripper_vel = 0.0f;    // s^{-1}
    double gripper_torque = 0.0f; // Nm
    EEFState() : pose_6d(Pose6d::Zero())
    {
    }
    EEFState(Pose6d pose_6d, double gripper_pos) : pose_6d(pose_6d), gripper_pos(gripper_pos)
    {
    }
    EEFState(Pose6d pose_6d, double gripper_pos, double gripper_vel, double gripper_torque)
        : pose_6d(pose_6d), gripper_pos(gripper_pos), gripper_vel(gripper_vel), gripper_torque(gripper_torque)
    {
    }
    EEFState operator+(const EEFState &other) const
    {
        return EEFState(pose_6d + other.pose_6d, gripper_pos + other.gripper_pos, gripper_vel + other.gripper_vel,
                        gripper_torque + other.gripper_torque);
    }
    EEFState operator*(const double &scalar) const
    {
        return EEFState(pose_6d * scalar, gripper_pos * scalar, gripper_vel * scalar, gripper_torque * scalar);
    }
    Pose6d &get_pose_6d_ref()
    {
        return pose_6d;
    }
};

} // namespace arx

#define sleep_ms(x) std::this_thread::sleep_for(std::chrono::milliseconds(x))
#define sleep_us(x) std::this_thread::sleep_for(std::chrono::microseconds(x))
#define get_time_us()                                                                                                  \
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()

#endif
