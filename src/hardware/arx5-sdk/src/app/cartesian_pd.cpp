#include "app/cartesian_pd.h"

#include <cmath>
#include <kdl/jacobian.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <limits>
#include <stdexcept>
#include <algorithm>

using namespace arx;

namespace
{
Eigen::Matrix3d skew(const Eigen::Vector3d &value)
{
    Eigen::Matrix3d result;
    result << 0.0, -value.z(), value.y(), value.z(), 0.0, -value.x(), -value.y(), value.x(), 0.0;
    return result;
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d &phi)
{
    const double angle = phi.norm();
    const Eigen::Matrix3d hat = skew(phi);
    if (angle < 1e-8)
        return Eigen::Matrix3d::Identity() + 0.5 * hat + (1.0 / 6.0) * hat * hat;
    return Eigen::Matrix3d::Identity() + (1.0 - std::cos(angle)) / (angle * angle) * hat +
           (angle - std::sin(angle)) / (angle * angle * angle) * hat * hat;
}

KDL::Rotation rotation_exp(const Eigen::Vector3d &phi)
{
    const double angle = phi.norm();
    if (angle < 1e-14)
        return KDL::Rotation::Identity();
    return KDL::Rotation::Rot(KDL::Vector(phi.x() / angle, phi.y() / angle, phi.z() / angle), angle);
}

Eigen::Vector3d rotation_log(const KDL::Rotation &rotation)
{
    const KDL::Vector value = rotation.GetRot();
    return Eigen::Vector3d(value.x(), value.y(), value.z());
}

bool finite_frame(const KDL::Frame &frame)
{
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            if (!std::isfinite(frame.M(row, column)))
                return false;
    return std::isfinite(frame.p.x()) && std::isfinite(frame.p.y()) && std::isfinite(frame.p.z());
}
} // namespace

CartesianReferenceInterpolator::CartesianReferenceInterpolator(double controller_dt, Pose6d velocity_limit,
                                                               double brake_duration_s)
    : controller_dt_(controller_dt), velocity_limit_(velocity_limit), brake_duration_s_(brake_duration_s)
{
    if (!std::isfinite(controller_dt_) || controller_dt_ <= 0.0)
        throw std::invalid_argument("Cartesian reference controller_dt must be finite and positive.");
    if (!velocity_limit_.array().isFinite().all() || (velocity_limit_.array() <= 0.0).any())
        throw std::invalid_argument("Cartesian reference velocity limits must be finite and positive.");
    if (!std::isfinite(brake_duration_s_) || brake_duration_s_ <= 0.0)
        throw std::invalid_argument("Cartesian reference brake duration must be finite and positive.");
}

KDL::Frame CartesianReferenceInterpolator::frame_from_pose(const Pose6d &pose)
{
    if (!pose.array().isFinite().all())
        throw std::invalid_argument("Cartesian pose must contain only finite values.");
    return KDL::Frame(KDL::Rotation::RPY(pose[3], pose[4], pose[5]), KDL::Vector(pose[0], pose[1], pose[2]));
}

Pose6d CartesianReferenceInterpolator::pose_from_frame(const KDL::Frame &frame)
{
    if (!finite_frame(frame))
        throw std::invalid_argument("Cartesian frame must contain only finite values.");
    Pose6d pose;
    pose.head<3>() << frame.p.x(), frame.p.y(), frame.p.z();
    frame.M.GetRPY(pose[3], pose[4], pose[5]);
    return pose;
}

void CartesianReferenceInterpolator::validate_point_(const KDL::Frame &frame, const Pose6d &twist,
                                                     double timestamp) const
{
    if (!finite_frame(frame) || !twist.array().isFinite().all() || !std::isfinite(timestamp))
        throw std::invalid_argument("Cartesian reference pose, twist, and timestamp must be finite.");
    if ((twist.cwiseAbs().array() > velocity_limit_.array() + 1e-12).any())
        throw std::invalid_argument("Cartesian reference endpoint twist exceeds ee_vel_max.");
}

void CartesianReferenceInterpolator::init_fixed(const KDL::Frame &frame, double timestamp)
{
    init(CartesianReferenceSample{timestamp, frame, Pose6d::Zero()});
}

void CartesianReferenceInterpolator::init(const CartesianReferenceSample &sample)
{
    validate_point_(sample.frame, sample.twist, sample.timestamp);
    trajectory_.assign(1, CartesianReferencePoint{sample.timestamp, sample.frame, sample.twist});
}

bool CartesianReferenceInterpolator::is_initialized() const
{
    return !trajectory_.empty();
}

CartesianReferenceSample CartesianReferenceInterpolator::interpolate_segment_(const CartesianReferencePoint &start,
                                                                              const CartesianReferencePoint &end,
                                                                              double timestamp)
{
    const double duration = end.timestamp - start.timestamp;
    if (!(duration > 0.0))
        throw std::runtime_error("Cartesian reference segment duration must be positive.");
    const double u = std::max(0.0, std::min(1.0, (timestamp - start.timestamp) / duration));
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 = u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 = u3 - u2;
    const double dh00 = (6.0 * u2 - 6.0 * u) / duration;
    const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    const double dh01 = (-6.0 * u2 + 6.0 * u) / duration;
    const double dh11 = 3.0 * u2 - 2.0 * u;

    const Eigen::Vector3d p0(start.frame.p.x(), start.frame.p.y(), start.frame.p.z());
    const Eigen::Vector3d p1(end.frame.p.x(), end.frame.p.y(), end.frame.p.z());
    const Eigen::Vector3d v0 = start.twist.head<3>();
    const Eigen::Vector3d v1 = end.twist.head<3>();
    const Eigen::Vector3d position = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1;
    const Eigen::Vector3d linear_velocity =
        dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1;

    const Eigen::Vector3d phi1 = rotation_log(end.frame.M * start.frame.M.Inverse());
    const Eigen::Vector3d phi_dot0 = start.twist.tail<3>();
    const Eigen::Vector3d phi_dot1 = so3_left_jacobian(phi1).fullPivLu().solve(end.twist.tail<3>());
    const Eigen::Vector3d phi = h10 * duration * phi_dot0 + h01 * phi1 + h11 * duration * phi_dot1;
    const Eigen::Vector3d phi_dot = dh10 * phi_dot0 + dh01 * phi1 + dh11 * phi_dot1;
    const Eigen::Vector3d angular_velocity = so3_left_jacobian(phi) * phi_dot;

    CartesianReferenceSample sample;
    sample.timestamp = timestamp;
    sample.frame = KDL::Frame(rotation_exp(phi) * start.frame.M,
                              KDL::Vector(position.x(), position.y(), position.z()));
    sample.twist << linear_velocity, angular_velocity;
    return sample;
}

CartesianReferenceSample CartesianReferenceInterpolator::interpolate(double timestamp) const
{
    if (trajectory_.empty())
        throw std::runtime_error("Cartesian reference is not initialized.");
    if (!std::isfinite(timestamp))
        throw std::invalid_argument("Cartesian interpolation timestamp must be finite.");
    if (trajectory_.size() == 1 || timestamp <= trajectory_.front().timestamp)
        return {timestamp, trajectory_.front().frame, trajectory_.front().twist};
    if (timestamp >= trajectory_.back().timestamp)
        return {timestamp, trajectory_.back().frame, trajectory_.back().twist};
    for (size_t index = 0; index + 1 < trajectory_.size(); ++index)
        if (timestamp <= trajectory_[index + 1].timestamp)
            return interpolate_segment_(trajectory_[index], trajectory_[index + 1], timestamp);
    throw std::runtime_error("Cartesian reference interpolation failed.");
}

void CartesianReferenceInterpolator::retime_trajectory_()
{
    for (size_t index = 0; index + 1 < trajectory_.size(); ++index)
    {
        for (int iteration = 0; iteration < 8; ++iteration)
        {
            double worst_ratio = 1.0;
            for (int sample_index = 0; sample_index <= 100; ++sample_index)
            {
                const double time = trajectory_[index].timestamp +
                                    (trajectory_[index + 1].timestamp - trajectory_[index].timestamp) *
                                        sample_index / 100.0;
                const Pose6d twist = interpolate_segment_(trajectory_[index], trajectory_[index + 1], time).twist;
                worst_ratio = std::max(worst_ratio, (twist.cwiseAbs().cwiseQuotient(velocity_limit_)).maxCoeff());
            }
            if (worst_ratio <= 1.0 + 1e-9)
                break;
            const double old_end = trajectory_[index + 1].timestamp;
            const double new_end = trajectory_[index].timestamp +
                                   (old_end - trajectory_[index].timestamp) * worst_ratio * 1.01;
            const double shift = new_end - old_end;
            for (size_t following = index + 1; following < trajectory_.size(); ++following)
                trajectory_[following].timestamp += shift;
        }
    }
}

void CartesianReferenceInterpolator::append_brake_if_needed_()
{
    if (trajectory_.empty() || trajectory_.back().twist.norm() <= 1e-12)
        return;
    const CartesianReferencePoint end = trajectory_.back();
    const double duration = std::max(brake_duration_s_, 2.0 * controller_dt_);
    const Eigen::Vector3d stop_position(end.frame.p.x() + 0.5 * duration * end.twist[0],
                                       end.frame.p.y() + 0.5 * duration * end.twist[1],
                                       end.frame.p.z() + 0.5 * duration * end.twist[2]);
    const Eigen::Vector3d rotation_step = 0.5 * duration * end.twist.tail<3>();
    const KDL::Frame stop_frame(rotation_exp(rotation_step) * end.frame.M,
                                KDL::Vector(stop_position.x(), stop_position.y(), stop_position.z()));
    trajectory_.push_back({end.timestamp + duration, stop_frame, Pose6d::Zero()});
}

void CartesianReferenceInterpolator::override_waypoint(double current_time, const KDL::Frame &frame,
                                                       const Pose6d &terminal_twist, double timestamp)
{
    validate_point_(frame, terminal_twist, timestamp);
    if (!is_initialized())
        throw std::runtime_error("Cartesian reference is not initialized.");
    if (!(timestamp > current_time))
        throw std::invalid_argument("Cartesian reference timestamp must be after current time.");
    const CartesianReferenceSample current = interpolate(current_time);
    trajectory_.clear();
    trajectory_.push_back({current_time, current.frame, current.twist});
    trajectory_.push_back({timestamp, frame, terminal_twist});
    append_brake_if_needed_();
    retime_trajectory_();
}

void CartesianReferenceInterpolator::override_trajectory(double current_time, const std::vector<KDL::Frame> &frames,
                                                         const std::vector<Pose6d> &twists,
                                                         const std::vector<double> &timestamps)
{
    if (!is_initialized())
        throw std::runtime_error("Cartesian reference is not initialized.");
    if (frames.empty() || frames.size() != twists.size() || frames.size() != timestamps.size())
        throw std::invalid_argument("Cartesian reference trajectory arrays must be nonempty and equal length.");
    double previous = current_time;
    for (size_t index = 0; index < frames.size(); ++index)
    {
        validate_point_(frames[index], twists[index], timestamps[index]);
        if (!(timestamps[index] > previous))
            throw std::invalid_argument("Cartesian reference timestamps must be strictly increasing and future.");
        previous = timestamps[index];
    }
    const CartesianReferenceSample current = interpolate(current_time);
    trajectory_.clear();
    trajectory_.push_back({current_time, current.frame, current.twist});
    for (size_t index = 0; index < frames.size(); ++index)
        trajectory_.push_back({timestamps[index], frames[index], twists[index]});
    append_brake_if_needed_();
    retime_trajectory_();
}

void arx::validate_cartesian_joint_velocity_filter_cutoff(double cutoff_hz, double controller_dt)
{
    if (!std::isfinite(controller_dt) || controller_dt <= 0.0)
        throw std::invalid_argument("controller_dt must be finite and positive for Cartesian velocity filtering.");
    const double nyquist_hz = 1.0 / (2.0 * controller_dt);
    if (!std::isfinite(cutoff_hz) || cutoff_hz < 0.0 || cutoff_hz >= nyquist_hz)
        throw std::invalid_argument("Cartesian joint velocity filter cutoff must be finite, nonnegative, and below " +
                                    std::to_string(nyquist_hz) + " Hz.");
}

CartesianJointVelocitySolution arx::compute_damped_cartesian_joint_velocity(
    const Eigen::MatrixXd &jacobian, const Pose6d &desired_twist, const VecDoF &joint_velocity_limit, double damping)
{
    CartesianJointVelocitySolution result;
    result.unlimited = VecDoF::Zero(jacobian.cols());
    result.limited = result.unlimited;
    if (jacobian.rows() != 6 || jacobian.cols() <= 0 || joint_velocity_limit.size() != jacobian.cols() ||
        !jacobian.allFinite() || !desired_twist.allFinite() || !joint_velocity_limit.allFinite() ||
        (joint_velocity_limit.array() <= 0.0).any() || !std::isfinite(damping) || damping <= 0.0)
        return result;

    const Eigen::Matrix<double, 6, 6> regularized =
        jacobian * jacobian.transpose() + damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    result.unlimited = jacobian.transpose() * regularized.ldlt().solve(desired_twist);
    if (!result.unlimited.allFinite())
        return result;

    double worst_ratio = 1.0;
    for (Eigen::Index i = 0; i < result.unlimited.size(); ++i)
        worst_ratio = std::max(worst_ratio, std::abs(result.unlimited[i]) / joint_velocity_limit[i]);
    result.scale = 1.0 / worst_ratio;
    result.limited = result.unlimited * result.scale;
    result.valid = result.limited.allFinite() && std::isfinite(result.scale) && result.scale > 0.0;
    return result;
}

CartesianJointVelocityFilter::CartesianJointVelocityFilter(int joint_dof, double controller_dt, double cutoff_hz)
    : joint_dof_(joint_dof), controller_dt_(controller_dt), cutoff_hz_(cutoff_hz),
      filtered_joint_velocity_(VecDoF::Zero(joint_dof)),
      previous_timestamp_(std::numeric_limits<double>::quiet_NaN())
{
    if (joint_dof_ <= 0)
        throw std::invalid_argument("Cartesian joint velocity filter requires a positive joint_dof.");
    validate_cartesian_joint_velocity_filter_cutoff(cutoff_hz_, controller_dt_);
}

void CartesianJointVelocityFilter::set_cutoff(double cutoff_hz)
{
    validate_cartesian_joint_velocity_filter_cutoff(cutoff_hz, controller_dt_);
    cutoff_hz_ = cutoff_hz;
}

double CartesianJointVelocityFilter::get_cutoff() const
{
    return cutoff_hz_;
}

VecDoF CartesianJointVelocityFilter::update(const VecDoF &measured_joint_velocity, double measured_timestamp)
{
    if (measured_joint_velocity.size() != joint_dof_)
        throw std::invalid_argument("Measured joint velocity must match the Cartesian velocity filter joint_dof.");

    const bool valid_sample = measured_joint_velocity.allFinite();
    const bool valid_timestamp = std::isfinite(measured_timestamp);
    if (!valid_sample || !valid_timestamp)
    {
        filtered_joint_velocity_ = measured_joint_velocity;
        initialized_ = false;
        previous_timestamp_ = std::numeric_limits<double>::quiet_NaN();
        return filtered_joint_velocity_;
    }

    if (cutoff_hz_ == 0.0)
    {
        filtered_joint_velocity_ = measured_joint_velocity;
        previous_timestamp_ = measured_timestamp;
        initialized_ = true;
        return filtered_joint_velocity_;
    }

    const double measured_dt = measured_timestamp - previous_timestamp_;
    if (!initialized_ || !std::isfinite(measured_dt) || measured_dt <= 0.0 ||
        measured_dt > 10.0 * controller_dt_)
    {
        filtered_joint_velocity_ = measured_joint_velocity;
        previous_timestamp_ = measured_timestamp;
        initialized_ = true;
        return filtered_joint_velocity_;
    }

    const double alpha = 1.0 - std::exp(-2.0 * std::acos(-1.0) * cutoff_hz_ * measured_dt);
    filtered_joint_velocity_ += alpha * (measured_joint_velocity - filtered_joint_velocity_);
    previous_timestamp_ = measured_timestamp;
    return filtered_joint_velocity_;
}

VecDoF CartesianJointVelocityFilter::get_filtered_joint_velocity() const
{
    return filtered_joint_velocity_;
}

CartesianKinematics::CartesianKinematics(const std::string &urdf_path, int joint_dof,
                                         const std::string &base_link_name, const std::string &eef_link_name)
    : joint_dof_(joint_dof)
{
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf_path, tree))
        throw std::runtime_error("Failed to parse URDF for Cartesian PD: " + urdf_path);
    if (!tree.getChain(base_link_name, eef_link_name, chain_))
        throw std::runtime_error("Failed to build KDL chain from '" + base_link_name + "' to '" + eef_link_name +
                                 "' for Cartesian PD.");
    if (static_cast<int>(chain_.getNrOfJoints()) != joint_dof_)
        throw std::runtime_error("Cartesian PD KDL chain has " + std::to_string(chain_.getNrOfJoints()) +
                                 " joints, but robot_config.joint_dof is " + std::to_string(joint_dof_) + ".");

    jacobian_solver_.reset(new KDL::ChainJntToJacSolver(chain_));
    fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(chain_));
}

KDL::JntArray CartesianKinematics::to_kdl_joints_(const VecDoF &joint_pos) const
{
    if (joint_pos.size() != joint_dof_)
        throw std::invalid_argument("Cartesian kinematics joint vector must match robot joint_dof.");
    if (!joint_pos.array().isFinite().all())
        throw std::invalid_argument("Cartesian kinematics joint vector must be finite.");

    KDL::JntArray joints(joint_dof_);
    for (int i = 0; i < joint_dof_; ++i)
        joints(i) = joint_pos[i];
    return joints;
}

Eigen::MatrixXd CartesianKinematics::jacobian(const VecDoF &joint_pos) const
{
    KDL::Jacobian jacobian(joint_dof_);
    if (jacobian_solver_->JntToJac(to_kdl_joints_(joint_pos), jacobian) < 0)
        throw std::runtime_error("KDL failed to compute the Cartesian PD Jacobian.");
    return jacobian.data;
}

KDL::Frame CartesianKinematics::frame(const VecDoF &joint_pos) const
{
    KDL::Frame result;
    if (fk_solver_->JntToCart(to_kdl_joints_(joint_pos), result) < 0)
        throw std::runtime_error("KDL failed to compute Cartesian PD forward kinematics.");
    return result;
}

Pose6d CartesianKinematics::pose(const VecDoF &joint_pos) const
{
    const KDL::Frame result = frame(joint_pos);
    Pose6d pose;
    pose.head<3>() << result.p.x(), result.p.y(), result.p.z();
    result.M.GetRPY(pose[3], pose[4], pose[5]);
    return pose;
}

void arx::validate_cartesian_vector(const Pose6d &values, const std::string &name)
{
    if (!values.array().isFinite().all())
        throw std::invalid_argument(name + " must contain only finite values.");
    if ((values.array() < 0.0).any())
        throw std::invalid_argument(name + " must contain only nonnegative values.");
}

void arx::validate_cartesian_pd_model(int joint_dof)
{
    if (joint_dof != 6)
        throw std::runtime_error("Cartesian PD is unsupported for this model: expected a six-joint X5/L5 variant, got " +
                                 std::to_string(joint_dof) + " joints.");
}

Pose6d arx::cartesian_pose_error(const KDL::Frame &desired, const KDL::Frame &measured)
{
    Pose6d error;
    const KDL::Vector translation_error = desired.p - measured.p;
    const KDL::Vector rotation_error = (desired.M * measured.M.Inverse()).GetRot();
    error << translation_error.x(), translation_error.y(), translation_error.z(), rotation_error.x(),
        rotation_error.y(), rotation_error.z();
    return error;
}

Pose6d arx::cartesian_pose_error(const Pose6d &desired, const Pose6d &measured)
{
    const KDL::Frame desired_frame(KDL::Rotation::RPY(desired[3], desired[4], desired[5]),
                                   KDL::Vector(desired[0], desired[1], desired[2]));
    const KDL::Frame measured_frame(KDL::Rotation::RPY(measured[3], measured[4], measured[5]),
                                    KDL::Vector(measured[0], measured[1], measured[2]));
    return cartesian_pose_error(desired_frame, measured_frame);
}

Pose6d arx::clamp_cartesian_pose_error(const Pose6d &error, double translation_limit_m, double rotation_limit_rad)
{
    if (!error.array().isFinite().all() || !std::isfinite(translation_limit_m) || translation_limit_m <= 0.0 ||
        !std::isfinite(rotation_limit_rad) || rotation_limit_rad <= 0.0)
        throw std::invalid_argument("Cartesian pose error and limits must be finite; limits must be positive.");
    Pose6d clamped = error;
    const double translation_norm = clamped.head<3>().norm();
    if (translation_norm > translation_limit_m)
        clamped.head<3>() *= translation_limit_m / translation_norm;
    const double rotation_norm = clamped.tail<3>().norm();
    if (rotation_norm > rotation_limit_rad)
        clamped.tail<3>() *= rotation_limit_rad / rotation_norm;
    return clamped;
}

bool arx::cartesian_tracking_error_exceeds(const Pose6d &error, double translation_limit_m,
                                           double rotation_limit_rad)
{
    (void)clamp_cartesian_pose_error(error, translation_limit_m, rotation_limit_rad);
    return error.head<3>().norm() > translation_limit_m || error.tail<3>().norm() > rotation_limit_rad;
}

bool arx::cartesian_tracking_error_recovered(const Pose6d &error, double translation_limit_m,
                                             double rotation_limit_rad, double recovery_ratio)
{
    (void)clamp_cartesian_pose_error(error, translation_limit_m, rotation_limit_rad);
    if (!std::isfinite(recovery_ratio) || recovery_ratio <= 0.0 || recovery_ratio >= 1.0)
        throw std::invalid_argument("Cartesian tracking recovery ratio must be finite and between zero and one.");
    return error.head<3>().norm() <= recovery_ratio * translation_limit_m &&
           error.tail<3>().norm() <= recovery_ratio * rotation_limit_rad;
}

bool arx::cartesian_joint_limit_guard_triggered(double position, double velocity, double lower_limit,
                                                double upper_limit, double margin_rad,
                                                double outward_velocity_threshold_rad_s)
{
    const double values[] = {position, velocity, lower_limit, upper_limit, margin_rad,
                             outward_velocity_threshold_rad_s};
    for (double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument("Cartesian joint-limit guard inputs must be finite.");
    if (!(lower_limit < upper_limit) || margin_rad <= 0.0 || outward_velocity_threshold_rad_s < 0.0)
        throw std::invalid_argument("Cartesian joint-limit guard limits are invalid.");
    if (position < lower_limit || position > upper_limit)
        return true;
    return (position - lower_limit <= margin_rad && velocity < -outward_velocity_threshold_rad_s) ||
           (upper_limit - position <= margin_rad && velocity > outward_velocity_threshold_rad_s);
}

VecDoF arx::compute_cartesian_pd_torque(const Eigen::MatrixXd &jacobian, const Pose6d &pose_error,
                                        const Pose6d &desired_twist, const Pose6d &actual_twist,
                                        const CartesianPDGain &gain, const Pose6d &wrench_limit,
                                        Pose6d *clamped_wrench)
{
    if (jacobian.rows() != 6)
        throw std::invalid_argument("Cartesian PD Jacobian must have six rows.");
    if (!jacobian.array().isFinite().all() || !pose_error.array().isFinite().all() ||
        !desired_twist.array().isFinite().all() || !actual_twist.array().isFinite().all())
        throw std::invalid_argument("Cartesian PD inputs must be finite.");
    validate_cartesian_vector(gain.kp, "Cartesian kp");
    validate_cartesian_vector(gain.kd, "Cartesian kd");
    validate_cartesian_vector(wrench_limit, "Cartesian wrench limit");

    Pose6d wrench = gain.kp.cwiseProduct(pose_error) + gain.kd.cwiseProduct(desired_twist - actual_twist);
    wrench = wrench.cwiseMin(wrench_limit).cwiseMax(-wrench_limit);
    if (clamped_wrench != nullptr)
        *clamped_wrench = wrench;
    return jacobian.transpose() * wrench;
}

VecDoF arx::slew_limit_cartesian_task_torque(const VecDoF &requested_torque, const VecDoF &previous_torque,
                                             const VecDoF &torque_limit, double controller_dt, double ramp_time_s)
{
    if (requested_torque.size() == 0 || requested_torque.size() != previous_torque.size() ||
        requested_torque.size() != torque_limit.size())
        throw std::invalid_argument("Cartesian task torque vectors must have equal nonzero dimensions.");
    if (!requested_torque.array().isFinite().all() || !previous_torque.array().isFinite().all() ||
        !torque_limit.array().isFinite().all() || (torque_limit.array() <= 0.0).any() ||
        !std::isfinite(controller_dt) || controller_dt <= 0.0 || !std::isfinite(ramp_time_s) || ramp_time_s <= 0.0)
        throw std::invalid_argument("Cartesian task torque slew inputs must be finite and limits positive.");
    const VecDoF maximum_delta = torque_limit * (controller_dt / ramp_time_s);
    const VecDoF delta = (requested_torque - previous_torque).cwiseMin(maximum_delta).cwiseMax(-maximum_delta);
    return previous_torque + delta;
}
