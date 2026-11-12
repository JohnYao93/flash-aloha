#ifndef CARTESIAN_PD_H
#define CARTESIAN_PD_H

#include "app/common.h"

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <memory>
#include <string>
#include <vector>

namespace arx
{

class CartesianJointVelocityFilter
{
  public:
    CartesianJointVelocityFilter(int joint_dof, double controller_dt, double cutoff_hz);

    void set_cutoff(double cutoff_hz);
    double get_cutoff() const;
    VecDoF update(const VecDoF &measured_joint_velocity, double measured_timestamp);
    VecDoF get_filtered_joint_velocity() const;

  private:
    int joint_dof_;
    double controller_dt_;
    double cutoff_hz_;
    VecDoF filtered_joint_velocity_;
    double previous_timestamp_;
    bool initialized_ = false;
};

class CartesianKinematics
{
  public:
    CartesianKinematics(const std::string &urdf_path, int joint_dof, const std::string &base_link_name,
                        const std::string &eef_link_name);

    Eigen::MatrixXd jacobian(const VecDoF &joint_pos) const;
    KDL::Frame frame(const VecDoF &joint_pos) const;
    Pose6d pose(const VecDoF &joint_pos) const;

  private:
    int joint_dof_;
    KDL::Chain chain_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jacobian_solver_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;

    KDL::JntArray to_kdl_joints_(const VecDoF &joint_pos) const;
};

struct CartesianReferencePoint
{
    double timestamp = 0.0;
    KDL::Frame frame = KDL::Frame::Identity();
    Pose6d twist = Pose6d::Zero(); // base-frame [linear, angular]
};

struct CartesianReferenceSample
{
    double timestamp = 0.0;
    KDL::Frame frame = KDL::Frame::Identity();
    Pose6d twist = Pose6d::Zero(); // base-frame [linear, angular]
};

// A Cartesian Hermite interpolator. Translation is cubic in R^3; orientation
// is cubic in the SO(3) tangent space and returns its analytic spatial twist.
class CartesianReferenceInterpolator
{
  public:
    CartesianReferenceInterpolator(double controller_dt, Pose6d velocity_limit, double brake_duration_s = 0.05);

    void init(const CartesianReferenceSample &sample);
    void init_fixed(const KDL::Frame &frame, double timestamp);
    void override_waypoint(double current_time, const KDL::Frame &frame, const Pose6d &terminal_twist,
                           double timestamp);
    void override_trajectory(double current_time, const std::vector<KDL::Frame> &frames,
                             const std::vector<Pose6d> &twists, const std::vector<double> &timestamps);
    CartesianReferenceSample interpolate(double timestamp) const;
    bool is_initialized() const;

    static KDL::Frame frame_from_pose(const Pose6d &pose);
    static Pose6d pose_from_frame(const KDL::Frame &frame);

  private:
    double controller_dt_;
    Pose6d velocity_limit_;
    double brake_duration_s_;
    std::vector<CartesianReferencePoint> trajectory_;

    static CartesianReferenceSample interpolate_segment_(const CartesianReferencePoint &start,
                                                          const CartesianReferencePoint &end, double timestamp);
    void validate_point_(const KDL::Frame &frame, const Pose6d &twist, double timestamp) const;
    void retime_trajectory_();
    void append_brake_if_needed_();
};

void validate_cartesian_joint_velocity_filter_cutoff(double cutoff_hz, double controller_dt);
void validate_cartesian_vector(const Pose6d &values, const std::string &name);
void validate_cartesian_pd_model(int joint_dof);
Pose6d cartesian_pose_error(const KDL::Frame &desired, const KDL::Frame &measured);
Pose6d cartesian_pose_error(const Pose6d &desired, const Pose6d &measured);
Pose6d clamp_cartesian_pose_error(const Pose6d &error, double translation_limit_m, double rotation_limit_rad);
bool cartesian_tracking_error_exceeds(const Pose6d &error, double translation_limit_m, double rotation_limit_rad);
bool cartesian_tracking_error_recovered(const Pose6d &error, double translation_limit_m, double rotation_limit_rad,
                                        double recovery_ratio = 0.5);
bool cartesian_joint_limit_guard_triggered(double position, double velocity, double lower_limit, double upper_limit,
                                           double margin_rad, double outward_velocity_threshold_rad_s);
VecDoF compute_cartesian_pd_torque(const Eigen::MatrixXd &jacobian, const Pose6d &pose_error,
                                   const Pose6d &desired_twist, const Pose6d &actual_twist,
                                   const CartesianPDGain &gain, const Pose6d &wrench_limit,
                                   Pose6d *clamped_wrench = nullptr);
struct CartesianJointVelocitySolution
{
    VecDoF unlimited;
    VecDoF limited;
    double scale = 1.0;
    bool valid = false;
};
CartesianJointVelocitySolution compute_damped_cartesian_joint_velocity(const Eigen::MatrixXd &jacobian,
                                                                        const Pose6d &desired_twist,
                                                                        const VecDoF &joint_velocity_limit,
                                                                        double damping = 0.01);
VecDoF slew_limit_cartesian_task_torque(const VecDoF &requested_torque, const VecDoF &previous_torque,
                                        const VecDoF &torque_limit, double controller_dt, double ramp_time_s);

} // namespace arx

#endif // CARTESIAN_PD_H
