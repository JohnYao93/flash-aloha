#ifndef CARTESIAN_CONTROLLER_H
#define CARTESIAN_CONTROLLER_H

#include "app/cartesian_pd.h"
#include "app/common.h"
#include "app/config.h"
#include "app/controller_base.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace arx
{
class Arx5CartesianController : public Arx5ControllerBase
{
  public:
    Arx5CartesianController(RobotConfig robot_config, ControllerConfig controller_config, std::string interface_name);
    Arx5CartesianController(std::string model, std::string interface_name);
    ~Arx5CartesianController() override;

    void set_eef_cmd(EEFState new_cmd);
    // Joint-space command access used for controlled transitions between
    // Cartesian and motor-local comparison conditions.
    void set_joint_cmd(JointState new_cmd);
    void set_joint_traj(std::vector<JointState> new_traj);
    void set_eef_traj(std::vector<EEFState> new_traj);
    void set_eef_cmd_with_twist(EEFState new_cmd, Pose6d terminal_twist);
    void set_eef_traj_with_twists(std::vector<EEFState> new_traj, std::vector<Pose6d> waypoint_twists);
    EEFState get_eef_cmd();
    Pose6d get_eef_cmd_twist();
    void set_cartesian_pd_gain(CartesianPDGain new_gain);
    CartesianPDGain get_cartesian_pd_gain();
    void set_cartesian_damping_mode(CartesianDampingMode mode);
    CartesianDampingMode get_cartesian_damping_mode() const;
    void set_cartesian_motor_kd(VecDoF motor_kd);
    VecDoF get_cartesian_motor_kd() const;
    void set_cartesian_joint_velocity_filter_cutoff(double cutoff_hz);
    double get_cartesian_joint_velocity_filter_cutoff();
    VecDoF get_filtered_joint_velocity();
    void enable_cartesian_pd();
    void disable_cartesian_pd();
    bool is_cartesian_pd_enabled() const;

    std::tuple<int, Eigen::VectorXd> multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                    Eigen::VectorXd current_joint_pos, int additional_trial_num = 5);

  protected:
    VecDoF compute_additional_torque_(const JointState &desired_joint_state,
                                      const JointState &previous_desired_joint_state,
                                      const JointState &measured_joint_state) override;
    bool arm_motor_joint_pd_enabled_() const override;
    double arm_motor_kp_(int joint_index) const override;
    double arm_motor_kd_(int joint_index) const override;
    void disable_cartesian_pd_for_safety_(bool freeze_desired) override;

  private:
    std::unique_ptr<CartesianKinematics> cartesian_kinematics_;
    CartesianJointVelocityFilter cartesian_joint_velocity_filter_;
    CartesianPDGain cartesian_pd_gain_;
    CartesianDampingMode cartesian_damping_mode_ = CartesianDampingMode::HostCartesian;
    VecDoF cartesian_motor_kd_;
    double cartesian_pseudoinverse_damping_ = 0.01;
    mutable std::mutex cartesian_pd_mutex_;
    mutable std::mutex cartesian_joint_velocity_filter_mutex_;
    mutable std::mutex cartesian_reference_mutex_;
    std::atomic<bool> cartesian_pd_enabled_{false};
    std::atomic<bool> cartesian_mode_transitioning_{false};
    CartesianReferenceInterpolator cartesian_reference_;
    CartesianReferenceSample last_safe_reference_;
    VecDoF previous_cartesian_task_torque_;
    bool tracking_error_active_ = false;
    double tracking_error_start_time_ = 0.0;

    void freeze_desired_at_measured_();
    void anchor_cartesian_reference_(const JointState &measured, double timestamp);
    void set_gripper_waypoint_(const EEFState &command);
    void set_gripper_trajectory_(const std::vector<EEFState> &trajectory);
    void set_eef_cmd_legacy_ik_(EEFState new_cmd);
    void set_eef_traj_legacy_ik_(std::vector<EEFState> new_traj);
    void disable_from_control_loop_(const JointState &measured, const std::string &reason);
};
} // namespace arx

#endif // CARTESIAN_CONTROLLER
