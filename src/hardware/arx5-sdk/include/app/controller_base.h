#ifndef CONTROLLER_BASE_H
#define CONTROLLER_BASE_H
#include "app/common.h"
#include "app/config.h"
#include "app/solver.h"
#include "hardware/arx_can.h"
#include "utils.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

namespace arx
{
class Arx5ControllerBase // parent class for the other two controllers
{
  public:
    Arx5ControllerBase(RobotConfig robot_config, ControllerConfig controller_config, std::string interface_name);
    virtual ~Arx5ControllerBase();
    JointState get_joint_cmd();
    JointState get_joint_state();
    ControlCycleTelemetry get_control_cycle_telemetry();
    EEFState get_eef_state();
    Pose6d get_home_pose();
    void set_gain(Gain new_gain);
    Gain get_gain();

    double get_timestamp();
    RobotConfig get_robot_config();
    ControllerConfig get_controller_config();
    void set_log_level(spdlog::level::level_enum level);
    void set_gravity_compensation(bool enabled);
    void set_friction_compensation(bool enabled);
    void set_joint_friction_compensation(bool enabled, VecDoF transition_velocity, VecDoF constant_term,
                                         VecDoF coulomb_coef, VecDoF viscous_coef);
    void set_gripper_friction_compensation(bool enabled, double transition_velocity, double constant_term,
                                           double coulomb_coef, double viscous_coef);

    void reset_to_home();
    void set_to_damping();

  protected:
    RobotConfig robot_config_;
    ControllerConfig controller_config_;

    int over_current_cnt_ = 0;
    JointState output_joint_cmd_{robot_config_.joint_dof};

    JointState joint_state_{robot_config_.joint_dof};
    Gain gain_{robot_config_.joint_dof};
    // bool prev_gripper_updated_ = false; // Declaring here leads to segfault

    ArxCan can_handle_;
    std::shared_ptr<spdlog::logger> logger_;
    std::thread background_send_recv_thread_;

    bool prev_gripper_updated_ = false; // To suppress the warning message
    std::atomic<bool> background_send_recv_running_{false};
    std::atomic<bool> destroy_background_threads_{false};
    bool shutdown_complete_ = false;

    std::mutex cmd_mutex_;
    std::mutex state_mutex_;
    mutable std::mutex control_telemetry_mutex_;

    // Written only by the control thread. pending_control_telemetry_ is
    // assembled throughout one send/receive cycle and copied atomically into
    // control_telemetry_ at the end for non-real-time readers.
    ControlCycleTelemetry pending_control_telemetry_{robot_config_.joint_dof};
    ControlCycleTelemetry control_telemetry_{robot_config_.joint_dof};
    std::uint64_t control_cycle_sequence_ = 0;

    long int start_time_us_;
    std::shared_ptr<Arx5Solver> solver_;
    JointStateInterpolator interpolator_{robot_config_.joint_dof, controller_config_.interpolation_method};
    void init_robot_();
    void update_joint_state_();
    void update_output_cmd_();
    void send_recv_();
    void recv_();
    void check_joint_state_sanity_();
    void over_current_protection_();
    void background_send_recv_();
    void enter_emergency_state_();
    void start_background_send_recv_();
    void shutdown_();
    virtual VecDoF compute_additional_torque_(const JointState &desired_joint_state,
                                               const JointState &previous_desired_joint_state,
                                               const JointState &measured_joint_state);
    virtual bool arm_motor_joint_pd_enabled_() const;
    virtual double arm_motor_kp_(int joint_index) const;
    virtual double arm_motor_kd_(int joint_index) const;
    virtual void disable_cartesian_pd_for_safety_(bool freeze_desired);

  private:
    friend class Arx5ControllerBaseFrictionTest;
    static VecDoF compute_joint_friction_adjustment_(const RobotConfig &robot_config,
                                                       const ControllerConfig &controller_config,
                                                       const VecDoF &joint_velocity);
    static double compute_gripper_friction_adjustment_(const RobotConfig &robot_config,
                                                       const ControllerConfig &controller_config,
                                                       double gripper_velocity);
};
} // namespace arx

#endif
