#ifndef JOINT_CONTROLLER_H
#define JOINT_CONTROLLER_H
#include "app/common.h"
#include "app/config.h"
#include "app/controller_base.h"
#include "app/solver.h"
#include "hardware/arx_can.h"
#include "utils.h"
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

VecDoF compute_host_joint_damping_torque(const VecDoF &joint_velocity, const VecDoF &joint_damping);

class Arx5JointController : public Arx5ControllerBase
{
  public:
    Arx5JointController(RobotConfig robot_config, ControllerConfig controller_config, std::string interface_name);
    Arx5JointController(std::string model, std::string interface_name);

    void set_joint_cmd(JointState new_cmd);

    void set_joint_traj(std::vector<JointState> new_traj);

    // Diagnostic joint-space damping evaluated in the 500 Hz host control
    // loop and added to t_ff.  Motor-local Kd remains controlled by Gain.
    void set_host_joint_damping(VecDoF joint_damping);
    VecDoF get_host_joint_damping();

    // Only works when background_send_recv is disabled
    void send_recv_once();
    void recv_once();

    void calibrate_gripper();
    void calibrate_joint(int joint_id);

  protected:
    VecDoF compute_additional_torque_(const JointState &desired_joint_state,
                                      const JointState &previous_desired_joint_state,
                                      const JointState &measured_joint_state) override;

  private:
    std::mutex host_joint_damping_mutex_;
    VecDoF host_joint_damping_;
};

} // namespace arx

#endif
