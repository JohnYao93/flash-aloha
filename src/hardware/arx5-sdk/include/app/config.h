#ifndef CONFIG_H
#define CONFIG_H

#include "app/common.h"
#include "utils.h"
#include <memory>
#include <unordered_map>
#include <vector>
namespace arx
{

class RobotConfig
{
  public:
    std::string robot_model;
    std::string interface_name;

    VecDoF joint_pos_min;
    VecDoF joint_pos_max;
    VecDoF joint_vel_max;    // rad/s
    VecDoF joint_torque_max; // N*m
    Pose6d ee_vel_max;
    // end effector speed: m/s for (x, y, z), rad/s for (roll, pitch, yaw)

    double gripper_vel_max; // m/s
    double gripper_torque_max;
    double gripper_width;        // m, fully opened: gripper_width, fully closed: 0
    double gripper_open_readout; // fully-opened gripper motor readout. Should be calibrated using
                                 // python/examples/calibrate.py
    int joint_dof;
    std::vector<int> motor_id;
    std::vector<MotorType> motor_type;
    // Converts velocity values decoded by the prebuilt hardware library into physical motor rad/s.
    // The library uses a fixed DaMiao VMAX of 45 rad/s, while some motors are configured differently.
    VecDoF joint_velocity_feedback_scale;
    int gripper_motor_id;
    MotorType gripper_motor_type; // Set to MotorType::NONE if the robot does not have a gripper
    double gripper_velocity_feedback_scale;

    // Will be used in inverse dynamics calculation.
    // Please change it to other values if the robot arm is not placed on the ground.
    Eigen::Vector3d gravity_vector;

    // Will be used in IK and FK.
    // ID will find stop at last active joint (instead of the eef link with a fixed joint) because of some KDL bugs
    std::string base_link_name;
    std::string eef_link_name;

    std::string urdf_path;

    RobotConfig(std::string robot_model, VecDoF joint_pos_min, VecDoF joint_pos_max, VecDoF joint_vel_max,
                VecDoF joint_torque_max, Pose6d ee_vel_max, double gripper_vel_max, double gripper_torque_max,
                double gripper_width, double gripper_open_readout, int joint_dof, std::vector<int> motor_id,
                std::vector<MotorType> motor_type, int gripper_motor_id, MotorType gripper_motor_type,
                Eigen::Vector3d gravity_vector, std::string base_link_name, std::string eef_link_name,
                std::string urdf_path)
        : robot_model(robot_model), joint_pos_min(joint_pos_min), joint_pos_max(joint_pos_max),
          joint_vel_max(joint_vel_max), joint_torque_max(joint_torque_max), ee_vel_max(ee_vel_max),
          gripper_vel_max(gripper_vel_max), gripper_torque_max(gripper_torque_max), gripper_width(gripper_width),
          gripper_open_readout(gripper_open_readout), joint_dof(joint_dof), motor_id(motor_id), motor_type(motor_type),
          joint_velocity_feedback_scale(VecDoF::Ones(joint_dof)), gripper_motor_id(gripper_motor_id),
          gripper_motor_type(gripper_motor_type), gripper_velocity_feedback_scale(1.0),
          gravity_vector(gravity_vector), base_link_name(base_link_name), eef_link_name(eef_link_name),
          urdf_path(urdf_path)
    {
    }
};

class RobotConfigFactory
{
  public:
    static RobotConfigFactory &get_instance()
    {
        static RobotConfigFactory instance;
        return instance;
    }

    RobotConfig get_config(const std::string &robot_model)
    {
        auto it = configurations.find(robot_model);
        if (it != configurations.end())
        {
            return *(it->second);
        }
        else
        {
            std::string available_models;
            for (auto &config : configurations)
            {
                available_models += config.first + " ";
            }
            throw std::runtime_error("Unknown robot model. Currently available: " + available_models);
        }
    }

  private:
    RobotConfigFactory()
    {
        std::string root_dir = get_root_dir();
        printf("Found root directory: %s\n", root_dir.c_str());
        std::string model_dir = root_dir + "/models";
        printf("URDF models directory: %s\n", model_dir.c_str());
        configurations["X5"] = std::make_shared<RobotConfig>(
            "X5",                                                          // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.088,                                                         // gripper_width
            5.03,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::EC_A4310, MotorType::EC_A4310, MotorType::EC_A4310, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            model_dir + "/X5.urdf"                                            // urdf_path
        );
        configurations["X5_umi"] = std::make_shared<RobotConfig>(
            "X5_umi",                                                      // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.086,                                                         // gripper_width
            4.90,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::EC_A4310, MotorType::EC_A4310, MotorType::EC_A4310, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            model_dir + "/X5_umi.urdf"                                        // urdf_path
        );

        configurations["X5_iphumi"] = std::make_shared<RobotConfig>(*configurations["X5_umi"]);
        configurations["X5_iphumi"]->robot_model = "X5_iphumi";
        configurations["X5_iphumi"]->urdf_path = model_dir + "/X5_iphumi.urdf";

        configurations["L5"] = std::make_shared<RobotConfig>(
            "L5",                                                          // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.100,                                                         // gripper_width
            5.1276,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            model_dir + "/L5.urdf"                                            // urdf_path
        );
        // libhardware.so decodes every DaMiao velocity with VMAX=45 rad/s. These motors are configured with
        // VMAX=10 rad/s (DM4340, joints 1-3) and VMAX=30 rad/s (DM4310, joints 4-6 and gripper).
        configurations["L5"]->joint_velocity_feedback_scale =
            (VecDoF(6) << 10.0 / 45.0, 10.0 / 45.0, 10.0 / 45.0, 30.0 / 45.0, 30.0 / 45.0, 30.0 / 45.0)
                .finished();
        configurations["L5"]->gripper_velocity_feedback_scale = 30.0 / 45.0;

        // Standalone profile: tune Trossen here without inheriting changes to L5.
        configurations["Trossen_L5"] = std::make_shared<RobotConfig>(
            "Trossen_L5",                                                  // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 2.0, 2.0, 2.0, 9.0, 9.0, 9.0).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.056,                                                         // gripper_width
            3.03,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                             // motor_id
            std::vector<MotorType>{MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                             // gripper_motor_id
            MotorType::DM_J4310,                                            // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                  // gravity_vector
            "base_link",                                                   // base_link_name
            "eef_link",                                                    // eef_link_name
            model_dir + "/Trossen_L5.urdf"                                 // urdf_path
        );
        // Same motor velocity calibration as before, explicitly owned by Trossen.
        configurations["Trossen_L5"]->joint_velocity_feedback_scale =
            (VecDoF(6) << 10.0 / 45.0, 10.0 / 45.0, 10.0 / 45.0, 30.0 / 45.0, 30.0 / 45.0, 30.0 / 45.0)
                .finished();
        configurations["Trossen_L5"]->gripper_velocity_feedback_scale = 30.0 / 45.0;

        configurations["L5_umi"] = std::make_shared<RobotConfig>(
            "L5_umi",                                                      // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.086,                                                         // gripper_width
            4.90,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            model_dir + "/L5_umi.urdf"                                        // urdf_path
        );

        configurations["L5_iphumi"] = std::make_shared<RobotConfig>(*configurations["L5_umi"]);
        configurations["L5_iphumi"]->robot_model = "L5_iphumi";
        configurations["L5_iphumi"]->urdf_path = model_dir + "/L5_iphumi.urdf";

        configurations["X7_left"] = std::make_shared<RobotConfig>(
            "X7_left",                                                                 // robot_model
            (VecDoF(7) << -2.09439, -1.5, -1.5, -1.5, -1.2, -0.3, -0.7854).finished(), // joint_pos_min
            (VecDoF(7) << 2.09439, 0.3, 1.5, 0.3, 1.2, 0.7854, 0.7854).finished(),     // joint_pos_max
            (VecDoF(7) << 3.0, 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),               // joint_vel_max
            (VecDoF(7) << 30.0, 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),        // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),                     // ee_vel_max
            0.3,                                                                       // gripper_vel_max
            1.5,                                                                       // gripper_torque_max
            0.088,                                                                     // gripper_width
            5.03,                                                                      // gripper_open_readout
            7,                                                                         // joint_dof
            std::vector<int>{1, 2, 3, 4, 5, 6, 7},                                     // motor_id
            std::vector<MotorType>{MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340,
                                   MotorType::DM_J4310, MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                                     // gripper_motor_id
            MotorType::DM_J4310,                                                                   // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                                        // gravity_vector
            "base_link",                                                                           // base_link_name
            "eef_link",                                                                            // eef_link_name
            model_dir + "/X7_left.urdf"                                                            // urdf_path
        );
        configurations["X7_right"] = std::make_shared<RobotConfig>(
            "X7_right",                                                                   // robot_model
            (VecDoF(7) << -2.09439, -0.3, -1.5, -0.3, -1.2, -0.7854, -0.7854).finished(), // joint_pos_min
            (VecDoF(7) << 2.09439, 1.5, 1.5, 1.5, 1.2, 0.3, 0.7854).finished(),           // joint_pos_max
            (VecDoF(7) << 3.0, 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),                  // joint_vel_max
            (VecDoF(7) << 30.0, 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),           // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),                        // ee_vel_max
            0.3,                                                                          // gripper_vel_max
            1.5,                                                                          // gripper_torque_max
            0.088,                                                                        // gripper_width
            5.03,                                                                         // gripper_open_readout
            7,                                                                            // joint_dof
            std::vector<int>{1, 2, 3, 4, 5, 6, 7},                                        // motor_id
            std::vector<MotorType>{MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340, MotorType::DM_J4340,
                                   MotorType::DM_J4310, MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                                     // gripper_motor_id
            MotorType::DM_J4310,                                                                   // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                                        // gravity_vector
            "base_link",                                                                           // base_link_name
            "eef_link",                                                                            // eef_link_name
            model_dir + "/X7_right.urdf"                                                           // urdf_path
        );
    }

    std::unordered_map<std::string, std::shared_ptr<RobotConfig>> configurations;

    // Disable copy constructor and assignment operator
    RobotConfigFactory(const RobotConfigFactory &) = delete;
    RobotConfigFactory &operator=(const RobotConfigFactory &) = delete;
};

class ControllerConfig
{
  public:
    std::string controller_type;
    VecDoF default_kp;
    VecDoF default_kd;
    double default_gripper_kp;
    double default_gripper_kd;
    int over_current_cnt_max;
    double controller_dt;
    bool gravity_compensation;
    Pose6d default_cartesian_kp = Pose6d::Zero();
    Pose6d default_cartesian_kd = Pose6d::Zero();
    Pose6d cartesian_wrench_limit =
        (Pose6d() << 30.0, 30.0, 30.0, 5.0, 5.0, 5.0).finished();
    double cartesian_joint_velocity_filter_cutoff_hz = 0.0;
    // Approximate task-space counterparts to the 0.2 rad joint-PD activation
    // discrepancy threshold: 0.2 rad over a ~0.5 m arm reach is ~0.1 m.
    double cartesian_tracking_error_translation_limit_m = 0.10;
    double cartesian_tracking_error_rotation_limit_rad = 0.7853981633974483;
    double cartesian_tracking_error_timeout_s = 0.25;
    double cartesian_task_torque_ramp_time_s = 0.02;
    double cartesian_joint_limit_margin_rad = 0.02;
    bool friction_compensation = false;
    bool joint_friction_compensation = false;
    VecDoF joint_friction_transition_velocity;
    VecDoF joint_friction_constant_term;
    VecDoF joint_friction_coulomb_coef;
    VecDoF joint_friction_viscous_coef;
    bool gripper_friction_compensation = false;
    double gripper_friction_transition_velocity = 0.0;
    double gripper_friction_constant_term = 0.0;
    double gripper_friction_coulomb_coef = 0.0;
    double gripper_friction_viscous_coef = 0.0;
    bool background_send_recv;
    bool shutdown_to_passive;
    // true: will set the arm to damping then passive mode when pressing `ctrl-C`. (recommended);
    //       pressing `ctrl-\` will directly kill the program so this process will be skipped
    // false: will keep the arm in the air when shutting down the controller (both `ctrl-\` and `ctrl-C`).
    //       X5 cannot be kept in the air.
    std::string interpolation_method; // "linear" or cubic Hermite
    double default_preview_time;      // The default value for preview time if the command has 0 timestamp

    ControllerConfig(std::string controller_type, VecDoF default_kp, VecDoF default_kd, double default_gripper_kp,
                     double default_gripper_kd, int over_current_cnt_max, double controller_dt,
                     bool gravity_compensation, bool background_send_recv, bool shutdown_to_passive,
                     std::string interpolation_method, double default_preview_time)
        : controller_type(controller_type), default_kp(default_kp), default_kd(default_kd),
          default_gripper_kp(default_gripper_kp), default_gripper_kd(default_gripper_kd),
          over_current_cnt_max(over_current_cnt_max), controller_dt(controller_dt),
          gravity_compensation(gravity_compensation), joint_friction_transition_velocity(VecDoF::Zero(default_kp.size())),
          joint_friction_constant_term(VecDoF::Zero(default_kp.size())),
          joint_friction_coulomb_coef(VecDoF::Zero(default_kp.size())),
          joint_friction_viscous_coef(VecDoF::Zero(default_kp.size())), background_send_recv(background_send_recv),
          shutdown_to_passive(shutdown_to_passive), interpolation_method(interpolation_method),
          default_preview_time(default_preview_time)
    {
    }
};

bool load_friction_compensation_from_urdf(const RobotConfig &robot_config, ControllerConfig &controller_config);

class ControllerConfigFactory
{
  public:
    static ControllerConfigFactory &get_instance()
    {
        static ControllerConfigFactory instance;
        return instance;
    }

    ControllerConfig get_config(const std::string &controller_type, const int joint_dof)
    {

        auto it = configurations.find(controller_type + "_" + std::to_string(joint_dof));
        if (it != configurations.end())
        {
            return *(it->second);
        }
        else
        {
            throw std::runtime_error(
                "Unknown controller type. Currently available: joint_controller, cartesian_controller");
        }
    }

    ControllerConfig get_config(const std::string &controller_type, const RobotConfig &robot_config)
    {
        ControllerConfig config = get_config(controller_type, robot_config.joint_dof);
        // Calibrated friction feedforward is a default of the joint-PD
        // controller only. Cartesian/hybrid experiments opt in explicitly so
        // their host task-space torque is not silently changed.
        if (controller_type == "joint_controller")
            load_friction_compensation_from_urdf(robot_config, config);
        return config;
    }

  private:
    ControllerConfigFactory()
    {
        configurations["joint_controller_7"] = std::make_shared<ControllerConfig>(
            "joint_controller",                                                 // controller_type
            (VecDoF(7) << 80.0, 70.0, 70.0, 70.0, 30.0, 30.0, 20.0).finished(), // default_kp
            (VecDoF(7) << 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.7).finished(),        // default_kd
            5.0,                                                                // default_gripper_kp
            0.2,                                                                // default_gripper_kd
            20,                                                                 // over_current_cnt_max
            0.002,                                                              // controller_dt
            true,                                                               // gravity_compensation
            true,                                                               // background_send_recv
            true,                                                               // shutdown_to_passive
            "linear",                                                           // interpolation_method
            0.0                                                                 // default_preview_time
        );
        configurations["joint_controller_6"] = std::make_shared<ControllerConfig>(
            "joint_controller",                                           // controller_type
            (VecDoF(6) << 80.0, 70.0, 70.0, 30.0, 30.0, 20.0).finished(), // default_kp
            (VecDoF(6) << 2.0, 2.0, 2.0, 1.0, 1.0, 0.7).finished(),       // default_kd
            5.0,                                                          // default_gripper_kp
            0.2,                                                          // default_gripper_kd
            20,                                                           // over_current_cnt_max
            0.002,                                                        // controller_dt
            true,                                                         // gravity_compensation
            true,                                                         // background_send_recv
            true,                                                         // shutdown_to_passive
            "linear",                                                     // interpolation_method
            0.0                                                           // default_preview_time
        );
        configurations["cartesian_controller_7"] = std::make_shared<ControllerConfig>(
            "cartesian_controller",                                                 // controller_type
            (VecDoF(7) << 300.0, 300.0, 300.0, 300.0, 80.0, 50.0, 40.0).finished(), // default_kp
            (VecDoF(7) << 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0).finished(),            // default_kd
            5.0,                                                                    // default_gripper_kp
            0.2,                                                                    // default_gripper_kd
            20,                                                                     // over_current_cnt_max
            0.002,                                                                  // controller_dt
            true,                                                                   // gravity_compensation
            true,                                                                   // background_send_recv
            true,                                                                   // shutdown_to_passive
            "linear",                                                               // interpolation_method
            0.1                                                                     // default_preview_time
        );
        configurations["cartesian_controller_6"] = std::make_shared<ControllerConfig>(
            "cartesian_controller",                                           // controller_type
            (VecDoF(6) << 200.0, 200.0, 200.0, 120.0, 80.0, 60.0).finished(), // default_kp
            (VecDoF(6) << 5.0, 5.0, 5.0, 1.0, 1.0, 1.0).finished(),           // default_kd
            5.0,                                                              // default_gripper_kp
            0.2,                                                              // default_gripper_kd
            20,                                                               // over_current_cnt_max
            0.002,                                                            // controller_dt
            true,                                                             // gravity_compensation
            true,                                                             // background_send_recv
            true,                                                             // shutdown_to_passive
            "linear",                                                         // interpolation_method
            0.1                                                               // default_preview_time
        );
    }
    std::unordered_map<std::string, std::shared_ptr<ControllerConfig>> configurations;

    // Disable copy constructor and assignment operator
    ControllerConfigFactory(const ControllerConfigFactory &) = delete;
    ControllerConfigFactory &operator=(const ControllerConfigFactory &) = delete;
};
} // namespace arx
#endif // CONFIG_H
