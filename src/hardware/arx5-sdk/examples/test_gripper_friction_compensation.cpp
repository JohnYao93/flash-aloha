#include "app/controller_base.h"
#include "app/config.h"
#include "app/common.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace arx;

namespace arx
{
class Arx5ControllerBaseFrictionTest
{
  public:
    static VecDoF compute_joint(const RobotConfig &robot_config, const ControllerConfig &controller_config,
                                const VecDoF &joint_velocity)
    {
        return Arx5ControllerBase::compute_joint_friction_adjustment_(robot_config, controller_config, joint_velocity);
    }

    static double compute_gripper(const RobotConfig &robot_config, const ControllerConfig &controller_config,
                                  double gripper_velocity)
    {
        return Arx5ControllerBase::compute_gripper_friction_adjustment_(robot_config, controller_config,
                                                                        gripper_velocity);
    }
};
} // namespace arx

namespace
{
void expect_near(double actual, double expected)
{
    assert(std::abs(actual - expected) < 1e-12);
}

void expect_vec_near(const VecDoF &actual, const VecDoF &expected)
{
    assert(actual.size() == expected.size());
    assert((actual - expected).cwiseAbs().maxCoeff() < 1e-12);
}
} // namespace

int main()
{
    RobotConfig robot_config = RobotConfigFactory::get_instance().get_config("L5");
    ControllerConfig controller_config =
        ControllerConfigFactory::get_instance().get_config("joint_controller", robot_config.joint_dof);
    ControllerConfig l5_model_config = ControllerConfigFactory::get_instance().get_config("joint_controller", robot_config);
    RobotConfig x5_robot_config = RobotConfigFactory::get_instance().get_config("X5");
    ControllerConfig x5_model_config = ControllerConfigFactory::get_instance().get_config("joint_controller", x5_robot_config);

    assert(!controller_config.friction_compensation);
    assert(!controller_config.joint_friction_compensation);
    assert(!controller_config.gripper_friction_compensation);
    assert(l5_model_config.friction_compensation);
    assert(l5_model_config.joint_friction_compensation);
    assert(l5_model_config.gripper_friction_compensation);
    expect_vec_near(l5_model_config.joint_friction_transition_velocity,
                    (VecDoF(6) << 0.005, 0.005, 0.01, 0.015, 0.02, 0.019).finished());
    expect_vec_near(l5_model_config.joint_friction_coulomb_coef,
                    (VecDoF(6) << 0.1, 0.1, 0.15, 0.05, 0.04, 0.05).finished());
    assert(!x5_model_config.friction_compensation);
    assert(!x5_model_config.joint_friction_compensation);
    assert(!x5_model_config.gripper_friction_compensation);

    controller_config.joint_friction_transition_velocity = VecDoF::Zero(robot_config.joint_dof);
    controller_config.joint_friction_constant_term = VecDoF::Zero(robot_config.joint_dof);
    controller_config.joint_friction_coulomb_coef = VecDoF::Zero(robot_config.joint_dof);
    controller_config.joint_friction_viscous_coef = VecDoF::Zero(robot_config.joint_dof);
    controller_config.joint_friction_transition_velocity[0] = 0.2;
    controller_config.joint_friction_transition_velocity[1] = 0.2;
    controller_config.joint_friction_constant_term[0] = 0.1;
    controller_config.joint_friction_constant_term[1] = 0.1;
    controller_config.joint_friction_coulomb_coef[0] = 0.3;
    controller_config.joint_friction_coulomb_coef[1] = 0.3;
    controller_config.joint_friction_viscous_coef[0] = 0.5;
    controller_config.joint_friction_viscous_coef[1] = 0.5;

    VecDoF joint_velocity = VecDoF::Zero(robot_config.joint_dof);
    joint_velocity[0] = 0.2;
    joint_velocity[1] = -0.2;
    expect_vec_near(Arx5ControllerBaseFrictionTest::compute_joint(robot_config, controller_config, joint_velocity),
                    VecDoF::Zero(robot_config.joint_dof));

    controller_config.joint_friction_compensation = true;
    expect_vec_near(Arx5ControllerBaseFrictionTest::compute_joint(robot_config, controller_config, joint_velocity),
                    VecDoF::Zero(robot_config.joint_dof));
    controller_config.friction_compensation = true;
    VecDoF joint_expected = VecDoF::Zero(robot_config.joint_dof);
    joint_expected[0] = 0.1 + 0.3 * std::tanh(1.0) + 0.5 * 0.2;
    joint_expected[1] = 0.1 - 0.3 * std::tanh(1.0) - 0.5 * 0.2;
    expect_vec_near(Arx5ControllerBaseFrictionTest::compute_joint(robot_config, controller_config, joint_velocity),
                    joint_expected);

    controller_config.joint_friction_transition_velocity[0] = 0.0;
    joint_expected[0] = 0.0;
    expect_vec_near(Arx5ControllerBaseFrictionTest::compute_joint(robot_config, controller_config, joint_velocity),
                    joint_expected);

    controller_config.gripper_friction_transition_velocity = 0.2;
    controller_config.gripper_friction_constant_term = 0.1;
    controller_config.gripper_friction_coulomb_coef = 0.3;
    controller_config.gripper_friction_viscous_coef = 0.5;

    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), 0.0);

    controller_config.friction_compensation = false;
    controller_config.gripper_friction_compensation = true;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), 0.0);
    controller_config.friction_compensation = true;
    controller_config.gripper_friction_transition_velocity = 0.0;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), 0.0);

    controller_config.gripper_friction_transition_velocity = 0.2;
    controller_config.gripper_friction_constant_term = 0.0;
    controller_config.gripper_friction_coulomb_coef = 0.0;
    controller_config.gripper_friction_viscous_coef = 0.0;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), 0.0);

    controller_config.gripper_friction_constant_term = 0.1;
    controller_config.gripper_friction_coulomb_coef = 0.3;
    controller_config.gripper_friction_viscous_coef = 0.5;
    const double positive_expected = 0.1 + 0.3 * std::tanh(1.0) + 0.5 * 0.2;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), positive_expected);

    const double negative_expected = 0.1 - 0.3 * std::tanh(1.0) - 0.5 * 0.2;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, -0.2), negative_expected);

    robot_config.gripper_motor_type = MotorType::NONE;
    expect_near(Arx5ControllerBaseFrictionTest::compute_gripper(robot_config, controller_config, 0.2), 0.0);

    std::cout << "friction compensation checks passed" << std::endl;
    return 0;
}
