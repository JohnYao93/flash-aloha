#include "app/config.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <tinyxml2.h>
#include <vector>

namespace arx
{
namespace
{
const char *FRICTION_TAG = "arx5_friction_compensation";

std::runtime_error friction_config_error(const RobotConfig &robot_config, const std::string &message)
{
    return std::runtime_error("Invalid friction compensation in " + robot_config.urdf_path + ": " + message);
}

bool required_bool(const tinyxml2::XMLElement &element, const char *name, const RobotConfig &robot_config)
{
    bool value = false;
    if (element.QueryBoolAttribute(name, &value) != tinyxml2::XML_SUCCESS)
        throw friction_config_error(robot_config,
                                    std::string("<") + element.Name() + "> requires boolean attribute '" + name + "'");
    return value;
}

int required_int(const tinyxml2::XMLElement &element, const char *name, const RobotConfig &robot_config)
{
    int value = 0;
    if (element.QueryIntAttribute(name, &value) != tinyxml2::XML_SUCCESS)
        throw friction_config_error(robot_config,
                                    std::string("<") + element.Name() + "> requires integer attribute '" + name + "'");
    return value;
}

double required_double(const tinyxml2::XMLElement &element, const char *name, const RobotConfig &robot_config)
{
    double value = 0.0;
    if (element.QueryDoubleAttribute(name, &value) != tinyxml2::XML_SUCCESS || !std::isfinite(value))
        throw friction_config_error(robot_config,
                                    std::string("<") + element.Name() + "> requires finite attribute '" + name + "'");
    return value;
}

void read_terms(const tinyxml2::XMLElement &element, const RobotConfig &robot_config, double &transition_velocity,
                double &constant_term, double &coulomb_coef, double &viscous_coef)
{
    transition_velocity = required_double(element, "transition_velocity", robot_config);
    constant_term = required_double(element, "constant_term", robot_config);
    coulomb_coef = required_double(element, "coulomb_coef", robot_config);
    viscous_coef = required_double(element, "viscous_coef", robot_config);
}
} // namespace

bool load_friction_compensation_from_urdf(const RobotConfig &robot_config, ControllerConfig &controller_config)
{
    tinyxml2::XMLDocument document;
    const tinyxml2::XMLError load_result = document.LoadFile(robot_config.urdf_path.c_str());
    if (load_result != tinyxml2::XML_SUCCESS)
        throw friction_config_error(robot_config, document.ErrorStr() ? document.ErrorStr() : "unable to read URDF");

    const tinyxml2::XMLElement *robot = document.FirstChildElement("robot");
    if (!robot)
        throw friction_config_error(robot_config, "missing <robot> root element");

    const tinyxml2::XMLElement *friction = robot->FirstChildElement(FRICTION_TAG);
    if (!friction)
        return false;
    if (friction->NextSiblingElement(FRICTION_TAG))
        throw friction_config_error(robot_config, std::string("multiple <") + FRICTION_TAG + "> elements");

    ControllerConfig loaded = controller_config;
    loaded.friction_compensation = required_bool(*friction, "enabled", robot_config);
    loaded.joint_friction_transition_velocity = VecDoF::Zero(robot_config.joint_dof);
    loaded.joint_friction_constant_term = VecDoF::Zero(robot_config.joint_dof);
    loaded.joint_friction_coulomb_coef = VecDoF::Zero(robot_config.joint_dof);
    loaded.joint_friction_viscous_coef = VecDoF::Zero(robot_config.joint_dof);

    std::vector<bool> seen_joint(robot_config.joint_dof, false);
    bool any_joint_enabled = false;
    for (const tinyxml2::XMLElement *joint = friction->FirstChildElement("joint"); joint;
         joint = joint->NextSiblingElement("joint"))
    {
        const int index = required_int(*joint, "index", robot_config);
        if (index < 0 || index >= robot_config.joint_dof)
            throw friction_config_error(robot_config, "joint friction index is outside robot joint_dof");
        if (seen_joint[index])
            throw friction_config_error(robot_config, "duplicate joint friction index " + std::to_string(index));
        seen_joint[index] = true;

        const bool enabled = required_bool(*joint, "enabled", robot_config);
        double transition_velocity = 0.0;
        double constant_term = 0.0;
        double coulomb_coef = 0.0;
        double viscous_coef = 0.0;
        read_terms(*joint, robot_config, transition_velocity, constant_term, coulomb_coef, viscous_coef);
        if (enabled && transition_velocity <= 0.0)
            throw friction_config_error(robot_config,
                                        "enabled joint " + std::to_string(index) + " requires transition_velocity > 0");

        if (enabled)
        {
            any_joint_enabled = true;
            loaded.joint_friction_transition_velocity[index] = transition_velocity;
            loaded.joint_friction_constant_term[index] = constant_term;
            loaded.joint_friction_coulomb_coef[index] = coulomb_coef;
            loaded.joint_friction_viscous_coef[index] = viscous_coef;
        }
    }

    for (int index = 0; index < robot_config.joint_dof; ++index)
    {
        if (!seen_joint[index])
            throw friction_config_error(robot_config, "missing joint friction index " + std::to_string(index));
    }
    loaded.joint_friction_compensation = any_joint_enabled;

    const tinyxml2::XMLElement *gripper = friction->FirstChildElement("gripper");
    if (!gripper)
        throw friction_config_error(robot_config, "missing <gripper> friction element");
    if (gripper->NextSiblingElement("gripper"))
        throw friction_config_error(robot_config, "multiple <gripper> friction elements");

    loaded.gripper_friction_compensation = required_bool(*gripper, "enabled", robot_config);
    read_terms(*gripper, robot_config, loaded.gripper_friction_transition_velocity,
               loaded.gripper_friction_constant_term, loaded.gripper_friction_coulomb_coef,
               loaded.gripper_friction_viscous_coef);
    if (loaded.gripper_friction_compensation && loaded.gripper_friction_transition_velocity <= 0.0)
        throw friction_config_error(robot_config, "enabled gripper requires transition_velocity > 0");

    controller_config = loaded;
    return true;
}
} // namespace arx
