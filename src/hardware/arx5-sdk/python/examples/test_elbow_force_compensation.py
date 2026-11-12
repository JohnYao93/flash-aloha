import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--urdf_path", "-u", default="../models/arx5.urdf", help="URDF file path")
def main(model: str, interface: str, urdf_path: str):

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config
    )
    controller_config.gravity_compensation = True

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )
    dof = robot_config.joint_dof
    gain = arx5.Gain(dof)
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    
    arx5_joint_controller.reset_to_home()
    arx5_joint_controller.set_gain(gain)
    
    elbow_acc = 0.0
    # last_elbow_vel = 0.0
    elbow_vels = [0.0] * 5
    dt = 0.005
    acc_threshold = 0.005
    torque_to_acc = 0.015

    
    while True:
        t1 = time.time()
        joint_state = arx5_joint_controller.get_joint_state()
        elbow_vels.append(joint_state.vel()[2])  # elbow is the 3rd joint (index 2)
        elbow_vels.pop(0)
        elbow_acc = 1/2 *(elbow_vels[-1] - elbow_vels[0]) / (dt * (len(elbow_vels) - 1)) +\
             1/2 * (elbow_vels[-2] - elbow_vels[2]) / (dt * (len(elbow_vels) - 3))
        if elbow_acc > acc_threshold:
            elbow_acc -= acc_threshold
        elif elbow_acc < -acc_threshold:
            elbow_acc += acc_threshold
        else:
            elbow_acc = 0.0

        assist_torque = torque_to_acc * elbow_acc
        print(
            f"elbow pos: {joint_state.pos()[2]:+.3f}, "
            f"vel: {joint_state.vel()[2]:+.3f}, "
            f"acc: {elbow_acc:+.3f}, "
            f"measured torque: {joint_state.torque()[2]:+.3f}, "
            f"assist torque: {assist_torque:+.3f}"
        )
        joint_cmd = arx5.JointState(dof)
        
        joint_cmd.torque()[2] = assist_torque
        arx5_joint_controller.set_joint_cmd(joint_cmd)
        t2 = time.time()
        print(f"loop time: {(t2 - t1) * 1000:.2f} ms")
        time.sleep(max(0, dt - (t2 - t1)))
    
    
    
if __name__ == "__main__":
    main()
