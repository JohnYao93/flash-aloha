import arx5_interface as arx5

# Get the controller configuration for your robot
robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_config)

# Check default gains
print("Default Kp:", joint_controller_config.default_kp)
print("Default Kd:", joint_controller_config.default_kd)
print("Default Gripper Kp:", joint_controller_config.default_gripper_kp)
print("Default Gripper Kd:", joint_controller_config.default_gripper_kd)
scale_kp = 0.7
scaled_kp = joint_controller_config.default_kp * scale_kp
print(f"Scaled Kp (x{scale_kp}):", scaled_kp)
scale_gripper_kp = 0.5
scaled_gripper_kp = joint_controller_config.default_gripper_kp * scale_gripper_kp
print(f"Scaled Gripper Kp (x{scale_gripper_kp}):", scaled_gripper_kp)
c = 0.5
ideal_kd = (scaled_kp ** 0.5) * c
print("Ideal Kd for critical damping (c=2.0):", ideal_kd)
ideal_gripper_kd = (scaled_gripper_kp ** 0.5) * c
print("Ideal Gripper Kd for critical damping (c=2.0):", ideal_gripper_kd)
