import unittest

import numpy as np

import arx5_interface as arx5


class CartesianPDBindingTest(unittest.TestCase):
    def test_host_joint_damping_kernel_and_binding(self):
        velocity = np.array([1.0, -2.0, 0.5, 0.0, -0.25, 3.0])
        damping = np.array([2.0, 1.0, 4.0, 3.0, 0.5, 0.25])
        np.testing.assert_allclose(
            arx5._compute_host_joint_damping_torque(velocity, damping),
            -damping * velocity,
        )
        self.assertTrue(hasattr(arx5.Arx5JointController, "_set_host_joint_damping"))

    def test_factory_defaults(self) -> None:
        config = arx5.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", 6
        )
        np.testing.assert_array_equal(config.default_cartesian_kp, np.zeros(6))
        np.testing.assert_array_equal(config.default_cartesian_kd, np.zeros(6))
        np.testing.assert_array_equal(
            config.cartesian_wrench_limit, [30, 30, 30, 5, 5, 5]
        )
        self.assertEqual(
            config.cartesian_joint_velocity_filter_cutoff_hz, 0.0
        )
        self.assertEqual(config.cartesian_tracking_error_translation_limit_m, 0.10)
        self.assertAlmostEqual(
            config.cartesian_tracking_error_rotation_limit_rad,
            np.deg2rad(45.0),
        )
        self.assertEqual(config.cartesian_tracking_error_timeout_s, 0.25)
        self.assertEqual(config.cartesian_task_torque_ramp_time_s, 0.02)
        self.assertEqual(config.cartesian_joint_limit_margin_rad, 0.02)

        for model in ("L5", "Trossen_L5"):
            robot = arx5.RobotConfigFactory.get_instance().get_config(model)
            model_config = (
                arx5.ControllerConfigFactory.get_instance().get_config(
                    "cartesian_controller", robot
                )
            )
            self.assertEqual(
                model_config.cartesian_joint_velocity_filter_cutoff_hz,
                0.0,
            )
            self.assertFalse(model_config.friction_compensation)
            self.assertFalse(model_config.joint_friction_compensation)
            self.assertFalse(model_config.gripper_friction_compensation)

            joint_config = (
                arx5.ControllerConfigFactory.get_instance().get_config(
                    "joint_controller", robot
                )
            )
            self.assertTrue(joint_config.friction_compensation)
            self.assertTrue(joint_config.joint_friction_compensation)
            self.assertTrue(joint_config.gripper_friction_compensation)

        trossen_l5 = arx5.RobotConfigFactory.get_instance().get_config(
            "Trossen_L5"
        )
        self.assertAlmostEqual(trossen_l5.gripper_width, 0.056)
        self.assertAlmostEqual(trossen_l5.gripper_open_readout, 3.03)
        np.testing.assert_array_equal(
            trossen_l5.ee_vel_max, [2.0, 2.0, 2.0, 9.0, 9.0, 9.0]
        )

        x5 = arx5.RobotConfigFactory.get_instance().get_config("X5")
        x5_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", x5
        )
        self.assertEqual(
            x5_config.cartesian_joint_velocity_filter_cutoff_hz, 0.0
        )

    def test_l5_velocity_scaling_config(self) -> None:
        expected = np.array([10 / 45] * 3 + [30 / 45] * 3)
        for model in ("L5", "Trossen_L5"):
            config = arx5.RobotConfigFactory.get_instance().get_config(model)
            np.testing.assert_allclose(
                config.joint_velocity_feedback_scale, expected
            )
            self.assertAlmostEqual(
                config.gripper_velocity_feedback_scale, 30 / 45
            )

        x5 = arx5.RobotConfigFactory.get_instance().get_config("X5")
        np.testing.assert_array_equal(
            x5.joint_velocity_feedback_scale, np.ones(6)
        )
        self.assertEqual(x5.gripper_velocity_feedback_scale, 1.0)

    def test_gain_validation(self) -> None:
        with self.assertRaises(ValueError):
            arx5.CartesianPDGain(
                np.array([-1, 0, 0, 0, 0, 0], dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            arx5.CartesianPDGain(
                np.zeros(6, dtype=np.float64),
                np.array([0, 0, np.nan, 0, 0, 0], dtype=np.float64),
            )

    def test_wrench_limit_validation(self) -> None:
        config = arx5.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", 6
        )
        with self.assertRaises(ValueError):
            config.cartesian_wrench_limit = np.array(
                [30, 30, -1, 5, 5, 5], dtype=np.float64
            )
        with self.assertRaises(ValueError):
            config.default_cartesian_kp = np.array(
                [100, 100, 100, np.inf, 10, 10], dtype=np.float64
            )
        for cutoff_hz in (-1.0, 250.0, np.inf):
            with self.assertRaises(ValueError):
                config.cartesian_joint_velocity_filter_cutoff_hz = cutoff_hz
        for field in (
            "cartesian_tracking_error_translation_limit_m",
            "cartesian_tracking_error_rotation_limit_rad",
            "cartesian_tracking_error_timeout_s",
            "cartesian_task_torque_ramp_time_s",
            "cartesian_joint_limit_margin_rad",
        ):
            with self.assertRaises(ValueError):
                setattr(config, field, 0.0)

    def test_controller_api_is_bound(self) -> None:
        for method in (
            "set_cartesian_pd_gain",
            "get_cartesian_pd_gain",
            "set_cartesian_damping_mode",
            "get_cartesian_damping_mode",
            "set_cartesian_motor_kd",
            "get_cartesian_motor_kd",
            "set_joint_cmd",
            "set_joint_traj",
            "set_cartesian_joint_velocity_filter_cutoff",
            "get_cartesian_joint_velocity_filter_cutoff",
            "get_filtered_joint_velocity",
            "enable_cartesian_pd",
            "disable_cartesian_pd",
            "is_cartesian_pd_enabled",
            "set_eef_cmd_with_twist",
            "set_eef_traj_with_twists",
            "get_eef_cmd_twist",
        ):
            self.assertTrue(hasattr(arx5.Arx5CartesianController, method))
        self.assertEqual(int(arx5.CartesianDampingMode.HostCartesian), 0)
        self.assertEqual(int(arx5.CartesianDampingMode.MotorLocalJoint), 1)

    def test_damped_pseudoinverse_binding_limits_uniformly(self) -> None:
        unlimited, limited, scale, valid = arx5._compute_damped_cartesian_joint_velocity(
            np.eye(6), np.array([1, 2, 3, 4, 5, 6], dtype=np.float64), np.ones(6)
        )
        self.assertTrue(valid)
        self.assertLess(scale, 1.0)
        np.testing.assert_allclose(limited, unlimited * scale)
        self.assertLessEqual(np.max(np.abs(limited)), 1.0 + 1e-12)

    def test_simulation_hooks_use_cpp_kernel(self) -> None:
        robot = arx5.RobotConfigFactory.get_instance().get_config("L5")
        kinematics = arx5._CartesianKinematics(
            robot.urdf_path,
            robot.joint_dof,
            robot.base_link_name,
            robot.eef_link_name,
        )
        q = np.array([0.0, 0.8, 1.0, -0.4, 0.3, 0.0], dtype=np.float64)
        jacobian = kinematics.jacobian(q)
        pose = kinematics.pose(q)
        np.testing.assert_equal(jacobian.shape, (6, 6))
        np.testing.assert_equal(pose.shape, (6,))

        gain = arx5.CartesianPDGain(
            np.ones(6, dtype=np.float64), np.zeros(6, dtype=np.float64)
        )
        pose_error = np.zeros(6, dtype=np.float64)
        pose_error[0] = 100.0
        torque, wrench = arx5._compute_cartesian_pd_torque(
            jacobian,
            pose_error,
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            gain,
            np.ones(6, dtype=np.float64),
        )
        self.assertEqual(np.asarray(torque).shape, (6,))
        np.testing.assert_array_equal(wrench, [1, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
