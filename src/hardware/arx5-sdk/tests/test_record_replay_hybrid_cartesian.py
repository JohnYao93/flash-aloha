import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from record_replay_hybrid_cartesian import (  # noqa: E402
    build_side_by_side_comparison,
    build_velocity_limit_report,
    calculate_fk_reference,
    capture_joint_replay_row,
    joint_replay_telemetry_columns,
    main,
    format_velocity_limit_report,
    prepare_recorded_joint_trajectory,
    project_cartesian_stiffness_along_path,
    replay_telemetry_columns,
    scale_joint_reference_for_limits,
    validate_six,
    write_side_by_side_comparison,
)


class HybridTeachReplayTest(unittest.TestCase):
    def test_velocity_limit_report_is_prominent_and_per_axis(self):
        joint_velocities = np.zeros((3, 6))
        joint_velocities[1, 0] = 6.0
        cartesian_twists = np.zeros((3, 6))
        cartesian_twists[1, 1] = 1.25
        report = build_velocity_limit_report(
            joint_velocities,
            np.full(6, 5.0),
            cartesian_twists,
            np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0]),
        )
        self.assertTrue(report["exceeded"])
        self.assertEqual(report["time_scale"], 1.25)
        message = format_velocity_limit_report(report)
        self.assertIn("VELOCITY LIMIT EXCEEDED", message)
        self.assertIn("Joint J1", message)
        self.assertIn("Cartesian Y", message)
        self.assertIn("1.250x", message)

    def test_velocity_limit_report_pass_message(self):
        report = build_velocity_limit_report(
            np.zeros((3, 6)), np.ones(6)
        )
        self.assertFalse(report["exceeded"])
        self.assertIn("RECORDED TIMING IS PRESERVED", format_velocity_limit_report(report))

    def test_prepare_recorded_joint_trajectory_preserves_position_knots(self):
        times = np.array([4.0, 4.009, 4.021])
        positions = np.outer(np.array([0.0, 0.5, 1.0]), np.arange(1, 7))
        replay_times, replay_positions, replay_velocities = (
            prepare_recorded_joint_trajectory(times, positions)
        )
        np.testing.assert_allclose(replay_times, [0.0, 0.009, 0.021])
        np.testing.assert_array_equal(replay_positions, positions)
        np.testing.assert_array_equal(replay_velocities[0], np.zeros(6))
        np.testing.assert_array_equal(replay_velocities[-1], np.zeros(6))

    def test_prepare_recorded_trajectory_deduplicates_controller_timestamps(self):
        times = np.array([4.0, 4.5, 4.5, 5.0])
        positions = np.zeros((4, 6))
        positions[:, 0] = [0.0, 0.5, 0.5, 1.0]
        replay_times, replay_positions, _ = prepare_recorded_joint_trajectory(
            times, positions
        )
        np.testing.assert_allclose(replay_times, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(replay_positions[:, 0], [0.0, 0.5, 1.0])

    def test_fk_reference_scales_time_and_twist_uniformly(self):
        class FakeKinematics:
            def jacobian(self, position):
                return np.eye(6)

            def pose(self, position):
                return np.asarray(position)

        times = np.array([0.0, 0.5, 1.0])
        positions = np.zeros((3, 6))
        positions[:, 0] = [0.0, 1.0, 2.0]
        velocities = np.zeros((3, 6))
        velocities[1, 0] = 2.0
        scaled_times, poses, twists, scale = calculate_fk_reference(
            FakeKinematics(),
            times,
            positions,
            velocities,
            np.ones(6),
            np.full(6, 0.5),
        )
        self.assertEqual(scale, 4.0)
        np.testing.assert_allclose(scaled_times, times * 4.0)
        np.testing.assert_allclose(poses, positions)
        self.assertEqual(twists[1, 0], 0.5)
        np.testing.assert_array_equal(twists[0], np.zeros(6))
        np.testing.assert_array_equal(twists[-1], np.zeros(6))

    def test_joint_reference_scales_time_and_velocity_uniformly(self):
        times = np.array([0.0, 0.5, 1.0])
        positions = np.zeros((3, 6))
        velocities = np.zeros((3, 6))
        velocities[1, 0] = 2.0
        scaled_times, scaled_positions, scaled_velocities, scale = (
            scale_joint_reference_for_limits(
                times,
                positions,
                velocities,
                np.ones(6),
            )
        )
        self.assertEqual(scale, 2.0)
        np.testing.assert_allclose(scaled_times, times * 2.0)
        np.testing.assert_array_equal(scaled_positions, positions)
        self.assertEqual(scaled_velocities[1, 0], 1.0)
        np.testing.assert_array_equal(scaled_velocities[0], np.zeros(6))
        np.testing.assert_array_equal(scaled_velocities[-1], np.zeros(6))

    def test_joint_replay_telemetry_uses_atomic_command_and_feedback(self):
        class Telemetry:
            sequence = 42
            cartesian_pd_enabled = False
            final_joint_position_command = np.arange(6, dtype=float) + 10
            feedback_joint_position = np.arange(6, dtype=float) + 20
            final_joint_velocity_command = np.arange(6, dtype=float) + 30
            feedback_joint_velocity = np.arange(6, dtype=float) + 40
            feedback_joint_torque = np.arange(6, dtype=float) + 50
            gravity_torque = np.arange(6, dtype=float) + 60
            final_joint_torque_command = np.arange(6, dtype=float) + 70
            final_motor_kp = np.arange(6, dtype=float) + 80
            final_motor_kd = np.arange(6, dtype=float) + 90
            final_motor_protocol_velocity = np.arange(6, dtype=float) + 100

        class Controller:
            calls = 0

            def _get_control_cycle_telemetry(self):
                self.calls += 1
                return Telemetry()

        controller = Controller()
        row = capture_joint_replay_row(controller, 0.25)
        columns = joint_replay_telemetry_columns()
        self.assertEqual(controller.calls, 1)
        self.assertEqual(len(row), len(columns))
        values = dict(zip(columns, row))
        self.assertEqual(values["elapsed_s"], 0.25)
        self.assertEqual(values["sequence"], 42)
        self.assertEqual(values["cartesian_pd_enabled"], 0)
        self.assertEqual(values["desired_joint_position_1"], 10)
        self.assertEqual(values["measured_joint_position_6"], 25)
        self.assertEqual(values["packed_motor_kd_6"], 95)

    def test_cartesian_stiffness_projection_reports_coupled_matrix_and_diagonal(self):
        class FakeKinematics:
            def jacobian(self, position):
                return np.diag(np.asarray(position, dtype=float))

        positions = np.array([[1, 2, 3, 4, 5, 6], [2, 2, 3, 4, 5, 6]], dtype=float)
        result = project_cartesian_stiffness_along_path(
            FakeKinematics(), positions, np.ones(6)
        )
        np.testing.assert_allclose(result["start_diagonal"], np.square(positions[0]))
        np.testing.assert_allclose(
            result["median_diagonal"],
            np.median(np.square(positions), axis=0),
        )

    def test_side_by_side_comparison_uses_common_reference_times(self):
        class FakeKinematics:
            def pose(self, position):
                return np.asarray(position, dtype=float)

        times = np.array([0.0, 0.5, 1.0])
        reference_joints = np.zeros((3, 6))
        reference_joints[:, 0] = times
        reference_poses = reference_joints.copy()
        hybrid_columns = replay_telemetry_columns()
        joint_columns = joint_replay_telemetry_columns()

        def row(columns, values):
            result = np.zeros(len(columns))
            index = {name: offset for offset, name in enumerate(columns)}
            for name, value in values.items():
                result[index[name]] = value
            return result.tolist()

        hybrid_rows = []
        joint_rows = []
        for elapsed, reference in zip(times + 1.0, reference_joints):
            hybrid_values = {"elapsed_s": elapsed}
            joint_values = {"elapsed_s": elapsed}
            for joint in range(1, 7):
                hybrid_values[f"joint_position_{joint}"] = reference[joint - 1] + 0.01
                joint_values[f"measured_joint_position_{joint}"] = reference[joint - 1] + 0.002
            for axis, value in zip(("x", "y", "z", "rx", "ry", "rz"), reference):
                hybrid_values[f"measured_pose_{axis}"] = value + 0.001
            hybrid_rows.append(row(hybrid_columns, hybrid_values))
            joint_rows.append(row(joint_columns, joint_values))

        summary, values = build_side_by_side_comparison(
            FakeKinematics(),
            times,
            reference_joints,
            reference_poses,
            hybrid_rows,
            1.0,
            joint_rows,
            1.0,
        )
        self.assertAlmostEqual(
            summary["hybrid-cartesian"]["translation_error_mm"]["rms"],
            np.sqrt(3.0),
        )
        self.assertGreater(
            summary["joint"]["translation_error_mm"]["rms"],
            summary["hybrid-cartesian"]["translation_error_mm"]["rms"],
        )
        stiffness = {
            "configured_cartesian_kp": {"median_diagonal": [1] * 6},
            "x_axis_only": {"median_diagonal": [0.5] * 6},
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_side_by_side_comparison(
                Path(directory) / "comparison", summary, values, stiffness
            )
            self.assertTrue(all(path.exists() for path in paths))

    def test_validation_and_cli_surface(self):
        with self.assertRaisesRegex(ValueError, "six finite"):
            validate_six([1, 2], "test")
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--cartesian-kp", result.output)
        self.assertIn("--motor-kd", result.output)
        self.assertIn("--replay-mode", result.output)
        self.assertIn("--trajectory-rate", result.output)
        self.assertIn("--record-rate", result.output)
        self.assertIn("100", result.output)
        self.assertNotIn("--replay-command-rate", result.output)
        self.assertIn("hybrid-cartesian", result.output)
        self.assertIn("joint", result.output)
        self.assertIn("comparison", result.output)
        self.assertIn("--joint-replay-kp", result.output)
        self.assertIn("--duration", result.output)
        self.assertIn("--start-joint-tolerance-rad", result.output)
        self.assertIn("--start-translation-tolerance-mm", result.output)
        self.assertIn("--start-rotation-tolerance-deg", result.output)


if __name__ == "__main__":
    unittest.main()
