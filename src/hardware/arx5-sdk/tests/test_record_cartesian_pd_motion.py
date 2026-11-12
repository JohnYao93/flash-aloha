import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from record_cartesian_pd_motion import (  # noqa: E402
    HOME_FINAL_SETTLE_S,
    LocalJointCommandGenerator,
    build_metadata,
    capture_row,
    main,
    project_cartesian_gains_to_joint_diagonal,
    recording_columns,
    return_home_after_recording,
    validate_paired_anchor,
    validate_motor_gain_vector,
    write_csv,
    write_html_report,
)


class CartesianPDRecordingTest(unittest.TestCase):
    def test_clean_recording_return_disables_cartesian_pd_then_homes(self) -> None:
        calls = []

        class FakeController:
            def disable_cartesian_pd(self):
                calls.append("disable")

            def reset_to_home(self):
                calls.append("home")

        return_home_after_recording(
            FakeController(), sleep_fn=lambda duration: calls.append(("sleep", duration))
        )
        self.assertEqual(
            calls,
            ["disable", "home", ("sleep", HOME_FINAL_SETTLE_S)],
        )

    def test_motor_local_return_enters_damping_then_homes(self) -> None:
        calls = []

        class FakeController:
            def set_to_damping(self):
                calls.append("damping")

            def reset_to_home(self):
                calls.append("home")

        return_home_after_recording(
            FakeController(),
            control_mode="motor-local",
            sleep_fn=lambda duration: calls.append(("sleep", duration)),
        )
        self.assertEqual(
            calls,
            ["damping", "home", ("sleep", HOME_FINAL_SETTLE_S)],
        )

    def test_hybrid_return_disables_cartesian_then_homes(self) -> None:
        calls = []

        class FakeController:
            def disable_cartesian_pd(self):
                calls.append("disable")

            def reset_to_home(self):
                calls.append("home")

        return_home_after_recording(
            FakeController(), control_mode="hybrid-cartesian", sleep_fn=lambda duration: None
        )
        self.assertEqual(calls, ["disable", "home"])

    def test_paired_anchor_joint_threshold_is_enforced(self) -> None:
        zero = np.zeros(6)
        validate_paired_anchor(zero, zero, np.full(6, 0.05), zero)
        with self.assertRaisesRegex(ValueError, "joint anchor"):
            validate_paired_anchor(zero, zero, np.array([0.051, 0, 0, 0, 0, 0]), zero)

    def test_cli_exposes_manual_positioning_controls(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--skip-positioning", result.output)
        self.assertIn("--axis", result.output)
        self.assertIn("--control-mode", result.output)
        self.assertIn("--motor-kd", result.output)
        self.assertIn("--paired-comparison", result.output)

    def test_paired_mode_refuses_positioning_bypass(self) -> None:
        result = CliRunner().invoke(
            main,
            ["Trossen_L5", "can2", "--axis", "y", "--paired-comparison", "--skip-positioning"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("requires manual positioning", result.output)

    def test_motor_gain_protocol_limits_are_validated(self) -> None:
        np.testing.assert_array_equal(
            validate_motor_gain_vector([1, 2, 3, 4, 5, 6], "Kp", 500.0),
            np.arange(1, 7),
        )
        with self.assertRaisesRegex(ValueError, "joint 6"):
            validate_motor_gain_vector([0, 0, 0, 0, 0, 5.1], "Kd", 5.0)

    def test_cartesian_gains_project_to_joint_diagonal(self) -> None:
        jacobian = np.diag([1, 2, 3, 4, 5, 6])
        joint_kp, joint_kd = project_cartesian_gains_to_joint_diagonal(
            jacobian, np.ones(6), 2.0 * np.ones(6)
        )
        np.testing.assert_allclose(joint_kp, np.square(np.arange(1, 7)))
        np.testing.assert_allclose(joint_kd, 2.0 * np.square(np.arange(1, 7)))

    def test_local_command_uses_ik_and_differential_velocity(self) -> None:
        class FakeSolver:
            def inverse_kinematics(self, pose, seed):
                np.testing.assert_array_equal(seed, np.zeros(6))
                return 0, np.asarray(pose) + 1.0

        class FakeKinematics:
            def jacobian(self, joint_position):
                return 2.0 * np.eye(6)

        generator = LocalJointCommandGenerator(
            FakeSolver(), FakeKinematics(), np.zeros(6), np.ones(6)
        )
        command = generator.make_command(
            np.zeros(6), np.full(6, 0.5), 0.02, 1.25
        )
        np.testing.assert_allclose(command.pos(), np.ones(6))
        np.testing.assert_allclose(command.vel(), np.full(6, 0.25))
        self.assertEqual(command.timestamp, 1.25)

    def test_columns_cover_cartesian_joint_and_timing_signals(self) -> None:
        columns = recording_columns()
        for expected in (
            "schedule_lateness_s",
            "control_cycle_sequence",
            "control_compute_end_s",
            "motor_snapshot_read_start_s",
            "feedback_publish_timestamp_s",
            "source_pose_rx",
            "desired_twist_rz",
            "measured_pose_x",
            "filtered_twist_z",
            "derivative_wrench_rx",
            "requested_cartesian_task_torque_4",
            "host_cartesian_damping_torque_4",
            "packed_motor_kp_1",
            "packed_motor_kd_6",
            "motor_protocol_velocity_3",
            "joint_position_6",
            "joint_velocity_command_3",
            "joint_torque_command_1",
            "joint_send_call_start_s_1",
            "joint_send_call_end_s_6",
        ):
            self.assertIn(expected, columns)
        self.assertEqual(len(columns), len(set(columns)))

    def test_capture_row_uses_one_atomic_controller_cycle(self) -> None:
        class Telemetry:
            sequence = 42
            cartesian_pd_enabled = True
            has_cartesian_data = True
            cycle_start_s = 1.000
            control_compute_start_s = 1.001
            control_compute_end_s = 1.002
            feedback_used_timestamp_s = 0.999
            motor_snapshot_read_start_s = 1.0030
            motor_snapshot_read_end_s = 1.0035
            feedback_publish_timestamp_s = 1.004
            cycle_end_s = 1.005
            desired_pose = np.arange(6, dtype=np.float64) + 10
            desired_twist = np.arange(6, dtype=np.float64) + 20
            measured_pose = np.arange(6, dtype=np.float64) + 30
            pose_error = np.arange(6, dtype=np.float64) + 40
            raw_twist = np.arange(6, dtype=np.float64) + 50
            filtered_twist = np.arange(6, dtype=np.float64) + 60
            proportional_wrench = np.arange(6, dtype=np.float64) + 70
            derivative_wrench = np.arange(6, dtype=np.float64) + 80
            clamped_wrench = np.arange(6, dtype=np.float64) + 90
            control_joint_position = np.arange(6, dtype=np.float64) + 100
            control_joint_velocity = np.arange(6, dtype=np.float64) + 110
            control_joint_torque = np.arange(6, dtype=np.float64) + 120
            feedback_joint_position = np.arange(6, dtype=np.float64) + 130
            feedback_joint_velocity = np.arange(6, dtype=np.float64) + 140
            filtered_joint_velocity = np.arange(6, dtype=np.float64) + 150
            feedback_joint_torque = np.arange(6, dtype=np.float64) + 160
            requested_cartesian_task_torque = np.arange(6, dtype=np.float64) + 170
            slewed_cartesian_task_torque = np.arange(6, dtype=np.float64) + 180
            final_joint_position_command = np.arange(6, dtype=np.float64) + 190
            final_joint_velocity_command = np.arange(6, dtype=np.float64) + 200
            final_joint_torque_command = np.arange(6, dtype=np.float64) + 210
            joint_send_call_start_s = np.arange(6, dtype=np.float64) + 220
            joint_send_call_end_s = np.arange(6, dtype=np.float64) + 230

        class Controller:
            calls = 0

            def _get_control_cycle_telemetry(self):
                self.calls += 1
                return Telemetry()

        controller = Controller()
        columns = recording_columns()
        row = capture_row(
            controller,
            object(),
            np.zeros(6),
            np.ones(6),
            0.25,
            0.002,
            0.0001,
            True,
            0.0002,
        )
        values = dict(zip(columns, row))
        self.assertEqual(controller.calls, 1)
        self.assertEqual(len(row), len(columns))
        self.assertEqual(values["control_cycle_sequence"], 42)
        self.assertEqual(values["controller_time_s"], 1.004)
        self.assertEqual(values["derivative_wrench_rx"], 83)
        self.assertEqual(values["joint_torque_command_6"], 215)
        self.assertEqual(values["joint_torque_measured_1"], 160)

    def test_dependency_free_report_and_csv(self) -> None:
        columns = recording_columns()
        rows = []
        for index in range(20):
            row = np.zeros(len(columns), dtype=np.float64)
            row[columns.index("host_elapsed_s")] = index * 0.002
            row[columns.index("controller_time_s")] = index * 0.002
            row[columns.index("sample_dt_s")] = 0.002
            row[columns.index("source_pose_x")] = np.sin(index * 0.1)
            row[columns.index("desired_pose_x")] = np.sin(index * 0.1)
            row[columns.index("measured_pose_x")] = np.sin(index * 0.1 - 0.02)
            rows.append(row.tolist())
        metadata = build_metadata(
            "L5",
            "can0",
            "x",
            np.ones(6),
            np.ones(6),
            50.0,
            0.002,
            0.02,
            2.0,
            rows,
            None,
        )
        self.assertIsNone(metadata["return_home"])
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "recording.csv"
            html_path = Path(directory) / "recording.html"
            write_csv(csv_path, columns, rows)
            write_html_report(html_path, columns, rows, metadata)
            self.assertTrue(csv_path.read_text().startswith("host_elapsed_s,"))
            report = html_path.read_text()
            self.assertIn("Cartesian motion recording", report)
            self.assertIn("Cartesian pose x", report)
            self.assertIn("Joint 6 torque", report)
            self.assertIn("Recorder timing", report)


if __name__ == "__main__":
    unittest.main()
