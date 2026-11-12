import sys
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from record_torque_latency import (  # noqa: E402
    DEFAULT_AMPLITUDE_NM,
    DM_TORQUE_MAX,
    DM_TORQUE_MIN,
    RawCanFrame,
    TorqueBurstProfile,
    classify_target_frames,
    compare_local_vs_host_damping,
    compare_motor_kd_frequency_response,
    decode_dm_command,
    decode_dm_feedback,
    fit_first_order_lag_plus_delay,
    frames_in_monotonic_window,
    main,
    motor_kd_condition_specs,
    safety_guard_reason,
)


def float_to_uint(value, minimum, maximum, bits):
    return int((value - minimum) * ((1 << bits) - 1) / (maximum - minimum))


def command_payload(position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0):
    p = float_to_uint(position, -12.5, 12.5, 16)
    v = float_to_uint(velocity, -45.0, 45.0, 12)
    p_gain = float_to_uint(kp, 0.0, 500.0, 12)
    d_gain = float_to_uint(kd, 0.0, 5.0, 12)
    t = float_to_uint(torque, DM_TORQUE_MIN, DM_TORQUE_MAX, 12)
    return bytes(
        [
            p >> 8,
            p & 0xFF,
            v >> 4,
            ((v & 0xF) << 4) | (p_gain >> 8),
            p_gain & 0xFF,
            d_gain >> 4,
            ((d_gain & 0xF) << 4) | (t >> 8),
            t & 0xFF,
        ]
    )


def feedback_payload(motor_id=7, position=0.0, velocity=0.0, torque=0.0):
    p = float_to_uint(position, -12.5, 12.5, 16)
    v = float_to_uint(velocity, -45.0, 45.0, 12)
    t = float_to_uint(torque, DM_TORQUE_MIN, DM_TORQUE_MAX, 12)
    return bytes(
        [
            motor_id,
            p >> 8,
            p & 0xFF,
            v >> 4,
            ((v & 0xF) << 4) | (t >> 8),
            t & 0xFF,
            30,
            31,
        ]
    )


class TorqueLatencyRecorderTest(unittest.TestCase):
    def test_default_amplitude_resolves_the_dm_torque_field(self):
        torque_lsb_nm = 0.424 * (DM_TORQUE_MAX - DM_TORQUE_MIN) / ((1 << 12) - 1)
        self.assertGreater(DEFAULT_AMPLITUDE_NM / torque_lsb_nm, 50.0)

    def test_profile_is_bounded_and_zero_between_integer_cycle_bursts(self):
        profile = TorqueBurstProfile(
            amplitude_nm=0.05,
            frequencies_hz=(5.0, 12.0),
            minimum_duration_s=1.0,
            ramp_duration_s=0.1,
            gap_duration_s=0.2,
            lead_duration_s=0.3,
        )
        samples = np.array(
            [profile.sample(value)[0] for value in np.linspace(0, profile.duration_s, 5000)]
        )
        self.assertLessEqual(np.max(np.abs(samples)), 0.05 + 1e-12)
        self.assertEqual(profile.sample(0.0), (0.0, 0.0, 0.0))
        for burst in profile.bursts:
            self.assertAlmostEqual(profile.sample(burst.start_s)[0], 0.0)
            self.assertAlmostEqual(
                np.sin(2 * np.pi * burst.frequency_hz * burst.duration_s),
                0.0,
                places=10,
            )

    def test_dm_command_and_feedback_decoding(self):
        command = decode_dm_command(
            command_payload(position=1.0, velocity=-2.0, kp=25.0, kd=0.5, torque=0.2)
        )
        self.assertAlmostEqual(command["position"], 1.0, places=3)
        self.assertAlmostEqual(command["velocity"], -2.0, delta=0.012)
        self.assertAlmostEqual(command["kp"], 25.0, delta=0.13)
        self.assertAlmostEqual(command["kd"], 0.5, delta=0.002)
        self.assertAlmostEqual(command["torque"], 0.2, delta=0.01)

        feedback = decode_dm_feedback(
            feedback_payload(position=-0.5, velocity=1.0, torque=-0.3)
        )
        self.assertEqual(feedback["motor_id"], 7)
        self.assertAlmostEqual(feedback["position"], -0.5, places=3)
        self.assertAlmostEqual(feedback["velocity"], 1.0, delta=0.012)
        self.assertAlmostEqual(feedback["torque"], -0.3, delta=0.01)

    def test_raw_target_frames_are_classified(self):
        frames = [
            RawCanFrame(1, 10, 11, 7, 8, command_payload(torque=0.2), 0, True),
            RawCanFrame(2, 12, 13, 0, 8, feedback_payload(torque=-0.3), 0, False),
        ]
        commands, feedback = classify_target_frames(frames, 7, 0.424, 2.0 / 3.0)
        self.assertEqual(len(commands), 1)
        self.assertEqual(len(feedback), 1)
        self.assertAlmostEqual(commands[0]["torque_sdk_nm"], 0.2 * 0.424, delta=0.005)
        self.assertAlmostEqual(feedback[0]["torque_sdk_nm"], -0.3 * 0.424, delta=0.005)
        self.assertAlmostEqual(
            feedback[0]["velocity_sdk_rad_s"],
            decode_dm_feedback(feedback_payload(torque=-0.3))["velocity"] * 2.0 / 3.0,
        )

    def test_same_session_ab_runs_damped_first_and_slices_frames(self):
        self.assertEqual(
            motor_kd_condition_specs(0.0, 0.7),
            [("damped", 0.7, 0.0), ("baseline", 0.0, 0.0)],
        )
        with self.assertRaises(ValueError):
            motor_kd_condition_specs(0.1, 0.7)
        frames = [
            RawCanFrame(0, time_ns, 0, 7, 8, bytes(8), 0, True)
            for time_ns in (9, 10, 15, 20, 21)
        ]
        selected = frames_in_monotonic_window(frames, 10, 20)
        self.assertEqual([frame.kernel_monotonic_ns for frame in selected], [10, 15, 20])

    def test_local_vs_host_conditions_and_comparison(self):
        self.assertEqual(
            motor_kd_condition_specs(0.0, None, 0.7, 1.0),
            [("local", 0.7, 0.0), ("host", 0.0, 1.0)],
        )
        frequencies = (5.0, 12.0, 20.0)

        def metadata(label, kd, host_damping, gain_scale, phase_offset):
            return {
                "model": "Trossen_L5",
                "joint": 6,
                "motor_id": 7,
                "failure": None,
                "anchor_joint_position_rad": np.zeros(6).tolist(),
                "host_joint_damping_nm_s_per_rad": host_damping,
                "analysis": {
                    "raw_command_kd_median": kd,
                    "raw_feedback_status_codes": [1],
                    "frequency_response": [
                        {
                            "frequency_hz": frequency,
                            "excitation_velocity_gain_rad_s_per_nm": (
                                gain_scale / frequency
                            ),
                            "excitation_velocity_phase_deg": (
                                -70.0 + phase_offset
                            ),
                        }
                        for frequency in frequencies
                    ],
                },
                "session_condition_label": label,
            }

        comparison = compare_local_vs_host_damping(
            metadata("local", 0.7, 0.0, 1.0, 0.0),
            metadata("host", 0.0, 1.0, 1.2, -5.0),
        )
        self.assertAlmostEqual(
            comparison["summary"]["median_host_over_local_gain"], 1.2
        )
        self.assertAlmostEqual(
            comparison["summary"]["median_host_minus_local_phase_deg"], -5.0
        )

    def test_motor_kd_ab_recovers_fresher_velocity_path(self):
        frequencies = (5.0, 12.0, 20.0)
        torque_constant = 0.424
        damped_kd = 0.5
        damping = torque_constant * damped_kd
        baseline_rows = []
        damped_rows = []
        for frequency in frequencies:
            omega = 2.0 * np.pi * frequency
            reported_filter = 1.0 / (1.0 + 1j * omega * 0.004)
            plant = 80.0 / (1j * omega + 4.0)
            g0 = reported_filter * plant
            internal_vs_reported = 1.0 / reported_filter
            gd = g0 / (1.0 + damping * internal_vs_reported * g0)

            def response_row(transfer):
                return {
                    "frequency_hz": frequency,
                    "raw_velocity_gain_rad_s_per_nm": abs(transfer),
                    "raw_velocity_phase_deg": np.angle(transfer, deg=True),
                    "reported_velocity_vs_position_derivative_gain": abs(
                        reported_filter
                    ),
                    "reported_velocity_vs_position_derivative_phase_deg": np.angle(
                        reported_filter, deg=True
                    ),
                }

            baseline_rows.append(response_row(g0))
            damped_rows.append(response_row(gd))

        common = {
            "model": "Trossen_L5",
            "joint": 6,
            "motor_id": 7,
            "torque_constant": torque_constant,
            "amplitude_nm": 0.2,
        }
        baseline = {
            **common,
            "selected_joint_motor_kd": 0.0,
            "analysis": {
                "raw_command_kd_median": 0.0,
                "raw_feedback_status_codes": [1],
                "frequency_response": baseline_rows,
            },
        }
        damped = {
            **common,
            "selected_joint_motor_kd": damped_kd,
            "analysis": {
                "raw_command_kd_median": damped_kd,
                "raw_feedback_status_codes": [1],
                "frequency_response": damped_rows,
            },
        }
        comparison = compare_motor_kd_frequency_response(baseline, damped)
        self.assertEqual(
            comparison["summary"]["closer_phase_hypothesis"],
            "fresher_less_filtered_velocity",
        )
        self.assertLess(
            comparison["summary"]["fresher_velocity_phase_rmse_deg"], 1e-9
        )
        for item in comparison["frequency_response"]:
            self.assertAlmostEqual(
                item["inferred_internal_vs_reported_gain"],
                item["fresher_velocity_expected_gain"],
                places=10,
            )

    def test_first_order_lag_is_not_misreported_as_transport_delay(self):
        frequencies = np.array([5.0, 8.0, 12.0, 16.0, 20.0, 25.0])
        omega = 2.0 * np.pi * frequencies
        expected_tau = 0.013
        expected_delay = 0.0045
        response = (
            1.01
            / (1.0 + 1j * omega * expected_tau)
            * np.exp(-1j * omega * expected_delay)
        )
        fit = fit_first_order_lag_plus_delay(frequencies, response)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["dc_gain"], 1.01, places=3)
        self.assertAlmostEqual(fit["time_constant_ms"], 13.0, places=2)
        self.assertAlmostEqual(fit["additional_delay_ms"], 4.5, places=2)
        self.assertLess(fit["phase_rmse_deg"], 0.01)

    def test_safety_guard_checks_travel_speed_and_limits(self):
        zeros = np.zeros(6)
        limits = np.full(6, 2.0)
        self.assertIsNone(
            safety_guard_reason(zeros, zeros, zeros, -limits, limits, 5, 0.2, 1.5, 0.1)
        )
        moved = zeros.copy()
        moved[5] = 0.21
        self.assertIn(
            "moved",
            safety_guard_reason(moved, zeros, zeros, -limits, limits, 5, 0.2, 1.5, 0.1),
        )
        fast = zeros.copy()
        fast[5] = 1.6
        self.assertIn(
            "rad/s",
            safety_guard_reason(zeros, fast, zeros, -limits, limits, 5, 0.2, 1.5, 0.1),
        )

    def test_cli_documents_raw_latency_controls(self):
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--amplitude", result.output)
        self.assertIn("--frequency", result.output)
        self.assertIn("--target-motor-kd", result.output)
        self.assertIn("--ab-motor-kd", result.output)
        self.assertIn("--local-vs-host-motor-kd", result.output)
        self.assertIn("--host-joint-damping", result.output)
        self.assertIn("--travel-limit", result.output)
        self.assertIn("--return-home", result.output)


if __name__ == "__main__":
    unittest.main()
