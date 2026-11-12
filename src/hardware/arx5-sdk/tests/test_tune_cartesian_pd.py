import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from tune_cartesian_pd import (  # noqa: E402
    MOTION_FREQUENCY_HZ,
    MOTION_JOINT_LIMIT_MARGIN_RAD,
    MOTION_OFF,
    MOTION_RAMP_DURATION_S,
    MOTION_RETURNING,
    MOTION_ROTATION_AMPLITUDE_RAD,
    MOTION_RUNNING,
    MOTION_STARTING,
    MOTION_TRANSLATION_AMPLITUDE_M,
    SinusoidalMotion,
    apply_motion_offset,
    load_gain_config,
    motion_amplitude,
    preflight_sinusoidal_motion,
    save_gain_config,
    sinusoidal_motion_offset,
    sinusoidal_motion_velocity,
    validate_filter_cutoff,
    validate_gain_vector,
)


class FakeSolver:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def inverse_kinematics(self, pose, initial_joint_position):
        self.calls.append(
            (
                np.asarray(pose, dtype=np.float64).copy(),
                np.asarray(initial_joint_position, dtype=np.float64).copy(),
            )
        )
        if self.response is None:
            return 0, np.zeros(6, dtype=np.float64)
        return self.response(len(self.calls) - 1, pose, initial_joint_position)

    def get_ik_status_name(self, status):
        return "fake failure"


class CartesianPDTunerConfigTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        kp = np.array([10, 20, 30, 1, 2, 3], dtype=np.float64)
        kd = np.array([4, 5, 6, 0.4, 0.5, 0.6], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gains.json"
            save_gain_config(path, "L5", kp, kd, 20.0, 0.002)
            loaded_kp, loaded_kd, loaded_cutoff_hz = load_gain_config(
                path, "L5"
            )
        np.testing.assert_array_equal(loaded_kp, kp)
        np.testing.assert_array_equal(loaded_kd, kd)
        self.assertEqual(loaded_cutoff_hz, 20.0)

    def test_legacy_config_uses_model_default_filter(self) -> None:
        config = {
            "model": "L5",
            "kp": [0.0] * 6,
            "kd": [0.0] * 6,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(config))
            _, _, loaded_cutoff_hz = load_gain_config(path, "L5")
        self.assertEqual(loaded_cutoff_hz, 0.0)

    def test_rejects_wrong_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gains.json"
            save_gain_config(path, "L5", np.zeros(6), np.zeros(6))
            with self.assertRaisesRegex(ValueError, "not 'X5'"):
                load_gain_config(path, "X5")

    def test_rejects_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "six values"):
            validate_gain_vector([1, 2], "kp")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_gain_vector([0, 0, 0, 0, 0, np.nan], "kd")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_gain_vector([0, 0, -1, 0, 0, 0], "kp")

    def test_rejects_missing_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gains.json"
            path.write_text(json.dumps({"model": "L5", "kp": [0] * 6}))
            with self.assertRaisesRegex(ValueError, "six values"):
                load_gain_config(path, "L5")

    def test_rejects_invalid_filter_cutoff(self) -> None:
        for cutoff_hz in (-1.0, 250.0, np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "below 250"):
                validate_filter_cutoff(cutoff_hz, 0.002)

        config = {
            "model": "L5",
            "kp": [0.0] * 6,
            "kd": [0.0] * 6,
            "cartesian_joint_velocity_filter_cutoff_hz": 250.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-filter.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "below 250"):
                load_gain_config(path, "L5")


class SinusoidalMotionTest(unittest.TestCase):
    def test_translation_and_rotation_amplitudes_and_timing(self) -> None:
        quarter_cycle_s = 0.25 / MOTION_FREQUENCY_HZ
        self.assertEqual(MOTION_TRANSLATION_AMPLITUDE_M, 0.040)
        self.assertAlmostEqual(MOTION_ROTATION_AMPLITUDE_RAD, np.deg2rad(20.0))
        self.assertEqual(motion_amplitude(0), MOTION_TRANSLATION_AMPLITUDE_M)
        self.assertEqual(motion_amplitude(3), MOTION_ROTATION_AMPLITUDE_RAD)
        self.assertAlmostEqual(
            sinusoidal_motion_offset(quarter_cycle_s, 0),
            MOTION_TRANSLATION_AMPLITUDE_M,
        )
        self.assertAlmostEqual(
            sinusoidal_motion_offset(quarter_cycle_s, 5),
            MOTION_ROTATION_AMPLITUDE_RAD,
        )
        self.assertAlmostEqual(
            sinusoidal_motion_offset(1.0 / MOTION_FREQUENCY_HZ, 0), 0.0
        )

    def test_profile_is_bounded_and_starts_smoothly(self) -> None:
        motion = SinusoidalMotion()
        anchor = np.arange(6, dtype=np.float64)
        motion.start(1, anchor, 0.037, 10.0)

        starting_pose, starting_twist = motion.command_reference(10.0)
        np.testing.assert_array_equal(starting_pose, anchor)
        np.testing.assert_array_equal(starting_twist, np.zeros(6))
        self.assertEqual(motion.state, MOTION_STARTING)

        times = np.linspace(0.0, 12.0, 1201)
        offsets = np.array(
            [motion.command_pose(10.0 + value)[1] - anchor[1] for value in times]
        )
        self.assertLessEqual(
            np.max(np.abs(offsets)),
            MOTION_TRANSLATION_AMPLITUDE_M + np.finfo(np.float64).eps,
        )
        # The cubic envelope and zero-crossing sinusoid both start at rest.
        self.assertLess(abs(offsets[1] / (times[1] - times[0])), 1e-4)
        self.assertEqual(motion.state, MOTION_RUNNING)
        self.assertEqual(motion.gripper_position, 0.037)

    def test_analytic_twist_matches_pose_derivative(self) -> None:
        for axis_index in range(6):
            motion = SinusoidalMotion()
            anchor = np.array([0.1, -0.2, 0.3, 0.2, -0.3, 0.4])
            motion.start(axis_index, anchor, 0.02, 1.0)
            now = 2.4
            pose, twist = motion.command_reference(now)
            epsilon = 1e-6
            before_pose = motion.command_pose(now - epsilon)
            after_pose = motion.command_pose(now + epsilon)
            if axis_index < 3:
                measured_velocity = (
                    after_pose[axis_index] - before_pose[axis_index]
                ) / (2.0 * epsilon)
            else:
                # The profile is a base-frame single-axis rotation, so its
                # spatial angular velocity is the scalar offset derivative.
                elapsed = now - motion.start_time
                measured_velocity = sinusoidal_motion_velocity(
                    elapsed, axis_index, 1.0, 0.0
                )
            self.assertAlmostEqual(twist[axis_index], measured_velocity, places=6)
            self.assertTrue(np.all(np.isfinite(pose)))

    def test_rotation_offset_is_composed_in_base_frame(self) -> None:
        anchor = np.array([0.1, 0.2, 0.3, 0.4, -0.2, 0.7])
        rotated = apply_motion_offset(anchor, 3, 0.1)
        self.assertFalse(np.array_equal(rotated[3:], anchor[3:] + [0.1, 0, 0]))
        np.testing.assert_array_equal(rotated[:3], anchor[:3])

    def test_ramp_out_returns_exactly_to_anchor(self) -> None:
        motion = SinusoidalMotion()
        anchor = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        motion.start(4, anchor, 0.02, 5.0)
        motion.command_pose(7.0)
        motion.begin_return(7.0)
        self.assertEqual(motion.state, MOTION_RETURNING)

        before_end = motion.command_pose(
            7.0 + MOTION_RAMP_DURATION_S - 1e-6
        )
        self.assertLessEqual(
            abs(before_end[4] - anchor[4]), MOTION_ROTATION_AMPLITUDE_RAD
        )
        returned, returned_twist = motion.command_reference(
            7.0 + MOTION_RAMP_DURATION_S
        )
        np.testing.assert_array_equal(returned, anchor)
        np.testing.assert_array_equal(returned_twist, np.zeros(6))
        self.assertEqual(motion.state, MOTION_OFF)
        self.assertFalse(motion.active)

    def test_stop_during_startup_does_not_jump(self) -> None:
        motion = SinusoidalMotion()
        anchor = np.zeros(6)
        motion.start(0, anchor, 0.01, 1.0)
        before_stop = motion.command_pose(1.4)
        motion.begin_return(1.4)
        at_stop = motion.command_pose(1.4)
        np.testing.assert_array_equal(at_stop, before_stop)


class SinusoidalPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = np.zeros(6, dtype=np.float64)
        self.measured_q = np.full(6, 0.25, dtype=np.float64)
        self.joint_min = np.full(6, -1.0, dtype=np.float64)
        self.joint_max = np.full(6, 1.0, dtype=np.float64)

    def test_accepts_complete_cycle_and_uses_measured_seed(self) -> None:
        solver = FakeSolver()
        sample_count = 17
        preflight_sinusoidal_motion(
            solver,
            self.anchor,
            self.measured_q,
            self.joint_min,
            self.joint_max,
            2,
            sample_count=sample_count,
        )
        self.assertEqual(len(solver.calls), sample_count)
        for _, seed in solver.calls:
            np.testing.assert_array_equal(seed, self.measured_q)
        offsets = np.array([pose[2] for pose, _ in solver.calls])
        self.assertAlmostEqual(offsets.max(), MOTION_TRANSLATION_AMPLITUDE_M)
        self.assertAlmostEqual(offsets.min(), -MOTION_TRANSLATION_AMPLITUDE_M)
        self.assertAlmostEqual(offsets[0], offsets[-1])

    def test_rejects_ik_failure(self) -> None:
        def fail_second_sample(index, _pose, _seed):
            if index == 1:
                return -3, np.zeros(6)
            return 0, np.zeros(6)

        solver = FakeSolver(fail_second_sample)
        with self.assertRaisesRegex(ValueError, "IK failed.*fake failure"):
            preflight_sinusoidal_motion(
                solver,
                self.anchor,
                self.measured_q,
                self.joint_min,
                self.joint_max,
                0,
                sample_count=5,
            )

    def test_rejects_insufficient_joint_limit_margin(self) -> None:
        too_close = np.zeros(6, dtype=np.float64)
        too_close[3] = (
            self.joint_max[3] - MOTION_JOINT_LIMIT_MARGIN_RAD
        )
        solver = FakeSolver(
            lambda _index, _pose, _seed: (0, too_close.copy())
        )
        with self.assertRaisesRegex(ValueError, "joint 4.*clearance"):
            preflight_sinusoidal_motion(
                solver,
                self.anchor,
                self.measured_q,
                self.joint_min,
                self.joint_max,
                3,
                sample_count=5,
            )

if __name__ == "__main__":
    unittest.main()
