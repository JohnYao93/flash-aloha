#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "setup"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from set_up_recording import (  # noqa: E402
    configured_controller_gain_snapshot,
    move_to_saved_start_position,
    save_start_position,
)


class FakeJointState:
    def __init__(self, qpos: list[float]) -> None:
        self._pos = np.asarray(qpos[:6], dtype=np.float64)
        self.gripper_pos = float(qpos[6])

    def pos(self) -> np.ndarray:
        return self._pos


class StartPositionSaveTest(unittest.TestCase):
    def test_saved_pose_move_enables_configured_gain_after_seeding_current_command(
        self,
    ) -> None:
        events = []

        class FakeGain:
            def __init__(self):
                self._kp = np.zeros(6, dtype=np.float64)
                self._kd = np.ones(6, dtype=np.float64)
                self.gripper_kp = 0.0
                self.gripper_kd = 0.0

            def kp(self):
                return self._kp

            def kd(self):
                return self._kd

        class FakeCommand:
            def __init__(self, _dof):
                self._pos = np.zeros(6, dtype=np.float64)
                self.gripper_pos = 0.0

            def pos(self):
                return self._pos

        class FakeController:
            def __init__(self):
                self.gain = FakeGain()
                self.state = FakeJointState([0, 0, 0, 0, 0, 0, 0.01])

            def get_gain(self):
                return self.gain

            def set_gain(self, gain):
                events.append(("gain", np.asarray(gain.kp()).copy()))
                self.gain = gain

            def set_friction_compensation(self, enabled):
                frictions.append(enabled)

            def get_timestamp(self):
                return 0.0

            def get_joint_state(self):
                return self.state

            def set_joint_cmd(self, command):
                events.append(("command", np.asarray(command.pos()).copy()))

        frictions = []
        controller = FakeController()
        config = SimpleNamespace(
            default_kp=np.asarray([80, 70, 70, 30, 30, 20], dtype=np.float64),
            default_kd=np.asarray([4, 4, 4, 2, 2, 1], dtype=np.float64),
            default_gripper_kp=5.0,
            default_gripper_kd=1.0,
        )
        gains = {"left": configured_controller_gain_snapshot(controller, config)}

        with (
            mock.patch("set_up_recording.time.sleep"),
            mock.patch("set_up_recording.time.monotonic", side_effect=[0.0, 1.0]),
        ):
            move_to_saved_start_position(
                SimpleNamespace(JointState=FakeCommand),
                SimpleNamespace(joint_dof=6),
                {"left": controller},
                {"left": np.asarray([1, 2, 3, 4, 5, 6, 0.02])},
                gains,
                duration=1.0,
                controller_fps=1.0,
            )

        self.assertEqual(events[0][0], "command")
        self.assertEqual(events[1][0], "gain")
        np.testing.assert_allclose(events[1][1], config.default_kp)
        self.assertEqual(events[2][0], "command")
        np.testing.assert_allclose(events[2][1], [1, 2, 3, 4, 5, 6])
        self.assertEqual(frictions, [False])  # friction feedforward is for hand guiding only

    def test_saved_pose_move_pauses_after_stall_instead_of_bursting(self) -> None:
        clock = [0.0]
        submissions = []

        class FakeCommand:
            def __init__(self, _dof):
                self._pos = np.zeros(6, dtype=np.float64)
                self.gripper_pos = 0.0

            def pos(self):
                return self._pos

        class FakeController:
            def get_gain(self):
                return SimpleNamespace(kp=lambda: np.zeros(6), kd=lambda: np.zeros(6))

            def set_gain(self, gain):
                pass

            def set_friction_compensation(self, enabled):
                pass

            def get_timestamp(self):
                return clock[0]

            def get_joint_state(self):
                return FakeJointState([0, 0, 0, 0, 0, 0, 0])

            def set_joint_cmd(self, command):
                submissions.append((clock[0], float(command.pos()[0])))
                clock[0] += 0.0005
                if len(submissions) == 4:
                    clock[0] += 0.45  # e.g. another thread holding the GIL

        def sleep(seconds):
            clock[0] += seconds

        gains = {"left": {"kp": np.ones(6), "kd": np.ones(6),
                          "gripper_kp": 1.0, "gripper_kd": 1.0}}
        with (
            mock.patch("set_up_recording.apply_controller_gain_snapshot"),
            mock.patch("set_up_recording.time.sleep", side_effect=sleep),
            mock.patch("set_up_recording.time.monotonic", side_effect=lambda: clock[0]),
            mock.patch("builtins.print") as printed,
        ):
            move_to_saved_start_position(
                SimpleNamespace(JointState=FakeCommand),
                SimpleNamespace(joint_dof=6),
                {"left": FakeController()},
                {"left": np.asarray([2.0, 0, 0, 0, 0, 0, 0])},
                gains,
                duration=2.0,
                controller_fps=200.0,
                smooth=True,
            )

        steps = submissions[1:]  # the first submission seeds the measured pose
        self.assertEqual(len(steps), 400)
        self.assertAlmostEqual(steps[-1][1], 2.0)
        # After the stall each target is still one plan step (5 ms) apart.
        gaps = np.diff([time for time, _ in steps[4:]])
        self.assertGreaterEqual(gaps.min(), 0.005 - 1e-9)
        # Lateness is measured against the next step's deadline: 450 - 5 + 0.5 ms.
        self.assertRegex(printed.call_args.args[0], r"stalled up to 44[56] ms at step 3/400")

    def test_single_arm_saves_merge_into_bimanual_file(self) -> None:
        left_qpos = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.1]
        right_qpos = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 0.2]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "start_position.npz"
            save_start_position(
                output_path,
                locked_states={"right": FakeJointState(right_qpos)},
                arm_can_interfaces={"right": "can1"},
                camera_names=["right", "overhead"],
            )
            save_start_position(
                output_path,
                locked_states={"left": FakeJointState(left_qpos)},
                arm_can_interfaces={"left": "can0"},
                camera_names=["left", "overhead"],
            )

            with np.load(output_path, allow_pickle=True) as data:
                self.assertEqual(data["arms"].tolist(), ["left", "right"])
                self.assertEqual(data["can_interfaces"].tolist(), ["can0", "can1"])
                np.testing.assert_allclose(data["left_qpos"], left_qpos)
                np.testing.assert_allclose(data["right_qpos"], right_qpos)
                np.testing.assert_allclose(data["qpos"], np.asarray(left_qpos + right_qpos))
                self.assertEqual(data["camera_names"].tolist(), ["left", "overhead", "right"])

    def test_gripper_only_save_preserves_existing_arm_joints(self) -> None:
        original_left = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.01]
        original_right = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 0.02]
        measured_left = [9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 0.04]
        measured_right = [8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 0.06]

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "start_position.npz"
            output_path = Path(tmpdir) / "start_position_grippers.npz"
            save_start_position(
                source_path,
                locked_states={
                    "left": FakeJointState(original_left),
                    "right": FakeJointState(original_right),
                },
                arm_can_interfaces={"left": "can0", "right": "can1"},
                camera_names=["left", "overhead", "right"],
            )
            save_start_position(
                output_path,
                locked_states={
                    "left": FakeJointState(measured_left),
                    "right": FakeJointState(measured_right),
                },
                arm_can_interfaces={"left": "can0", "right": "can1"},
                camera_names=["left", "overhead", "right"],
                gripper_only=True,
                source_position_path=source_path,
            )

            with np.load(output_path, allow_pickle=True) as data:
                np.testing.assert_allclose(
                    data["left_qpos"], original_left[:6] + [measured_left[6]]
                )
                np.testing.assert_allclose(
                    data["right_qpos"], original_right[:6] + [measured_right[6]]
                )
            with np.load(source_path, allow_pickle=True) as original_data:
                np.testing.assert_allclose(original_data["left_qpos"], original_left)
                np.testing.assert_allclose(original_data["right_qpos"], original_right)

    def test_gripper_only_save_requires_existing_arm_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "start_position.npz"
            with self.assertRaisesRegex(ValueError, "no saved left arm pose"):
                save_start_position(
                    output_path,
                    locked_states={
                        "left": FakeJointState([1, 2, 3, 4, 5, 6, 0.04])
                    },
                    arm_can_interfaces={"left": "can0"},
                    camera_names=["left"],
                    gripper_only=True,
                )


if __name__ == "__main__":
    unittest.main()
