#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPTS_DIR = REPO_ROOT / "scripts" / "setup"
if str(SETUP_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_SCRIPTS_DIR))

import hold_start_position as hold  # noqa: E402


def pressed_h():
    """The operator presses 'h' as soon as damping starts."""
    from hardware import safe_exit
    keys = SimpleNamespace(home=SimpleNamespace(is_set=lambda: True))
    return mock.patch.object(safe_exit, "HomeKeyReader", lambda: keys)


class HoldStartPositionTest(unittest.TestCase):
    def test_load_selected_start_positions_rejects_missing_arm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "start_position.npz"
            np.savez(path, left_qpos=np.arange(7, dtype=np.float64))

            with self.assertRaisesRegex(ValueError, "right"):
                hold.load_selected_start_positions(
                    path,
                    {"left": "can0", "right": "can1"},
                )

    def test_run_holds_saved_pose_then_interrupt_damps_until_h_and_homes(self):
        args = SimpleNamespace(
            arm="both",
            start_position="unused.npz",
            can_interface=None,
            left_can_interface=None,
            right_can_interface=None,
            move_duration=2.0,
            controller_fps=50.0,
        )
        controllers = {"left": mock.Mock(), "right": mock.Mock()}
        qpos = {
            "left": np.arange(7, dtype=np.float64),
            "right": np.arange(7, 14, dtype=np.float64),
        }
        commands = {"left": object(), "right": object()}

        with (
            pressed_h(),
            mock.patch.object(
                hold,
                "load_selected_start_positions",
                return_value=(Path("unused.npz"), qpos),
            ),
            mock.patch.object(hold, "load_arx5_interface", return_value="arx5"),
            mock.patch.object(
                hold,
                "build_arx_controllers",
                return_value=("robot_config", "controller_config", controllers),
            ),
            mock.patch.object(
                hold,
                "configured_controller_gain_snapshot",
                return_value={"gain": True},
            ),
            mock.patch.object(hold, "move_to_saved_start_position") as move,
            mock.patch.object(
                hold,
                "build_qpos_commands",
                return_value=commands,
            ) as build_commands,
            mock.patch.object(hold, "send_hold_commands") as send,
            mock.patch.object(hold, "return_arms_home") as home,
            mock.patch.object(hold.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            exit_code = hold.run_hold(args)

        self.assertEqual(exit_code, 130)
        for controller in controllers.values():
            controller.set_to_damping.assert_called_once()  # failure exit: damping
            controller.reset_to_home.assert_called_once()   # then 'h' homes
        home.assert_not_called()  # no second, normal-exit home
        move.assert_called_once_with(
            "arx5",
            "robot_config",
            controllers,
            qpos,
            {"left": {"gain": True}, "right": {"gain": True}},
            duration=2.0,
            controller_fps=50.0,
        )
        build_commands.assert_called_once_with("arx5", "robot_config", qpos)
        send.assert_called_once_with(controllers, commands)

    def test_motion_failure_damps_until_h_then_homes(self):
        args = SimpleNamespace(
            arm="left",
            start_position="unused.npz",
            can_interface=None,
            left_can_interface=None,
            right_can_interface=None,
            move_duration=2.0,
            controller_fps=50.0,
        )
        controllers = {"left": mock.Mock()}

        with (
            pressed_h(),
            mock.patch.object(
                hold,
                "load_selected_start_positions",
                return_value=(Path("unused.npz"), {"left": np.arange(7)}),
            ),
            mock.patch.object(hold, "load_arx5_interface", return_value="arx5"),
            mock.patch.object(
                hold,
                "build_arx_controllers",
                return_value=("robot_config", "controller_config", controllers),
            ),
            mock.patch.object(
                hold,
                "configured_controller_gain_snapshot",
                return_value={"gain": True},
            ),
            mock.patch.object(
                hold,
                "move_to_saved_start_position",
                side_effect=RuntimeError("motion failed"),
            ),
            mock.patch.object(hold, "return_arms_home") as home,
        ):
            with self.assertRaisesRegex(RuntimeError, "motion failed"):
                hold.run_hold(args)

        controllers["left"].set_to_damping.assert_called_once()
        controllers["left"].reset_to_home.assert_called_once()
        home.assert_not_called()

    def test_homing_attempts_every_arm_after_one_fails(self):
        left = mock.Mock()
        right = mock.Mock()
        left.reset_to_home.side_effect = RuntimeError("left bus failed")

        with self.assertRaisesRegex(RuntimeError, "left: left bus failed"):
            hold.return_arms_home({"left": left, "right": right})

        left.reset_to_home.assert_called_once_with()
        right.reset_to_home.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
