#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from data_processing import episode_store
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "src"
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

from recording import record_episode_single as recorder  # noqa: E402


class RecordEpisodeSingleHelpersTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "task_name": "unit_task",
            "robot": None,
            "single_arm": None,
            "start_position_path": recorder.DEFAULT_START_POSITION_PATH,
            "start_position_move_duration": 3.0,
            "fps": 50,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_resolve_context_uses_dataset_dir_and_config_robot(self):
        old = recorder.TASK_CONFIGS.get("unit_task")
        recorder.TASK_CONFIGS["unit_task"] = {
            "dataset_dir": "/tmp/direct_episode_unit",
            "episode_len": 12,
            "camera_names": ["video_right", "video_overhead"],
            "robot_type": "arx",
            "can_interface_left": "can3",
            "single_arm": True,
        }
        try:
            ctx = recorder.resolve_run_context(self._args())
        finally:
            if old is None:
                del recorder.TASK_CONFIGS["unit_task"]
            else:
                recorder.TASK_CONFIGS["unit_task"] = old

        self.assertEqual(ctx.dataset_dir, "/tmp/direct_episode_unit")
        self.assertEqual(ctx.robot_type, "arx")
        self.assertTrue(ctx.single_arm)
        self.assertEqual(ctx.camera_short_names, ["right", "overhead"])
        self.assertEqual(ctx.camera_side, "right")
        self.assertEqual(ctx.qvel_dim, 7)

    def test_robot_override_and_deprecated_single_arm_override(self):
        old = recorder.TASK_CONFIGS.get("unit_task")
        recorder.TASK_CONFIGS["unit_task"] = {
            "dataset_dir": "/tmp/direct_episode_unit",
            "episode_len": 12,
            "camera_names": ["video_left", "video_overhead"],
            "robot_type": "arx",
            "single_arm": False,
            "can_interface_left": "can0",
            "single_arm_side": "left",
        }
        try:
            ctx = recorder.resolve_run_context(self._args(robot="arx", single_arm=True))
        finally:
            if old is None:
                del recorder.TASK_CONFIGS["unit_task"]
            else:
                recorder.TASK_CONFIGS["unit_task"] = old

        self.assertEqual(ctx.robot_type, "arx")
        self.assertTrue(ctx.single_arm)
        self.assertTrue(ctx.deprecated_single_arm_override)
        self.assertEqual(ctx.can_interface_left, "can0")
        self.assertEqual(ctx.qvel_dim, 7)

    def test_arx_qvel_includes_gripper_velocity(self):
        class FakeState:
            gripper_vel = 0.125

            @staticmethod
            def vel():
                return np.arange(6, dtype=np.float64)

        class FakeController:
            @staticmethod
            def get_joint_state():
                return FakeState()

        qvel = recorder.arx_qvel(FakeController())

        self.assertEqual(qvel.shape, (7,))
        self.assertEqual(qvel.dtype, np.float32)
        np.testing.assert_allclose(qvel, [0, 1, 2, 3, 4, 5, 0.125])


    def test_episode_writer_rejects_samples_without_saved_references(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "original stream"):
                recorder.write_episode(Path(directory)/"episode.zarr",
                    [recorder.EpisodeSample({}, np.zeros(7), np.zeros(7), 0.)], [], 7, 7)


if __name__ == "__main__":
    unittest.main()
