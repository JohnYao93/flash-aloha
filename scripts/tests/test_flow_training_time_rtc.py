from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


SRC_ACT_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ACT_DIR))

from policy import imitate_episodes_train  # noqa: E402
from policy.policy import LeRobotFlowPolicy  # noqa: E402


class DummyDiffusion:
    def __init__(self, prediction):
        self.prediction = prediction
        self.last_x_t = None

    def _prepare_global_conditioning(self, batch):
        return None

    def unet(self, x_t, t_embed, global_cond=None):
        self.last_x_t = x_t.detach().clone()
        return self.prediction.to(device=x_t.device, dtype=x_t.dtype)


def make_policy(
    *,
    prediction,
    training=True,
    enabled=True,
    min_delay=0,
    max_delay=2,
    distribution="uniform",
    exp_decay=1.0,
):
    policy = LeRobotFlowPolicy.__new__(LeRobotFlowPolicy)
    torch.nn.Module.__init__(policy)
    policy.training_time_rtc = enabled
    policy.training_time_rtc_min_delay = min_delay
    policy.training_time_rtc_max_delay = max_delay
    policy.training_time_rtc_delay_distribution = distribution
    policy.training_time_rtc_exp_decay = exp_decay
    policy._strict_flow_diffusion_module = DummyDiffusion(prediction)
    policy.train(training)
    return policy






class FlowTrainingTimeRTCTest(unittest.TestCase):
    def test_uniform_delay_sampling_uses_inclusive_bounds(self):
        policy = make_policy(prediction=torch.zeros((3, 2, 1)), min_delay=2, max_delay=4)

        with mock.patch("policy.policy.torch.randint", return_value=torch.tensor([2, 3, 4])) as randint:
            delays = policy._sample_training_time_rtc_delays(3, torch.device("cpu"))

        self.assertEqual(delays.tolist(), [2, 3, 4])
        self.assertEqual(randint.call_args.args[0], 2)
        self.assertEqual(randint.call_args.args[1], 5)

    def test_flow_delay_sampling_is_uniform_even_with_legacy_exp_config(self):
        policy = make_policy(
            prediction=torch.zeros((3, 2, 1)),
            min_delay=2,
            max_delay=4,
            distribution="exp",
            exp_decay=0.5,
        )

        with mock.patch("policy.policy.torch.randint", return_value=torch.tensor([2, 3, 4])) as randint:
            delays = policy._sample_training_time_rtc_delays(3, torch.device("cpu"))

        self.assertEqual(delays.tolist(), [2, 3, 4])
        self.assertEqual(randint.call_args.args[0], 2)
        self.assertEqual(randint.call_args.args[1], 5)

    def test_training_config_validation_failures(self):
        base = {
            "no_training_time_rtc": False,
            "training_time_rtc_min_delay": 0,
            "training_time_rtc_max_delay": 2,
            "training_time_rtc_delay_distribution": "uniform",
            "training_time_rtc_exp_decay": 1.0,
        }

        for updates, message in [
            ({"training_time_rtc_min_delay": -1}, "min_delay must be >= 0"),
            ({"training_time_rtc_min_delay": 3, "training_time_rtc_max_delay": 2}, "max_delay must be >="),
            ({"training_time_rtc_max_delay": 4}, "max_delay must be < chunk_size"),
        ]:
            args = dict(base)
            args.update(updates)
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    imitate_episodes_train.validate_flow_training_time_rtc_args(args, chunk_size=4)

    def test_flow_training_config_drops_delay_distribution_fields(self):
        config = imitate_episodes_train.validate_flow_training_time_rtc_args(
            {
                "no_training_time_rtc": False,
                "training_time_rtc_min_delay": 1,
                "training_time_rtc_max_delay": 2,
                "training_time_rtc_delay_distribution": "exp",
                "training_time_rtc_exp_decay": 0.25,
            },
            chunk_size=4,
        )

        self.assertEqual(
            config,
            {
                "training_time_rtc": True,
                "training_time_rtc_min_delay": 1,
                "training_time_rtc_max_delay": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
