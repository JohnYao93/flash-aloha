from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_ACT_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ACT_DIR))

from inference import serve_policy  # noqa: E402


class DummyPolicy:
    def forward_inf(self, qpos, image):
        return torch.zeros((1, 3, 7), device=qpos.device)


class DummyFlowPolicy:
    def __init__(self):
        self.last_qpos = None
        self.last_image = None
        self.last_rtc_prefix = None
        self.forward_inf_calls = 0
        self.rtc_calls = 0

    def forward_inf(self, qpos, image):
        self.forward_inf_calls += 1
        self.last_qpos = qpos.detach().cpu()
        self.last_image = image.detach().cpu()
        return torch.ones((1, 50, 7), device=qpos.device) * 2.0

    def forward_inf_training_time_rtc(self, qpos, image, rtc_action_prefix):
        self.rtc_calls += 1
        self.last_qpos = qpos.detach().cpu()
        self.last_image = image.detach().cpu()
        self.last_rtc_prefix = rtc_action_prefix.detach().cpu()
        actions = torch.ones((1, 50, 7), device=qpos.device) * 3.0
        prefix_len = min(rtc_action_prefix.shape[1], actions.shape[1])
        actions[:, :prefix_len] = rtc_action_prefix[:, :prefix_len]
        return actions


class DummyRTCPolicy:
    """Pins the RTC prefix it is given, like the Flow sampler; every other action is -1."""

    def forward_inf(self, qpos, image):
        return torch.ones((1, 5, 7), device=qpos.device) * -1.0

    def forward_inf_training_time_rtc(self, qpos, image, prefix):
        actions = self.forward_inf(qpos, image)
        actions[:, :prefix.shape[1]] = prefix
        return actions


def make_config(state_dim=7, action_dim=7, policy_class="Flow", chunk_size=50):
    return {
        "control_freq": 100,
        "policy_class": policy_class,
        "ckpt_dir": "ckpts/example",
        "task_name": "arx_right",
        "task_config": {},
        "policy_config": {
            "camera_names": ["video_overhead", "video_right"],
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_queries": chunk_size,
            "height": 2,
            "width": 2,
        },
    }


def make_stats(state_dim=7, action_dim=7):
    return {
        "qpos_mean": np.zeros((state_dim,), dtype=np.float32),
        "qpos_std": np.ones((state_dim,), dtype=np.float32),
        "action_mean": np.zeros((action_dim,), dtype=np.float32),
        "action_std": np.ones((action_dim,), dtype=np.float32),
    }


class WebsocketPolicyServerTest(unittest.TestCase):
    def test_parser_defaults_to_no_fill_in(self):
        parser = serve_policy.build_parser()
        args = parser.parse_args(["--config", "ckpts/example/all_configs.json"])

        self.assertFalse(args.fill_in)
        self.assertFalse(args.no_fill_in)

    def test_parser_fill_in_is_explicit_opt_in(self):
        parser = serve_policy.build_parser()
        args = parser.parse_args(["--config", "ckpts/example/all_configs.json", "--fill_in"])

        self.assertTrue(args.fill_in)
        self.assertFalse(args.no_fill_in)

    def test_stats_state_dim_mismatch_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "qpos_mean dim 7.*state_dim 14"):
            serve_policy._validate_config_and_stats(make_config(state_dim=14), make_stats(state_dim=7))


    def test_flow_metadata_advertises_local_request_schema_with_training_time_rtc(self):
        server = serve_policy.LocalPolicyServer(
            policy=DummyFlowPolicy(),
            stats=make_stats(),
            config=make_config(policy_class="Flow"),
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )

        self.assertEqual(server.metadata["policy_class"], "Flow")
        self.assertEqual(server.metadata["request_schema"], "native")
        self.assertEqual(server.metadata["rtc_modes"], ["training_time"])
        self.assertEqual(server.metadata["camera_names"], ["video_overhead", "video_right"])
        self.assertEqual(server.metadata["state_dim"], 7)
        self.assertEqual(server.metadata["action_dim"], 7)

    def test_flow_infer_normalizes_inputs_converts_bgr_and_denormalizes_actions(self):
        config = make_config(policy_class="Flow")
        stats = make_stats()
        stats["qpos_mean"] = np.ones((7,), dtype=np.float32)
        stats["qpos_std"] = np.full((7,), 2.0, dtype=np.float32)
        stats["action_mean"] = np.ones((7,), dtype=np.float32)
        stats["action_std"] = np.full((7,), 0.5, dtype=np.float32)
        policy = DummyFlowPolicy()
        server = serve_policy.LocalPolicyServer(
            policy=policy,
            stats=stats,
            config=config,
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )
        obs = {
            "observation/state": np.arange(7, dtype=np.float32),
            "client/image_color_space": "BGR",
            "observation/images/video_overhead": np.full((2, 2, 3), [1, 2, 3], dtype=np.uint8),
            "observation/images/video_right": np.full((2, 2, 3), [4, 5, 6], dtype=np.uint8),
        }

        response = server.infer(obs)

        self.assertEqual(response["actions"].shape, (1, 50, 7))
        np.testing.assert_allclose(response["actions"], np.full((1, 50, 7), 2.0, dtype=np.float32))
        np.testing.assert_allclose(policy.last_qpos.numpy()[0], (np.arange(7, dtype=np.float32) - 1.0) / 2.0)
        self.assertEqual(tuple(policy.last_image.shape), (1, 2, 3, 2, 2))
        np.testing.assert_allclose(policy.last_image.numpy()[0, 0, :, 0, 0], np.array([3, 2, 1], dtype=np.float32) / 255.0)
        self.assertEqual(policy.forward_inf_calls, 1)
        self.assertEqual(policy.rtc_calls, 0)

    def test_flow_training_time_rtc_calls_policy_method_with_normalized_prefix(self):
        config = make_config(policy_class="Flow")
        stats = make_stats()
        stats["action_mean"] = np.ones((7,), dtype=np.float32)
        stats["action_std"] = np.full((7,), 2.0, dtype=np.float32)
        policy = DummyFlowPolicy()
        server = serve_policy.LocalPolicyServer(
            policy=policy,
            stats=stats,
            config=config,
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )
        prefix = np.stack(
            [
                np.arange(7, dtype=np.float32),
                np.arange(7, dtype=np.float32) + 10.0,
                np.arange(7, dtype=np.float32) + 20.0,
            ],
            axis=0,
        )
        obs = {
            "observation/state": np.zeros((7,), dtype=np.float32),
            "client/image_color_space": "BGR",
            "observation/images/video_overhead": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/images/video_right": np.zeros((2, 2, 3), dtype=np.uint8),
            "rtc/enabled": True,
            "rtc/inference_delay_steps": 2,
            "rtc/prev_chunk_left_over": prefix,
        }

        response = server.infer(obs)

        self.assertEqual(policy.forward_inf_calls, 0)
        self.assertEqual(policy.rtc_calls, 1)
        expected_normalized_prefix = (prefix[:2] - 1.0) / 2.0
        np.testing.assert_allclose(policy.last_rtc_prefix.numpy()[0], expected_normalized_prefix)
        np.testing.assert_allclose(response["actions"][0, :2], prefix[:2])

    def test_flow_training_time_rtc_rejects_wrong_action_dim(self):
        server = serve_policy.LocalPolicyServer(
            policy=DummyFlowPolicy(),
            stats=make_stats(),
            config=make_config(policy_class="Flow"),
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )

        with self.assertRaisesRegex(ValueError, "does not match Flow action_dim"):
            server._build_training_time_rtc_prefix(
                {
                    "rtc/enabled": True,
                    "rtc/inference_delay_steps": 2,
                    "rtc/prev_chunk_left_over": np.zeros((2, 6), dtype=np.float32),
                }
            )

    def test_infer_converts_bgr_images_and_denormalizes_actions(self):
        config = make_config(chunk_size=3)
        stats = make_stats()
        stats["action_mean"] = np.arange(7, dtype=np.float32)
        server = serve_policy.WebsocketPolicyServer(
            policy=DummyPolicy(),
            stats=stats,
            config=config,
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )
        obs = {
            "observation/state": np.zeros((7,), dtype=np.float32),
            "client/image_color_space": "BGR",
            "observation/images/video_overhead": np.full((2, 2, 3), [1, 2, 3], dtype=np.uint8),
            "observation/images/video_right": np.full((2, 2, 3), [4, 5, 6], dtype=np.uint8),
        }

        response = server.infer(obs)

        self.assertEqual(response["actions"].shape, (1, 3, 7))
        np.testing.assert_array_equal(response["actions"][0, 0], stats["action_mean"])


    def test_infer_rejects_a_chunk_of_any_other_length_than_the_checkpoint(self):
        server = serve_policy.WebsocketPolicyServer(
            policy=DummyFlowPolicy(), stats=make_stats(), config=make_config(chunk_size=64),
            ckpt_path="ckpts/example/policy_best.ckpt", host="127.0.0.1", port=8000,
            device=torch.device("cpu"), use_fill_in=False)
        self.assertEqual(server.metadata["prediction_horizon"], 64)
        obs = {
            "observation/state": np.zeros((7,), dtype=np.float32),
            "client/image_color_space": "BGR",
            "observation/images/video_overhead": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/images/video_right": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        with self.assertRaisesRegex(ValueError, "chunk_size is 64"):
            server.infer(obs)  # the dummy policy predicts 50 actions

    def test_checkpoint_without_a_chunk_size_is_rejected(self):
        config = make_config()
        del config["policy_config"]["num_queries"]
        with self.assertRaisesRegex(ValueError, "chunk length"):
            serve_policy.WebsocketPolicyServer(
                policy=DummyFlowPolicy(), stats=make_stats(), config=config,
                ckpt_path="ckpts/example/policy_best.ckpt", host="127.0.0.1", port=8000,
                device=torch.device("cpu"), use_fill_in=False)

    def test_training_time_rtc_clamps_action_prefix(self):
        config = make_config(chunk_size=5)
        stats = make_stats()
        server = serve_policy.WebsocketPolicyServer(
            policy=DummyRTCPolicy(),
            stats=stats,
            config=config,
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )
        prefix = np.stack(
            [
                np.arange(7, dtype=np.float32),
                np.arange(7, dtype=np.float32) + 10.0,
                np.arange(7, dtype=np.float32) + 20.0,
            ],
            axis=0,
        )
        obs = {
            "observation/state": np.zeros((7,), dtype=np.float32),
            "client/image_color_space": "BGR",
            "observation/images/video_overhead": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/images/video_right": np.zeros((2, 2, 3), dtype=np.uint8),
            "rtc/enabled": True,
            "rtc/inference_delay_steps": 2,
            "rtc/prev_chunk_left_over": prefix,
        }

        response = server.infer(obs)

        np.testing.assert_array_equal(response["actions"][0, :2], prefix[:2])

    def test_training_time_rtc_rejects_wrong_action_dim(self):
        config = make_config()
        stats = make_stats()
        server = serve_policy.WebsocketPolicyServer(
            policy=DummyRTCPolicy(),
            stats=stats,
            config=config,
            ckpt_path="ckpts/example/policy_best.ckpt",
            host="127.0.0.1",
            port=8000,
            device=torch.device("cpu"),
            use_fill_in=False,
        )

        with self.assertRaisesRegex(ValueError, "does not match Flow action_dim"):
            server._build_training_time_rtc_prefix(
                {
                    "rtc/enabled": True,
                    "rtc/inference_delay_steps": 2,
                    "rtc/prev_chunk_left_over": np.zeros((2, 6), dtype=np.float32),
                }
            )


if __name__ == "__main__":
    unittest.main()
