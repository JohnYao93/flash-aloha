from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from data_processing import episode_store
import numpy as np
import pytest

SRC_ACT_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ACT_DIR))
from contracts.action_contract import require_absolute_actions

try:
    from policy import training_utils
except ImportError as exc:
    training_utils = None
    TRAINING_UTILS_IMPORT_ERROR = exc
else:
    TRAINING_UTILS_IMPORT_ERROR = None


def write_episode(path):
    qpos = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.01],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.02],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 0.03],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 0.04],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 0.05],
        ],
        dtype=np.float32,
    )
    action = qpos + np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 10.0], dtype=np.float32)
    qvel = np.zeros_like(qpos)
    image = np.zeros((len(qpos), 2, 2, 3), dtype=np.uint8)

    with episode_store.Store(path, "w") as root:
        root.attrs["compress"] = False
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        images = obs.create_group("images")
        images.create_dataset("video_left", data=image)
        root.create_dataset("action", data=action)
    from episode_fixtures import attach_streams
    attach_streams(path)
    return qpos, action


@unittest.skipIf(training_utils is None, f"training_utils import failed: {TRAINING_UTILS_IMPORT_ERROR}")
class AbsoluteActionDatasetTest(unittest.TestCase):
    def test_route_policy_mask_excludes_reset_frames_and_stops_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_masked.zarr"
            _, action = write_episode(path)
            policy_mask = np.array([True, True, False, False, True])
            with episode_store.Store(path, "a") as root:
                route = root.create_group("route")
                route.create_dataset("policy_training_mask", data=policy_mask)
                root.attrs["policy_training_mask_path"] = "/route/policy_training_mask"

            stats, episode_lengths = training_utils.get_norm_stats(
                [str(path)],
                observation_name=["qpos"],
                outputs_name=["action"],
            )
            self.assertEqual(episode_lengths, [3])
            np.testing.assert_allclose(
                stats["action_mean"],
                action[policy_mask].mean(axis=0),
                atol=1e-6,
            )
            dataset = training_utils.EpisodicDataset(
                None,
                [str(path)],
                ["video_left"],
                stats,
                np.array([0]),
                episode_lengths,
                3,
                "Flow",
                observation_name=["qpos"],
            )
            self.assertEqual(dataset._locate_transition(0), (0, 0))
            self.assertEqual(dataset._locate_transition(1), (0, 1))
            self.assertEqual(dataset._locate_transition(2), (0, 4))
            _, _, _, is_pad = dataset[1]
            np.testing.assert_array_equal(
                is_pad.numpy(), np.array([False, True, True])
            )

            included = training_utils.EpisodicDataset(
                None,
                [str(path)],
                ["video_left"],
                stats,
                np.array([0]),
                [len(action)],
                3,
                "Flow",
                observation_name=["qpos"],
                include_masked_segments=True,
            )
            self.assertEqual(included._locate_transition(2), (0, 2))

    def test_absolute_mode_keeps_existing_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_0.zarr"
            _, action = write_episode(path)

            stats, _ = training_utils.get_norm_stats(
                [str(path)],
                observation_name=["qpos"],
                outputs_name=["action"],
                action_command_style="absolute",
            )
            dataset = training_utils.EpisodicDataset(
                None,
                [str(path)],
                ["video_left"],
                stats,
                np.array([0]),
                [len(action)],
                3,
                "Flow",
                observation_name=["qpos"],
                action_command_style="absolute",
            )

            _, _, normalized_action, is_pad = dataset[2]
            action_chunk = normalized_action.numpy() * stats["action_std"] + stats["action_mean"]
            np.testing.assert_allclose(action_chunk, action[2:5], atol=1e-6)
            np.testing.assert_array_equal(is_pad.numpy(), np.array([False, False, False]))


@pytest.mark.parametrize('declaration', [
    {'action_command_style': 'relative_chunk_delta'},
    {'action_command_style': 'absolute', 'policy_config': {'action_command_style': 'relative_chunk_delta'}},
    {'task_config': {'action_command_style': 'relative_chunk_delta'}},
    {'action_command_style': 'absolute', 'relative_action_base': 'chunk_start_qpos'},
])
def test_explicit_relative_contract_cannot_be_overridden(declaration):
    with pytest.raises(ValueError, match='Only absolute'):
        require_absolute_actions(declaration, {'action_command_style': 'absolute'})


def test_legacy_and_explicit_absolute_contracts():
    assert require_absolute_actions({}, {'policy_config': {'action_command_style': b'absolute'}}) == 'absolute'


def test_relative_dataset_rejected_by_statistics_and_samples(tmp_path):
    path = tmp_path / 'episode.zarr'
    write_episode(path)
    stats, lengths = training_utils.get_norm_stats([str(path)], observation_name=['qpos'])
    with episode_store.Store(path, 'a') as root:
        root.attrs['action_command_style'] = 'relative_chunk_delta'
    with pytest.raises(ValueError, match='Only absolute'):
        training_utils.get_norm_stats([str(path)], observation_name=['qpos'])
    with pytest.raises(ValueError, match='Only absolute'):
        training_utils.EpisodicDataset(None, [str(path)], ['video_left'], stats,
                                      np.array([0]), lengths, 3, 'Flow', observation_name=['qpos'])


def test_relative_checkpoint_rejected_before_model_loading_and_serving():
    import torch
    from inference import inference_direct
    from inference import inference_remote
    from inference.serve_policy import LocalPolicyServer
    config = {'policy_config': {'action_command_style': 'relative_chunk_delta'}}
    with pytest.raises(ValueError, match='Only absolute'):
        inference_direct.load_policy_and_stats(config, 'nonexistent.ckpt')
    with pytest.raises(ValueError, match='Only absolute'):
        LocalPolicyServer(policy=None, stats={}, config=config, ckpt_path='synthetic',
                          host='unused', port=0, device=torch.device('cpu'), use_fill_in=False)
    with pytest.raises(ValueError, match='Only absolute'):
        inference_remote.validate_policy_metadata(config)


def test_relative_recording_payload_rejected_before_write(tmp_path):
    from recording.distributed_recording import ControlWriter
    writer = ControlWriter(tmp_path / 'control.zarr')
    with pytest.raises(ValueError, match='Only absolute'):
        writer.append({'action_command_style': 'relative_chunk_delta'})
    assert writer.count == 0
    from data_processing.episode_store import publish_episode
    from test_raw_training import make_raw
    source = make_raw(tmp_path / "source")
    output = tmp_path / 'rejected.zarr'
    with pytest.raises(ValueError, match='Only absolute'):
        publish_episode(output, [(source, [0, 1])], metadata={'action_command_style': 'relative_chunk_delta'})
    assert not output.exists()


