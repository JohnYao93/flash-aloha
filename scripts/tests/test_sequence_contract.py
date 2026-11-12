"""Training continuity is shared; primitive semantics belong to the task."""
from data_processing import episode_store
import numpy as np
import pytest

from policy import training_utils
from contracts.sequence_contract import contiguous_runs, training_runs
from test_action_contract import write_episode


def dataset(path, **kwargs):
    stats, lengths = training_utils.get_norm_stats([str(path)], observation_name=['qpos'],
        task_boundary_provider=kwargs.get('task_boundary_provider'))
    return training_utils.EpisodicDataset(None, [str(path)], ['video_left'], stats,
        np.array([0]), lengths, 3, 'Flow', observation_name=['qpos'], **kwargs)


@pytest.mark.parametrize('continuity_path', ['observations/capture_continuity_index', 'recording_timing/continuity_id'])
def test_chunks_cannot_cross_recording_gap(tmp_path, continuity_path):
    path = tmp_path / 'episode.zarr'
    write_episode(path)
    with episode_store.Store(path, 'a') as root:
        root[continuity_path] = [0, 0, 1, 1, 1]
    assert dataset(path)[1][3].tolist() == [False, True, True]




def test_history_does_not_cross_gap(tmp_path):
    path = tmp_path / 'episode.zarr'
    write_episode(path)
    with episode_store.Store(path, 'a') as root:
        root['observations/capture_continuity_index'] = [0, 0, 1, 1, 1]
        root['observations/images/video_left'][:] = 255
    ds = dataset(path, history_len=1)
    images, state, _, _ = ds[2]
    assert images.ndim == 4
    # Zero padding is normalized just like the existing episode-start padding.
    np.testing.assert_allclose(state[7:], -ds.norm_stats['qpos_mean'] / ds.norm_stats['qpos_std'])


def test_raw_future_actions_stop_at_anchor_continuity(tmp_path):
    path = tmp_path / 'episode.zarr'
    _, actions = write_episode(path)
    with episode_store.Store(path, 'a') as root:
        root['action_raw'] = actions
        root['action_raw_timestamps'] = [0., .01, 2., 2.01, 2.02]
        root['action_timestamps'] = [0., .01, 2., 2.01, 2.02]
        root['action_raw_capture_continuity_index'] = [0, 0, 1, 1, 1]
        root['observations/capture_continuity_index'] = [0, 0, 1, 1, 1]
    ds = dataset(path, use_raw_future_actions=True)
    assert ds[0][3].tolist() == [False, True, True]
    assert ds[1][3].tolist() == [True, True, True]
    assert ds[2][3].tolist() == [False, False, True]
    with episode_store.Store(path, 'a') as root:
        del root['action_raw_capture_continuity_index']
    with episode_store.Store(path) as root:
        np.testing.assert_array_equal(root['action_raw_capture_continuity_index'], np.zeros(5))












