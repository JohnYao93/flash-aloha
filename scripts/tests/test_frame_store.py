"""The training frame store must reproduce live exact-frame preprocessing bit for bit."""
import os
import pickle

import numpy as np
import pytest
import torch

from data_processing.episode_data import clear_episode_cache
from test_raw_training import make_raw


CAMERAS = ['video_left', 'video_overhead']


@pytest.fixture(autouse=True)
def caches(tmp_path, monkeypatch):
    monkeypatch.setenv('FLASH_ALOHA_VIDEO_INDEX_CACHE', str(tmp_path / 'indices'))
    clear_episode_cache()
    yield
    clear_episode_cache()


@pytest.fixture
def raw(tmp_path):
    return make_raw(tmp_path / 'data/raw/episode')


def dataset(raw, store=None, **extra):
    from policy import training_utils as training
    stats, lengths = training.get_norm_stats([str(raw)], observation_name=['qpos'])
    args = dict(observation_name=['qpos'], width=27, height=19)
    args.update(extra)
    return training.EpisodicDataset(None, [str(raw)], CAMERAS, stats, [0], lengths, 3, 'Flow',
                                    frame_store_dir=store, **args)


@pytest.mark.parametrize('size', [dict(width=27, height=19), dict(width=None, height=None)])
def test_store_matches_live_decoding_exactly(raw, tmp_path, size):
    live = dataset(raw, **size)
    stored = dataset(raw, store=tmp_path / 'store', **size)
    assert stored.frame_store is not None and live.frame_store is None
    for index in range(int(live.cumulative_len[-1])):
        for actual, expected in zip(stored[index], live[index]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_store_is_reused_and_rebuilt_when_episode_changes(raw, tmp_path):
    store = tmp_path / 'store'
    first = dataset(raw, store=store).frame_store.entries
    stamps = {path: os.stat(path).st_mtime_ns for path in first.values()}
    assert dataset(raw, store=store).frame_store.entries == first
    assert {path: os.stat(path).st_mtime_ns for path in first.values()} == stamps
    manifest = raw / 'manifest.json'
    manifest.write_text(manifest.read_text())
    os.utime(manifest, ns=(1, 1))
    clear_episode_cache()
    assert set(dataset(raw, store=store).frame_store.entries.values()).isdisjoint(first.values())


def test_store_pickles_as_paths_only(raw, tmp_path):
    frames = dataset(raw, store=tmp_path / 'store').frame_store
    frames.get(0, CAMERAS[0], int(frames.start_indices[0][0]))
    assert frames._arrays
    restored = pickle.loads(pickle.dumps(frames))
    assert restored._arrays == {} and restored.entries == frames.entries
    with pytest.raises(KeyError):
        restored.get(0, CAMERAS[0], 10_000)


def test_store_feeds_dataloader_workers(raw, tmp_path):
    from policy import training_utils as training
    common = dict(policy_class='Flow', observation_name=['qpos'], width=32, height=24)
    live, *_ = training.load_data(0, None, str(raw.parent.parent), lambda _: True, CAMERAS, 2, 2, 3, **common)
    stored, *_ = training.load_data(2, None, str(raw.parent.parent), lambda _: True, CAMERAS, 2, 2, 3,
                                    frame_store_dir=str(tmp_path / 'store'), **common)
    iterator = iter(stored)
    batch = next(iterator)
    assert batch[0].shape == (2, 2, 3, 24, 32)
    iterator._shutdown_workers()


def test_store_refuses_feature_loss(raw, tmp_path):
    with pytest.raises(ValueError, match='feature-loss'):
        dataset(raw, store=tmp_path / 'store', feature_loss=True)
