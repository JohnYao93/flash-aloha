"""Raw training/viewing integration with generated videos and no live hardware."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
from data_processing import episode_store
import numpy as np
import pytest
import torch
import zarr

from data_processing.episode_data import Episode, clear_episode_cache, find_episodes, open_episode
from data_processing.video_processing import ExactVideo


@pytest.fixture(autouse=True)
def caches(tmp_path, monkeypatch):
    monkeypatch.setenv('FLASH_ALOHA_VIDEO_INDEX_CACHE', str(tmp_path / 'indices'))
    clear_episode_cache()
    yield
    clear_episode_cache()


def make_raw(path, *, keep=None, frame_count=32):
    path.mkdir(parents=True)
    count = 9
    names = ['video_left', 'video_overhead']
    manifest = dict(complete=True, retained=True, alignable=True, sample_count=count,
                    fps=100, camera_names=names, metadata=dict(task_name='unit', robot_type='arx'))
    (path / 'manifest.json').write_text(json.dumps(manifest))
    (path / 'alignment_report.json').write_text(json.dumps(dict(alignable=True, errors=[], retained_robot_samples=len(keep) if keep is not None else count)))
    if keep is not None:
        (path / 'edits.json').write_text(json.dumps(dict(retained_robot_indices=keep)))
    control = zarr.open_group(str(path / 'control.zarr'), mode='w')
    control.attrs['action_command_style'] = 'absolute'
    for name, value in dict(qpos=np.arange(count*7,dtype='f4').reshape(count,7)/100,
        qvel=np.ones((count,7), 'f4'), commanded_action=np.arange(count*7,dtype='f4').reshape(count,7)/50,
        timestamps=100.+np.arange(count)*.01, elapsed_seconds=np.arange(count)*.01,
        continuity_id=np.zeros(count,dtype='i8'), metadata_id=np.zeros(count,dtype='i8'),
        intervention_active=np.zeros(count,dtype='u1')).items():
        control.create_dataset(name, data=value)
    control.create_dataset('camera_observation/snapshot_timestamp', data=100.+np.arange(count)*.01+.001)
    for camera in names:
        video = path / f'{camera}.zarr/main.mp4'
        video.parent.mkdir()
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 100, (64,48))
        assert writer.isOpened()
        for i in range(frame_count):
            image = np.zeros((48,64,3),np.uint8)
            image[:,:] = [i % 240, (i*3) % 240, (i*7) % 240]
            image[:, :i % 64, :] = [10,80,140]
            writer.write(image)
        writer.release()
        (path / f'{camera}.timing.jsonl').write_text(''.join(json.dumps(dict(video_frame=i,
            frame_number=1000+i,capture_timestamp=99.995+i*.005,continuity_id=0))+'\n' for i in range(frame_count)))
        refs = np.array([0,1,1,3,4,5,6,7,8])
        for key, value in dict(frame_number=1000+refs, capture_timestamp=99.995+refs*.005,
            consumer_received_timestamp=100.+np.arange(count)*.01).items():
            control.create_dataset(f'camera_observation/{camera}/{key}', data=value)
    return path


@pytest.fixture
def raw(tmp_path):
    return make_raw(tmp_path / 'data/raw/episode')





def test_raw_training_and_viewer_return_identical_exact_frames(raw, monkeypatch):
    from policy import training_utils as training
    from visualization.episode_visualization import VisualizationEpisode
    stats, lengths = training.get_norm_stats([str(raw)], observation_name=['qpos'])
    dataset = training.EpisodicDataset(None, [str(raw)], ['video_left','video_overhead'],
        stats, [0], lengths, 4, 'Flow', observation_name=['qpos'], width=64, height=48)
    with VisualizationEpisode(raw) as viewer:
        images, state, actions, padding = dataset[2]
        expected = np.stack([cv2.cvtColor(viewer.images[name][2], cv2.COLOR_BGR2RGB).transpose(2,0,1) for name in dataset.camera_names]) / 255.
        np.testing.assert_allclose(images.numpy(), expected, atol=1e-7)
        np.testing.assert_array_equal(viewer.images['video_left'][1], viewer.images['video_left'][2])
        np.testing.assert_allclose(state.numpy(), (viewer.qpos[2]-stats['qpos_mean'])/stats['qpos_std'], atol=1e-6)
        np.testing.assert_allclose(actions.numpy(), (viewer.actions[2:6]-stats['action_mean'])/stats['action_std'], atol=1e-6)
        assert not padding.any()
    assert len(list(raw.parent.rglob('*.mp4'))) == 2


def test_published_episode_matches_stream_tensor_preprocessing(raw, tmp_path, monkeypatch):
    from policy import training_utils as training
    published = tmp_path / 'published.zarr'
    from data_processing.episode_store import publish_episode
    publish_episode(published, [(raw, np.arange(9))])
    stats, lengths = training.get_norm_stats([str(raw)], observation_name=['qpos','qvel'])
    args = dict(masking_strategy=None, camera_names=['video_left','video_overhead'], norm_stats=stats,
        episode_ids=[0], episode_len=lengths, chunk_size=3, policy_class='Flow', width=27, height=19,
        observation_name=['qpos','qvel'], history_len=2, grayscale=True)
    raw_data = training.EpisodicDataset(dataset_path_list=[str(raw)], **args)
    published_data = training.EpisodicDataset(dataset_path_list=[str(published)], **args)
    for index in (0,2,5,8):
        for actual, expected in zip(raw_data[index], published_data[index]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_reads_stream_data_directly(raw, monkeypatch):
    from policy import training_utils as training
    from contracts.action_rate_contract import dataset_rate
    assert dataset_rate([str(raw)]) == 100
    train, val, stats, *_ = training.load_data(0, None, str(raw.parent.parent), lambda _: True,
        ['video_left','video_overhead'], 2, 2, 3, policy_class='Flow', observation_name=['qpos'], width=32, height=24,
        train_ratio=0.99)
    assert len(next(iter(train))) == 4
    assert len(next(iter(val))) == 4
    # Default train_ratio=1.0 trains on every episode and builds no validation loader.
    train, val, *_ = training.load_data(0, None, str(raw.parent.parent), lambda _: True,
        ['video_left','video_overhead'], 2, 2, 3, policy_class='Flow', observation_name=['qpos'], width=32, height=24)
    assert len(next(iter(train))) == 4
    assert val is None


def test_raw_workers_have_independent_decoders(raw, monkeypatch):
    from policy import training_utils as training
    train, val, *_ = training.load_data(2, None, str(raw.parent.parent), lambda _: True,
        ['video_left','video_overhead'], 2, 2, 3, policy_class='Flow', observation_name=['qpos'], width=32, height=24)
    iterator = iter(train)
    for _ in range(3):
        assert next(iterator)[0].shape == (2,2,3,24,32)
    iterator._shutdown_workers()


def test_retained_gaps_break_action_chunks_and_history(tmp_path, monkeypatch):
    from policy import training_utils as training
    raw = make_raw(tmp_path / 'episode', keep=[0,1,4,5,8])
    stats, lengths = training.get_norm_stats([str(raw)])
    dataset = training.EpisodicDataset(None,[str(raw)],['video_left'],stats,[0],lengths,4,'Flow',
        observation_name=['qpos'], history_len=1, width=64,height=48)
    assert dataset[1][3].tolist() == [False,True,True,True]
    with open_episode(raw) as data:
        assert data.runs.tolist() == [0,0,1,1,2]


@pytest.mark.parametrize('failure', ['incomplete','unaligned','missing_frame','duplicate_indices','missing_camera'])
def test_invalid_raw_data_fails_before_training(raw, failure):
    if failure in ('incomplete','unaligned'):
        path=raw/'manifest.json'; doc=json.loads(path.read_text())
        doc['complete' if failure=='incomplete' else 'alignable']=False
        path.write_text(json.dumps(doc))
    elif failure == 'missing_frame':
        path=raw/'video_left.timing.jsonl'; lines=path.read_text().splitlines(); lines[1]=lines[1].replace('1001','9999'); path.write_text('\n'.join(lines))
    elif failure == 'missing_camera':
        (raw/'video_left.zarr/main.mp4').unlink()
    else:
        (raw/'edits.json').write_text(json.dumps(dict(retained_robot_indices=[0,0])))
    with pytest.raises(ValueError): Episode(raw)


def test_discovery_prefers_raw_over_duplicate_exports(raw):
    duplicate=raw.parent.parent/'episode_00000.zarr'; duplicate.write_bytes(b'not opened')
    assert find_episodes(raw.parent.parent) == [str(raw)]




def test_review_exclusions_keep_exact_references_and_break_sequences(raw):
    from data_processing.dagger_review import source_fingerprint
    control = zarr.open_group(str(raw / 'control.zarr'), mode='a')
    control['metadata_id'][2:4] = 1
    (raw / 'review.json').write_text(json.dumps(dict(discarded_segments=['2:4'],
        source_fingerprint=source_fingerprint(control))))
    with Episode(raw) as data:
        assert data.retained_indices.tolist() == [0,1,4,5,6,7,8]
        assert data.runs.tolist() == [0,0,1,1,1,1,1]
        assert data.images['video_left'].indices.tolist() == [0,1,4,5,6,7,8]


def test_missing_requested_observation_does_not_create_fake_statistics(raw):
    from policy import training_utils as training
    with pytest.raises(KeyError):
        training.get_norm_stats([str(raw)], observation_name=['missing'])




def test_camera_subset_does_not_decode_unused_stream(raw, monkeypatch):
    with Episode(raw) as data:
        monkeypatch.setattr(data.images['video_overhead'].video, 'get',
                            lambda _: pytest.fail('Unused camera decoded'))
        assert list(data.frame_bundle(0, ['video_left'])) == ['video_left']


def test_b_frame_video_random_seeks_preserve_presentation_order(tmp_path):
    import av
    path = tmp_path / 'bframes.mp4'
    with av.open(str(path), mode='w') as output:
        stream = output.add_stream('libx264', rate=100)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        stream.options = {'g': '12', 'bf': '2'}
        rng = np.random.default_rng(42)
        for _ in range(96):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (48,64,3), dtype=np.uint8), format='bgr24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    capture = cv2.VideoCapture(str(path))
    expected = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        expected.append(frame)
    capture.release()
    assert len(expected) == 96
    video = ExactVideo(path, cache_bytes=0)
    try:
        for index in [85,4,70,71,2,95,0]:
            np.testing.assert_array_equal(video.get(index), expected[index])
    finally:
        video.close()


def test_indexed_seeks_match_sequential_decoding_and_cache_invalidates(tmp_path):
    raw=make_raw(tmp_path/'episode', frame_count=240)
    path=raw/'video_left.zarr/main.mp4'
    capture=cv2.VideoCapture(str(path)); expected=[]
    while True:
        ok, image=capture.read()
        if not ok: break
        expected.append(image)
    capture.release()
    video=ExactVideo(path,cache_bytes=0)
    for index in [200,5,170,2,239,80,180]:
        np.testing.assert_array_equal(video.get(index),expected[index])
    assert video.decoded_frames < 240  # Random seeks do not decode from frame zero.
    video.close()
    assert list((tmp_path/'indices').glob('*.npy'))
    before=len(list((tmp_path/'indices').glob('*.npy')))
    import os
    stat=path.stat(); os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    other=ExactVideo(path,cache_bytes=0); other.get(170); other.close()
    assert len(list((tmp_path/'indices').glob('*.npy'))) == before+1
