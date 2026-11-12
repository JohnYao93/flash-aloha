"""Exact raw frame alignment and Rerun timestamps, without hardware access."""
import json
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from visualization.episode_visualization import VisualizationEpisode, resolve_episode_directory
@pytest.fixture
def episode_data(tmp_path):
    sid = 'synthetic'
    raw = tmp_path / 'data' / 'streams' / sid
    raw.mkdir(parents=True)
    video = raw / 'video_overhead.zarr' / 'main.mp4'
    video.parent.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    (raw / 'manifest.json').write_text(json.dumps(dict(state='complete', complete=True,
        alignable=True, camera_names=['video_overhead'], sample_count=5, fps=100,
        metadata=dict(session_id=sid, collection='generic'))))
    (raw / 'task_manifest.json').write_text(json.dumps(dict(session_id=sid, stage='complete')))
    (raw / 'alignment_report.json').write_text(json.dumps(dict(alignable=True, errors=[], retained_robot_samples=4)))
    (raw / 'edits.json').write_text(json.dumps(dict(retained_robot_indices=[0, 1, 3, 4])))
    # Deliberately shuffled label rows: joins must use raw_index, never row number.
    (raw / 'collection_labels.jsonl').write_text(''.join(json.dumps(dict(raw_index=i,
        label=dict(stage='searching', step_index=0, target=f'raw-{i}', episode_index=0)))+'\n'
        for i in [4, 2, 0, 3, 1]))
    (raw / 'video_overhead.timing.jsonl').write_text(''.join(json.dumps(dict(video_frame=i,
        frame_number=100+i, capture_timestamp=10+i/30, continuity_id=0))+'\n' for i in range(10)))
    control = zarr.open_group(str(raw / 'control.zarr'), mode='w')
    for name, values in dict(timestamps=[10, 11, 12, 13, 14], continuity_id=[0]*5,
            elapsed_seconds=[5., 5.02, 5.03, 5.11, 5.5],
            sampled_at=[1000.1, 1000.2, 1000.3, 1000.5, 1001.]).items():
        control.create_dataset(name, data=np.asarray(values))
    references = control.require_group('camera_observation/video_overhead')
    for name, values in dict(frame_number=[101, 101, 103, 106, 108],
            capture_timestamp=[10+i/30 for i in [1, 1, 3, 6, 8]],
            consumer_received_timestamp=[10, 11, 12, 13, 14]).items():
        references.create_dataset(name, data=np.asarray(values))
    yield None, sid, None, None, raw


@pytest.fixture
def raw(episode_data):
    directory = episode_data[-1]
    control = zarr.open_group(str(directory/'control.zarr'), mode='a')
    for name in ('qpos', 'qvel', 'commanded_action'):
        control.create_dataset(name, data=np.repeat(np.arange(5)[:, None], 14, axis=1).astype('f4'))
    return directory


def test_raw_visualization_resolves_episode_and_selects_retained_observations(raw):
    assert resolve_episode_directory(raw) == raw
    assert resolve_episode_directory(raw.parents[1]) == raw
    assert resolve_episode_directory(raw.parents[2]) == raw
    with VisualizationEpisode(raw.parents[2]) as data:
        np.testing.assert_array_equal(data.qpos[:, 0], [0, 1, 3, 4])
        np.testing.assert_array_equal(data.actions, data.qpos)
        assert data.timestamps == pytest.approx([0, .02, .11, .5])
        video = data.images['video_overhead']
        assert video.capture is None  # Lazy opening, no images held for the whole episode.
        for index, expected in [(0, 20), (1, 20), (2, 120), (3, 160), (0, 20)]:
            assert abs(video[index].mean()-expected) < 6
        assert video.position == 8  # Earlier frame came from cache; decoder did not restart.
    assert video.capture is None and video.image is None


def test_visualization_rejects_missing_reference_without_timestamp_fallback(raw):
    path = raw/'video_overhead.timing.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[6]['frame_number'] = 999
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='referenced frame'):
        VisualizationEpisode(raw)


def test_truncated_video_fails_and_closes_decoder(raw):
    path = raw/'video_overhead.zarr/main.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64,48))
    for _ in range(3):
        writer.write(np.zeros((48,64,3),np.uint8))
    writer.release()
    with VisualizationEpisode(raw) as data:
        video = data.images['video_overhead']
        assert video[0].shape == (48,64,3)
        with pytest.raises(ValueError, match='Missing referenced video frame 6'):
            video[2]
        assert video.capture is None


def test_rerun_uses_saved_seconds_and_logs_every_retained_observation(raw, monkeypatch):
    from visualization import visualize_episode as visualization
    times, images, logs = [], [], []
    monkeypatch.setattr(visualization.rr, 'set_time', lambda name, **value: times.append((name,value)))
    monkeypatch.setattr(visualization.rr, 'log', lambda name, value: logs.append(name))
    monkeypatch.setattr(visualization.rr, 'Image', lambda image: images.append(image.copy()) or mock.Mock())
    masker = mock.Mock()
    masker.mask.side_effect = lambda images, qpos: (images,qpos)
    with VisualizationEpisode(raw) as data:
        visualization.run_arx_data_rerun_visualization({}, data.qpos, data.images, data.qvel,
            data.actions, {}, True, observation_timestamps=data.timestamps,
            raw_indices=data.retained_indices)
    assert [value['duration'] for name,value in times if name == 'episode_time'] == pytest.approx([0,.02,.11,.5])
    assert [value['sequence'] for name,value in times if name == 'frame'] == [0,1,2,3]
    assert len(images) == 4
    assert logs.count('timing/sample_interval_ms') == 3
    assert logs.count('timing/robot_index') == 4
    assert logs.count('processed_action/right_arm/joint_5') == 4


def test_raw_visualization_rejects_physical_replay_before_loading(raw, monkeypatch):
    from visualization import visualize_episode as visualization
    with mock.patch.object(visualization, 'run_rerun_visualization') as run:
        with pytest.raises(ValueError, match='visualization only'):
            visualization.main(dict(robot='arx',data_dir=raw,action_replay=True))
        run.assert_not_called()


def test_save_rrd_does_not_spawn_viewer(raw, tmp_path, monkeypatch):
    from visualization import visualize_episode as visualization
    output = tmp_path/'episode.rrd'
    monkeypatch.setattr(visualization.rr, 'spawn', mock.Mock(side_effect=AssertionError('Do not spawn')))
    monkeypatch.setattr(visualization, 'open_rrd', mock.Mock(side_effect=AssertionError('Do not open')))
    visualization.main(dict(robot='arx',data_dir=raw,save_rrd=output))
    assert output.stat().st_size > 1000


def test_episode_cli_still_requires_task_and_index(monkeypatch):
    from visualization import visualize_episode as visualization
    monkeypatch.setattr(sys, 'argv', ['visualization/visualize_episode.py','--robot','arx','--rerun'])
    with pytest.raises(SystemExit):
        visualization.parse_args()
    monkeypatch.setattr(sys, 'argv', ['visualization/visualize_episode.py','--robot','arx','--data-dir','/example'])
    assert visualization.parse_args()['data_dir'] == Path('/example')


def fake_meminfo(available_gib, total_gib=32):
    return lambda: dict(MemTotal=total_gib * 1024**3, MemAvailable=int(available_gib * 1024**3))


def test_memory_guard_floor_and_viewer_limit(monkeypatch):
    from visualization.visualization_utils.memory_guard import GIB, MemoryGuard
    monkeypatch.delenv('FLASH_ALOHA_MIN_FREE_MEMORY_GB', raising=False)
    assert MemoryGuard(read=fake_meminfo(20)).floor == 8 * GIB  # 25% of total
    assert MemoryGuard(read=fake_meminfo(20, total_gib=8)).floor == 4 * GIB  # 4 GiB minimum
    guard = MemoryGuard(read=fake_meminfo(20))
    assert guard.viewer_limit() == 12 * GIB
    with pytest.raises(MemoryError, match='robot processes'):
        MemoryGuard(read=fake_meminfo(7.9)).check()
    monkeypatch.setenv('FLASH_ALOHA_MIN_FREE_MEMORY_GB', '2')
    assert MemoryGuard(read=fake_meminfo(20)).floor == 2 * GIB


def test_rerun_saves_file_then_opens_it_instead_of_streaming(raw, tmp_path, monkeypatch):
    from visualization import visualize_episode as visualization
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(visualization.rr, 'spawn', mock.Mock(side_effect=AssertionError('Do not stream')))
    opened = mock.Mock()
    monkeypatch.setattr(visualization, 'open_rrd', opened)
    visualization.main(dict(robot='arx', data_dir=raw, rerun=True))
    path = opened.call_args.args[0]
    assert path.parent == tmp_path / 'cache/flash-aloha/episode-views'
    assert path.stat().st_size > 1000
    assert list(path.parent.iterdir()) == [path]  # No temporary file left behind.


def test_export_stops_below_memory_floor_and_keeps_no_partial_file(raw, tmp_path, monkeypatch):
    from visualization import visualize_episode as visualization
    from visualization.visualization_utils import memory_guard
    readings = iter([20, 20, 20, 3])  # Constructor, start check, frame 0, frame 1.
    monkeypatch.setattr(memory_guard, 'meminfo',
                        lambda: dict(MemTotal=32 * 1024**3, MemAvailable=next(readings) * 1024**3))
    monkeypatch.setattr(visualization, 'open_rrd', mock.Mock(side_effect=AssertionError('Do not open')))
    output = tmp_path / 'episode.rrd'
    with pytest.raises(MemoryError, match='frame 1/4'):
        visualization.main(dict(robot='arx', data_dir=raw, save_rrd=output))
    assert list(tmp_path.glob('*.rrd')) == [] and list(tmp_path.glob('.*.rrd')) == []


def test_lazy_frames_decode_one_frame_per_access():
    from visualization.visualize_episode import LazyFrames
    calls = []
    frames = LazyFrames(np.arange(5), lambda value: calls.append(value) or value * 10)
    assert len(frames) == 5 and calls == []
    assert frames[-1] == 40 and frames[1:3] == [10, 20]
    assert calls == [4, 1, 2]
