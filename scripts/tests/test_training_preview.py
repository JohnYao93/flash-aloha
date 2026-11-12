"""Resolution-only offline previews preserve saved images, clocks and actions."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from test_visualize_episode_data import raw, episode_data
from visualization import training_preview as module
from visualization import visualize_episode as viewer
from visualization.episode_visualization import VisualizationEpisode

@pytest.fixture
def recipe(raw, tmp_path):
    task = dict(schema_version=1, task=dict(name='preview_test'),
                control_freq=100, data=dict(directory=str(raw)),
                robot=dict(type='arx', arms=dict(left=dict(can_interface='can0'), right=dict(can_interface='can1'))),
                cameras=dict(names=['video_overhead']), collection=dict(max_duration_seconds=1),
                training=dict(observations=['qpos'], outputs=['action'], state_dim=14, action_dim=14))
    task_path = tmp_path / 'task.yaml'
    task_path.write_text(yaml.safe_dump(task))
    config = dict(task=str(task_path), model_name='preview_test', model=dict(type='Flow', chunk_size=3),
                  training=dict(training_time_rtc=False, width=32, height=24))
    path = tmp_path / 'training.yaml'
    path.write_text(yaml.safe_dump(config))
    yield raw, path, config


def rewrite(path, config):
    path.write_text(yaml.safe_dump(config))



def test_native_resolution_is_preserved():
    pixels = np.zeros((800, 1280, 3), dtype=np.uint8)
    assert viewer.prepare_visualization_image(pixels) is pixels


def test_config_only_resizes_and_does_not_transform_trajectories(recipe, monkeypatch):
    raw, path, config = recipe
    config['training'].update(data_aug=True, randomize_color=True, grayscale=True,
                              randomize_data=True, use_mask=True)
    config['data'] = dict(directory='/nonexistent/training/data')
    config['curriculum'] = dict(camera_curriculum={'video_overhead': dict(initial_prob=1)})
    rewrite(path, config)
    preview = module.TrainingPreview(raw, path)
    import rerun as rr
    images, logs, times = [], {}, []
    monkeypatch.setattr(rr, 'Image', lambda image: images.append(image.copy()))
    monkeypatch.setattr(rr, 'Tensor', mock.Mock(side_effect=AssertionError('No tensors')))
    monkeypatch.setattr(rr, 'Scalars', lambda value: value)
    monkeypatch.setattr(rr, 'log', lambda name, value: logs.update({name: value}))
    monkeypatch.setattr(rr, 'set_time', lambda name, **value: times.append((name, value)))
    with VisualizationEpisode(raw, transform=preview.transform) as data:
        module.log_frame(rr, data, preview, 2)
        assert logs['recorded/action/dim_0'] == 3.
        assert logs['recorded/qpos/dim_0'] == 3.
        assert logs['recorded/qvel/dim_0'] == 3.
    assert len(images) == 1
    assert images[0].shape == (24, 32, 3) and images[0].dtype == np.uint8
    assert abs(images[0].mean() - 120) < 6  # Not camera dropout, despite recipe.
    assert dict(times)['episode_time']['duration'] == pytest.approx(.11)
    assert not any(name.startswith('training/') for name in logs)


def test_disabled_resize_keeps_native_pixels(recipe):
    raw, path, config = recipe
    config['training'].update(width=-1, height=-1)
    rewrite(path, config)
    preview = module.TrainingPreview(raw, path)
    pixels = np.arange(48*64*3, dtype=np.uint8).reshape(48, 64, 3)
    assert preview.transform(pixels) is pixels


def test_resize_matches_shared_operation_without_grayscale(recipe):
    from contracts.standard_image_processing import resize_image
    raw, path, config = recipe
    config['training'].update(grayscale=True, randomize_color=True, data_aug=True)
    rewrite(path, config)
    pixels = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    actual = module.TrainingPreview(raw, path).transform(pixels)
    np.testing.assert_array_equal(actual, resize_image(pixels, 32, 24))
    assert not np.array_equal(actual[..., 0], actual[..., 1])


def test_replay_conflict_fails_before_loading():
    with mock.patch.object(module, 'TrainingPreview') as preview:
        with pytest.raises(ValueError, match='offline resolution-only'):
            viewer.main(dict(training_config='unused', action_replay=True))
        preview.assert_not_called()


def test_prepared_viewer_opens_after_all_recorded_frames(recipe, tmp_path, monkeypatch):
    raw, path, _ = recipe
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    order = []
    original = module.log_frame
    def log(*args):
        order.append('frame')
        return original(*args)
    def open_file(output):
        assert output.stat().st_size > 1000
        assert order == ['frame'] * 4
        order.append('viewer')
    monkeypatch.setattr(module, 'log_frame', log)
    monkeypatch.setattr(module, 'open_prepared_preview', open_file)
    import rerun as rr
    monkeypatch.setattr(rr, 'spawn', mock.Mock(side_effect=AssertionError('No live streaming')))
    output = viewer.main(dict(data_dir=raw, training_config=str(path), rerun=True))
    assert order[-1] == 'viewer' and output.exists()


def test_failed_rebuild_preserves_finished_file(recipe, tmp_path, monkeypatch):
    raw, path, _ = recipe
    output = tmp_path / 'completed.rrd'
    output.write_bytes(b'previous complete preview')
    monkeypatch.setattr(module, 'log_frame', mock.Mock(side_effect=ValueError('test decoder failure')))
    with pytest.raises(ValueError, match='test decoder failure'):
        viewer.main(dict(data_dir=raw, training_config=str(path), save_rrd=output))
    assert output.read_bytes() == b'previous complete preview'
    assert not list(tmp_path.glob('.completed-*.rrd'))


def test_prepare_only_and_frame_bounds(recipe, tmp_path, monkeypatch):
    raw, path, _ = recipe
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(module, 'open_prepared_preview', mock.Mock(side_effect=AssertionError('No GUI')))
    output = viewer.main(dict(data_dir=raw, training_config=str(path), prepare_only=True, preview_frame=0))
    assert output.stat().st_size > 1000
    with pytest.raises(ValueError, match='outside the episode'):
        viewer.main(dict(data_dir=raw, training_config=str(path), save_rrd=output, preview_frame=999))
    with pytest.raises(ValueError, match='requires --training-config'):
        viewer.main(dict(data_dir=raw, prepare_only=True))


