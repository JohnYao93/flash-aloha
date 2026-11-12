"""The optional masking tuner: edits, rendering, saving and the episode source. Offline, no robot."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from config.masking_config import DEFAULT_MASKING_CONFIG, load_masking_config, masking_contract
from utils.masking import Masker
from utils.masking_tuner import MaskTuner

pytest.importorskip('PyKDL', reason='projected masks use the ARX kinematics (PyKDL)')
from test_masking import CAMERAS, H, REACHING, W, custom_config, images, write_bimanual_episode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/setup'))
import tune_masking  # noqa: E402


@pytest.fixture
def config(tmp_path):
    return custom_config(tmp_path, lambda document: None)


def test_moving_a_polygon_moves_its_mask_and_only_that_one(config):
    tuner = MaskTuner('end_effector', config)
    assert len(tuner.projects) == 4 and tuner.selected['camera'] == 'overhead' and not tuner.changed
    before = [p.copy() for p in tuner.masker.polygons(REACHING, 'video_overhead', (W, H))]
    wrist_before = [p.copy() for p in tuner.masker.polygons(REACHING, 'video_left', (W, H))]
    tuner.move('x', 5)  # the whole rectangle, 5 cm along +x of the arm base
    after = tuner.masker.polygons(REACHING, 'video_overhead', (W, H))
    assert tuner.changed and all(np.abs(a - b).max() > 2 for a, b in zip(after, before))
    for a, b in zip(tuner.masker.polygons(REACHING, 'video_left', (W, H)), wrist_before):
        np.testing.assert_array_equal(a, b)  # the other operations are untouched
    assert np.allclose(np.asarray(tuner.selected['corners'])[:, 0], np.array([-0.15, 0.1, 0.1, -0.35]) + 0.05)
    tuner.reset()
    assert not tuner.changed
    for a, b in zip(tuner.masker.polygons(REACHING, 'video_overhead', (W, H)), before):
        np.testing.assert_array_equal(a, b)


def test_corner_selection_scaling_and_step(config):
    tuner = MaskTuner('end_effector', config, step=0.02)
    tuner.select_corner()
    tuner.move('y', -1)
    corners = np.asarray(tuner.selected['corners'])
    assert tuner.corner == 0 and corners[0, 1] == pytest.approx(-0.17) and corners[1, 1] == pytest.approx(-0.15)
    for _ in range(4):
        tuner.select_corner()
    assert tuner.corner is None  # past the last corner: the whole polygon again
    centre = np.asarray(tuner.selected['corners']).mean(axis=0)
    spread = np.ptp(np.asarray(tuner.selected['corners']), axis=0)
    tuner.scale(2)
    np.testing.assert_allclose(np.asarray(tuner.selected['corners']).mean(axis=0), centre, atol=1e-6)
    np.testing.assert_allclose(np.ptp(np.asarray(tuner.selected['corners']), axis=0), spread * 2, atol=1e-5)
    tuner.scale_step(2)
    assert tuner.step == pytest.approx(0.04)
    tuner.select_operation(1)
    assert tuner.operation == 1 and tuner.selected['camera'] == 'wrist'
    tuner.select_operation(-2)
    assert tuner.operation == 3
    with pytest.raises(ValueError, match='no projected polygons'):
        MaskTuner('prioproceptive', config)


def test_render_shows_every_camera_with_the_mask_dimmed(config):
    tuner = MaskTuner('end_effector', config)
    frames = images()
    view = tuner.render(frames, REACHING, scale=2)
    assert view.shape == (24 + H * 2, W * 2 * len(CAMERAS), 3) and view.dtype == np.uint8
    assert all((frames[name] == 200).all() for name in CAMERAS)  # inputs untouched
    masked, _ = tuner.masker.mask(frames, REACHING)
    overhead = view[24:, W * 2:W * 4]  # second tile, in the order the frames were given
    hidden = (masked['video_overhead'] == 0).all(axis=2).repeat(2, axis=0).repeat(2, axis=1)
    interior = hidden & np.roll(hidden, 4, 0) & np.roll(hidden, -4, 0) & np.roll(hidden, 4, 1) & np.roll(hidden, -4, 1)
    assert np.median(overhead[interior]) == 70 and np.median(overhead[~hidden]) == 200  # dimmed mask, clear elsewhere
    assert 'op 1/4' in tuner.describe() and 'all corners' in tuner.describe()


def test_saving_writes_a_valid_config_and_updates_every_strategy_that_shares_the_polygon(config, tmp_path):
    tuner = MaskTuner('end_effector', config)
    tuner.move('z', -3)
    output = tuner.save()
    assert output == config.with_suffix('.tuned.yaml') and not tuner.changed
    tuned = load_masking_config(output)
    for strategy in ('end_effector', 'end_effector_and_prioproceptive'):
        assert tuned['strategies'][strategy][0]['project']['corners'][0][2] == pytest.approx(-0.13)
    assert tuned['strategies']['moving_block'] == load_masking_config(config)['strategies']['moving_block']
    assert 'definitions' not in yaml.safe_load(output.read_text())
    # The tuned polygons are another mask for existing checkpoints; the calibration is the same files.
    assert masking_contract('end_effector', output) != masking_contract('end_effector', config)
    assert masking_contract('moving_block', output) == masking_contract('moving_block', config)
    np.testing.assert_array_equal(Masker('end_effector', output).polygons(REACHING, 'video_overhead', (W, H))[0],
                                  tuner.masker.polygons(REACHING, 'video_overhead', (W, H))[0])
    # A second save after more edits still finds the operations, and another directory gets absolute paths.
    tuner.move('z', 1)
    elsewhere = tuner.save(tmp_path / 'elsewhere/masking.yaml')
    moved = load_masking_config(elsewhere)
    assert moved['strategies']['end_effector'][0]['project']['corners'][0][2] == pytest.approx(-0.12)
    assert Path(yaml.safe_load(elsewhere.read_text())['calibration']['arms']['left']['intrinsics']).is_absolute()
    assert Masker('end_effector', elsewhere).is_projected


def test_keys_drive_the_tuner_and_the_episode_source(config, tmp_path):
    episode = tmp_path / 'episode.zarr'
    qpos = write_bimanual_episode(episode)
    tuner = MaskTuner('end_effector', config)
    source = tune_masking.EpisodeSource(episode, tuner.masker.cameras, (W, H))
    try:
        frames, state, status = source.read()
        assert set(frames) == set(CAMERAS) and frames['video_overhead'].shape == (H, W, 3) and status == 'frame 1/4'
        np.testing.assert_array_equal(state, qpos[0])
        press = lambda character: tune_masking.handle_key(ord(character), tuner, source, tmp_path / 'saved.yaml')
        assert press('.') and source.read()[2] == 'frame 2/4'
        assert press(',') and press(',') and source.index == 0
        press(' ')
        assert source.playing and source.read()[2] == 'frame 2/4'
        press(']'); press('c'); press('Y'); press('+'); press('>')
        assert (tuner.operation, tuner.corner, tuner.step) == (1, 0, pytest.approx(0.02)) and tuner.changed
        press('r')
        assert not tuner.changed
        press('x')
        assert press('s') and (tmp_path / 'saved.yaml').is_file() and not tuner.changed
        assert tune_masking.handle_key(255, tuner, source, None)  # unmapped keys are ignored
        assert tuner.render(*source.read()[:2]).shape[1] == W * 2 * 3
        assert not press('q')
    finally:
        source.close()


def test_live_mode_uses_the_existing_arx_setup_and_the_safe_exit():
    import inspect
    source = inspect.getsource(tune_masking.run_live)
    assert 'from calibrate_arx_arm import build_controller, load_arx5_interface, safe_home, set_passive_mode' in source
    assert 'with damping_exit(lambda: ARXDamping(controllers) if controllers else None):' in source
    assert source.index('damping_exit(') < source.index('controller.reset_to_home()') < source.index('set_passive_mode(controller)')
    args = tune_masking.build_parser().parse_args(['end_effector', '--episode', 'x'])
    assert not args.live
    with pytest.raises(SystemExit):
        tune_masking.build_parser().parse_args(['end_effector'])  # a source is required
