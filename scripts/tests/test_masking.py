"""Image/proprioceptive masking: config-defined strategies, geometry, resolution and train/deploy wiring.

Offline, no robot.
"""
import shutil

import numpy as np
import pytest
import torch
import yaml

from config import masking_config
from config.masking_config import (DEFAULT_MASKING_CONFIG, load_masking_config, masking_contract,
                                   masking_strategy_names, require_masking_contract, require_masking_strategy)
from config.supported_policy import require_unmasked_for_serving
from data_processing import episode_store
from policy import training_utils
from utils.masking import Masker

pytest.importorskip('PyKDL', reason='projected masks use the ARX kinematics (PyKDL)')
import cv2  # noqa: E402

W, H = 320, 180
CAMERAS = ['video_left', 'video_overhead', 'video_right']
SHIPPED = ('end_effector', 'end_effector_and_prioproceptive', 'moving_block', 'prioproceptive_and_moving_block',
           'prioproceptive', 'arm_cameras', 'arm_cameras_and_prioproceptive', 'overhead_left_90px',
           'left_right_overhead')
# Two recorded bimanual poses (left arm, gripper, right arm, gripper), radians.
RETRACTED = np.array([0.006, 0.004, 0.193, -0.393, -0.088, -0.118, 0.001,
                      -0.151, 1.969, 1.563, -1.211, -0.152, -0.133, 0.015], dtype=np.float32)
REACHING = np.array([0.25, 1.4, 1.2, -0.6, 0.1, 0.0, 0.02,
                     -0.3, 1.5, 1.1, -0.8, 0.0, 0.1, 0.02], dtype=np.float32)


def images(value=200, size=(W, H)):
    return {name: np.full((size[1], size[0], 3), value, np.uint8) for name in CAMERAS}


def custom_config(tmp_path, edit):
    """A copy of the shipped masking config and calibration, edited before it is written."""
    root = tmp_path / 'masking'
    shutil.copytree(DEFAULT_MASKING_CONFIG.parent, root)
    document = yaml.safe_load((root / 'masking.yaml').read_text())
    document.pop('definitions', None)
    document['calibration']['urdf'] = load_masking_config()['calibration']['urdf']
    edit(document)
    (root / 'masking.yaml').write_text(yaml.safe_dump(document, sort_keys=False))
    return root / 'masking.yaml'


def test_shipped_strategies_are_defined_in_the_masking_config():
    assert masking_strategy_names() == SHIPPED
    assert require_masking_strategy(None) is None
    for strategy in SHIPPED:
        assert Masker(strategy).strategy == strategy
    for check in (require_masking_strategy, Masker):
        with pytest.raises(ValueError, match='Unknown masking strategy'):
            check('robot')
    with pytest.raises(ValueError, match='Remote serving cannot omit the mask'):
        require_unmasked_for_serving({'task_config': {'masking_strategy': 'end_effector'}})
    require_unmasked_for_serving({'task_config': {}})
    assert Masker('end_effector').is_projected and not Masker('end_effector').masks_proprioception
    assert Masker('prioproceptive').masks_proprioception and not Masker('prioproceptive').is_projected


def test_no_strategy_returns_the_inputs_untouched():
    frames, qpos = images(), torch.from_numpy(RETRACTED)
    masked, state = Masker(None).mask(frames, qpos)
    assert masked is frames and state is qpos and Masker(None).contract() is None


@pytest.mark.parametrize('strategy', [s for s in SHIPPED if s != 'prioproceptive'])
def test_masks_never_modify_their_inputs_and_do_not_depend_on_call_order(strategy):
    masker = Masker(strategy)
    frames, qpos = images(), torch.from_numpy(REACHING)
    first, _ = masker.mask(frames, qpos)
    assert all((frames[name] == 200).all() for name in CAMERAS)  # e.g. read-only frame-store arrays
    masker.mask(images(), torch.from_numpy(RETRACTED))  # another sample in between (random-access training)
    again, _ = masker.mask(frames, qpos)
    for name in CAMERAS:
        np.testing.assert_array_equal(first[name], again[name])
    assert any((first[name] == 0).any() for name in CAMERAS)


def test_overhead_rectangles_follow_each_wrist():
    masker = Masker('end_effector')
    for qpos in (RETRACTED, REACHING):
        wrists = masker.link_pixels(qpos, (W, H))
        polygons = dict(zip(('left', 'right'), masker.polygons(qpos, 'video_overhead', (W, H))))
        for name, (x, y) in wrists.items():
            # The wrist projects inside its own arm's rectangle and outside the other arm's.
            assert cv2.pointPolygonTest(polygons[name].reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
            other = 'right' if name == 'left' else 'left'
            assert cv2.pointPolygonTest(polygons[other].reshape(-1, 1, 2), (float(x), float(y)), False) < 0
    # Moving only the left arm moves only the left rectangle, by about the wrist's own displacement.
    moved = RETRACTED.copy()
    moved[:6] = REACHING[:6]
    before, after = (masker.polygons(q, 'video_overhead', (W, H)) for q in (RETRACTED, moved))
    np.testing.assert_array_equal(before[1], after[1])
    wrist_shift = masker.link_pixels(moved, (W, H))['left'] - masker.link_pixels(RETRACTED, (W, H))['left']
    polygon_shift = after[0].mean(0) - before[0].mean(0)
    assert np.linalg.norm(wrist_shift) > 20
    assert np.linalg.norm(polygon_shift - wrist_shift) < 0.35 * np.linalg.norm(wrist_shift)


def test_overhead_mask_is_the_filled_rectangles_and_the_state_is_kept():
    masker = Masker('end_effector')
    masked, qpos = masker.mask(images(), torch.from_numpy(REACHING))
    expected = np.full((H, W, 3), 200, np.uint8)
    for points in masker.polygons(REACHING, 'video_overhead', (W, H)):
        cv2.fillPoly(expected, [points.reshape(-1, 1, 2)], (0, 0, 0))
    np.testing.assert_array_equal(masked['video_overhead'], expected)
    assert 0 < (masked['video_overhead'] == 0).all(axis=2).mean() < 1
    assert torch.equal(qpos, torch.from_numpy(REACHING))  # image masks leave the state alone


@pytest.mark.parametrize('scale', [2, 4])
def test_masks_are_resolution_independent(scale):
    masker, size = Masker('end_effector'), (W * scale, H * scale)
    for camera in CAMERAS:
        for small, large in zip(masker.polygons(REACHING, camera, (W, H)), masker.polygons(REACHING, camera, size)):
            visible = (np.abs(small) < 2900).all(axis=1)  # corners clamped far off-screen do not scale
            assert np.abs(large[visible] - small[visible] * scale).max() <= scale  # integer rounding only
    # The masked fraction of the image is the same at every resolution, and regions are fractions.
    low, _ = masker.mask(images(), REACHING)
    high, _ = masker.mask(images(size=size), REACHING)
    for camera in CAMERAS:
        assert abs((low[camera] == 0).mean() - (high[camera] == 0).mean()) < 0.02
    columns, _ = Masker('left_right_overhead').mask(images(size=size), RETRACTED)
    assert not columns['video_overhead'][:, :90 * scale].any() and not columns['video_overhead'][:, 235 * scale:].any()
    assert (columns['video_overhead'][:, 90 * scale:235 * scale] == 200).all()
    # A non-uniform resize (another aspect ratio) rescales each axis by its own factor.
    wide = masker.polygons(REACHING, 'video_overhead', (W * 2, H))
    for small, stretched in zip(masker.polygons(REACHING, 'video_overhead', (W, H)), wide):
        assert np.abs(stretched[:, 0] - small[:, 0] * 2).max() <= 2 and np.abs(stretched[:, 1] - small[:, 1]).max() <= 1


def test_masks_accept_frames_that_are_views():
    # Readers hand out channel-flipped or strided views; the mask is the same as for a contiguous copy.
    masker = Masker('end_effector')
    contiguous = {name: np.random.default_rng(1).integers(1, 255, (H, W, 3), dtype=np.uint8) for name in CAMERAS}
    flipped = {name: np.ascontiguousarray(frame[..., ::-1])[..., ::-1] for name, frame in contiguous.items()}
    assert not flipped['video_overhead'].flags['C_CONTIGUOUS']
    want, _ = masker.mask(contiguous, REACHING)
    got, _ = masker.mask(flipped, REACHING)
    for name in CAMERAS:
        np.testing.assert_array_equal(got[name], want[name])


def test_fixed_and_camera_masks():
    frames = images()
    left, _ = Masker('overhead_left_90px').mask(frames, RETRACTED)
    assert not left['video_overhead'][:, :90].any() and (left['video_overhead'][:, 90:] == 200).all()
    both, _ = Masker('left_right_overhead').mask(frames, RETRACTED)
    assert not both['video_overhead'][:, 235:].any() and (both['video_overhead'][:, 90:235] == 200).all()
    arms, qpos = Masker('arm_cameras_and_prioproceptive').mask(frames, torch.from_numpy(RETRACTED))
    assert not arms['video_left'].any() and not arms['video_right'].any() and (arms['video_overhead'] == 200).all()
    assert not qpos.any()
    block, _ = Masker('moving_block').mask(frames, RETRACTED)
    assert not block['video_overhead'][:H // 4].any() and (block['video_overhead'][H // 4:] == 200).all()


def test_a_custom_config_defines_new_strategies_cameras_and_a_single_arm(tmp_path):
    def edit(document):
        calibration = document['calibration']
        calibration['arms'] = {'right': calibration['arms']['right']}  # a single-arm setup
        document['strategies'] = {
            'my_wrist_disc': [{'project': {'camera': 'overhead', 'frame': 'wrist', 'arm': 'self', 'corners': [
                [float(0.08 * np.cos(a)), float(0.08 * np.sin(a)), 0.0]
                for a in np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()]}}],
            'hide_other_arm': [{'project': {'camera': 'wrist', 'frame': 'end_effector', 'arm': 'other',
                                            'corners': [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]}}],
            'bottom_strip': [{'region': {'camera': 'extra_camera', 'x': [0, 1], 'y': [0.8, 1]}}, {'blank': 'video_right'}]}
    path = custom_config(tmp_path, edit)
    assert masking_strategy_names(path) == ('my_wrist_disc', 'hide_other_arm', 'bottom_strip')
    single = RETRACTED[7:]  # the one arm's seven values
    disc = Masker('my_wrist_disc', path)
    assert disc.arm_names == ('right',) and disc.cameras == ('video_right', 'video_overhead')
    masked, _ = disc.mask({'video_overhead': np.full((H, W, 3), 200, np.uint8)}, single)
    (polygon,) = disc.polygons(single, 'video_overhead', (W, H))
    x, y = disc.link_pixels(single, (W, H))['right']
    assert len(polygon) == 12 and cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
    assert 0 < (masked['video_overhead'] == 0).all(axis=2).mean() < 0.2
    # The same wrist as in the bimanual config: the single-arm state indexes its own arm.
    np.testing.assert_array_equal(disc.link_pixels(single, (W, H))['right'],
                                  Masker('end_effector').link_pixels(RETRACTED, (W, H))['right'])
    # With one arm there is no "other" arm: that operation draws nothing.
    untouched, _ = Masker('hide_other_arm', path).mask(images(), single)
    assert all((untouched[name] == 200).all() for name in CAMERAS)
    strip, _ = Masker('bottom_strip', path).mask({**images(), 'extra_camera': np.full((100, 50, 3), 9, np.uint8)}, single)
    assert not strip['extra_camera'][80:].any() and (strip['extra_camera'][:80] == 9).all()
    assert not strip['video_right'].any()
    with pytest.raises(ValueError, match="declares this task's arms"):
        Masker('end_effector').mask(images(), single)  # the bimanual config needs both arms' state


def test_the_environment_selects_the_masking_config(tmp_path, monkeypatch):
    path = custom_config(tmp_path, lambda document: document.update(
        strategies={'only_this': [{'zero': 'proprioception'}]}))
    monkeypatch.setenv(masking_config.ENVIRONMENT_VARIABLE, str(path))
    assert masking_strategy_names() == ('only_this',)
    with pytest.raises(ValueError, match='Unknown masking strategy'):
        Masker('end_effector')
    assert Masker('only_this').masks_proprioception


@pytest.mark.parametrize('broken,message', [
    (lambda d: d['strategies'].update(bad=[{'project': {'camera': 'side', 'frame': 'wrist', 'arm': 'self',
                                                        'corners': [[0, 0, 0]] * 3}}]), 'camera is overhead'),
    (lambda d: d['strategies'].update(bad=[{'region': {'camera': 'overhead', 'x': [0.5, 0.2], 'y': [0, 1]}}]), 'fractions'),
    (lambda d: d['strategies'].update(bad=[{'erase': 'everything'}]), 'must be one of'),
    (lambda d: d['calibration'].update(image_size=[320]), 'image_size'),
    (lambda d: d['calibration']['arms']['left'].pop('intrinsics'), 'missing keys'),
    (lambda d: d.pop('calibration'), 'need a calibration section'),
])
def test_invalid_masking_configs_fail_when_loaded(tmp_path, broken, message):
    with pytest.raises(ValueError, match=message):
        load_masking_config(custom_config(tmp_path, broken))
    with pytest.raises(FileNotFoundError, match='Masking config not found'):
        load_masking_config(tmp_path / 'absent.yaml')


def test_checkpoint_contract_pins_the_strategy_definition_and_calibration(tmp_path):
    contract = masking_contract('end_effector')
    assert contract == Masker('end_effector').contract() and contract['strategy'] == 'end_effector'
    assert require_masking_contract(contract, 'end_effector') == contract
    assert require_masking_contract(None, None) is None
    with pytest.raises(ValueError, match='stores no masking contract'):
        require_masking_contract(None, 'end_effector')
    with pytest.raises(ValueError, match='but no masking strategy'):
        require_masking_contract(contract, None)
    # A changed rectangle, or a recalibrated file, is another mask: deployment refuses it.
    moved = custom_config(tmp_path / 'geometry', lambda d: d['strategies']['end_effector'][0]['project'][
        'corners'].__setitem__(0, [-0.2, -0.15, -0.1]))
    with pytest.raises(ValueError, match='Masking contract mismatch'):
        require_masking_contract(contract, 'end_effector', moved)
    recalibrated = custom_config(tmp_path / 'calibration', lambda d: None)
    assert masking_contract('end_effector', recalibrated) == contract  # the same definition and files
    file = recalibrated.parent / 'head-to-robot_video_left_hand/head_to_robot_base.npz'
    transform = np.load(file)['average_transform'].copy()
    transform[0, 3] += 0.01
    np.savez(file, average_transform=transform)
    with pytest.raises(ValueError, match='Masking contract mismatch'):
        require_masking_contract(contract, 'end_effector', recalibrated)
    # Strategies without projection do not depend on the calibration files.
    assert masking_contract('prioproceptive', recalibrated) == masking_contract('prioproceptive')


def write_bimanual_episode(path, rows=4):
    rng = np.random.default_rng(0)
    qpos = np.stack([RETRACTED + (REACHING - RETRACTED) * i / (rows - 1) for i in range(rows)]).astype(np.float32)
    with episode_store.Store(path, 'w') as root:
        root.attrs['compress'] = False
        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos)
        obs.create_dataset('qvel', data=np.zeros_like(qpos))
        group = obs.create_group('images')
        for name in CAMERAS:
            group.create_dataset(name, data=rng.integers(1, 255, (rows, H, W, 3), dtype=np.uint8))
        root.create_dataset('action', data=qpos + 0.1)
    from episode_fixtures import attach_streams
    attach_streams(path)
    return qpos


def dataset(path, strategy, **kwargs):
    stats, lengths = training_utils.get_norm_stats([str(path)], observation_name=['qpos'])
    return training_utils.EpisodicDataset(strategy, [str(path)], CAMERAS, stats, np.array([0]), lengths, 3, 'ACT',
                                          observation_name=['qpos'], **kwargs)


@pytest.mark.parametrize('size', [(W, H), (160, 90)])
def test_training_samples_carry_the_mask_of_their_own_row_at_the_policy_resolution(tmp_path, size):
    path = tmp_path / 'episode.zarr'
    qpos = write_bimanual_episode(path)
    resize = {} if size == (W, H) else dict(width=size[0], height=size[1])
    plain, masked = dataset(path, None, **resize), dataset(path, 'end_effector', **resize)
    masker = Masker('end_effector')
    for row in (2, 0, 3):  # random access: each sample's mask comes from its own qpos
        image, state, _, _ = masked[row]
        reference, reference_state, _, _ = plain[row]
        torch.testing.assert_close(state, reference_state)  # normalized as without a mask
        assert image.shape == reference.shape == (3, 3, size[1], size[0])
        for index, name in enumerate(CAMERAS):
            canvas = np.full((size[1], size[0], 3), 255, np.uint8)
            for points in masker.polygons(qpos[row], name, size):
                cv2.fillPoly(canvas, [points.reshape(-1, 1, 2)], (0, 0, 0))
            hidden = (canvas == 0).all(axis=2)
            assert not image[index][:, hidden].any()
            torch.testing.assert_close(image[index][:, ~hidden], reference[index][:, ~hidden])
            if name == 'video_overhead':
                assert 0 < hidden.mean() < 1  # part of the overhead view stays visible


def test_proprioceptive_masks_zero_the_state_without_normalizing_it(tmp_path):
    path = tmp_path / 'episode.zarr'
    write_bimanual_episode(path)
    image, state, action, _ = dataset(path, 'prioproceptive')[1]
    reference, _, reference_action, _ = dataset(path, None)[1]
    assert not state.any()
    torch.testing.assert_close(image, reference)
    torch.testing.assert_close(action, reference_action)


def test_recipes_declare_the_mask_and_deployment_cannot_change_it(tmp_path):
    from config.task_config import CONFIG_ROOT
    from config.training_args import build_parser
    from contracts import standard_image_processing
    recipe = yaml.safe_load((CONFIG_ROOT / 'training/bimanual_act.yaml').read_text())
    recipe.update(task=str(CONFIG_ROOT / 'tasks/bimanual.yaml'), output_dir=str(tmp_path))
    recipe['data'] = {'masking_strategy': 'end_effector'}
    path = tmp_path / 'masked.yaml'
    path.write_text(yaml.safe_dump(recipe))
    args = build_parser().parse_args(['--training-config', str(path)])
    assert args._training_task_options['masking_strategy'] == 'end_effector'
    recipe['data'] = {'masking_strategy': 'robot'}
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--training-config', str(path)])
    # Deployment may confirm the saved mask but never change it.
    config = {'task_config': {'masking_strategy': 'end_effector'}}
    assert standard_image_processing.masking_strategy(config) == 'end_effector'
    with pytest.raises(ValueError, match='differs from training'):
        standard_image_processing.masking_strategy(config, 'moving_block')
