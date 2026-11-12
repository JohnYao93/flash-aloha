"""Offline task YAML and artifact compatibility checks; no robot operation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from data_processing import episode_store
import pytest

from conftest import TASK_FIXTURES
import yaml

from config import constants
from config.task_config import CONFIG_ROOT, load_task_config


TASK = TASK_FIXTURES / 'arx_right_wide_state.yaml'


@pytest.mark.parametrize('fps', [50, 100, 200])
def test_task_fps_scales_collection_duration_and_owns_no_chunk_size(fps):
    task = load_task_config(TASK, overrides={'fps': fps})
    assert task.prediction_steps is None  # the training config owns the chunk size
    assert not hasattr(task, 'execution_steps')
    assert task.dt == 1 / fps
    assert task.episode_steps == fps * 5


def test_chunk_size_binds_from_training_config_or_checkpoint_and_must_agree():
    task = load_task_config(TASK)
    bound = task.with_chunk_size(128, source='training config')
    assert bound.prediction_steps == 128 and task.prediction_steps is None
    assert bound.require_checkpoint({'fps': 100, 'num_queries': 128}).prediction_steps == 128
    with pytest.raises(ValueError, match='differs'):
        bound.require_checkpoint({'fps': 100, 'num_queries': 100})
    for bad in (0, -1, 1.5, True, '128'):
        with pytest.raises(ValueError, match='positive integer'):
            task.with_chunk_size(bad)


def test_task_chunks_section_is_rejected_with_its_new_location():
    with pytest.raises(ValueError, match='model.chunk_size in the training config'):
        load_task_config(TASK, overrides={'chunks': {'prediction_seconds': 1.0}})


def test_custom_durations_and_explicit_can_override():
    task = load_task_config(TASK, overrides={
        'fps': 50, 'robot': {'arms': {'right': {'can_interface': 'can7'}}}})
    assert task.episode_steps == 250
    assert task.as_dict()['robot']['arms']['right']['can_interface'] == 'can7'


def test_arm_override_preserves_other_arm():
    task = load_task_config(TASK_FIXTURES / 'arx_bimanual.yaml',
                            overrides={'robot': {'arms': {'right': {'can_interface': 'can7'}}}})
    assert task.as_dict()['robot']['arms'] == {
        'left': {'can_interface': 'can0'}, 'right': {'can_interface': 'can7'}}


def test_camera_less_left_task_is_the_left_arm_on_can0():
    task = load_task_config(TASK_FIXTURES / 'arx_left_no_camera.yaml')
    assert task.as_dict()['robot']['arms']['left']['can_interface'] == 'can0'
    assert task.as_dict()['cameras']['names'] == []
    with pytest.raises(ValueError, match='Unknown robot.type'):
        load_task_config(TASK_FIXTURES / 'arx_left_no_camera.yaml', overrides={'robot': {'type': 'other'}})


@pytest.mark.parametrize('overrides', [
    {'fps': 0}, {'fps': -100}, {'fps': True}, {'fps': float('nan')}, {'fps': float('inf')},
    {'fps': '100'}, {'fps': 99.1},
    {'chunks': {'execution_seconds': 2}}, {'chunks': {'prediction_seconds': 0}},
    {'chunks': {'execution_seconds': .005}}, {'chunks': {'prediction_frames': 100}},
    {'collection': {'fps': 50}}, {'training': {'fps': 50}}, {'inference': {'fps': 50}},
    {'runtime': {'control_hz': 500}}, {'runtime': {'record_rollout': 'false'}},
    {'robot': {'arms': {'right': {'can_interace': 'can1'}}}},
    {'cameras': {'names': ['video_right', 'video_right']}},
    {'task': {'behavior': 'generic'}}, {'task': {'subtask_count': 3}}, {'task': {'category': 'native'}},
    {'schema_version': True}, {'schema_version': 2},
])
def test_invalid_or_conflicting_config_fails(overrides):
    with pytest.raises(ValueError):
        load_task_config(TASK, overrides=overrides)


def test_duplicate_keys_fail_instead_of_silently_overwriting(tmp_path):
    path = tmp_path / 'duplicate.yaml'
    path.write_text(TASK.read_text() + '\ncontrol_freq: 50\n')
    with pytest.raises(ValueError, match='Duplicate configuration key'):
        load_task_config(path)


def test_relative_paths_are_based_on_task_file(tmp_path, monkeypatch):
    doc = load_task_config(TASK).as_dict()
    doc['data']['directory'] = 'episodes'
    doc['task'] = dict(name='task')
    path = tmp_path / 'task.yaml'
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.chdir('/')
    task = load_task_config(path)
    assert task.as_dict()['data']['directory'] == str(tmp_path / 'episodes')


def test_catalog_exposes_validated_task_configs():
    from config.task_config import task_directories
    paths = [path for directory in task_directories() for path in directory.glob('*.yaml')]
    assert {path.stem for path in paths} == set(constants.TASK_CONFIGS)
    assert {'single_arm', 'bimanual'} == {path.stem for path in (CONFIG_ROOT / 'tasks').glob('*.yaml')}
    for path in paths:
        task = load_task_config(path)
        old = constants.TASK_CONFIGS[task.name]
        doc = task.as_dict()
        assert task.fps == constants.FPS
        assert task.episode_steps == old['episode_len']
        assert doc['data']['directory'] == str(Path(old['dataset_dir']).expanduser().resolve())
        assert doc['cameras']['names'] == old['camera_names']
        assert doc['robot']['type'] == old['robot_type']
        arms = doc['robot']['arms']
        # Arms may also declare their model and safety timeouts.
        arms = {side: {'can_interface': arm['can_interface']} for side, arm in arms.items()}
        if old['single_arm']:
            side = old.get('single_arm_side', 'left')
            # Existing ARX behavior resolves a named side via the fixed map.
            can = constants.resolve_arx_single_arm_can(old)
            assert arms == {side: {'can_interface': can}}
        else:
            assert arms == {
                side: {'can_interface': old.get('can_interface_' + side, default)}
                for side, default in (('left', 'can0'), ('right', 'can1'))}


def test_nominal_rate_agreement_does_not_reject_measurement_jitter(tmp_path):
    task = load_task_config(TASK)
    path = tmp_path / 'episode.zarr'
    with episode_store.Store(path, 'w') as root:
        root.attrs.update(fps=100, nominal_recording_hz=100, measured_recording_hz=98.1)
    with episode_store.Store(path, 'r') as root:
        assert task.require_recording(root.attrs) == 100
        with pytest.raises(ValueError, match='requires 100 FPS'):
            load_task_config(TASK, overrides={'fps': 50}).require_recording(root.attrs)


def test_legacy_rate_must_be_explicit_and_cannot_override_saved_rate():
    task = load_task_config(TASK)
    with pytest.raises(ValueError, match='no control_freq metadata'):
        task.require_recording({})
    assert task.require_recording({}, legacy_fps=100) == 100
    with pytest.raises(ValueError, match='conflicts'):
        task.require_recording({'fps': 50}, legacy_fps=100)


@pytest.mark.parametrize('metadata', [
    {'action_rate_hz': 100, 'policy_config': {'num_queries': 100}},
    {'fps': 100, 'prediction_horizon': 100},
    {'fps': 100, 'action_horizon': 100},
])
def test_matching_checkpoint(metadata):
    assert load_task_config(TASK).require_checkpoint(metadata).prediction_steps == 100


@pytest.mark.parametrize('metadata', [
    {'action_rate_hz': 50, 'num_queries': 100},
    {'action_rate_hz': 100, 'num_queries': 0}, {'action_rate_hz': 100, 'num_queries': 12.5},
    {'action_rate_hz': 100, 'num_queries': 100, 'recording_rate_hz': 50},
    {'action_rate_hz': 100, 'num_queries': 100, 'task_config': {'fps': 50}},
    {'action_rate_hz': 100, 'num_queries': 100, 'policy_config': {'prediction_horizon': 50}},
    {'action_rate_hz': 100, 'num_queries': 100, 'state_dim': 7},
    {'action_rate_hz': 100, 'num_queries': 100, 'camera_names': ['video_right', 'video_overhead']},
    {'action_rate_hz': 100}, {'num_queries': 100},
])
def test_checkpoint_rate_horizon_and_shape_mismatches_fail(metadata):
    with pytest.raises(ValueError):
        load_task_config(TASK).require_checkpoint(metadata)


def test_cli_prints_scaled_config_and_checks_artifacts(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'fps': 50, 'num_queries': 64}))
    command = [sys.executable, str(CONFIG_ROOT / 'check_task_config.py'),
               str(TASK), '--set', 'fps=50', '--checkpoint-config', str(checkpoint)]
    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['derived']['prediction_steps'] == 64  # from the checkpoint
    checkpoint.write_text(json.dumps({'fps': 100, 'num_queries': 100}))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0


def test_storage_locations_follow_the_environment(tmp_path):
    """FLASH_ALOHA_DATA_DIR/CKPT_DIR move every repository-relative data and checkpoint path together."""
    code = (
        "import json\n"
        "from config import paths\n"
        "from config.task_config import load_task_config, CONFIG_ROOT\n"
        "from config.training_args import build_parser\n"
        "from config.training_config import training_summary\n"
        "from config.inference_config import load_inference_config\n"
        "task = load_task_config(CONFIG_ROOT / 'tasks/single_arm.yaml')\n"
        "args = vars(build_parser().parse_args(['--training-config', 'single_arm_act']))\n"
        "print(json.dumps(dict(data=str(paths.DATA_DIR), rollouts=str(paths.ROLLOUT_DIR), logs=str(paths.INFERENCE_LOG_DIR),\n"
        "    task=task.as_dict()['data']['directory'], ckpt=training_summary(args)['ckpt_dir'],\n"
        "    checkpoint=load_inference_config('single_arm_act')[1]['_inference_checkpoint'],\n"
        "    outside=str(paths.storage_path('/somewhere/else/data/x')))))\n")
    root = Path(__file__).resolve().parents[2]

    def run(**environment):
        env = {key: value for key, value in os.environ.items() if not key.startswith('FLASH_ALOHA_')}
        env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), env.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, '-c', code], env={**env, **environment}, capture_output=True, text=True,
                                check=True)
        return json.loads(result.stdout.strip().splitlines()[-1])
    default = run()
    assert default['task'] == str(root / 'data/single_arm') and default['ckpt'] == str(root / 'ckpts/single_arm_act')
    assert default['checkpoint'] == str(root / 'ckpts/single_arm_act/all_configs.json')
    moved = run(FLASH_ALOHA_DATA_DIR=str(tmp_path / 'datasets'), FLASH_ALOHA_CKPT_DIR=str(tmp_path / 'checkpoints'),
                FLASH_ALOHA_LOG_DIR=str(tmp_path / 'logs'))
    assert moved['data'] == str(tmp_path / 'datasets') and moved['rollouts'] == str(tmp_path / 'datasets/rollouts')
    assert moved['task'] == str(tmp_path / 'datasets/single_arm')
    assert moved['ckpt'] == str(tmp_path / 'checkpoints/single_arm_act')
    assert moved['checkpoint'] == str(tmp_path / 'checkpoints/single_arm_act/all_configs.json')
    assert moved['logs'] == str(tmp_path / 'logs/inference')
    assert moved['outside'] == default['outside'] == '/somewhere/else/data/x'  # only the repository's own defaults move
