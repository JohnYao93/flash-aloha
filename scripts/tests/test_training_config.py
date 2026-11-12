"""Offline recipe resolution, catalog migration, and trainer wiring regressions."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from conftest import TASK_FIXTURES
import yaml

from config.task_config import CONFIG_ROOT, load_task_config, select_task_config
from config.training_args import build_parser
from config.training_config import (
    checkpoint_directory, load_training_config,
    resolve_training_task, training_summary,
)


def write_recipe(tmp_path, **update):
    doc = dict(task='arx_right', model_name='unit', model={'type': 'Flow', 'chunk_size': 64})
    doc.update(update)
    doc.setdefault('training', {}).setdefault('training_time_rtc', doc['model'].get('type') in ('Flow', 'Flow'))
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(doc))
    return path






@pytest.mark.parametrize('task', sorted(path.stem for path in (CONFIG_ROOT / 'tasks').glob('*.yaml')))
def test_every_task_trains_one_image_backbone_per_camera(tmp_path, task):
    # One arm or two: the default recipe never shares a backbone across cameras.
    doc = yaml.safe_load((CONFIG_ROOT / 'training/single_arm_flow.yaml').read_text())
    doc.update(task=str(CONFIG_ROOT / f'tasks/{task}.yaml'), output_dir=str(tmp_path))
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(doc))
    try:
        args = vars(build_parser().parse_args(['--training-config', str(path)]))
    except SystemExit:
        pytest.skip(f'{task} is not trainable with the default recipe')
    assert args['same_backbones'] is False


@pytest.mark.parametrize('name,policy', [('single_arm_flow', 'Flow')])
def test_shipped_recipes_and_cli_precedence(name, policy):
    parser = build_parser()
    args = vars(parser.parse_args(['--training-config', name, '--batch_size', '17']))
    assert args['batch_size'] == 17
    assert args['policy_class'] == policy
    assert args['chunk_size'] == 128  # model.chunk_size, owned by the training config
    assert args['wandb'] is True
    assert args['finetune'] is False
    # ckpts/ may be a symlink to scratch storage; compare the real locations.
    assert Path(checkpoint_directory(args)).resolve() == (Path(__file__).resolve().parents[2] / 'ckpts' / name).resolve()
    assert parser.parse_args([]).batch_size == 32  # No sticky parser defaults.




def test_training_config_requires_explicit_rtc_choice(tmp_path):
    path = tmp_path / 'unspecified_rtc.yaml'
    path.write_text('task: arx_right\nmodel_name: missing_rtc\nmodel:\n  type: Flow\n  chunk_size: 64\n')
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--training-config', str(path)])


def test_paths_resolve_relative_to_training_config(tmp_path):
    path = write_recipe(tmp_path, output_dir='models',
                        training={'pretrained_path': 'source', 'wandb': False, 'num_steps': 32},
                        data={'directory': ['episodes', 'extra']})
    args = vars(build_parser().parse_args(['--training-config', str(path)]))
    assert args['wandb'] is False
    assert args['pretrained_path'] == str(tmp_path / 'source')
    assert checkpoint_directory(args) == str(tmp_path / 'models/unit')
    _, settings = resolve_training_task(args)
    assert settings['dataset_dir'] == [str(tmp_path / 'episodes'), str(tmp_path / 'extra')]
    assert settings['camera_names'] == ['video_overhead', 'video_right']
    overridden = vars(build_parser().parse_args(['--training-config', str(path), '--data_dir', str(tmp_path / 'root')]))
    assert resolve_training_task(overridden)[1]['dataset_dir'] == [str(tmp_path / 'root/episodes'), str(tmp_path / 'root/extra')]



def test_explicit_false_cli_overrides_yaml_true():
    args = build_parser().parse_args(['--training-config', 'single_arm_flow', '--no-wandb', '--no-data_aug'])
    assert args.wandb is False and args.data_aug is False


def test_custom_task_reference_and_task_contract_cannot_be_overridden(tmp_path):
    doc = load_task_config(TASK_FIXTURES / 'arx_right.yaml').as_dict()
    doc['control_freq'] = 50
    doc['task']['name'] = 'custom'
    (tmp_path / 'task.yaml').write_text(yaml.safe_dump(doc))
    path = write_recipe(tmp_path, task='task.yaml')
    args = vars(build_parser().parse_args(['--training-config', str(path)]))
    assert args['task_name'] == 'custom'
    assert args['chunk_size'] == 64
    for flags in [['--chunk_size', '100'], ['--task_name', 'route']]:
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--training-config', str(path), *flags])


@pytest.mark.parametrize('update', [
    {'modle': {}}, {'model': {'type': 'typo'}}, {'model': {'type': 'Flow', 'hidden_dim': True}},
    {'model': {'type': 'Flow'}}, {'model': {'type': 'Flow', 'chunk_size': 0}},
    {'model': {'type': 'Flow', 'chunk_size': 12.5}},
    {'model': {'type': 'Flow', 'chunk_size': 64}, 'training': {'chunk_size': 64}},
    {'training': {'batch_szie': 3}}, {'training': {'batch_size': '32'}},
    {'training': {'wandb': 'false'}}, {'training': {'lr': None}},
    {'training': {'num_steps': -1}}, {'training': {'style': 'typo'}},
    {'training': {'finetune': True}}, {'training': {'finetune': 'false'}}, {'training': {'lr': float('nan')}},
    {'data': {'camera_names': ['new']}}, {'data': {'train_ratio': 1.5}}, {'data': {'train_ratio': 0}},
    {'data': {'sample_weights': [-1]}}, {'model_name': '../escape'},
    {'task': 'no_such_task'}, {'schema_version': True},
    {'curriculum': {'proprio_noise': {'initial_sclae': 1}}},
    {'curriculum': {'camera_curriculum': {'typo_camera': {'initial_prob': 0.5}}}},
])
def test_invalid_recipes_fail_before_training(tmp_path, update):
    path = write_recipe(tmp_path, **update)
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--training-config', str(path)])


def test_duplicate_keys_and_inheritance_fail(tmp_path):
    path = tmp_path / 'duplicate.yaml'
    path.write_text('task: arx_right\ntask: route\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_training_config(path)
    path.write_text('extends: duplicate.yaml\n')
    with pytest.raises(ValueError, match='unknown keys'):
        load_training_config(path)






def test_offline_checker_imports_no_training_or_hardware_modules(tmp_path):
    script = CONFIG_ROOT / 'check_training_config.py'
    code = '''import runpy, sys
sys.argv = [sys.argv[1], '--training-config', 'single_arm_flow']
runpy.run_path(sys.argv[0], run_name='__main__')
assert not {'torch', 'policy.imitate_episodes_train', 'recording.record_episode_single'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c',
        f'import sys; sys.path.insert(0, {str(CONFIG_ROOT.parent)!r});\n' + code,
        str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc['chunk_size'] == 128
    assert doc['ckpt_dir'].endswith('/ckpts/single_arm_flow')
    assert not list(tmp_path.iterdir())


def test_cli_checkpoint_directory_naming():
    args = vars(build_parser().parse_args(['--task_name', 'arx_right', '--chunk_size', '100', '--ckpt_dir', 'ckpts/base']))
    assert training_summary(args)['ckpt_dir'] == 'ckpts/base_arx_right_ACT_resnet18_False_None'




@pytest.mark.parametrize("tagged", [False, True])
def test_flow_trainer_exports_checkpoint_contract(tmp_path, monkeypatch, tagged):
    from policy import imitate_episodes_train as trainer
    from policy import training_utils
    from contracts import action_rate_contract, robot_contract
    from camera_utils import camera_settings_contract
    import torch

    path = write_recipe(tmp_path, model={'type': 'Flow', 'hidden_dim': 32, 'chunk_size': 64}, output_dir='models',
                        training={'num_steps': 1, 'training_time_rtc': True})
    args = vars(build_parser().parse_args(['--training-config', str(path)]))
    identity = {'robot_contract_version': 1, 'arms': {'right': {'model': 'L5'}}} if tagged else None
    monkeypatch.setattr(robot_contract, 'dataset_robot_contract', lambda *a, **k: identity)
    monkeypatch.setattr(training_utils, 'find_episodes', lambda *a: ['synthetic'])
    monkeypatch.setattr(action_rate_contract, 'dataset_rate', lambda *a, **k: 100)
    monkeypatch.setattr(camera_settings_contract, 'dataset_camera_settings', lambda *a: {})
    stats = {'robot_contract': identity} if tagged else {}
    monkeypatch.setattr(training_utils, 'load_data', lambda *a, **k: ([], [], stats, None, None))
    calls = []
    def train(*args):
        calls.append('Flow')
        return 0, 0.0, {}
    monkeypatch.setattr(trainer, 'train_bc', train)
    monkeypatch.setattr(torch, 'save', lambda *a: None)
    trainer.main_train(args)
    assert calls == ['Flow']
    exported = json.loads((tmp_path / 'models/unit/all_configs.json').read_text())
    assert exported['policy_class'] == 'Flow'
    assert exported['task_config']['camera_names'] == ['video_overhead', 'video_right']
    assert exported['policy_config']['num_queries'] == 64
    assert exported['policy_config']['use_separate_rgb_encoder_per_camera'] is True  # one encoder per camera
    # The recipe's model.chunk_size is the checkpoint's one predicted chunk length,
    # and inference binds the task to it.
    from config.task_config import checkpoint_chunk_size
    assert exported['chunk_size'] == checkpoint_chunk_size(exported) == 64
    assert load_task_config(TASK_FIXTURES / 'arx_right.yaml').require_checkpoint(exported).prediction_steps == 64
    assert exported['policy_config']['training_time_rtc'] is True
    assert exported['policy_config']['training_time_rtc_max_delay'] == 10
    assert exported['action_command_style'] == 'absolute'

    assert exported.get('robot_contract') == identity
    assert exported['policy_config'].get('robot_contract') == identity
    assert exported['task_config'].get('robot_contract') == identity
    import pickle
    with (tmp_path / 'models/unit/dataset_stats.pkl').open('rb') as stream:
        assert pickle.load(stream).get('robot_contract') == identity
