"""Inference choices and checkpoint compatibility, using no robot hardware."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from config.inference_config import load_inference_config, configure_execution
from config.task_config import CONFIG_ROOT, configure_inference_task
from config.rtc_config import require_rtc_compatibility


def document(backend='direct'):
    doc = dict(task='arx_right', backend=backend, execution={'chunk_steps': 50},
               rtc={'training_time': True, 'enabled': backend == 'remote'},
               recording={'rollouts': False, 'dagger': False})
    if backend == 'direct':
        doc['checkpoint'] = 'model/all_configs.json'
    else:
        doc['execution'].update(async_inference=True, overlap_steps=10)
    return doc


def write_config(tmp_path, doc):
    path = tmp_path / 'inference.yaml'
    path.write_text(yaml.safe_dump(doc))
    return path


def metadata(trained=True):
    return dict(task_name='arx_right', action_rate_hz=100,
                state_dim=7, action_dim=7, policy_class='Flow', policy_family='flash_aloha',
                request_schema='native', action_command_style='absolute',
                policy_config=dict(num_queries=100, state_dim=7, action_dim=7,
                    camera_names=['video_overhead', 'video_right'], training_time_rtc=trained,
                    training_time_rtc_min_delay=0, training_time_rtc_max_delay=10))


@pytest.mark.parametrize('name', ['single_arm_flow'])
def test_shipped_checkpoint_paths_stay_in_repository(name):
    _, args = load_inference_config(CONFIG_ROOT / 'inference' / f'{name}.yaml')
    # ckpts/ may be a symlink to scratch storage; compare the real locations.
    assert Path(args['config']).resolve() == (Path(__file__).resolve().parents[2] / 'ckpts' / name / 'all_configs.json').resolve()


def test_independent_first_chunk_and_blending_are_inference_choices(tmp_path):
    doc = document()
    doc['execution']['first_chunk_steps'] = 25
    doc['blending'] = {'method': 'linear', 'steps': 7}
    doc['recording']['dagger'] = True
    task, args = load_inference_config(write_config(tmp_path, doc))
    configure_execution(args, task, metadata(), backend='direct')
    assert (args['query_frequency'], args['first_chunk_size']) == (50, 25)
    assert args['blend_actions'] and args['blending_method'] == 'linear' and args['mixing_steps'] == 7
    assert args['record_dagger_corrections'] and not args['record_rollout']
    assert args['config'] == str(tmp_path / 'model/all_configs.json')
    assert 'execution_steps' not in task.legacy()
    assert 'runtime' not in task.as_dict() and 'inference' not in task.as_dict()


@pytest.mark.parametrize('key,value', [
    ('fps', 50), ('camera_names', ['different']), ('state_dim', 14), ('width', 224),
    ('action_dim', 14), ('masking_strategy', 'robot'), ('gripper_lookahead', 0),
    ('robot', {}), ('image_preprocessing', {}), ('extends', 'other.yaml'),
])
def test_invariant_and_hardware_options_are_not_configurable(tmp_path, key, value):
    doc = document()
    doc[key] = value
    with pytest.raises(ValueError, match='unknown keys'):
        load_inference_config(write_config(tmp_path, doc))
    doc.pop(key)
    doc['options'] = {key: value}
    with pytest.raises(ValueError, match='unknown keys'):
        load_inference_config(write_config(tmp_path, doc))


@pytest.mark.parametrize('section,values', [
    ('execution', {'chunk_steps': 0}), ('execution', {'chunk_steps': True}),
    ('blending', {'method': 'typo'}),
    ('blending', {'method': 'sigmoid', 'steps': -1}), ('recording', {'rollouts': 'false'}),
    ('rtc', {'training_time': True}), ('rtc', {'training_time': 'true', 'enabled': False}),
    ('rtc', {'training_time': False, 'enabled': True}),
])
def test_invalid_or_unsupported_direct_options_fail(tmp_path, section, values):
    doc = document()
    doc[section] = values
    with pytest.raises(ValueError):
        load_inference_config(write_config(tmp_path, doc))


@pytest.mark.parametrize('override', [
    {'execution': {'chunk_steps': 50, 'first_chunk_steps': 25}},
    {'execution': {'chunk_steps': 50, 'overlap_steps': 10, 'max_inference_delay_ms': 100}},
    {'execution': {'chunk_steps': 50, 'async_inference': False}},
    {'blending': {'method': 'sigmoid'}},
])
def test_remote_limits_fail_explicitly(tmp_path, override):
    doc = document('remote')
    doc.update(override)
    with pytest.raises(ValueError):
        load_inference_config(write_config(tmp_path, doc))


@pytest.mark.parametrize('backend,execution', [
    ('direct', {'chunk_steps': 86}),  # plus the action/gripper lookahead exceeds 100
    ('direct', {'chunk_steps': 50, 'first_chunk_steps': 101}),
    ('remote', {'chunk_steps': 95, 'async_inference': True, 'overlap_steps': 10}),
])
def test_execution_limits_use_the_checkpoint_chunk_size(tmp_path, backend, execution):
    # The chunk size belongs to the checkpoint's training config, so execution
    # limits are checked against it when the checkpoint is known (before hardware).
    doc = document(backend)
    doc['execution'] = execution
    task, args = load_inference_config(write_config(tmp_path, doc))
    assert task.prediction_steps is None
    with pytest.raises(ValueError, match='horizon'):
        configure_execution(args, task, metadata(), backend=backend)
    larger = metadata()
    larger['policy_config']['num_queries'] = 128
    assert configure_execution(args, task, larger, backend=backend).prediction_steps == 128


def test_simultaneous_recordings_only_supported_by_remote(tmp_path):
    for backend in ('direct', 'remote'):
        doc = document(backend)
        doc['recording'] = {'rollouts': True, 'dagger': True}
        path = write_config(tmp_path, doc)
        if backend == 'direct':
            with pytest.raises(ValueError, match='simultaneous'):
                load_inference_config(path)
        else:
            assert load_inference_config(path)[1]['record_rollout']


def test_rtc_checkpoint_mismatch_missing_and_delay_range_fail(tmp_path):
    task, args = load_inference_config(write_config(tmp_path, document('remote')))
    configure_execution(args, task, metadata(), backend='remote')
    with pytest.raises(ValueError, match='mismatch'):
        configure_execution(args, task, metadata(False), backend='remote')
    missing = metadata()
    del missing['policy_config']['training_time_rtc']
    with pytest.raises(ValueError, match='missing an explicit'):
        configure_execution(args, task, missing, backend='remote')
    args.update(rtc_prefix_steps=20, async_prefetch_steps=20)
    with pytest.raises(ValueError, match='trained range'):
        configure_execution(args, task, metadata(), backend='remote')


def test_unconditioned_inference_requires_trained_zero_delay():
    info = metadata()
    require_rtc_compatibility(info, trained=True, enabled=False)
    info['policy_config']['training_time_rtc_min_delay'] = 1
    with pytest.raises(ValueError, match='zero delay'):
        require_rtc_compatibility(info, trained=True, enabled=False)


def test_conflicting_rtc_checkpoint_declarations_fail():
    info = metadata()
    info['training_time_rtc'] = False
    with pytest.raises(ValueError, match='Conflicting'):
        require_rtc_compatibility(info, trained=True, enabled=True, prefix_steps=10, overlap_steps=10)


def test_cli_precedence_and_entrypoint_selection(tmp_path):
    from inference import inference_direct, inference_remote
    path = write_config(tmp_path, document())
    args = inference_direct.build_parser().parse_args(['--inference-config', str(path), '--first_chunk_size', '20', '--blending-method', 'exponential'])
    assert args.first_chunk_size == 20 and args.blending_method == 'exponential'
    with pytest.raises(SystemExit):
        inference_remote.build_parser().parse_args(['--inference-config', str(path)])
    args = vars(inference_direct.build_parser().parse_args(['--inference-config', str(path), '--first_chunk_size', '99']))
    task, _ = load_inference_config(path)
    with pytest.raises(ValueError, match='horizon'):  # 99 + lookahead exceeds the checkpoint's 100
        configure_execution(args, task, metadata(), backend='direct')


def test_direct_rejects_rtc_checkpoint_before_hardware(tmp_path, monkeypatch):
    from inference import inference_direct
    doc = document()
    path = write_config(tmp_path, doc)
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/all_configs.json').write_text(json.dumps(metadata(False)))
    args = vars(inference_direct.build_parser().parse_args(['--inference-config', str(path)]))
    monkeypatch.setattr(inference_direct, 'init_arm', lambda *a, **k: pytest.fail('hardware reached'))
    with pytest.raises(ValueError, match='RTC training mismatch'):
        inference_direct.main_eval(args)


def test_direct_passes_execution_blending_and_recording_to_existing_runner(tmp_path, monkeypatch):
    from inference import inference_direct
    doc = document()
    doc['execution']['first_chunk_steps'] = 20
    doc['blending'] = {'method': 'exponential', 'steps': 8}
    doc['recording']['dagger'] = True
    path = write_config(tmp_path, doc)
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    info = metadata()
    info['ckpt_dir'] = str(model_dir)
    (model_dir / 'all_configs.json').write_text(json.dumps(info))
    (model_dir / 'policy_best.ckpt').touch()
    args = vars(inference_direct.build_parser().parse_args(['--inference-config', str(path)]))
    calls = []
    monkeypatch.setattr(inference_direct, 'eval_bc', lambda *a, **k: calls.append(k))
    monkeypatch.setattr(inference_direct, 'init_arm', lambda *a, **k: pytest.fail('hardware reached'))
    inference_direct.main_eval(args)
    assert len(calls) == 1
    assert (calls[0]['query_frequency'], calls[0]['first_chunk_size']) == (50, 20)
    assert calls[0]['blending_method'] == 'exponential' and calls[0]['mixing_steps'] == 8
    assert calls[0]['record_dagger_corrections'] and not calls[0]['record_rollout']


@pytest.mark.parametrize('trained', [True, False])
def test_native_server_publishes_actual_rtc_training_declarations(trained):
    import torch
    from inference.serve_policy import LocalPolicyServer
    from test_policy_server import make_config, make_stats, DummyFlowPolicy
    config = make_config(policy_class='Flow')
    config['policy_config'].update(training_time_rtc=trained,
        training_time_rtc_min_delay=0, training_time_rtc_max_delay=10)
    server = LocalPolicyServer(policy=DummyFlowPolicy(), stats=make_stats(), config=config,
        ckpt_path='dummy.ckpt', host='localhost', port=8000, device=torch.device('cpu'), use_fill_in=False)
    assert server.metadata['training_time_rtc'] is trained
    assert server.metadata['training_time_rtc_max_delay'] == 10


def test_remote_uses_config_and_checks_real_server_metadata(tmp_path):
    from inference import inference_remote
    path = write_config(tmp_path, document('remote'))
    args = inference_remote.build_parser().parse_args(['--inference-config', str(path)])
    inference_remote.configure_runtime_shape(args, metadata())
    inference_remote.configure_yaml_policy_timing(args, metadata())
    assert args.fps == 100 and args.query_frequency == 50
    assert args.rtc and args.async_prefetch_steps == 10
    with pytest.raises(ValueError, match='RTC training mismatch'):
        inference_remote.configure_yaml_policy_timing(args, metadata(False))


def test_collection_invariants_still_reject_incompatible_checkpoints(tmp_path):
    task, args = load_inference_config(write_config(tmp_path, document()))
    for patch in ({'action_rate_hz': 50}, {'camera_names': ['different']}, {'action_dim': 14}):
        info = metadata()
        info.update(patch)
        with pytest.raises(ValueError):
            configure_execution(deepcopy(args), task, info, backend='direct')


def test_offline_checker_validates_checkpoint_without_hardware_imports(tmp_path):
    path = write_config(tmp_path, document())
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/all_configs.json').write_text(json.dumps(metadata()))
    code = f'''import runpy, sys
sys.path.insert(0, {str(CONFIG_ROOT.parent)!r})
sys.argv = ['check_inference_config.py', {str(path)!r}]
runpy.run_path({str(CONFIG_ROOT / 'check_inference_config.py')!r}, run_name='__main__')
assert not {{'torch', 'inference.inference_direct', 'inference.inference_remote'}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['checkpoint_verified'] is True


def test_training_configs_explicitly_declare_rtc_and_export_it(tmp_path, monkeypatch):
    from config.training_args import build_parser
    from policy.imitate_episodes_train import validate_training_time_rtc_args
    for name, enabled in [('single_arm_flow', True)]:
        args = vars(build_parser().parse_args(['--training-config', name]))
        assert args['training_time_rtc'] is enabled
        assert args['no_training_time_rtc'] is not enabled
    args = vars(build_parser().parse_args(['--training-config', 'single_arm_flow', '--no-training-time-rtc']))
    assert validate_training_time_rtc_args(args, 100)['training_time_rtc'] is False
    args = vars(build_parser().parse_args(['--training-config', 'single_arm_flow', '--no_training_time_rtc']))
    assert args['training_time_rtc'] is False


def direct_rtc(**rtc):
    doc = document()
    doc['rtc'] = {'training_time': True, 'enabled': True, **rtc}
    doc['blending'] = {'method': 'sigmoid', 'steps': 4}  # 4 actions after the handoff
    return doc


def test_direct_rtc_uses_the_checkpoint_trained_maximum_delay(tmp_path):
    task, args = load_inference_config(write_config(tmp_path, direct_rtc()))
    assert args['rtc'] and args['blend_actions'] and args['mixing_steps'] == 4  # boundary blending stays on
    trained = metadata()
    trained['policy_config']['training_time_rtc_max_delay'] = 15
    configure_execution(args, task, trained, backend='direct')
    assert args['rtc_prefix_steps'] == 15


def test_direct_rtc_runs_the_gripper_on_the_pinned_row(tmp_path):
    """The pinned prefix is whole plan rows, so RTC sends the gripper the arm's row (no gripper lookahead)."""
    from inference.inference_direct import rtc_planned_prefix, select_action_from_chunk
    task, args = load_inference_config(write_config(tmp_path, direct_rtc()))
    trained = metadata()
    trained['policy_config']['training_time_rtc_max_delay'] = 5
    configure_execution(args, task, trained, backend='direct')  # the default (unset) is accepted
    args['gripper_lookahead'] = 5
    with pytest.raises(ValueError, match='gripper_lookahead and RTC'):
        configure_execution(args, task, trained, backend='direct')
    plan = np.arange(100, dtype=float)[None, :, None].repeat(7, axis=2)  # value = plan index
    prefix = rtc_planned_prefix(plan, 0, 45, 5)
    sent = np.stack([select_action_from_chunk(t, 50, 50, 0, 0, plan, False)[0] for t in range(45, 50)])
    np.testing.assert_array_equal(sent, prefix[0])  # arm and gripper both execute the pinned rows


@pytest.mark.parametrize('rtc,extra,match', [
    ({'prefix_steps': 10}, {}, 'remote-only'),
    ({'training_time': False}, {}, 'requires rtc.training_time'),
])
def test_direct_rtc_rejects_conflicting_settings(tmp_path, rtc, extra, match):
    doc = direct_rtc(**rtc)
    doc.update(extra)
    with pytest.raises(ValueError, match=match):
        load_inference_config(write_config(tmp_path, doc))


def test_direct_rtc_checks_the_checkpoint_contract(tmp_path):
    task, args = load_inference_config(write_config(tmp_path, direct_rtc()))
    with pytest.raises(ValueError, match='RTC training mismatch'):
        configure_execution(dict(args), task, metadata(False), backend='direct')
    late = metadata()
    late['policy_config']['training_time_rtc_min_delay'] = 2  # no zero-delay bootstrap
    with pytest.raises(ValueError, match='zero delay'):
        configure_execution(dict(args), task, late, backend='direct')
    # Chunks after the first run from index `prefix`: 50 + prefix + blending 4 <= 100 (RTC has no
    # gripper lookahead: the gripper executes the pinned row).
    long = metadata()
    long['policy_config']['training_time_rtc_max_delay'] = 47  # 50 + 47 + 4 > 100, i.e. 47 + 4 > chunk 50
    with pytest.raises(ValueError, match='plus blending.steps 4 exceeds execution.chunk_steps 50'):
        configure_execution(dict(args), task, long, backend='direct')
    fits = metadata()
    fits['policy_config']['training_time_rtc_max_delay'] = 46  # 50 + 46 + 4 = 100
    configure_execution(dict(args), task, fits, backend='direct')


@pytest.mark.parametrize('execution', [{'chunk_steps': 10, 'first_chunk_steps': 50},
                                       {'chunk_steps': 50, 'first_chunk_steps': 10}])
def test_direct_rtc_queries_are_submitted_within_the_chunk_they_follow(tmp_path, execution):
    doc = direct_rtc()  # 10-action prefix
    doc['blending'] = {'method': 'none'}
    doc['execution'] = execution
    task, args = load_inference_config(write_config(tmp_path, doc))
    with pytest.raises(ValueError, match='must be shorter than execution.chunk_steps'):
        configure_execution(args, task, metadata(), backend='direct')


@pytest.mark.parametrize('chunk_steps,ok', [(14, True), (13, False)])
def test_direct_rtc_prefix_and_blend_must_finish_within_the_executed_chunk(tmp_path, chunk_steps, ok):
    doc = direct_rtc()  # 10-action prefix, 4 blended actions after the handoff
    doc['execution'] = {'chunk_steps': chunk_steps, 'first_chunk_steps': chunk_steps}
    task, args = load_inference_config(write_config(tmp_path, doc))
    if ok:
        configure_execution(args, task, metadata(), backend='direct')
    else:
        with pytest.raises(ValueError, match='exceeds execution.chunk_steps'):
            configure_execution(args, task, metadata(), backend='direct')


@pytest.mark.parametrize('steps,ok', [(9, True), (30, True), (31, False)])
def test_direct_rtc_blending_window_needs_previous_plan_actions_after_the_handoff(tmp_path, steps, ok):
    # Not limited by the 10-action prefix: the previous plan (100 actions) must still hold the
    # longest executed first chunk 70 and the blended actions after the handoff.
    doc = direct_rtc()
    doc['execution']['first_chunk_steps'] = 70
    doc['blending'] = {'method': 'sigmoid', 'steps': steps}
    task, args = load_inference_config(write_config(tmp_path, doc))
    trained = metadata()  # trained maximum delay 10
    if ok:
        configure_execution(args, task, trained, backend='direct')
    else:
        with pytest.raises(ValueError, match='executed chunk, next prefix and blending window'):
            configure_execution(args, task, trained, backend='direct')


def test_direct_rtc_and_action_lookahead_are_mutually_exclusive(tmp_path):
    from inference import inference_direct
    path = write_config(tmp_path, direct_rtc())
    task, _ = load_inference_config(path)
    args = vars(inference_direct.build_parser().parse_args(['--inference-config', str(path)]))
    assert args['action_lookahead'] is None
    configure_execution(dict(args), task, metadata(), backend='direct')
    with pytest.raises(SystemExit):  # rejected while parsing, before any checkpoint or hardware
        inference_direct.build_parser().parse_args(['--inference-config', str(path), '--action_lookahead', '10'])
    with pytest.raises(ValueError, match='mutually exclusive'):
        configure_execution(dict(args, action_lookahead=10), task, metadata(), backend='direct')


def test_direct_rtc_prefix_is_the_previous_plans_next_unexecuted_actions():
    import numpy as np
    from inference.inference_direct import rtc_planned_prefix
    plan = np.arange(128, dtype=float).reshape(1, 128, 1)
    # Chunk started at index 10 (the lookahead) and 50 actions were executed: next is 60.
    np.testing.assert_array_equal(rtc_planned_prefix(plan, 10, 50, 15)[0, :, 0], np.arange(60, 75))
    assert rtc_planned_prefix(plan, 10, 110, 15).shape[1] == 8  # only what remains
    assert rtc_planned_prefix(plan, 10, 118, 15) is None


def test_launcher_accepts_checkpoints_copied_from_the_training_machine(tmp_path):
    from config.run import check_checkpoint
    metadata_path = tmp_path / 'model/all_configs.json'
    metadata_path.parent.mkdir()
    # Saved by training on another machine: that host's path, not where the copy now lives.
    metadata_path.write_text(json.dumps({'ckpt_dir': '/scr/elsewhere/model'}))
    plan = {'checkpoint_config': str(metadata_path)}
    with pytest.raises(ValueError, match='statistics not found'):
        check_checkpoint(plan)
    (tmp_path / 'model/dataset_stats.pkl').write_bytes(b'')
    check_checkpoint(plan)
