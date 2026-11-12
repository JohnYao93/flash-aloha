"""Saved tuning compatibility checks with synthetic NPZs and no camera/robot access."""
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from data_processing import episode_store
import numpy as np
import pytest

from conftest import TASK_FIXTURES

import camera_utils.camera_settings_contract as contract


def tuned(path, *, exposure=156, mode=(640, 480, 50), names=('overhead', 'right'), matrix=False):
    settings = dict(camera_names=np.array(names), devices=np.array(['/dev/' + n for n in names]),
                    widths=np.full(len(names), mode[0]), heights=np.full(len(names), mode[1]),
                    fps_values=np.full(len(names), mode[2]))
    if matrix:
        settings.update(param_names=np.array(['exposure_absolute', 'gain']),
                        values=np.tile([exposure, 30], (len(names), 1)))
    else:
        settings.update(exposure_absolute=np.full(len(names), exposure), gain=np.full(len(names), 30))
    np.savez(path, **settings)


@pytest.fixture
def params(tmp_path, monkeypatch):
    path = tmp_path / 'camera_params.npz'
    tuned(path)
    monkeypatch.setenv('FLASH_ALOHA_CAMERA_PARAMS', str(path))
    return path


def test_snapshot_uses_saved_controls_modes_and_only_selected_cameras(params):
    snapshot = contract.snapshot_camera_settings(['video_overhead'])
    assert snapshot == {'video_overhead': {'settings': {'gain': 30, 'exposure_absolute': 156},
                                         'mode': [640, 480, 50]}}
    tuned(params, matrix=True)
    assert contract.snapshot_camera_settings(['video_overhead']) == snapshot
    # Record order, unused cameras and device paths do not affect the contract.
    tuned(params, names=('right', 'overhead', 'torso'))
    assert contract.snapshot_camera_settings(['video_overhead']) == snapshot


@pytest.mark.parametrize('empty', [None, '', {}, '{}', 'null', b'{}'])
def test_empty_metadata_is_unrestricted_even_with_different_local_settings(params, empty):
    assert contract.check_inference_camera_settings({'camera_settings': empty}) == {}


def test_missing_file_or_camera_record_is_allowed(params):
    expected = contract.snapshot_camera_settings(['video_overhead'])
    tuned(params, names=('right',))
    assert contract.check_inference_camera_settings({'camera_settings': expected}) == expected
    params.unlink()
    assert contract.snapshot_camera_settings(['video_overhead']) == {}
    assert contract.check_inference_camera_settings({'camera_settings': expected}) == expected
    assert contract.check_inference_camera_settings({}) == {}


@pytest.mark.parametrize('change', [dict(exposure=200), dict(mode=(1280, 720, 50)), dict(mode=(640, 480, 100))])
def test_changed_tuning_is_rejected(params, change):
    expected = contract.snapshot_camera_settings(['video_overhead'])
    tuned(params, **change)
    with pytest.raises(ValueError, match='camera settings mismatch.*video_overhead'):
        contract.check_inference_camera_settings({'camera_settings': expected})


def test_corrupt_existing_file_is_not_treated_as_missing(params):
    params.write_text('not an npz')
    with pytest.raises(ValueError, match='Invalid camera settings file'):
        contract.snapshot_camera_settings(['video_overhead'])


def test_recording_request_snapshots_before_stream_start_and_preserves_empty_override(params, monkeypatch):
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    recorder = DistributedRecorder.__new__(DistributedRecorder)
    recorder.config = {'require_camera_timing': False}
    sent = []
    recorder._send_and_wait = lambda command, timeout: sent.append(command)
    def request(metadata):
        return EpisodeRequest('test', '/unused', ['video_overhead'], False, False,
                                 '', '', '', '', [], False, 100, metadata=metadata)
    original = contract.snapshot_camera_settings(['video_overhead'])
    repeated_request = request({'task': 'example'})
    recorder.start_episode(repeated_request, 1)
    tuned(params, exposure=200)
    assert sent[0]['request']['metadata']['camera_settings'] == original
    assert sent[0]['request']['metadata']['task'] == 'example'
    assert 'camera_settings' not in repeated_request.metadata
    recorder.start_episode(repeated_request, 1)
    assert sent[1]['request']['metadata']['camera_settings']['video_overhead']['settings']['exposure_absolute'] == 200
    recorder.start_episode(request({'camera_settings': {}}), 1)
    assert sent[2]['request']['metadata']['camera_settings'] == {}


def test_episode_buffer_keeps_start_snapshot_after_file_changes(params, tmp_path):
    from recording.episode_buffer import EpisodeBuffer
    class Stream:
        def __init__(self, **kwargs):
            pass
        def start_episode(self, request, timeout):
            self.request = request
            self.directory = Path(request.data_dir) / request.session_id
            self.directory.mkdir(parents=True)
        def stop_episode(self, **kwargs):
            (self.directory/'manifest.json').write_text(json.dumps(dict(asdict(self.request), complete=True)))
        def close(self):
            pass
    buffer = EpisodeBuffer(camera_names=['video_overhead'], directory=tmp_path, stream_factory=Stream)
    expected = deepcopy(buffer.camera_settings)
    tuned(params, exposure=200)
    buffer.finish()
    manifest = json.loads((buffer.data_dir/'manifest.json').read_text())
    assert manifest['metadata']['camera_settings'] == expected
    assert buffer.camera_settings == expected


def test_export_and_training_use_start_snapshot_not_current_npz(params, tmp_path):
    from recording import record_episode_single as recording
    saved = []
    expected = contract.snapshot_camera_settings(['video_overhead'])
    buffer = SimpleNamespace(camera_settings=expected, data_dir=tmp_path/'raw', finish=lambda: None,
                             saved_export=lambda path, metadata, events: saved.append(metadata))
    class Images(dict):
        pass
    images = Images(video_overhead=np.zeros((4, 6, 3), dtype=np.uint8))
    images.buffer = buffer
    sample = recording.EpisodeSample(images, np.zeros(7), np.zeros(7), 0.)
    from episode_fixtures import recorded_samples
    recorded_samples([sample])
    buffer.data_dir = Path(sample.episode_directory)
    manifest_path = buffer.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metadata']['camera_settings'] = contract.camera_settings_json(expected)
    manifest_path.write_text(json.dumps(manifest))
    tuned(params, exposure=200)
    path = tmp_path/'episode.zarr'
    recording.write_episode(path, [sample], ['video_overhead'], 7, 7, metadata={'fps': 100})
    with episode_store.Store(path) as root:
        assert json.loads(root.attrs['camera_settings']) == expected
    assert contract.dataset_camera_settings([path]) == expected
    unset = tmp_path/'unset.zarr'
    unset_sample = recording.EpisodeSample(
        {'video_overhead': np.zeros((4, 6, 3), dtype=np.uint8)}, np.zeros(7), np.zeros(7), 0.)
    recorded_samples([unset_sample])
    recording.write_episode(unset, [unset_sample], ['video_overhead'], 7, 7, metadata={'fps': 100})
    assert contract.dataset_camera_settings([unset, path]) == expected
    with episode_store.Store(unset, 'a') as root:
        root.attrs['camera_settings'] = contract.camera_settings_json(contract.snapshot_camera_settings(['video_overhead']))
    with pytest.raises(ValueError, match='camera settings mismatch'):
        contract.dataset_camera_settings([path, unset])


def test_yaml_contract_checks_known_values_and_can_be_cleared(params):
    from config.task_config import CONFIG_ROOT, load_task_config
    expected = contract.snapshot_camera_settings(['video_overhead'])
    task = load_task_config(TASK_FIXTURES/'arx_right_wide_state.yaml', overrides={'camera_settings': expected})
    task.require_recording({'fps': task.fps, 'camera_settings': expected})
    task.require_recording({'fps': task.fps})
    changed = deepcopy(expected)
    changed['video_overhead']['settings']['gain'] = 1
    with pytest.raises(ValueError, match='camera settings mismatch'):
        task.require_recording({'fps': task.fps, 'camera_settings': changed})
    from config.task_config import _merge
    assert _merge(task.as_dict(), {'camera_settings': {}})['camera_settings'] == {}
    assert contract.metadata_camera_settings({'camera_settings': {}, 'task_config': {'camera_settings': expected}}) == {}


def test_native_inference_rejects_mismatch_before_any_hardware_setup(params):
    expected = contract.snapshot_camera_settings(['video_overhead'])
    tuned(params, exposure=200)
    source = Path(__file__).resolve().parents[2]/'src/inference/inference_direct.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'eval_bc')
    function.decorator_list = []
    namespace = dict(check_inference_camera_settings=contract.check_inference_camera_settings,
                     DEFAULT_START_POSITION_PATH='unused')
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    with pytest.raises(ValueError, match='camera settings mismatch'):
        namespace['eval_bc']({'camera_settings': expected}, 'unused')


def test_checkpoint_server_metadata_reaches_remote_inference_check(params):
    import torch
    from test_policy_server import make_config, make_stats, DummyPolicy
    from inference.serve_policy import LocalPolicyServer
    from inference.inference_remote import validate_policy_metadata
    config = make_config()
    expected = contract.snapshot_camera_settings(['video_overhead'])
    config['camera_settings'] = expected
    server = LocalPolicyServer(policy=DummyPolicy(), stats=make_stats(), config=config,
        ckpt_path='unused', host='127.0.0.1', port=8000, device=torch.device('cpu'), use_fill_in=False)
    assert server.metadata['camera_settings'] == expected
    tuned(params, exposure=200)
    with pytest.raises(ValueError, match='camera settings mismatch'):
        validate_policy_metadata(server.metadata)
