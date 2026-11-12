"""Shared YAML task configuration and compatibility checks; no hardware imports."""
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
import math
import os
from pathlib import Path

import yaml

from config.paths import storage_path
from contracts.action_rate_contract import resolve_rate
from camera_utils.camera_settings_contract import normalize_camera_settings, metadata_camera_settings, merge_camera_settings

CONFIG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_ROOT.parents[1]
# Repo-relative start poses: the shared file, and optional per-task
# files written by scripts/setup/set_up_recording.py --task_name.
SHARED_START_POSITION_PATHS = {'arx': 'calibration/start_position.npz'}
TASK_START_POSITION_DIR = 'calibration/start_positions'


class ConfigLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError('Configuration keys must be strings')
        if key in result:
            raise ValueError(f'Duplicate configuration key {key!r} at line {key_node.start_mark.line + 1}')
        result[key] = loader.construct_object(value_node, deep=True)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path):
    with Path(path).open() as stream:
        value = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return value


def _merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key != 'camera_settings' and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _keys(value, allowed, required, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be a mapping')
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(f'{context}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}')


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be finite and positive')
    return value


def _names(value, field, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f'{field} must be a list of names')
    for item in value:
        _text(item, field)
    if len(set(value)) != len(value):
        raise ValueError(f'{field} must not contain duplicates')


def _steps(seconds, fps, field):
    value = Decimal(str(_number(seconds, field))) * Decimal(str(fps))
    if value != value.to_integral_value() or value < 1:
        raise ValueError(f'{field} × fps must be a positive whole number of samples; got {value}')
    return int(value)


def _path(value, directory, field, *, exists=False):
    path = Path(_text(value, field)).expanduser()
    if not path.is_absolute():
        path = directory / path
    path = storage_path(path)  # data/ckpts/logs under the repository follow config.paths
    if exists and not path.is_file():
        raise ValueError(f'{field} does not exist: {path}')
    return str(path)


def chunk_size_value(value, field='chunk_size'):
    """A predicted action chunk length: a positive integer number of actions."""
    if type(value) is not int or value <= 0:
        raise ValueError(f'{field} must be a positive integer number of actions')
    return value


def require_chunk_length(actions, chunk_size):
    """Fail closed unless a predicted batch of chunks (B, T, D) has T == chunk_size."""
    shape = tuple(getattr(actions, 'shape', ()))
    if len(shape) != 3 or shape[1] != chunk_size:
        raise ValueError(f'Policy predicted a chunk of shape {shape}; the checkpoint chunk_size is {chunk_size}')
    return actions


def checkpoint_chunk_size(metadata):
    """The one predicted chunk length a checkpoint (or policy server) declares."""
    horizons = []
    for item in (metadata, metadata.get('policy_config', {}), metadata.get('task_config', {})):
        if not isinstance(item, dict):
            raise ValueError('Checkpoint config sections must be mappings')
        for key in ('prediction_horizon', 'num_queries', 'chunk_size'):
            if item.get(key) is not None:
                horizons.append(item[key])
    if not horizons:
        # OpenPI serving metadata commonly calls the predicted output
        # length action_horizon. It is not the runtime execution length.
        horizons = [metadata['action_horizon']] if 'action_horizon' in metadata else []
    if not horizons:
        raise ValueError('Checkpoint is missing its predicted action chunk length')
    if any(type(n) is not int or n <= 0 or n != horizons[0] for n in horizons):
        raise ValueError(f'Checkpoint declares inconsistent predicted chunk lengths {horizons}')
    return horizons[0]


@dataclass(frozen=True)
class TaskConfig:
    """A task's robot/data/camera contract.

    The predicted chunk length is owned by the training config, not the task.
    ``prediction_steps`` is None until bound to a training config or checkpoint
    with ``with_chunk_size``; downstream consumers use the bound copy.
    """
    source: Path
    _document: dict
    episode_steps: int
    prediction_steps: int | None = None

    def with_chunk_size(self, chunk_size, *, source='checkpoint'):
        chunk_size = chunk_size_value(chunk_size, f'{source} chunk_size')
        if self.prediction_steps is not None and chunk_size != self.prediction_steps:
            raise ValueError(f'{source} chunk_size {chunk_size} differs from the bound chunk_size {self.prediction_steps}')
        return replace(self, prediction_steps=chunk_size)

    @property
    def robot_contract(self):
        from contracts.robot_contract import normalize_robot_contract
        robot = self._document["robot"]
        return normalize_robot_contract({"robot_contract_version": 1, "arms": {
            role: {"model": arm.get("model", "L5")} for role, arm in robot["arms"].items()}})

    @property
    def control_freq(self):
        return self._document['control_freq']

    @property
    def fps(self):
        # Compatibility for recording/hardware adapters; action-rate configs use control_freq.
        return self.control_freq

    @property
    def dt(self):
        return 1.0 / self.fps

    @property
    def name(self):
        return self._document['task']['name']

    def as_dict(self):
        return deepcopy(self._document)

    def resolved(self):
        return dict(self.as_dict(), derived=dict(
            sample_interval_seconds=self.dt, prediction_steps=self.prediction_steps,
            max_episode_steps=self.episode_steps))

    def legacy(self, base=None):
        """Compatibility view for existing task consumers, with YAML authoritative."""
        doc = self.as_dict()
        arms = doc['robot']['arms']
        single = len(arms) == 1
        side = next(iter(arms)) if single else None
        result = dict(base or {})
        result.update(task_name=self.name, dataset_dir=doc['data']['directory'],
            camera_names=doc['cameras']['names'], robot_type=doc['robot']['type'],
            single_arm=single, single_arm_side=side, episode_len=self.episode_steps,
            fps=self.control_freq, control_freq=self.control_freq,
            camera_settings=doc.get('camera_settings', {}),
            prediction_steps=self.prediction_steps,
            _task_config_path=str(self.source))
        for key in ('execution_steps', 'runtime', 'inference'):
            result.pop(key, None)
        result['can_interface_left'] = arms[side if single else 'left']['can_interface']
        result['can_interface_right'] = None if single else arms['right']['can_interface']
        result['single_arm_can'] = result['can_interface_left'] if single else None
        result['robot_contract'] = self.robot_contract
        result['hardware_config'] = doc['robot']
        for key, value in doc.get('training', {}).items():
            result[{'observations': 'observation_name', 'outputs': 'outputs_name'}.get(key, key)] = value
        return result

    def _require_robot(self, metadata, *, context):
        from contracts.robot_contract import metadata_robot_contract, require_robot_match
        # Untagged existing ARX artifacts retain their legacy path.
        if metadata_robot_contract(metadata) is not None:
            require_robot_match(self.robot_contract, metadata, context=context)

    def require_recording(self, metadata, *, legacy_fps=None):
        """Match nominal recorded FPS, never measured jitter or camera latency."""
        self._require_robot(metadata, context='task recording')
        merge_camera_settings(self._document.get('camera_settings'), metadata_camera_settings(metadata),
                              context='task recording')
        return resolve_rate(metadata, self.fps, legacy_rate=legacy_fps, context='task recording')

    def require_checkpoint(self, metadata):
        """Require matching rates before using a checkpoint; return a copy bound to its chunk size.

        The chunk size comes from the checkpoint (its training config). When this
        task is already bound (e.g. to a training config), the two must agree.
        """
        self._require_robot(metadata, context='task checkpoint')
        merge_camera_settings(self._document.get('camera_settings'), metadata_camera_settings(metadata),
                              context='task checkpoint')
        # Check all declarations; do not prefer output FPS over conflicting
        # observation FPS (e.g. a checkpoint trained with strided actions).
        resolve_rate(metadata, self.fps, context='task checkpoint')
        for item in (metadata, metadata.get('policy_config', {}), metadata.get('task_config', {})):
            if not isinstance(item, dict):
                raise ValueError('Checkpoint config sections must be mappings')
            for key in ('recording_rate_hz', 'training_action_frequency_hz'):
                if item.get(key) is not None:
                    resolve_rate({'fps': item[key]}, self.fps, context=f'checkpoint {key}')
        bound = self.with_chunk_size(checkpoint_chunk_size(metadata))
        training = self._document.get('training', {})
        for item in (metadata, metadata.get('policy_config', {}), metadata.get('task_config', {})):
            for key in ('state_dim', 'action_dim'):
                if key in training and key in item and item[key] != training[key]:
                    raise ValueError(f'Checkpoint {key} differs from the task training configuration')
            names = item.get('base_camera_names', item.get('camera_names'))
            if names is not None and names != self._document['cameras']['names']:
                raise ValueError('Checkpoint camera names/order differ from the task configuration')
        return bound


def load_task_config(path, *, overrides=None, defaults_path=None):
    """Resolve defaults → task YAML → explicit nested overrides, then validate.

    Relative paths are anchored to the task file. Mappings are merged by key;
    lists/scalars are replaced. There are no implicit environment overrides.
    """
    path = Path(path).expanduser().resolve()
    defaults = _read(defaults_path or CONFIG_ROOT / 'task_defaults.yaml')
    supplied = _read(path)
    for part in (defaults, supplied):
        if 'fps' in part:
            if 'control_freq' in part and part['control_freq'] != part['fps']:
                raise ValueError('Conflicting control_freq and legacy fps')
            part['control_freq'] = part.pop('fps')
    document = _merge(defaults, supplied)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError('Overrides must be a mapping')
        overrides = deepcopy(overrides)
        if 'fps' in overrides:
            if 'control_freq' in overrides and overrides['control_freq'] != overrides['fps']:
                raise ValueError('Conflicting control_freq and legacy fps')
            overrides['control_freq'] = overrides.pop('fps')
        document = _merge(document, overrides)
    if 'chunks' in document:
        raise ValueError('Task chunks.prediction_seconds was removed: the predicted chunk length is '
                         'model.chunk_size in the training config')
    _keys(document, ('schema_version', 'task', 'control_freq', 'data', 'robot',
                     'cameras', 'camera_settings', 'collection', 'training'),
          ('schema_version', 'task', 'control_freq', 'data', 'robot', 'cameras', 'collection'), 'config')
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError('Unsupported task config schema_version')
    document['camera_settings'] = normalize_camera_settings(document.get('camera_settings'))
    task = document['task']
    _keys(task, ('name',), ('name',), 'task')
    _text(task['name'], 'task.name')
    fps = _number(document['control_freq'], 'control_freq')
    _keys(document['data'], ('directory',), ('directory',), 'data')
    document['data']['directory'] = _path(document['data']['directory'], path.parent, 'data.directory')
    robot = document['robot']
    _keys(robot, ('type', 'arms'), ('type', 'arms'), 'robot')
    if robot['type'] != 'arx':
        raise ValueError('Unknown robot.type (expected arx)')
    _keys(robot['arms'], ('left', 'right'), (), 'robot.arms')
    if not robot['arms']:
        raise ValueError('At least one robot arm must be configured')
    ports = []
    for side, arm in robot['arms'].items():
        allowed = ('can_interface', 'model', 'command_timeout_s', 'max_feedback_age_s')
        _keys(arm, allowed, ('can_interface',), f'robot.arms.{side}')
        if arm.get('model', 'L5') not in ('L5', 'Trossen_L5'):
            raise ValueError(f'Unsupported {robot["type"]} robot model')
        for key in ('command_timeout_s', 'max_feedback_age_s'):
            if key in arm:
                _number(arm[key], key)
        if fps > 500:
            raise ValueError('control_freq exceeds the selected native motor loop rate')
        ports.append(_text(arm['can_interface'], f'robot.arms.{side}.can_interface'))
    if len(set(ports)) != len(ports):
        raise ValueError('Robot arms cannot share a CAN interface')
    _keys(document['cameras'], ('names',), ('names',), 'cameras')
    _names(document['cameras']['names'], 'cameras.names', empty=True)
    _keys(document['collection'], ('max_duration_seconds',), ('max_duration_seconds',), 'collection')
    episode_steps = _steps(document['collection']['max_duration_seconds'], fps, 'collection.max_duration_seconds')
    training = document.get('training', {})
    _keys(training, ('observations', 'outputs', 'state_dim', 'action_dim'), (), 'training')
    for key in ('observations', 'outputs'):
        if key in training:
            _names(training[key], f'training.{key}')
    for key in ('state_dim', 'action_dim'):
        if key in training and (type(training[key]) is not int or training[key] <= 0):
            raise ValueError(f'training.{key} must be a positive integer')
    return TaskConfig(path, document, episode_steps)


def select_task_config(name=None, path=None):
    """Load an explicit file or a catalog task; unknown legacy names return None."""
    if path:
        config = load_task_config(path)
        if name is not None and name != config.name:
            raise ValueError(f'Task name {name!r} differs from YAML task {config.name!r}')
        return config
    if name is None:
        return None
    if Path(name).name != name:
        raise ValueError('Task name must not contain path components')
    for directory in task_directories():
        path = directory / f'{name}.yaml'
        if path.is_file():
            return load_task_config(path)
    return None


def task_directories():
    """Where task names resolve: this package's tasks/, then FLASH_ALOHA_TASK_DIRS (os.pathsep-separated)."""
    extra = [Path(part).expanduser() for part in os.environ.get('FLASH_ALOHA_TASK_DIRS', '').split(os.pathsep) if part]
    return [CONFIG_ROOT / 'tasks', *extra]


def task_catalog():
    result = {}
    for directory in task_directories():
        for path in sorted(directory.glob('*.yaml')):
            config = load_task_config(path)
            if config.name != path.stem or config.name in result:
                raise ValueError(f'Duplicate or mismatched task name in {path}')
            result[config.name] = config.legacy()
    return result


def task_start_position_path(task_name):
    """Repo-relative per-task start pose path (the file may not exist yet)."""
    if not task_name or Path(task_name).name != task_name:
        raise ValueError(f'Invalid task name for a start pose: {task_name!r}')
    return f'{TASK_START_POSITION_DIR}/{task_name}.npz'


def default_start_position_path(task_name, robot_type):
    """The task's saved start pose when present, else the shared pose."""
    if task_name:
        path = task_start_position_path(task_name)
        if (REPO_ROOT / path).is_file():
            return path
    return SHARED_START_POSITION_PATHS['arx']


def require_task_value(actual, expected, field):
    """Legacy frame/rate CLI flags are assertions, never independent overrides."""
    if actual is not None and actual != expected:
        raise ValueError(f'{field}={actual} conflicts with task YAML value {expected}; edit the task YAML')
    return expected



def resolve_inference_config_path(args):
    """CLI checkpoint config wins; otherwise use the explicitly selected YAML."""
    if args.get('config'):
        return str(Path(args['config']).expanduser())
    raise ValueError('Supply --config checkpoint/all_configs.json or checkpoint in --inference-config YAML')


def load_inference_checkpoint_config(args):
    """Load all_configs.json; its own directory is the checkpoint directory.

    The saved ckpt_dir is the training machine's absolute path, so checkpoints
    copied from another machine or checkout load from where they now live.
    """
    import json
    path = Path(resolve_inference_config_path(args)).resolve()
    with path.open() as stream:
        config = json.load(stream)
    config['ckpt_dir'] = str(path.parent)
    return config


def configure_inference_task(config, args):
    """Validate before hardware setup and apply task deployment settings."""
    saved = config.get('task_config', {})
    name = config.get('task_name') or saved.get('task_name')
    task = select_task_config(name, args.get('task_config'))
    if task is None:
        return None
    if args.get('legacy_policy_fps') is not None:
        config['control_freq'] = resolve_rate(config, legacy_rate=args['legacy_policy_fps'], context='task checkpoint')
    task = task.require_checkpoint(config)  # bound to the checkpoint's chunk size
    config.setdefault('camera_settings', metadata_camera_settings(config))
    from camera_utils.camera_settings_contract import check_inference_camera_settings
    check_inference_camera_settings(task.legacy())
    args['fps'] = require_task_value(args.get('fps'), task.fps, 'fps')
    from config.inference_config import configure_execution
    configure_execution(args, task, config, backend='direct')
    config['task_config'] = task.legacy(saved)
    config['control_freq'] = task.fps
    return task
