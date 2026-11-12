"""Self-contained training configs; no torch, dataset, or hardware imports."""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys

from config.paths import DATA_DIR, storage_path
from config.task_config import CONFIG_ROOT, _keys, _merge, _read, _text, chunk_size_value, select_task_config

TRAINING_ROOT = CONFIG_ROOT / 'training'
DEFAULT_DATA_DIR = str(DATA_DIR)
SECTIONS = {'schema_version', 'task', 'model_name', 'output_dir',
            'model', 'training', 'data', 'curriculum'}
DATA_FIELDS = {'directory': 'dataset_dir', 'stats_dir': 'stats_dir',
               'sample_weights': 'sample_weights', 'train_ratio': 'train_ratio',
               # Select replay or hand-guided episodes by their recording metadata.
               'replay_episodes': 'replay_episodes',
               # Image/proprioceptive mask applied in training and by the direct runner (utils.masking).
               'masking_strategy': 'masking_strategy'}
REPLAY_EPISODE_CHOICES = ('all', 'only', 'exclude')
CURRICULUM_FIELDS = {'proprio_noise', 'camera_curriculum', 'camera_blur'}
MODEL_TYPES = ('Flow', 'ACT')
PATH_ARGS = {'data_dir', 'pretrained_path', 'resume_ckpt_path', 'actuator_network_dir'}
RESERVED_ARGS = {'help', 'config', 'eval', 'task_name', 'task_config', 'training_config',
                 'model_name', 'ckpt_dir', 'dry_run', 'model_type', 'policy_class', 'training_method'}


def _path(value, directory):
    path = Path(_text(value, 'path')).expanduser()
    return str(storage_path(path if path.is_absolute() else directory / path))


def _recipe_path(value):
    path = Path(_text(str(value), 'training config')).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path, TRAINING_ROOT / path,
        TRAINING_ROOT / f'{path}.yaml']
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f'Training config does not exist: {value}')


def _document(path):
    doc = _read(path)
    _keys(doc, SECTIONS, (), str(path))
    if type(doc.get('schema_version', 1)) is not int or doc.get('schema_version', 1) != 1:
        raise ValueError(f'{path}: unsupported schema_version')
    for section in ('model', 'training', 'data', 'curriculum'):
        if section in doc and not isinstance(doc[section], dict):
            raise ValueError(f'{section} must be a mapping')
    # Resolve paths relative to this training config.
    if 'task' in doc:
        task = _text(doc['task'], 'task')
        if '/' in task or task.endswith(('.yaml', '.yml')):
            doc['task'] = _path(task, path.parent)
    if 'output_dir' in doc:
        doc['output_dir'] = _path(doc['output_dir'], path.parent)
    for key in ('directory', 'stats_dir'):
        value = doc.get('data', {}).get(key)
        if value is not None:
            doc['data'][key] = ([_path(v, path.parent) for v in value]
                                if isinstance(value, list) else _path(value, path.parent))
    for section in ('model', 'training'):
        for key in PATH_ARGS & doc.get(section, {}).keys():
            if doc[section][key] is not None:
                doc[section][key] = _path(doc[section][key], path.parent)
    return doc


def resolve_training_task(args):
    """Reuse the existing task contract, adding only training-owned settings."""
    name = args.get('task_name')
    task = select_task_config(name, args.get('task_config'))
    if task:
        name = args['task_name'] = task.name
    settings = {}
    if task:
        settings = task.legacy(settings)
        if args.get('data_dir'):
            settings['dataset_dir'] = str(Path(args['data_dir']).expanduser() / Path(settings['dataset_dir']).name)
    else:
        raise ValueError(f'Unknown training task {name!r}')
    settings.update(deepcopy(args.get('_training_task_options', {})))
    if args.get('data_dir') and 'dataset_dir' in args.get('_training_task_options', {}):
        directory = settings['dataset_dir']
        def relocate(path):
            return str(Path(args['data_dir']).expanduser() / Path(path).name)
        settings['dataset_dir'] = [relocate(p) for p in directory] if isinstance(directory, list) else relocate(directory)
    return task, settings


def training_chunk_size(args, settings=None):
    """The predicted chunk length, owned by the training config (model.chunk_size).

    A CLI --chunk_size may restate it but never silently replace it.
    """
    given, catalog = args.get('chunk_size'), (settings or {}).get('chunk_size')
    recipe = args.get('_training_recipe')
    if recipe is not None:
        declared = chunk_size_value(recipe.get('model', {}).get('chunk_size'), 'model.chunk_size')
        if given is not None and given != declared:
            raise ValueError(f'chunk_size={given} conflicts with model.chunk_size {declared}; edit the training config')
        return declared
    if given is not None and catalog is not None and given != catalog:
        raise ValueError(f'chunk_size={given} conflicts with the training catalog value {catalog}')
    value = given if given is not None else catalog
    if value is None:
        raise ValueError('Set model.chunk_size in the training config')
    return chunk_size_value(value, 'model.chunk_size')


def checkpoint_directory(args, *, legacy_suffix=''):
    root = Path(args.get('ckpt_dir', 'ckpts/')).expanduser()
    name = args.get('model_name')
    if name is not None:
        _text(name, 'model_name')
        if name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('model_name must be a single directory name')
        return str(root / name)
    # Preserve the historical string suffix, including a trailing slash.
    return str(Path(str(args.get('ckpt_dir', 'ckpts/')) + legacy_suffix).expanduser())


def _validate_curriculum(curriculum):
    for name, values in curriculum.items():
        if values is None:
            continue
        fields = {'proprio_noise': ('initial_scale', 'final_scale'),
                  'camera_curriculum': ('initial_prob', 'final_prob'),
                  'camera_blur': ('initial_sigma', 'final_sigma')}[name]
        if not isinstance(values, dict):
            raise ValueError(f'curriculum.{name} must be a mapping or null')
        schedules = {name: values} if name == 'proprio_noise' else values
        for camera, schedule in schedules.items():
            context = f'curriculum.{name}.{camera}'
            _keys(schedule, (*fields, 'anneal_steps', 'schedule'), (), context)
            for field in fields:
                value = schedule.get(field, 0)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{context}.{field} must be finite and nonnegative')
                if name == 'camera_curriculum' and value > 1:
                    raise ValueError(f'{context}.{field} must be at most 1')
            steps = schedule.get('anneal_steps', -1)
            if type(steps) is not int or steps < -1:
                raise ValueError(f'{context}.anneal_steps must be an integer >= -1')
            if schedule.get('schedule', 'linear') not in ('linear', 'exponential'):
                raise ValueError(f'{context}.schedule must be linear or exponential')


@dataclass(frozen=True)
class TrainingConfig:
    source: Path
    document: dict

    def arguments(self, parser):
        doc = deepcopy(self.document)
        _keys(doc, SECTIONS, ('task', 'model_name'), 'training config')
        checkpoint_directory({'model_name': doc['model_name']})
        task_ref = _text(doc['task'], 'task')
        result = {'model_name': doc['model_name'], 'training_config': str(self.source)}
        if '/' in task_ref or task_ref.endswith(('.yaml', '.yml')):
            result['task_config'] = task_ref
        else:
            result['task_name'] = task_ref
        if 'output_dir' in doc:
            result['ckpt_dir'] = doc['output_dir']
        model, training = doc.get('model', {}), doc.get('training', {})
        if 'training_time_rtc' not in training:
            raise ValueError('Specify training.training_time_rtc explicitly as true or false')
        if 'chunk_size' in training:
            raise ValueError('chunk_size belongs in the model section (model.chunk_size)')
        if 'chunk_size' not in model:
            raise ValueError('Specify model.chunk_size: the number of actions each predicted chunk contains')
        chunk_size_value(model['chunk_size'], 'model.chunk_size')
        if 'type' in model:
            if model['type'] not in MODEL_TYPES:
                raise ValueError(f"Unsupported model.type: {model['type']!r}")
            result['policy_class'] = model['type']
        if 'method' in training:
            result['training_method'] = training['method']
        actions = {action.dest: action for action in parser._actions}
        for section, values, special in [('model', model, {'type'}), ('training', training, {'method'})]:
            for key, value in values.items():
                if key in special:
                    continue
                if key not in actions or key in RESERVED_ARGS or (section == 'model' and key in ('finetune', 'training_time_rtc')):
                    raise ValueError(f'Unknown or misplaced {section} option {key!r}')
                if key in result:
                    raise ValueError(f'Training option {key!r} specified more than once')
                result[key] = value
        result.setdefault('finetune', False)
        method = result.get('training_method', 'standard')
        if method != 'standard':
            raise ValueError(f'Unknown training.method {method!r}')
        for key, value in result.items():
            action = actions.get(key)
            if action is None:
                continue
            if value is None:
                if action.default is not None:
                    raise ValueError(f'{key} cannot be null')
                continue
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
                if type(value) is not bool:
                    raise ValueError(f'{key} must be a boolean')
            elif action.type is not None:
                expected = action.type
                valid = (type(value) in (int, float) if expected is float else type(value) is expected)
                if not valid or (isinstance(value, float) and not math.isfinite(value)):
                    raise ValueError(f'{key} must be a finite {expected.__name__}')
            if action.choices is not None and value not in action.choices:
                raise ValueError(f'{key} must be one of {action.choices}')
        data = doc.get('data', {})
        _keys(data, DATA_FIELDS, (), 'data')
        ratio = data.get('train_ratio', 1.0)
        # 1.0 trains on every episode with no validation split.
        if type(ratio) not in (float, int) or not 0 < ratio <= 1:
            raise ValueError('data.train_ratio must be in (0, 1]')
        weights = data.get('sample_weights')
        if weights is not None and (not isinstance(weights, list) or not weights or any(
                type(w) not in (float, int) or not math.isfinite(w) or w < 0 for w in weights) or not any(weights)):
            raise ValueError('data.sample_weights must be nonnegative finite weights with a positive sum')
        if data.get('masking_strategy') is not None:
            from config.masking_config import require_masking_strategy
            require_masking_strategy(data['masking_strategy'])
        if data.get('replay_episodes', 'all') not in REPLAY_EPISODE_CHOICES:
            raise ValueError(f'data.replay_episodes must be one of {REPLAY_EPISODE_CHOICES}')
        curriculum = doc.get('curriculum', {})
        _keys(curriculum, CURRICULUM_FIELDS, (), 'curriculum')
        _validate_curriculum(curriculum)
        result['_training_task_options'] = {DATA_FIELDS[k]: v for k, v in data.items()}
        result['_training_task_options'].update(curriculum)
        result['_training_recipe'] = doc
        return result


def load_training_config(path):
    source = _recipe_path(path)
    return TrainingConfig(source, _document(source))


class TrainingArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        selected, _ = super().parse_known_args(argv)
        if not selected.training_config:
            return super().parse_args(argv, namespace)
        try:
            recipe = load_training_config(selected.training_config)
            values = recipe.arguments(self)
            seed = argparse.Namespace(**values)
            if namespace is not None:
                vars(seed).update(vars(namespace))
            parsed = super().parse_args(argv, seed)
            resolved = vars(parsed)
            if '--no_training_time_rtc' in argv and '--training-time-rtc' not in argv:
                resolved['training_time_rtc'] = False
            resolved['_training_task_options']['finetune'] = resolved['finetune']
            # An explicit task flag may confirm the recipe, but not silently replace it.
            recipe_task, _ = resolve_training_task(dict(values))
            task, settings = resolve_training_task(resolved)
            if (task.name if task else resolved.get('task_name')) != (recipe_task.name if recipe_task else values.get('task_name')):
                raise ValueError('CLI task conflicts with training config task')
            if recipe_task and (not task or task.source != recipe_task.source):
                raise ValueError('CLI task config conflicts with training config task')
            for key in ('camera_curriculum', 'camera_blur'):
                unknown = set(settings.get(key) or {}) - set(settings['camera_names'])
                if unknown:
                    raise ValueError(f'{key} references unknown task cameras: {sorted(unknown)}')
            if resolved.get('chunk_size') != values['chunk_size']:
                raise ValueError(f"--chunk_size {resolved.get('chunk_size')} conflicts with model.chunk_size "
                                 f"{values['chunk_size']}; edit the training config")
            resolved['chunk_size'] = training_chunk_size(resolved, settings)
            for key in ('batch_size', 'num_steps', 'eval_every', 'validate_every', 'save_every', 'lr'):
                value = resolved[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be finite and positive')
            if resolved['policy_class'] not in MODEL_TYPES:
                raise ValueError(f'policy_class must be one of {MODEL_TYPES}')
            normalize_training_rtc(resolved)
            if resolved.get('finetune'):
                raise ValueError('Action-head finetuning is not supported for Flow; use load_pretrain')
            if (resolved.get('load_pretrain') or resolved.get('finetune')) and not resolved.get('pretrained_path'):
                raise ValueError('Pretraining/finetuning requires training.pretrained_path')
            checkpoint_directory(resolved)
            resolved['training_config'] = str(recipe.source)
            return parsed
        except (ValueError, OSError) as exc:
            self.error(str(exc))


def normalize_training_rtc(args):
    """Translate the positive config option to existing trainer switches."""
    choice = args.get('training_time_rtc')
    method = args.get('training_method', 'standard')
    if choice is None:
        return
    if type(choice) is not bool:
        raise ValueError('training.training_time_rtc must be a boolean')
    if choice and args.get('no_training_time_rtc'):
        raise ValueError('Conflicting training-time RTC options')
    args['no_training_time_rtc'] = not choice


def training_summary(args):
    task, settings = resolve_training_task(dict(args))
    values = {k: deepcopy(v) for k, v in args.items() if not k.startswith('_')}
    method = args.get('training_method', 'standard')
    suffix = f"_{settings.get('task_name', args.get('task_name'))}_{args['policy_class']}_{args['backbone']}_{args['same_backbones']}_{settings.get('masking_strategy')}"
    values['ckpt_dir'] = checkpoint_directory(args, legacy_suffix=suffix)
    values['task_config'] = settings
    values['finetune'] = args.get('finetune', False) or settings.get('finetune', False)
    values['use_raw_future_actions'] = args.get('use_raw_future_actions', False) or settings.get('use_raw_future_actions', False)
    if 'use_raw_future_actions' in args.get('_training_recipe', {}).get('training', {}):
        values['use_raw_future_actions'] = args['use_raw_future_actions']
    values['training_recipe'] = deepcopy(args.get('_training_recipe'))
    if method == 'standard':
        values['chunk_size'] = training_chunk_size(args, settings)
    return values
