"""Create validated task and training YAML files without loading hardware or models."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
from pathlib import Path
import re
import tempfile

import yaml

from config.paths import DATA_DIR
from config.task_config import CONFIG_ROOT, ConfigLoader, load_task_config
from config.training_args import build_parser as training_parser
from config.training_config import PATH_ARGS, load_training_config, resolve_training_task


MODEL_TEMPLATES = {'ACT': 'single_arm_act', 'Flow': 'single_arm_flow'}


def template_path(kind, value):
    """Accept a catalog name or an explicit YAML path relative to the caller."""
    path = Path(value).expanduser()
    if path.is_absolute() or '/' in value or path.is_file():
        return path.resolve()
    filename = value if path.suffix in ('.yaml', '.yml') else f'{value}.yaml'
    return CONFIG_ROOT / ('tasks' if kind == 'task' else 'training') / filename


def set_field(document, key, value):
    parts = key.split('.')
    if not all(parts):
        raise ValueError('Override keys must be dotted field names')
    current = document
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f'Cannot set {key}: its parent is not a mapping')
    current[parts[-1]] = value


def absolute_path(value):
    return str(Path(value).expanduser().resolve())


def normalize_paths(kind, document):
    """Template paths are already resolved; CLI paths resolve from the caller's cwd."""
    if kind == 'task':
        fields = [(document.get('data', {}), 'directory')]
    else:
        fields = [(document, 'output_dir')]
        task = document.get('task')
        if isinstance(task, str) and ('/' in task or task.endswith(('.yaml', '.yml'))):
            fields.append((document, 'task'))
        fields.extend((document.get('data', {}), key) for key in ('directory', 'stats_dir'))
        fields.extend((document.get(section, {}), key)
                      for section in ('model', 'training') for key in PATH_ARGS)
    for mapping, key in fields:
        if not isinstance(mapping, dict) or mapping.get(key) is None:
            continue
        value = mapping[key]
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f'{key} must contain paths')
            mapping[key] = [absolute_path(item) for item in value]
        elif isinstance(value, str):
            mapping[key] = absolute_path(value)
        else:
            raise ValueError(f'{key} must be a path')


def build_document(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name) or args.name.endswith(('.yaml', '.yml')):
        raise ValueError('NAME must be a plain config name without a directory or YAML extension')
    if args.kind == 'task':
        path = template_path('task', args.template or 'single_arm')
        document = load_task_config(path).as_dict()
        document['task']['name'] = args.name
        document['data']['directory'] = str(DATA_DIR / args.name)
    else:
        default = MODEL_TEMPLATES[args.model_type or 'ACT']
        path = template_path('training', args.template or default)
        document = deepcopy(load_training_config(path).document)
        document['schema_version'] = 1
        document['model_name'] = args.name
        if args.model_type is not None:
            if document.get('model', {}).get('type') != args.model_type:
                raise ValueError('--model-type must match --template; omit --template to use model-specific defaults')

    for field, value in vars(args).items():
        if ('.' in field or field in ('control_freq', 'task', 'output_dir')) and value is not None:
            set_field(document, field, value)
    for assignment in args.set:
        key, separator, text = assignment.partition('=')
        if not separator or not key or not text:
            raise ValueError('--set requires KEY=VALUE (use null or quoted YAML for empty values)')
        if key in ('task.name', 'model_name') or (args.kind == 'task' and key == 'task'):
            raise ValueError('Use the positional NAME to name the new config')
        set_field(document, key, yaml.load(text, Loader=ConfigLoader))
    normalize_paths(args.kind, document)
    return document


def validate_document(kind, document):
    """Use the existing validators, including training/task compatibility rules."""
    with tempfile.TemporaryDirectory(prefix='flash-aloha-config-') as directory:
        path = Path(directory) / 'candidate.yaml'
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        if kind == 'task':
            load_task_config(path)
        else:
            # Resolve the task first, then run the same parser the trainer uses.
            recipe = load_training_config(path)
            _, settings = resolve_training_task(recipe.arguments(training_parser()))
            parser = training_parser()
            def invalid(message):
                raise ValueError(message)
            parser.error = invalid
            parser.parse_args(['--training-config', str(path)])


def create_config(args):
    """Return the target and generated YAML; publish only a validated, new file."""
    document = build_document(args)
    validate_document(args.kind, document)
    folder = 'tasks' if args.kind == 'task' else 'training'
    target = Path(args.output).expanduser().resolve() if args.output else CONFIG_ROOT / folder / f'{args.name}.yaml'
    if target.suffix not in ('.yaml', '.yml'):
        raise ValueError('--output must be a .yaml or .yml file')
    if args.kind == 'task' and target.stem != args.name:
        raise ValueError('Task YAML filename must match NAME for task catalog lookup')
    text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    if not args.dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Validate before touching the target; never replace existing configuration.
        with target.open('x') as stream:
            stream.write(text)
    return target, text


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    task = commands.add_parser('task', help='Create a task; defaults come from single_arm')
    training = commands.add_parser('training', help='Create a training config; defaults come from the chosen model template')
    for command in (task, training):
        command.add_argument('name', help='New config name, without .yaml')
        command.add_argument('--template', help='Existing config name or YAML path to copy defaults from')
        command.add_argument('--output', type=Path, help='Output YAML (default: src/config/tasks or training/NAME.yaml)')
        command.add_argument('--set', action='append', default=[], metavar='FIELD=YAML',
                             help='Override any schema field, e.g. training.seed=42; applied after named flags')
        command.add_argument('--dry-run', action='store_true', help='Validate and print YAML without creating a file')
    task.add_argument('--control-freq', dest='control_freq', type=float, help='Action/capture frequency in Hz')
    task.add_argument('--data-dir', dest='data.directory', help='Dataset directory (default: data/NAME in this repository)')
    task.add_argument('--cameras', dest='cameras.names', nargs='+', help='Ordered camera names')
    task.add_argument('--max-duration-seconds', dest='collection.max_duration_seconds', type=float)
    training.add_argument('--task', dest='task', help='Existing task name or task YAML path (default: template task)')
    training.add_argument('--model-type', choices=tuple(MODEL_TEMPLATES), help='Select default model template (default: ACT)')
    training.add_argument('--output-dir', dest='output_dir', help='Checkpoint parent directory (default: template output_dir)')
    for flag, field, kind in (
        ('batch-size', 'training.batch_size', int), ('lr', 'training.lr', float),
        ('num-steps', 'training.num_steps', int), ('seed', 'training.seed', int),
        ('eval-every', 'training.eval_every', int), ('validate-every', 'training.validate_every', int),
        ('save-every', 'training.save_every', int), ('hidden-dim', 'model.hidden_dim', int),
        ('unet-depth', 'model.unet_depth', int), ('chunk-size', 'model.chunk_size', int),
        ('backbone', 'model.backbone', str),
    ):
        training.add_argument('--' + flag, dest=field, type=kind)
    for flag in ('data-aug', 'wandb', 'finetune', 'training-time-rtc'):
        training.add_argument('--' + flag, dest='training.' + flag.replace('-', '_'),
                              action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target, text = create_config(args)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(text, end='')
    else:
        print(f'Created {target}')


if __name__ == '__main__':
    main()
