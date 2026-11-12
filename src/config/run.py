"""Run any workflow by task or training-config name.

    python scripts/run.py collect single_arm
    python scripts/run.py train single_arm_flow
    python scripts/run.py infer single_arm_flow
    python scripts/run.py serve single_arm_flow --device cuda:0
    python scripts/run.py infer single_arm_flow --inference-config single_arm_remote --remote_host HOST
    python scripts/run.py view single_arm 24
    python scripts/run.py replay single_arm 24 27

Names resolve to the scripts/infer.py, record.py, visualize.py and training
launchers; --dry-run prints the resolved command without starting it.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path
import sys

from config.inference_config import INFERENCE_ROOT, configure_execution, load_inference_config
from config.task_config import load_task_config, select_task_config
from config.training_args import build_parser as training_parser
from config.training_config import load_training_config, resolve_training_task, training_summary

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SOURCE_ROOT.parent / 'scripts'


def resolve_training(value):
    recipe = load_training_config(value)
    task, settings = resolve_training_task(recipe.arguments(training_parser()))
    parser = training_parser()
    def invalid(message):
        raise ValueError(message)
    parser.error = invalid
    arguments = vars(parser.parse_args(['--training-config', str(recipe.source)]))
    summary = training_summary(arguments)
    return recipe, task, arguments, summary


def resolve_task(value):
    path = Path(value).expanduser()
    if path.is_file() or '/' in value or path.suffix in ('.yaml', '.yml'):
        task = load_task_config(path)
    else:
        task = select_task_config(value)
    if task is None:
        raise ValueError(f'Unknown task {value!r}')
    return task


def script(name, *args):
    """A scripts/ launcher invocation; the launchers own the src/ entry-point paths."""
    return [str(SCRIPTS_ROOT / name), *map(str, args)]


def require_standard_training(training):
    if training.get('training_method', 'standard') != 'standard':
        raise ValueError(
            'Only training.method: standard is supported.')


def resolve_inference_profile(args, recipe, task, training):
    """The explicit or same-named inference profile, checked against the training config."""
    # Optional inference profiles own execution settings. The training recipe
    # always owns model selection, including a custom output_dir/model_name.
    profile = args.inference_config
    if profile is None:
        candidate = INFERENCE_ROOT / f'{recipe.source.stem}.yaml'
        if candidate.is_file():
            profile = str(candidate)
    if profile is None:
        return None
    inference_task, inference = load_inference_config(profile)
    backend = inference['_inference_document']['backend']
    if inference_task.source != task.source:
        raise ValueError('Inference profile and training config must reference the same task YAML')
    if inference['_expected_training_time_rtc'] != training['training_time_rtc']:
        raise ValueError('Inference profile training-time RTC does not match the training config')
    configure_execution(inference, task, backend=backend)
    return inference, backend


def build_plan(args, extra=()):
    """Resolve names without importing runtime code or starting any processes.

    EXTRA options are appended unchanged for the selected entry point.
    """
    plan = dict(workflow=args.workflow)
    task = None
    if args.workflow in ('collect', 'view', 'replay'):
        task = resolve_task(args.name)
        if args.workflow == 'collect':
            runner = script('record.py', 'single',
                            '--task_name', task.name, '--task-config', task.source)
        else:
            robot = task.legacy()['robot_type']
            runner = script('visualize.py', 'episode', '--robot', robot,
                            '--task_name', task.name, '--task-config', task.source)
            if args.workflow == 'view':
                runner += ['--episode_idx', str(args.episode), '--rerun']
            else:
                # The collector's review replay: start-pose overlay and kp/kd prompt.
                runner += ['--review_replay', *map(str, args.episodes)]
    else:
        recipe, task, training, summary = resolve_training(args.name)
        plan.update(training_config=str(recipe.source), model_name=training['model_name'],
                    checkpoint_directory=str(Path(summary['ckpt_dir']).resolve()))
        checkpoint = Path(plan['checkpoint_directory']) / 'all_configs.json'
        if args.workflow == 'train':
            runner = script('policy/imitate_episodes_train.py', '--training-config', recipe.source)
        else:
            if task is None:
                raise ValueError('Named inference requires a training config referencing a task YAML')
            require_standard_training(training)
            profile = resolve_inference_profile(args, recipe, task, training) if args.workflow == 'infer' else None
            if profile is not None and profile[1] == 'remote':
                if args.policy_step is not None:
                    raise ValueError('A remote client cannot select the checkpoint step; '
                                     'pass --policy-step to "run.py serve"')
                plan['inference_config'] = profile[0]['inference_config']
                runner = script('infer.py', 'remote', '--inference-config', plan['inference_config'],
                                '--task-config', task.source)
            else:
                plan['checkpoint_config'] = str(checkpoint)
                runner = script('infer.py', 'serve' if args.workflow == 'serve' else 'direct')
                if profile is not None:
                    plan['inference_config'] = profile[0]['inference_config']
                    runner += ['--inference-config', plan['inference_config']]
                runner += ['--config', str(checkpoint), '--task-config', str(task.source)]
                if args.policy_step is not None:
                    runner += ['--policy_step', str(args.policy_step)]
    plan['task'] = task.name if task else training['task_name']
    plan['task_config'] = str(task.source) if task else None
    plan['command'] = [sys.executable, *runner, *extra]
    return plan


def check_checkpoint(plan):
    """Catch missing or relocated model metadata before importing the live runner."""
    path = Path(plan['checkpoint_config'])
    if not path.is_file():
        raise ValueError(f'Checkpoint metadata not found: {path}. Train this config first.')
    with path.open() as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f'Checkpoint metadata must be an object: {path}')
    # The runner loads weights and statistics from this file's directory, whatever
    # ckpt_dir the training machine saved (load_inference_checkpoint_config).
    if not (path.parent / 'dataset_stats.pkl').is_file():
        raise ValueError(f'Checkpoint statistics not found: {path.parent / "dataset_stats.pkl"}')


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Options not listed for a workflow are passed unchanged to its entry point.')
    commands = parser.add_subparsers(dest='workflow', required=True)

    def add(name, help, target):
        command = commands.add_parser(name, help=help, allow_abbrev=False)
        command.add_argument('name', help=target)
        command.add_argument('--dry-run', action='store_true',
                             help='Print the resolved command as JSON without starting a process')
        return command

    add('collect', 'Collect demonstrations with the existing recorder', 'Task name or YAML path')
    add('train', 'Train a policy', 'Training-config name or YAML path')
    infer = add('infer', 'Run the robot with a policy (direct, or remote client by profile)',
                'Training-config name or YAML path')
    infer.add_argument('--inference-config',
                       help='Execution profile; defaults to the same-named catalog profile if present. '
                            'A backend: remote profile runs the remote client.')
    serve = add('serve', 'Serve a checkpoint to a remote inference client', 'Training-config name or YAML path')
    for command in (infer, serve):
        command.add_argument('--policy-step', type=int,
                             help='Existing runner checkpoint-step option (default: profile or best)')
    view = add('view', 'View a recorded episode in Rerun', 'Task name or YAML path')
    view.add_argument('episode', type=int, help='Episode index')
    replay = add('replay', 'Replay recorded episodes on the robot', 'Task name or YAML path')
    replay.add_argument('episodes', type=int, nargs='+',
                        help='Episode indices, reviewed in order with the start-pose overlay and gain prompt')
    return parser


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    try:
        plan = build_plan(args, extra)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        if 'checkpoint_config' in plan:
            check_checkpoint(plan)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # Preserve interpreter flags, environment, cwd, exit status and signals.
    interpreter_args = sys.orig_argv[1:len(sys.orig_argv) - len(sys.argv)]
    command = [sys.executable, *interpreter_args, *plan['command'][1:]]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
