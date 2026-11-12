"""Self-contained inference choices, separate from task and checkpoint invariants."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from config.paths import storage_path
from config.task_config import CONFIG_ROOT, _keys, _read, _steps, _text, select_task_config
from config.rtc_config import require_rtc_compatibility

INFERENCE_ROOT = CONFIG_ROOT / 'inference'
FIELDS = {'schema_version', 'task', 'backend', 'checkpoint', 'execution', 'blending', 'rtc', 'recording', 'connection', 'options'}
DIRECT_OPTIONS = {'policy_step': int, 'diffusion_inference_steps': int, 'rerun': bool, 'fast_inference': bool}
REMOTE_OPTIONS = {'prompt': str, 'num_rollouts': int, 'max_timesteps': int, 'timing': bool, 'timing_interval': int}


def _typed(value, kind, field, *, minimum=None):
    if type(value) is not kind or (kind is float and not math.isfinite(value)):
        raise ValueError(f'{field} must be {kind.__name__}')
    if minimum is not None and value < minimum:
        raise ValueError(f'{field} must be >= {minimum}')
    return value


def _path(value, directory):
    path = Path(_text(value, 'path')).expanduser()
    return str(storage_path(path if path.is_absolute() else directory / path))


def default_execution_steps(task):
    """Historical CLI-only default, owned by inference rather than the task."""
    return _steps(0.5, task.fps, 'default inference execution duration')


def load_inference_config(value):
    path = Path(value).expanduser()
    candidates = [path, INFERENCE_ROOT / path, INFERENCE_ROOT / f'{path}.yaml']
    source = next((p.resolve() for p in candidates if p.is_file()), None)
    if source is None:
        raise ValueError(f'Inference config does not exist: {value}')
    doc = _read(source)
    _keys(doc, FIELDS, ('task', 'backend', 'execution', 'rtc'), 'inference config')
    if type(doc.get('schema_version', 1)) is not int or doc.get('schema_version', 1) != 1:
        raise ValueError('Unsupported inference schema_version')
    backend = doc['backend']
    if backend not in ('direct', 'remote'):
        raise ValueError('backend must be direct or remote')
    ref = _text(doc['task'], 'task')
    task = select_task_config(path=_path(ref, source.parent)) if '/' in ref or ref.endswith(('.yaml', '.yml')) else select_task_config(ref)
    if task is None:
        raise ValueError(f'Unknown inference task {ref!r}')
    args = dict(task_config=str(task.source), _inference_task_name=task.name,
                inference_config=str(source), _inference_document=deepcopy(doc))
    if backend == 'remote':
        args['task_name'] = task.name
    if 'checkpoint' in doc:
        if backend == 'remote':
            raise ValueError('Remote inference validates the server handshake; checkpoint paths belong on the server')
        args['_inference_checkpoint'] = _path(doc['checkpoint'], source.parent)
        if backend == 'direct':
            args['config'] = args['_inference_checkpoint']
    elif backend == 'direct':
        raise ValueError('Direct inference requires checkpoint: path/to/all_configs.json')
    execution = doc['execution']
    _keys(execution, ('chunk_steps', 'first_chunk_steps', 'async_inference', 'overlap_steps', 'max_inference_delay_ms'), ('chunk_steps',), 'execution')
    args['query_frequency'] = _typed(execution['chunk_steps'], int, 'execution.chunk_steps', minimum=1)
    args['first_chunk_size'] = _typed(execution.get('first_chunk_steps', args['query_frequency']), int, 'execution.first_chunk_steps', minimum=1)
    asynchronous = _typed(execution.get('async_inference', backend == 'remote'), bool, 'execution.async_inference')
    if backend == 'direct' and (asynchronous or set(execution) & {'overlap_steps', 'max_inference_delay_ms'}):
        raise ValueError('Direct inference does not support asynchronous execution options')
    if backend == 'remote':
        args['async_inference'] = asynchronous
        args['async_prefetch_steps'] = _typed(execution.get('overlap_steps', 20), int, 'execution.overlap_steps', minimum=1)
        if 'max_inference_delay_ms' in execution:
            delay = execution['max_inference_delay_ms']
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                raise ValueError('execution.max_inference_delay_ms must be finite and nonnegative')
            args['max_inference_delay_ms'] = delay
    blending = doc.get('blending', {})
    _keys(blending, ('method', 'steps'), (), 'blending')
    method = blending.get('method', 'sigmoid' if backend == 'direct' else 'none')
    if method not in ('none', 'linear', 'exponential', 'sigmoid'):
        raise ValueError('blending.method must be none, linear, exponential, or sigmoid')
    if backend == 'remote' and (method != 'none' or 'steps' in blending):
        raise ValueError('Remote inference uses fixed-overlap handoffs and supports only blending.method: none')
    if backend == 'direct':
        args.update(blend_actions=method != 'none', blending_method=method if method != 'none' else 'sigmoid',
                    mixing_steps=_typed(blending.get('steps', 10), int, 'blending.steps', minimum=1))
    rtc = doc['rtc']
    _keys(rtc, ('training_time', 'enabled', 'prefix_steps'), ('training_time', 'enabled'), 'rtc')
    args['_expected_training_time_rtc'] = _typed(rtc['training_time'], bool, 'rtc.training_time')
    args['rtc'] = _typed(rtc['enabled'], bool, 'rtc.enabled')
    if args['rtc'] and not args['_expected_training_time_rtc']:
        raise ValueError('rtc.enabled requires rtc.training_time: true')
    if backend == 'direct' and 'prefix_steps' in rtc:
        raise ValueError("Direct RTC uses the checkpoint's trained maximum delay; rtc.prefix_steps is remote-only")
    if backend == 'remote':
        args['rtc_prefix_steps'] = _typed(rtc.get('prefix_steps', 10), int, 'rtc.prefix_steps', minimum=0)
    recording = doc.get('recording', {})
    _keys(recording, ('rollouts', 'dagger', 'eval', 'save_video', 'rollout_dir', 'dagger_dir'), (), 'recording')
    for key, dest in [('rollouts', 'record_rollout'), ('dagger', 'record_dagger_corrections'), ('eval', 'eval'), ('save_video', 'save_episode')]:
        args[dest] = _typed(recording.get(key, False), bool, f'recording.{key}')
    for key in ('rollout_dir', 'dagger_dir'):
        if key in recording:
            args[key] = _path(recording[key], source.parent)
    connection = doc.get('connection', {})
    _keys(connection, ('host', 'port'), (), 'connection')
    if backend == 'direct' and connection:
        raise ValueError('connection is only supported by remote inference')
    if backend == 'remote':
        args['remote_host'] = _text(connection.get('host', 'localhost'), 'connection.host')
        args['remote_port'] = _typed(connection.get('port', 8000), int, 'connection.port', minimum=1)
        if args['remote_port'] > 65535:
            raise ValueError('connection.port must be <= 65535')
    options = doc.get('options', {})
    allowed = DIRECT_OPTIONS if backend == 'direct' else REMOTE_OPTIONS
    _keys(options, allowed, (), 'options')
    for key, val in options.items():
        args[key] = _typed(val, allowed[key], f'options.{key}')
    configure_execution(args, task, backend=backend)
    return task, args


def configure_execution(args, task, metadata=None, *, backend):
    """Resolve inference-owned options and validate before any robot setup.

    The predicted chunk length (horizon) comes from the checkpoint's training
    config. Without checkpoint metadata, horizon checks wait until it is known.
    Returns the task bound to the checkpoint's chunk size (or ``task`` unchanged).
    """
    if metadata is not None:
        task = task.require_checkpoint(metadata)
        if backend == 'direct' and args.get('rtc'):
            # The prefix length is the training config's maximum RTC delay.
            from config.rtc_config import checkpoint_rtc
            args['rtc_prefix_steps'] = checkpoint_rtc(metadata)['training_time_rtc_max_delay']
            if args.get('_inference_document'):
                # Fail on an untrained or incompatible checkpoint before execution checks.
                require_rtc_compatibility(metadata, trained=args['_expected_training_time_rtc'], enabled=True,
                                          prefix_steps=args['rtc_prefix_steps'], overlap_steps=args['rtc_prefix_steps'])
    horizon = task.prediction_steps
    if args.get('query_frequency') is None:
        args['query_frequency'] = default_execution_steps(task)
    for field in ('query_frequency', 'first_chunk_size'):
        if args.get(field) is None:
            args[field] = args['query_frequency']
        _typed(args[field], int, field, minimum=1)
        if horizon is not None and args[field] > horizon:
            raise ValueError(f'{field} exceeds the checkpoint prediction horizon {horizon}')
    for field in ('eval', 'record_rollout', 'save_episode'):
        if args.get(field) is None:
            args[field] = False
    if args.get('temporal_agg') or args.get('inference_on_refresh'):
        raise ValueError('Fixed chunk scheduling does not support temporal aggregation/refresh querying')
    if backend == 'direct' and (args.get('record_rollout') or args.get('eval')) and args.get('record_dagger_corrections'):
        raise ValueError('Use remote inference for simultaneous rollout and DAgger recording')
    if backend == 'remote':
        if args['first_chunk_size'] != args['query_frequency']:
            raise ValueError('Remote inference requires first_chunk_steps equal to chunk_steps; a separate first chunk is supported by direct inference')
        if args.get('async_inference'):
            overlap = args['async_prefetch_steps']
            _typed(overlap, int, 'overlap_steps', minimum=1)
            if horizon is not None and args['query_frequency'] + overlap > horizon:
                raise ValueError('Execution chunk plus overlap exceeds prediction horizon')
            delay = args.get('max_inference_delay_ms')
            if delay is not None and overlap / task.fps <= delay / 1000:
                raise ValueError('Configured overlap must be greater than maximum inference delay')
    if not args.get('_inference_document'):
        return task
    if backend == 'remote' and args.get('rtc') and not args.get('async_inference'):
        raise ValueError('RTC prefix conditioning requires asynchronous remote inference')
    if backend == 'direct':
        chunk = max(args['query_frequency'], args['first_chunk_size'])
        gripper = args.get('gripper_lookahead')
        if args.get('rtc'):
            # The pinned prefix is whole plan rows; the gripper executes the arm's row.
            if gripper not in (None, 0):
                raise ValueError('gripper_lookahead and RTC are mutually exclusive; the pinned prefix is what is sent')
            gripper = 0
            # Asynchronous RTC: each query is sent the prefix length before its chunk boundary and
            # the arm executes the pinned prefix meanwhile; action_lookahead does not apply.
            if args.get('action_lookahead') is not None:
                raise ValueError('action_lookahead and RTC are mutually exclusive; RTC executes its prefix first')
            prefix = args.get('rtc_prefix_steps')
            if type(prefix) is int:
                blend = args['mixing_steps'] if args.get('blend_actions') else 0
                if prefix >= min(args['query_frequency'], args['first_chunk_size']):
                    # Each query is submitted within the chunk it follows.
                    raise ValueError(f'Direct RTC prefix {prefix} must be shorter than execution.chunk_steps '
                                     'and execution.first_chunk_steps')
                if prefix + blend > args['query_frequency']:
                    # The new predictions (and their blend) must take over before the next query.
                    raise ValueError(f'Direct RTC prefix {prefix} plus blending.steps {blend} exceeds '
                                     f'execution.chunk_steps {args["query_frequency"]}')
                # Chunks after the first run from index `prefix`. The old plan must still cover the
                # executed window and gripper lookahead, the next prefix and the blending window.
                covered = max(args['first_chunk_size'], args['query_frequency'] + prefix)
                if horizon is not None and covered + max(blend, gripper) > horizon:
                    raise ValueError('Direct RTC executed chunk, next prefix and blending window exceed prediction horizon')
        else:
            action_lookahead = args.get('action_lookahead')
            lookahead = max(10, 10 if action_lookahead is None else action_lookahead) + (5 if gripper is None else gripper)
            if horizon is not None and chunk + lookahead > horizon:
                raise ValueError('Direct execution chunks plus existing action/gripper lookahead exceed prediction horizon')
        if args.get('blend_actions'):
            _typed(args['mixing_steps'], int, 'mixing_steps', minimum=1)
    if task.name != args['_inference_task_name']:
        raise ValueError('Inference config task does not match checkpoint/CLI task')
    if metadata is not None:
        # Remote conditions min(prefix, overlap) actions; direct conditions its whole prefix.
        overlap = args.get('async_prefetch_steps', 0) if backend == 'remote' else args.get('rtc_prefix_steps', 0)
        require_rtc_compatibility(metadata, trained=args['_expected_training_time_rtc'],
            enabled=args.get('rtc', False), prefix_steps=args.get('rtc_prefix_steps', 0),
            overlap_steps=overlap)
    return task


class InferenceArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, backend, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_backend = backend
        self.add_argument('--inference-config', help='Self-contained inference YAML or catalog name.')

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        selected, _ = super().parse_known_args(argv)
        if not selected.inference_config:
            return super().parse_args(argv, namespace)
        try:
            task, values = load_inference_config(selected.inference_config)
            if values['_inference_document']['backend'] != self.inference_backend:
                raise ValueError(f'This entry point requires backend: {self.inference_backend}')
            seed = argparse.Namespace(**values)
            if namespace is not None:
                vars(seed).update(vars(namespace))
            parsed = super().parse_args(argv, seed)
            resolved = vars(parsed)
            if resolved.get('task_config') != str(task.source):
                selected_task = select_task_config(path=resolved['task_config'])
                if selected_task.source != task.source:
                    raise ValueError('CLI task config conflicts with inference config')
            if resolved.get('task_name') not in (None, task.name):
                raise ValueError('CLI task name conflicts with inference config')
            configure_execution(resolved, task, backend=self.inference_backend)
            return parsed
        except (ValueError, OSError) as exc:
            self.error(str(exc))
