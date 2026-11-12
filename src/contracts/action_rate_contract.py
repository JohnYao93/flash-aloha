"""control_freq is the policy action rate in Hz, never the 500 Hz SDK loop.

Old artifact field names are accepted only at this read boundary. New policy
metadata and task configs publish control_freq."""
import math


def positive_rate(value):
    if isinstance(value, bool):
        raise ValueError('control_freq must be a positive number')
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('control_freq must be finite and positive')
    return rate


def declared_rate(metadata):
    declarations = []
    for key in ('control_freq', 'action_rate_hz', 'action_sample_hz', 'fps', 'nominal_recording_hz'):
        value = metadata.get(key)
        if value is not None:
            declarations.append(positive_rate(value))
    for key in ('policy_config', 'task_config'):
        if isinstance(metadata.get(key), dict):
            rate = declared_rate(metadata[key])
            if rate is not None:
                declarations.append(rate)
    if not declarations:
        return None
    if any(not math.isclose(value, declarations[0], rel_tol=1e-6) for value in declarations):
        raise ValueError(f'Conflicting control_freq declarations: {declarations}')
    return declarations[0]


def resolve_rate(metadata, requested=None, *, legacy_rate=None, context='episode/policy'):
    rate = declared_rate(metadata)
    if rate is None:
        if legacy_rate is None:
            raise ValueError(f'{context} has no control_freq metadata; explicitly declare its legacy FPS')
        rate = positive_rate(legacy_rate)
    elif legacy_rate is not None and not math.isclose(rate, positive_rate(legacy_rate), rel_tol=1e-6):
        raise ValueError(f'{context} legacy FPS conflicts with recorded metadata ({rate:g} Hz)')
    if requested is not None and not math.isclose(rate, positive_rate(requested), rel_tol=1e-6):
        raise ValueError(f'{context} requires {rate:g} FPS, got {requested}; changing SDK Hz does not change action FPS')
    return rate


def dataset_rate(paths, *, legacy_rate=None, raw=False):
    from data_processing.episode_data import open_episode
    rate = None
    for path in paths:
        with open_episode(path) as root:
            metadata = ({'action_rate_hz': root.attrs.get('raw_action_rate_hz')} if raw else root.attrs)
            current = resolve_rate(metadata, legacy_rate=legacy_rate, context=str(path))
        if rate is not None and not math.isclose(rate, current, rel_tol=1e-6):
            raise ValueError(f'Mixed action FPS in training data: {rate:g} and {current:g} ({path})')
        rate = current
    if rate is None:
        raise ValueError('Cannot determine action FPS from an empty dataset')
    return rate


def resolve_policy_rate(metadata, requested=None, *, legacy_rate=None, context='policy'):
    # A trained output stream can have a different rate from its observation
    # dataset (e.g. explicitly strided raw actions). Prefer its action declaration.
    rates = [item[key] for item in
             (metadata, metadata.get('policy_config', {}), metadata.get('task_config', {}))
             if isinstance(item, dict)
             for key in ('control_freq', 'action_rate_hz', 'action_sample_hz')
             if item.get(key) is not None]
    if rates:
        if any(not math.isclose(positive_rate(rate), positive_rate(rates[0]), rel_tol=1e-6) for rate in rates):
            raise ValueError(f'Conflicting policy action rates: {rates}')
        metadata = {'control_freq': rates[0]}
    return resolve_rate(metadata, requested, legacy_rate=legacy_rate, context=context)


def canonical_policy_metadata(metadata):
    """Publish one action-rate name without rewriting saved training artifacts."""
    from copy import deepcopy
    result = deepcopy(metadata)
    rate = resolve_policy_rate(result)
    for part in (result, result.get('policy_config', {}), result.get('task_config', {})):
        for key in ('fps', 'action_rate_hz', 'action_sample_hz', 'control_hz'):
            part.pop(key, None)
    result['control_freq'] = rate
    return result
