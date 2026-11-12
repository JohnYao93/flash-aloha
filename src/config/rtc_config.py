"""Checkpoint RTC declarations and inference compatibility; no execution logic."""


def checkpoint_rtc(metadata):
    """Read saved declarations only; a model class is not evidence of RTC training."""
    result = {}
    for key in ('training_time_rtc', 'training_time_rtc_min_delay', 'training_time_rtc_max_delay'):
        values = [part[key] for part in (metadata, metadata.get('policy_config', {}))
                  if part.get(key) is not None]
        expected = bool if key == 'training_time_rtc' else int
        if any(type(value) is not expected for value in values):
            raise ValueError(f'Checkpoint {key} must be {expected.__name__}')
        if values and any(value != values[0] for value in values):
            raise ValueError(f'Conflicting checkpoint {key} declarations')
        result[key] = values[0] if values else None
    return result


def require_rtc_compatibility(metadata, *, trained, enabled, prefix_steps=0, overlap_steps=0):
    saved = checkpoint_rtc(metadata)
    actual = saved['training_time_rtc']
    if type(actual) is not bool:
        raise ValueError('Checkpoint is missing an explicit training_time_rtc boolean; verify its training configuration')
    if actual != trained:
        raise ValueError(f'RTC training mismatch: inference expects training_time_rtc={trained}, checkpoint declares {actual}')
    if enabled and not trained:
        raise ValueError('RTC inference requires an RTC-trained checkpoint')
    if not trained:
        return
    lower, upper = saved['training_time_rtc_min_delay'], saved['training_time_rtc_max_delay']
    if type(lower) is not int or type(upper) is not int or not 0 <= lower <= upper:
        raise ValueError('RTC checkpoint must declare a valid trained delay range')
    if lower > 0:
        raise ValueError('Inference bootstrap requires zero delay in the checkpoint RTC training range')
    if not enabled:
        return
    # The existing remote runner conditions min(overlap, prefix length) actions.
    delay = min(prefix_steps, overlap_steps)
    if prefix_steps <= 0 or not lower <= delay <= upper:
        raise ValueError(f'RTC inference delay {delay} lies outside the trained range [{lower}, {upper}]')
