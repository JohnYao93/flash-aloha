"""Scope checks for the supported policies (decoder-only ACT and Flow)."""


from config.masking_config import require_masking_strategy  # noqa: E402  (strategies are defined in the masking config)


def require_supported_policy(config):
    """Supported checkpoints: decoder-only ACT or native Flow, each optionally RTC-prefixed."""
    if config.get('policy_class') not in ('Flow', 'ACT'):
        raise ValueError('flash_aloha supports only Flow and ACT checkpoints')
    for part in (config, config.get('task_config', {}), config.get('policy_config', {})):
        require_masking_strategy(part.get('masking_strategy'))
        if part.get('training_method', 'standard') != 'standard':
            raise ValueError('Only standard native training is supported')


def require_unmasked_for_serving(config):
    """The model server and the remote client do not apply masks: refuse masked checkpoints."""
    for part in (config, config.get('task_config', {}), config.get('policy_config', {})):
        if part.get('masking_strategy') is not None:
            raise ValueError(f"Checkpoint was trained with masking strategy {part['masking_strategy']!r}; masked "
                             'checkpoints run through direct inference, which applies the shared Masker. '
                             'Remote serving cannot omit the mask')
