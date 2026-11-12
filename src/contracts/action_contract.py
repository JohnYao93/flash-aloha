"""Task-independent action representation checks at data and policy boundaries.

Only absolute targets are supported. Missing declarations in legacy absolute
files retain their existing meaning; an explicit unsupported declaration must
never be hidden by an override or silently reinterpreted.
"""


def require_absolute_action_style(style=None):
    if isinstance(style, bytes):
        style = style.decode('utf-8')
    if style is not None and (not isinstance(style, str) or style != 'absolute'):
        raise ValueError(f'Only absolute actions are supported; got action_command_style={style!r}. '
                         'Relative-action datasets/checkpoints cannot be used as absolute targets.')
    return 'absolute'


def require_absolute_actions(*declarations):
    """Validate configs, server metadata, recording metadata, or Zarr/Zarr attrs."""
    for declaration in declarations:
        if declaration is None:
            continue
        require_absolute_action_style(declaration.get('action_command_style'))
        if declaration.get('relative_action_base') is not None:
            raise ValueError('Only absolute actions are supported; legacy relative_action_base is present. '
                             'Use the original absolute dataset/checkpoint.')
        for name in ('policy_config', 'task_config', 'metadata'):
            nested = declaration.get(name)
            if nested is not None:
                require_absolute_actions(nested)
    return 'absolute'
