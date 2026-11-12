"""Build inert adapters from model/role configuration; never connect on import."""
from collections.abc import Mapping

from contracts.robot_contract import normalize_robot_contract


def normalize_hardware_config(config, *, base_dir=None):
    if not isinstance(config, Mapping) or set(config) - {'type', 'arms'} or not config.get('arms'):
        raise ValueError('Hardware config requires type and arms')
    backend = config.get('type')
    if backend != 'arx':
        raise ValueError('Hardware type must be arx')
    if not isinstance(config['arms'], Mapping) or set(config['arms']) - {'left', 'right'}:
        raise ValueError('Hardware arms must name left and/or right')
    arms, ports = {}, set()
    for role in ('left', 'right'):
        if role not in config['arms']:
            continue
        arm = dict(config['arms'][role])
        if set(arm) - {'model', 'can_interface', 'command_timeout_s', 'max_feedback_age_s'}:
            raise ValueError(f'Unknown hardware options for {role}')
        model = arm.get('model')
        supported = ('L5', 'Trossen_L5')
        if model not in supported:
            raise ValueError(f'{role}: explicitly choose one of {supported}')
        port = arm.get('can_interface')
        if not isinstance(port, str) or not port.strip() or port in ports:
            raise ValueError('Each arm needs a distinct CAN interface')
        ports.add(port)
        arms[role] = arm
    return {'type': backend, 'arms': arms}


def hardware_robot_contract(config):
    return normalize_robot_contract({'robot_contract_version': 1,
        'arms': {role: {'model': arm['model']} for role, arm in config['arms'].items()}})


def build_arms(config, *, base_dir=None):
    config = normalize_hardware_config(config, base_dir=base_dir)
    # Validate every arm's configuration before connecting any hardware.
    from hardware.backends.arx import ARXAdapter as Adapter
    return {role: Adapter(role=role, **arm) for role, arm in config['arms'].items()}
