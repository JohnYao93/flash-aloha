"""Robot model and arm-role compatibility. No SDK imports or coordinate conversion."""
from collections.abc import Mapping
import json

VERSION = 1
ROLES = ('left', 'right')


def normalize_robot_contract(value):
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if not isinstance(value, Mapping) or set(value) != {'robot_contract_version', 'arms'}:
        raise ValueError('Robot contract requires robot_contract_version and arms')
    if type(value['robot_contract_version']) is not int or value['robot_contract_version'] != VERSION:
        raise ValueError('Unsupported robot contract version')
    arms = value['arms']
    if not isinstance(arms, Mapping) or not arms or set(arms) - set(ROLES):
        raise ValueError('Robot contract arms must name left and/or right')
    result = {}
    for role in ROLES:
        if role not in arms:
            continue
        arm = arms[role]
        if not isinstance(arm, Mapping) or set(arm) != {'model'}:
            raise ValueError(f'Robot contract {role} requires only model')
        model = arm['model']
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError(f'Robot contract {role} model must be a nonempty name')
        result[role] = {'model': model}
    return {'robot_contract_version': VERSION, 'arms': result}


def metadata_robot_contract(metadata, *, required=False):
    """Check every nested declaration; conflicting overrides cannot hide a mismatch."""
    declarations = []
    def visit(part):
        if not isinstance(part, Mapping) and not hasattr(part, 'get'):
            raise ValueError('Robot metadata sections must be mappings')
        if part.get('robot_contract') is not None:
            declarations.append(normalize_robot_contract(part['robot_contract']))
        for key in ('metadata', 'task_config', 'policy_config'):
            if part.get(key) is not None:
                visit(part[key])
    visit(metadata)
    if not declarations:
        if required:
            raise ValueError('Missing robot_contract; declare robot model and arm roles explicitly')
        return None
    if any(item != declarations[0] for item in declarations[1:]):
        raise ValueError('Conflicting robot model/arm-role contract declarations')
    return declarations[0]


def require_robot_match(expected, metadata, *, context='robot artifact'):
    expected = normalize_robot_contract(expected)
    actual = metadata_robot_contract(metadata, required=True)
    if actual != expected:
        raise ValueError(f'{context}: robot model/arm-role mismatch: expected {expected["arms"]}, got {actual["arms"]}')
    return actual


def dataset_robot_contract(paths, *, required=True):
    """Never mix declared and undeclared episodes, models, or left/right roles."""
    from data_processing.episode_data import open_episode
    contracts = []
    for path in paths:
        with open_episode(path) as root:
            contract = metadata_robot_contract(root.attrs, required=required)
            # Raw episodes can change task labels, but cannot change the robot.
            for segment in getattr(root, 'manifest', {}).get('metadata_segments', []):
                declared = metadata_robot_contract(segment)
                if declared is not None and declared != contract:
                    raise ValueError(f'{path}: robot contract changes within an episode')
        contracts.append(contract)
    if not contracts:
        raise ValueError('Cannot determine robot contract from empty data')
    if any(item != contracts[0] for item in contracts[1:]):
        raise ValueError('Mixed robot models/arm roles or missing contracts in training data')
    return contracts[0]
