import json
from copy import deepcopy

import pytest

from contracts.robot_contract import (normalize_robot_contract, metadata_robot_contract,
    require_robot_match, dataset_robot_contract)


def contract(model='Trossen_L5', role='left'):
    return {'robot_contract_version': 1, 'arms': {role: {'model': model}}}


@pytest.mark.parametrize('value', [None, {}, {'robot_contract_version': True, 'arms': {'left': {'model': 'L5'}}},
    {'robot_contract_version': 2, 'arms': {'left': {'model': 'L5'}}},
    {'robot_contract_version': 1, 'arms': {}}, {'robot_contract_version': 1, 'arms': {'active': {'model': 'L5'}}}])
def test_invalid_contracts_fail(value):
    with pytest.raises(ValueError):
        normalize_robot_contract(value)


def test_native_details_are_outside_contract():
    value = contract()
    value['arms']['left']['units'] = 'radians'
    with pytest.raises(ValueError):
        normalize_robot_contract(value)
    assert normalize_robot_contract(json.dumps(contract())) == contract()


@pytest.mark.parametrize('wrong', [contract('L5'), contract(role='right')])
def test_models_and_roles_cannot_be_interchanged(wrong):
    with pytest.raises(ValueError, match='mismatch'):
        require_robot_match(contract(), {'robot_contract': wrong})


def test_missing_and_conflicting_nested_declarations_fail():
    with pytest.raises(ValueError, match='Missing'):
        require_robot_match(contract(), {})
    with pytest.raises(ValueError, match='Conflicting'):
        metadata_robot_contract({'robot_contract': contract(), 'policy_config': {'robot_contract': contract('L5')}})
    result = metadata_robot_contract({'task_config': {'robot_contract': contract()}})
    result['arms']['left']['model'] = 'changed'
    assert contract()['arms']['left']['model'] == 'Trossen_L5'


def test_dataset_contract_checks_all_episodes_and_segments(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from data_processing import episode_data
    roots = {}
    @contextmanager
    def open_fake(path):
        yield roots[path]
    monkeypatch.setattr(episode_data, 'open_episode', open_fake)
    roots['a'] = SimpleNamespace(attrs={'robot_contract': contract()}, manifest={})
    roots['b'] = SimpleNamespace(attrs={'robot_contract': contract()}, manifest={})
    assert dataset_robot_contract(['a', 'b']) == contract()
    roots['b'].attrs = {}
    with pytest.raises(ValueError, match='Mixed'):
        dataset_robot_contract(['a', 'b'], required=False)
    roots['b'].attrs = {'robot_contract': contract(role='right')}
    with pytest.raises(ValueError, match='Mixed'):
        dataset_robot_contract(['a', 'b'])
    roots['a'].manifest = {'metadata_segments': [{'robot_contract': contract('L5')}]}
    with pytest.raises(ValueError, match='within an episode'):
        dataset_robot_contract(['a'])
    roots['a'] = SimpleNamespace(attrs={}, manifest={})
    assert dataset_robot_contract(['a'], required=False) is None
    with pytest.raises(ValueError, match='Missing'):
        dataset_robot_contract(['a'])
