"""Standalone scope and preserved execution math."""
import ast
from pathlib import Path

import numpy as np
import pytest

from config.supported_policy import require_supported_policy
from policy.model_util import make_policy

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('policy_class', ['Diffusion', 'CNNMLP', 'TransformerPolicy', 'unknown'])
def test_excluded_policies_rejected_before_loading(policy_class):
    with pytest.raises(ValueError, match='Only Flow and ACT'):
        make_policy(policy_class, {})


@pytest.mark.parametrize('section', [None, 'task_config', 'policy_config'])
def test_masked_checkpoint_is_never_silently_changed(section):
    from config.supported_policy import require_unmasked_for_serving

    def config(strategy):
        return {'policy_class': 'Flow', **({section: {'masking_strategy': strategy}} if section
                                           else {'masking_strategy': strategy})}
    with pytest.raises(ValueError, match='Unknown masking strategy'):
        require_supported_policy(config('robot'))
    require_supported_policy(config('end_effector'))  # a known mask: direct inference applies it
    with pytest.raises(ValueError, match='Remote serving cannot omit the mask'):
        require_unmasked_for_serving(config('end_effector'))


def test_no_imports_of_excluded_subsystems():
    for path in (ROOT / 'src').rglob('*.py'):
        if 'hardware' in path.parts or 'kdl_parser' in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name.startswith(('inpainting', 'policy.detr')) for name in names), path


def execution_functions():
    tree = ast.parse((ROOT / 'src/inference/inference_direct.py').read_text())
    names = {'blend_actions', 'get_chunk_step_offset', 'get_action_index'}
    scope = {'np': np}
    exec(compile(ast.Module(body=[node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]), '<execution>', 'exec'), scope)
    return scope


def test_chunk_offset_tracks_first_and_later_execution_windows():
    index = execution_functions()['get_action_index']
    assert [index(t, 25, 50, 10, 100) for t in (0, 24, 25, 74, 75)] == [10, 34, 10, 59, 10]
    assert index(24, 25, 50, 95, 100) == 99


@pytest.mark.parametrize('method', ['linear', 'exponential', 'sigmoid'])
def test_blending_preserves_input_and_unblended_actions(method):
    blend = execution_functions()['blend_actions']
    old = np.zeros((1, 100, 7), dtype=np.float32)
    new = np.ones_like(old)
    blended = blend(old, new, 10, 50, 10, 20, interp_type=method)
    np.testing.assert_array_equal(old, 0)
    np.testing.assert_array_equal(new, 1)
    np.testing.assert_array_equal(blended[:, :20], new[:, :20])
    np.testing.assert_array_equal(blended[:, 30:], new[:, 30:])
    assert np.all(np.diff(blended[0, 20:30, 0]) >= 0)
    assert blended[0, 20, 0] < blended[0, 29, 0]
