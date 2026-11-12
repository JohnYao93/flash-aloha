"""The predicted chunk size: owned by the training config, enforced at training and inference."""
import numpy as np
import pytest

from config.task_config import CONFIG_ROOT, require_chunk_length
from config.training_args import build_parser
from config.training_config import TRAINING_ROOT
from policy.policy import LeRobotFlowPolicy

DEPTH_4 = (256, 512, 1024, 2048)


def test_incompatible_unet_depth_fails_instead_of_shrinking():
    with pytest.raises(ValueError, match=r'chunk_size=100 is not divisible by 2\*\*unet_depth=16'):
        LeRobotFlowPolicy._select_compatible_down_dims(None, 100, DEPTH_4)
    assert LeRobotFlowPolicy._select_compatible_down_dims(None, 128, DEPTH_4) == DEPTH_4


@pytest.mark.parametrize('recipe', sorted(path.stem for path in TRAINING_ROOT.glob('*.yaml')))
def test_shipped_recipes_declare_a_chunk_size_their_unet_depth_divides(recipe):
    args = vars(build_parser().parse_args(['--training-config', recipe]))
    assert args['chunk_size'] > 0
    if args['policy_class'] == 'Flow':  # only the Flow U-Net constrains the chunk length
        assert args['chunk_size'] % 2 ** args['unet_depth'] == 0


@pytest.mark.parametrize('recipe', sorted(path.stem for path in TRAINING_ROOT.glob('*_act*.yaml')))
def test_act_recipes_train_from_per_camera_pretrained_resnet18_on_one_gpu(recipe):
    args = vars(build_parser().parse_args(['--training-config', recipe]))
    assert args['policy_class'] == 'ACT'
    assert (args['chunk_size'], args['batch_size'], args['num_steps']) == (100, 128, 100000)
    assert (args['backbone'], args['pretrained_backbone_weights']) == ('resnet18', 'IMAGENET1K_V1')
    assert args['same_backbones'] is False  # one image backbone per camera
    assert args['frame_store'] and args['compile']


def test_act_is_the_default_policy():
    assert build_parser().get_default('policy_class') == 'ACT'


@pytest.mark.parametrize('recipe', sorted(path.stem for path in TRAINING_ROOT.glob('*_flow*.yaml')))
def test_flow_recipes_train_depth_3_from_per_camera_pretrained_resnet18_on_one_gpu(recipe):
    args = vars(build_parser().parse_args(['--training-config', recipe]))
    assert args['policy_class'] == 'Flow'
    assert (args['chunk_size'], args['unet_depth'], args['batch_size']) == (128, 3, 128)
    assert (args['backbone'], args['pretrained_backbone_weights']) == ('resnet18', 'IMAGENET1K_V1')
    assert args['same_backbones'] is False  # one image backbone per camera
    assert args['frame_store'] and args['compile']
    if not recipe.endswith('_smoke'):
        assert args['num_steps'] == 100000


def test_task_yamls_no_longer_declare_chunks():
    for path in (CONFIG_ROOT / 'tasks').glob('*.yaml'):
        assert 'chunks:' not in path.read_text(), path


def test_predicted_chunk_must_have_exactly_the_checkpoint_length():
    actions = np.zeros((1, 128, 7))
    assert require_chunk_length(actions, 128) is actions
    for wrong in (np.zeros((1, 100, 7)), np.zeros((128, 7)), np.zeros((1, 129, 7))):
        with pytest.raises(ValueError, match='chunk_size is 128'):
            require_chunk_length(wrong, 128)
