"""Fast Flow inference: same sampler and outputs up to fp16 conv-weight rounding. No robot."""
import pytest
import torch
from torch import nn

from policy.fast_flow_inference import (ChannelsLastConv1d, GraphedUNetStep, accelerate_flow_policy,
                                        convert_convolutions)
from test_flow_per_action_rtc import small_policy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.mark.parametrize('stride,padding,dilation,bias', [(1, 2, 1, True), (2, 1, 1, False), (1, 2, 2, True)])
def test_channels_last_conv_matches_conv1d(stride, padding, dilation, bias):
    torch.manual_seed(0)
    conv = nn.Conv1d(6, 10, 5, stride=stride, padding=padding, dilation=dilation, bias=bias)
    x = torch.randn(2, 6, 32)
    fast = ChannelsLastConv1d(conv, torch.float32)  # exact dtype on CPU; fp16 is checked on CUDA
    torch.testing.assert_close(fast(x), conv(x), rtol=1e-5, atol=1e-5)
    assert fast(x).dtype == torch.float32


def test_convert_replaces_every_conv1d_and_nothing_else():
    model = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.Sequential(nn.Conv1d(4, 4, 1), nn.ConvTranspose1d(4, 4, 4)))
    assert convert_convolutions(model, torch.float32) == 2
    assert isinstance(model[0], ChannelsLastConv1d) and isinstance(model[1][1], nn.ConvTranspose1d)


@cuda
def test_accelerated_policy_matches_eager_and_keeps_the_rtc_prefix():
    torch.manual_seed(0)
    reference = small_policy().cuda().eval()
    fast = small_policy().cuda().eval()
    fast.deserialize(reference.serialize())
    accelerate_flow_policy(fast)
    assert isinstance(fast.model.diffusion.unet, GraphedUNetStep)
    qpos, image = torch.randn(1, 7, device='cuda'), torch.rand(1, 1, 3, 24, 32, device='cuda')
    prefix = torch.randn(1, 4, 7, device='cuda')
    for call in (lambda p: p.forward_inf(qpos, image), lambda p: p.forward_inf_training_time_rtc(qpos, image, prefix)):
        torch.manual_seed(1)
        expected = call(reference)
        torch.manual_seed(1)
        actual = call(fast)
        assert actual.shape == expected.shape == (1, 16, 7)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)
    torch.manual_seed(1)
    assert torch.equal(fast.forward_inf_training_time_rtc(qpos, image, prefix)[:, :4], prefix)
    # Replays draw fresh noise, and other batch sizes run the converted U-Net eagerly.
    assert not torch.equal(fast.forward_inf(qpos, image), fast.forward_inf(qpos, image))
    assert fast.forward_inf(qpos.expand(2, -1), image.expand(2, -1, -1, -1, -1)).shape == (2, 16, 7)


@cuda
def test_accelerated_policy_without_graph_and_its_guards():
    policy = small_policy().cuda().eval()
    accelerate_flow_policy(policy, cuda_graph=False)
    assert not isinstance(policy.model.diffusion.unet, GraphedUNetStep)
    assert policy.forward_inf(torch.randn(1, 7, device='cuda'), torch.rand(1, 1, 3, 24, 32, device='cuda')).shape == (1, 16, 7)
    with pytest.raises(ValueError, match='eval-mode'):
        accelerate_flow_policy(small_policy().cuda().train())


def test_accelerate_requires_cuda():
    with pytest.raises(ValueError, match='requires CUDA'):
        accelerate_flow_policy(small_policy().eval())


def test_fast_inference_is_a_typed_direct_option_and_a_server_flag(tmp_path):
    from config.inference_config import load_inference_config
    from inference.serve_policy import build_parser
    from test_inference_config import document, write_config
    assert build_parser().parse_args(['--fast_inference']).fast_inference is True
    doc = document()
    doc['options'] = {'fast_inference': True}
    assert load_inference_config(write_config(tmp_path, doc))[1]['fast_inference'] is True
    doc['options'] = {'fast_inference': 'yes'}
    with pytest.raises(ValueError, match='options.fast_inference must be bool'):
        load_inference_config(write_config(tmp_path, doc))
