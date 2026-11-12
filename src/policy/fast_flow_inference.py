"""Faster Flow sampling on CUDA, for inference only.

cuDNN otherwise reorders every large Conv1d weight into its NHWC layout on each
call (most of the U-Net's GPU time). Here each Conv1d runs as the equivalent
Conv2d whose fp16 weights are stored once in that layout; activations, the
time/condition encoders and everything outside the convolutions stay fp32.
The per-step U-Net call is then replayed from a CUDA graph. The checkpoint,
objective and sampler are unchanged; only the convolutions' weight precision is.
A bf16 policy keeps its bf16 weights (and bf16 activations) instead of fp16.
"""
import logging

import torch
from torch import nn
from torch.nn import functional as F


class ChannelsLastConv1d(nn.Module):
    """A Conv1d computed as a channels-last Conv2d with fp16 weights; input and output dtypes are kept."""

    def __init__(self, conv, dtype=torch.float16):
        super().__init__()
        if conv.padding_mode != 'zeros' or isinstance(conv.padding, str):
            raise ValueError('Only zero-padded Conv1d layers with explicit padding are supported')
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        self.dilation, self.groups = conv.dilation[0], conv.groups
        weight = conv.weight.detach().to(dtype)[:, :, None]
        self.weight = nn.Parameter(weight.contiguous(memory_format=torch.channels_last), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().to(dtype), requires_grad=False)

    def forward(self, x):
        y = F.conv2d(x.to(self.weight.dtype)[:, :, None].contiguous(memory_format=torch.channels_last),
                     self.weight, self.bias, (1, self.stride), (0, self.padding), (1, self.dilation), self.groups)
        return y[:, :, 0].to(x.dtype)


def convert_convolutions(module, dtype=torch.float16):
    """Replace every Conv1d below ``module`` in place; returns how many were replaced."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Conv1d):
            setattr(module, name, ChannelsLastConv1d(child, dtype))
            count += 1
        else:
            count += convert_convolutions(child, dtype)
    return count


class GraphedUNetStep(nn.Module):
    """Replays one U-Net call from a CUDA graph for the captured shapes; other shapes run eagerly."""

    def __init__(self, unet, x, time, condition):
        super().__init__()
        self.unet = unet
        self.x, self.time, self.condition = x.clone(), time.clone(), condition.clone()
        with torch.inference_mode():
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm up cuDNN/cuBLAS outside the capture
                    unet(self.x, self.time, global_cond=self.condition)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = unet(self.x, self.time, global_cond=self.condition)

    def forward(self, x, timestep, global_cond=None):
        if (global_cond is None or x.shape != self.x.shape or timestep.shape != self.time.shape
                or global_cond.shape != self.condition.shape or x.dtype != self.x.dtype):
            return self.unet(x, timestep, global_cond=global_cond)
        self.x.copy_(x)
        self.time.copy_(timestep)
        self.condition.copy_(global_cond)
        self.graph.replay()
        return self.out.clone()  # the next replay overwrites the static output


def accelerate_flow_policy(policy, *, cuda_graph=True):
    """Speed up an eval-mode Flow policy on CUDA in place. Not for training or saving."""
    from policy.flow_rtc import PerActionTimeUNet
    diffusion = policy._get_strict_flow_diffusion_module()
    unet = diffusion.unet
    if not isinstance(unet, PerActionTimeUNet):
        raise ValueError('Fast Flow inference expects the per-action-time U-Net')
    if policy.training:
        raise ValueError('Fast Flow inference is for eval-mode policies only')
    device = next(unet.parameters()).device
    if device.type != 'cuda':
        raise ValueError(f'Fast Flow inference requires CUDA, got {device}')
    weight_dtype = unet.diffusion_step_encoder[1].weight.dtype
    conv_dtype = torch.float16 if weight_dtype == torch.float32 else weight_dtype
    replaced = convert_convolutions(unet, conv_dtype)
    message = (f'Fast Flow inference: {replaced} convolutions with {str(conv_dtype).removeprefix("torch.")} '
               'channels-last weights')
    if cuda_graph:
        cond_dim = unet.down_modules[0][0].cond_encoder[1].in_features - unet.diffusion_step_encoder[-1].out_features
        horizon = policy.prediction_horizon
        try:
            graphed = GraphedUNetStep(unet, torch.zeros(1, horizon, policy.action_dim, device=device),
                                      torch.zeros(1, horizon, device=device), torch.zeros(1, cond_dim, device=device))
        except RuntimeError as error:  # keep the faster convolutions, without the graph
            logging.warning('CUDA graph capture failed (%s); running the U-Net eagerly', error)
        else:
            diffusion.unet = graphed
            message += ', CUDA-graphed U-Net step'
    policy.fast_inference = True
    logging.info(message)
    print(message)
    return policy
