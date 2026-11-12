"""The Flow objective: per-action noise times for the LeRobot U-Net.

* Noise is at t=1 and data at t=0: x_t = t * noise + (1 - t) * actions, the
  target velocity is noise - actions, and sampling integrates t from 1 to 0.
* Every action has its own noise time. Training-time RTC gives the first
  ``delay`` actions (the action prefix) time 0, i.e. explicitly clean, rather
  than only overwriting their values.
* Continuous sine/cosine time embedding with periods from 4e-3 to 4.
* Velocity MSE, averaged over valid (non-prefix, non-padding) actions within
  each example and then over examples, so every example has equal weight.
* Low-precision (bf16) weights compute in their own dtype, but noise times and
  their sine/cosine embedding stay fp32: at the 4e-3 period the angles reach
  ~1.6e3 rad, far beyond bf16 resolution.
"""
import torch
from torch import nn
from torch.nn import functional as F

OBJECTIVE_MARKER = '_per_action_time'  # weights buffer identifying this objective


class ContinuousTimeEmbedding(nn.Module):
    """Continuous-time sine/cosine embedding, at the U-Net's embedding width."""

    def __init__(self, dimension):
        super().__init__()
        if dimension % 2:
            raise ValueError('Time embedding width must be even')
        self.dimension = dimension

    def forward(self, time):
        fraction = torch.linspace(0, 1, self.dimension // 2, device=time.device, dtype=time.dtype)
        period = 0.004 * 1000.0 ** fraction  # min_period 4e-3, max_period 4.0
        angle = time[..., None] * (2 * torch.pi / period)
        return torch.cat((angle.sin(), angle.cos()), dim=-1)


class PerActionTimeUNet(nn.Module):
    """The LeRobot conditional U-Net, with its time FiLM applied per action.

    Reuses the U-Net's modules and parameters; only the parameter-free sinusoidal
    timestep embedding is replaced. Each position's condition (its own time
    embedding plus the global observation features) is resampled to every U-Net
    resolution.
    """

    def __init__(self, unet):
        super().__init__()
        for name in ('diffusion_step_encoder', 'down_modules', 'mid_modules', 'up_modules', 'final_conv'):
            setattr(self, name, getattr(unet, name))
        self.diffusion_step_encoder[0] = ContinuousTimeEmbedding(self.diffusion_step_encoder[1].in_features)
        self.channels_last = False

    @staticmethod
    def _residual(block, x, condition):
        out = block.conv1(x)
        local = F.interpolate(condition.transpose(1, 2), size=out.shape[-1], mode='nearest')
        encoded = block.cond_encoder(local.transpose(1, 2)).transpose(1, 2)
        if block.use_film_scale_modulation:
            scale, bias = encoded.split(block.out_channels, dim=1)
            out = scale * out + bias
        else:
            out = out + encoded
        return block.conv2(out) + block.residual_conv(x)

    def use_channels_last(self):
        """Training kernel layout: the same parameters and math, run as channels-last 2D convolutions.

        cuDNN otherwise transposes every Conv1d's activations and weights to NHWC and
        back on each call. Here activations are (B, C, 1, L) channels-last and each
        convolution weight is stored channels-last once (the parameter keeps its
        (out, in, k) shape and state_dict name), so no per-call transposes remain.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                module.weight.data = module.weight.data.transpose(1, 2).contiguous().transpose(1, 2)
        self.channels_last = True
        return self

    @staticmethod
    def _conv_channels_last(conv, x):
        if isinstance(conv, nn.Identity):
            return x
        weight = conv.weight.unsqueeze(2)
        if isinstance(conv, nn.ConvTranspose1d):
            return F.conv_transpose2d(x, weight, conv.bias, (1, conv.stride[0]), (0, conv.padding[0]),
                                      (0, conv.output_padding[0]), conv.groups, (1, conv.dilation[0]))
        if conv.padding_mode != 'zeros' or isinstance(conv.padding, str):
            raise ValueError('Channels-last layout supports zero-padded convolutions with explicit padding')
        return F.conv2d(x, weight, conv.bias, (1, conv.stride[0]), (0, conv.padding[0]),
                        (1, conv.dilation[0]), conv.groups)

    @classmethod
    def _block_channels_last(cls, block, x):
        conv, norm, activation = block.block  # LeRobot Conv1d -> GroupNorm -> Mish
        return activation(norm(cls._conv_channels_last(conv, x)))

    @classmethod
    def _residual_channels_last(cls, block, x, condition):
        out = cls._block_channels_last(block.conv1, x)
        length = out.shape[-1]
        # Nearest resampling by an integer factor selects every step-th position.
        step = condition.shape[1] // length
        local = (condition[:, ::step] if step * length == condition.shape[1] else
                 F.interpolate(condition.transpose(1, 2), size=length, mode='nearest').transpose(1, 2))
        encoded = block.cond_encoder(local).transpose(1, 2).unsqueeze(2)
        if block.use_film_scale_modulation:
            scale, bias = encoded.split(block.out_channels, dim=1)
            out = scale * out + bias
        else:
            out = out + encoded
        return cls._block_channels_last(block.conv2, out) + cls._conv_channels_last(block.residual_conv, x)

    def _forward_channels_last(self, x, condition):
        x = x.transpose(1, 2).unsqueeze(2)
        skips = []
        for first, second, down in self.down_modules:
            x = self._residual_channels_last(second, self._residual_channels_last(first, x, condition), condition)
            skips.append(x)
            x = self._conv_channels_last(down, x)
        for middle in self.mid_modules:
            x = self._residual_channels_last(middle, x, condition)
        for first, second, up in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = self._conv_channels_last(
                up, self._residual_channels_last(second, self._residual_channels_last(first, x, condition), condition))
        block, conv = self.final_conv
        return self._conv_channels_last(conv, self._block_channels_last(block, x)).squeeze(2).transpose(1, 2)

    def forward(self, x, timestep, global_cond=None):
        if timestep.shape != x.shape[:2]:
            raise ValueError(f'Per-action noise times must have shape {tuple(x.shape[:2])}, got {tuple(timestep.shape)}')
        batch, horizon = x.shape[:2]
        # The time encoder's Linear is never replaced by inference accelerations, so it names the weight dtype.
        dtype = self.diffusion_step_encoder[1].weight.dtype
        embedding = self.diffusion_step_encoder[0](timestep.reshape(-1).float()).to(dtype)
        condition = self.diffusion_step_encoder[1:](embedding).reshape(batch, horizon, -1)
        if global_cond is not None:
            global_cond = global_cond.to(dtype)
            condition = torch.cat((condition, global_cond[:, None].expand(-1, horizon, -1)), dim=-1)
        x = x.to(dtype)
        if self.channels_last:
            return self._forward_channels_last(x, condition)
        x = x.transpose(1, 2)
        skips = []
        for first, second, down in self.down_modules:
            x = self._residual(second, self._residual(first, x, condition), condition)
            skips.append(x)
            x = down(x)
        for middle in self.mid_modules:
            x = self._residual(middle, x, condition)
        for first, second, up in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = up(self._residual(second, self._residual(first, x, condition), condition))
        return self.final_conv(x).transpose(1, 2)


def sample_noise_time(batch_size, *, sampling, device, dtype):
    """Noise time ~ Beta(1.5, 1) * 0.999 + 0.001 (or uniform on the same range)."""
    if sampling == 'beta':
        distribution = torch.distributions.Beta(torch.tensor(1.5, device=device, dtype=dtype),
                                                torch.tensor(1.0, device=device, dtype=dtype))
        return distribution.sample((batch_size,)) * 0.999 + 0.001
    return torch.rand((batch_size,), device=device, dtype=dtype) * 0.999 + 0.001


def per_action_flow_loss(actions, noise, noise_time, delays, predict, is_pad=None):
    """Noise at t=1, data at t=0; the first ``delays`` actions are clean (time 0) and unsupervised."""
    if actions.ndim != 3 or noise.shape != actions.shape:
        raise ValueError('Actions and noise must have the same [batch, horizon, action_dim] shape')
    if noise_time.shape != actions.shape[:1] or delays.shape != actions.shape[:1]:
        raise ValueError('Noise time and delay must have shape [batch]')
    if is_pad is not None and (is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool):
        raise ValueError('Padding mask must be boolean [batch, horizon]')
    prefix_mask = torch.arange(actions.shape[1], device=actions.device)[None] < delays[:, None]
    times = torch.where(prefix_mask, 0.0, noise_time[:, None].expand(actions.shape[:2]))
    x_t = times[..., None] * noise + (1 - times[..., None]) * actions
    velocity = predict(x_t, times)
    if velocity.shape != actions.shape:
        raise ValueError('Flow velocity shape must match actions')
    error = (velocity - (noise - actions)).square().mean(dim=-1)
    valid = ~prefix_mask if is_pad is None else ~prefix_mask & ~is_pad
    # Average valid actions within each example, then examples equally.
    loss = ((error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()
    return {
        'loss': loss,
        'flow_loss': loss,
        'flow_t_mean': noise_time.mean().detach(),
        'training_time_rtc_delay_mean': delays.float().mean().detach(),
        'training_time_rtc_prefix_fraction': prefix_mask.float().mean().detach(),
    }


def per_action_sample(unet, condition, noise, num_steps, prefix=None):
    """Integrate t from 1 (noise) to 0 (data). A prefix is pinned at time 0 on every step."""
    batch, horizon = noise.shape[:2]
    prefix_len = 0 if prefix is None else prefix.shape[1]
    sample = noise
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=noise.device, dtype=noise.dtype)
    for index in range(num_steps):
        per_action_time = times[index].expand(batch, horizon).clone()
        if prefix_len:
            sample = sample.clone()
            sample[:, :prefix_len] = prefix
            per_action_time[:, :prefix_len] = 0
        velocity = unet(sample, per_action_time, condition)
        sample = sample + (times[index + 1] - times[index]) * velocity
    if prefix_len:
        sample = sample.clone()
        sample[:, :prefix_len] = prefix
    return sample
