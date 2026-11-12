"""Decoder-only ACT: pretrained ResNet features and a BERT-style transformer.

One token sequence -- the proprio projection, every camera's
projected backbone feature map (cameras side by side) and one zero-initialized
query per action -- passes through self-attention blocks with a learned
positional embedding per position; a linear head reads each action from its
query's output. Training minimizes L1 over valid (non-padding) actions.

The model uses a ResNet with frozen BatchNorm (pretrained, ImageNet-normalized
inputs), a 1x1 input projection, a learned positional embedding for every token,
separate backbones per camera unless ``same_backbones``, and the backbone trained
at its own learning rate. There is no VAE/encoder-decoder variant.

Training-time RTC (``training_time_rtc``) is opt-in: the first d ~ U{min..max}
action queries carry the committed prefix (a linear projection of the clean
normalized action plus a learned prefix flag) and are excluded from the loss, so
inference can condition each chunk on the previous plan's next d actions (as
Flow RTC).

The attention has no mask, so PyTorch's scaled-dot-product attention can use its
FlashAttention-2 kernel. ACT is bf16 only (fp32 master weights in the optimizer);
the frozen BatchNorm statistics stay fp32.
--fast_inference replays the whole batch-1 forward from a CUDA graph
(``accelerate_act_policy``; a fixed-size masked RTC prefix is a graph input).
"""

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from contracts import standard_image_processing


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters (eps inside the rsqrt)."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict.pop(prefix + 'num_batches_tracked', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale.to(x.dtype).reshape(1, -1, 1, 1) + bias.to(x.dtype).reshape(1, -1, 1, 1)


class Backbone(nn.Module):
    """A torchvision ResNet's layer4 features, with frozen BatchNorm; weights load separately."""

    def __init__(self, name):
        super().__init__()
        if not name.startswith('resnet'):
            raise ValueError(f'ACT supports ResNet backbones, got {name!r}')
        resnet = getattr(torchvision.models, name)(weights=None, norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(resnet, return_layers={'layer4': '0'})
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

    def forward(self, x):
        return self.body(x)['0']


class TransformerBlock(nn.Module):
    """Post-norm self-attention block (LayerNorm, attention, LayerNorm, GELU MLP, LayerNorm)."""

    def __init__(self, latent_dim, num_head, dropout_rate):
        super().__init__()
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 4 * latent_dim), nn.GELU(),
                                 nn.Linear(4 * latent_dim, latent_dim), nn.Dropout(dropout_rate))
        self.ln_3 = nn.LayerNorm(latent_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.dropout1(self.attn(x, x, x, need_weights=False)[0])
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return self.ln_3(x)


class ACTDecoder(nn.Module):
    def __init__(self, *, context_len, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.weight_pos_embed = nn.Embedding(context_len, d_model)
        self.attention_blocks = nn.Sequential(*[TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, tokens):  # (sequence, batch, d_model)
        return self.attention_blocks(tokens + self.weight_pos_embed.weight[:, None])


class ACTModel(nn.Module):
    def __init__(self, *, backbone, camera_names, same_backbones, state_dim, action_dim, num_queries, hidden_dim,
                 nheads, dec_layers, dropout, image_shape, rtc=False):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = list(camera_names)
        count = 1 if same_backbones else len(self.camera_names)
        self.backbones = nn.ModuleList(Backbone(backbone) for _ in range(count))
        self.input_proj = nn.Conv2d(self.backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        with torch.no_grad():
            height, width = self.backbones[0](torch.zeros(1, 3, *image_shape)).shape[-2:]
        self.feature_hw = (height, width * len(self.camera_names))
        self.image_tokens = height * width * len(self.camera_names)
        self.decoder = ACTDecoder(context_len=1 + self.image_tokens + num_queries, d_model=hidden_dim, nhead=nheads,
                                  num_layers=dec_layers, dropout=dropout)
        if rtc:
            # Training-time RTC: committed prefix actions enter their queries as content.
            self.rtc_prefix_proj = nn.Linear(action_dim, hidden_dim)
            self.rtc_prefix_flag = nn.Parameter(torch.zeros(hidden_dim))

    def use_channels_last(self):
        """Run the ResNet backbones and input projection in channels-last (NHWC) memory, as Flow's
        U-Net trains: same parameters and math, faster bf16 convolutions; state_dict layout unchanged."""
        self.backbones.to(memory_format=torch.channels_last)
        self.input_proj.to(memory_format=torch.channels_last)
        self.channels_last = True

    def _image_features(self, image):
        batch = image.shape[0]
        layout = torch.channels_last if getattr(self, 'channels_last', False) else torch.contiguous_format
        if len(self.backbones) > 1:
            return torch.cat([self.input_proj(self.backbones[i](image[:, i].contiguous(memory_format=layout)))
                              for i in range(len(self.camera_names))], dim=3)
        projected = self.input_proj(self.backbones[0](image.flatten(0, 1).contiguous(memory_format=layout)))
        return torch.cat(projected.unflatten(0, (batch, len(self.camera_names))).unbind(1), dim=3)

    def forward(self, qpos, image, prefix=None, prefix_mask=None):
        batch = image.shape[0]
        # Cameras side by side along the width.
        src = self._image_features(image).flatten(2).permute(2, 0, 1)  # (image tokens, batch, hidden)
        proprio = self.input_proj_robot_state(qpos)[None]
        queries = src.new_zeros(self.num_queries, batch, src.shape[-1])
        if prefix is not None:
            # prefix (B, queries, action_dim) normalized; prefix_mask (B, queries) marks committed actions.
            content = self.rtc_prefix_proj(prefix.to(src.dtype)) + self.rtc_prefix_flag.to(src.dtype)
            queries = (content * prefix_mask[..., None].to(src.dtype)).transpose(0, 1)
        hidden = self.decoder(torch.cat((proprio, src, queries), dim=0))
        return self.action_head(hidden[-self.num_queries:].transpose(0, 1))


class ACTPolicy(nn.Module):
    """Decoder-only ACT behind the trainer/inference interface of the Flow policy."""

    def __init__(self, args_override):
        standard_image_processing.require_single_frame(args_override)
        super().__init__()
        self.training_time_rtc = bool(args_override.get('training_time_rtc', False))
        self.training_time_rtc_min_delay = int(args_override.get('training_time_rtc_min_delay', 0))
        self.training_time_rtc_max_delay = int(args_override.get('training_time_rtc_max_delay', 0))
        if self.training_time_rtc and not (
                0 <= self.training_time_rtc_min_delay <= self.training_time_rtc_max_delay
                < int(args_override['num_queries'])):
            raise ValueError('ACT RTC delays must satisfy 0 <= min <= max < chunk size')
        self.precision = str(args_override.get('precision', 'bf16'))
        if self.precision != 'bf16':
            raise ValueError(f'ACT is bf16 only, got precision {self.precision!r}')
        self.compute_dtype = torch.bfloat16
        self.lr = float(args_override['lr'])
        self.lr_backbone = float(args_override.get('lr_backbone', 1e-5))
        self.weight_decay = float(args_override.get('weight_decay', 1e-4))
        self.camera_names = list(args_override['camera_names'])
        self.base_camera_names = list(args_override.get('base_camera_names', self.camera_names))
        self.num_queries = int(args_override['num_queries'])
        self.state_dim = int(args_override['state_dim'])
        self.action_dim = int(args_override['action_dim'])
        self.state_idx = args_override.get('state_idx')
        self.action_idx = args_override.get('action_idx')
        self.backbone_name = args_override.get('backbone', 'resnet18')
        self.pretrained_backbone_weights = args_override.get('pretrained_backbone_weights')
        if args_override.get('height') is None or args_override.get('width') is None:
            raise ValueError('ACT requires explicit image height and width')
        self.image_shape = (int(args_override['height']), int(args_override['width']))
        hidden_dim = int(args_override.get('hidden_dim') or 512)
        self.model = ACTModel(
            rtc=self.training_time_rtc,
            backbone=self.backbone_name, camera_names=self.camera_names,
            same_backbones=bool(args_override.get('same_backbones', False)),
            state_dim=self.state_dim, action_dim=self.action_dim, num_queries=self.num_queries,
            hidden_dim=hidden_dim, nheads=int(args_override.get('nheads', 8)),
            dec_layers=int(args_override.get('dec_layers', 7)), dropout=float(args_override.get('dropout', 0.1)),
            image_shape=self.image_shape)
        self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.model.to(self.compute_dtype)
        for module in self.model.modules():  # frozen statistics stay fp32 (cast per call)
            if isinstance(module, FrozenBatchNorm2d):
                module.float()
        self.optimizer = None

    def load_pretrained_backbone(self):
        """Initialize every camera backbone with the configured torchvision weights (training only)."""
        if not self.pretrained_backbone_weights:
            raise ValueError('No pretrained_backbone_weights configured')
        reference = getattr(torchvision.models, self.backbone_name)(weights=self.pretrained_backbone_weights)
        state = {k: v for k, v in reference.state_dict().items() if not k.startswith('fc.')}
        for backbone in self.model.backbones:
            backbone.body.load_state_dict(state)  # strict; num_batches_tracked is dropped by FrozenBatchNorm2d
        return self

    def _inputs(self, qpos, image):
        if image.dim() != 5 or image.shape[1] != len(self.camera_names):
            raise ValueError(f'Expected images (B, {len(self.camera_names)}, C, H, W), got {tuple(image.shape)}')
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        # Backbone normalization inside the model, as for Flow; then the policy's compute dtype.
        image = ((image - self._imagenet_mean) / self._imagenet_std).to(self.compute_dtype)
        return qpos.to(self.compute_dtype), image

    def _compute_loss(self, qpos, image, actions, is_pad):
        if self.action_idx is not None:
            actions = actions[:, :, self.action_idx]
        actions = actions[:, :self.num_queries]
        is_pad = (torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device) if is_pad is None
                  else is_pad[:, :self.num_queries])
        args = self._inputs(qpos, image)
        if self.training_time_rtc:
            # Committed prefixes of d ~ U{min..max} clean actions; only the rest is predicted.
            delay = torch.randint(self.training_time_rtc_min_delay, self.training_time_rtc_max_delay + 1,
                                  (actions.shape[0],), device=actions.device)
            prefix_mask = torch.arange(self.num_queries, device=actions.device)[None] < delay[:, None]
            a_hat = self.model(*args, actions, prefix_mask).float()
            is_pad = is_pad | prefix_mask
        else:
            a_hat = self.model(*args).float()
        valid = (~is_pad).unsqueeze(-1).to(a_hat.dtype)
        l1 = (F.l1_loss(a_hat, actions, reduction='none') * valid).sum() / valid.expand_as(a_hat).sum().clamp(min=1.0)
        return {'loss': l1, 'l1': l1}

    def _model_args(self, qpos, image, rtc_action_prefix=None):
        """The model's inputs, always in the same structure for a variant (so one CUDA graph serves
        every query): (qpos, image[, prefix (B, queries, action_dim), prefix_mask])."""
        batch = image.shape[0]
        args = self._inputs(qpos, image)
        if self.training_time_rtc:
            delay = 0 if rtc_action_prefix is None else rtc_action_prefix.shape[1]
            if delay and (rtc_action_prefix.dim() != 3 or rtc_action_prefix.shape[-1] != self.action_dim):
                raise ValueError(f'RTC prefix must have shape (B, delay, {self.action_dim})')
            if delay > self.training_time_rtc_max_delay:
                raise ValueError(f'RTC prefix length {delay} exceeds the trained maximum {self.training_time_rtc_max_delay}')
            prefix = torch.zeros((batch, self.num_queries, self.action_dim), device=image.device)
            if delay:
                prefix[:, :delay] = rtc_action_prefix.to(prefix)
            prefix_mask = (torch.arange(self.num_queries, device=image.device) < delay)[None].expand(batch, -1)
            args += (prefix, prefix_mask)
        elif rtc_action_prefix is not None:
            raise ValueError('An RTC prefix requires an ACT model trained with training-time RTC')
        return args

    @torch.no_grad()
    def forward_inf(self, qpos, image):
        """A chunk; RTC models run at delay 0."""
        return self.model(*self._model_args(qpos, image)).float()

    @torch.no_grad()
    def forward_inf_training_time_rtc(self, qpos, image, rtc_action_prefix):
        """A chunk conditioned on the committed (normalized) prefix (B or 1, d, action_dim); its first d
        actions are the prefix itself, as Flow pins its RTC prefix."""
        if not self.training_time_rtc:
            raise ValueError('An RTC prefix requires an ACT model trained with training-time RTC')
        args = self._model_args(qpos, image, rtc_action_prefix)
        chunk = self.model(*args).float()
        delay = 0 if rtc_action_prefix is None else rtc_action_prefix.shape[1]
        chunk[:, :delay] = args[-2][:, :delay]
        return chunk

    def __call__(self, qpos, image, actions=None, is_pad=None):
        if actions is None:
            return self.forward_inf(qpos, image)
        return self._compute_loss(qpos, image, actions, is_pad)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer
        backbone = [p for n, p in self.model.named_parameters() if 'backbone' in n and p.requires_grad]
        rest = [p for n, p in self.model.named_parameters() if 'backbone' not in n and p.requires_grad]
        from policy.mixed_precision import MasterWeightAdamW
        groups = [{'params': rest}, {'params': backbone, 'lr': self.lr_backbone}]
        self.optimizer = MasterWeightAdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        return self.optimizer

    def serialize(self):
        state = self.model.state_dict()
        for key, value in state.items():
            state[key] = value.detach().cpu().contiguous()
        return state

    def deserialize(self, model_dict):
        if isinstance(model_dict, dict) and 'model_state_dict' in model_dict:
            model_dict = model_dict['model_state_dict']
        return self.model.load_state_dict(model_dict, strict=True)


class GraphedACTModel(nn.Module):
    """Replays the whole ACT forward from a CUDA graph for the captured input shapes; others run eagerly.

    Inputs are the variant's fixed argument structure (``ACTPolicy._model_args``): the
    fixed-size masked RTC prefix is a graph input like qpos and the images.
    """

    def __init__(self, model, *example):
        super().__init__()
        self.model = model
        self.inputs = tuple(x.clone() for x in example)
        with torch.inference_mode():
            device = example[0].device
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm up cuDNN/cuBLAS outside the capture
                    model(*self.inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = model(*self.inputs)

    def forward(self, *args):
        if len(args) != len(self.inputs) or any(
                a.shape != s.shape or a.dtype != s.dtype for a, s in zip(args, self.inputs)):
            return self.model(*args)
        for static, value in zip(self.inputs, args):
            static.copy_(value)
        self.graph.replay()
        return self.out.clone()  # the next replay overwrites the static output


def accelerate_act_policy(policy):
    """Channels-last backbones and a CUDA-graphed batch-1 forward for an eval-mode ACT policy, in place
    (every variant). Not for training or saving."""
    if policy.training:
        raise ValueError('Fast ACT inference is for eval-mode policies only')
    device = next(policy.model.parameters()).device
    if device.type != 'cuda':
        raise ValueError(f'Fast ACT inference requires CUDA, got {device}')
    # NHWC backbones as in training (~0.6 ms of a ~3 ms query on a laptop RTX 3070 Ti).
    policy.model.use_channels_last()
    qpos = torch.zeros(1, policy.state_dim, device=device)
    image = torch.zeros(1, len(policy.camera_names), 3, *policy.image_shape, device=device)
    example = policy._model_args(qpos, image)
    policy.model = GraphedACTModel(policy.model, *example)
    policy.fast_inference = True
    print('Fast ACT inference: channels-last backbones, CUDA-graphed forward')
    return policy
