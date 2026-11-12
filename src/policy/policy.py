from contracts import standard_image_processing
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np
import importlib
import pathlib
import sys
import types
import IPython
e = IPython.embed


def _install_lerobot_policies_import_shim():
    import lerobot

    policies_dir = pathlib.Path(lerobot.__file__).parent / "policies"
    if not policies_dir.is_dir():
        return

    shim = types.ModuleType("lerobot.policies")
    shim.__path__ = [str(policies_dir)]
    shim.__package__ = "lerobot"
    sys.modules["lerobot.policies"] = shim


def _import_lerobot_diffusion_modules():
    candidates = [
        (
            "lerobot.common.policies.diffusion.modeling_diffusion",
            "lerobot.common.policies.diffusion.configuration_diffusion",
        ),
        (
            "lerobot.policies.diffusion.modeling_diffusion",
            "lerobot.policies.diffusion.configuration_diffusion",
        ),
    ]
    last_err = None
    for modeling_path, config_path in candidates:
        try:
            if modeling_path.startswith("lerobot.policies."):
                _install_lerobot_policies_import_shim()
            modeling_module = importlib.import_module(modeling_path)
            config_module = importlib.import_module(config_path)
            types_module = importlib.import_module("lerobot.configs.types")
            return (
                getattr(modeling_module, "DiffusionPolicy"),
                getattr(config_module, "DiffusionConfig"),
                getattr(types_module, "PolicyFeature"),
                getattr(types_module, "FeatureType"),
                getattr(types_module, "NormalizationMode"),
            )
        except Exception as err:
            last_err = err
    raise ImportError(
        "LeRobot diffusion policy is unavailable. Install lerobot, then retry with "
        "--policy_class Flow."
    ) from last_err


class _FlowBackbone(nn.Module):
    def __init__(self, args_override):
        standard_image_processing.require_single_frame(args_override)
        super().__init__()

        (
            diffusion_policy_cls,
            diffusion_config_cls,
            policy_feature_cls,
            feature_type_cls,
            normalization_mode_cls,
        ) = _import_lerobot_diffusion_modules()

        from policy.mixed_precision import policy_dtype
        self.precision = str(args_override.get('precision', 'fp32'))
        self.compute_dtype = policy_dtype(self.precision)
        self.lr = args_override['lr']
        self.weight_decay = args_override.get('weight_decay', 1e-6)
        self.camera_names = list(args_override['camera_names'])
        self.base_camera_names = list(args_override.get('base_camera_names', self.camera_names))
        self.height = args_override['height']
        self.width = args_override['width']
        self.state_dim = args_override['state_dim']
        self.action_dim = args_override['action_dim']
        self.observation_horizon = 1
        self.action_horizon = int(args_override.get('action_horizon', args_override['num_queries']))
        self.prediction_horizon = int(args_override.get('prediction_horizon', args_override['num_queries']))
        self.num_inference_timesteps = args_override.get('num_inference_timesteps', 100)
        sampler_type = args_override.get("noise_scheduler_type", "DDIM")
        self.noise_scheduler_type = str(sampler_type).upper()
        if self.noise_scheduler_type not in ("DDIM", "DDPM"):
            raise ValueError(
                f"Unsupported diffusion sampler '{sampler_type}'. Expected one of: DDIM, DDPM."
            )
        self.state_idx = args_override.get('state_idx')
        self.action_idx = args_override.get('action_idx')
        raw_hidden_dim = args_override.get("hidden_dim", 512)
        if raw_hidden_dim is None:
            raw_hidden_dim = 512
        self.hidden_dim = int(raw_hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0 for diffusion/flow policies, got {self.hidden_dim}.")
        raw_unet_depth = args_override.get("unet_depth", 3)
        self.unet_depth = int(raw_unet_depth)
        if self.unet_depth <= 0:
            raise ValueError(f"unet_depth must be > 0 for diffusion/flow policies, got {self.unet_depth}.")
        default_down_dims = tuple((2 ** i) * self.hidden_dim for i in range(self.unet_depth))
        self.down_dims = tuple(args_override.get("down_dims", default_down_dims))
        self.crop_shape = None
        self.crop_is_random = bool(args_override.get("crop_is_random", True))
        # Without weights: the from-scratch GroupNorm encoder of older checkpoints, kept so they still
        # load for inference; the trainer always sets weights. Torchvision weights (e.g. "IMAGENET1K_V1") keep the backbone's BatchNorm, since
        # GroupNorm would discard the pretrained statistics. They only initialize training
        # (load_pretrained_backbone); the policy itself never downloads them.
        self.pretrained_backbone_weights = args_override.get("pretrained_backbone_weights", None)
        self.use_group_norm = bool(args_override.get("use_group_norm", self.pretrained_backbone_weights is None))
        if self.pretrained_backbone_weights and self.use_group_norm:
            raise ValueError("A pretrained backbone keeps BatchNorm; it cannot use use_group_norm")
        # One shared image encoder for every camera (LeRobot's default) unless set.
        self.use_separate_rgb_encoder_per_camera = bool(args_override.get("use_separate_rgb_encoder_per_camera", False))
        self.do_mask_loss_for_padding = bool(args_override.get("do_mask_loss_for_padding", True))
        # We train/infer in z-score action space; disable diffusion sample clipping by default.
        self.clip_sample = bool(args_override.get("clip_sample", False))
        self.clip_sample_range = float(args_override.get("clip_sample_range", 1.0))
        self.image_normalization = str(
            args_override.get("image_normalization", "imagenet")
        ).lower()
        if self.image_normalization not in ("imagenet", "minus_one_one"):
            raise ValueError(
                "image_normalization must be 'imagenet' or 'minus_one_one', "
                f"got {self.image_normalization!r}."
            )
        if self.pretrained_backbone_weights and self.image_normalization != "imagenet":
            raise ValueError("A pretrained (ImageNet) backbone requires image_normalization 'imagenet'")
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        if self.height is None or self.width is None:
            raise ValueError(
                "Diffusion policy requires explicit image dimensions. Set --height and --width "
                "to positive values."
            )

        input_features = {
            "observation.state": policy_feature_cls(type=feature_type_cls.STATE, shape=(self.state_dim,))
        }
        for camera_name in self.camera_names:
            input_features[f"observation.images.{camera_name}"] = policy_feature_cls(
                type=feature_type_cls.VISUAL,
                shape=(3, self.height, self.width),
            )
        output_features = {
            "action": policy_feature_cls(type=feature_type_cls.ACTION, shape=(self.action_dim,))
        }

        config_kwargs = {
            "n_obs_steps": self.observation_horizon,
            "horizon": self.prediction_horizon,
            "n_action_steps": self.action_horizon,
            "num_inference_steps": self.num_inference_timesteps,
            "noise_scheduler_type": self.noise_scheduler_type,
            "down_dims": self._select_compatible_down_dims(self.prediction_horizon, self.down_dims),
            # Explicitly override LeRobot defaults (some versions default to crop_shape=(84, 84)).
            "crop_shape": self.crop_shape,
            "crop_is_random": self.crop_is_random,
            # Architecture only; see load_pretrained_backbone.
            "pretrained_backbone_weights": None,
            "use_group_norm": self.use_group_norm,
            "use_separate_rgb_encoder_per_camera": self.use_separate_rgb_encoder_per_camera,
            "do_mask_loss_for_padding": self.do_mask_loss_for_padding,
            "clip_sample": self.clip_sample,
            "clip_sample_range": self.clip_sample_range,
            "input_features": input_features,
            "output_features": output_features,
        }
        if 'backbone' in args_override:
            config_kwargs['vision_backbone'] = args_override['backbone']

        self.config = diffusion_config_cls(**config_kwargs)
        try:
            self.model = diffusion_policy_cls(self.config, dataset_stats=None)
        except TypeError:
            self.model = diffusion_policy_cls(self.config)
        # Built on first use, after the policy reaches its training device.
        self.optimizer = None

    def load_pretrained_backbone(self):
        """Initialize each camera encoder's backbone with the configured torchvision weights."""
        if not self.pretrained_backbone_weights:
            raise ValueError("No pretrained_backbone_weights configured")
        import torchvision
        reference = getattr(torchvision.models, self.config.vision_backbone)(weights=self.pretrained_backbone_weights)
        state = nn.Sequential(*list(reference.children())[:-2]).state_dict()  # LeRobot's backbone slice
        encoders = self.model.diffusion.rgb_encoder
        for encoder in (encoders if isinstance(encoders, nn.ModuleList) else [encoders]):
            encoder.backbone.load_state_dict(state)  # strict; copies into the policy's dtype/layout
        return self

    def _select_compatible_down_dims(self, horizon, down_dims):
        """Keep the configured U-Net depth; never shrink it to fit the chunk.

        The (lerobot) U-Net requires the horizon to be a multiple of
        2 ** len(down_dims). A mismatch is a configuration error: training a
        shallower network than the recipe declares would be silent.
        """
        factor = 2 ** len(down_dims)
        if len(down_dims) and horizon % factor:
            raise ValueError(
                f"Flow chunk_size={horizon} is not divisible by 2**unet_depth={factor} "
                f"(down_dims={tuple(down_dims)}). Choose model.chunk_size as a multiple of {factor} "
                "or a compatible model.unet_depth in the training config.")
        return down_dims

    def _prepare_observation_tensors(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]

        if image.dim() != 5:
            raise ValueError(f"Expected image tensor of shape (B, V, C, H, W), got {tuple(image.shape)}")
        if image.shape[1] != len(self.camera_names):
            expected_desc = f"{len(self.camera_names)} effective image views"
            raise ValueError(
                f"Expected {expected_desc}, got {image.shape[1]}. "
                "Ensure camera names match the saved single-frame policy config."
            )
        if self.image_normalization == "minus_one_one":
            # RGB [0, 1] -> [-1, 1].
            image = image * 2.0 - 1.0
        else:
            # Preserve the standard Flow/Diffusion/ACT image contract.
            image = (image - self._imagenet_mean) / self._imagenet_std
        # Normalized in fp32 as before; a bf16 policy then runs its encoder in bf16.
        image = image.to(self.compute_dtype)

        # LeRobot diffusion expects temporal observation dimensions.
        qpos_obs = qpos.unsqueeze(1).repeat(1, self.observation_horizon, 1)
        image_obs = image.unsqueeze(1).repeat(1, self.observation_horizon, 1, 1, 1, 1)
        return qpos_obs, image_obs

    def _build_batch(self, qpos, image, actions=None, is_pad=None):
        qpos_obs, image_obs = self._prepare_observation_tensors(qpos, image)
        batch = {"observation.state": qpos_obs}
        # Some LeRobot diffusion versions expect a stacked observation.images tensor
        # with shape (B, S, N, C, H, W), while others read per-camera keys.
        # Provide both for cross-version compatibility.
        batch["observation.images"] = image_obs
        for cam_idx, camera_name in enumerate(self.camera_names):
            batch[f"observation.images.{camera_name}"] = image_obs[:, :, cam_idx]

        if actions is not None:
            batch["action"] = actions
            if is_pad is not None:
                batch["action_is_pad"] = is_pad[:, :self.prediction_horizon]
        return batch



    def configure_optimizers(self):
        if hasattr(self, "optimizer") and self.optimizer is not None:
            return self.optimizer
        if hasattr(self.model, "get_optim_params"):
            params = list(self.model.get_optim_params())
        else:
            params = list(self.model.parameters())
        if self.compute_dtype != torch.float32:
            from policy.mixed_precision import MasterWeightAdamW
            self.optimizer = MasterWeightAdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            fused = True if all(p.is_cuda for p in params) else None
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, fused=fused)
        return self.optimizer

    def serialize(self):
        state = self.model.state_dict()
        for key, value in state.items():
            # Canonical layout whatever the training kernel layout. The device copy keeps the
            # strides (no GPU temporaries, which a full GPU cannot spare); the layout changes on CPU.
            state[key] = value.detach().cpu().contiguous()
        return state

    def deserialize(self, model_dict):
        if isinstance(model_dict, dict) and "model_state_dict" in model_dict:
            model_dict = model_dict["model_state_dict"]
        return self.model.load_state_dict(model_dict, strict=False)


class LeRobotFlowPolicy(_FlowBackbone):
    def __init__(self, args_override):
        standard_image_processing.require_single_frame(args_override)
        # Per-action noise times with the RTC prefix explicitly clean (see policy/flow_rtc.py).
        self.flow_loss_type = str(args_override.get("flow_loss_type", "flow_matching_mse"))
        if self.flow_loss_type != "flow_matching_mse":
            raise ValueError("The Flow objective requires flow_loss_type flow_matching_mse")
        self.flow_t_sampling = str(args_override.get("flow_t_sampling", "beta")).lower()
        if self.flow_t_sampling not in ("beta", "uniform"):
            raise ValueError(
                f"Unsupported flow_t_sampling '{self.flow_t_sampling}'. Expected one of: beta, uniform."
            )
        self.training_time_rtc = bool(args_override.get("training_time_rtc", True))
        self.training_time_rtc_min_delay = int(args_override.get("training_time_rtc_min_delay", 0))
        self.training_time_rtc_max_delay = int(args_override.get("training_time_rtc_max_delay", 10))
        self._training_time_rtc_horizon = args_override.get(
            "prediction_horizon", args_override.get("num_queries")
        )
        self._validate_training_time_rtc_config()
        super().__init__(args_override)
        from policy.flow_rtc import OBJECTIVE_MARKER, PerActionTimeUNet
        # Same parameters (the optimizer's references stay valid); per-action time FiLM.
        self.model.diffusion.unet = PerActionTimeUNet(self.model.diffusion.unet)
        # Saved with the weights: checkpoints of the removed global-time objective lack it and are rejected.
        self.model.register_buffer(OBJECTIVE_MARKER, torch.tensor(1, dtype=torch.int64))
        # bf16 policies store, train and infer with bf16 weights (the marker stays int64).
        self.model.to(self.compute_dtype)
        # BatchNorm (pretrained backbones) keeps fp32 weights and running statistics; it
        # normalizes bf16 activations directly, and bf16 running averages would drift.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.float()
        self._strict_flow_diffusion_module = getattr(self.model, "diffusion", None)
        if self._strict_flow_diffusion_module is None:
            raise RuntimeError(
                "FlowPolicy strict path requires `self.model.diffusion`, but it was not found."
            )
        if not hasattr(self._strict_flow_diffusion_module, "_prepare_global_conditioning"):
            raise RuntimeError(
                "FlowPolicy strict path requires diffusion._prepare_global_conditioning(batch)."
            )
        if not hasattr(self._strict_flow_diffusion_module, "unet"):
            raise RuntimeError(
                "FlowPolicy strict path requires diffusion.unet for timestep-conditioned prediction."
            )

    def _get_strict_flow_diffusion_module(self):
        return self._strict_flow_diffusion_module

    def _prepare_global_conditioning(self, batch):
        """Build Flow conditioning through an overridable policy hook."""
        return self._get_strict_flow_diffusion_module()._prepare_global_conditioning(batch)

    def _validate_training_time_rtc_config(self):
        if self.training_time_rtc_min_delay < 0:
            raise ValueError(
                f"training_time_rtc_min_delay must be >= 0, got {self.training_time_rtc_min_delay}."
            )
        if self.training_time_rtc_max_delay < self.training_time_rtc_min_delay:
            raise ValueError(
                "training_time_rtc_max_delay must be >= training_time_rtc_min_delay, "
                f"got {self.training_time_rtc_max_delay} < {self.training_time_rtc_min_delay}."
            )
        if self.training_time_rtc and self._training_time_rtc_horizon is not None:
            horizon = int(self._training_time_rtc_horizon)
            if self.training_time_rtc_max_delay >= horizon:
                raise ValueError(
                    "training_time_rtc_max_delay must be < prediction_horizon when training-time RTC is enabled, "
                    f"got max_delay={self.training_time_rtc_max_delay}, prediction_horizon={horizon}."
                )
    def _sample_training_time_rtc_delays(self, batch_size, device):
        min_delay = self.training_time_rtc_min_delay
        max_delay = self.training_time_rtc_max_delay
        if max_delay == min_delay:
            return torch.full((batch_size,), min_delay, device=device, dtype=torch.long)
        return torch.randint(min_delay, max_delay + 1, (batch_size,), device=device)

    def _extract_unet_prediction(self, model_pred):
        if torch.is_tensor(model_pred):
            return model_pred
        if hasattr(model_pred, "sample") and torch.is_tensor(model_pred.sample):
            return model_pred.sample
        raise RuntimeError(
            "FlowPolicy strict path expected diffusion.unet(...) to return a tensor "
            "or an object with tensor attribute `.sample`."
        )

    def _compute_flow_matching_loss(self, batch):
        from policy.flow_rtc import per_action_flow_loss, sample_noise_time
        actions = batch["action"]
        if actions.dim() != 3:
            raise ValueError(f"FlowPolicy expects action tensor with shape (B, T, D), got {tuple(actions.shape)}.")
        noise_time = sample_noise_time(len(actions), sampling=self.flow_t_sampling,
                                       device=actions.device, dtype=actions.dtype)
        # Validation uses the same delayed-prefix objective as training.
        delays = (self._sample_training_time_rtc_delays(len(actions), actions.device) if self.training_time_rtc
                  else torch.zeros(len(actions), device=actions.device, dtype=torch.long))
        condition = self._prepare_global_conditioning(batch)
        unet = self._get_strict_flow_diffusion_module().unet
        return per_action_flow_loss(
            actions, torch.randn_like(actions), noise_time, delays,
            lambda x, t: self._extract_unet_prediction(unet(x, t, global_cond=condition)),
            batch.get("action_is_pad"))

    @torch.no_grad()
    def forward_inf(self, qpos, image, prefix=None):
        from policy.flow_rtc import per_action_sample
        batch = self._build_batch(qpos, image)
        diffusion_module = self._get_strict_flow_diffusion_module()
        condition = self._prepare_global_conditioning(batch)
        state_obs = batch["observation.state"]
        batch_size = state_obs.shape[0]
        noise = torch.randn((batch_size, self.prediction_horizon, self.action_dim),
                            dtype=state_obs.dtype, device=state_obs.device)
        if prefix is not None:
            if prefix.dim() != 3 or prefix.shape[-1] != self.action_dim or prefix.shape[0] not in (1, batch_size):
                raise ValueError(f"RTC prefix must have shape (B, delay, {self.action_dim}), got {tuple(prefix.shape)}.")
            if prefix.shape[1] and not self.training_time_rtc:
                raise ValueError("An RTC prefix requires a model trained with training-time RTC.")
            if not prefix.shape[1] <= self.training_time_rtc_max_delay < self.prediction_horizon:
                raise ValueError(f"RTC prefix length {prefix.shape[1]} exceeds the trained delay range "
                                 f"[{self.training_time_rtc_min_delay}, {self.training_time_rtc_max_delay}].")
            prefix = prefix.to(noise).expand(batch_size, -1, -1) if prefix.shape[1] else None
        num_steps = max(1, int(getattr(diffusion_module, "num_inference_steps", self.num_inference_timesteps)))
        # The sampler integrates in fp32 whatever the U-Net's weight dtype.
        sample = per_action_sample(
            lambda x, t, c: self._extract_unet_prediction(diffusion_module.unet(x, t, global_cond=c)).to(noise.dtype),
            condition, noise, num_steps, prefix)
        return sample[:, :self.action_horizon]

    @torch.no_grad()
    def forward_inf_training_time_rtc(self, qpos, image, rtc_action_prefix):
        return self.forward_inf(qpos, image, rtc_action_prefix)

    def deserialize(self, model_dict):
        from policy.flow_rtc import OBJECTIVE_MARKER
        if isinstance(model_dict, dict) and "model_state_dict" in model_dict:
            model_dict = model_dict["model_state_dict"]
        model_dict = dict(model_dict)
        # The first per-action-time run (wandb ywkjc4xk) saved the marker as _flow_rtc_version=2.
        early = model_dict.pop("_flow_rtc_version", None)
        if early is not None and int(early) == 2:
            model_dict.setdefault(OBJECTIVE_MARKER, torch.tensor(1, dtype=torch.int64))
        if OBJECTIVE_MARKER not in model_dict:
            raise ValueError("Checkpoint was trained with the removed global-time Flow objective; retrain it.")
        return self.model.load_state_dict(model_dict, strict=True)

    def __call__(self, qpos, image, actions=None, is_pad=None):
        if actions is None:
            return self.forward_inf(qpos, image)

        batch = self._build_batch(qpos, image, actions, is_pad)
        return self._compute_flow_matching_loss(batch)



