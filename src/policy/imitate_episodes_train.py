# Support direct execution as well as python -m policy.imitate_episodes_train.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.action_contract import require_absolute_actions
from contracts import standard_image_processing
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib

# Try different backends - prefer interactive when a display is available,
# otherwise avoid probing GUI backends on headless training nodes.
_interactive_backend = False
_backend_candidates = ['QtAgg', 'TkAgg', 'Agg'] if os.environ.get('DISPLAY') else ['Agg']
for _backend in _backend_candidates:
    try:
        matplotlib.use(_backend, force=True)
        import matplotlib.pyplot as plt
        # Actually test if backend works by creating a figure
        _test_fig = plt.figure()
        plt.close(_test_fig)
        if _backend != 'Agg':
            _interactive_backend = True
        break
    except Exception:
        continue
else:
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import json
import wandb

from config import constants as constants_module
from config.constants import FPS

from policy.model_util import make_policy, make_optimizer
from policy.frame_store import default_frame_store_dir

import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta

from torch.nn.parallel import DistributedDataParallel as DDP


from transformers import get_cosine_schedule_with_warmup



torch.backends.cudnn.benchmark = True


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def load_data(*args, **kwargs):
    from policy.training_utils import load_data as implementation

    return implementation(*args, **kwargs)


def compute_dict_mean(*args, **kwargs):
    from policy.training_utils import compute_dict_mean as implementation

    return implementation(*args, **kwargs)


def set_seed(*args, **kwargs):
    from utils.random_utils import set_seed as implementation

    return implementation(*args, **kwargs)


def _load_standard_training_utils():
    """Import standard dataset utilities only for the standard training method."""
    global load_data, compute_dict_mean, set_seed
    from policy.training_utils import compute_dict_mean as _compute_dict_mean
    from policy.training_utils import load_data as _load_data
    from utils.random_utils import set_seed as _set_seed

    load_data = _load_data
    compute_dict_mean = _compute_dict_mean
    set_seed = _set_seed


def resolve_single_arm_side_for_training(task_name, task_config):
    """Resolve ARX single-arm side from constants.py for checkpoint metadata."""
    if int(task_config.get('action_dim', 40)) > 7:
        return None

    runtime_task_config = task_config if task_config.get('_task_config_path') else constants_module.TASK_CONFIGS.get(task_name, {})
    side = runtime_task_config.get('single_arm_side', task_config.get('single_arm_side'))
    if side is None:
        return None
    return constants_module.resolve_arx_single_arm_side({'single_arm_side': side})


def resolve_single_arm_can_for_training(task_name, task_config, single_arm_side):
    if single_arm_side is None:
        return None
    runtime_task_config = task_config if task_config.get('_task_config_path') else constants_module.TASK_CONFIGS.get(task_name, {})
    can_task_config = dict(task_config)
    can_task_config.update(runtime_task_config)
    can_task_config['single_arm_side'] = single_arm_side
    return constants_module.resolve_arx_single_arm_can(can_task_config)


def build_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Build a cosine decay scheduler with linear warmup.

    Uses Hugging Face's implementation when available, with a torch fallback
    to keep training functional even if transformers is not installed.
    """
    num_training_steps = max(1, int(num_training_steps))
    num_warmup_steps = max(0, min(int(num_warmup_steps), num_training_steps))

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def compute_annealed_value(step, initial_value, final_value, anneal_steps, schedule='linear'):
    """
    Compute an annealed value (for noise scale, dropout probability, etc.).

    Args:
        step: Current training step
        initial_value: Starting value
        final_value: Final value at end of annealing
        anneal_steps: Total steps over which to anneal
        schedule: 'linear' or 'exponential' annealing schedule

    Returns:
        Current annealed value
    """
    if anneal_steps <= 0:
        return initial_value

    progress = min(step / anneal_steps, 1.0)

    if schedule == 'exponential':
        # Exponential interpolation
        if initial_value > 0 and final_value > 0:
            ratio = final_value / initial_value
            value = initial_value * (ratio ** progress)
        elif initial_value > 0:
            # Decay to ~0
            decay_rate = 5.0
            value = initial_value * np.exp(-decay_rate * progress)
        else:
            value = final_value
    else:
        # Linear interpolation
        value = initial_value * (1.0 - progress) + final_value * progress

    return value


def should_dropout_camera(step, dropout_config, num_steps):
    """
    Determine if a camera should be dropped out based on curriculum config.

    Args:
        step: Current training step
        dropout_config: Dict with keys:
            - 'initial_prob': Starting dropout probability (e.g., 1.0 = always drop)
            - 'final_prob': Final dropout probability (e.g., 0.0 = never drop)
            - 'anneal_steps': Steps to anneal over (-1 for num_steps)
            - 'schedule': 'linear' or 'exponential'
        num_steps: Total training steps

    Returns:
        bool: True if camera should be dropped this step
    """
    if dropout_config is None:
        return False

    anneal_steps = dropout_config.get('anneal_steps', -1)
    if anneal_steps < 0:
        anneal_steps = num_steps

    dropout_prob = compute_annealed_value(
        step,
        dropout_config.get('initial_prob', 1.0),
        dropout_config.get('final_prob', 0.0),
        anneal_steps,
        dropout_config.get('schedule', 'linear')
    )

    return torch.rand(1).item() < dropout_prob


def apply_gaussian_blur(image_tensor, sigma):
    """
    Apply Gaussian blur to an image tensor.

    Args:
        image_tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        sigma: Standard deviation for Gaussian kernel. 0 means no blur.

    Returns:
        Blurred image tensor
    """
    if sigma <= 0:
        return image_tensor

    # Compute kernel size (must be odd, typically 6*sigma rounded up to odd)
    kernel_size = int(np.ceil(sigma * 6))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)  # Minimum kernel size of 3

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=image_tensor.device) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel via outer product
    kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # Handle input dimensions
    squeeze_batch = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze_batch = True

    B, C, H, W = image_tensor.shape
    padding = kernel_size // 2

    # Apply blur to each channel
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)
    blurred = torch.nn.functional.conv2d(
        image_tensor, kernel_2d, padding=padding, groups=C
    )

    if squeeze_batch:
        blurred = blurred.squeeze(0)

    return blurred


def compute_camera_blur_sigma(step, blur_config, num_steps):
    """
    Compute the blur sigma for a camera based on curriculum config.

    Args:
        step: Current training step
        blur_config: Dict with keys:
            - 'initial_sigma': Starting blur sigma (e.g., 10.0 = heavy blur)
            - 'final_sigma': Final blur sigma (e.g., 0.0 = no blur)
            - 'anneal_steps': Steps to anneal over (-1 for num_steps)
            - 'schedule': 'linear' or 'exponential'
        num_steps: Total training steps

    Returns:
        float: Current blur sigma
    """
    if blur_config is None:
        return 0.0

    anneal_steps = blur_config.get('anneal_steps', -1)
    if anneal_steps < 0:
        anneal_steps = num_steps

    return compute_annealed_value(
        step,
        blur_config.get('initial_sigma', 0.0),
        blur_config.get('final_sigma', 0.0),
        anneal_steps,
        blur_config.get('schedule', 'linear')
    )


def validate_training_time_rtc_args(args, chunk_size):
    min_delay = int(args.get('training_time_rtc_min_delay', 0))
    max_delay = int(args.get('training_time_rtc_max_delay', 10))
    distribution = str(args.get('training_time_rtc_delay_distribution', 'uniform')).lower()
    exp_decay = float(args.get('training_time_rtc_exp_decay', 1.0))
    enabled = not bool(args.get('no_training_time_rtc', False))

    if min_delay < 0:
        raise ValueError(f"training_time_rtc_min_delay must be >= 0, got {min_delay}.")
    if max_delay < min_delay:
        raise ValueError(
            "training_time_rtc_max_delay must be >= training_time_rtc_min_delay, "
            f"got {max_delay} < {min_delay}."
        )
    if chunk_size is None:
        raise ValueError("chunk_size is required for training-time RTC validation.")
    if enabled and max_delay >= int(chunk_size):
        raise ValueError(
            "training_time_rtc_max_delay must be < chunk_size when training-time RTC is enabled, "
            f"got max_delay={max_delay}, chunk_size={chunk_size}."
        )
    if distribution not in ('uniform', 'exp'):
        raise ValueError(
            "training_time_rtc_delay_distribution must be one of: uniform, exp; "
            f"got {distribution!r}."
        )
    if exp_decay <= 0:
        raise ValueError(f"training_time_rtc_exp_decay must be > 0, got {exp_decay}.")

    return {
        'training_time_rtc': enabled,
        'training_time_rtc_min_delay': min_delay,
        'training_time_rtc_max_delay': max_delay,
        'training_time_rtc_delay_distribution': distribution,
        'training_time_rtc_exp_decay': exp_decay,
    }


def validate_flow_training_time_rtc_args(args, chunk_size):
    # Flow RTC always samples prefix delays uniformly. Keep the distribution
    # flags available for ACT without carrying them into Flow checkpoints.
    uniform_args = dict(args)
    uniform_args['training_time_rtc_delay_distribution'] = 'uniform'
    uniform_args['training_time_rtc_exp_decay'] = 1.0
    config = validate_training_time_rtc_args(uniform_args, chunk_size)
    return {
        'training_time_rtc': config['training_time_rtc'],
        'training_time_rtc_min_delay': config['training_time_rtc_min_delay'],
        'training_time_rtc_max_delay': config['training_time_rtc_max_delay'],
    }


####===================FOR ROBOT EVAL===================####


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def inspect_training_data_interactive(data, batch_idx, camera_names):
    """Interactive data inspection function to examine training data"""
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cpu(), qpos_data.cpu(), action_data.cpu(), is_pad.cpu()
    batch_size = image_data.shape[0]

    print(f"\n=== Data Inspection - Batch {batch_idx} ===")
    print(f"Batch size: {batch_size}")
    print(f"Image data shape: {image_data.shape}")
    print(f"Qpos data shape: {qpos_data.shape}")
    print(f"Action data shape: {action_data.shape}")
    print(f"Is_pad shape: {is_pad.shape}")
    print(f"Camera names: {camera_names}")

    # Non-interactive mode: save images to files
    if not _interactive_backend:
        num_samples_to_save = min(3, batch_size)
        for sample_idx in range(num_samples_to_save):
            num_cameras = len(camera_names)
            images_per_camera = image_data.shape[1] // num_cameras if image_data.shape[1] > num_cameras else 1

            fig, axes = plt.subplots(1, num_cameras, figsize=(5*num_cameras, 5))
            if num_cameras == 1:
                axes = [axes]

            for cam_idx, cam_name in enumerate(camera_names):
                img_idx = cam_idx * images_per_camera if images_per_camera > 1 else cam_idx
                if img_idx < image_data.shape[1]:
                    img = image_data[sample_idx, img_idx].numpy().transpose(1, 2, 0)

                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    if img.shape[2] == 1:
                        img = np.squeeze(img, axis=2)
                        axes[cam_idx].imshow(img, cmap='gray')
                    else:
                        axes[cam_idx].imshow(img)

                    axes[cam_idx].set_title(f'{cam_name}')
                    axes[cam_idx].axis('off')

            plt.suptitle(f'Batch {batch_idx} - Sample {sample_idx}')
            plt.tight_layout()
            output_path = f'inspect_batch{batch_idx}_sample{sample_idx}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'Saved {output_path}')
            plt.close()

            print(f"Sample {sample_idx} - Qpos (first 10): {qpos_data[sample_idx, :10].numpy()}")
            print(f"Sample {sample_idx} - Action (first 10): {action_data[sample_idx, :10].numpy()}")

        return 'quit'

    # Interactive mode
    sample_idx = 0
    while True:
        print(f"\nExamining sample {sample_idx}/{batch_size-1}")

        # Display images for this sample
        num_cameras = len(camera_names)
        images_per_camera = image_data.shape[1] // num_cameras if image_data.shape[1] > num_cameras else 1

        fig, axes = plt.subplots(1, num_cameras, figsize=(5*num_cameras, 5))
        if num_cameras == 1:
            axes = [axes]

        for cam_idx, cam_name in enumerate(camera_names):
            # Get the first image for this camera
            img_idx = cam_idx * images_per_camera if images_per_camera > 1 else cam_idx
            if img_idx < image_data.shape[1]:
                img = image_data[sample_idx, img_idx].numpy().transpose(1, 2, 0)
                print(img.shape)

                # Normalize image for display if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # Handle grayscale images
                if img.shape[2] == 1:
                    img = np.squeeze(img, axis=2)
                    axes[cam_idx].imshow(img, cmap='gray')
                else:
                    axes[cam_idx].imshow(img)

                axes[cam_idx].set_title(f'{cam_name}')
                axes[cam_idx].axis('off')
            else:
                axes[cam_idx].text(0.5, 0.5, f'No image\nfor {cam_name}',
                                  ha='center', va='center', transform=axes[cam_idx].transAxes)
                axes[cam_idx].axis('off')

        plt.tight_layout()
        plt.show()

        # Display other data info
        print(f"Qpos data sample (first 10): {qpos_data[sample_idx, :10].numpy()}")
        print(f"Action data sample (first 10): {action_data[sample_idx, :10].numpy()}")
        print(f"Is_pad: {is_pad[sample_idx].numpy()}")

        print("\nControls:")
        print("  'n' or Enter - next sample")
        print("  'p' - previous sample")
        print("  '2' - next batch")
        print("  'q' - quit inspection")
        print("  's' - skip to next batch without viewing more samples")

        try:
            user_input = input("Command: ").strip().lower()

            if user_input == 'n' or user_input == '':
                sample_idx = min(sample_idx + 1, batch_size - 1)
                if sample_idx == batch_size - 1:
                    print("Reached last sample in batch. Use '2' for next batch.")
            elif user_input == 'p':
                sample_idx = max(sample_idx - 1, 0)
                if sample_idx == 0:
                    print("At first sample in batch.")
            elif user_input == '2' or user_input == 's':
                plt.close('all')
                return 'next_batch'
            elif user_input == 'q':
                plt.close('all')
                return 'quit'
            else:
                print("Invalid command. Use 'n', 'p', '2', 's', or 'q'.")

        except KeyboardInterrupt:
            plt.close('all')
            return 'quit'


def forward_pass(data, policy, image_history_len=0, proprio_noise_scale=0.0, qpos_std=None,
                 camera_dropout_indices=None, camera_blur_sigmas=None):
    """
    Forward pass through the policy.

    Args:
        data: Tuple of (image_data, qpos_data, action_data, is_pad)
        policy: The policy model
        image_history_len: Deprecated compatibility argument; must be zero.
        proprio_noise_scale: Multiplier for noise relative to qpos_std.
                             Set to 0.0 to disable noise (e.g., during validation).
        qpos_std: Standard deviation of qpos from dataset stats (tensor on GPU).
                  If provided, noise is scaled per-dimension by qpos_std.
        camera_dropout_indices: List of camera indices to zero out (dropout).
                                None means no dropout.
        camera_blur_sigmas: Dict mapping camera index to blur sigma.
                            None means no blur.
    """
    standard_image_processing.require_single_frame({"image_history_len": image_history_len})
    image_data, qpos_data, action_data, is_pad = (t.cuda(non_blocking=True) for t in data)

    # Add noise to proprioception (qpos) during training to encourage visual reliance
    # Noise is scaled by the dataset's qpos standard deviation per dimension
    if proprio_noise_scale > 0.0:
        if qpos_std is not None:
            # Scale noise by per-dimension std: noise ~ N(0, scale * qpos_std)
            noise_std = qpos_std
            if noise_std.shape[-1] != qpos_data.shape[-1]:
                if qpos_data.shape[-1] % noise_std.shape[-1] != 0:
                    raise ValueError(
                        f"qpos_std shape {tuple(noise_std.shape)} is incompatible with "
                        f"qpos_data shape {tuple(qpos_data.shape)}."
                    )
                repeats = qpos_data.shape[-1] // noise_std.shape[-1]
                noise_std = noise_std.repeat(repeats)
            noise = torch.randn_like(qpos_data) * proprio_noise_scale * noise_std
        else:
            # Fallback: use absolute noise scale
            noise = torch.randn_like(qpos_data) * proprio_noise_scale
        qpos_data = qpos_data + noise

    # Apply camera blur (before dropout, so dropped cameras don't waste compute)
    if camera_blur_sigmas is not None:
        for cam_idx, sigma in camera_blur_sigmas.items():
            if sigma > 0 and cam_idx < image_data.shape[1]:
                if image_data.dim() == 6:
                    for frame_idx in range(image_data.shape[2]):
                        image_data[:, cam_idx, frame_idx] = apply_gaussian_blur(
                            image_data[:, cam_idx, frame_idx],
                            sigma,
                        )
                else:
                    # image_data shape: (batch, num_cameras, C, H, W)
                    image_data[:, cam_idx] = apply_gaussian_blur(image_data[:, cam_idx], sigma)

    # Apply camera dropout (zero out specified cameras)
    if camera_dropout_indices is not None and len(camera_dropout_indices) > 0:
        for cam_idx in camera_dropout_indices:
            if cam_idx < image_data.shape[1]:
                image_data[:, cam_idx] = 0.0

    if image_data.ndim != 5:
        raise ValueError('Policies require one current image per camera (B, cameras, C, H, W)')

    return policy(qpos_data, image_data, action_data, is_pad)



def train_bc(train_dataloader, val_dataloader, config, stats=None):
    num_steps = config['num_steps']
    total_training_steps = num_steps + 1
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    validate_every = config['validate_every']
    save_every = config['save_every']
    # Continuation runs name checkpoints and log wandb steps by the total trained steps.
    step_offset = int(config.get('step_offset', 0))

    # Get curriculum configs from task_config
    proprio_noise_config = config.get('proprio_noise', None)
    camera_curriculum = config.get('camera_curriculum', None)
    camera_blur_config = config.get('camera_blur', None)
    camera_names = config.get('camera_names', [])

    # Get qpos_std from stats for noise scaling
    qpos_std = None
    if stats is not None and 'qpos_std' in stats and proprio_noise_config is not None:
        qpos_std = torch.from_numpy(stats['qpos_std']).float().cuda()
        print(f"Proprioception noise enabled: {proprio_noise_config}")
        print(f"Noise will be scaled by qpos_std (shape={qpos_std.shape})")

    # Parse camera curriculum (dropout) - map camera names to indices
    camera_dropout_configs = {}
    if camera_curriculum is not None:
        print(f"Camera dropout curriculum enabled: {camera_curriculum}")
        for cam_name, dropout_config in camera_curriculum.items():
            if cam_name in camera_names:
                cam_idx = camera_names.index(cam_name)
                camera_dropout_configs[cam_idx] = dropout_config
                print(f"  Dropout '{cam_name}' (idx={cam_idx}): {dropout_config}")

    # Parse camera blur config - map camera names to indices
    camera_blur_configs = {}
    if camera_blur_config is not None:
        print(f"Camera blur curriculum enabled: {camera_blur_config}")
        for cam_name, blur_config in camera_blur_config.items():
            if cam_name in camera_names:
                cam_idx = camera_names.index(cam_name)
                camera_blur_configs[cam_idx] = blur_config
                print(f"  Blur '{cam_name}' (idx={cam_idx}): {blur_config}")

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if policy_config.get('pretrained_backbone_weights') and not (config['load_pretrain'] or config['resume_ckpt_path']):
        policy.load_pretrained_backbone()
        print(f"Backbone initialized from torchvision {policy_config['backbone']} {policy_config['pretrained_backbone_weights']}")
    weights_only = bool(config.get('pretrained_weights_only'))
    if weights_only and policy_class != 'ACT':
        raise ValueError('pretrained_weights_only warm-starts ACT variants only')
    if config['load_pretrain'] and weights_only:
        # Warm start (e.g. ACT + RTC from a plain ACT checkpoint): every pretrained weight is used;
        # only the RTC prefix input the new variant adds may be missing (it starts at zero).
        state = torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda')
        loading_status = policy.model.load_state_dict(state, strict=False)
        new = [k for k in loading_status.missing_keys if not k.startswith('rtc_prefix_')]
        if new or loading_status.unexpected_keys:
            raise ValueError(f'Warm start does not match the pretrained checkpoint: missing {new}, '
                             f'unexpected {loading_status.unexpected_keys}')
        print(f'warm-started from {config["pretrained_path"]}; new parameters: {loading_status.missing_keys}')
    elif config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda'))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    if policy_class == 'Flow':
        # Training-only kernel layout (same parameters and math); before DDP, the optimizer and compile.
        policy._get_strict_flow_diffusion_module().unet.use_channels_last()
    else:
        policy.model.use_channels_last()  # ACT: NHWC ResNet backbones (same parameters and math)
    if config.get('compile'):
        import torch._inductor.config as inductor_config
        inductor_config.coordinate_descent_tuning = bool(config.get('compile_coordinate_descent'))
        mode = config.get('compile_mode', 'max-autotune')
        # The policies override __call__, so nn.Module.compile() would never take effect.
        # Compile the training loss path on this instance instead: parameters, state_dict and
        # inference code are untouched. Autotuning only selects among kernels.
        loss_path = '_compute_flow_matching_loss' if policy_class == 'Flow' else '_compute_loss'
        setattr(policy, loss_path, torch.compile(getattr(policy, loss_path), mode=mode))
        print(f'compile: mode={mode} coordinate_descent={inductor_config.coordinate_descent_tuning} '
              f'matmul_tf32={torch.backends.cuda.matmul.allow_tf32} cudnn_tf32={torch.backends.cudnn.allow_tf32}')
    model = policy
    if dist.is_initialized():
        # DDP broadcasts rank 0's weights; per-rank seeds give each rank its own
        # batches, flow noise and RTC delays. `policy` stays the unwrapped module.
        set_seed(seed + dist.get_rank())
        # Gradients are bf16, so DDP already all-reduces them in bf16.
        model = DDP(policy, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)
    optimizer = make_optimizer(policy_class, policy)
    if config['load_pretrain'] and not weights_only:
        optimizer.load_state_dict(torch.load(f'{config["pretrained_path"]}/optimizer_last.ckpt', map_location='cuda'))
    lr_warmup_steps = config.get('lr_warmup_steps', 1000)
    lr_scheduler = build_cosine_warmup_scheduler(
        optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=total_training_steps,
    )
        

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = cuda_prefetch(repeater(train_dataloader))
    log_every = max(1, int(config.get('wandb_log_every', 1)))
    log_sums, log_count = {}, 0


    for step in tqdm(range(num_steps+1), disable=not _is_main_process()):
        # validation (rank 0 only, on the unwrapped policy)
        if val_dataloader is not None and step % validate_every == 0 and _is_main_process():
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, 0)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step + step_offset, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)     
            if config['wandb']:       
                wandb.log(validation_summary, step=step + step_offset)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
                
        # training
        model.train()
        optimizer.zero_grad()
        data = next(train_dataloader)

        # Compute annealed proprioception noise scale from config
        current_noise_scale = 0.0
        if proprio_noise_config is not None:
            anneal_steps = proprio_noise_config.get('anneal_steps', -1)
            if anneal_steps < 0:
                anneal_steps = num_steps
            current_noise_scale = compute_annealed_value(
                step,
                proprio_noise_config.get('initial_scale', 0.0),
                proprio_noise_config.get('final_scale', 0.0),
                anneal_steps,
                proprio_noise_config.get('schedule', 'linear')
            )

        # Determine which cameras to dropout based on curriculum
        camera_dropout_indices = []
        for cam_idx, dropout_config in camera_dropout_configs.items():
            if should_dropout_camera(step, dropout_config, num_steps):
                camera_dropout_indices.append(cam_idx)

        # Compute blur sigmas for each camera
        camera_blur_sigmas = {}
        for cam_idx, blur_config in camera_blur_configs.items():
            sigma = compute_camera_blur_sigma(step, blur_config, num_steps)
            if sigma > 0:
                camera_blur_sigmas[cam_idx] = sigma

        forward_dict = forward_pass(data, model, 0,
                                    current_noise_scale, qpos_std, camera_dropout_indices,
                                    camera_blur_sigmas)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        if config['wandb']:
            # Sum metrics on the GPU and sync once per window instead of every step.
            # Summing copies values out of compiled CUDA-graph outputs before the next replay.
            for k, v in forward_dict.items():
                log_sums[k] = log_sums.get(k, 0) + v.detach().float()
            log_count += 1
            if (step + 1) % log_every == 0 or step == num_steps:
                keys = list(log_sums)
                values = torch.stack([log_sums[k] for k in keys]) / log_count
                log_sums, log_count = {}, 0
                if dist.is_initialized():
                    dist.all_reduce(values)
                    values /= dist.get_world_size()
                log_dict = dict(zip(keys, values.tolist()))
                log_dict['proprio_noise_scale'] = current_noise_scale
                log_dict['num_cameras_dropped'] = len(camera_dropout_indices)
                log_dict['lr'] = current_lr
                if camera_blur_sigmas:
                    log_dict['max_blur_sigma'] = max(camera_blur_sigmas.values())
                if _is_main_process():
                    wandb.log(log_dict, step=step + step_offset)

        if not _is_main_process():
            continue
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step + step_offset}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            #save optimizer state
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_step_{step + step_offset}_seed_{seed}.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            #save best ckpt
            # if best_ckpt_info is not None:
            #     ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
            #     torch.save(best_ckpt_info[2], ckpt_path)
        if step % 2000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            


    if not _is_main_process():
        return None
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
    torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    if best_ckpt_info is None:
        # No validation split: the final weights are the "best" checkpoint.
        best_ckpt_info = (num_steps + step_offset, float('nan'), deepcopy(policy.serialize()))
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def cuda_prefetch(batches):
    """Copy the next (pinned) batch to the GPU on a side stream while the current step computes."""
    stream = torch.cuda.Stream()

    def load():
        data = next(batches)
        with torch.cuda.stream(stream):
            return [t.cuda(non_blocking=True) for t in data]

    upcoming = load()
    while True:
        torch.cuda.current_stream().wait_stream(stream)
        current = upcoming
        for tensor in current:
            tensor.record_stream(torch.cuda.current_stream())
        upcoming = load()
        yield current


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

def main_train(args, *, task_boundary_provider=None):
    if not args.get('pretrained_backbone_weights'):
        raise ValueError('Flow training always starts from a pretrained backbone; set '
                         'model.pretrained_backbone_weights (e.g. IMAGENET1K_V1)')
    from config.training_config import normalize_training_rtc
    normalize_training_rtc(args)
    training_method = str(args.get('training_method', 'standard')).lower()
    if training_method != 'standard':
        raise ValueError('Only training.method: standard is supported')

    _load_standard_training_utils()
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    if policy_class not in ('Flow', 'ACT'):
        raise ValueError('Only Flow and ACT are supported')
    if args.get('finetune'):
        raise ValueError('Action-head finetuning is not supported; use load_pretrain')
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1:
        # Launched by torchrun: split the configured (global) batch evenly across GPUs.
        if args['inspect_data']:
            raise ValueError('--inspect_data is single-process; run without torchrun')
        if args['batch_size'] % world_size:
            raise ValueError(f"batch_size {args['batch_size']} must be divisible by {world_size} GPUs")
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(hours=1))
    batch_size_train = args['batch_size'] // world_size
    # Validation runs on rank 0 only, over the full batch.
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    backbone = args['backbone']
    same_backbones = args['same_backbones']
    num_dataloader_workers = args['num_dataloader_workers']
    legacy_model_type = args.get('model_type')
    if legacy_model_type not in (None, ""):
        print(
            f"[Deprecated] --model_type={legacy_model_type} is ignored in imitate_episodes_train.py. "
            "Use --policy_class to choose the policy."
        )

    from config.training_config import resolve_training_task, training_chunk_size
    yaml_task, task_config = resolve_training_task(args)
    task_name = args['task_name']
    # The training config owns the predicted chunk length; the checkpoint records it.
    args['chunk_size'] = training_chunk_size(args, task_config)
    if 'chunk_size' in task_config:  # a task-supplied value the recipe overrides
        task_config['chunk_size'] = args['chunk_size']
    if yaml_task:
        yaml_task = yaml_task.with_chunk_size(args['chunk_size'], source='training config')

    aloha = task_config.get('aloha', False)

    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    # episode_len is now optional - actual lengths are read from raw or legacy episodes
    # This value is only used for config logging, not for data loading
    episode_len = task_config.get('episode_len', None)
    camera_names = list(task_config['camera_names'])
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    # Default: train on every episode (no validation split); set data.train_ratio < 1 to hold some out.
    train_ratio = task_config.get('train_ratio', 1.0)
    name_filter = task_config.get('name_filter', lambda n: True)
    replay_episodes = task_config.get('replay_episodes', 'all')
    if replay_episodes != 'all':
        from policy.replay_data import replay_filter
        base_filter, select = name_filter, replay_filter(replay_episodes)
        name_filter = lambda path: base_filter(path) and select(path)
    from contracts.action_rate_contract import dataset_rate
    from policy import training_utils as training_utils_module
    timing_paths = [path for directory in ([dataset_dir] if isinstance(dataset_dir, str) else dataset_dir)
        for path in training_utils_module.find_episodes(directory, args['skip_mirrored_data']) if name_filter(path)]
    from contracts.robot_contract import dataset_robot_contract, require_robot_match
    robot_contract = dataset_robot_contract(timing_paths, required=False)
    if robot_contract is not None:
        expected_robot = yaml_task.robot_contract if yaml_task else task_config.get('robot_contract')
        if expected_robot is None:
            raise ValueError('Tagged training data requires an explicit task robot_contract')
        require_robot_match(expected_robot, {'robot_contract': robot_contract}, context='training data')
    recording_fps = dataset_rate(timing_paths, legacy_rate=args.get('legacy_dataset_fps'))
    from camera_utils.camera_settings_contract import dataset_camera_settings, merge_camera_settings
    camera_settings = dataset_camera_settings(timing_paths)
    if yaml_task:
        yaml_task.require_recording({'fps': recording_fps, 'camera_settings': camera_settings,
                                     'robot_contract': robot_contract})
        camera_settings = merge_camera_settings(camera_settings, yaml_task.as_dict().get('camera_settings'))
        stat_dirs = stats_dir if stats_dir is not None else dataset_dir
        for directory in ([stat_dirs] if isinstance(stat_dirs, str) else stat_dirs):
            paths = training_utils_module.find_episodes(directory, args['skip_mirrored_data'])
            stats_robot = dataset_robot_contract(paths, required=robot_contract is not None)
            if stats_robot != robot_contract:
                raise ValueError('Normalization data robot model/arm roles differ from training data')
            yaml_task.require_recording({'fps': dataset_rate(paths, legacy_rate=args.get('legacy_dataset_fps')),
                                         'robot_contract': stats_robot})
    randomize_index = task_config.get('randomize_index', False)
    from config.masking_config import masking_contract, require_masking_strategy
    masking_strategy = require_masking_strategy(task_config.get('masking_strategy'))
    if masking_strategy is not None:
        # Stored with the checkpoint: direct inference requires the same mask definition and calibration.
        task_config['masking_contract'] = masking_contract(masking_strategy)
    finetune = args.get('finetune', False) or task_config.get('finetune', False)
    outputs_name = task_config.get('outputs_name', ['action'])
    use_raw_future_actions = args.get('use_raw_future_actions', False) or task_config.get('use_raw_future_actions', False)
    if 'use_raw_future_actions' in args.get('_training_recipe', {}).get('training', {}):
        use_raw_future_actions = args['use_raw_future_actions']
    action_command_style = args.get('action_command_style') or 'absolute'
    include_masked_segments = bool(args.get('include_masked_segments', False))
    if use_raw_future_actions and outputs_name != ['action']:
        raise ValueError(
            f"use_raw_future_actions requires outputs_name=['action'], got {outputs_name} for task {task_name}"
        )
    
    standard_image_processing.require_single_frame(task_config, dict(args, task_config=task_config))
    camera_names_for_model = list(camera_names)

    from config.training_config import checkpoint_directory
    ckpt_dir = checkpoint_directory(args, legacy_suffix=f'_{task_name}_{policy_class}_{backbone}_{same_backbones}_{masking_strategy}')
    ckpt_dir_exists = os.path.isdir(ckpt_dir)
    if world_size > 1:
        dist.barrier()  # every rank checks before rank 0 creates it
    if ckpt_dir_exists and not args['inspect_data']:
        print(f'ckpt_dir {ckpt_dir} already exists, exiting')
        return
    args['ckpt_dir'] = ckpt_dir 
    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    print(f'===========================START===========================:')
    print(f"{task_name}")
    print(f'===========================Config===========================:')
    print(f'ckpt_dir: {ckpt_dir}')
    print(f'policy_class: {policy_class}')
    print(f'use_raw_future_actions: {use_raw_future_actions}')
    print(f'action_command_style: {action_command_style}')
    print(f'lr_warmup_steps: {args["lr_warmup_steps"]}')
    # fixed parameters
    state_dim = task_config.get('state_dim', 40)
    action_dim = task_config.get('action_dim', 40)
    state_mask = task_config.get('state_mask', [1]*state_dim)
    action_mask = task_config.get('action_mask', [1]*action_dim)
    if args['use_mask']:
        state_dim = sum(state_mask)
        action_dim = sum(action_mask)
        state_idx = np.where(state_mask)[0].tolist()
        action_idx = np.where(action_mask)[0].tolist()
    else:
        state_idx = np.arange(state_dim).tolist()
        action_idx = np.arange(action_dim).tolist()
    single_arm_side = resolve_single_arm_side_for_training(task_name, task_config)
    single_arm_can = resolve_single_arm_can_for_training(task_name, task_config, single_arm_side)
    if action_dim <= 7:
        print(f'single_arm_side: {single_arm_side or "legacy/unspecified"}')
        if single_arm_can is not None:
            print(f'single_arm_can: {single_arm_can}')
    backbone = args['backbone']
    if args['width'] < 0 or args['height'] < 0:
        raise ValueError(f"{policy_class} requires an explicit image size; provide positive --width and --height.")
    if policy_class == 'Flow':
        flow_hidden_dim = args['hidden_dim'] if args['hidden_dim'] is not None else 512
        flow_unet_depth = args['unet_depth']
        flow_training_time_rtc_config = validate_flow_training_time_rtc_args(args, args['chunk_size'])
        policy_config = {
                        'observation_horizon': args['observation_horizon'],
                        'action_horizon': args['chunk_size'],
                        'prediction_horizon': args['chunk_size'],
                        'num_queries': args['chunk_size'],
                        'num_inference_timesteps': args['num_inference_timesteps'],
                        'flow_loss_type': args.get('flow_loss_type', 'flow_matching_mse'),
                        'flow_t_sampling': args.get('flow_t_sampling', 'beta'),
                        **flow_training_time_rtc_config,
                        'ema_power': args['ema_power'],
                        'lr': args['lr'],
                        'action_dim': action_dim,
                        'state_dim': state_dim,
                        'state_idx': state_idx,
                        'action_idx': action_idx,
                        'height': args['height'],
                        'width': args['width'],
                        'hidden_dim': flow_hidden_dim,
                        'unet_depth': flow_unet_depth,
                        'crop_shape': None,  # Force-disable LeRobot default image crop (e.g., 84x84 in some versions).
                        'backbone': backbone,
                        # Torchvision weights initialize the backbone (BatchNorm kept); training never starts from scratch.
                        'pretrained_backbone_weights': args['pretrained_backbone_weights'],
                        # One image encoder per camera unless same_backbones; recorded explicitly so
                        # older shared-encoder checkpoints (no key) still rebuild as shared.
                        'use_separate_rgb_encoder_per_camera': not args['same_backbones'],
                        'camera_names': camera_names_for_model,
                        # Always bf16: weights, activations, gradients and checkpoints (fp32 master weights).
                        'precision': 'bf16',
                        'state_mask': state_mask,
                        'action_mask': action_mask,
                        }
    elif policy_class == 'ACT':
        act_rtc_config = validate_flow_training_time_rtc_args(args, args['chunk_size'])
        policy_config = {
                        'num_queries': args['chunk_size'],
                        **act_rtc_config,
                        'lr': args['lr'],
                        'lr_backbone': args['lr_backbone'],
                        'action_dim': action_dim,
                        'state_dim': state_dim,
                        'state_idx': state_idx,
                        'action_idx': action_idx,
                        'height': args['height'],
                        'width': args['width'],
                        'hidden_dim': args['hidden_dim'] if args['hidden_dim'] is not None else 512,
                        'dec_layers': args['dec_layers'],
                        'nheads': args['nheads'],
                        'dropout': args['dropout'],
                        'backbone': backbone,
                        'same_backbones': args['same_backbones'],
                        'pretrained_backbone_weights': args['pretrained_backbone_weights'],
                        'camera_names': camera_names_for_model,
                        'precision': 'bf16',
                        'state_mask': state_mask,
                        'action_mask': action_mask,
                        }
    else:
        raise NotImplementedError
    policy_config.update({
        'camera_names': list(camera_names_for_model),
        'base_camera_names': list(camera_names),
        'action_command_style': action_command_style,
        'single_arm_side': single_arm_side,
        'single_arm_can': single_arm_can,
    })
    print(f'====================FINISH INIT POLICY========================:')
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        # Export the resolved model dimension, not the unused --action_dim CLI default.
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'data_dir': os.path.expanduser(args.get('data_dir') or constants_module.DATA_DIR),
        'model_name': args.get('model_name'),
        'training_recipe': deepcopy(args.get('_training_recipe')),
        'training_time_rtc': policy_config.get('training_time_rtc', False),
        'seed': args['seed'],
        'temporal_agg': False,
        'camera_names': list(camera_names),  # Original camera names for data loading
        'camera_names_for_model': list(camera_names_for_model),
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'pretrained_weights_only': args['pretrained_weights_only'],
        'height':args['height'],
        'width':args['width'],
        'normalize_resnet': args['normalize_resnet'],
        'grayscale': args['grayscale'],
        'aloha': aloha,
        'wandb': args['wandb'],
        'wandb_log_every': args['wandb_log_every'],
        'world_size': world_size,
        'compile': args['compile'],
        'compile_mode': args['compile_mode'],
        'compile_coordinate_descent': args['compile_coordinate_descent'],
        'precision': 'bf16',
        'pretrained_path': args['pretrained_path'],
        'randomize_data_degree': args['randomize_data_degree'],
        'randomize_data': args['randomize_data'],
        'lr_warmup_steps': args['lr_warmup_steps'],
        'step_offset': args['step_offset'],
        'finetune': finetune,
        # Curriculum learning configs from task_config
        'proprio_noise': task_config.get('proprio_noise', None),
        'camera_curriculum': task_config.get('camera_curriculum', None),
        'camera_blur': task_config.get('camera_blur', None),
        'single_arm': task_config.get('single_arm', True),
        'single_arm_side': single_arm_side,
        'single_arm_can': single_arm_can,
        'use_raw_future_actions': use_raw_future_actions,
        'action_command_style': action_command_style,
    }
    args_for_export = {key: value for key, value in args.items() if not key.startswith('_')}
    # `model_type` is a legacy training arg that is not used by this trainer and
    # causes confusion with `policy_class` in exported config files.
    args_for_export.pop('model_type', None)
    action_fps = recording_fps
    if use_raw_future_actions:
        action_fps = dataset_rate(timing_paths, legacy_rate=args.get('raw_action_fps'), raw=True)
    if yaml_task:
        yaml_task.require_recording({'fps': action_fps, 'robot_contract': robot_contract})
    if robot_contract is not None:
        config['robot_contract'] = robot_contract
        policy_config['robot_contract'] = robot_contract
        task_config['robot_contract'] = robot_contract
        # A resume/pretrain must carry compatible identity before loading weights.
        sources = []
        if config.get('load_pretrain'):
            sources.append(config['pretrained_path'])
        if config.get('resume_ckpt_path'):
            sources.append(os.path.dirname(config['resume_ckpt_path']) or '.')
        for source in sources:
            with open(os.path.join(source, 'all_configs.json')) as stream:
                require_robot_match(robot_contract, json.load(stream), context='pretrained/resumed checkpoint')
    config['control_freq'] = action_fps
    config['camera_settings'] = camera_settings
    config['recording_rate_hz'] = recording_fps
    policy_config['control_freq'] = action_fps
    task_config_for_export = dict(task_config)
    if robot_contract is None:
        task_config_for_export.pop('robot_contract', None)
    task_config_for_export.pop('fps', None)
    task_config_for_export.pop('action_rate_hz', None)
    task_config_for_export['action_command_style'] = action_command_style
    task_config_for_export['single_arm_side'] = single_arm_side
    task_config_for_export['single_arm_can'] = single_arm_can
    all_configs = {**args_for_export, **config, "task_config": task_config_for_export}
    
    if args['width'] < 0:
        args['width'] = None
    if args['height'] < 0:
        args['height'] = None

    is_main = _is_main_process()
    if is_main:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        config_path = os.path.join(ckpt_dir, 'config.pkl')
        all_config_path = os.path.join(ckpt_dir, 'all_configs.json')
        expr_name = ckpt_dir.split('/')[-1]
        if not is_eval and args['wandb'] and not args['inspect_data']:
            wandb.init(project="flash-aloha", name=expr_name)
            wandb.config.update(config)
        elif args['inspect_data']:
            print("Wandb logging disabled during data inspection.")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        with open(all_config_path, 'w') as fp:
            print(all_configs)
            json.dump(all_configs, fp, indent=4)

    # Rank 0 builds any missing frame-store entries; the other ranks then reuse them.
    if world_size > 1 and not is_main:
        dist.barrier()
    train_dataloader, val_dataloader, stats, _, debug_image = load_data(num_dataloader_workers, masking_strategy, dataset_dir, name_filter, camera_names, 
                                                           batch_size_train, batch_size_val, args['chunk_size'], 
                                                           args['skip_mirrored_data'], config['load_pretrain'], 
                                                           policy_class, stats_dir_l=stats_dir, 
                                                           sample_weights=sample_weights, 
                                                           train_ratio=train_ratio,
                                                           width=args['width'],
                                                           height=args['height'],
                                                           normalize_resnet=args['normalize_resnet'],
                                                           data_aug=args['data_aug'],
                                                           observation_name=task_config['observation_name'],
                                                           outputs_name=outputs_name,
                                                           feature_loss = args['feature_loss_weight'] > 0,
                                                           grayscale = args['grayscale'],
                                                           randomize_color = args['randomize_color'],
                                                            randomize_data_degree = args['randomize_data_degree'],
                                                            randomize_data = args['randomize_data'],
                                                            history_len=args['history_len'],
                                                            history_noise= args['history_noise'],
                                                            noise_type = args.get('noise_type', 'gaussian'),
                                                            aloha=aloha,
                                                            task_boundary_provider=task_boundary_provider,
                                                            use_raw_future_actions=use_raw_future_actions,
                                                            action_command_style=action_command_style,
                                                            include_masked_segments=include_masked_segments,
                                                            frame_store_dir=(args['frame_store_dir'] or str(default_frame_store_dir())) if args['frame_store'] else None,
                                                           )
    if world_size > 1 and is_main:
        dist.barrier()
    
    if args['inspect_data']:
        print('Inspecting data...')
        if not _interactive_backend:
            print('No interactive backend available. Images will be saved to files.')

        # Display debug image first if available
        if debug_image is not None:
            # Convert to numpy if it's a tensor
            if not isinstance(debug_image, np.ndarray):
                debug_image_np = debug_image.cpu().numpy()
            else:
                debug_image_np = debug_image.copy()

            # Handle multiple cameras
            if len(debug_image_np.shape) == 4:  # (num_cameras, channels, height, width)
                num_cameras = debug_image_np.shape[0]
                fig, axes = plt.subplots(1, num_cameras, figsize=(5*num_cameras, 5))
                if num_cameras == 1:
                    axes = [axes]

                for cam_idx in range(num_cameras):
                    img = debug_image_np[cam_idx]
                    if img.shape[0] == 3:  # CHW format
                        img = img.transpose(1, 2, 0)  # Convert to HWC
                    elif img.shape[0] == 1:  # Grayscale CHW
                        img = img.squeeze(0)  # Remove channel dimension

                    if len(img.shape) == 3:  # Color image
                        axes[cam_idx].imshow(img)
                    else:  # Grayscale image
                        axes[cam_idx].imshow(img, cmap='gray')

                    axes[cam_idx].set_title(f'Camera {cam_idx+1}')
                    axes[cam_idx].axis('off')

                plt.suptitle('Debug Images from Data Loader')
                plt.tight_layout()
                if _interactive_backend:
                    plt.show()
                else:
                    plt.savefig('inspect_debug_images.png', dpi=150, bbox_inches='tight')
                    print('Saved debug images to inspect_debug_images.png')
                    plt.close()
            else:
                # Single image
                plt.figure(figsize=(8, 6))
                if len(debug_image_np.shape) == 3:
                    if debug_image_np.shape[0] == 3:  # CHW format
                        debug_image_np = debug_image_np.transpose(1, 2, 0)  # Convert to HWC
                    plt.imshow(debug_image_np)
                elif len(debug_image_np.shape) == 2:
                    plt.imshow(debug_image_np, cmap='gray')

                plt.title('Debug Image from Data Loader')
                plt.axis('off')
                if _interactive_backend:
                    plt.show()
                else:
                    plt.savefig('inspect_debug_image.png', dpi=150, bbox_inches='tight')
                    print('Saved debug image to inspect_debug_image.png')
                    plt.close()

            if _interactive_backend:
                print("Press Enter to continue to batch inspection...")
                input()
        
        print('Creating separate data iterator for inspection...')
        
        # Create a separate iterator for inspection to avoid consuming the main dataloader
        inspection_dataloader = iter(train_dataloader)
        
        # Main inspection loop
        batch_idx = 0
        while True:
            try:
                print(f"\nLoading batch {batch_idx}...")
                data = next(inspection_dataloader)
                result = inspect_training_data_interactive(data, batch_idx, camera_names)
                
                if result == 'quit':
                    break
                elif result == 'next_batch':
                    batch_idx += 1
                    continue
                    
                if batch_idx > 10:  # Limit to prevent infinite inspection
                    print("\nReached batch limit (10). Use 'q' to quit or '2' to continue.")
                    user_choice = input("Continue inspection? (y/n): ").strip().lower()
                    if user_choice != 'y':
                        break
                        
                batch_idx += 1
                        
            except StopIteration:
                print("Reached end of dataset.")
                break
            except KeyboardInterrupt:
                print("\nInspection interrupted by user.")
                break
                
        print('Data inspection done.')
        return

    # save dataset stats
    if is_main:
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, stats)
    if not is_main:
        return
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    if args['wandb']:
        wandb.finish()


from config.training_args import build_parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run:
        from config.training_config import training_summary
        print(json.dumps(training_summary(vars(args)), indent=2))
        return
    try:
        main_train(vars(args))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
