"""Training CLI definitions, shared with YAML validation and offline inspection."""
import argparse
from config.training_config import TrainingArgumentParser


def build_parser():
    parser = TrainingArgumentParser()
    parser.add_argument('--inspect_data', action='store_true', default=False, help='inspect data')
    parser.add_argument('--config', action='store', type=str, help='config file', required=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', default="ckpts/", action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', default="ACT", choices=["ACT", "Flow"], action='store', type=str,
                        help='Decoder-only ACT (default) or Flow')
    parser.add_argument('--task_name', default=None, action='store', type=str, help='task_name')
    parser.add_argument('--data_dir', default=None, action='store', type=str,
                        help='Override the training dataset root for this run.')
    
    parser.add_argument('--batch_size', default=32, action='store', type=int, help='batch_size')
    parser.add_argument('--seed', default=0, action='store', type=int, help='seed')
    parser.add_argument('--num_steps', default=100000, action='store', type=int, help='num_steps')
    parser.add_argument('--lr', default=1e-5, action='store', type=float, help='lr')
    parser.add_argument('--lr_warmup_steps', default=1000, action='store', type=int,
                        help='Linear warmup steps before cosine decay.')
    parser.add_argument('--load_pretrain', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pretrained_path', action='store', type=str, help='pretrained_path', required=False)
    parser.add_argument('--pretrained_weights_only', action=argparse.BooleanOptionalAction, default=False,
                        help='With load_pretrain: warm-start a new run from the weights only (fresh optimizer). '
                             'Only parameters the new variant adds (the RTC prefix input) may be missing.')
    parser.add_argument('--step_offset', action='store', type=int, default=0,
                        help='Steps already trained (continuing from load_pretrain): offsets checkpoint names and '
                             'wandb steps only; the LR warmup/cosine schedule spans this run\'s num_steps.')
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=False, help='finetune only the action head')

    parser.add_argument('--eval_every', action='store', type=int, default=100000, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=1000, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--legacy-dataset-fps', type=float, default=None, help='Explicit FPS for legacy episodes without rate metadata; existing declarations must agree.')
    parser.add_argument('--raw-action-fps', type=float, default=None, help='Explicit rate of a legacy action_raw stream without raw_action_rate_hz metadata.')
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int, default=0, help='history_len for state observations (qpos, qvel)')
    parser.add_argument('--history_noise', action='store', type=float, default=0.0, help='history_noise')
    parser.add_argument('--noise_type', action='store', type=str, default='gaussian', help='noise_type')
    parser.add_argument('--use_raw_future_actions', action='store_true', default=False,
                        help='Train with chunk_size targets sampled from /action_raw strictly after each /action_timestamps anchor')
    parser.add_argument('--action_command_style', action='store', type=str, default=None,
                        choices=['absolute'],
                        help='Only absolute action targets are supported.')
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    parser.add_argument('--implicit_loss', action='store_true', default=False, help='use implicit loss')

    parser.add_argument('--num_dataloader_workers', action='store', type=int, default=8, help='num_dataloader_workers')

    # Flow sampling and backbone
    parser.add_argument('--observation_horizon', action='store', type=int, default=1, help='observation_horizon')
    parser.add_argument('--num_inference_timesteps', action='store', type=int, default=16, help='num_inference_timesteps')
    parser.add_argument('--unet_depth', action='store', type=int, default=3,
                        help='Flow U-Net depth x. Default down_dims are d*2^i for i in [0, x-1].')
    parser.add_argument('--flow_loss_type', action='store', type=str, default='flow_matching_mse',
                        help='Flow policy loss objective identifier (used when --policy_class Flow).')
    parser.add_argument('--flow_t_sampling', action='store', type=str, default='beta',
                        choices=['beta', 'uniform'],
                        help='Flow policy t-sampling distribution for flow matching.')
    parser.add_argument('--training-time-rtc', action=argparse.BooleanOptionalAction, default=None,
                        help='Explicitly select training-time RTC; saved in checkpoint metadata.')
    parser.add_argument('--no_training_time_rtc', action='store_true', default=False,
                        help='Disable training-time RTC action-prefix conditioning.')
    parser.add_argument('--training_time_rtc_min_delay', action='store', type=int, default=0,
                        help='Minimum training-time RTC prefix delay in action steps.')
    parser.add_argument('--training_time_rtc_max_delay', action='store', type=int, default=10,
                        help='Maximum training-time RTC prefix delay in action steps.')
    parser.add_argument('--ema_power', action='store', type=float, default=0.75, help='ema_power')
    # This should be removed or be default 14
    parser.add_argument('--action_dim', action='store', type=int, default=14, help='action_dim')

    # Model dimensions
    parser.add_argument('--task-config', dest='task_config', help='Explicit task YAML; otherwise selected by task_name.')
    parser.add_argument('--chunk_size', action='store', type=int, help='Predicted actions per chunk; set model.chunk_size in the training config.', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    #dec_layers
    
    #feature_loss_weight
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.0)
    #self_attention
    
    #for backbone 
    parser.add_argument('--backbone', type=str, default='resnet18')
    # Decoder-only ACT
    parser.add_argument('--dec_layers', type=int, default=7, help='ACT: transformer blocks.')
    parser.add_argument('--nheads', type=int, default=8, help='ACT: attention heads.')
    parser.add_argument('--dropout', type=float, default=0.1, help='ACT: transformer dropout.')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='ACT: backbone learning rate.')
    parser.add_argument('--pretrained_backbone_weights', type=str, default='IMAGENET1K_V1',
                        help='Torchvision weights that initialize the backbone (required: Flow never trains it from '
                             'scratch); it keeps BatchNorm. Downloaded once at training start, never at inference.')
    parser.add_argument('--same_backbones', action='store_true',
                        help='Share one image backbone across cameras (default: one per camera, Flow and ACT).')
    #use mask
    parser.add_argument('--use_mask', action='store_true', default=False)
    
    # for image 
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--data_aug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--normalize_resnet', action='store_true') ### not used - always normalize - in the model.forward
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--randomize_color', action='store_true')
    parser.add_argument('--randomize_data', action='store_true')
    parser.add_argument('--randomize_data_degree', action='store', type=int, default=3)
    
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_log_every', action='store', type=int, default=50,
                        help='Log training metrics to wandb every N steps, averaged over those N steps '
                             '(avoids a GPU sync per step).')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                        help='Compile the policy in place with torch.compile. Numeric precision settings are '
                             'not changed and the checkpoint format is unchanged.')
    parser.add_argument('--compile_mode', default='max-autotune',
                        choices=['default', 'reduce-overhead', 'max-autotune-no-cudagraphs', 'max-autotune'],
                        help='torch.compile mode used with --compile.')
    parser.add_argument('--compile_coordinate_descent', action=argparse.BooleanOptionalAction, default=False,
                        help='Additionally coordinate-descent tune generated Triton kernels (longer compile).')
    parser.add_argument('--frame_store', action=argparse.BooleanOptionalAction, default=False,
                        help='Decode each episode once into a memory-mapped store of preprocessed frames '
                             '(bit-identical to live decoding) instead of seeking video per sample.')
    parser.add_argument('--frame_store_dir', default=None,
                        help='Frame store location (default $FLASH_ALOHA_FRAME_STORE_DIR or ~/.cache/flash-aloha/frame-store).')
    
    parser.add_argument('--model_type', type=str, default=None,
                        help='[Deprecated] Ignored in this trainer. Use --policy_class instead.')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--training-config', help='Training recipe YAML or catalog name.')
    parser.add_argument('--model-name', help='Exact output directory name beneath --ckpt_dir.')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved training settings without loading data or models.')
    return parser


