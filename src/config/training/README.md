# Training recipes

ACT is the default policy. `single_arm_act.yaml` (2 cameras, 1 arm; the
`create_config.py training` template) and `bimanual_act.yaml` (3 cameras, 2 arms)
select decoder-only ACT (`model.type: ACT`, `src/policy/act.py`): per-camera
ImageNet ResNet18 backbones with frozen BatchNorm (`same_backbones: true` shares one),
a 1x1 projection, and a BERT-style transformer over the proprio token, every camera's
feature tokens and one query per action, with a learned positional embedding per
position and an L1 loss over non-padded actions. The recipes use `hidden_dim` 512,
`dec_layers` 7, `nheads` 8, `dropout` 0.1, lr 1.2e-4 with `lr_backbone` 1e-5, weight
decay 1e-4, batch 128, 100-action chunks and 100k steps on one GPU, with the frame
store and torch.compile on. ACT is bf16 only. Its attention is unmasked, so it runs on
PyTorch's FlashAttention-2 kernel.

Training-time RTC is opt-in (`training.training_time_rtc: true` with
`training_time_rtc_max_delay`): the first d ~ U{min..max} action queries carry
the committed prefix (a linear projection of the clean normalized action plus a
learned prefix flag) and are excluded from the L1 loss. Direct inference then pins
the previous plan's next actions as the prefix of each new chunk
(`forward_inf_training_time_rtc`). A plain ACT checkpoint can warm-start an RTC run
with `load_pretrain` and `pretrained_weights_only: true`: every weight is reused and
only the RTC prefix input is new.

`single_arm_flow.yaml` and `bimanual_flow.yaml` select Flow (`model.type: Flow`, `src/policy/policy.py`): a
depth-3 U-Net with a separate ImageNet-pretrained ResNet18 backbone per camera, batch
128 on one GPU for 100k steps, 128-action chunks.

Validate offline with `python scripts/config/check_training_config.py --training-config NAME`.
`model_name` controls the output directory. `training.training_time_rtc` is explicit.
`data.train_ratio` defaults to 1.0: every episode is used for training, validation is
skipped, and `policy_best.ckpt` holds the final weights. Set a value below 1 to hold
out episodes for validation loss and best-checkpoint selection.
`data.replay_episodes` (`all`, `only`, `exclude`) selects recorded replays
(metadata `replay_recording`) or hand-guided takes.
`data.masking_strategy` trains with an image and/or proprioceptive mask defined in the
masking config (for example `end_effector`); the checkpoint records it and direct
inference applies the same mask.
See [masking](../../../docs/image_preprocessing.md#masking).

Training runs on one GPU. `wandb_log_every` averages metrics over N steps. With `frame_store: true`, loader
workers keep every episode open (`FLASH_ALOHA_EPISODE_CACHE_SIZE` defaults to the
episode count); otherwise set it to at least the episode count yourself.

Training is always bf16; there is no higher-precision option. Weights, activations,
gradients and saved policy weights are bf16; AdamW keeps fp32 master weights and
moments (plain bf16 updates would round away). The checkpoint records
`policy_config.precision: bf16`, and inference runs it in bf16, exactly the numerics
it trained with. For Flow, noise times and their sine/cosine embedding, the flow-matching loss, and
the sampler's integration stay fp32. `optimizer_*.ckpt` carries the fp32 master
weights, so `load_pretrain` resumes exactly. During training the U-Net runs its
Conv1d layers as channels-last 2D convolutions (same parameters and math, no
per-call cuDNN layout transposes); checkpoints keep the standard layout.

`model.pretrained_backbone_weights` (default `IMAGENET1K_V1`) initializes the vision
backbone from torchvision at training start. It is required: training never
starts the backbone from scratch. A pretrained
backbone keeps its BatchNorm layers (GroupNorm would discard the pretrained
statistics), which stay fp32 in bf16 policies, and requires the default ImageNet
image normalization. The weights download once on the training host; inference
builds the same architecture without downloading and loads everything from the
checkpoint. `load_pretrain`/`resume_ckpt_path` skip the torchvision initialization.

Multi-GPU: launch the same command under `torchrun --standalone --nproc_per_node N`.
`training.batch_size` stays the global batch and is split evenly across GPUs; rank 0
writes checkpoints and wandb. On GPUs without P2P, 2-GPU DDP at batch 128 was no
faster than one GPU, so separate runs on `--gpu_id 0` and `--gpu_id 1` are usually the
better use of two cards.

`model.chunk_size` (required) is the number of actions each predicted chunk
contains; the task YAML does not set it. For Flow it must be a multiple of
2**`model.unet_depth` (8 for depth 3): an incompatible pair fails instead of the
U-Net being silently made shallower. The checkpoint records the chunk size, and
inference uses it: execution windows are checked against it and a predicted
chunk of any other length is rejected.

Flow objective and training-time RTC:
noise is at t=1 and data at t=0, every action has its own noise time, the RTC
prefix (the first 0..`training_time_rtc_max_delay` actions, inclusive) is given
time 0 (explicitly clean) and excluded from the velocity-MSE loss, and each
example is weighted equally. Sampling integrates from t=1 to 0 with any RTC prefix
pinned at time 0. Weights carry a marker for this objective.

Flow takes up to three cameras and two arms (14-dim state/action). Each camera gets its
own image encoder by default (`model.same_backbones: false`, recorded in the checkpoint as
`policy_config.use_separate_rgb_encoder_per_camera`); `same_backbones: true` shares one
encoder across cameras. One encoder per camera costs about 2% step time at 3 cameras
(same FLOPs, more parameters).
