"""Decoder-only ACT and multi-camera/bimanual shapes for every policy. Offline, no robot."""
import io

import pytest
import torch

from policy.act import ACTPolicy, FrozenBatchNorm2d
from policy.mixed_precision import MasterWeightAdamW
from policy.model_util import make_optimizer, make_policy
from test_flow_per_action_rtc import small_policy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
SHAPES = [(1, 7), (2, 7), (3, 14)]  # (cameras, state/action dim): up to three cameras and two arms
CAMERAS = ['video_overhead', 'video_left', 'video_right']


def act_config(cameras=1, dim=7, **override):
    config = dict(lr=1e-4, camera_names=CAMERAS[:cameras], height=64, width=64, state_dim=dim, action_dim=dim,
                  num_queries=10, hidden_dim=32, nheads=4, dec_layers=2, backbone='resnet18',
                  pretrained_backbone_weights='IMAGENET1K_V1', training_time_rtc=False)
    config.update(override)
    return config


def inputs(cameras, dim, batch=3, horizon=10, size=64):
    return (torch.randn(batch, dim), torch.rand(batch, cameras, 3, size, size), torch.randn(batch, horizon, dim),
            torch.zeros(batch, horizon, dtype=torch.bool))


@pytest.mark.parametrize('cameras,dim', SHAPES)
def test_act_token_layout_losses_and_inference(cameras, dim):
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(cameras, dim))
    model = policy.model
    assert len(model.backbones) == cameras  # one backbone per camera unless same_backbones
    assert model.image_tokens == cameras * 2 * 2  # 64x64 -> 2x2 layer4 features per camera
    assert model.decoder.weight_pos_embed.num_embeddings == 1 + model.image_tokens + 10
    qpos, image, actions, is_pad = inputs(cameras, dim)
    out = policy(qpos, image, actions, is_pad)
    assert set(out) == {'loss', 'l1'} and out['loss'].dtype == torch.float32
    out['loss'].backward()
    assert all(p.grad is not None for p in policy.model.parameters() if p.requires_grad)
    policy.eval()
    prediction = policy(qpos, image)
    assert prediction.shape == (3, 10, dim) and prediction.dtype == torch.float32
    torch.testing.assert_close(policy.forward_inf(qpos, image), prediction)


def test_act_l1_ignores_padding_and_shares_one_backbone_when_asked():
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(2, same_backbones=True))
    assert len(policy.model.backbones) == 1
    qpos, image, actions, is_pad = inputs(2, 7)
    base = policy(qpos, image, actions, is_pad)['loss']
    is_pad[:, 5:] = True
    changed = actions.clone()
    changed[:, 5:] += 100.0  # padded targets never contribute
    policy.eval()  # no dropout, so the two evaluations are comparable
    with torch.no_grad():
        a = policy(qpos, image, actions, is_pad)['loss']
        b = policy(qpos, image, changed, is_pad)['loss']
    torch.testing.assert_close(a, b)
    assert torch.isfinite(base)


def test_act_bf16_keeps_frozen_statistics_fp32_and_constructs_without_downloading(monkeypatch):
    import torchvision
    requested, real = [], torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, 'resnet18',
                        lambda weights=None, **kw: requested.append(weights) or real(weights=None, **kw))
    policy = make_policy('ACT', act_config(3, 14, precision='bf16'))
    assert requested and all(w is None for w in requested)
    norms = [m for m in policy.model.modules() if isinstance(m, FrozenBatchNorm2d)]
    assert norms and all(b.dtype == torch.float32 for m in norms for b in m.buffers())
    assert policy.model.input_proj.weight.dtype == torch.bfloat16
    assert all(t.dtype in (torch.bfloat16, torch.float32) for t in policy.serialize().values())
    qpos, image, actions, is_pad = inputs(3, 14)
    optimizer = make_optimizer('ACT', policy)
    assert isinstance(optimizer, MasterWeightAdamW)
    policy(qpos, image, actions, is_pad)['loss'].backward()
    optimizer.step()
    assert policy.eval()(qpos, image).dtype == torch.float32


def test_act_pretrained_backbone_loads_exactly_into_every_camera(monkeypatch):
    import torchvision
    torch.manual_seed(5)
    reference = torchvision.models.resnet18(weights=None)
    policy = ACTPolicy(act_config(2))
    monkeypatch.setattr(torchvision.models, 'resnet18', lambda weights=None, **kw: reference)
    policy.load_pretrained_backbone()
    expected = {k: v for k, v in reference.state_dict().items() if not k.startswith('fc.')}
    for backbone in policy.model.backbones:
        loaded = backbone.body.state_dict()
        assert loaded.keys() == {k for k in expected if not k.endswith('num_batches_tracked')}
        assert all(torch.equal(loaded[k], expected[k].to(loaded[k].dtype)) for k in loaded)  # bf16 convs


def test_act_backbone_learning_rate_survives_master_weight_buckets():
    from transformers import get_cosine_schedule_with_warmup
    policy = make_policy('ACT', act_config(2, precision='bf16', lr=1e-4, lr_backbone=1e-5))
    optimizer = policy.configure_optimizers()
    backbone_ids = {id(p) for n, p in policy.model.named_parameters() if 'backbone' in n}
    for group, bucket in zip(optimizer.param_groups, optimizer.model_param_groups):
        is_backbone = {id(p) in backbone_ids for p in bucket}
        assert len(is_backbone) == 1  # buckets never mix the two groups
        assert group['lr'] == (1e-5 if is_backbone.pop() else 1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    scheduler.step()
    assert sorted({g['lr'] for g in optimizer.param_groups}) == pytest.approx([1e-6, 1e-5])


def test_act_round_trips_through_a_saved_checkpoint():
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(2, precision='bf16')).eval()
    buffer = io.BytesIO()
    torch.save(policy.serialize(), buffer)
    buffer.seek(0)
    restored = make_policy('ACT', act_config(2, precision='bf16')).eval()
    restored.deserialize(torch.load(buffer))
    qpos, image, _, _ = inputs(2, 7)
    torch.testing.assert_close(restored(qpos, image), policy(qpos, image))


def test_act_is_bf16_only():
    policy = make_policy('ACT', act_config())
    assert policy.precision == 'bf16' and policy.model.action_head.weight.dtype == torch.bfloat16
    with pytest.raises(ValueError, match='bf16 only'):
        make_policy('ACT', act_config(precision='fp32'))


def test_act_is_a_supported_checkpoint_with_or_without_rtc():
    from config.supported_policy import require_supported_policy
    policy = make_policy('ACT', act_config())
    assert not policy.training_time_rtc
    require_supported_policy({'policy_class': 'ACT', 'policy_config': {'training_time_rtc': False}})
    require_supported_policy({'policy_class': 'ACT', 'training_time_rtc': True})
    with pytest.raises(ValueError, match='delays'):
        make_policy('ACT', act_config(training_time_rtc=True, training_time_rtc_max_delay=10))
    assert make_policy('ACT', act_config(training_time_rtc=True, training_time_rtc_max_delay=3)).training_time_rtc
    for other in ('Diffusion', 'CNNMLP', 'TransformerPolicy'):
        with pytest.raises(ValueError, match='Flow and ACT'):
            require_supported_policy({'policy_class': other})


def test_act_rtc_prefix_is_pinned_conditions_the_suffix_and_leaves_the_loss():
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(2, 7, training_time_rtc=True, training_time_rtc_max_delay=3)).eval()
    qpos, image, actions, _ = inputs(2, 7, batch=1)
    prefix = torch.randn(1, 3, 7)
    chunk = policy.forward_inf_training_time_rtc(qpos, image, prefix)
    assert chunk.shape == (1, 10, 7) and torch.equal(chunk[:, :3], prefix)
    other = policy.forward_inf_training_time_rtc(qpos, image, prefix + 1)
    assert not torch.allclose(chunk[:, 3:], other[:, 3:])  # the committed prefix conditions the rest
    # Delay 0 is the plain chunk; prefixes beyond the trained maximum are refused.
    assert torch.equal(policy.forward_inf_training_time_rtc(qpos, image, prefix[:, :0]), policy.forward_inf(qpos, image))
    with pytest.raises(ValueError, match='maximum'):
        policy.forward_inf_training_time_rtc(qpos, image, torch.randn(1, 4, 7))
    # With every non-prefix action padded, only prefix positions remain and they never count.
    policy.train()
    policy.training_time_rtc_min_delay = policy.training_time_rtc_max_delay = 3
    is_pad = torch.ones(1, 10, dtype=torch.bool)
    is_pad[:, :3] = False
    assert policy(qpos, image, actions, is_pad)['loss'].item() == 0.0


def test_fresh_start_refuses_a_first_target_far_from_the_arm():
    import numpy as np
    from inference.inference_direct import FRESH_START_MAX_TARGET_GAP, check_fresh_start_target
    qpos = np.array([-0.3097, 0.5264, 0.5985, -0.6636, 0.0297, -0.0481, 0.7246])
    fault = np.array([[-0.2609, 1.18, 0.7673, -0.2103, -0.1194, -0.0508, 0.867]])
    with pytest.raises(RuntimeError, match='0.654 rad .* action index 1'):
        check_fresh_start_target(fault, qpos)
    near = qpos.copy()
    near[:6] += 0.09  # a normal start (plain ACT's lookahead reaches 0.09 rad)
    near[6] += 1.0  # the gripper is not checked
    check_fresh_start_target(near[None], qpos)
    both = np.concatenate([qpos, qpos])
    far = both.copy()
    far[9] += FRESH_START_MAX_TARGET_GAP + 0.01  # second arm's joint 3
    with pytest.raises(RuntimeError, match='action index 9'):
        check_fresh_start_target(far[None], both, is_double_arm=True)


def test_direct_rtc_worker_returns_the_prefixed_chunk():
    import time
    from inference.inference_direct import DirectRTCQueries
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(2, 7, training_time_rtc=True, training_time_rtc_max_delay=3)).eval()
    qpos, image, _, _ = inputs(2, 7, batch=1)
    prefix = torch.randn(1, 3, 7)
    rtc = DirectRTCQueries(policy, 3, execution_dt=1.0, first_chunk_size=6, query_frequency=6)
    try:
        rtc.submit(qpos, image, prefix, 3)
        while rtc.worker.has_work_pending() and rtc.result is None:
            rtc.poll(3)
            time.sleep(0.001)
        chunk, length = rtc.take(6)
    finally:
        rtc.stop()
    assert length == 3
    assert (chunk == policy.forward_inf_training_time_rtc(qpos, image, prefix).numpy()).all()


def test_replay_filter_selects_episodes_by_recording_metadata(monkeypatch):
    from policy import replay_data
    monkeypatch.setattr(replay_data, 'is_replay_episode', lambda path: 'replay' in path)
    paths = ['take_0', 'replay_0']
    assert [p for p in paths if replay_data.replay_filter('all')(p)] == paths
    assert [p for p in paths if replay_data.replay_filter('only')(p)] == ['replay_0']
    assert [p for p in paths if replay_data.replay_filter('exclude')(p)] == ['take_0']
    with pytest.raises(ValueError, match='replay_episodes'):
        replay_data.replay_filter('some')


@pytest.mark.parametrize('cameras,dim', SHAPES)
def test_flow_handles_up_to_three_cameras_and_two_arms(cameras, dim):
    torch.manual_seed(0)
    policy = small_policy(camera_names=CAMERAS[:cameras], state_dim=dim, action_dim=dim, precision='bf16')
    policy.model.diffusion.unet.use_channels_last()
    qpos, image = torch.randn(2, dim), torch.rand(2, cameras, 3, 24, 32)
    out = policy(qpos, image, torch.randn(2, 16, dim), torch.zeros(2, 16, dtype=torch.bool))
    out['loss'].backward()
    assert policy.eval()(qpos, image).shape == (2, 16, dim)


def test_act_recipes_parse_with_their_hyperparameters():
    from config.training_args import build_parser
    for name, dim, cams in (('single_arm_act', 7, 2), ('bimanual_act', 14, 3)):
        args = build_parser().parse_args(['--training-config', name])
        assert args.policy_class == 'ACT' and args.no_training_time_rtc
        assert (args.hidden_dim, args.dec_layers, args.nheads, args.dropout) == (512, 7, 8, 0.1)
        assert (args.lr, args.lr_backbone, args.batch_size, args.chunk_size) == (1.2e-4, 1e-5, 128, 100)
        assert args.pretrained_backbone_weights == 'IMAGENET1K_V1'
    rtc = build_parser().parse_args(['--training-config', 'single_arm_act', '--training-time-rtc',
                                     '--training_time_rtc_max_delay', '5'])
    assert rtc.policy_class == 'ACT' and not rtc.no_training_time_rtc and rtc.training_time_rtc_max_delay == 5


@cuda
@pytest.mark.parametrize('cameras,dim', [(2, 7), (3, 14)])
def test_act_cuda_graph_matches_eager_and_attention_runs_on_flash(cameras, dim):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from policy.model_util import accelerate_policy
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(cameras, dim, precision='bf16')).cuda()
    qpos, image, actions, is_pad = (t.cuda() for t in inputs(cameras, dim))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # errors if attention cannot use FlashAttention
        policy(qpos, image, actions, is_pad)['loss'].backward()
    policy.eval()
    expected = policy.forward_inf(qpos[:1], image[:1])
    accelerate_policy(policy)
    assert policy.fast_inference
    # Channels-last backbones pick other cuDNN kernels: equal up to bf16 rounding.
    torch.testing.assert_close(policy.forward_inf(qpos[:1], image[:1]), expected, atol=0.03, rtol=0.02)
    assert policy.forward_inf(qpos, image).shape == (3, 10, dim)  # other batch sizes run eagerly


@cuda
def test_act_cuda_graph_serves_every_rtc_prefix_length():
    from policy.model_util import accelerate_policy
    torch.manual_seed(0)
    policy = make_policy('ACT', act_config(2, 7, training_time_rtc=True, training_time_rtc_max_delay=3)).cuda()
    torch.nn.init.normal_(policy.model.rtc_prefix_proj.weight)
    policy.eval()
    qpos, image, _, _ = (t.cuda() for t in inputs(2, 7, batch=1))
    cases = [None] + [torch.randn(1, d, 7).cuda() for d in (1, 3)]
    run = lambda prefix: policy.forward_inf_training_time_rtc(qpos, image, prefix)
    expected = [run(prefix) for prefix in cases]
    accelerate_policy(policy)
    assert policy.fast_inference
    for prefix, want in zip(cases, expected):  # one graph serves every prefix length
        got = run(prefix)
        torch.testing.assert_close(got, want, atol=0.03, rtol=0.02)  # bf16 rounding (channels-last kernels)
        if prefix is not None:
            assert torch.equal(got[:, :prefix.shape[1]], prefix)  # the prefix stays pinned exactly


def test_flow_can_use_a_separate_encoder_per_camera(monkeypatch):
    import torchvision
    torch.manual_seed(0)
    policy = small_policy(camera_names=CAMERAS, state_dim=14, action_dim=14, precision='bf16', height=64, width=64,
                          pretrained_backbone_weights='IMAGENET1K_V1', use_separate_rgb_encoder_per_camera=True)
    encoders = policy.model.diffusion.rgb_encoder
    assert isinstance(encoders, torch.nn.ModuleList) and len(encoders) == 3
    reference = torchvision.models.resnet18(weights=None)
    monkeypatch.setattr(torchvision.models, 'resnet18', lambda weights=None, **kw: reference)
    policy.load_pretrained_backbone()  # every camera's encoder starts from the same weights
    qpos, image = torch.randn(2, 14), torch.rand(2, 3, 3, 64, 64)
    policy(qpos, image, torch.randn(2, 16, 14), torch.zeros(2, 16, dtype=torch.bool))['loss'].backward()
    assert policy.eval()(qpos, image).shape == (2, 16, 14)
    assert not small_policy().use_separate_rgb_encoder_per_camera  # shared by default
