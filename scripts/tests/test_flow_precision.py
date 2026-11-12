"""bf16 Flow training/inference and the channels-last training layout. Offline, no robot."""
import pytest
import torch

from policy.flow_rtc import PerActionTimeUNet
from policy.mixed_precision import MasterWeightAdamW, policy_dtype
from test_flow_per_action_rtc import small_policy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def unet_inputs(policy, batch=3):
    unet = policy.model.diffusion.unet
    cond_dim = unet.down_modules[0][0].cond_encoder[1].in_features - unet.diffusion_step_encoder[-1].out_features
    return torch.randn(batch, 16, 7), torch.rand(batch, 16), torch.randn(batch, cond_dim)


def test_channels_last_layout_is_the_same_function_with_the_same_parameters():
    torch.manual_seed(0)
    policy = small_policy()
    unet = policy.model.diffusion.unet
    x, time, condition = unet_inputs(policy)
    expected = unet(x, time, global_cond=condition)
    names = {name: param for name, param in policy.model.named_parameters()}
    unet.use_channels_last()
    assert unet.channels_last
    assert {name: param for name, param in policy.model.named_parameters()} == names  # same Parameter objects
    weight = unet.down_modules[0][0].conv1.block[0].weight
    assert weight.shape[-1] == 5 and weight.unsqueeze(2).is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(unet(x, time, global_cond=condition), expected, rtol=1e-5, atol=1e-5)
    # Gradients flow to the same parameters with the same values.
    reference = small_policy()
    reference.deserialize(policy.serialize())
    unet(x, time, global_cond=condition).square().sum().backward()
    reference.model.diffusion.unet(x, time, global_cond=condition).square().sum().backward()
    for (name, param), other in zip(policy.model.named_parameters(), reference.model.parameters()):
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-4, atol=1e-5, msg=name)


def test_serialized_weights_are_canonical_whatever_the_training_layout():
    policy = small_policy()
    policy.model.diffusion.unet.use_channels_last()
    state = policy.serialize()
    assert all(value.is_contiguous() for value in state.values())
    fresh = small_policy()
    fresh.deserialize(state)
    for value, other in zip(state.values(), fresh.serialize().values()):
        assert torch.equal(value, other)


def test_bf16_policy_holds_bf16_weights_but_fp32_times_and_sampler():
    torch.manual_seed(0)
    policy = small_policy(precision='bf16')
    assert policy.compute_dtype == torch.bfloat16
    floats = [t for t in policy.model.state_dict().values() if t.is_floating_point()]
    assert floats and all(t.dtype == torch.bfloat16 for t in floats)
    assert policy.model.state_dict()['_per_action_time'].dtype == torch.int64
    assert all(t.dtype == torch.bfloat16 for t in policy.serialize().values() if t.is_floating_point())
    policy.eval()
    qpos, image = torch.randn(2, 7), torch.rand(2, 1, 3, 24, 32)
    actions = policy(qpos, image)
    assert actions.dtype == torch.float32 and actions.shape == (2, 16, 7) and torch.isfinite(actions).all()
    prefix = torch.randn(2, 3, 7)
    assert torch.equal(policy.forward_inf_training_time_rtc(qpos, image, prefix)[:, :3], prefix)
    # Distinct noise times keep distinct fp32 embeddings even where bf16 would round them together.
    unet = policy.model.diffusion.unet
    embedding = unet.diffusion_step_encoder[0]
    times = torch.tensor([0.5, 0.5 + 2 ** -10])
    assert torch.tensor(0.5).bfloat16() == torch.tensor(0.5 + 2 ** -10).bfloat16()
    assert not torch.equal(embedding(times)[0], embedding(times)[1])


def test_bf16_policy_trains_with_fp32_loss_and_master_weights():
    torch.manual_seed(0)
    policy = small_policy(precision='bf16')
    policy.model.diffusion.unet.use_channels_last()
    optimizer = policy.configure_optimizers()
    assert isinstance(optimizer, MasterWeightAdamW) and policy.configure_optimizers() is optimizer
    before = [p.detach().clone() for p in policy.model.parameters()]
    qpos, image = torch.randn(4, 7), torch.rand(4, 1, 3, 24, 32)
    actions, is_pad = torch.randn(4, 16, 7), torch.zeros(4, 16, dtype=torch.bool)
    out = policy(qpos, image, actions, is_pad)
    assert out['loss'].dtype == torch.float32 and torch.isfinite(out['loss'])
    out['loss'].backward()
    assert all(p.grad.dtype == torch.bfloat16 for p in policy.model.parameters() if p.grad is not None)
    optimizer.step()
    optimizer.zero_grad()
    assert all(p.grad is None for p in policy.model.parameters())
    masters = [m for group in optimizer.param_groups for m in group['params']]
    assert all(m.dtype == torch.float32 for m in masters)
    assert all(torch.equal(p.detach(), m.bfloat16()) for p, m in zip(policy.model.parameters(), masters))
    assert any(not torch.equal(p, b) for p, b in zip(policy.model.parameters(), before))


def test_legacy_fp32_policy_keeps_plain_adamw_and_unknown_precision_is_rejected():
    assert type(small_policy().configure_optimizers()) is torch.optim.AdamW
    assert policy_dtype('fp32') == torch.float32  # checkpoints from before bf16 training
    with pytest.raises(ValueError, match='precision'):
        small_policy(precision='fp16')


def test_master_weights_keep_updates_below_bf16_resolution():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = MasterWeightAdamW([param], lr=1e-4, weight_decay=0.0, betas=(0.0, 0.999))
    for _ in range(100):
        param.grad = torch.ones_like(param)
        optimizer.step()
    # 100 updates of ~1e-4 move the weight by ~1e-2, though each is below bf16's 2**-8 spacing at 1.0.
    master = optimizer.param_groups[0]['params'][0]
    assert 0.98 < float(master[0]) < 0.995
    assert param.dtype == torch.bfloat16 and torch.equal(param.detach(), master.bfloat16())
    pure = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    plain = torch.optim.AdamW([pure], lr=1e-4, weight_decay=0.0, betas=(0.0, 0.999))
    for _ in range(100):
        pure.grad = torch.ones_like(pure)
        plain.step()
    assert torch.equal(pure.detach(), torch.ones_like(pure))  # what fp32 masters prevent


def test_master_weight_state_round_trip_and_buckets():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16)) for n in (5, 7, 3)]
    optimizer = MasterWeightAdamW(params, lr=1e-2, bucket_numel=8)
    assert [len(bucket) for bucket in optimizer.model_param_groups] == [1, 1, 1]
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        optimizer.step()
    import io
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)  # as optimizer_*.ckpt; in-memory state dicts alias tensors
    buffer.seek(0)
    state = torch.load(buffer)
    assert len(state['master_params']) == 3
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = MasterWeightAdamW(clones, lr=1e-2, bucket_numel=8)
    restored.load_state_dict(state)
    for a, b in zip([m for g in optimizer.param_groups for m in g['params']],
                    [m for g in restored.param_groups for m in g['params']]):
        assert torch.equal(a, b)
    grads = [torch.randn_like(p) for p in params]
    for opt, ps in ((optimizer, params), (restored, clones)):
        for p, g in zip(ps, grads):
            p.grad = g.clone()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(params, clones))


def test_lr_scheduler_drives_every_bucket():
    from transformers import get_cosine_schedule_with_warmup
    params = [torch.nn.Parameter(torch.zeros(6, dtype=torch.bfloat16)) for _ in range(3)]
    optimizer = MasterWeightAdamW(params, lr=1.0, bucket_numel=6)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    scheduler.step()
    assert len(optimizer.param_groups) == 3
    assert all(group['lr'] == pytest.approx(0.1) for group in optimizer.param_groups)


def test_training_is_always_bf16():
    from config.training_args import build_parser
    from config.training_config import load_training_config
    for flag in ('--precision', '--bf16', '--ddp_bf16_grad_compress'):
        with pytest.raises(SystemExit):
            build_parser().parse_args([flag])
    recipe = load_training_config('single_arm_flow')
    assert 'precision' not in recipe.document['training'] and 'bf16' not in recipe.document['training']
    import inspect
    from policy import imitate_episodes_train
    source = inspect.getsource(imitate_episodes_train.main_train)
    assert source.count("'precision': 'bf16'") == 3  # Flow and ACT policy_config, checkpoint config
    assert 'autocast' not in inspect.getsource(imitate_episodes_train.train_bc)


@cuda
def test_bf16_policy_on_cuda_trains_and_its_fast_inference_matches():
    from policy.fast_flow_inference import GraphedUNetStep, accelerate_flow_policy
    torch.manual_seed(0)
    policy = small_policy(precision='bf16').cuda()
    policy.model.diffusion.unet.use_channels_last()
    optimizer = policy.configure_optimizers()
    assert optimizer.defaults['fused']
    qpos, image = torch.randn(4, 7, device='cuda'), torch.rand(4, 1, 3, 24, 32, device='cuda')
    actions, is_pad = torch.randn(4, 16, 7, device='cuda'), torch.zeros(4, 16, dtype=torch.bool, device='cuda')
    policy(qpos, image, actions, is_pad)['loss'].backward()
    optimizer.step()
    reference = small_policy(precision='bf16').cuda().eval()
    reference.deserialize(policy.serialize())
    fast = small_policy(precision='bf16').cuda().eval()
    fast.deserialize(policy.serialize())
    accelerate_flow_policy(fast)
    assert isinstance(fast.model.diffusion.unet, GraphedUNetStep)
    conv = fast.model.diffusion.unet.unet.down_modules[0][0].conv1.block[0]
    assert conv.weight.dtype == torch.bfloat16
    torch.manual_seed(1)
    expected = reference.forward_inf(qpos[:1], image[:1])
    torch.manual_seed(1)
    actual = fast.forward_inf(qpos[:1], image[:1])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-2)


def test_pretrained_backbone_keeps_batchnorm_in_fp32_and_constructs_without_downloading(monkeypatch):
    import torchvision
    requested = []
    real = torchvision.models.resnet18

    def resnet18(*args, weights=None, **kwargs):
        requested.append(weights)
        return real(weights=None)

    monkeypatch.setattr(torchvision.models, 'resnet18', resnet18)
    policy = small_policy(precision='bf16', pretrained_backbone_weights='IMAGENET1K_V1', height=64, width=64)
    assert requested and all(w is None for w in requested)  # construction (and so inference) never downloads
    norms = [m for m in policy.model.diffusion.rgb_encoder.backbone.modules()
             if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm))]
    assert norms and all(isinstance(m, torch.nn.BatchNorm2d) for m in norms)
    assert all(t.dtype == torch.float32 for m in norms for t in (*m.parameters(), m.running_mean, m.running_var))
    assert policy.model.diffusion.rgb_encoder.backbone[0].weight.dtype == torch.bfloat16
    # Training initialization loads exactly the torchvision weights (here a seeded stand-in).
    torch.manual_seed(3)
    reference = real(weights=None)
    monkeypatch.setattr(torchvision.models, 'resnet18', lambda weights=None: reference)
    policy.load_pretrained_backbone()
    expected = torch.nn.Sequential(*list(reference.children())[:-2]).state_dict()
    for key, value in policy.model.diffusion.rgb_encoder.backbone.state_dict().items():
        assert torch.equal(value, expected[key].to(value.dtype)), key
    # BatchNorm needs >1 value per channel in the shape dry run: 64x64 gives a 2x2 feature map.
    # bf16 activations through fp32 BatchNorm train, step and infer.
    optimizer = policy.configure_optimizers()
    qpos, image = torch.randn(4, 7), torch.rand(4, 1, 3, 64, 64)
    policy(qpos, image, torch.randn(4, 16, 7), torch.zeros(4, 16, dtype=torch.bool))['loss'].backward()
    optimizer.step()
    assert norms[0].weight.dtype == torch.float32
    fresh = small_policy(precision='bf16', pretrained_backbone_weights='IMAGENET1K_V1', height=64, width=64)
    fresh.deserialize(policy.serialize())
    assert fresh.eval()(qpos, image).shape == (4, 16, 7)


def test_pretrained_backbone_rejects_groupnorm_and_other_image_normalization():
    with pytest.raises(ValueError, match='BatchNorm'):
        small_policy(pretrained_backbone_weights='IMAGENET1K_V1', use_group_norm=True)
    with pytest.raises(ValueError, match='imagenet'):
        small_policy(pretrained_backbone_weights='IMAGENET1K_V1', image_normalization='minus_one_one')
    assert small_policy().use_group_norm  # checkpoints without the key stay GroupNorm from scratch


def test_single_arm_flow_recipe_trains_from_the_imagenet_backbone():
    from config.training_args import build_parser
    args = build_parser().parse_args(['--training-config', 'single_arm_flow'])
    assert args.pretrained_backbone_weights == 'IMAGENET1K_V1'


def test_training_never_starts_the_backbone_from_scratch():
    from config.training_args import build_parser
    from policy.imitate_episodes_train import main_train
    assert build_parser().parse_args([]).pretrained_backbone_weights == 'IMAGENET1K_V1'
    for weights in (None, ''):
        with pytest.raises(ValueError, match='pretrained backbone'):
            main_train({'pretrained_backbone_weights': weights})
