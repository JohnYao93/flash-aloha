"""Policy and optimizer factories: Flow and decoder-only ACT."""
POLICY_CLASSES = ('Flow', 'ACT')


def make_policy(policy_class, policy_config):
    if policy_class == 'Flow':
        from policy.policy import LeRobotFlowPolicy
        return LeRobotFlowPolicy(policy_config)
    if policy_class == 'ACT':
        from policy.act import ACTPolicy
        return ACTPolicy(policy_config)
    raise ValueError(f'Only Flow and ACT are supported, got {policy_class!r}')


def accelerate_policy(policy):
    """Inference acceleration for an eval-mode CUDA policy (Flow: fp16/bf16 channels-last convolutions and a
    CUDA-graphed U-Net step; ACT: a CUDA-graphed forward)."""
    from policy.act import ACTPolicy, accelerate_act_policy
    if isinstance(policy, ACTPolicy):
        return accelerate_act_policy(policy)
    from policy.fast_flow_inference import accelerate_flow_policy
    return accelerate_flow_policy(policy)


def make_optimizer(policy_class, policy):
    if policy_class not in POLICY_CLASSES:
        raise ValueError(f'Only Flow and ACT are supported, got {policy_class!r}')
    return policy.configure_optimizers()
