"""bf16 training with fp32 master weights.

The model's parameters, activations and gradients are bf16. AdamW keeps fp32
master copies of the parameters and fp32 moments, so updates far below bf16
resolution (~0.4% of a weight) still accumulate; after each step the master
weights are rounded back into the bf16 model. Gradients are upcast one bucket
at a time to bound the transient fp32 memory.
"""
import torch

# Training always uses bf16; fp32 only describes checkpoints trained before bf16 (no
# precision key), which still load and infer at the precision they were trained with.
PRECISIONS = ('fp32', 'bf16')


def policy_dtype(precision):
    """Parameter dtype of a policy trained at ``precision``."""
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}, got {precision!r}')
    return torch.bfloat16 if precision == 'bf16' else torch.float32


class MasterWeightAdamW(torch.optim.AdamW):
    """AdamW over fp32 master copies of low-precision model parameters.

    ``zero_grad`` clears the model's gradients and ``step`` reads them. The state
    dict also carries the master weights so a resumed run continues exactly.
    """

    def __init__(self, params, *, bucket_numel=2 ** 26, **kwargs):
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{'params': params}]
        buckets, master_groups = [], []
        for group in groups:  # each group (e.g. a backbone learning rate) is split into buckets
            options = {k: v for k, v in group.items() if k != 'params'}
            current, size = [], 0
            for param in (p for p in group['params'] if p.requires_grad):
                if current and size + param.numel() > bucket_numel:
                    buckets.append(current)
                    current, size = [], 0
                current.append(param)
                size += param.numel()
            if current:
                buckets.append(current)
            master_groups += [dict(options, params=[p.detach().to(torch.float32, copy=True) for p in bucket])
                              for bucket in buckets[len(master_groups):]]
        if not buckets:
            raise ValueError('MasterWeightAdamW needs trainable parameters')
        if 'fused' not in kwargs and 'foreach' not in kwargs:
            kwargs['fused'] = all(p.is_cuda for bucket in buckets for p in bucket)
        super().__init__(master_groups, **kwargs)
        self.model_param_groups = buckets

    def zero_grad(self, set_to_none=True):
        for bucket in self.model_param_groups:
            for param in bucket:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('MasterWeightAdamW does not support closures')
        groups = self.param_groups
        try:
            for group, bucket in zip(groups, self.model_param_groups):
                for master, param in zip(group['params'], bucket):
                    master.grad = None if param.grad is None else param.grad.float()
                self.param_groups = [group]
                super().step()
                for master in group['params']:
                    master.grad = None
                torch._foreach_copy_(bucket, group['params'])
        finally:
            self.param_groups = groups

    def state_dict(self):
        state = super().state_dict()
        # CPU copies: a snapshot that needs no GPU memory (training may leave none to spare).
        state['master_params'] = [p.detach().cpu() for group in self.param_groups for p in group['params']]
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        masters = state_dict.pop('master_params', None)
        super().load_state_dict(state_dict)
        if masters is None:
            return  # keep masters initialized from the (already loaded) model weights
        current = [p for group in self.param_groups for p in group['params']]
        if len(masters) != len(current):
            raise ValueError('Saved master weights do not match the optimizer parameters')
        for target, saved in zip(current, masters):
            target.copy_(saved)
        for group, bucket in zip(self.param_groups, self.model_param_groups):
            torch._foreach_copy_(bucket, group['params'])
