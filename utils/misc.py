import math
import copy

import torch
from torch import nn, Tensor
import torch.distributed as dist


def pad(kernel_size, dilation=1) -> int:
    padding = (kernel_size - 1) // 2 * dilation
    return padding


def _make_divisible(value: float, divisor=8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def round_repeats(num_repeats: int, depth_mult: float) -> int:
    if depth_mult == 1.0:
        return num_repeats
    return int(math.ceil(num_repeats * depth_mult))


def round_filters(filters: int, width_mult: float) -> int:
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def add_weight_decay(model, weight_decay=1e-5):
    """ Applying weight decay to only weights, not biases """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 and name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0}, {"params": decay, "weight_decay": weight_decay}]


def _stochastic_depth(x: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    assert 0.0 < p < 1.0, f"drop probability has to be between 0 and 1, but got {p}"
    assert mode in ["batch", "row"], f"mode has to be either 'batch' or 'row', but got {mode}"
    if not training or p == 0.0:
        return x

    survival_rate = 1.0 - p
    size = [x.shape[0]] + [1] * (x.ndim - 1)

    noise = torch.empty(size, dtype=x.dtype, device=x.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return x * noise


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return _stochastic_depth(x, self.p, self.mode, self.training)


class EMA(nn.Module):

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.fn = lambda e, m: self.decay * e + (1. - self.decay) * m

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                new_value = self.fn(ema_v, model_v)
                ema_v.copy_(new_value)
