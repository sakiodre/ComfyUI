import torch
from contextlib import contextmanager

import comfy.cli_args

ENABLE_AIT = comfy.cli_args.enable_ait

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None, use_swish=False):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
        if use_swish:
            self.swish = torch.nn.SiLU()
        else:
            self.swish = torch.nn.Identity()

    def forward(self, input):
        return self.swish(super().forward(input))

def time_embed(model_channels, embed_dim, dtype, device):
    return torch.nn.Sequential(
            Linear(model_channels, embed_dim, dtype=dtype, device=device),
            torch.nn.SiLU(),
            Linear(embed_dim, embed_dim, dtype=dtype, device=device),
        )

if ENABLE_AIT:
    from aitemplate.frontend import nn, Tensor
    class Linear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
            return super().__init__(in_features, out_features, bias, dtype=dtype)

    def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        if bias == True and padding_mode == 'zeros':
            return nn.Conv2dBias(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype=dtype)
        else:
            print("error UNIMPLEMENTED")

    def time_embed(model_channels, embed_dim, dtype, device):
        return nn.Sequential(
                nn.Linear(model_channels, embed_dim, specialization="swish", dtype=dtype),
                nn.Identity(),
                nn.Linear(embed_dim, embed_dim, dtype=dtype),
            )
    GroupNorm = nn.GroupNorm

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


@contextmanager
def use_comfy_ops(device=None, dtype=None): # Kind of an ugly hack but I can't think of a better way
    old_torch_nn_linear = torch.nn.Linear
    force_device = device
    force_dtype = dtype
    def linear_with_dtype(in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        if force_device is not None:
            device = force_device
        if force_dtype is not None:
            dtype = force_dtype
        return Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    torch.nn.Linear = linear_with_dtype
    try:
        yield
    finally:
        torch.nn.Linear = old_torch_nn_linear
