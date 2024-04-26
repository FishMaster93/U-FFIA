import math
from functools import partial
import torch
from torch import nn, Tensor
from utils.misc import pad, round_filters, round_repeats, StochasticDepth

"""
EfficientNet params
name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
"""


class SqueezeExcitation(torch.nn.Module):

    def __init__(self, in_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(in_channels, squeeze_channels, 1),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(squeeze_channels, in_channels, 1),
                                   nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.block(x)


class Conv2dNormAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 groups: int = 1,
                 norm=torch.nn.BatchNorm2d,
                 act=torch.nn.SiLU,
                 dilation: int = 1,
                 ) -> None:
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding=pad(kernel_size, dilation) if padding is None else padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=False)

        self.norm = norm(out_channels)
        self.act = act(inplace=True) if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(self,
                 expand_ratio: int,
                 kernel: int,
                 stride: int,
                 in_channels: int,
                 out_channels: int,
                 stochastic_depth_prob: float,
                 norm) -> None:
        super().__init__()
        # condition of using residual connection
        self.use_residual = stride == 1 and in_channels == out_channels
        assert stride in [1, 2], f"{stride} is illegal stride value"

        # squeezed channels
        squeeze_channels = max(1, in_channels // 4)
        # expanded channels
        exp_channels = round_filters(in_channels, expand_ratio)
        condition = exp_channels != in_channels

        self.block = nn.Sequential(
            Conv2dNormAct(in_channels, exp_channels, kernel_size=1, norm=norm) if condition else nn.Identity(),
            # depthwise
            Conv2dNormAct(exp_channels, exp_channels, kernel_size=kernel, stride=stride, groups=exp_channels,
                          norm=norm),
            # squeeze and excitation
            SqueezeExcitation(exp_channels, squeeze_channels),
            # project
            Conv2dNormAct(exp_channels, out_channels, kernel_size=1, norm=norm, act=None)
        )

        self.stc_depth = StochasticDepth(stochastic_depth_prob, "row")  # stochastic depth

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_residual:
            result = self.stc_depth(result)
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            classes_num: int = 4,
            norm=None,
            width_mult=1.0,
            depth_mult=1.0
    ) -> None:
        super().__init__()
        self._init_weight()

        self.t = [1, 6, 6, 6, 6, 6, 6]  # expand ratios
        self.k = [3, 3, 5, 3, 5, 5, 3]  # kernel sizes
        self.s = [1, 2, 2, 2, 1, 2, 1]  # strides

        self.ic = [32, 16, 24, 40, 80, 112, 192]  # input channels
        self.oc = [16, 24, 40, 80, 112, 192, 320]  # output channels
        self.nr = [1, 2, 2, 3, 3, 4, 1]  # number of repeats

        # applying width_mult and depth_mult
        self.ic = [round_filters(in_channels, width_mult) for in_channels in self.ic]
        self.oc = [round_filters(out_channels, width_mult) for out_channels in self.oc]
        self.nr = [round_repeats(num_repeats, depth_mult) for num_repeats in self.nr]

        configs = zip(self.t, self.k, self.s, self.ic, self.oc, self.nr)
        norm = norm if norm is not None else nn.BatchNorm2d

        # building first layer
        layers = [Conv2dNormAct(1, self.ic[0], kernel_size=3, stride=2, norm=norm)]

        # building inverted residual blocks
        total_stage_blocks = sum(self.nr)
        stage_block_id = 0
        for expand_ratio, kernel_size, stride, in_channels, out_channels, repeats in configs:
            for idx in range(repeats):
                stride = stride if idx == 0 else 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                layers += [MBConv(expand_ratio, kernel_size, stride, in_channels, out_channels, sd_prob, norm)]
                stage_block_id += 1
                in_channels = out_channels

        # building last several layers
        last_input = self.oc[-1]
        last_output = round_filters(1280, width_mult)
        layers.append(Conv2dNormAct(last_input, last_output, kernel_size=1, norm=norm))

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(last_output, classes_num))

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.size(0))
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x


def efficientnet_b0(**kwargs) -> EfficientNet:
    return EfficientNet(0.2, width_mult=1.0, depth_mult=1.0, **kwargs)


def efficientnet_b1(**kwargs) -> EfficientNet:
    return EfficientNet(0.2, width_mult=1.0, depth_mult=1.1, **kwargs)


def efficientnet_b2(**kwargs) -> EfficientNet:
    return EfficientNet(0.3, width_mult=1.1, depth_mult=1.2, **kwargs)


def efficientnet_b3(**kwargs) -> EfficientNet:
    return EfficientNet(0.3, width_mult=1.2, depth_mult=1.4, **kwargs)


def efficientnet_b4(**kwargs) -> EfficientNet:
    return EfficientNet(0.4, width_mult=1.4, depth_mult=1.8, **kwargs)


def efficientnet_b5(**kwargs) -> EfficientNet:
    batch_norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    return EfficientNet(0.4, norm=batch_norm, width_mult=1.6, depth_mult=2.2, **kwargs)


def efficientnet_b6(**kwargs) -> EfficientNet:
    batch_norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    return EfficientNet(0.5, norm=batch_norm, width_mult=1.8, depth_mult=2.6, **kwargs)


def efficientnet_b7(**kwargs) -> EfficientNet:
    batch_norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    return EfficientNet(0.5, norm=batch_norm, width_mult=2.0, depth_mult=3.1, **kwargs)


def num_params(model):
    return "Num. of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    b1 = efficientnet_b7()
    input = torch.randn(128, 1, 126, 64)
    output_dict = b1(input)
    print(num_params(b1))