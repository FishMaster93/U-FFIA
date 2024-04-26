from typing import Callable, Any, Optional, List
from einops import rearrange
import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.SiLU,
        dilation: int = 1
    ):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation = dilation, groups = groups, bias = norm_layer is None)

        self.norm_layer = nn.BatchNorm2d(out_channels) if norm_layer is None else norm_layer(out_channels)
        self.act = activation_layer() if activation_layer is not None else activation_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, dim_head = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5

        self.scale_factor = dim ** -0.5

        # Weight matrix for output, Size: num_heads*dim_head X dim
        # Final linear transformation layer
        self.w_out = nn.Linear(_weight_dim, dim, bias = False)

    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = torch.softmax(dots, dim = -1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.w_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetv2 InvertedResidual block
    """
    def __init__(self, in_channels, out_channels, stride = 1, expand_ratio = 2, act_layer = nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size = 1, activation_layer = None))

        # Depth-wise convolution
        layers.append(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size = 3, stride = stride,
                        padding = 1, groups = hidden_dim, activation_layer = act_layer)
        )
        # Point-wise convolution
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, bias = False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim):
        super(MobileVitBlock, self).__init__()
        # Local representation
        self.local_representation = nn.Sequential(
            # Encode local spatial information
            ConvNormAct(in_channels, in_channels, 3),
            # Projects the tensor to a high-diementional space
            ConvNormAct(in_channels, d_model, 1)
        )

        self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)

        # Fusion block
        self.fusion_block1 = nn.Conv2d(d_model, in_channels, kernel_size = 1)
        self.fusion_block2 = nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1)

    def forward(self, x):
        local_repr = self.local_representation(x)
        # global_repr = self.global_representation(local_repr)
        _, _, h, w = local_repr.shape
        global_repr = rearrange(local_repr, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=2, pw=2)
        global_repr = self.transformer(global_repr)
        global_repr = rearrange(global_repr, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//2, w=w//2, ph=2, pw=2)

        # Fuse the local and gloval features in the concatenation tensor
        fuse_repr = self.fusion_block1(global_repr)
        result = self.fusion_block2(torch.cat([x, fuse_repr], dim = 1))
        return result


model_cfg = {
    "xxs":{
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        "layers": [2, 4, 3]
    },
    "xs":{
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
}


class MobileViT(nn.Module):
    def __init__(self, img_size, features_list, d_list, transformer_depth, expansion, classes_num = 4):
        super(MobileViT, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features_list[0], kernel_size=3, stride=2, padding=1),
            InvertedResidual(in_channels=features_list[0], out_channels=features_list[1], stride=1, expand_ratio=expansion),
        )

        self.stage1 = nn.Sequential(
            InvertedResidual(in_channels=features_list[1], out_channels=features_list[2], stride=2, expand_ratio=expansion),
            InvertedResidual(in_channels=features_list[2], out_channels=features_list[2], stride=1, expand_ratio=expansion),
            InvertedResidual(in_channels=features_list[2], out_channels=features_list[3], stride=1, expand_ratio=expansion)
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels=features_list[3], out_channels=features_list[4], stride=2, expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[4], out_channels=features_list[5], d_model=d_list[0],
                           layers=transformer_depth[0], mlp_dim=d_list[0] * 2)
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels=features_list[5], out_channels=features_list[6], stride=2, expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[6], out_channels=features_list[7], d_model=d_list[1],
                           layers=transformer_depth[1], mlp_dim=d_list[1] * 4)
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels=features_list[7], out_channels=features_list[8], stride=2, expand_ratio=expansion),
            MobileVitBlock(in_channels=features_list[8], out_channels=features_list[9], d_model=d_list[2],
                           layers=transformer_depth[2], mlp_dim=d_list[2] * 4),
            nn.Conv2d(in_channels=features_list[9], out_channels=features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=img_size // 32)
        # self.avgpool = nn.AvgPool2d((4, 2))
        self.fc = nn.Linear(features_list[10], classes_num)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        # Body
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def MobileViT_XXS(img_size=128, classes_num=4):
    cfg_xxs = model_cfg["xxs"]
    model_xxs = MobileViT(img_size, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"], classes_num)
    return model_xxs


def MobileViT_XS(img_size=256, classes_num=1000):
    cfg_xs = model_cfg["xs"]
    model_xs = MobileViT(img_size, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"], classes_num)
    return model_xs


def MobileViT_S(img_size=256, classes_num=1000):
    cfg_s = model_cfg["s"]
    model_s = MobileViT(img_size, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"], classes_num)
    return model_s


if __name__ == "__main__":
    img = torch.randn(10, 1, 126, 64)
    net = MobileViT_XXS()
    out = net(img)

    # cfg_xs = model_cfg["xs"]
    # model_xs = MobileViT(256, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"])
    #
    # cfg_s = model_cfg["s"]
    # model_s = MobileViT(256, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"])
    #
    # print(model_s)
    #
    # # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    # print("XXS params: ", sum(p.numel() for p in model_xxs.parameters()))
    # print(" XS params: ", sum(p.numel() for p in model_xs.parameters()))
    # print("  S params: ", sum(p.numel() for p in model_s.parameters()))