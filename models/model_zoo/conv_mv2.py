import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class Mobilev2Block(nn.Module):
#     def __init__(self, in_channels, out_channels, r=1):
#         super(Mobilev2Block, self).__init__()

#         size = 3
#         pad = size // 2
#         r = r

#         self.conv1a = nn.Conv2d(in_channels=in_channels,
#                                 out_channels=out_channels * r,
#                                 kernel_size=(1, 1), stride=(1, 1),
#                                 dilation=(1, 1),
#                                 padding=(0, 0), bias=False)

#         self.conv1b = nn.Conv2d(in_channels=out_channels * r,
#                                 out_channels=out_channels * r,
#                                 kernel_size=(size, size), stride=(1, 1),
#                                 dilation=(1, 1), groups=out_channels * r,
#                                 padding=(pad, pad), bias=False)

#         self.conv1c = nn.Conv2d(in_channels=out_channels * r,
#                                 out_channels=out_channels,
#                                 kernel_size=(1, 1), stride=(1, 1),
#                                 dilation=(1, 1),
#                                 padding=(0, 0), bias=False)

#         self.conv2a = nn.Conv2d(in_channels=out_channels,
#                                 out_channels=out_channels * r,
#                                 kernel_size=(1, 1), stride=(1, 1),
#                                 dilation=(1, 1),
#                                 padding=(0, 0), bias=False)

#         self.conv2b = nn.Conv2d(in_channels=out_channels * r,
#                                 out_channels=out_channels * r,
#                                 kernel_size=(size, size), stride=(1, 1),
#                                 dilation=(1, 1), groups=out_channels * r,
#                                 padding=(pad, pad), bias=False)

#         self.conv2c = nn.Conv2d(in_channels=out_channels * r,
#                                 out_channels=out_channels,
#                                 kernel_size=(1, 1), stride=(1, 1),
#                                 dilation=(1, 1),
#                                 padding=(0, 0), bias=False)

#         self.bn1a = nn.BatchNorm2d(in_channels)
#         self.bn1b = nn.BatchNorm2d(out_channels * r)
#         self.bn1c = nn.BatchNorm2d(out_channels)
#         self.bn2a = nn.BatchNorm2d(out_channels)
#         self.bn2b = nn.BatchNorm2d(out_channels * r)
#         self.bn2c = nn.BatchNorm2d(out_channels)

#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#             self.is_shortcut = True
#         else:
#             self.is_shortcut = False

#         self.init_weights()

#     def init_weights(self):
#         init_layer(self.conv1a)
#         init_layer(self.conv1b)
#         init_layer(self.conv1c)
#         init_layer(self.conv2a)
#         init_layer(self.conv2b)
#         init_layer(self.conv2c)
#         init_bn(self.bn1a)
#         init_bn(self.bn1b)
#         init_bn(self.bn1c)
#         init_bn(self.bn2a)
#         init_bn(self.bn2b)
#         init_bn(self.bn2c)

#         if self.is_shortcut:
#             init_layer(self.shortcut)

#     def forward(self, input, pool_size=(2, 2), pool_type='avg'):

#         origin = input
#         x = self.conv1a(F.leaky_relu_(self.bn1a(origin), negative_slope=0.01))
#         x = self.conv1b(F.leaky_relu_(self.bn1b(x), negative_slope=0.01))
#         x = self.conv1c(F.leaky_relu_(self.bn1c(x), negative_slope=0.01))

#         if self.is_shortcut:
#             origin = self.shortcut(origin) + x
#         else:
#             origin = origin + x

#         x = self.conv2a(F.leaky_relu_(self.bn2a(origin), negative_slope=0.01))
#         x = self.conv2b(F.leaky_relu_(self.bn2b(x), negative_slope=0.01))
#         x = self.conv2c(F.leaky_relu_(self.bn2c(x), negative_slope=0.01))

#         x = origin + x

#         x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)

#         return x

class Mobilev2Block(nn.Module):
    def __init__(self, in_channels, out_channels, r=1):
        super(Mobilev2Block, self).__init__()

        size = 3
        pad = size // 2
        r = r

        self.conv1a = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels * r,
                                kernel_size=(1, 1), stride=(1, 1),
                                dilation=(1, 1),
                                padding=(0, 0), bias=False)

        self.conv1b = nn.Conv2d(in_channels=out_channels * r,
                                out_channels=out_channels * r,
                                kernel_size=(size, size), stride=(1, 1),
                                dilation=(1, 1), groups=out_channels * r,
                                padding=(pad, pad), bias=False)

        self.conv1c = nn.Conv2d(in_channels=out_channels * r,
                                out_channels=out_channels,
                                kernel_size=(1, 1), stride=(1, 1),
                                dilation=(1, 1),
                                padding=(0, 0), bias=False)

        self.conv2a = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels * r,
                                kernel_size=(1, 1), stride=(1, 1),
                                dilation=(1, 1),
                                padding=(0, 0), bias=False)

        self.conv2b = nn.Conv2d(in_channels=out_channels * r,
                                out_channels=out_channels * r,
                                kernel_size=(size, size), stride=(1, 1),
                                dilation=(1, 1), groups=out_channels * r,
                                padding=(pad, pad), bias=False)

        self.conv2c = nn.Conv2d(in_channels=out_channels * r,
                                out_channels=out_channels,
                                kernel_size=(1, 1), stride=(1, 1),
                                dilation=(1, 1),
                                padding=(0, 0), bias=False)

        self.bn1a = nn.BatchNorm2d(in_channels)
        self.bn1b = nn.BatchNorm2d(out_channels * r)
        self.bn1c = nn.BatchNorm2d(out_channels)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels * r)
        self.bn2c = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1a)
        init_layer(self.conv1b)
        init_layer(self.conv1c)
        init_layer(self.conv2a)
        init_layer(self.conv2b)
        init_layer(self.conv2c)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn1c)
        init_bn(self.bn2a)
        init_bn(self.bn2b)
        init_bn(self.bn2c)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        origin = input
        x = self.conv1a(self.GELU(self.bn1a(origin)))
        x = self.conv1b(self.GELU(self.bn1b(x)))
        x = self.conv1c(self.GELU(self.bn1c(x)))

        if self.is_shortcut:
            origin = self.shortcut(origin) + x
        else:
            origin = origin + x

        x = self.conv2a(self.GELU(self.bn2a(origin)))
        x = self.conv2b(self.GELU(self.bn2b(x)))
        x = self.conv2c(self.GELU(self.bn2c(x)))

        x = origin + x

        x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)

        return x



class Cnn14_mobilev2(nn.Module):
    def __init__(self, classes_num=4, dim=16, patch_size=4):
        super(Cnn14_mobilev2, self).__init__()
        # self.conv2d1 = nn.Sequential(
        #     nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
        #     nn.GELU(),
        #     nn.BatchNorm2d(dim)
        # )
        self.conv_block0 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block1 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block2 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block3 = Mobilev2Block(in_channels=64, out_channels=128)
        self.conv_block4 = Mobilev2Block(in_channels=128, out_channels=256)
        self.conv_block5 = Mobilev2Block(in_channels=256, out_channels=512)
        # self.conv_block6 = Mobilev2Block(in_channels=512, out_channels=1024)

        self.fc1 = nn.Linear(512, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()
        self.GELU = nn.GELU()
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        """
        Input: (batch_size, data_length)"""
        # x = self.conv2d1(x)
        x = self.conv_block0(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.leaky_relu_(self.fc1(x), negative_slope=0.01)
        x = self.GELU(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        return clipwise_output, embedding