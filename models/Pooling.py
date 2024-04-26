import torch
import torch.nn as nn
import torch.nn.functional as F
from models.specpool2d import SpectralPool2d


class Pooling_layer(nn.Module):
    def __init__(self, factor=0.75):
        super(Pooling_layer, self).__init__()
        self.factor = factor
        self.SpecPool2d = SpectralPool2d(scale_factor=(factor, 1))

    def forward(self, x):
        """
        args:
            x: input mel spectrogram [batch, 1, time, frequency]
        return:
            out: reduced features [batch, 1, factor*time, frequency]
        """
        out = self.SpecPool2d(x)
        return out


if __name__ == '__main__':
    input = torch.randn(4, 1, 128, 128)
    model = Pooling_layer(factor=0.5)
    print(model(input).shape)