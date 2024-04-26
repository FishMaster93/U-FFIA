import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim=16, depth=6, kernel_size=3, patch_size=4, n_classes=4):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.conv2d1(x)

        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        x = self.head(x)

        return x


if __name__ == '__main__':
    model = ConvMixer().to(device)
    input = torch.randn(5, 1, 128, 128).to(device)
    out = model(input)
    print(out.shape)
    # summary(model, (10, 3, 224, 224))
