import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.modules import InvertedResidual, init_layer, init_bn
import os


class MobileNetV2(nn.Module):
    def __init__(self, classes_num=4):

        super().__init__()
        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # self.fc1 = nn.Linear(1280, 1024, bias=True)
        # self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        """
        Input: (batch_size, 1, time_steps, mel_bins)"""  # (128, 1, 251, 64)
        x = self.features(x)  # (128, 1280, 7, 2)
        x1 = x.transpose(3, 1)
        x1 = self.fc1(x1)
        B, T, M, C= x1.size()
        audio_embed = x1.reshape(B, T*M, C)  # (128, 14, 1024)
        audio_logits = F.avg_pool2d(audio_embed, (audio_embed.size(1),1))
        x = torch.mean(x, dim=3)  # (128, 1280, 7)
        (x1, _) = torch.max(x, dim=2)  # x1 (128,1280)
        x2 = torch.mean(x, dim=2)   # x2: (128, 1280)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        clip_embedding = F.dropout(x, p=0.5, training=self.training)    # clip_embed (128, 1024)

        clipwise_output = self.fc_audioset(x)  # (128, 4)

        # TODO (frame)clip_embed: [B, T, Dim]
        return clipwise_output,  audio_embed


def load_MobileNetV2_weight(model, file_weight='pretrained_models/S3D_kinetics400.pt'):
    if os.path.isfile(file_weight):
        weight_dict = torch.load(file_weight)  # load pretrained weights
        model_dict = model.state_dict()        # model's weights
        for name, param in weight_dict.items():
            if "module" in name and "module.fc" not in name:
                name = ".".join(name.split(".")[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(" size? " + name, param.size(), model_dict[name].size())
            else:
                pass
        return model
    else:
        print("weight file?")



if __name__ == '__main__':
    model = MobileNetV2(classes_num=4)
    input = torch.randn(128,1,251,64)
    output_dict = model(input)