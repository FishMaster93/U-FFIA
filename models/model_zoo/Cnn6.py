import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.modules import ConvBlock5x5, init_layer, init_bn


class Cnn6(nn.Module):
    def __init__(self, classes_num=4):

        super(Cnn6, self).__init__()

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        # self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        # init_layer(self.fc_audioset)

    def forward(self, x):
        """
        Input: (batch_size, data_length)"""

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = x.permute(0, 2, 1)  # batch x time x channel (512)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        # clipwise_output = self.fc_audioset(x)

        # output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        # return output_dict