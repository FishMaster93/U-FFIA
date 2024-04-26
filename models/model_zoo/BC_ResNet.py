import torch.nn as  nn
import torch
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from models.model_zoo.modules import TransitionBlock, BroadcastedBlock, init_bn, init_layer



class BC_ResNet(nn.Module):
    def __init__(self, classes_num=4, norm=False):
        
        super(BC_ResNet, self).__init__()

        c = 4
        c = 10 * c
        self.conv1 = nn.Conv2d(1, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.block2_1 = nn.MaxPool2d(2)

        self.block3_1 = TransitionBlock(c, int(1.5 * c))
        self.block3_2 = BroadcastedBlock(int(1.5 * c))

        self.block4_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(1.5 * c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), classes_num, 1)

        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = norm
        self.fc_audioset = nn.Linear(1, classes_num, bias=True)
        if norm:
           self.one = nn.InstanceNorm2d(1)
           self.two = nn.InstanceNorm2d(int(1))
           self.three = nn.InstanceNorm2d(int(1))
           self.four = nn.InstanceNorm2d(int(1))
           self.five = nn.InstanceNorm2d(int(1))

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc_audioset)


 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        out = input
        if self.norm:
               out =self.lamb*out + self.one(out)
        out = self.conv1(out)

        out = self.block1_1(out)

        out = self.block1_2(out)
        if self.norm:
           out =self.lamb*out + self.two(out)

        out = self.block2_1(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        if self.norm:
           out =self.lamb*out + self.three(out)

        out = self.block4_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        if self.norm:
           out =self.lamb*out + self.four(out)

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        if self.norm:
           out =self.lamb*out + self.five(out)

        out = self.block7_1(out)

        out = self.block8_1(out)
        out = self.block8_1(out)

        embedding = F.dropout(out, p=0.2, training=self.training)
        clipwise_output = torch.squeeze(torch.squeeze(out,dim=2),dim=2)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding}

        return clipwise_output, embedding



if __name__ == '__main__':

    model = BC_ResNet(classes_num=4)
    input = torch.randn(128,1,251,64)
    output_dict = model(input)
