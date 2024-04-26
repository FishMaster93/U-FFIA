import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.modules import ConvBlock, ConvBlock5x5, init_layer
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from utils.pytorch_utils import do_mixup, interpolate, pad_framewise_output


class PANNS_Cnn6(nn.Module):
    def __init__(self, classes_num):
        
        super(PANNS_Cnn6, self).__init__()
       
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        init_layer(self.fc_audioset)   

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        # output_dict = {
        #     'clipwise_output': clipwise_output, 
        #     'embedding': embedding}

        return clipwise_output

class PANNS_Cnn10(nn.Module):
    def __init__(self, classes_num):
        
        super(PANNS_Cnn10, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_layer(self.fc_audioset) 

    def forward(self, x):



        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        # output_dict = {
        #     'clipwise_output': clipwise_output, 
        #     'embedding': embedding}

        # return output_dict
        return clipwise_output


class PANNS_Cnn14(nn.Module):
    def __init__(self, classes_num):
        
        super(PANNS_Cnn14, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_layer(self.fc_audioset) 
        # self.load_from_ckpt()  

    # def load_from_ckpt(self):
    #     pretrained_cnn = torch.load('pretrained_models/Cnn14_mAP=0.431.pth')['model']
    #     dict_new = self.state_dict().copy()
    #     trained_list = [i for i in pretrained_cnn.keys() if not ('fc' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
    #     for i in range(len(trained_list)):
    #         dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
    #     self.load_state_dict(dict_new)

    def forward(self, input, mixup_lambda=None):


        # Mixup on spectrogram

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
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': embedding}

        return output_dict

if __name__ == '__main__':
    model = PANNS_Cnn14()
    print(model(torch.randn(1, 1, 51, 40)))