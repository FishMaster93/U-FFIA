import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
 
 
def vgg_block(num_convs, in_channels, out_channels):
    """
    vgg block: Conv2d ReLU MaxPool2d
    """
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    return blk
 
 
class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        features = []
        features.extend(vgg_block(2, 3, 64))
        features.extend(vgg_block(2, 64, 128))
        features.extend(vgg_block(3, 128, 256))
        self.index_pool3 = len(features)  # pool3
        features.extend(vgg_block(3, 256, 512))
        self.index_pool4 = len(features)  # pool4
        features.extend(vgg_block(3, 512, 512))  # pool5
 
        self.features = nn.Sequential(*features)  # 模型容器，有state_dict参数(字典类型)
 
        """ 将传统CNN中的全连接操作，变成卷积操作conv6 conv7 此时不进行pool操作，图像大小不变，此时图像不叫feature map而是heatmap"""
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=1)   # conv6
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)  # conv7
 
        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()  # state_dict()存放训练过程中需要学习的权重和偏置系数,字典类型
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.features.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.features.load_state_dict(new_dict)  # load_state_dict必须传入字典对象，将预训练的参数权重加载到features中
 
    def forward(self, x):
        
        pool3 = self.features[:self.index_pool3](x)  # 图像大小为原来的1/8
        pool4 = self.features[self.index_pool3:self.index_pool4](pool3)  # 图像大小为原来的1/16
        # pool4 = self.features[:self.index_pool4](x)    # pool4的第二种写法，较浪费时间(从头开始)
 
        pool5 = self.features[self.index_pool4:](pool4)  # 图像大小为原来的1/32
 
        conv6 = self.relu(self.conv6(pool5))  # 图像大小为原来的1/32
        conv7 = self.relu(self.conv7(conv6))  # 图像大小为原来的1/32
        return pool3, pool4, conv7
 
 
class FCN(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        """
        Args:
            num_classes: 分类数目
            backbone: 骨干网络 VGG
        """
        super(FCN, self).__init__()
        if backbone == 'vgg':
            self.features = VGG16()  # 参数初始化
 
        # 1*1卷积，将通道数映射为类别数
        self.scores1 = nn.Conv2d(4096, num_classes, kernel_size=1)  # 对conv7操作
        self.relu = nn.ReLU(inplace=True)
        self.scores2 = nn.Conv2d(512, num_classes, kernel_size=1)   # 对pool4操作 
        self.scores3 = nn.Conv2d(256, num_classes, kernel_size=1)   # 对pool3操作
 
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8)  # 转置卷积
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
 
 
    def forward(self, x):
        b, c, h, w = x.shape
        pool3, pool4, conv7 = self.features(x)
 
        conv7 = self.relu(self.scores1(conv7)) #4,12,11,15
 
        pool4 = self.relu(self.scores2(pool4)) #4,12,22,30
 
        pool3 = self.relu(self.scores3(pool3)) #4, 12,45,60

        # 融合之前调整一下h w
        conv7_2x = F.interpolate(self.upsample_2x(conv7), size=(pool4.size(2), pool4.size(3)))  # conv7 2倍上采样，调整到pool4的大小
        s=conv7_2x+pool4  # conv7 2倍上采样与pool4融合
 
        s=F.interpolate(self.upsample_2x(s),size=(pool3.size(2),pool3.size(3)))  # 融合后的特征2倍上采样，调整到pool3的大小
        s = pool3 + s     # 融合后的特征与pool3融合
 
        out_8s=F.interpolate(self.upsample_8x(s) ,size=(h,w))  # 8倍上采样得到 FCN-8s，得到和原特征x一样大小的特征
        import ipdb; ipdb.set_trace()
 
        return out_8s
 
if __name__=='__main__':
    model = FCN(num_classes=12)
 
    fake_img=torch.randn((4,3,360,480))  # B C H W
 
    output_8s=model(fake_img)
    print(output_8s.shape)
 
 