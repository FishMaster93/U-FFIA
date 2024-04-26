import torch
import torch.nn as nn
from models.model_zoo.S3D import load_S3D_weight
from models.model_zoo.ResNet3D import load_R3D_weight
from einops import rearrange, repeat, reduce

# class VideoModel(nn.Module):
#     def __init__(self, backbone, **kwargs):
#         super().__init__(**kwargs)

#         self.backbone = backbone



#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         """

#         clipwise_output = self.backbone(input)
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict




# class VideoModel(nn.Module):
#     def __init__(self, backbone, **kwargs):
#         super().__init__(**kwargs)
#         # ckpt = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/video_best.pt')
#         # self.backbone = load_S3D_weight(backbone, file_weight= '/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/S3D400.pt')
#         self.backbone = backbone.load_state_dict(ckpt['model_state_dict'])
#         self.encoder = backbone
#         old_pretrained_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/video_best.pt')['model_state_dict']
#         dict_new = self.encoder.state_dict().copy()
#         pretrained_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_encoder.items()}
#         trained_list = [i for i in pretrained_encoder.keys() if not ('head' in i or 'pos' in i)]
#         for i in range(len(trained_list)):
#             dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
#         self.encoder.load_state_dict(dict_new)
#         self.backbone = backbone



#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         """
#         # clipwise_output = self.backbone(input)
#         clipwise_output, _ = self.backbone(input)
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict


# class VideoModel(nn.Module):
#     def __init__(self, backbone, **kwargs):
#         super().__init__(**kwargs)
#         # self.backbone=backbone
#         self.backbone = load_S3D_weight(backbone, file_weight= '/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/S3D400.pt')


#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         """
#         # input = input.transpose(2,1)
#         input = torch.mean(input ,dim=1)
#         clipwise_output, _ = self.backbone(input)
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict


class VideoModel_Pre_S3D(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.encoder = backbone
        old_pretrained_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/video_best.pt')['model_state_dict']
        dict_new = self.encoder.state_dict().copy()
        pretrained_encoder = {k.replace('module.', ''): v for k, v in old_pretrained_encoder.items()}
        trained_list = [i for i in pretrained_encoder.keys() if not ('fc' in i )]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.encoder.load_state_dict(dict_new)

    def forward(self, input):
   
        # input = rearrange(input, 'b T C H W -> b (T C) H W')
        clipwise_output = self.encoder(input)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict