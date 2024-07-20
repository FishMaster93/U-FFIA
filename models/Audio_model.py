import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from models.model_zoo.modules import init_bn
from transformer.SubLayers import MultiHeadAttention
import torch.nn.functional as F
from models.Pooling import Pooling_layer

class Audio_Frontend(nn.Module):
    """
    Wav2Mel transformation & Mel Sampling frontend
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax):
        super(Audio_Frontend, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.mel_bins = mel_bins
        # self.Pooling= Pooling_layer(factor=0.5)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins) 100,1,251,1025
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) 100,1 251,64
        # TODO expand 126 to 128
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)

        x = x.transpose(1, 3) 
        x = self.bn0(x)
        x = x.transpose(1, 3)  # 100,1,128,128

        if self.training:
            x = self.spec_augmenter(x)

        # if sampler:
        #     x = self.Pooling(x)

        return x

class AudioModel(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        # """

        clipwise_output = self.backbone(self.frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

# class AudioModel(nn.Module):
#     def __init__(self, backbone, sampler, **kwargs):
#         super().__init__(**kwargs)

#         self.backbone = backbone
#         self.sampler = sampler
#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         # """
#         if self.sampler is not None:
#             x = self.sampler(input)
#         clipwise_output = self.backbone(x)
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict

class AudioModel_Trasformer(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend
        # self.backone = load_mobilevit_weights(backbone, file_weight='/vol/research/Fish_tracking_master/FishMM/pretrained_models/xxsmodel_best.pth.tar')['state_dict']
        # self.backbone = backbone
        self.encoder = backbone
        old_pretrained_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/xxsmodel_best.pth.tar')['state_dict']
        # new_shape = torch.mean(old_pretrained_encoder['module.stem.0.weight'], dim=1)
        # new_shape = new_shape.unsqueeze(1)
        dict_new = self.encoder.state_dict().copy()
        # old_pretrained_encoder['module.stem.0.weight'] = new_shape
        pretrained_encoder = {k.replace('module.', ''): v for k, v in old_pretrained_encoder.items()}
        trained_list = [i for i in pretrained_encoder.keys() if not ('fc' in i or i.startswith('stem.0'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.encoder.load_state_dict(dict_new)

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        # """

        clipwise_output = self.encoder(self.frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict


class AudioModel_ownpretrained(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone
        self.audio_frontend = frontend
        old_pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/audio-visual_pretrainedmodel/audio_best.pt')['model_state_dict']
        dict_new = self.audio_encoder.state_dict().copy()
        dict_new_frontend = self.audio_frontend.state_dict().copy()
        pretrained_audio_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_audio_encoder.items()}
        pretrained_audio1_encoder = {k.replace('frontend.', ''): v for k, v in pretrained_audio_encoder.items()}
        pretrained_audio_encoder = {k:v for k, v in pretrained_audio1_encoder.items() if k in dict_new}
        pretrained_frontend = {k: v for k, v in pretrained_audio1_encoder.items() if k in dict_new_frontend}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        trained_list1 = [i for i in pretrained_frontend.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list1)):
            dict_new_frontend[trained_list1[i]] = pretrained_frontend[trained_list1[i]]
        self.audio_frontend.load_state_dict(dict_new_frontend)
        # self.att_linear = nn.Linear(1024, 512)
        self.slf_attn = MultiHeadAttention(n_head=1, d_model=1024, d_k=1024, d_v=1024, dropout=0.1)
        self.fusion_linear = nn.Linear(1024, 4)
        
        # pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/MobileNetV2.pth')['model']
        # dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        # # trained_list = [i for i in pretrained_audio_encoder.keys() if not ('head' in i or 'pos' in i)]
        # trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        # for i in range(len(trained_list)):
        #     dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        # self.audio_encoder.load_state_dict(dict_new)

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """
        clipwise_output, audio_embed = self.audio_encoder(self.audio_frontend(input))
        # audio_features = F.relu_(self.att_linear(audio_embed))
        audio_features = audio_embed
        dec_output, dec_slf_attn = self.slf_attn(
                audio_features, audio_features, audio_features, mask=None)
        audio = torch.mean(dec_output, 1)
        clipwise_output = self.fusion_linear(audio)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

class AudioModel_Cnn6(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, backbone, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        self.backbone = backbone
        pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))
        clipwise_output = self.backbone(output)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

class AudioModel_Panns6(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))

        return output

class AudioModel_pretrained(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone
        self.audio_frontend = frontend
        pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        for name, param in self.audio_encoder.named_parameters():
            # if "fc_audioset" not in name:
            param.requires_grad = False


    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        clipwise_output = self.audio_encoder(self.audio_frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict
    

class AudioModel_pre_Cnn10(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone
        self.audio_frontend = frontend
        pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/Cnn10.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)


    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        clipwise_output = self.audio_encoder(self.audio_frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict


class AudioModel_MV2(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        pretrained_audio_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/MV2/audio_best.pt')['model_state_dict']
        dict_new = self.audio_encoder.state_dict().copy()
        pretrained_encoder = {k.replace('backbone.', ''): v for k, v in pretrained_audio_encoder.items()}
        pretrained_encoder = {k.replace('frontend.', ''): v for k, v in pretrained_encoder.items()}
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))
        return output
