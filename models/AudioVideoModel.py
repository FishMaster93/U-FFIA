import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.S3D import S3D, load_S3D_weight
from models.model_zoo.MobileNetV2 import load_MobileNetV2_weight
from transformer.SubLayers import MultiHeadAttention
from Transformer_tools.blocks.encoder_layer import EncoderLayer
import pdb


class Audio_video_Model(nn.Module):
    def __init__(self, audio_frontend, audio_backbone, video_backbone, classes_num, fusion_type, **kwargs):
        super().__init__(**kwargs)
        self.fusion_type = fusion_type
        self.num_class = classes_num
        self.video_encoder = video_backbone
        old_pretrained_video_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/audio-visual_pretrainedmodel/video_best.pt')['model_state_dict']
        dict_new = self.video_encoder.state_dict().copy()
        pretrained_video_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_video_encoder.items()}
        trained_list = [i for i in pretrained_video_encoder.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_video_encoder[trained_list[i]]
        self.video_encoder.load_state_dict(dict_new)

        self.audio_encoder = audio_backbone
        self.audio_frontend = audio_frontend
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

        self.att_linear = nn.Linear(1024, 512)

        if self.fusion_type == 'fc':
            # fc fusion of two clipwise_embed
            self.fusion_linear = nn.Linear(1024, self.num_class)

        elif self.fusion_type == 'atten':
            # attn fusion of two framewise_embed
            # n_head = 1
            # d_model = 512
            # d_k = 512
            # d_v = 512
            dropout = 0.1
            self.slf_attn = MultiHeadAttention(n_head=1, d_model=512, d_k=512, d_v=512, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=dropout)
            self.dec_attn = MultiHeadAttention(n_head=4, d_model=512, d_k=128, d_v=128, dropout=dropout)
            self.fusion_linear1 = nn.Linear(512, 4)

        elif self.fusion_type == 'MBT':
            n_head = 1
            d_model = 512
            n_layers = 4
            drop_prob = 0.1
            ffn_hidden = 2048
            self.Bottleneck = nn.Parameter(torch.randn(20, 2, 512))
            self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                      ffn_hidden=ffn_hidden,
                                                      n_head=n_head,
                                                      drop_prob=drop_prob)
                                         for _ in range(n_layers)])
            # self.slf_attn = MultiHeadAttention(n_head=1, d_model=512, d_k=512, d_v=512, dropout=0.1)
            # self.enc_attn = MultiHeadAttention(n_head=1, d_model=512, d_k=512, d_v=512, dropout=0.1)
            self.fusion_linear_B = nn.Linear(512, 4)

    def forward(self, audio, video):
        """
        Input: (batch_size, data_length)ave_precision
        """

        if self.fusion_type == 'fc':
            clipwise_output_video, video_embed = self.video_encoder(video)

            clipwise_output_audio, audio_embed = self.audio_encoder(self.audio_frontend(audio))

            # av = clipwise_output_video + clipwise_output_audio

            av = torch.cat((audio_embed, video_embed), dim =1)

            av = torch.mean(av, 1)
            clipwise_output = self.fusion_linear(av)
            output_dict = {'clipwise_output': clipwise_output}

        elif self.fusion_type == 'atten':
            # TODO feature fusion and Cls
            _, video_embed = self.video_encoder(video)
            _, audio_embed = self.audio_encoder(self.audio_frontend(audio))

            video_features = F.relu_(self.att_linear(video_embed))
            audio_features = F.relu_(self.att_linear(audio_embed))
            dec_output, dec_slf_attn = self.slf_attn(
                video_features, video_features, video_features, mask=None)
            enc_output, enc_slf_attn = self.slf_attn(
                audio_features, audio_features, audio_features, mask=None)
            dec_enc_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=None)
            # dec_enc_output, dec_enc_attn = self.enc_attn(
            #     video_features, audio_features, audio_features, mask=None)
            av = torch.mean(dec_enc_output, 1)
            clipwise_output = self.fusion_linear1(av)
            output_dict = {'clipwise_output': clipwise_output}

        elif self.fusion_type == 'fused-crossatt':
            # TODO mutual fusion
            _, video_embed = self.video_encoder(video)
            _, audio_embed = self.audio_encoder(self.audio_frontend(audio))
            av_embed = torch.cat((audio_embed, video_embed),dim=1)
            video_features = F.relu_(self.att_linear(video_embed))
            audio_features = F.relu_(self.att_linear(audio_embed))
            av_features = F.relu_(self.att_linear(av_embed))
            video_output, dec_video_attn = self.slf_attn(
                video_features, video_features, video_features, mask=None)
            audio_output, dec_audio_attn = self.slf_attn(
                audio_features, audio_features, audio_features, mask=None)
            av_output, dec_av_attn = self.slf_attn(
                av_features, av_features, av_features, mask=None)
            dec1_enc_output, dec1_enc_attn = self.enc_attn(
                video_output, av_output, av_output, mask=None)
            dec2_enc_output, dec2_enc_attn = self.enc_attn(
                audio_output, av_output, av_output, mask=None)
            dec_enc_output, dec_enc_attn = self.dec_attn(
                dec2_enc_output, dec1_enc_output, dec1_enc_output, mask=None)

            dec_enc_output = torch.mean(dec_enc_output, 1)
            clipwise_output = self.fusion_linear1(dec_enc_output)
            output_dict = {'clipwise_output': clipwise_output}
        elif self.fusion_type == 'MBT':
            _, video_embed = self.video_encoder(video)
            _, audio_embed = self.audio_encoder(self.audio_frontend(audio))
            video_embedding = self.att_linear(video_embed)  # video_embedding[20, 36, 512]
            audio_embedding = self.att_linear(audio_embed)  # audio_embedding[20, 6, 512]
            fused_video_embedding = torch.cat((video_embedding, self.Bottleneck), dim=1)  # size[20, 40, 512]
            for layer in self.layers:
                video_output = layer(fused_video_embedding, s_mask=None)
            Bottleneck_fused = video_output[:, 36:, :]  # size[20, 6, 512]
            video_fused = video_output[:, :36, :]
            fused_audio_embedding = torch.cat((audio_embedding, Bottleneck_fused), dim=1)
            for layer in self.layers:
                output = layer(fused_audio_embedding, s_mask=None)  # [20, 10, 512]
            Bottleneck_fused = output[:, 6:, :] 
            audio_fused = output[:, :6, :]
            # dec_output, dec_slf_attn = self.slf_attn(
            #     video_embedding, self.Bottleneck, self.Bottleneck, mask=None)
            # Bottleneck_fused = dec_output[:, 36:, :]  # size[10, 6, 512]
            # dec_enc_output, dec_enc_attn = self.enc_attn(
            #     dec_output, Bottleneck_fused, Bottleneck_fused, mask=None)
            fused_embedding = torch.cat((audio_fused, video_fused), dim=1)
            output = torch.mean(fused_embedding, 1)
            clipwise_output = self.fusion_linear_B(output)
            output_dict = {'clipwise_output': clipwise_output}

        return output_dict


