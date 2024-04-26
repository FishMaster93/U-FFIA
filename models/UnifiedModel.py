import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Unified import Unified_model
from models.Audio_model import AudioModel_Panns6
from models.Audio_model import AudioModel_MV2
from models.model_zoo.Cnn6 import Cnn6
from models.model_zoo.revised_mobilenet import Cnn14_mobilev2


class Unified_Model(nn.Module):
    def __init__(self, frontend, classes_num, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend
        self.classes_num = classes_num
        self.unified_model = Unified_model(
            image_size=224,
            # audio_size=128,
            image_patch_size=16,
            frames=1,
            frame_patch_size=1,
            num_classes=4,
            dim=768,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.MV2 = Cnn14_mobilev2()
        self.AudioModel_MV2 = AudioModel_MV2(frontend_pre=self.frontend, backbone_pre=self.MV2)
        # self.Cnn6 = Cnn6()
        # self.AudioModel_Cnn6 = AudioModel_Panns6(frontend_pre=self.frontend, backbone_pre=self.Cnn6)

    def forward(self, audio, video, modality):
        """
        Input: (batch_size, data_length)
        """
        # audio: (20, 128000)  video: (20, 8, 3, 224, 224)
        if modality == 'a':
            # audio_input = self.frontend(audio)  # [10, 1, 128, 128]
            audio_features = self.AudioModel_MV2(audio)
            out = self.unified_model(audio_features, None, 'a')
        elif modality == 'v':
            # video_input = video.transpose(2, 1)
            out = self.unified_model(None, video, 'v')
        elif modality == 'av':
            # audio_input = self.frontend(audio)  # [10, 1, 128, 128]
            audio_features = self.AudioModel_MV2(audio)
            # video_input = video.transpose(2, 1)
            out = self.unified_model(audio_features,video, 'av')
        clipwise_output = out
        output_dict = {'clipwise_output': clipwise_output}
        return  output_dict





