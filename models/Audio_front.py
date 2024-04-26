import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from models.model_zoo.modules import init_bn
from transformer.SubLayers import MultiHeadAttention
import torch.nn.functional as F


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

        x = x.transpose(1, 3)  # 100,64,251,1
        x = self.bn0(x)
        x = x.transpose(1, 3)  # 100,1,251,64

        if self.training:
            x = self.spec_augmenter(x)

        return x