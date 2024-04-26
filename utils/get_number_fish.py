import os
import glob
from itertools import chain
import numpy as np


def get_wav_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    path = '/vol/research/Fish_tracking_master/Fish_av_dataset/audio_dataset'

    audio =[]
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path, dir))
        for dir1 in l2:
            wav_dir = os.path.join(path, dir, dir1, split, '*.wav')
            audio.append(glob.glob(wav_dir))
    return list(chain.from_iterable(audio))


strong_list = get_wav_name(split='medium')

print(len(strong_list))
