import warnings

warnings.filterwarnings("ignore")
# import decord
# decord.bridge.set_bridge("torch")
# import dtk.transforms as dtf
import random
import glob
import cv2
import time
# from utils.video_transform import RandomHorizontalFlipVideo, CenterCropVideo
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from itertools import chain


def get_video_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    path = '/mnt/fast/nobackup/users/mc02229/Fish_av_dataset/video_dataset'
    video = []
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path, dir))
        for dir1 in l2:
            video_dir = os.path.join(path, dir, dir1, split, '*.mp4')
            video.append(glob.glob(video_dir))
    return list(chain.from_iterable(video))


import pickle


def save_pickle(obj, fname):
    # print("Save pickle at " + fname)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    # print("Load pickle at " + fname)
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res


def data_generator(seed, test_sample_per_class):
    """
    class to label mapping:
    none: 0
    strong: 1
    middle: 2
    weak: 3
    """
    random_state = np.random.RandomState(seed)
    strong_v_list = get_video_name(split='strong')
    medium_v_list = get_video_name(split='medium')
    weak_v_list = get_video_name(split='weak')
    none_v_list = get_video_name(split='none')

    random_state.shuffle(strong_v_list)
    random_state.shuffle(medium_v_list)
    random_state.shuffle(weak_v_list)
    random_state.shuffle(none_v_list)

    strong_v_train = strong_v_list[2*test_sample_per_class:]
    medium_v_train = medium_v_list[2*test_sample_per_class:]
    weak_v_train = weak_v_list[2*test_sample_per_class:]
    none_v_train = none_v_list[2*test_sample_per_class:]

    strong_v_test = strong_v_list[:test_sample_per_class]
    medium_v_test = medium_v_list[:test_sample_per_class]
    weak_v_test = weak_v_list[:test_sample_per_class]
    none_v_test = none_v_list[:test_sample_per_class]

    strong_v_val = strong_v_list[test_sample_per_class:2 * test_sample_per_class]
    medium_v_val = medium_v_list[test_sample_per_class:2 * test_sample_per_class]
    weak_v_val = weak_v_list[test_sample_per_class:2 * test_sample_per_class]
    none_v_val = none_v_list[test_sample_per_class:2 * test_sample_per_class]

    train_v_dict = []
    test_v_dict = []
    val_v_dict = []

    for wav in strong_v_train:
        train_v_dict.append([wav, 1])

    for wav in medium_v_train:
        train_v_dict.append([wav, 2])

    for wav in weak_v_train:
        train_v_dict.append([wav, 3])

    for wav in none_v_train:
        train_v_dict.append([wav, 0])

    for wav in strong_v_test:
        test_v_dict.append([wav, 1])

    for wav in medium_v_test:
        test_v_dict.append([wav, 2])

    for wav in weak_v_test:
        test_v_dict.append([wav, 3])

    for wav in none_v_test:
        test_v_dict.append([wav, 0])

    for wav in strong_v_val:
        val_v_dict.append([wav, 1])

    for wav in medium_v_val:
        val_v_dict.append([wav, 2])

    for wav in weak_v_val:
        val_v_dict.append([wav, 3])

    for wav in none_v_val:
        val_v_dict.append([wav, 0])

    random_state.shuffle(train_v_dict)
    random_state.shuffle(test_v_dict)
    random_state.shuffle(val_v_dict)

    return train_v_dict, test_v_dict, val_v_dict


def gaussian(image, mean, std):
    image = image
    t, h, w, c = image.shape
    noise = torch.randn([t, h, w, c])*std + mean
    new_noise = noise+image
    return new_noise


class Fish_Video_Dataset(Dataset):
    def __init__(self, seed, epoch, split):
        """
        split: train or test
        if sample_rate=None, read audio with the default sr
        """
        self.seed = seed
        self.split = split
        train_v_dict, test_v_dict, val_v_dict = data_generator(self.seed, test_sample_per_class=700)
        if self.split == 'train':
            self.data_dict = train_v_dict
        elif self.split == 'test':
            self.data_dict = test_v_dict
        elif self.split == 'val':
            self.data_dict = val_v_dict
        self.epoch = epoch

        # transform_chain = [dtf.ToTensorVideo()]
        # transform_chain += [dtf.NormalizeVideo([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        # transform_chain += [CenterCropVideo()]
        # transform_chain += [RandomHorizontalFlipVideo(0.5)]

        # TODO add data augmentation from dtk
        # self.video_transform = transforms.Compose(transform_chain)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        # save_path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/mc0229/datasets/Fish_dataset'
        save_path = '/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_video_dataset/Fish_video_1/'
        #################################################
        # video_name, target = self.data_dict[index]
        # target = np.eye(4)[target]
        # # random sample one frame from video
        # vr = decord.VideoReader(video_name, height=196, width=196)
        # full_vid_length = len(vr)
        # video_frames = vr.get_batch(range(0, full_vid_length))

        # X = np.arange(0, full_vid_length - 1)
        # Y = sorted(np.random.choice(X, 4, replace=False))
        # vf = video_frames[Y, ...]
        # vf = self.video_transform(vf)
        # #################################################
        # data_dict = {'video_name': video_name, 'video_form': vf, 'target': target}
        # os.makedirs(os.path.join(save_path, str(self.epoch), self.split), exist_ok=True)
        # save_pickle(data_dict, os.path.join(save_path, str(self.epoch), self.split, '%s.pkl' % index))

        #################################################
        pickle_path = os.path.join(save_path, str(self.epoch), self.split, '%s.pkl' % index)
        data_dict = load_pickle(pickle_path)

        return data_dict


# def collate_fn(batch):
#     target = [data['target'] for data in batch]
#     video_name = [data['video_name'] for data in batch]
#     vf = torch.FloatTensor([np.array(data['video_form']) for data in batch])

#     target = torch.FloatTensor(np.array(target))

#     data_dict = {'video_name': video_name, 'video_form': vf, 'target': target}
#     return data_dict
def collate_fn(batch):
    target = [data['target'] for data in batch]
    video_name = [data['video_name'] for data in batch]
    vf = torch.stack([data['video_form'] for data in batch])
    target = torch.FloatTensor(np.array(target))
    # vf = vf.squeeze(1)
    vf = vf.permute(0, 2, 1, 3, 4)
    data_dict = {'video_name': video_name, 'video_form': vf, 'target': target}

    return data_dict

def get_dataloader(split,
                   batch_size,
                   seed,
                   epoch,
                   shuffle=False,
                   drop_last=False,
                   num_workers=8):
    dataset = Fish_Video_Dataset(split=split, seed=seed, epoch=epoch)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':   
    # for i in range(10):
    #     train_loader = get_dataloader(split='train', batch_size=8, seed=20, num_workers=4, epoch=i)

    # test_loader = get_dataloader(split='test', batch_size=2, seed=20, num_workers=4)
    # val_loader = get_dataloader(split='val', batch_size=2, seed=20, num_workers=4)
    from tqdm import tqdm
    for i in range(5):
        train_loader = get_dataloader(split='train', batch_size=8, seed=25, num_workers=8, epoch=i)
        for item in tqdm(train_loader):
            pass
        #
        print("Start save test_loader")
        test_loader = get_dataloader(split='test', batch_size=8, seed=25, num_workers=8, epoch=i)
        for item in tqdm(test_loader):
            pass

        print("Start save val_loader")
        val_loader = get_dataloader(split='val', batch_size=8, seed=25, num_workers=8, epoch=i)
        for item in tqdm(val_loader):
            pass
