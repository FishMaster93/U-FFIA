import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
from models.Audio_model import Audio_Frontend
from models.model_zoo.MobileNetV2 import MobileNetV2
from models.AudioVideoModel import Audio_video_Model
from models.model_zoo.S3D import S3D
from dataset.fish_av_dataset import get_dataloader
from models.AudioVideoModel import Audio_video_Model
import os
import time
import logging as log_config
import argparse
from tasks.av_task import trainer
from omegaconf import OmegaConf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/audiovisual/exp2_av.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    modality = config['Modality']
    Model = config['Model']
    Training = config['Training']
    audio_features = config['Audio_features']
    audio_type = Model['audio_name']
    video_type = Model['video_name']

    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']
    sample_rate = audio_features['sample_rate']

    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    image_dir = os.path.join(workspace, exp_name, 'images')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir,
                                                '%s-%d.log' % (exp_name, time.time()))),
            log_config.StreamHandler()
        ]
    )

    logger = log_config.getLogger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    frontend = Audio_Frontend(**audio_features)
    Model_audio = eval(audio_type)
    audio_backbone = Model_audio(classes_num=classes_num)

    Model_video = eval(video_type)
    video_backbone = Model_video(classes_num=classes_num)

    model = Audio_video_Model(audio_frontend=frontend, audio_backbone=audio_backbone, video_backbone=video_backbone,
                              classes_num=classes_num, fusion_type='MBT')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    train_loader = get_dataloader(split='train', batch_size=batch_size, seed=seed, epoch=0,sample_rate=sample_rate, num_workers=8, drop_last=True)
    test_loader = get_dataloader(split='test', batch_size=batch_size, seed=seed, epoch=0,
                                 sample_rate=sample_rate, num_workers=8)
    val_loader = get_dataloader(split='val', batch_size=batch_size, seed=seed, epoch=0,
                                sample_rate=sample_rate, num_workers=8)
    logger.info(config)
    logger.info(model)
    logger.info(f"{modality} modality experiments running on {device}")
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir, image_dir)
