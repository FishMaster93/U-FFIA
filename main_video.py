from re import I
from dataset.fish_video_dataset import get_dataloader
import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
from models.Video_model import VideoModel
from models.model_zoo.ResNet3D import generate_model
from models.model_zoo.S3D import S3D
import os
import time
import logging as log_config
import argparse
from tasks.video_task import trainer
from omegaconf import OmegaConf
from models.model_zoo.vit3D import ViT3D
from models.model_zoo.vivit import ViViT
from models.model_zoo.Vit import ViT
from models.model_zoo.MobileVitV1 import MobileViT_XXS



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/vision/exp3_video.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    modality = config['Modality']
    Model = config['Model']
    model_type = Model['name']

    Training = config['Training']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']

    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
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

    Model = eval(model_type)
    # backbone = Model(classes_num=classes_num)
    backbone = Model(  
       image_size=224,
        image_patch_size=16,
        frames=2,
        frame_patch_size=1,
        num_classes=4,
        dim=768,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1)
        
    # backbone = Model(  
    #     image_size=224,
    #     patch_size=16,
    #     num_classes=4,
    #     dim=768,
    #     depth=6,
    #     heads=8,
    #     mlp_dim=1024,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    model = VideoModel(backbone=backbone)
    # model = VideoModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    train_loader = get_dataloader(split='train', batch_size=batch_size, seed=seed, epoch=0, num_workers=8)
    test_loader = get_dataloader(split='test', batch_size=batch_size, seed=seed, epoch=0, num_workers=8)
    val_loader = get_dataloader(split='val', batch_size=batch_size, seed=seed, epoch=0, num_workers=8)
    logger.info(config)
    logger.info(model)
    logger.info(f"{modality} modality experiments running on {device}")
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir)
