import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
from models.UnifiedModel import Unified_Model
from models.Audio_front import Audio_Frontend
from dataset.fish_av_dataset import get_dataloader
import os
import time
import logging as log_config
import argparse
from tasks.unified_tasks import trainer
from omegaconf import OmegaConf
from utils.warmupCosineScheduler import WarmupCosineScheduler
from torch.optim import AdamW


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/unified/exp1_av.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    modality = config['Modality']
    Training = config['Training']
    audio_features = config['Audio_features']

    modality_dropout = Training['modality_drop']
    audio_dropout = Training['audio_drop']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']
    sample_rate = audio_features['sample_rate']

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
    frontend = Audio_Frontend(**audio_features, sampler=True)
    # frontend = Audio_Frontend(**audio_features)
    model = Unified_Model(frontend=frontend, classes_num=classes_num)
    model = model.to(device)
    
    train_loader = get_dataloader(split='train', batch_size=batch_size, seed=seed, epoch=0,
                                  sample_rate=sample_rate, num_workers=8, drop_last=True)
    test_loader = get_dataloader(split='test', batch_size=batch_size, seed=seed, epoch=0,
                                 sample_rate=sample_rate, num_workers=8, drop_last=True)
    val_loader = get_dataloader(split='val', batch_size=batch_size, seed=seed, epoch=0,
                                sample_rate=sample_rate, num_workers=8, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # opt_cls = AdamW
    # optimizer = opt_cls(
    #     [
    #         {
    #             "name": "model",
    #             "params":model.parameters(),
    #             "lr": learning_rate,
    #         }
    #     ],
    #     weight_decay = 1e-5,
    #     betas = (0.9, 0.999),
    # )

    # scheduler = WarmupCosineScheduler(
    #     optimizer,
    #     warmup_epochs=5,
    #     num_epochs=max_epoch,
    #     iter_per_epoch=len(train_loader)
    # )
    logger.info(config)
    logger.info(model)
    logger.info(f"{modality} modality experiments running on {device}")
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir, modality_dropout, audio_dropout)
