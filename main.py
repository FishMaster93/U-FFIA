from dataset.fish_audio_dataset import get_dataloader
# from dataset.audio_dataset import get_dataloader
import warnings
import os
from models.model_zoo.BC_ResNet import BC_ResNet
warnings.filterwarnings("ignore")
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from warmup_scheduler import GradualWarmupScheduler
from utils.warmupCosineScheduler import WarmupCosineScheduler
import torch
from models.model_zoo.AST import ASTModel
from models.model_zoo.Vit import ViT
from models.model_zoo.vit_for_small_dataset import small_data_ViT
# from models.model_zoo.revised_mobilenet import Cnn14_mobilev2
from models.model_zoo.Cnn14_mobilev2 import Cnn14_mobilev2
from models.model_zoo.mcv2 import patch_cmv2
# from models.model_zoo.CBAM_mobilenet import Cnn14_mobilev2
from models.model_zoo.MobileVitV1 import MobileViT_XXS
# from models.model_zoo.Cnn6 import Cnn6
from models.model_zoo.models import MobileNetV1, ResNet18, ResNet22, ResNet38, Cnn10, Cnn14, Cnn4, Cnn6, Wavegram_Cnn14
from models.Audio_model import AudioModel, Audio_Frontend, AudioModel_pretrained, AudioModel_ownpretrained, AudioModel_Trasformer, AudioModel_Cnn6, AudioModel_pre_Cnn10
from models.model_zoo.MobileNetV2 import MobileNetV2
from models.model_zoo.MobileNetV3 import MobileNetV3_Small
import time
import logging as log_config
import argparse
from tasks.audio_task import trainer
from omegaconf import OmegaConf
from models.model_zoo.ConvMixer import ConvMixer
from models.Pooling import Pooling_layer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/audio/pre_exp.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    workspace = config['Workspace']
    exp_name = config['Exp_name']
    modality = config['Modality']
    model_type = config['Model']
    # model_type_pre = config['Model_pretrained']

    Training = config['Training']
    audio_features = config['Audio_features']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']
    # embed = Training['embed']
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
    frontend = Audio_Frontend(**audio_features)

    Model = eval(model_type)
    backbone = Model()
    """if you want to use pretrained model please change the AudioModel_Cnn6 function"""
    model = AudioModel_pre_Cnn10(frontend=frontend, backbone=backbone)

    model = model.to(device)

    train_loader = get_dataloader(split='train', batch_size=batch_size, sample_rate=sample_rate, seed=seed, drop_last=True)
    test_loader = get_dataloader(split='test', batch_size=batch_size, sample_rate=sample_rate, seed=seed, drop_last=True)
    val_loader = get_dataloader(split='val', batch_size=batch_size, sample_rate=sample_rate, seed=seed, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    logger.info(config)
    logger.info(model)
    logger.info(f"{modality} modality experiments running on {device}")
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir)