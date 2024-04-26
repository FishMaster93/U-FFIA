import matplotlib.pyplot as plt
from dataset.fish_audio_dataset import get_dataloader
import torch.optim as optim
import torch
import torch.nn as nn
from utils.losses import get_loss_func
import os
import time
import logging as log_config
import numpy as np
from utils.evaluate import Evaluator
import argparse
from utils.early_stopping import save_model
from tqdm import tqdm
import wandb
# wandb.init(project = 'ASTlog-project')
# wandb.config = {
#     "learning_rate" : 0.001,
#     "epoch":400,
#     "batch_size":100

# }

def trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir):
    logger = log_config.getLogger()
    logger.info("Starting new training run")
    loss_func = get_loss_func('clip_ce')
    evaluator = Evaluator(model=model)
    best_acc = 0
    best_epoch = 0
    for epoch in range(max_epoch):
        mean_loss = 0
        test_loader.dataset.epoch = epoch
        for data_dict in tqdm(train_loader):
            # scheduler.step()
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)
            model.train()
            output_dict = model(data_dict['waveform'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            target_dict = {'target': data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
            import ipdb; ipdb.set_trace()
            loss = loss_func(output_dict, target_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            mean_loss += loss
            # wandb.log({"loss": loss})
        epoch_loss = mean_loss / len(train_loader)
        # wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        # train_statistics = evaluator.evaluate_audio(train_loader)
        # train_mAP = np.mean(train_statistics['average_precision'])
        # train_acc = np.mean(train_statistics['accuracy'])
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")
        # logger.info(f'Train mAP: {train_mAP}, accuracy: {train_acc}')

        if epoch % 1 == 0:
            model.eval()
            val_statistics = evaluator.evaluate_audio(val_loader)
            # val_cm = val_statistics['confu_matrix']
            val_mAP = np.mean(val_statistics['average_precision'])
            val_acc = np.mean(val_statistics['accuracy'])
            # wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})
            # wandb.log({"val_mAP": val_mAP, "epoch": epoch})
            # wandb.log({"val_acc": val_acc, "epoch": epoch})
            # val_message = val_statistics['message']
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_mAP = val_mAP
                # best_cm = val_cm
                # ConfusionMatrixDisplay(best_cm).plot()
                # plt.title("confusion_matrix")
                # fig_name = ckpt_dir + str(best_epoch) + '.png'
                # plt.savefig(fig_name)
                # best_message = val_message
                save_model(os.path.join(ckpt_dir, 'audio_best.pt'), model, optimizer, val_acc, best_epoch)
            model.train()
        logger.info(f' best mAP:{best_mAP}, val_best_acc: {best_acc}, best_epoch: {best_epoch}')
        
        # logger.info(f'Metrics report by val_class: {best_message}')
        # logger.info(f'confusion_matrix: {val_cm}')

    # Test evaluate
    logger.info('Evaluate on the Test dataset')
    model_path = os.path.join(ckpt_dir, 'audio_best.pt')
    # model_path = '/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/audio-visual_pretrainedmodel/audio_best1024-2.pt'
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    test_statistics = evaluator.evaluate_audio(test_loader)
    ave_precision = np.mean(test_statistics['average_precision'])
    ave_acc = np.mean(test_statistics['accuracy'])
    # message = test_statistics['message']
    logger.info(f' test_dataset mAP: {ave_precision}, accuracy: {ave_acc}')
    # logger.info(f'Metrics report by test_class: {message}')






