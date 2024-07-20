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
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)
            model.train()
            output_dict = model(data_dict['waveform'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            target_dict = {'target': data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
            loss = loss_func(output_dict, target_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            mean_loss += loss
        epoch_loss = mean_loss / len(train_loader)
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")

        if epoch % 1 == 0:
            model.eval()
            val_statistics = evaluator.evaluate_audio(val_loader)
            val_mAP = np.mean(val_statistics['average_precision'])
            val_acc = np.mean(val_statistics['accuracy'])
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_mAP = val_mAP
                save_model(os.path.join(ckpt_dir, 'audio_best.pt'), model, optimizer, val_acc, best_epoch)
            model.train()
        logger.info(f' best mAP:{best_mAP}, val_best_acc: {best_acc}, best_epoch: {best_epoch}')
        

    # Test evaluate
    logger.info('Evaluate on the Test dataset')
    model_path = os.path.join(ckpt_dir, 'audio_best.pt')
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    test_statistics = evaluator.evaluate_audio(test_loader)
    ave_precision = np.mean(test_statistics['average_precision'])
    ave_acc = np.mean(test_statistics['accuracy'])
    # message = test_statistics['message']
    logger.info(f' test_dataset mAP: {ave_precision}, accuracy: {ave_acc}')






