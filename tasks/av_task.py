import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import torch.nn as nn
from utils.losses import get_loss_func
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging as log_config
import numpy as np
from utils.evaluate import Evaluator
from models.model_zoo.S3D import load_S3D_weight
import argparse
from utils.early_stopping import save_model
from tqdm import tqdm
from utils.early_stopping import save_model


def trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir, image_dir):
    logger = log_config.getLogger()
    logger.info("Starting new training run")
    loss_func = get_loss_func('clip_ce')
    evaluator = Evaluator(model=model)
    best_acc = 0
    best_epoch = 0

    for epoch in range(max_epoch):
        mean_loss = 0
        # train_loader.dataset.epoch = epoch
        # val_loader.dataset.epoch = epoch
        # test_loader.dataset.epoch = epoch
        for data_dict in tqdm(train_loader):
            data_dict['video_form'] = data_dict['video_form'].to(device)
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)
            model.train()

            output_dict1 = model(data_dict['waveform'], data_dict['video_form'])
            # output_dict2 = model(data_dict['waveform'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            target_dict = {'target': data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
            loss = loss_func(output_dict1, target_dict)
            # loss2 = loss_func(output_dict2, target_dict)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            mean_loss += loss
        epoch_loss = mean_loss / len(train_loader)
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")
        if epoch % 1 == 0:
            model.eval()
            val_statistics = evaluator.evaluate_av(val_loader)
            val_cm = val_statistics['confu_matrix']
            val_mAP = np.mean(val_statistics['average_precision'])
            val_acc = np.mean(val_statistics['accuracy'])
            val_message = val_statistics['message']
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_mAP = val_mAP
                best_cm = val_cm
                save_model(os.path.join(ckpt_dir, 'atten8_best.pt'), model, optimizer, val_acc, best_epoch)
            model.train()
        logger.info(f'val_best_acc: {best_acc}, best mAP:{best_mAP}, best_epoch: {best_epoch}')

    # Test evaluate
    logger.info('Evaluate on the Test dataset')
    model_path = os.path.join(ckpt_dir, 'atten8_best.pt')
    # model_path = '/mnt/fast/nobackup/users/mc02229/FishMM/Fish_workspace/exp1_av_126/save_models/av_best.pt'
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    test_statistics = evaluator.evaluate_av(test_loader)
    test_cm = test_statistics['confu_matrix']
    ConfusionMatrixDisplay(test_cm).plot()
    plt.title("confusion_matrix_test")
    plt.savefig(os.path.join(image_dir,'test-{}.png'.format(epoch)))
    ave_precision = np.mean(test_statistics['average_precision'])
    ave_acc = np.mean(test_statistics['accuracy'])
    logger.info(f' test_dataset mAP: {ave_precision}, accuracy: {ave_acc}')
    # logger.info(f' confusion_matrix: \n {test_cm}')
     

