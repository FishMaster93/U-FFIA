import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.losses import get_loss_func
import os
import logging as log_config
import numpy as np
from utils.unified_evaluate import Evaluator
from utils.early_stopping import save_model
from tqdm import tqdm


def trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir, modality_dropout, audio_dropout, self.Pre_AudioModel, self.Pre_VideoModel):
    logger = log_config.getLogger()
    logger.info("Starting new training run")
    loss_func = get_loss_func('clip_ce')
    evaluator = Evaluator(model=model)
    best_acc_audio = 0
    best_acc_video = 0
    best_acc_av = 0
    best_epoch = 0
    temp = 2.5
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    # change the probabilities of audio and video
    modality_dropout = 0.5
    audio_dropout = 0.5
    for epoch in range(max_epoch):
        mean_loss = 0
        for data_dict in tqdm(train_loader):
            data_dict['video_form'] = data_dict['video_form'].to(device)
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)
            modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()  #return[0.0, 1.0)
            model.train()
            if modality_drop_prob < modality_dropout:
                if audio_drop_prob < audio_dropout:
                    logger.info("Now is using only video modality")
                    video_output = model(None, data_dict['video_form'], 'v')
                    Pre_video_output = self.Pre_VideoModel(data_dict['video_form_all'])
                    KL_video_loss = soft_loss(
                        F.log_softmax(video_output['clipwise_output'], dim=1),
                        F.softmax(Pre_video_output['clipwise_output']/temp, dim=1)
                    )
                    target_dict = {'target': data_dict['target']}
                    loss = 0.5*loss_func(video_output, target_dict) + 0.5*KL_video_loss
                else:
                    logger.info("Now is using only audio modality")
                    audio_output = model(data_dict['waveform'], None, 'a')
                    Pre_audio_output = self.Pre_AudioModel(data_dict['waveform'])
                    KL_audio_loss = soft_loss(
                        F.log_softmax(audio_output['clipwise_output'], dim=1),
                        F.softmax(Pre_audio_output['clipwise_output']/temp, dim=1)
                    )
                    target_dict = {'target': data_dict['target']}
                    loss = 0.5*loss_func(audio_output, target_dict) + 0.5*KL_audio_loss
            else:
                logger.info("Now is using audio-video modality")
                av_output = model(data_dict['waveform'], data_dict['video_form'], 'av')
                target_dict = {'target': data_dict['target']}
                loss = loss_func(av_output, target_dict)
  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            mean_loss += loss
        epoch_loss = mean_loss / len(train_loader)
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")

        if epoch % 1 == 0:
            model.eval()
            # Evaluate on audio modality
            val_statistics_audio = evaluator.evaluate_audio(val_loader)
            val_mAP_audio = np.mean(val_statistics_audio['average_precision'])
            val_acc_audio = np.mean(val_statistics_audio['accuracy'])

            # Evaluate on video modality
            val_statistics_video = evaluator.evaluate_video(val_loader)
            val_mAP_video = np.mean(val_statistics_video['average_precision'])
            val_acc_video = np.mean(val_statistics_video['accuracy'])

            # Evaluator on audio-video modality
            val_statistics_av = evaluator.evaluate_av(val_loader)
            val_mAP_av = np.mean(val_statistics_av['average_precision'])
            val_acc_av = np.mean(val_statistics_av['accuracy'])

            if val_acc_video > best_acc_video:
                best_epoch_video = epoch
                best_acc_video = val_acc_video
                best_mAP_video = val_mAP_video
                save_model(os.path.join(ckpt_dir, 'video_best.pt'), model, optimizer, best_acc_video, best_epoch_video)

            if val_acc_audio > best_acc_audio:
                best_epoch_audio = epoch
                best_acc_audio = val_acc_audio
                best_mAP_audio = val_mAP_audio
                save_model(os.path.join(ckpt_dir, 'audio_best.pt'), model, optimizer, best_acc_audio, best_epoch_audio)

            if val_acc_av > best_acc_av:
                best_epoch_av = epoch
                best_acc_av = val_acc_av
                best_mAP_av = val_mAP_av
                save_model(os.path.join(ckpt_dir, 'av_best.pt'), model, optimizer, best_acc_av, best_epoch_av)

            model.train()
        logger.info(f'val_best_acc_audio: {best_acc_audio}, best mAP_audio:{best_mAP_audio}, best_epoch: {best_epoch_audio}')
        logger.info(f'val_best_acc_video: {best_acc_video}, best mAP_video:{best_mAP_video}, best_epoch: {best_epoch_video}')
        logger.info(f'val_best_acc_av: {best_acc_av}, best mAP_av:{best_mAP_av}, best_epoch: {best_epoch_av}')

    # Test evaluate
    logger.info('Evaluate on the Test dataset')
    # Evaluate on audio-video modality
    logger.info('Evaluate on the audio-video dataset')
    model_av_path = os.path.join(ckpt_dir, 'av_best.pt')
    model.load_state_dict(torch.load(model_av_path)['model_state_dict'])
    model.eval()
    test_statistics_av = evaluator.evaluate_av(test_loader)
    ave_precision_av = np.mean(test_statistics_av['average_precision'])
    ave_acc_av = np.mean(test_statistics_av['accuracy'])
    logger.info(f' test_dataset mAP: {ave_precision_av}, accuracy: {ave_acc_av}')

    # Evaluate on audio modality
    logger.info('Evaluate on the audio dataset')
    model_audio_path = os.path.join(ckpt_dir, 'audio_best.pt')
    model.load_state_dict(torch.load(model_audio_path)['model_state_dict'])
    test_statistics_a = evaluator.evaluate_audio(test_loader)
    ave_precision_a = np.mean(test_statistics_a['average_precision'])
    ave_acc_a = np.mean(test_statistics_a['accuracy'])
    logger.info(f' test_dataset mAP: {ave_precision_a}, accuracy: {ave_acc_a}')

    # Evaluate on video modality
    logger.info('Evaluate on the video dataset')
    model_video_path = os.path.join(ckpt_dir, 'video_best.pt')
    model.load_state_dict(torch.load(model_video_path)['model_state_dict'])
    test_statistics_v = evaluator.evaluate_video(test_loader)
    ave_precision_v = np.mean(test_statistics_v['average_precision'])
    ave_acc_v = np.mean(test_statistics_v['accuracy'])
    logger.info(f' test_dataset mAP: {ave_precision_v}, accuracy: {ave_acc_v}')