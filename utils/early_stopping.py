import numpy as np
import torch
import logging



def save_model(path, model, optimizer, ave_precision, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ave_precision': ave_precision,
    }, path)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=30, verbose=False, delta=0, trace_func=logging):
        """
       Args:
           patience (int): How long to wait after last time validation loss improved.
                           Default: 30
           verbose (bool): If True, prints a message for each validation loss improvement.
                           Default: False
           delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
           path (str): Path for the checkpoint to be saved to.
                           Default: 'checkpoint.pt'
           trace_func (function): trace print function.
                           Default: print
       """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func.info

    def __call__(self, val_loss, model, optimizer, epoch, ckpt_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, ckpt_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, ckpt_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, ckpt_path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Test result increased ({-self.val_loss_min:.4f} --> {-val_loss:.4f}).  Saving model ...')

        save_model(path=ckpt_path, model=model, optimizer=optimizer, epoch=epoch)
        self.val_loss_min = val_loss