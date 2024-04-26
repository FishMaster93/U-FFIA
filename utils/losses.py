import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])

def clip_ce(output_dict, target_dict):
    """ crossentropy loss.
    """
    return F.cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])

def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
