import torch
from torch import nn


def generalized_cross_entropy_binary(p, q=0.2):
    '''
        p.shape = batch_size * 1
    '''
    return torch.mean((1-torch.pow(p,q))/q)


def constrained_loss(pr, pt):

    return torch.mean(torch.log(1-pr*(1-pt)))


