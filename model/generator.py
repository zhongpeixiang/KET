import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import print_dims

class Generator(nn.Module):
    """
    A standard linear + softmax generation step
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return F.log_softmax(self.proj(x), dim=1)