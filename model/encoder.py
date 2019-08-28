import torch
import torch.nn as nn
from .modules import clones, LayerNorm, SublayerConnection
from .constants import print_dims

class Encoder(nn.Module):
    "Encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
            x: (batch_size, src_seq_len, d_model)
            mask: (batch_size, 1, src_seq_len)
        """
        "Pass the input token ids and mask through each layer in turn"
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward layers"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        """norm -> self_attn -> dropout -> add -> 
        norm -> feed_forward -> dropout -> add"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
