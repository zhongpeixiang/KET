import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import clones, LayerNorm, SublayerConnection
from .constants import print_dims

class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, src_states, src_mask, tgt_mask):
        """
            x: (batch_size, tgt_seq_len, d_model)
            src_states: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, src_seq_len)
            tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: src_states: type: {1}, shape: {2}".format(self.__class__.__name__, src_states.type(), src_states.shape))
            print("{0}: src_mask: type: {1}, shape: {2}".format(self.__class__.__name__, src_mask.type(), src_mask.shape))
            print("{0}: tgt_mask: type: {1}, shape: {2}".format(self.__class__.__name__, tgt_mask.type(), tgt_mask.shape))
        for layer in self.layers:
            x = layer(x, src_states, src_mask, tgt_mask)
        x = self.norm(x) # (batch_size, tgt_seq_len, d_model)
        if print_dims:
            print("{0}: x (output): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        
        # add max pooling across sequences
        x = F.max_pool1d(x.permute(0,2,1), x.shape[1]).squeeze(-1) # (batch_size, d_model)
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn and feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, src_states, src_mask, tgt_mask):
        """norm -> self_attn -> dropout -> add -> 
        norm -> src_attn -> dropout -> add ->
        norm -> feed_forward -> dropout -> add"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, src_states, src_states, src_mask))
        return self.sublayer[2](x, self.feed_forward)

