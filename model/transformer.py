import copy
import torch
import torch.nn as nn

from .embedding import Embeddings, PositionalEncoding
from .attention import MultiHeadAttention, GraphAttention
from .modules import PositionWiseFeedForward
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, model_variant=1, graph_attention=None, concept_emb=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.model_variant = model_variant
        self.graph_attention = graph_attention
        self.concept_emb = concept_emb
        
        self.embed_size = self.src_embed[0].embedding.weight.shape[1]
        self.src_mlp = nn.Linear(2*self.embed_size, self.embed_size)

        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: (batch_size, context_length * seq_len)
        # src_mask: (batch_size, 1, context_length * seq_len)
        # tgt: (batch_size, seq_len)
        # tgt_mask: (batch_size, 1, seq_len)
        # concept: (batch_size, (context_length + 1) * 30)

        src_embed = self.src_embed(src) # src_embed: (batch_size, context_length * seq_len, d_model)
        tgt_embed = self.tgt_embed(tgt) # tgt_embed: (batch_size, seq_len, d_model)
        if self.graph_attention is not None:
            # graph attention to compute concept representations
            src_concept_embed, tgt_concept_embed = self.graph_attention(src, src_embed, tgt, tgt_embed)

            # concatenate concept representations with utterance embeddings
            src_embed = self.src_mlp(torch.cat([src_embed, src_concept_embed], dim=-1))
            tgt_embed = self.src_mlp(torch.cat([tgt_embed, tgt_concept_embed], dim=-1))
        
        if self.model_variant == 1:
            # self-attention
            src_states = self.encoder(src_embed, src_mask)
            # cross-attention
            return self.decoder(tgt_embed, src_states, src_mask, tgt_mask)
        if self.model_variant == 2:
            seq_len = tgt.shape[1]
            context_length = src.shape[1]//seq_len
            
            # utterance-level self-attention for each src sentence
            src_states = []
            for i in range(context_length):
                s, e = i*seq_len, (i+1)*seq_len
                src_states.append(self.encoder(src_embed[:,s:e,:], src_mask[:,:,s:e]))
            src_states = torch.cat(src_states, dim=1)
            
            # context-level self-attention
            src_states = self.encoder(src_states, src_mask)
            
            # cross-attention
            return self.decoder(tgt_embed, src_states, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, output_size=0, dropout=0.1, KB=False, \
        model_variant=1, context_length=0, graph_attention_variant=0):
    c = copy.deepcopy
    enc_attn = MultiHeadAttention(h, d_model)
    dec_attn = MultiHeadAttention(h, d_model)
    enc_dec_attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    emb = Embeddings(d_model, src_vocab) # share src and tgt embedding
    concept_embed = None
    graph_attention = None
    if KB:
        # dynamic graph
        graph_attention = GraphAttention(vocab_size=src_vocab, embed_size=d_model, variant=graph_attention_variant, concept_embed=concept_embed)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(enc_attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(dec_attn), c(enc_dec_attn), c(ff), dropout), N),
        nn.Sequential(emb, c(position)),
        nn.Sequential(emb, c(position)),
        Generator(d_model, output_size),
        model_variant=model_variant,
        graph_attention=graph_attention
    )

    return model






