import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import clones
from .constants import print_dims

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product self attention
        query: (batch_size, h, seq_len, d_k), seq_len can be either src_seq_len or tgt_seq_len
        key: (batch_size, h, seq_len, d_k), seq_len in key, value and mask are the same
        value: (batch_size, h, seq_len, d_k)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) (legacy)
    """
    if print_dims:
        print("{0}: query: type: {1}, shape: {2}".format("attention func", query.type(), query.shape))
        print("{0}: key: type: {1}, shape: {2}".format("attention func", key.type(), key.shape))
        print("{0}: value: type: {1}, shape: {2}".format("attention func", value.type(), value.shape))
        print("{0}: mask: type: {1}, shape: {2}".format("attention func", mask.type(), mask.shape))
    d_k = query.size(-1)

    # scores: (batch_size, h, seq_len, seq_len) for self_attn, (batch_size, h, tgt_seq_len, src_seq_len) for src_attn
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k)) 
    # print(query.shape, key.shape, mask.shape, scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        if print_dims:
            print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
            print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
            print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k)
        if print_dims:
            print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x) # (batch_size, seq_len, d_model)
        if print_dims:
            print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x


class GraphAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, GAW=None, variant=0, concept_embed=None):
        super(GraphAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.variant = variant
        if concept_embed is None:
            self.concept_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.concept_embed = concept_embed

    def init_params(self, GAW=None, edge_matrix=None, affectiveness=None, concentration_factor=1):
        self.GAW = GAW
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=0)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range==0).float()) # normalization
        self.edge_matrix = edge_matrix
        self.affectiveness = affectiveness
        self.concentration_factor = concentration_factor
        if self.GAW is not None:
            self._lambda = self.GAW
        else:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), 0.5))


    def get_hierarchical_sentence_representation(self, sent_embed):
        # sent_embed: (batch_size, seq_len, d_model)
        seq_len = sent_embed.shape[1]
        N = 3 # n-gram hierarchical pooling
        
        # max pooling for each ngram
        ngram_embeddings = [[] for i in range(N-1)] # one list for each n
        for n in range(1, N):
            for i in range(seq_len):
                ngram_embeddings[n-1].append(sent_embed[:,i:i+n+1,:].max(dim=1)[0])
        
        # mean pooling across ngram embeddings
        pooled_ngram_embeddings = [sent_embed.mean(dim=1)] # unigram
        for ngram_embedding in ngram_embeddings:
            ngram_embedding = torch.stack(ngram_embedding, dim=1).mean(dim=1)
            pooled_ngram_embeddings.append(ngram_embedding)

        sent_embed = torch.stack(pooled_ngram_embeddings, dim=1).mean(dim=1)
        return sent_embed

    def get_context_representation(self, src_embed, tgt_embed):
        # src_embed: (batch_size, context_length*seq_len, d_model)
        # tgt_embed: (batch_size, seq_len, d_model)
        seq_len = tgt_embed.shape[1]
        context_length = src_embed.shape[1]//seq_len
        sentence_representations = []
        for i in range(context_length):
            sentence_representations.append(self.get_hierarchical_sentence_representation(src_embed[:, i*seq_len:(i+1)*seq_len]))
        sentence_representations.append(self.get_hierarchical_sentence_representation(tgt_embed))
        context_representation = torch.stack(sentence_representations, dim=1).mean(dim=1) # (batch_size, d_model)
        return context_representation

    def forward(self, src, src_embed, tgt, tgt_embed):
        # src: (batch_size, context_length * seq_len)
        # src_embed: (batch_size, context_length*seq_len, d_model)
        # tgt: (batch_size, seq_len)
        # tgt_embed: (batch_size, seq_len, d_model)
        # embed: shared embedding layer: (vocab_size, d_model)

        # get context representation
        if self.variant == 1:
            # standard attention as in GAT
            src_len = src.shape[1]
            src = torch.cat([src, tgt], dim=1)
            src_embed = torch.cat([src_embed, tgt_embed], dim=1) # (batch_size, (context_length+1)*seq_len, d_model), self.concept_embed.weight: (vocab_size, d_model)
            concept_weights = (self.edge_matrix[src] > 0).float() * torch.matmul(src_embed, self.concept_embed.weight.transpose(0,1)) # (batch_size, (context_length+1)*seq_len, vocab_size)
            concept_embedding = torch.matmul(torch.softmax(concept_weights * self.concentration_factor, dim=2), self.concept_embed.weight)
            return concept_embedding[:, :src_len, :], concept_embedding[:, src_len:, :]
        if self.variant == 2:
            context_representation = self.get_context_representation(src_embed, tgt_embed) # (batch_size, d_model)
            # get concept embedding
            src_len = src.shape[1]
            src = torch.cat([src, tgt], dim=1)
            cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(1), \
                self.concept_embed.weight.unsqueeze(0), dim=2)) # (batch_size, vocab_size)
            relatedness = self.edge_matrix[src] * cosine_similarity.unsqueeze(1) # (batch_size, (context_length+1)*seq_len, vocab_size)
            concept_weights = self._lambda*relatedness + (1-self._lambda)*(self.edge_matrix[src] > 0).float()*self.affectiveness # (batch_size, (context_length+1)*seq_len, vocab_size)
            concept_embedding = torch.matmul(torch.softmax(concept_weights * self.concentration_factor, dim=2), self.concept_embed.weight) # (batch_size, (context_length+1)*seq_len, d_model)
            return concept_embedding[:, :src_len, :], concept_embedding[:, src_len:, :]

    

