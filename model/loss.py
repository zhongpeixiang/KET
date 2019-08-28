import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, classification_report

from .constants import print_dims

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        "Update parameter and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, 
                  torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: target: type: {1}, shape: {2}".format(self.__class__.__name__, target.type(), target.shape))
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        if print_dims:
            print("{0}: true_dist: type: {1}, shape: {2}".format(self.__class__.__name__, true_dist.type(), true_dist.shape))
        return self.criterion(x, true_dist)


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, dataset, emotion2id, opt=None, test=False):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.dataset = dataset
        self.emotion2id = emotion2id
        self.test = test
        self.outputs = []
        self.tgts = []

    @staticmethod
    def compute_score(predictions, ground, dataset, emotion2id, test=False):
        pred_y = predictions.astype(int)
        val_y = ground.astype(int)
        
        if dataset in ["EC"]:
            labels = [emotion2id["happy"], emotion2id["angry"], emotion2id["sad"]]
            score = f1_score(val_y, pred_y, average='micro', labels=labels)
            print("Micro-F1 (exclude neutral): {0}".format(score))
            if test:
                print(classification_report(val_y, pred_y, labels=labels, digits=4))
        elif dataset in ["DD"]:
            labels = [emotion2id[str(i)] for i in range(1, 7)]
            score = f1_score(val_y, pred_y, average='micro', labels=labels)
            print("Micro-F1 (exclude neutral): {0}".format(score))
            if test:
                print(classification_report(val_y, pred_y, labels=labels, digits=4))
        elif dataset in ["MELD", "IEMOCAP", "EmoryNLP"]:
            score = f1_score(val_y, pred_y, average='weighted')
            print("Weighted Macro-F1: {0}".format(score))
            if test:
                print(classification_report(val_y, pred_y, digits=4))
        else:
            score = mean_absolute_error(val_y, pred_y)
            print("MAE: {0}".format(score))
            if test:
                print(mean_absolute_error(val_y, pred_y))
        return score


    def score(self):
        score = self.compute_score(np.array(self.outputs), np.array(self.tgts), self.dataset, self.emotion2id, test=self.test)
        return score


    def clear(self):
        self.outputs = []
        self.tgts = []
        
    def __call__(self, x, y, norm):
        """
        x: (batch_size, tgt_seq_len, d_model)
        y: (batch_size, tgt_seq_len)
        norm: ()
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: y: type: {1}, shape: {2}".format(self.__class__.__name__, y.type(), y.shape))
            print("{0}: norm: type: {1}, shape: {2}".format(self.__class__.__name__, norm.type(), norm.shape))
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        self.outputs += x.contiguous().view(-1, x.size(-1)).argmax(dim=-1).tolist()
        self.tgts += y.contiguous().view(-1).tolist()
        if print_dims:
            print("{0}: loss: type: {1}, shape: {2}".format(self.__class__.__name__, loss.type(), loss.shape))
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            if isinstance(self.opt, NoamOpt):
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.item() * norm