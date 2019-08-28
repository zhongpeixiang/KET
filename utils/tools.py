
from collections import Counter
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def counter_to_distribution(counter):
    distribution = {}
    total = sum([v for k, v in counter.items()])
    for k, v in counter.items():
        distribution[k] = round(v/total, 4)
    return distribution


def label_distribution_transformer(examples):
    counter = Counter([ex[-1] for ex in examples])
    percentatges = []
    for i in range(len(counter)):
        percentatges.append(counter[i]/len(examples))
    return percentatges
