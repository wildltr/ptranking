#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 19/01/08 | https://y-research.github.io

"""Description

"""

import torch
from torch.nn import functional as F


EPS = 1e-20


def gumbel_softmax(logits, samples_per_query, temperature=1.0, cuda=False, cuda_device=None):
    '''

    :param logits: [1, ranking_size]
    :param num_samples_per_query: number of stochastic rankings to generate
    :param temperature:
    :return:
    '''
    assert 1 == logits.size(0) and 2 == len(logits.size())

    unif = torch.rand(samples_per_query, logits.size(1)) # [num_samples_per_query, ranking_size]
    if cuda: unif = unif.to(cuda_device)

    gumbel = -torch.log(-torch.log(unif + EPS) + EPS) # Sample from gumbel distribution

    logit = (logits + gumbel) / temperature

    y = F.softmax(logit, dim=1)

    # i.e., #return F.softmax(logit, dim=1)
    return y