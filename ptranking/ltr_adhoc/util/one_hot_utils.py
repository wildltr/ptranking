#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

def get_one_hot_reprs(batch_stds, gpu=False):
    """ Get one-hot representation of batch ground-truth labels """
    batch_size = batch_stds.size(0)
    hist_size = batch_stds.size(1)
    int_batch_stds = batch_stds.type(torch.cuda.LongTensor) if gpu else batch_stds.type(torch.LongTensor)

    hot_batch_stds = torch.cuda.FloatTensor(batch_size, hist_size, 3) if gpu else torch.FloatTensor(batch_size, hist_size, 3)
    hot_batch_stds.zero_()
    hot_batch_stds.scatter_(2, torch.unsqueeze(int_batch_stds, 2), 1)

    return hot_batch_stds
