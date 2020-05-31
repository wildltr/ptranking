#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/03/06 | https://y-research.github.io

"""Description

"""


import torch
import torch.nn.functional as F

from org.archive.base.ranker import NeuralRanker

from org.archive.l2r_global import global_gpu as gpu, global_device as device

EPS = 1e-20

class STListNet(NeuralRanker):
    '''
    author = {Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc},
    title = {A Stochastic Treatment of Learning to Rank Scoring Functions},
    year = {2020},
    booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
    pages = {61–69},
    '''

    def __init__(self, sf_para_dict=None, temperature=1.0):
        super(STListNet, self).__init__(id='STListNet', sf_para_dict=sf_para_dict)
        self.temperature = temperature

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        unif = torch.rand(batch_preds.size())  # [num_samples_per_query, ranking_size]
        if gpu: unif = unif.to(device)

        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

        batch_preds = (batch_preds + gumbel) / self.temperature

        # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
        # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        batch_loss = torch.sum(-torch.sum(F.softmax(batch_stds, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss