#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch

from org.archive.base.ranker import NeuralRanker
from org.archive.metric.adhoc_metric import torch_ideal_dcg

from org.dl.virtual.indicator.virtual_indicator_family import robust_sigmoid

from org.archive.l2r_global import global_gpu as gpu


def get_approx_ranks(input, alpha=10):
    ''' get approximated rank positions: Equation-11 in the paper'''
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)  # computing pairwise differences, i.e., Sij or Sxy

    batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), alpha) # using {-1.0*} may lead to a poor performance when compared with the above way;

    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

    return batch_hat_pis


def approxNDCG(batch_preds=None, batch_stds=None, alpha=10):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha)

    ''' since the input standard labels are sorted in advance, thus directly used '''
    # sorted_labels, _ = torch.sort(batch_stds, dim=1, descending=True)  # for optimal ltr_adhoc based on standard labels
    batch_idcgs = torch_ideal_dcg(batch_sorted_labels=batch_stds, gpu=gpu)  # ideal dcg given standard labels

    batch_gains = torch.pow(2.0, batch_stds) - 1.0

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    return batch_approx_nDCG


def approxNDCG_loss(batch_preds=None, batch_stds=None, alpha=10):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha)

    ''' since the input standard labels are sorted in advance, thus directly used '''
    # sorted_labels, _ = torch.sort(batch_stds, dim=1, descending=True)  # for optimal ltr_adhoc based on standard labels
    batch_idcgs = torch_ideal_dcg(batch_sorted_labels=batch_stds, gpu=gpu)  # ideal dcg given standard labels

    batch_gains = torch.pow(2.0, batch_stds) - 1.0

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    batch_loss = -torch.mean(batch_approx_nDCG)
    return batch_loss




class AppoxNDCG(NeuralRanker):
    '''
    Tao Qin, Tie-Yan Liu, and Hang Li. 2010.
    A general approximation framework for direct optimization of information retrieval measures.
    Journal of Information Retrieval 13, 4 (2010), 375–397.
    '''

    def __init__(self, sf_para_dict=None, apxNDCG_para_dict=None):
        super(AppoxNDCG, self).__init__(id='AppoxNDCG', sf_para_dict=sf_para_dict)
        self.alpha = apxNDCG_para_dict['alpha']

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        batch_loss = approxNDCG_loss(batch_preds, batch_stds, self.alpha)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


def get_apxndcg_paras_str(model_para_dict, log=False):
    s1 = ':' if log else '_'
    apxNDCG_paras_str = s1.join(['Alpha', str(model_para_dict['alpha'])])

    return apxNDCG_paras_str
