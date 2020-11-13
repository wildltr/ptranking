#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
"""

import torch

from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.util.gather_utils import torch_triu_indice

def ranknet_loss(batch_pred=None, batch_label=None, sigma=1.0, pair_row_inds=None, pair_col_inds=None):
    '''
    RankNet's loss function, cf. {From RankNet to LambdaRank to LambdaMART: An Overview}
    :param batch_preds: [batch, ranking_size]
    :param batch_stds:  [batch, ranking_size]
    :return:
    '''
    batch_pred_diffs = torch.unsqueeze(batch_pred, dim=2) - torch.unsqueeze(batch_pred, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = batch_pred_diffs[:, pair_row_inds, pair_col_inds]

    batch_std_diffs = torch.unsqueeze(batch_label, dim=2) - torch.unsqueeze(batch_label, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = batch_Sij[:, pair_row_inds, pair_col_inds]

    batch_loss_1st = 0.5 * sigma * batch_s_ij * (1.0 - batch_Sij)     # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_s_ij) + 1.0)   # cf. the 1st equation in page-3
    batch_loss = torch.sum(batch_loss_1st + batch_loss_2nd)

    return batch_loss


class RankNet(NeuralRanker):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''
    def __init__(self, sf_para_dict=None, gpu=False, device=None):
        super(RankNet, self).__init__(id='RankNet', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.sigma = 1.0
        self.pair  = 'All'
        self.dict_indice = dict()  # buffering pair indice to avoid duplicate computation, which is refreshed per ranker

    def inner_train(self, batch_pred, batch_label, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_label:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        qid = kwargs['qid']
        if qid in self.dict_indice:
            pair_row_inds, pair_col_inds = self.dict_indice[qid]
        else:
            pair_row_inds, pair_col_inds = torch_triu_indice(k=1, pair_type=self.pair, batch_label=batch_label)
            self.dict_indice[qid] = [pair_row_inds, pair_col_inds]

        batch_loss = ranknet_loss(batch_pred, batch_label, sigma=self.sigma, pair_row_inds=pair_row_inds, pair_col_inds=pair_col_inds)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss
