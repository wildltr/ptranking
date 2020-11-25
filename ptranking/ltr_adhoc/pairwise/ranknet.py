#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
"""

import torch
import torch.nn.functional as F

from ptranking.base.ranker import NeuralRanker

class RankNet(NeuralRanker):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''
    def __init__(self, sf_para_dict=None, gpu=False, device=None):
        super(RankNet, self).__init__(id='RankNet', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.sigma = 1.0

    def inner_train(self, batch_pred, batch_label, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_label:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        batch_s_ij = torch.unsqueeze(batch_pred, dim=2) - torch.unsqueeze(batch_pred, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)

        batch_std_diffs = torch.unsqueeze(batch_label, dim=2) - torch.unsqueeze(batch_label, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

        # about reduction, both mean & sum would work, mean seems straightforward due to the fact that the number of pairs differs from query to query
        batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='mean')

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss
