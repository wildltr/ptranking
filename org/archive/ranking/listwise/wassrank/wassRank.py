#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/12/02 | https://y-research.github.io

"""Description

"""

import torch

from org.archive.ranker.ranker import AbstractNeuralRanker

from org.archive.ranking.listwise.wassrank.wasserstein_cost_mat import get_cost_mat, get_normalized_distributions
from org.archive.ranking.listwise.wassrank.wasserstein_loss_layer import Y_WassersteinLossVanilla, Y_WassersteinLossStab, tor_sinkhorn, tor_sinkhorn_stabilized
wasserstein_distance_vanilla = Y_WassersteinLossVanilla.apply
wasserstein_distance = Y_WassersteinLossStab.apply

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

























class WassRank(AbstractNeuralRanker):
    '''
    Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory.
    Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 2019.2.
    '''

    def __init__(self, ranking_function, wass_para_dict=None, dict_cost_mats=None, dict_std_dists=None):
        super(WassRank, self).__init__(id='WassRank', ranking_function=ranking_function)

        self.TL_AF = self.ranking_function.get_tl_af()
        self.wass_para_dict = wass_para_dict
        if dict_cost_mats is not None:
            self.dict_cost_mats = dict_cost_mats
        if dict_std_dists is not None:
            self.dict_std_dists = dict_std_dists

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        qid = kwargs['qid']
        if qid in self.dict_cost_mats:
            tor_cost_mat = self.dict_cost_mats[qid]  # using buffered cost matrices to avoid re-computation
            #if debug: cost_mat = tor_cost_mat.data.numpy()
        else:
            tor_cost_mat, cost_mat = get_cost_mat(kwargs['cpu_tor_batch_std_label_vec'], wass_para_dict=self.wass_para_dict)
            if gpu: tor_cost_mat = tor_cost_mat.to(device)
            self.dict_cost_mats[qid] = tor_cost_mat

        tor_batch_std_dist, dists_pred = get_normalized_distributions(tor_batch_std_label_vec=batch_stds, tor_batch_prediction=batch_preds,
            wass_dict_std_dists=self.dict_std_dists, qid=qid, wass_para_dict=self.wass_para_dict, TL_AF=self.TL_AF)

        wass_mode, sh_itr, lam = self.wass_para_dict['mode'], self.wass_para_dict['sh_itr'], self.wass_para_dict['lam']
        if wass_mode == 'WassLossSta':
            batch_loss, = wasserstein_distance(dists_pred, tor_batch_std_dist.to(device), tor_cost_mat, lam, sh_itr)

        elif wass_mode == 'WassLoss':
            batch_loss, = wasserstein_distance_vanilla(dists_pred, tor_batch_std_dist.to(device), tor_cost_mat, lam, sh_itr)

        elif wass_mode == 'Tor_WassLossSta':
            batch_ot = tor_sinkhorn_stabilized(a=torch.squeeze(dists_pred), b=torch.squeeze(tor_batch_std_dist), M=tor_cost_mat, reg=lam, numItermax=sh_itr)
            batch_loss = torch.sum(batch_ot * tor_cost_mat)
        # print('batch_loss', batch_loss)
        elif wass_mode == 'Tor_WassLoss':
            batch_ot = tor_sinkhorn(a=torch.squeeze(dists_pred), b=torch.squeeze(tor_batch_std_dist), M=tor_cost_mat, reg=lam, numItermax=sh_itr)
            batch_loss = torch.sum(batch_ot * tor_cost_mat)
        else:
            raise NotImplementedError

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss