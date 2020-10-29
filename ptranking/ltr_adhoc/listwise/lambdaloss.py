#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
The implementation of ranking models proposed in the following paper, which is inspired by https://github.com/allegro/allRank

@inproceedings{
author = {Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc},
title = {The LambdaLoss Framework for Ranking Metric Optimization},
year = {2018},
url = {https://doi.org/10.1145/3269206.3271784},
booktitle = {Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
pages = {1313–1322},
}
"""

import torch
from itertools import product

from ptranking.base.ranker import NeuralRanker
from ptranking.eval.parameter import ModelParameter
from ptranking.metric.adhoc_metric import torch_dcg_at_k
from ptranking.ltr_global import global_gpu as gpu, global_device as device, tensor, epsilon


LAMBDALOSS_TYPE = ['NDCG_Loss1', 'NDCG_Loss2', 'NDCG_Loss2++'] # todo add 'ARP_Loss1', 'ARP_Loss2',


#def arp_loss1_power_weights(batch_preds=None, batch_stds=None):

#def arp_loss2_power_weights(batch_preds=None, batch_stds=None):

def ndcg_loss1_power_weights(batch_n_gains=None, discounts=None):
    return batch_n_gains/discounts

def ndcg_loss2_power_weights(batch_n_gains=None, discounts=None):
    ranks = torch.arange(batch_n_gains.size(1)).type(tensor) + 1.0
    abs_rank_deltas = torch.abs(ranks[:, None] - ranks[None, :]).type(torch.LongTensor)
    #delta_ij = torch.abs(torch.pow(discounts[0, abs_rank_deltas - 1], -1.) - torch.pow(discounts[0, abs_rank_deltas], -1.))
    delta_ij = torch.abs(torch.pow(discounts[abs_rank_deltas - 1], -1.) - torch.pow(discounts[abs_rank_deltas], -1.))
    delta_ij.diagonal().zero_()

    power_weights = delta_ij[None, :, :] * torch.abs(batch_n_gains[:, :, None] - batch_n_gains[:, None, :])
    return power_weights

def ndcg_loss2plusplus_power_weights(batch_n_gains=None, discounts=None, mu=5.):
    #
    rho_ij = torch.abs(torch.pow(discounts[:, None], -1.) - torch.pow(discounts[None, :], -1.))

    ranks = torch.arange(batch_n_gains.size(1)).type(tensor) + 1.0
    abs_rank_deltas = torch.abs(ranks[:, None] - ranks[None, :]).type(torch.LongTensor)
    delta_ij = torch.abs(torch.pow(discounts[abs_rank_deltas - 1], -1.) - torch.pow(discounts[abs_rank_deltas], -1.))
    delta_ij.diagonal().zero_()

    power_weights = (rho_ij + mu * delta_ij) * torch.abs(batch_n_gains[:, :, None] - batch_n_gains[:, None, :])
    return power_weights


class LambdaLoss(NeuralRanker):
    '''
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None):
        super(LambdaLoss, self).__init__(id='LambdaLoss', sf_para_dict=sf_para_dict)
        self.lambdaloss_dict = model_para_dict
        self.multi_level_rele = False if model_para_dict['std_rele_is_permutation'] else True
        self.k, self.sigma, self.loss_type = model_para_dict['k'], model_para_dict['sigma'], model_para_dict['loss_type']
        if 'NDCG_Loss2++' == self.loss_type: self.mu = model_para_dict['mu']

    def inner_train(self, batch_preds, batch_labels, **kwargs):
        '''
        per-query training process
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_stds_sorted_via_preds = torch.gather(batch_labels, dim=1, index=batch_preds_sorted_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_ranks = torch.arange(batch_preds.size(1)).type(tensor)
        dists_1D = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients

        # assuming that batch_labels is pre-sorted, i.e., presort=True for efficiency
        batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_labels)

        if self.multi_level_rele:
            batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
        else:
            batch_gains = batch_stds_sorted_via_preds

        batch_n_gains = batch_gains / batch_idcgs  # normalised gains

        if 'NDCG_Loss1' == self.loss_type:
            power_weights = ndcg_loss1_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D)
        elif 'NDCG_Loss2' == self.loss_type:
            power_weights = ndcg_loss2_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D)
        elif 'NDCG_Loss2++' == self.loss_type:
            power_weights = ndcg_loss2plusplus_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D, mu=self.mu)

        batch_pred_diffs = (torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)).clamp(min=-1e8, max=1e8)  # computing pairwise differences, i.e., s_i - s_j
        batch_pred_diffs[torch.isnan(batch_pred_diffs)] = 0.

        weighted_probas = (torch.sigmoid(self.sigma * batch_pred_diffs).clamp(min=epsilon) ** power_weights).clamp(min=epsilon)
        log_weighted_probas = torch.log2(weighted_probas)

        # mask for truncation based on cutoff k
        trunc_mask = torch.zeros((batch_preds.shape[1], batch_preds.shape[1]), dtype=torch.bool, device=device)
        trunc_mask[:self.k, :self.k] = 1

        if self.loss_type in ['NDCG_Loss2', 'NDCG_Loss2++']:
            batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
            padded_pairs_mask = batch_std_diffs>0
            padded_log_weighted_probas = log_weighted_probas [padded_pairs_mask & trunc_mask]
        else:
            padded_log_weighted_probas = log_weighted_probas [trunc_mask[None, :, :]]

        batch_loss = -torch.sum(padded_log_weighted_probas)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

###### Parameter of LambdaLoss ######

class LambdaLossParameter(ModelParameter):
    ''' Parameter class for LambdaLoss '''
    def __init__(self, debug=False, std_rele_is_permutation=False):
        super(LambdaLossParameter, self).__init__(model_id='LambdaLoss')
        self.debug = debug
        self.std_rele_is_permutation = std_rele_is_permutation

    def default_para_dict(self):
        """
        Default parameter setting for LambdaLoss
        :return:
        """
        self.lambdaloss_para_dict = dict(model_id=self.model_id, std_rele_is_permutation=self.std_rele_is_permutation,
                                         loss_type='NDCG_Loss2++', sigma=1.0, k=5, mu=5.0)
        return self.lambdaloss_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        lambdaloss_para_dict = given_para_dict if given_para_dict is not None else self.lambdaloss_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        if 'NDCG_Loss2++' == lambdaloss_para_dict['loss_type']:
            lambdaloss_paras_str = s1.join([lambdaloss_para_dict['loss_type'], 'Sigma', '{:,g}'.format(lambdaloss_para_dict['sigma']),
                 'Mu', '{:,g}'.format(lambdaloss_para_dict['mu'])])
            return lambdaloss_paras_str
        else:
            lambdaloss_paras_str = s1.join(
                [lambdaloss_para_dict['loss_type'], 'Sigma', '{:,g}'.format(lambdaloss_para_dict['sigma'])])
            return lambdaloss_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaLoss
        :param debug:
        :return:
        """
        choice_loss_type = ['NDCG_Loss2'] if self.debug else ['NDCG_Loss2']  #
        choice_sigma = [1.0] if self.debug else [1.0]  #
        choice_mu = [5.0] if self.debug else [5.0]  #
        choice_k = [5] if self.debug else [5]

        for loss_type, sigma, k in product(choice_loss_type, choice_sigma, choice_k):
            if 'NDCG_Loss2++' == loss_type:
                for mu in choice_mu:
                    self.lambdaloss_para_dict = dict(model_id='LambdaLoss',
                                                     std_rele_is_permutation=self.std_rele_is_permutation,
                                                     sigma=sigma, loss_type=loss_type, mu=mu, k=k)
                    yield self.lambdaloss_para_dict
            else:
                self.lambdaloss_para_dict = dict(model_id='LambdaLoss',
                                                 std_rele_is_permutation=self.std_rele_is_permutation,
                                                 sigma=sigma, loss_type=loss_type, k=k)
                yield self.lambdaloss_para_dict
