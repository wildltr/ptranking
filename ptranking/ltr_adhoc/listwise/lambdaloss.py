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

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.metric.adhoc.adhoc_metric import torch_dcg_at_k
from ptranking.ltr_global import epsilon

LAMBDALOSS_TYPE = ['NDCG_Loss1', 'NDCG_Loss2', 'NDCG_Loss2++'] # todo add 'ARP_Loss1', 'ARP_Loss2',


#def arp_loss1_power_weights(batch_preds=None, batch_stds=None):

#def arp_loss2_power_weights(batch_preds=None, batch_stds=None):

def ndcg_loss1_power_weights(batch_n_gains=None, discounts=None):
    return batch_n_gains/discounts

def ndcg_loss2_power_weights(batch_n_gains=None, discounts=None, gpu=False):
    torch_arange = torch.arange(batch_n_gains.size(1)).type(torch.cuda.FloatTensor) if gpu else torch.arange(batch_n_gains.size(1)).type(torch.FloatTensor)
    ranks = torch_arange + 1.0
    abs_rank_deltas = torch.abs(ranks[:, None] - ranks[None, :]).type(torch.LongTensor)
    #delta_ij = torch.abs(torch.pow(discounts[0, abs_rank_deltas - 1], -1.) - torch.pow(discounts[0, abs_rank_deltas], -1.))
    delta_ij = torch.abs(torch.pow(discounts[abs_rank_deltas - 1], -1.) - torch.pow(discounts[abs_rank_deltas], -1.))
    delta_ij.diagonal().zero_()

    power_weights = delta_ij[None, :, :] * torch.abs(batch_n_gains[:, :, None] - batch_n_gains[:, None, :])
    return power_weights

def ndcg_loss2plusplus_power_weights(batch_n_gains=None, discounts=None, mu=5., gpu=False):
    #
    rho_ij = torch.abs(torch.pow(discounts[:, None], -1.) - torch.pow(discounts[None, :], -1.))

    torch_arange = torch.arange(batch_n_gains.size(1)).type(torch.cuda.FloatTensor) if gpu else torch.arange(batch_n_gains.size(1)).type(torch.FloatTensor)
    ranks = torch_arange + 1.0
    abs_rank_deltas = torch.abs(ranks[:, None] - ranks[None, :]).type(torch.LongTensor)
    delta_ij = torch.abs(torch.pow(discounts[abs_rank_deltas - 1], -1.) - torch.pow(discounts[abs_rank_deltas], -1.))
    delta_ij.diagonal().zero_()

    power_weights = (rho_ij + mu * delta_ij) * torch.abs(batch_n_gains[:, :, None] - batch_n_gains[:, None, :])
    return power_weights


class LambdaLoss(AdhocNeuralRanker):
    '''
    author = {Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc},
    title = {The LambdaLoss Framework for Ranking Metric Optimization},
    year = {2018},
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(LambdaLoss, self).__init__(id='LambdaLoss', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.lambdaloss_dict = model_para_dict
        self.k, self.sigma, self.loss_type = model_para_dict['k'], model_para_dict['sigma'], model_para_dict['loss_type']
        if 'NDCG_Loss2++' == self.loss_type: self.mu = model_para_dict['mu']

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        label_type = kwargs['label_type']
        assert label_type == LABEL_TYPE.MultiLabel

        if 'presort' in kwargs and kwargs['presort']:
            target_batch_preds, batch_ideal_rankings = batch_preds, batch_std_labels
        else:
            batch_ideal_rankings, batch_ideal_desc_inds = torch.sort(batch_std_labels, dim=1, descending=True)
            target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_ideal_desc_inds)

        batch_descending_preds, batch_pred_desc_inds = torch.sort(target_batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_predict_rankings = torch.gather(batch_ideal_rankings, dim=1, index=batch_pred_desc_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        #batch_std_ranks = torch.arange(target_batch_preds.size(1)).type(torch.cuda.FloatTensor) if self.gpu else torch.arange(target_batch_preds.size(1)).type(torch.FloatTensor)
        batch_std_ranks = torch.arange(target_batch_preds.size(1), dtype=torch.float, device=self.device)
        dists_1D = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients

        # ideal dcg values based on optimal order
        batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, device=self.device)

        if label_type == LABEL_TYPE.MultiLabel:
            batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0
        elif label_type == LABEL_TYPE.Permutation:
            batch_gains = batch_predict_rankings
        else:
            raise NotImplementedError

        batch_n_gains = batch_gains / batch_idcgs  # normalised gains

        if 'NDCG_Loss1' == self.loss_type:
            power_weights = ndcg_loss1_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D)
        elif 'NDCG_Loss2' == self.loss_type:
            power_weights = ndcg_loss2_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D)
        elif 'NDCG_Loss2++' == self.loss_type:
            power_weights = ndcg_loss2plusplus_power_weights(batch_n_gains=batch_n_gains, discounts=dists_1D, mu=self.mu)

        batch_pred_diffs = (torch.unsqueeze(batch_descending_preds, dim=2) - torch.unsqueeze(batch_descending_preds, dim=1)).clamp(min=-1e8, max=1e8)  # computing pairwise differences, i.e., s_i - s_j
        batch_pred_diffs[torch.isnan(batch_pred_diffs)] = 0.

        weighted_probas = (torch.sigmoid(self.sigma * batch_pred_diffs).clamp(min=epsilon) ** power_weights).clamp(min=epsilon)
        log_weighted_probas = torch.log2(weighted_probas)

        # mask for truncation based on cutoff k
        trunc_mask = torch.zeros((target_batch_preds.shape[1], target_batch_preds.shape[1]), dtype=torch.bool, device=self.device)
        trunc_mask[:self.k, :self.k] = 1

        if self.loss_type in ['NDCG_Loss2', 'NDCG_Loss2++']:
            batch_std_diffs = torch.unsqueeze(batch_predict_rankings, dim=2) - torch.unsqueeze(batch_predict_rankings, dim=1)  # standard pairwise differences, i.e., S_{ij}
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
    def __init__(self, debug=False, para_json=None):
        super(LambdaLossParameter, self).__init__(model_id='LambdaLoss', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for LambdaLoss
        :return:
        """
        self.lambdaloss_para_dict = dict(model_id=self.model_id, loss_type='NDCG_Loss2++', sigma=1.0, k=5, mu=5.0)
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
        if self.use_json:
            choice_k = self.json_dict['k']
            choice_mu = self.json_dict['mu']
            choice_sigma = self.json_dict['sigma']
            choice_loss_type = self.json_dict['loss_type']
        else:
            choice_loss_type = ['NDCG_Loss2'] if self.debug else ['NDCG_Loss2']  #
            choice_sigma = [1.0] if self.debug else [1.0]  #
            choice_mu = [5.0] if self.debug else [5.0]  #
            choice_k = [5] if self.debug else [5]

        for loss_type, sigma, k in product(choice_loss_type, choice_sigma, choice_k):
            if 'NDCG_Loss2++' == loss_type:
                for mu in choice_mu:
                    self.lambdaloss_para_dict = dict(model_id='LambdaLoss', sigma=sigma, loss_type=loss_type, mu=mu, k=k)
                    yield self.lambdaloss_para_dict
            else:
                self.lambdaloss_para_dict = dict(model_id='LambdaLoss', sigma=sigma, loss_type=loss_type, k=k)
                yield self.lambdaloss_para_dict
