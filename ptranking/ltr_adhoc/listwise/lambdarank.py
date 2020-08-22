#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch
import torch.nn.functional as F

from ptranking.ltr_global import torch_zero
from ptranking.base.ranker import NeuralRanker
from ptranking.eval.parameter import ModelParameter
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.ltr_adhoc.util.gather_utils import torch_triu_indice

def lambdaRank_loss_diagonal(batch_preds=None, batch_stds=None, sigma=None):
    '''
    This method will impose explicit bias to highly ranked documents that are essentially ties
    :param batch_preds:
    :param batch_stds:
    :return:
    '''

    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    # get unique document pairs, which is dynamically different per training iteration
    pair_row_inds, pair_col_inds = torch_triu_indice(k=1, pair_type='NoTies', batch_labels=batch_stds_sorted_via_preds)

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}
    batch_std_Sij = batch_std_Sij[:, pair_row_inds, pair_col_inds]  # necessary S_{ij}

    batch_pred_diffs = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j
    batch_pred_s_ij = batch_pred_diffs[:, pair_row_inds, pair_col_inds] # unique pairwise comparisons according to a ltr_adhoc of documents

    batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds)
    batch_delta_ndcg = batch_delta_ndcg[:, pair_row_inds, pair_col_inds]

    batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij) # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_s_ij) + 1.0)  # cf. the 1st equation in page-3

    batch_loss = torch.sum((batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg)    # weighting with delta-nDCG

    return batch_loss


def lambdaRank_loss_full(batch_preds=None, batch_stds=None, sigma=None, multi_level_rele=None):
    '''
    Instead of strictly getting the uppper diagonal entries, here we compute the lambdaloss by fully making use of the properties as follows:
    (1) using the full pairwise difference matrix is twice the loss of using merely the uppper diagonal entries
    (2) for ties, the delta nDCG will be zero, thus no need to explicitly remove pairs of ties
    '''
    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}

    batch_pred_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j

    batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds, multi_level_rele=multi_level_rele)

    batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij) # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_s_ij) + 1.0)  # cf. the 1st equation in page-3

    # the coefficient of 0.5 is added due to all pairs are used
    batch_loss = torch.sum(0.5 * (batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg)    # weighting with delta-nDCG

    return batch_loss


def lambdaRank_loss_full_soft(batch_preds=None, batch_stds=None, sigma=None):
    '''
    Instead of strictly getting the uppper diagonal entries, here we compute the lambdaloss by fully making use of the properties as follows:
    (1) using the full pairwise difference matrix is twice the loss of using merely the uppper diagonal entries
    (2) for ties, the delta nDCG will be zero, thus no need to explicitly remove pairs of ties
    '''

    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}

    batch_pred_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j

    batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds)

    batch_loss = torch.sum(sigma * (F.softplus(batch_pred_s_ij, beta=sigma) - batch_std_Sij * batch_pred_s_ij) * batch_delta_ndcg)

    return batch_loss


###### Function - LambdaRank ######

def log_1_add_exp_minus_sigma(x, sigma=1.0):
    res = torch.where(x>0, torch.log(1.0+torch.exp(-sigma*x)), torch_zero)
    res = torch.where(x<0, torch.log(1.0+torch.exp(sigma*x)) - sigma*x, res)
    return res

def reciprocal_1_add_exp_sigma(x, sigma=1.0):
    res = torch.where(x<0, 1.0/(1.0 + torch.exp(sigma*x)), torch_zero)
    res = torch.where(x>0, 1.0-1.0/(1.0+torch.exp(-sigma*x)), res)
    return res

class LambdaRank_OP(torch.autograd.Function):
    """ Aiming at mannual gradient computation """

    @staticmethod
    def forward(ctx, batch_preds, batch_stds, sigma):
        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)   # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij   = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}

        batch_pred_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j

        batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds)

        batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij)  # cf. the 1st equation in page-3
        batch_loss_2nd = log_1_add_exp_minus_sigma(batch_pred_s_ij, sigma=sigma)  # cf. the 1st equation in page-3

        batch_loss = torch.sum((batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg * 0.5)  # weighting with delta-nDCG, '0.5' is multiplied due to the symmetric property

        #- gradient -#
        batch_grad = sigma * (0.5 * (1 - batch_std_Sij) - reciprocal_1_add_exp_sigma(batch_pred_s_ij, sigma=sigma))

        batch_grad = batch_grad * batch_delta_ndcg
        batch_grad = torch.sum(batch_grad, dim=1, keepdim=True) # relying on the symmetric property, i-th row-sum corresponding to the cumulative gradient w.r.t. i-th document.
        ctx.save_for_backward(batch_grad)

        return batch_loss

    @staticmethod
    def backward(ctx, grad_output):
        batch_grad = ctx.saved_tensors[0]
        target_gradients = grad_output*batch_grad
        # it is a must that keeping the same number w.r.t. the input of forward function
        return target_gradients, None, None

apply_LambdaRank_OP = LambdaRank_OP.apply

###### LambdaRank Ranker ######

class LambdaRank(NeuralRanker):
    '''
    Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
    Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None):
        super(LambdaRank, self).__init__(id='LambdaRank', sf_para_dict=sf_para_dict)
        self.sigma = model_para_dict['sigma']
        self.loss_version = model_para_dict['loss_version']
        # todo revise all loss functions correspondingly [now only full is done]
        self.multi_level_rele = False if model_para_dict['std_rele_is_permutation'] else True

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        if 'FullSoft' == self.loss_version: # ''' softplus version '''
            batch_loss = lambdaRank_loss_full_soft(batch_preds, batch_stds, sigma=self.sigma)

        elif 'Full' == self.loss_version:
            '''
            The following comparison between full and diagonal shows that: (1) the performance is almost the same; (2) explicit extracting indices in diagonal implementation is quite time-comsuming.
            Thus lambdaRank_loss_full is used as the default.
            Elapsed time: 0:04:04.392835 LambdaRank 2-fold cross validation scores: nDCG@1:0.4855, nDCG@3:0.4911, nDCG@5:0.5028, nDCG@10:0.5330, nDCG@20:0.5987, nDCG@50:0.0158
            '''
            batch_loss = lambdaRank_loss_full(batch_preds, batch_stds, sigma=self.sigma,
                                              multi_level_rele=self.multi_level_rele)

        elif 'Diag' == self.loss_version:
            # Elapsed time: 0:06:19.067998 LambdaRank 2-fold cross validation scores: nDCG@1:0.4849, nDCG@3:0.4909, nDCG@5:0.5028, nDCG@10:0.5328, nDCG@20:0.5985, nDCG@50:0.0158
            batch_loss = lambdaRank_loss_diagonal(batch_preds, batch_stds, sigma=self.sigma)

        elif 'OP' == self.loss_version:
            batch_loss = apply_LambdaRank_OP(batch_preds, batch_stds, self.sigma, self.pair, self.focal)

        else:
            raise NotImplementedError

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of LambdaRank ######

class LambdaRankParameter(ModelParameter):
    ''' Parameter class for LambdaRank '''
    def __init__(self, debug=False, std_rele_is_permutation=False):
        super(LambdaRankParameter, self).__init__(model_id='LambdaRank')
        self.debug = debug
        self.std_rele_is_permutation = std_rele_is_permutation

    def default_para_dict(self):
        """
        Default parameter setting for LambdaRank
        :return:
        """
        self.lambda_para_dict = dict(model_id=self.model_id, sigma=1.0, loss_version='Full',
                                     std_rele_is_permutation=self.std_rele_is_permutation)
        return self.lambda_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        lambda_para_dict = given_para_dict if given_para_dict is not None else self.lambda_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        lambdarank_para_str = s1.join([lambda_para_dict['loss_version'], 'Sigma',
                                    '{:,g}'.format(lambda_para_dict['sigma'])])
        return lambdarank_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        :param debug:
        :return:
        """
        plus_choice_sigma = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0
        for sigma in plus_choice_sigma:
            self.lambda_para_dict = dict(model_id=self.model_id, sigma=sigma, loss_version='Full',
                                         std_rele_is_permutation=self.std_rele_is_permutation)
            yield self.lambda_para_dict
