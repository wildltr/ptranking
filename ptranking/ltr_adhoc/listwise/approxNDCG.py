#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Tao Qin, Tie-Yan Liu, and Hang Li. 2010.
A general approximation framework for direct optimization of information retrieval measures.
Journal of Information Retrieval 13, 4 (2010), 375–397.
"""

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.metric.adhoc.adhoc_metric import torch_dcg_at_k
from ptranking.base.utils import robust_sigmoid


def get_approx_ranks(input, alpha=10, device=None):
    ''' get approximated rank positions: Equation-11 in the paper'''
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)  # computing pairwise differences, i.e., Sij or Sxy

    batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), alpha, device) # using {-1.0*} may lead to a poor performance when compared with the above way;

    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

    return batch_hat_pis


def approxNDCG(batch_preds=None, batch_stds=None, alpha=10, label_type=None, device=None):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha, device=device)

    ''' since the input standard labels are sorted in advance, thus directly used '''
    # sorted_labels, _ = torch.sort(batch_stds, dim=1, descending=True)  # for optimal ltr_adhoc based on standard labels
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None, label_type=label_type, device=device)  # ideal dcg given standard labels

    batch_gains = torch.pow(2.0, batch_stds) - 1.0

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    return batch_approx_nDCG


def approxNDCG_loss(batch_preds=None, batch_ideal_rankings=None, alpha=10, label_type=None, device=None):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha, device=device)

    # ideal dcg given optimally ordered labels
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, cutoff=None, label_type=label_type, device=device)

    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_ideal_rankings) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_ideal_rankings
    else:
        raise NotImplementedError

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    batch_loss = -torch.sum(batch_approx_nDCG)
    return batch_loss




class ApproxNDCG(AdhocNeuralRanker):
    '''
    Tao Qin, Tie-Yan Liu, and Hang Li. 2010.
    A general approximation framework for direct optimization of information retrieval measures.
    Journal of Information Retrieval 13, 4 (2010), 375–397.
    '''

    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(ApproxNDCG, self).__init__(id='ApproxNDCG', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.alpha = model_para_dict['alpha']

    def uniform_eval_setting(self, **kwargs):
        eval_dict = kwargs['eval_dict']
        if eval_dict["do_validation"] and not eval_dict['vali_metric']=='nDCG':
            eval_dict['vali_metric'] = "nDCG"

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

        '''
        Given the ideal rankings, the optimization objective is to maximize the approximated nDCG based on differentiable rank positions
        '''
        batch_loss = approxNDCG_loss(batch_preds=target_batch_preds, batch_ideal_rankings=batch_ideal_rankings,
                                     alpha=self.alpha, label_type=label_type, device=self.device)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

#-------
def get_apxndcg_paras_str(model_para_dict, log=False):
    s1 = ':' if log else '_'
    apxNDCG_paras_str = s1.join(['Alpha', str(model_para_dict['alpha'])])

    return apxNDCG_paras_str

###### Parameter of ApproxNDCG ######

class ApproxNDCGParameter(ModelParameter):
    ''' Parameter class for ApproxNDCG '''
    def __init__(self, debug=False, para_json=None):
        super(ApproxNDCGParameter, self).__init__(model_id='ApproxNDCG', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for ApproxNDCG
        :return:
        """
        self.apxNDCG_para_dict = dict(model_id=self.model_id, alpha=10.)
        return self.apxNDCG_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        apxNDCG_para_dict = given_para_dict if given_para_dict is not None else self.apxNDCG_para_dict

        s1 = ':' if log else '_'
        apxNDCG_paras_str = s1.join(['Alpha', str(apxNDCG_para_dict['alpha'])])
        return apxNDCG_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for ApproxNDCG
        """
        if self.use_json:
            choice_alpha = self.json_dict['alpha']
        else:
            choice_alpha = [10.0] if self.debug else [10.0]  # 1.0, 10.0, 50.0, 100.0

        for alpha in choice_alpha:
            self.apxNDCG_para_dict = dict(model_id=self.model_id, alpha=alpha)
            yield self.apxNDCG_para_dict
