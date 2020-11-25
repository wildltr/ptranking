#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
"""

import json

import torch
import torch.nn.functional as F

from ptranking.base.ranker import NeuralRanker
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.ltr_adhoc.eval.parameter import ModelParameter

class LambdaRank(NeuralRanker):
    '''
    Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
    Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(LambdaRank, self).__init__(id='LambdaRank', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.sigma = model_para_dict['sigma']

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        '''
        label_type = kwargs['label_type']
        assert LABEL_TYPE.MultiLabel == label_type
        assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking

        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_std_Sij)

        batch_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j
        batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)

        batch_delta_ndcg = get_delta_ndcg(batch_ideally_sorted_stds=batch_stds, batch_stds_sorted_via_preds=batch_stds_sorted_via_preds, label_type=label_type, gpu=self.gpu)

        # about reduction, mean leads to poor performance, a probable reason is that the small values due to * lambda_weight * mean
        batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                            target=torch.triu(batch_std_p_ij, diagonal=1),
                                            weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='sum')
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of LambdaRank ######

class LambdaRankParameter(ModelParameter):
    ''' Parameter class for LambdaRank '''
    def __init__(self, debug=False, para_json=None):
        super(LambdaRankParameter, self).__init__(model_id='LambdaRank')
        self.debug = debug
        self.para_json = para_json

    def default_para_dict(self):
        """
        Default parameter setting for LambdaRank
        :return:
        """
        self.lambda_para_dict = dict(model_id=self.model_id, sigma=1.0)
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
        lambdarank_para_str = s1.join(['Sigma', '{:,g}'.format(lambda_para_dict['sigma'])])
        return lambdarank_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.para_json is not None:
            with open(self.para_json) as json_file:
                json_dict = json.load(json_file)
            choice_sigma = json_dict['sigma']
        else:
            choice_sigma = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0

        for sigma in choice_sigma:
            self.lambda_para_dict = dict(model_id=self.model_id, sigma=sigma)
            yield self.lambda_para_dict
