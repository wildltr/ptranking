#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
"""

import torch
import torch.nn.functional as F

from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs

class RankNet(AdhocNeuralRanker):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(RankNet, self).__init__(id='RankNet', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.sigma = model_para_dict['sigma']

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_preds, batch_std_labels=batch_std_labels,
                                                             sigma=self.sigma)
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

###### Parameter of RankNet ######

class RankNetParameter(ModelParameter):
    ''' Parameter class for RankNet '''
    def __init__(self, debug=False, para_json=None):
        super(RankNetParameter, self).__init__(model_id='RankNet', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for RankNet
        """
        self.ranknet_para_dict = dict(model_id=self.model_id, sigma=1.0)
        return self.ranknet_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        ranknet_para_dict = given_para_dict if given_para_dict is not None else self.ranknet_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        ranknet_para_str = s1.join(['Sigma', '{:,g}'.format(ranknet_para_dict['sigma'])])
        return ranknet_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for RankNet
        """
        if self.use_json:
            choice_sigma = self.json_dict['sigma']
        else:
            choice_sigma = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0

        for sigma in choice_sigma:
            self.ranknet_para_dict = dict(model_id=self.model_id, sigma=sigma)
            yield self.ranknet_para_dict
