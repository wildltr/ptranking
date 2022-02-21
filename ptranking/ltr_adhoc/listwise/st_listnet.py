#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
author = {Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc},
title = {A Stochastic Treatment of Learning to Rank Scoring Functions},
year = {2020},
booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
pages = {61–69}
"""

import torch
import torch.nn.functional as F

from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter


EPS = 1e-20

class STListNet(AdhocNeuralRanker):
    '''
    author = {Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc},
    title = {A Stochastic Treatment of Learning to Rank Scoring Functions},
    year = {2020},
    booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
    pages = {61–69}
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(STListNet, self).__init__(id='STListNet', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.temperature = model_para_dict['temperature']

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        unif = torch.rand(batch_preds.size(), device=self.device)  # [batch_size, ranking_size]

        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

        batch_preds = (batch_preds + gumbel) / self.temperature

        # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
        # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        batch_loss = torch.sum(-torch.sum(F.softmax(batch_std_labels, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

###### Parameter of STListNet ######

class STListNetParameter(ModelParameter):
    ''' Parameter class for STListNet '''
    def __init__(self, debug=False, para_json=None):
        super(STListNetParameter, self).__init__(model_id='STListNet', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for STListNet
        :return:
        """
        self.stlistnet_para_dict = dict(model_id=self.model_id, temperature=1.0)
        return self.stlistnet_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        stlistnet_para_dict = given_para_dict if given_para_dict is not None else self.stlistnet_para_dict

        s1 = ':' if log else '_'
        stlistnet_para_str = s1.join(['Tem', str(stlistnet_para_dict['temperature'])])
        return stlistnet_para_str


    def grid_search(self):
        """
        Iterator of parameter settings for STListNet
        :param debug:
        :return:
        """
        if self.use_json:
            choice_temperature = self.json_dict['temperature']
        else:
            choice_temperature = [1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0

        for temperature in choice_temperature:
            self.stlistnet_para_dict = dict(model_id=self.model_id, temperature=temperature)
            yield self.stlistnet_para_dict
