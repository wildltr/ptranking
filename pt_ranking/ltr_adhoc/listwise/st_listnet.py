#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""


import torch
import torch.nn.functional as F

from pt_ranking.base.ranker import NeuralRanker
from pt_ranking.eval.parameter import ModelParameter

from pt_ranking.ltr_global import global_gpu as gpu, global_device as device

EPS = 1e-20

class STListNet(NeuralRanker):
    '''
    author = {Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc},
    title = {A Stochastic Treatment of Learning to Rank Scoring Functions},
    year = {2020},
    booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
    pages = {61–69},
    '''

    def __init__(self, sf_para_dict=None, model_para_dict=None):
        super(STListNet, self).__init__(id='STListNet', sf_para_dict=sf_para_dict)
        self.temperature = model_para_dict['temperature']

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        unif = torch.rand(batch_preds.size())  # [num_samples_per_query, ranking_size]
        if gpu: unif = unif.to(device)

        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

        batch_preds = (batch_preds + gumbel) / self.temperature

        # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
        # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        batch_loss = torch.sum(-torch.sum(F.softmax(batch_stds, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

###### Parameter of STListNet ######

class STListNetParameter(ModelParameter):
    ''' Parameter class for STListNet '''
    def __init__(self, debug=False):
        super(STListNetParameter, self).__init__(model_id='STListNet')
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
        plus_choice_temperature = [1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0
        for temperature in plus_choice_temperature:
            self.stlistnet_para_dict = dict(model_id=self.model_id, temperature=temperature)
            yield self.stlistnet_para_dict
