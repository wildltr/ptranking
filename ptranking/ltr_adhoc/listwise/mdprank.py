#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from itertools import product

from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_k
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.sampling_utils import sample_ranking_PL, sample_ranking_PL_gumbel_softmax

class MDPRank(AdhocNeuralRanker):
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(MDPRank, self).__init__(id='MDPRank', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.gamma = model_para_dict['gamma']
        self.top_k = model_para_dict['top_k']
        self.temperature = model_para_dict['temperature']
        self.distribution = model_para_dict['distribution']  # 'PL', 'STPL'

        # checking the quality of sampled rankings
        self.pg_checking = False

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        # aiming for meaningful batch-normalization, please set {train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 1, 1, 1}
        assert 1 == batch_preds.size(0)

        assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking

        if 'PL' == self.distribution:
            batch_sample_inds, batch_action_preds = sample_ranking_PL(batch_preds=batch_preds, only_indices=False, temperature=self.temperature)

        elif 'STPL' == self.distribution:
            batch_sample_inds, batch_action_preds = sample_ranking_PL_gumbel_softmax(
                batch_preds=batch_preds, only_indices=False, temperature=self.temperature, device=self.device)
        else:
            raise NotImplementedError

        top_k = batch_std_labels.size(1) if self.top_k is None else self.top_k
        batch_action_stds = torch.gather(batch_std_labels, dim=1, index=batch_sample_inds)

        if self.pg_checking:
            sample_metric_values = torch_ndcg_at_k(batch_predict_rankings=batch_action_stds,
                                                   batch_ideal_rankings=batch_std_labels, k=5, device=self.device)

        # TODO alternative metrics, such as AP and NERR
        batch_gains = torch.pow(2.0, batch_action_stds) - 1.0
        batch_ranks = torch.arange(top_k, dtype=torch.float, device=self.device).view(1, -1)
        batch_discounts = torch.log2(2.0 + batch_ranks)
        batch_rewards = batch_gains[:, 0:top_k] / batch_discounts
        # the long-term return of the sampled episode starting from t
        """ this is also the key difference, equivalently, weighting is different """
        batch_G_t = torch.flip(torch.cumsum(torch.flip(batch_rewards, dims=[1]), dim=1), dims=[1])

        if self.gamma != 1.0:
            return_discounts = torch.cumprod(torch.ones(top_k).view(1, -1) * self.gamma, dim=1)
            batch_G_t = batch_G_t * return_discounts

        m, _ = torch.max(batch_action_preds, dim=1, keepdim=True)  # a transformation aiming for higher stability when computing softmax() with exp()
        y = batch_action_preds - m
        y = torch.exp(y)
        y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])  # row-wise cumulative sum, from tail to head
        batch_logcumsumexps = torch.log(y_cumsum_t2h) + m  # corresponding to the '-m' operation
        batch_neg_log_probs = batch_logcumsumexps[:, 0:top_k] - batch_action_preds[:, 0:top_k]
        batch_loss = torch.sum(torch.sum(batch_neg_log_probs * batch_G_t[:, 0:top_k], dim=1))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        if self.pg_checking:
            return sample_metric_values
        else:
            return batch_loss

###### Parameter of FastMDPRank ######

class MDPRankParameter(ModelParameter):
    ''' Parameter class for FastMDPRank '''

    def __init__(self, debug=False, para_json=None):
        super(MDPRankParameter, self).__init__(model_id='MDPRank', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for FastMDPRank
        """
        self.MDPRank_para_dict = dict(model_id=self.model_id, temperature=1.0, gamma=1.0, top_k=10,
                                      distribution='PL')
        return self.MDPRank_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        """
        # using specified para-dict or inner para-dict
        MDPRank_para_dict = given_para_dict if given_para_dict is not None else self.MDPRank_para_dict

        s1 = ':' if log else '_'
        top_k, distribution, gamma, temperature=MDPRank_para_dict['top_k'], MDPRank_para_dict['distribution'],\
                                                MDPRank_para_dict['gamma'], MDPRank_para_dict['temperature']
        fastMDPRank_para_str = s1.join([str(top_k), distribution, 'G', '{:,g}'.format(gamma),
                                        'T', '{:,g}'.format(temperature)])
        return fastMDPRank_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for FastMDPRank
        """
        if self.use_json:
            #choice_metric = json_dict['metric']
            choice_topk = self.json_dict['top_k']
            choice_distribution = self.json_dict['distribution']
            choice_temperature = self.json_dict['temperature']
            choice_gamma = self.json_dict['gamma']
        else:
            #choice_metric = ['NERR', 'nDCG', 'AP'] # 'nDCG', 'AP', 'NERR'
            choice_topk = [10] if self.debug else [10]
            choice_distribution = ['PL']
            choice_temperature = [.1] if self.debug else [1.0]  # 1.0, 10.0
            choice_gamma = [1.0]

        for top_k, distribution, temperature, gamma in product(choice_topk, choice_distribution, choice_temperature, choice_gamma):
            self.MDPRank_para_dict = dict(model_id=self.model_id, top_k=top_k, gamma=gamma,
                                          distribution=distribution, temperature=temperature)
            yield self.MDPRank_para_dict
