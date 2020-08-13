#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Direct stochastic optimization of IR metrics based on twin-sigmoid and its variants
"""

from itertools import product

from org.archive.base.ranker import NeuralRanker
from org.archive.eval.parameter import ModelParameter
from org.archive.ltr_adhoc.util.twin_sigmoid import TWIN_SIGMOID
from org.archive.ltr_adhoc.util.ir_metric_as_a_loss import precision_as_a_loss, AP_as_a_loss, NERR_as_a_loss, nDCG_as_a_loss

IRMETRIC = ['P', 'AP', 'NERR', 'NDCG'] # supported ir metrics

class DirectOpt(NeuralRanker):
    '''
    Direct stochastic optimization of IR metrics based on twin-sigmoid and its variants
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None):
        super(DirectOpt, self).__init__(id='DirectOpt', sf_para_dict=sf_para_dict)
        self.direct_opt_dict = model_para_dict
        self.multi_level_rele = False if model_para_dict['std_rele_is_permutation'] else True
        self.metric, self.twin_sigmoid_id, self.b_sigma, self.margin = model_para_dict['metric'],\
                               model_para_dict['twin_sigmoid_id'], model_para_dict['b_sigma'], model_para_dict['margin']
        assert self.metric in IRMETRIC and self.twin_sigmoid_id in TWIN_SIGMOID

    def inner_train(self, batch_pred, batch_label, **kwargs):
        '''
        per-query training process
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        if 'P' == self.metric:
            batch_loss = precision_as_a_loss(batch_preds=batch_pred, batch_stds=batch_label, margin=self.margin,
                                             twin_sigmoid_id=self.twin_sigmoid_id, b_sigma=self.b_sigma)
        elif 'AP' == self.metric:
            batch_loss = AP_as_a_loss(batch_preds=batch_pred, batch_stds=batch_label,
                                      twin_sigmoid_id=self.twin_sigmoid_id, b_sigma=self.b_sigma, margin=self.margin)
        elif 'NERR' == self.metric:
            batch_loss = NERR_as_a_loss(batch_preds=batch_pred, batch_stds=batch_label, k=self.direct_opt_dict['k'],
                                        twin_sigmoid_id=self.twin_sigmoid_id, b_sigma=self.b_sigma, margin=self.margin)
        elif 'NDCG' == self.metric:
            batch_loss = nDCG_as_a_loss(batch_preds=batch_pred, batch_stds=batch_label, b_sigma=self.b_sigma,
                                        twin_sigmoid_id=self.twin_sigmoid_id, margin=self.margin,
                                        multi_level_rele=self.multi_level_rele)

        self.optimizer.zero_grad()
        batch_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.sf.get_parameters(), max_norm=20)
        self.optimizer.step()

        return batch_loss


###### Parameter of DirectOpt ######

class DirectOptParameter(ModelParameter):
    ''' Parameter class for DirectOpt '''
    def __init__(self, debug=False, std_rele_is_permutation=False):
        super(DirectOptParameter, self).__init__(model_id='DirectOpt')
        self.debug = debug
        self.std_rele_is_permutation = std_rele_is_permutation

    def default_para_dict(self):
        """
        Default parameter setting for DirectOpt
        :return:
        """
        # 'P', 'AP', 'NERR', 'NDCG'
        self.direct_opt_dict = dict(model_id='DirectOpt', metric='NDCG', twin_sigmoid_id='Type3', b_sigma=1, margin=None,
                                    std_rele_is_permutation=self.std_rele_is_permutation)
        if 'NERR' == self.direct_opt_dict['metric']: self.direct_opt_dict['k'] = 10
        return self.direct_opt_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        direct_opt_dict = given_para_dict if given_para_dict is not None else self.direct_opt_dict

        s1, s2 = (':', '\n') if log else ('_', '_')

        twin_sigmoid_id = direct_opt_dict['twin_sigmoid_id']
        assert twin_sigmoid_id in TWIN_SIGMOID

        k = direct_opt_dict['k'] if 'k' in direct_opt_dict else None
        metric = direct_opt_dict['metric']
        margin, b_sigma = direct_opt_dict['margin'], direct_opt_dict['b_sigma']
        twin_sigmoid_str = s1.join([twin_sigmoid_id, '{:,g}'.format(b_sigma)])
        metric_str = metric if k is None else s2.join([metric, str(k)])

        direct_opt_paras_str = s2.join([metric_str, twin_sigmoid_str])

        if margin is not None:
            margin_str = s1.join(['M', '{:,g}'.format(margin)])
            direct_opt_paras_str = s2.join([direct_opt_paras_str, margin_str])

        return direct_opt_paras_str

    def grid_search(self, debug=False):
        """
        Iterator of parameter settings for DirectOpt
        :param debug:
        :return:
        """
        choice_metric = ['NDCG'] if self.debug else ['P', 'AP', 'NERR', 'NDCG']  # 'Type1', 'Type2', 'Type3'
        choice_twin_sigmoid = ['Type3'] if self.debug else ['Type3']  # 'Type1', 'Type2', 'Type3'
        choice_b_sigma = [1.0] if self.debug else [1.0]  # 1.0, 2.0, 4.0, 6.0, 8.0, 10., 12.0, 14.0, 16.0, 18.0, 20.
        choice_k = [None] if self.debug else [10]  # limited to 'NERR'

        # choice_margin = [None] if self.debug else [0.0001, 0.001, 0.01, 0.1, 1.0] # 0.001, 0.01, 0.1, 1.0, 3.0, 5.0
        choice_margin = [None] if self.debug else [None]  # 0.001, 0.01, 0.1, 1.0, 3.0, 5.0

        for b_sigma, margin, twin_sigmoid_id in product(choice_b_sigma, choice_margin, choice_twin_sigmoid):
            for metric in choice_metric:
                if metric == 'NERR':
                    for k in choice_k:
                        direct_opt_para_dict = dict(model_id=self.model_id, metric=metric, k=k, b_sigma=b_sigma,
                                                    std_rele_is_permutation=self.std_rele_is_permutation,
                                                    margin=margin, twin_sigmoid_id=twin_sigmoid_id)
                        yield direct_opt_para_dict
                else:
                    direct_opt_para_dict = dict(model_id=self.model_id, metric=metric, b_sigma=b_sigma,
                                                std_rele_is_permutation=self.std_rele_is_permutation,
                                                twin_sigmoid_id=twin_sigmoid_id, margin=margin)
                    yield direct_opt_para_dict
