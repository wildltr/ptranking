#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Direct stochastic optimization of IR metrics based on twin-sigmoid and its variants
"""

from itertools import product

from org.archive.base.ranker import NeuralRanker

from org.archive.ltr_adhoc.util.twin_sigmoid import TWIN_SIGMOID
from org.archive.ltr_adhoc.util.ir_metric_as_a_loss import precision_as_a_loss, AP_as_a_loss, NERR_as_a_loss, nDCG_as_a_loss

IRMETRIC = ['P', 'AP', 'NERR', 'NDCG'] # supported ir metrics

class DirectOptIRMetric(NeuralRanker):
    '''
    Direct stochastic optimization of IR metrics based on twin-sigmoid and its variants
    '''
    def __init__(self, sf_para_dict=None, direct_opt_dict=None):
        super(DirectOptIRMetric, self).__init__(id='DirectOptIRMetric', sf_para_dict=sf_para_dict)
        self.direct_opt_dict = direct_opt_dict
        self.metric, self.twin_sigmoid_id, self.b_sigma, self.margin = direct_opt_dict['metric'],\
                               direct_opt_dict['twin_sigmoid_id'], direct_opt_dict['b_sigma'], direct_opt_dict['margin']
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
            batch_loss = nDCG_as_a_loss(batch_preds=batch_pred, batch_stds=batch_label,
                                        twin_sigmoid_id=self.twin_sigmoid_id, b_sigma=self.b_sigma, margin=self.margin)

        self.optimizer.zero_grad()
        batch_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.sf.get_parameters(), max_norm=20)
        self.optimizer.step()

        return batch_loss

###### Parameter of VirtualAP ######

def get_direct_opt_paras_str(direct_opt_dict, log=False):
    ''' convert parameter-setting as a string identifier '''
    s1, s2 = (':', '\n') if log else ('_', '_')

    twin_sigmoid_id = direct_opt_dict['twin_sigmoid_id']
    assert twin_sigmoid_id in TWIN_SIGMOID

    k      = direct_opt_dict['k'] if 'k' in direct_opt_dict else None
    metric = direct_opt_dict['metric']
    margin, b_sigma = direct_opt_dict['margin'], direct_opt_dict['b_sigma']
    twin_sigmoid_str = s1.join([twin_sigmoid_id, '{:,g}'.format(b_sigma)])
    metric_str = metric if k is None else s2.join([metric, str(k)])

    direct_opt_paras_str = s2.join([metric_str, twin_sigmoid_str])

    if margin is not None:
        margin_str = s1.join(['M', '{:,g}'.format(margin)])
        direct_opt_paras_str = s2.join([direct_opt_paras_str, margin_str])

    return direct_opt_paras_str


def default_direct_opt_para_dict(metric):
    '''
    An empirical setting for direct optimization of ir metrics
    '''
    direct_opt_dict = dict(model_id='DirectOptIRMetric', metric=metric, twin_sigmoid_id='Type3', b_sigma=1, margin=None)
    if 'NERR' == metric: direct_opt_dict['k'] = 10
    return direct_opt_dict


def direct_opt_grid(model_id=None, metric=None, direct_opt_choice_k=None, direct_opt_choice_twin_sigmoid=None, direct_opt_choice_b_sigma=None, direct_opt_choice_margin=None):
    ''' grid-search over hyper-parameters '''
    for twin_sigmoid_id in direct_opt_choice_twin_sigmoid:
        for b_sigma, margin in product(direct_opt_choice_b_sigma, direct_opt_choice_margin):
            if metric == 'NERR':
                for k in direct_opt_choice_k:
                    direct_opt_para_dict = dict(model_id=model_id, metric=metric, k=k, b_sigma=b_sigma, margin=margin, twin_sigmoid_id=twin_sigmoid_id)
                    yield direct_opt_para_dict
            else:
                direct_opt_para_dict = dict(model_id=model_id, metric=metric, b_sigma=b_sigma, twin_sigmoid_id=twin_sigmoid_id, margin=margin)
                yield direct_opt_para_dict



def direct_opt_para_iterator(debug, model_id=None, metric=None):
    ''' testing bsed on grid-serch '''

    assert metric is not None and metric in IRMETRIC

    direct_opt_choice_twin_sigmoid = ['Type3'] if debug else ['Type3'] # 'Type1', 'Type2', 'Type3'
    direct_opt_choice_b_sigma = [1.0] if debug else [1.0]  # 1.0, 2.0, 4.0, 6.0, 8.0, 10., 12.0, 14.0, 16.0, 18.0, 20.

    if 'NERR' == metric:
        direct_opt_choice_k = [10]
    else:
        direct_opt_choice_k = [None]

    #direct_opt_choice_margin = [None] if debug else [0.0001, 0.001, 0.01, 0.1, 1.0] # 0.001, 0.01, 0.1, 1.0, 3.0, 5.0
    direct_opt_choice_margin = [None] if debug else [None] # 0.001, 0.01, 0.1, 1.0, 3.0, 5.0

    return direct_opt_grid(model_id=model_id, metric=metric,
                           direct_opt_choice_k=direct_opt_choice_k, direct_opt_choice_twin_sigmoid=direct_opt_choice_twin_sigmoid,
                           direct_opt_choice_b_sigma=direct_opt_choice_b_sigma, direct_opt_choice_margin=direct_opt_choice_margin)