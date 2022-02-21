#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory.
Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 2019.2.
"""
import numpy as np
from itertools import product

import torch

from ptranking.base.list_ranker import ListNeuralRanker
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.listwise.wassrank.pytorch_wasserstein import SinkhornOT, EntropicOT, OldSinkhornOT
from ptranking.ltr_adhoc.listwise.wassrank.wasserstein_cost_mat import get_explicit_cost_mat, get_normalized_histograms

wasserstein_distance = OldSinkhornOT.apply

class WassRank(AdhocNeuralRanker):
    '''
    Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory.
    Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 2019.2.
    '''

    def __init__(self, sf_para_dict, wass_para_dict=None, dict_cost_mats=None, dict_std_dists=None, gpu=False, device=None):
        super(WassRank, self).__init__(id='WassRank', sf_para_dict=sf_para_dict, gpu=gpu, device=device)

        self.TL_AF = self.get_tl_af()
        self.wass_para_dict = wass_para_dict
        if dict_cost_mats is not None:
            self.dict_cost_mats = dict_cost_mats
        if dict_std_dists is not None:
            self.dict_std_dists = dict_std_dists

        if 'EntropicOT' == self.wass_para_dict['mode']:
            self.entropic_ot_loss = EntropicOT(eps=self.wass_para_dict['lam'], max_iter=self.wass_para_dict['sh_itr'])
            self.pi = None


    def custom_loss_function(self, batch_preds, batch_stds, **kwargs):
        batch_ids = kwargs['batch_ids']
        #print('batch_ids', batch_ids)
        print('batch_preds', batch_preds.size(), batch_preds)
        #print('batch_stds', batch_stds.size(), batch_stds)
        if len(batch_ids) == 1:
            qid = batch_ids[0]
        else:
            qid = '_'.join(batch_ids)

        #print('qid', qid)

        '''
        if qid in self.dict_cost_mats:
            batch_cost_mats = self.dict_cost_mats[qid]  # using buffered cost matrices to avoid re-computation
        else:
            batch_cost_mats = get_explicit_cost_mat(batch_stds, wass_para_dict=self.wass_para_dict, gpu=self.gpu)
            self.dict_cost_mats[qid] = batch_cost_mats
        '''

        batch_cost_mats = get_explicit_cost_mat(batch_stds, wass_para_dict=self.wass_para_dict, gpu=self.gpu)

        batch_std_hists, batch_pred_hists = get_normalized_histograms(batch_std_labels=batch_stds, batch_preds=batch_preds,
                                                                      wass_dict_std_dists=self.dict_std_dists, qid=qid,
                                                                      wass_para_dict=self.wass_para_dict, TL_AF=self.TL_AF)

        wass_mode = self.wass_para_dict['mode']
        if wass_mode == 'SinkhornOT':
            sh_itr, lam = self.wass_para_dict['sh_itr'], self.wass_para_dict['lam']
            if self.gpu: batch_std_hists = batch_std_hists.type(torch.cuda.FloatTensor)
            batch_loss = wasserstein_distance(batch_pred_hists, batch_std_hists, torch.squeeze(batch_cost_mats, dim=0), lam, sh_itr)

        elif wass_mode == 'EntropicOT':
            if self.gpu: batch_std_hists = batch_std_hists.type(torch.cuda.FloatTensor)
            batch_loss, self.pi = self.entropic_ot_loss(batch_pred_hists, batch_std_hists, batch_cost_mats)

        else:
            raise NotImplementedError

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of WassRank ######

class WassRankParameter(ModelParameter):
    ''' Parameter class for WassRank '''
    def __init__(self, debug=False, para_json=None):
        super(WassRankParameter, self).__init__(model_id='WassRank', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for WassRank. EntropicOT | SinkhornOT
        :return:
        """
        self.wass_para_dict = dict(model_id=self.model_id, mode='SinkhornOT', sh_itr=20, lam=0.1, smooth_type='ST',
                                   norm_type='BothST', cost_type='eg', non_rele_gap=100, var_penalty=np.e, gain_base=4)
        return self.wass_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        wass_para_dict = given_para_dict if given_para_dict is not None else self.wass_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')

        cost_type, smooth_type, norm_type = wass_para_dict['cost_type'], wass_para_dict['smooth_type'], wass_para_dict[
            'norm_type']

        mode_str = s1.join(['mode', wass_para_dict['mode']]) if log else wass_para_dict['mode']

        if smooth_type in ['ST', 'NG']:
            smooth_str = s1.join(['smooth_type', smooth_type]) if log else s1.join(['ST', smooth_type])
        else:
            raise NotImplementedError

        if cost_type.startswith('Group'):
            gain_base, non_rele_gap, var_penalty = wass_para_dict['gain_base'], wass_para_dict['non_rele_gap'], \
                                                   wass_para_dict['var_penalty']
            cost_str = s2.join([s1.join(['cost_type', cost_type]),
                                s1.join(['gain_base', '{:,g}'.format(gain_base)]),
                                s1.join(['non_rele_gap', '{:,g}'.format(non_rele_gap)]),
                                s1.join(['var_penalty', '{:,g}'.format(var_penalty)])]) if log \
                else s1.join(
                [cost_type, '{:,g}'.format(non_rele_gap), '{:,g}'.format(gain_base), '{:,g}'.format(var_penalty)])
        else:
            cost_str = s1.join(['cost_type', cost_type]) if log else cost_type

        sh_itr, lam = wass_para_dict['sh_itr'], wass_para_dict['lam']
        horn_str = s2.join([s1.join(['Lambda', '{:,g}'.format(lam)]), s1.join(['ShIter', str(sh_itr)])]) if log \
            else s1.join(['Lambda', '{:,g}'.format(lam), 'ShIter', str(sh_itr)])

        wass_paras_str = s2.join([mode_str, smooth_str, cost_str, horn_str])

        return wass_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for WassRank
        """
        if self.use_json:
            wass_choice_mode = self.json_dict['mode']
            wass_choice_itr = self.json_dict['itr']
            wass_choice_lam = self.json_dict['lam']

            wass_cost_type = self.json_dict['cost_type']
            # member parameters of 'Group' include margin, div, group-base
            wass_choice_non_rele_gap = self.json_dict['non_rele_gap']
            wass_choice_var_penalty = self.json_dict['var_penalty']
            wass_choice_group_base = self.json_dict['group_base']

            wass_choice_smooth = self.json_dict['smooth']
            wass_choice_norm = self.json_dict['norm']
        else:
            wass_choice_mode = ['WassLossSta']  # EOTLossSta | WassLossSta
            wass_choice_itr = [10]  # number of iterations w.r.t. sink-horn operation
            wass_choice_lam = [0.1]  # 0.01 | 1e-3 | 1e-1 | 10  regularization parameter

            wass_cost_type = ['eg']  # p1 | p2 | eg | dg| ddg
            # member parameters of 'Group' include margin, div, group-base
            wass_choice_non_rele_gap = [10]  # the gap between a relevant document and an irrelevant document
            wass_choice_var_penalty = [np.e]  # variance penalty
            wass_choice_group_base = [4]  # the base for computing gain value

            wass_choice_smooth = ['ST']  # 'ST', i.e., ST: softmax | Gain, namely the way on how to get the normalized distribution histograms
            wass_choice_norm = ['BothST']  # 'BothST': use ST for both prediction and standard labels

        for mode, wsss_lambda, sinkhorn_itr in product(wass_choice_mode, wass_choice_lam, wass_choice_itr):
            for wass_smooth, norm in product(wass_choice_smooth, wass_choice_norm):
                for cost_type in wass_cost_type:
                    for non_rele_gap, var_penalty, group_base in product(wass_choice_non_rele_gap,
                                                                         wass_choice_var_penalty,
                                                                         wass_choice_group_base):
                        self.wass_para_dict = dict(model_id='WassRank', mode=mode, sh_itr=sinkhorn_itr, lam=wsss_lambda,
                                                   cost_type=cost_type, smooth_type=wass_smooth, norm_type=norm,
                                                   gain_base=group_base, non_rele_gap=non_rele_gap, var_penalty=var_penalty)
                        yield self.wass_para_dict
