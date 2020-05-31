#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/12/02 | https://y-research.github.io

"""Description

"""

import ot
import numpy as np
from itertools import product

import torch

from org.archive.base.ranker import NeuralRanker
from org.archive.ltr_adhoc.listwise.otrank.ot_loss_layer import EntropicOTLoss
from org.archive.ltr_adhoc.listwise.wassrank.wasserstein_loss_layer import Y_WassersteinLossStab
from org.archive.ltr_adhoc.listwise.wassrank.wasserstein_cost_mat import get_explicit_cost_mat, get_normalized_histograms

from org.archive.l2r_global import global_gpu as gpu, global_device as device, tensor

wasserstein_distance = Y_WassersteinLossStab.apply

class _EMD_OP(torch.autograd.Function):
    """ Aiming at mannual gradient computation """

    @staticmethod
    def forward(ctx, batch_pred_hists, batch_std_hists, batch_cost_mats):
        #print(batch_pred_hists.type())
        pred_hists = torch.squeeze(batch_pred_hists)*100.
        std_hists  = torch.squeeze(batch_std_hists)*100.
        cost_mat   = torch.squeeze(batch_cost_mats)/100.

        np_pred_hists = pred_hists.cpu().numpy() if gpu else pred_hists.data.numpy()
        np_std_hists  = std_hists.cpu().numpy() if gpu else std_hists.data.numpy()
        np_cost_mat   = cost_mat.cpu().numpy() if gpu else cost_mat.data.numpy()
        #print(np_pred_hists.dtype)

        print('np_pred_hists', np_pred_hists)
        print('np_std_hists',  np_std_hists)
        print('np_cost_mat', np_cost_mat)
        pi = ot.emd(a=np_pred_hists, b=np_std_hists, M=np_cost_mat)
        print('pi', pi)

        emd = np.sum(pi * np_cost_mat)

        batch_loss = torch.tensor([emd]).to(device) if gpu else torch.tensor([emd])
        print(batch_loss)

        # mannual gradients
        pi[pi >= 1e-8] = 1.0
        pi[pi < 1e-8]  = 0.0

        grads = np.sum(pi * np_cost_mat, axis=1)
        torch_grads = torch.tensor(grads).to(device) if gpu else torch.tensor(grads)
        batch_grad = torch_grads.view(batch_pred_hists.size())

        ctx.save_for_backward(batch_grad)

        return batch_loss

    @staticmethod
    def backward(ctx, grad_output):
        batch_grad = ctx.saved_tensors[0]
        target_gradients = grad_output*batch_grad
        # it is a must that keeping the same number w.r.t. the input of forward function
        return target_gradients, None, None


class WassRank(NeuralRanker):
    '''
    Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory.
    Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 2019.2.
    '''

    def __init__(self, sf_para_dict, wass_para_dict=None, dict_cost_mats=None, dict_std_dists=None):
        super(WassRank, self).__init__(id='WassRank', sf_para_dict=sf_para_dict)

        self.TL_AF = self.get_tl_af()
        self.wass_para_dict = wass_para_dict
        if dict_cost_mats is not None:
            self.dict_cost_mats = dict_cost_mats
        if dict_std_dists is not None:
            self.dict_std_dists = dict_std_dists

        if 'EOTLossSta' == self.wass_para_dict['mode']:
            self.entropic_ot_loss = EntropicOTLoss(eps=self.wass_para_dict['lam'], max_iter=self.wass_para_dict['sh_itr'])
            self.pi = None


    def inner_train(self, batch_preds, batch_stds, **kwargs):
        qid = kwargs['qid']
        if qid in self.dict_cost_mats:
            batch_cost_mats = self.dict_cost_mats[qid]  # using buffered cost matrices to avoid re-computation
        else:
            batch_cost_mats = get_explicit_cost_mat(batch_stds, wass_para_dict=self.wass_para_dict)
            self.dict_cost_mats[qid] = batch_cost_mats

        batch_std_hists, batch_pred_hists = get_normalized_histograms(batch_std_labels=batch_stds, batch_preds=batch_preds,
                                                                      wass_dict_std_dists=self.dict_std_dists, qid=qid,
                                                                      wass_para_dict=self.wass_para_dict, TL_AF=self.TL_AF)

        #'''
        wass_mode = self.wass_para_dict['mode']
        if wass_mode == 'WassLossSta':
            sh_itr, lam = self.wass_para_dict['sh_itr'], self.wass_para_dict['lam']
            if gpu: batch_std_hists = batch_std_hists.type(tensor)
            batch_loss, = wasserstein_distance(batch_pred_hists, batch_std_hists, torch.squeeze(batch_cost_mats, dim=0), lam, sh_itr)

        elif wass_mode == 'EOTLossSta':
            if gpu: batch_std_hists = batch_std_hists.type(tensor)
            batch_loss, self.pi = self.entropic_ot_loss(batch_pred_hists, batch_std_hists, batch_cost_mats)

        else:
            raise NotImplementedError
        #'''

        #batch_loss = apply_EMD_OP(batch_pred_hists, batch_std_hists, batch_cost_mats)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of WassRank ######

def wass_grid(wass_choice_mode=None, wass_choice_lam=None, wass_choice_itr=None,
              wass_choice_smooth=None, wass_choice_norm=None,
              wass_cost_type=None,
              wass_choice_non_rele_gap=None, wass_choice_var_penalty=None, wass_choice_group_base=None):
    """  """
    for mode, wsss_lambda, sinkhorn_itr in product(wass_choice_mode, wass_choice_lam, wass_choice_itr):
        for wass_smooth, norm in product(wass_choice_smooth, wass_choice_norm):
            for cost_type in wass_cost_type:
                for non_rele_gap, var_penalty, group_base in product(wass_choice_non_rele_gap, wass_choice_var_penalty, wass_choice_group_base):
                    w_para_dict = dict(model_id='WassRank', mode=mode, sh_itr=sinkhorn_itr, lam=wsss_lambda, cost_type=cost_type,
                                       smooth_type=wass_smooth, norm_type=norm, gain_base=group_base, non_rele_gap=non_rele_gap, var_penalty=var_penalty)
                    yield w_para_dict


def wassrank_para_iterator():
    """  """
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

    return wass_grid(wass_choice_mode=wass_choice_mode, wass_choice_lam=wass_choice_lam, wass_choice_itr=wass_choice_itr,
                     wass_choice_smooth=wass_choice_smooth, wass_choice_norm=wass_choice_norm,
                     wass_cost_type=wass_cost_type, wass_choice_non_rele_gap=wass_choice_non_rele_gap,
                     wass_choice_var_penalty=wass_choice_var_penalty, wass_choice_group_base=wass_choice_group_base)



def get_wass_para_str(ot_para_dict, log=False):
    s1, s2 = (':', '\n') if log else ('_', '_')

    cost_type, smooth_type, norm_type = ot_para_dict['cost_type'], ot_para_dict['smooth_type'], ot_para_dict['norm_type']

    mode_str = s1.join(['mode', ot_para_dict['mode']]) if log else ot_para_dict['mode']

    if smooth_type in ['ST', 'NG']:
        smooth_str = s1.join(['smooth_type', smooth_type]) if log else s1.join(['ST', smooth_type])
    else:
        raise NotImplementedError

    if cost_type.startswith('Group'):
        gain_base, non_rele_gap, var_penalty = ot_para_dict['gain_base'], ot_para_dict['non_rele_gap'],  ot_para_dict['var_penalty']
        cost_str = s2.join([s1.join(['cost_type', cost_type]),
                            s1.join(['gain_base',    '{:,g}'.format(gain_base)]),
                            s1.join(['non_rele_gap', '{:,g}'.format(non_rele_gap)]),
                            s1.join(['var_penalty',  '{:,g}'.format(var_penalty)])]) if log \
                   else s1.join([cost_type, '{:,g}'.format(non_rele_gap), '{:,g}'.format(gain_base), '{:,g}'.format(var_penalty)])
    else:
        cost_str = s1.join(['cost_type', cost_type]) if log else cost_type

    sh_itr, lam = ot_para_dict['sh_itr'], ot_para_dict['lam']
    horn_str = s2.join([s1.join(['Lambda', '{:,g}'.format(lam)]), s1.join(['ShIter', str(sh_itr)])]) if log \
                   else s1.join(['Lambda', '{:,g}'.format(lam), 'ShIter', str(sh_itr)])

    wass_paras_str = s2.join([mode_str, smooth_str, cost_str, horn_str])

    return wass_paras_str


def get_wsdm2019_para_dict(model_id=None):
    w_para_dict = dict(model_id=model_id, mode='EOTLossSta',
                       sh_itr=20, lam=0.1,
                       smooth_type='ST', norm_type='BothST',
                       cost_type='eg', non_rele_gap=100, var_penalty=np.e, gain_base=4)
    return w_para_dict