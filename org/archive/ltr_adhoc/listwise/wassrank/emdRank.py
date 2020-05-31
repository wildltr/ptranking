#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import ot

import numpy as np

import torch
import torch.nn.functional as F

from org.archive.base.ranker import NeuralRanker
from org.archive.l2r_global import global_gpu as gpu, global_device as device


def EMD_1DClosedForm_Loss(batch_preds=None, batch_stds=None):
    ''' Viewing the 1D closed form EMD of pushing the predicted histogram to standard histogram as the ltr_adhoc loss '''
    batch_hists_pred = F.softmax(batch_preds, dim=1)
    batch_cumsum_pred = torch.cumsum(batch_hists_pred, dim=1)

    batch_hists_std = F.softmax(batch_stds, dim=1)
    batch_cumsum_std = torch.cumsum(batch_hists_std, dim=1)
    # the earth mover's distance using the closed form solution
    emd = torch.sum(torch.abs(batch_cumsum_pred - batch_cumsum_std)) / torch.sum(batch_preds)

    return emd


def get_cost_mat_const(size_dist, const=1.0):
    cost_mat = np.full((size_dist, size_dist), const) - np.eye(size_dist)*const
    return cost_mat

from org.archive.ltr_adhoc.listwise.lambdarank import get_delta_ndcg

def get_std_delta_gains(batch_stds):
    batch_gains = torch.pow(2.0, batch_stds) - 1.0
    batch_gain_diffs = torch.unsqueeze(batch_gains, dim=2) - torch.unsqueeze(batch_gains, dim=1)
    batch_delta_gain = torch.abs(batch_gain_diffs)  # absolute delta gains w.r.t. pairwise swapping

    return batch_delta_gain

def minimum_emd_pair_indice(batch_ideally_sorted_labels=None, batch_stds_sorted_via_preds=None):
    ideally_sorted_labels = torch.squeeze(batch_ideally_sorted_labels)
    stds_sorted_via_preds = torch.squeeze(batch_stds_sorted_via_preds)

    np_ideally_sorted_labels = ideally_sorted_labels.cpu().numpy() if gpu else ideally_sorted_labels.data.numpy()
    np_stds_sorted_via_preds = stds_sorted_via_preds.cpu().numpy() if gpu else stds_sorted_via_preds.data.numpy()

    cost_mat = get_cost_mat_const(len(np_ideally_sorted_labels))

    #print('source: ', np_stds_sorted_via_preds)
    #print('target: ', np_ideally_sorted_labels)
    #print('const cost matrix:\n', cost_mat)

    pi = ot.emd(a=np_stds_sorted_via_preds, b=np_ideally_sorted_labels, M=cost_mat)
    np.fill_diagonal(pi, 0)

    #print(pi)

    row_inds, col_inds = np.nonzero(pi)
    num_pairs = len(row_inds)
    if not num_pairs > 0:
        #print('num of pairs', num_pairs)
        #print(row_inds)
        #print('source: ', np_stds_sorted_via_preds)
        #print('target: ', np_ideally_sorted_labels)
        return None, None, True

    #print('row_inds', row_inds)
    #print('col_inds', col_inds)

    tor_row_inds = torch.LongTensor(row_inds).to(device) if gpu else torch.LongTensor(row_inds)
    tor_col_inds = torch.LongTensor(col_inds).to(device) if gpu else torch.LongTensor(col_inds)
    #batch_triu = batch_mats[:, tor_row_inds, tor_col_inds]

    return tor_row_inds, tor_col_inds, False  # shape: [number of pairs]

'''
todo-as-note:
selecting specific pairs but still using delta-nDCG will lead to small values compared with the original lambdaRank
'''

def lambdaRank_emd_loss(batch_preds=None, batch_stds=None, sigma=1.0):
    '''
    This method will impose explicit bias to highly ranked documents that are essentially ties
    :param batch_preds:
    :param batch_stds:
    :return:
    '''
    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    # get unique document pairs, which is dynamically different per training iteration
    pair_row_inds, pair_col_inds, zero_emd = minimum_emd_pair_indice(batch_ideally_sorted_labels=batch_stds, batch_stds_sorted_via_preds=batch_stds_sorted_via_preds)
    if zero_emd:
        return None, True

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_Sij = batch_std_Sij[:, pair_row_inds, pair_col_inds]

    batch_pred_diffs = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j
    batch_pred_s_ij = batch_pred_diffs[:, pair_row_inds, pair_col_inds] # unique pairwise comparisons according to a ltr_adhoc of documents

    #batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds)
    batch_delta_ndcg = get_std_delta_gains(batch_stds)
    batch_delta_ndcg = batch_delta_ndcg[:, pair_row_inds, pair_col_inds] + .5 # due to much 1.0
    #print('batch_delta_ndcg', batch_delta_ndcg)

    batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij) # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_s_ij) + 1.0)  # cf. the 1st equation in page-3

    batch_loss = torch.sum((batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg)    # weighting with delta-nDCG

    return batch_loss, False


class EMDRank(NeuralRanker):
    '''
    '''
    def __init__(self, scoring_function=None):
        super(EMDRank, self).__init__(id='EMDRank', scoring_function=scoring_function)

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        #batch_loss = EMD_1DClosedForm_Loss(batch_preds=batch_preds, batch_stds=batch_stds)

        batch_loss, zero_loss = lambdaRank_emd_loss(batch_preds=batch_preds, batch_stds=batch_stds)

        if not zero_loss:
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            return batch_loss
        else:
            return torch.tensor([0.0]).to(device) if gpu else torch.tensor([0.0])