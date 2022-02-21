#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
"""
import numpy as np

import torch
import torch.nn.functional as F

from ptranking.metric.adhoc.adhoc_metric import rele_gain

def tor_sum_norm(histogram):
    probs = torch.div(histogram, torch.sum(histogram, dim=1, keepdim=True))
    return probs

def _cost_mat_group(cpu_tor_batch_std_label_vec, non_rele_gap=100.0, var_penalty=0.01, gain_base=4.0):
    """
    Numpy reference
    Take into account the group information among documents, namely whether two documents are of the same standard relevance degree
    @param non_rele_gap the gap between a relevant document and an irrelevant document
    @param var_penalty variance penalty
    @param gain_base the base for computing gain value
    """
    size_ranking = cpu_tor_batch_std_label_vec.size(1)
    std_label_vec = cpu_tor_batch_std_label_vec[0, :].numpy()

    cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float32)
    for i in range(size_ranking):
        i_rele_level = std_label_vec[i]
        for j in range(size_ranking):
            if i==j:
                cost_mat[i, j] = 0
            else:
                j_rele_level = std_label_vec[j]

                if i_rele_level == j_rele_level:
                    cost_mat[i, j] = var_penalty
                else:
                    cost_mat[i, j] = np.abs(rele_gain(i_rele_level, gain_base=gain_base) - rele_gain(j_rele_level, gain_base=gain_base))

                    if 0 == i_rele_level or 0 == j_rele_level:  #enforce the margin between relevance and non-relevance
                        cost_mat[i, j] += non_rele_gap

    return cost_mat

def torch_cost_mat_dist(batch_std_labels, exponent=1.0, gpu=False):
    """ Viewing the absolute difference (with a exponent value) between two rank positions as the cost """
    batch_size = batch_std_labels.size(0)
    ranking_size = batch_std_labels.size(1)

    positions = (torch.arange(ranking_size) + 1.0).type(torch.cuda.FloatTensor) if gpu else (torch.arange(ranking_size) + 1.0).type(torch.FloatTensor)
    C = positions.view(-1, 1) - positions.view(1, -1)
    C = torch.abs(C)
    batch_C = C.expand(batch_size, -1, -1)

    if exponent > 1.0:
        batch_C = torch.pow(batch_C, exponent)

    return batch_C


def get_delta_gains(batch_stds, discount=False, gpu=False):
    '''
    Delta-gains w.r.t. pairwise swapping of the ideal ltr_adhoc
    :param batch_stds: the standard labels sorted in a descending order
    :return:
    '''
    batch_gains = torch.pow(2.0, batch_stds) - 1.0
    batch_g_diffs = torch.unsqueeze(batch_gains, dim=2) - torch.unsqueeze(batch_gains, dim=1)

    if discount:
        batch_std_ranks = torch.arange(batch_stds.size(1)).type(torch.cuda.FloatTensor) if gpu else torch.arange(batch_stds.size(1)).type(torch.FloatTensor)
        batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
        batch_dists = torch.unsqueeze(batch_dists, dim=0)
        batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
        batch_delta_gs = torch.abs(batch_g_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping
    else:
        batch_delta_gs = torch.abs(batch_g_diffs)  # absolute delta gains w.r.t. pairwise swapping

    return batch_delta_gs


def torch_cost_mat_group(batch_std_labels, non_rele_gap=100.0, var_penalty=0.01, gain_base=2.0, gpu=False):
    """
    Take into account the group information among documents, namely whether two documents are of the same standard relevance degree
    :param batch_std_labels: standard relevance labels
    :param non_rele_gap:     the gap between a relevant document and an irrelevant document
    :param var_penalty:      variance penalty w.r.t. the transportation among documents of the same label
    :param gain_base:        the base for computing gain value
    :return:                 cost matrices
    """
    batch_size = batch_std_labels.size(0)
    ranking_size = batch_std_labels.size(1)

    batch_std_gains = torch.pow(gain_base, batch_std_labels) - 1.0
    torch_non_rele_gap = torch.cuda.FloatTensor([non_rele_gap]) if gpu else torch.FloatTensor([non_rele_gap])
    batch_std_gains_gaps = torch.where(batch_std_gains < 1.0, -torch_non_rele_gap, batch_std_gains) # add the gap between relevance and non-relevance
    batch_std_costs = batch_std_gains_gaps.view(batch_size, ranking_size, 1) - batch_std_gains_gaps.view(batch_size, 1, ranking_size)
    batch_std_costs = torch.abs(batch_std_costs) # symmetric cost, i.e., C_{ij} = C_{ji}

    # add variance penalty, i.e., the cost of transport among positions of the same relevance level. But the diagonal entries need to be revised later
    torch_var_penalty = torch.cuda.FloatTensor([var_penalty]) if gpu else torch.FloatTensor([var_penalty])
    batch_C = torch.where(batch_std_costs < 1.0, torch_var_penalty, batch_std_costs)

    torch_eye = torch.eye(ranking_size).type(torch.cuda.FloatTensor) if gpu else torch.eye(ranking_size).type(torch.FloatTensor)
    diag = torch_eye * var_penalty
    batch_diags = diag.expand(batch_size, -1, -1)
    batch_C = batch_C - batch_diags              # revise the diagonal entries to zero again

    return batch_C

def get_explicit_cost_mat(batch_std_labels, wass_para_dict=None, gpu=False):
    """
    Initialize the cost matrix based on pre-defined (prior) knowledge
    :param batch_std_labels:
    :param wass_para_dict:
    :return:
    """
    cost_type = wass_para_dict['cost_type']
    if cost_type   == 'p1': # |x-y|
        batch_C = torch_cost_mat_dist(batch_std_labels, gpu=gpu)

    elif cost_type == 'p2': # |x-y|^2
        batch_C = torch_cost_mat_dist(batch_std_labels, exponent=2.0, gpu=gpu)

    elif cost_type == 'eg': # explicit grouping of relevance labels
        gain_base, non_rele_gap, var_penalty = wass_para_dict['gain_base'], wass_para_dict['non_rele_gap'], wass_para_dict['var_penalty']
        batch_C = torch_cost_mat_group(batch_std_labels, non_rele_gap=non_rele_gap, var_penalty=var_penalty, gain_base=gain_base, gpu=gpu)

    elif cost_type == 'dg': # delta gain
        batch_C = get_delta_gains(batch_std_labels, gpu=gpu)

    elif cost_type == 'ddg': # delta discounted gain
        batch_C = get_delta_gains(batch_std_labels, discount=True, gpu=gpu)
    else:
        raise NotImplementedError

    return batch_C


def get_standard_normalized_histogram_ST(batch_std_labels, non_rele_as=0.0, adjust_softmax=True):
    """
    Convert to a normalized histogram based on softmax
    The underlying trick is to down-weight the mass of non-relevant labels: treat them as one, then average the probability mass
    """
    if adjust_softmax:
        batch_ones = torch.ones_like(batch_std_labels)
        batch_zeros = torch.zeros_like(batch_std_labels)
        batch_non_rele_ones = torch.where(batch_std_labels > 0.0, batch_zeros, batch_ones)
        batch_non_cnts = torch.sum(batch_non_rele_ones, dim=1)

        batch_std_exps = torch.exp(batch_std_labels)
        if non_rele_as != 0.0:
            batch_rele_ones = 1.0 - batch_non_rele_ones
            batch_non_vals = batch_non_rele_ones * non_rele_as
            batch_non_avgs = (torch.exp(batch_non_vals) - batch_rele_ones) /batch_non_cnts

        else:
            batch_non_avgs  = batch_non_rele_ones / batch_non_cnts

        batch_std_adjs = batch_std_exps - batch_non_rele_ones + batch_non_avgs
        batch_histograms = batch_std_adjs/torch.sum(batch_std_adjs)

    else:
        batch_histograms = F.softmax(batch_std_labels, dim=1)

    return batch_histograms

def get_standard_normalized_histogram_GN(batch_std_labels, gain_base=2.0):
    """
    Convert to a normalized histogram based on gain values, i.e., each entry equals to gain_value/sum_gain_value
    :param batch_std_labels:
    :return:
    """
    batch_std_gains = torch.pow(gain_base, batch_std_labels) - 1.0
    batch_histograms = batch_std_gains/torch.sum(batch_std_gains, dim=1).view(-1, 1)
    return batch_histograms


def get_normalized_histograms(batch_std_labels=None, batch_preds=None,
                              wass_dict_std_dists=None, qid=None, wass_para_dict=None, TL_AF=None):
    """ Convert both standard labels and predictions w.r.t. a query to normalized histograms """
    smooth_type, norm_type = wass_para_dict['smooth_type'], wass_para_dict['norm_type']

    if 'ST' == smooth_type:
        if wass_dict_std_dists is not None:
            if qid in wass_dict_std_dists:  # target distributions
                batch_std_hists = wass_dict_std_dists[qid]
            else:
                batch_std_hists = get_standard_normalized_histogram_ST(batch_std_labels, adjust_softmax=False)
                wass_dict_std_dists[qid] = batch_std_hists
        else:
            batch_std_hists = get_standard_normalized_histogram_ST(batch_std_labels, adjust_softmax=False)

        if 'BothST' == norm_type:
            if 'S' == TL_AF or 'ST' == TL_AF:  # first convert to the same relevance level
                max_rele_level = torch.max(batch_std_labels)
                batch_preds = batch_preds * max_rele_level

            batch_pred_hists = F.softmax(batch_preds, dim=1)
        else:
            raise NotImplementedError

    elif 'NG' == smooth_type: # normalization of gain, i.e., gain/sum_gain
        if wass_dict_std_dists is not None:
            if qid in wass_dict_std_dists:  # target distributions
                batch_std_hists = wass_dict_std_dists[qid]
            else:
                batch_std_hists = get_standard_normalized_histogram_GN(batch_std_labels)
                wass_dict_std_dists[qid] = batch_std_hists
        else:
            batch_std_hists = get_standard_normalized_histogram_GN(batch_std_labels)

        #print(batch_std_hists.size())
        #print('batch_std_hists', batch_std_hists)

        ''' normalizing predictions, where negative values should be taken into account '''
        mini = torch.min(batch_preds)
        # print('mini', mini)

        # (1)
        #'''
        if mini > 0.0:
            batch_pred_hists = batch_preds / torch.sum(batch_preds, dim=1).view(-1, 1)
        else:
            batch_preds = batch_preds - mini
            batch_pred_hists = batch_preds / torch.sum(batch_preds, dim=1).view(-1, 1)
        #'''

        # (2) treat all negative values as zero, which seems to be a bad way
        '''
        if mini > 0.0:
            batch_pred_hists = batch_preds / torch.sum(batch_preds, dim=1).view(-1, 1)
        else:
            batch_deltas = -torch.clamp(batch_preds, min=-1e3, max=0)
            batch_preds  = batch_preds + batch_deltas
            batch_pred_hists = batch_preds / torch.sum(batch_preds, dim=1).view(-1, 1)
        '''

        #print(batch_pred_hists.size())
        #print('batch_pred_hists', batch_pred_hists)
    else:
        raise NotImplementedError

    return batch_std_hists, batch_pred_hists