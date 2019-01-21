#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import ot
import numpy as np

import torch

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device


""" Evaluation Metrics """

def rele_gain(rele_level, gain_base=2.0):
	gain = np.power(gain_base, rele_level) - 1.0
	return gain


""" Precision """
def tor_p_at_ks(sys_sorted_labels, ks=None):
    '''	precision at ks
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    rele_ones = torch.ones(max_cutoff)
    non_rele_zeros = torch.zeros(max_cutoff)
    positions = torch.arange(max_cutoff) + 1.0

    sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
    binarized_sys = torch.where(sys_sorted_labels > 0, rele_ones, non_rele_zeros)
    cumu_binarized_sys = torch.cumsum(binarized_sys, dim=0)

    sys_positionwise_precision = cumu_binarized_sys/positions
    sys_p_at_ks = sys_positionwise_precision[inds]
    if valid_max < max(ks):
        padded_p_at_ks = torch.zeros(len(ks))
        padded_p_at_ks[0:len(used_ks)] = sys_p_at_ks
        return padded_p_at_ks
    else:
        return sys_p_at_ks

""" Average Precision """
def tor_ap_at_ks(sys_sorted_labels, ks=None):
    ''' average precision at ks
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    rele_ones = torch.ones(max_cutoff)
    non_rele_zeros = torch.zeros(max_cutoff)
    positions = torch.arange(max_cutoff) + 1.0

    sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
    binarized_sys = torch.where(sys_sorted_labels > 0, rele_ones, non_rele_zeros)
    cumu_binarized_sys = torch.cumsum(binarized_sys, dim=0)

    sys_poswise_precision = cumu_binarized_sys / positions
    zeroed_sys_poswise_precision = torch.where(sys_sorted_labels>0, sys_poswise_precision, non_rele_zeros)    # for non-rele positions, use zero rather than cumulative sum of precisions

    cumsum_sys_poswise_precision = torch.cumsum(zeroed_sys_poswise_precision, dim=0)
    sys_poswise_ap = cumsum_sys_poswise_precision / positions
    sys_ap_at_ks = sys_poswise_ap[inds]

    if valid_max < max(ks):
        padded_ap_at_ks = torch.zeros(len(ks))
        padded_ap_at_ks[0:len(used_ks)] = sys_ap_at_ks
        return padded_ap_at_ks
    else:
        return sys_ap_at_ks


""" ERR """
def tor_err_at_ks(sys_sorted_labels, ks=None, multi_level_rele=True, max_rele_level=None):
    '''
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :param multi_level_rele:
    :param max_rele_level:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    if multi_level_rele:
        positions = torch.arange(max_cutoff) + 1.0
        expt_ranks = 1.0 / positions    # expected stop positions

        tor_max_rele = torch.Tensor([max_rele_level]).float()
        satis_pros = (torch.pow(2.0, sys_sorted_labels[0:max_cutoff]) - 1.0)/torch.pow(2.0, tor_max_rele)
        non_satis_pros = torch.ones(max_cutoff) - satis_pros
        cum_non_satis_pros = torch.cumprod(non_satis_pros, dim=0)

        cascad_non_satis_pros = positions
        cascad_non_satis_pros[1:max_cutoff] = cum_non_satis_pros[0:max_cutoff-1]
        expt_satis_ranks = expt_ranks * satis_pros * cascad_non_satis_pros  # w.r.t. all rank positions
        err_at_ranks = torch.cumsum(expt_satis_ranks, dim=0)

        err_at_ks = err_at_ranks[inds]
        if valid_max < max(ks):
            padded_err_at_ks = torch.zeros(len(ks))
            padded_err_at_ks[0:len(used_ks)] = err_at_ks
            return padded_err_at_ks
        else:
            return err_at_ks
    else:
        raise NotImplementedError

""" nDCG """
def tor_discounted_cumu_gain_at_k(sorted_labels, cutoff, multi_level_rele=True):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
    :param max_cutoff: the maximum rank position to be considered
    :param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: cumulative gains for each rank position
    '''
    if multi_level_rele:    #the common case with multi-level labels
        nums = torch.pow(2.0, sorted_labels[0:cutoff]) - 1.0
    else:
        nums = sorted_labels[0:cutoff]  #the case like listwise ranking, where the relevance is labeled as (n-rank_position)

    denoms = torch.log2(torch.arange(cutoff) + 2.0)   #discounting factor
    dited_cumu_gain = torch.sum(nums/denoms)   # discounted cumulative gain value

    return dited_cumu_gain

def tor_discounted_cumu_gain_at_ks(sorted_labels, max_cutoff, multi_level_rele=True):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
    :param max_cutoff: the maximum rank position to be considered
    :param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: cumulative gains for each rank position
    '''

    if multi_level_rele:    #the common case with multi-level labels
        nums = torch.pow(2.0, sorted_labels[0:max_cutoff]) - 1.0
    else:
        nums = sorted_labels[0:max_cutoff]  #the case like listwise ranking, where the relevance is labeled as (n-rank_position)

    denoms = torch.log2(torch.arange(max_cutoff) + 2.0)   #discounting factor
    dited_cumu_gains = torch.cumsum(nums/denoms, dim=0)   # discounted cumulative gain value w.r.t. each position

    return dited_cumu_gains

def tor_nDCG_at_k(sys_sorted_labels, ideal_sorted_labels, k=None, multi_level_rele=True):
    sys_dited_cg_at_k = tor_discounted_cumu_gain_at_k(sys_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)  # only using the cumulative gain at the final rank position
    ideal_dited_cg_at_k = tor_discounted_cumu_gain_at_k(ideal_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)
    ndcg_at_k = sys_dited_cg_at_k / ideal_dited_cg_at_k
    return ndcg_at_k

def tor_nDCG_at_ks(sys_sorted_labels, ideal_sorted_labels, ks=None, multi_level_rele=True):
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k<=valid_max] if valid_max < max(ks) else ks

    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    sys_dited_cgs = tor_discounted_cumu_gain_at_ks(sys_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
    sys_dited_cg_at_ks = sys_dited_cgs[inds]  # get cumulative gains at specified rank positions
    ideal_dited_cgs = tor_discounted_cumu_gain_at_ks(ideal_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
    ideal_dited_cg_at_ks = ideal_dited_cgs[inds]

    ndcg_at_ks = sys_dited_cg_at_ks / ideal_dited_cg_at_ks

    if valid_max < max(ks):
        padded_ndcg_at_ks = torch.zeros(len(ks))
        padded_ndcg_at_ks[0:len(used_ks)] = ndcg_at_ks
        return padded_ndcg_at_ks
    else:
        return ndcg_at_ks


def np_metric_at_ks(ranker=None, test_Qs=None, ks=[1, 5, 10], multi_level_rele=True, max_rele_level=None):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    cnt = 0
    sum_ndcg_at_ks = torch.zeros(len(ks))
    sum_err_at_ks = torch.zeros(len(ks))
    sum_p_at_ks = torch.zeros(len(ks))

    list_ndcg_at_ks_per_q = []
    list_err_at_ks_per_q = []
    list_p_at_ks_per_q = []

    for entry in test_Qs:
        tor_test_ranking, tor_test_std_label_vec = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0)  # remove the size 1 of dim=0 from loader itself

        if gpu:
            tor_rele_pred = ranker(tor_test_ranking.to(device))
            tor_rele_pred = torch.squeeze(tor_rele_pred)
            tor_rele_pred = tor_rele_pred.cpu()
        else:
            tor_rele_pred = ranker(tor_test_ranking)
            tor_rele_pred = torch.squeeze(tor_rele_pred)

        _, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)

        sys_sorted_labels = tor_test_std_label_vec[tor_sorted_inds]
        ideal_sorted_labels, _ = torch.sort(tor_test_std_label_vec, descending=True)

        ndcg_at_ks_per_query = tor_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=multi_level_rele)
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks_per_query)
        list_ndcg_at_ks_per_q.append(ndcg_at_ks_per_query.numpy())

        err_at_ks_per_query = tor_err_at_ks(sys_sorted_labels, ks=ks, multi_level_rele=multi_level_rele, max_rele_level=max_rele_level)
        sum_err_at_ks = torch.add(sum_err_at_ks, err_at_ks_per_query)
        list_err_at_ks_per_q.append(err_at_ks_per_query.numpy())

        p_at_ks_per_query = tor_p_at_ks(sys_sorted_labels=sys_sorted_labels, ks=ks)
        sum_p_at_ks = torch.add(sum_p_at_ks, p_at_ks_per_query)
        list_p_at_ks_per_q.append(p_at_ks_per_query.numpy())

        cnt += 1

    ndcg_at_ks = sum_ndcg_at_ks/cnt
    err_at_ks = sum_err_at_ks/cnt
    p_at_ks = sum_p_at_ks/cnt

    return ndcg_at_ks.numpy(), err_at_ks.numpy(), p_at_ks.numpy(), list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_p_at_ks_per_q


def np_stable_softmax_e(histogram):
    histogram = np.asarray(histogram, dtype=np.float64)
    max_v, _ = np.max(histogram, dim=0)  # a transformation aiming for higher stability when computing softmax() with exp()
    hist = histogram - max_v
    hist_exped = np.exp(hist)
    probs = np.divide(hist_exped, np.sum(hist_exped, dim=0))
    return probs


def eval_cost_mat_group(sorted_std_labels, group_div_cost=np.e, margin_to_non_rele=100.0, rele_gain_base=4.0):
    size_ranking = len(sorted_std_labels)
    cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float64)

    for i in range(size_ranking):
        i_rele_level = sorted_std_labels[i]
        for j in range(size_ranking):
            if i==j:
                cost_mat[i, j] = 0
            else:
                j_rele_level = sorted_std_labels[j]

                if i_rele_level == j_rele_level:
                    cost_mat[i, j] = group_div_cost
                else:
                    cost_mat[i, j] = np.abs(rele_gain(i_rele_level, gain_base=rele_gain_base) - rele_gain(j_rele_level, gain_base=rele_gain_base))
                    if 0 == i_rele_level or 0 == j_rele_level:
                        cost_mat[i, j] += margin_to_non_rele

    return cost_mat


def EMD_at_k(k, ideal_desc_labels, sys_corresponding_scores, group_div_cost=np.e, margin_to_non_rele=100.0, rele_gain_base=4.0):
    if k>len(ideal_desc_labels):
        return 0.0

    cost_mat = eval_cost_mat_group(ideal_desc_labels, group_div_cost=group_div_cost, margin_to_non_rele=margin_to_non_rele, rele_gain_base=rele_gain_base)

    ideal_histogram = np_stable_softmax_e(ideal_desc_labels)
    sys_historgram = np_stable_softmax_e(sys_corresponding_scores)

    # %% EMD
    G0 = ot.emd(a=sys_historgram, b=ideal_histogram, M=cost_mat)
    emd_value = np.sum(G0 * cost_mat)

    return emd_value