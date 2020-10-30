#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc_metric import torch_dcg_at_k
from ptranking.ltr_global import global_gpu as gpu, tensor

#######
# For Delta Metrics
#######

def get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds, label_type=LABEL_TYPE.MultiLabel):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_stds: the standard labels sorted in a descending order
    :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, label_type=label_type)

    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_stds_sorted_via_preds
    else:
        raise NotImplementedError

    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_stds_sorted_via_preds.size(1)).type(tensor)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


def get_sharp_swap_deltas(batch_stds, batch_stds_sorted_via_preds, pos_swap_const=1., neg_swap_const=1.):
    '''
    pure changes w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    pure changes w.r.t. pairwise swapping is given that: (1) (1/D_i - 1/D_j)(G_j - G_i) (2)(G_i - G_j)(1/D_j - 1/D_i)

    :param batch_stds: the standard labels sorted in a descending order
    :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds)                      # ideal discount cumulative gains

    batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_stds_sorted_via_preds.size(1)).type(tensor)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    t_batch_dists_diffs = torch.transpose(batch_dists_diffs, dim0=1, dim1=2)

    batch_swap_ndcg = batch_ng_diffs * t_batch_dists_diffs  # pure changes

    batch_pos_swap_ones = (batch_swap_ndcg > 0).type(tensor) # s_ij is one for positive swap, otherwise 0
    batch_pos_swap_cofs = batch_pos_swap_ones * pos_swap_const

    batch_neg_swap_ones = (batch_swap_ndcg < 0).type(tensor) # negative swap means that the current pairwise order is consistent with the standard order
    batch_neg_swap_cofs = batch_neg_swap_ones * neg_swap_const

    batch_all_cofs = batch_pos_swap_cofs + batch_neg_swap_cofs

    #1 what is the meaning?
    #batch_swap_ndcg = torch.clamp(batch_swap_ndcg, min=0.0, max=100000.) # keeping positive swapping
    #batch_swap_streths = batch_swap_ndcg + batch_neg_swap_cofs

    #2
    #batch_delta_ndcg   = torch.abs(batch_swap_ndcg)
    #batch_swap_streths = batch_all_cofs * batch_delta_ndcg

    #3 all constant
    batch_swap_streths = torch.ones_like(batch_swap_ndcg)

    return batch_swap_streths


def metric_results_to_string(list_scores=None, list_cutoffs=None, split_str=', '):
    """
    Convert metric results to a string representation
    :param list_scores:
    :param list_cutoffs:
    :param split_str:
    :return:
    """
    list_str = []
    for i in range(len(list_scores)):
        list_str.append('nDCG@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)
