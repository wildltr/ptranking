#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc_metric import torch_dcg_at_k

#######
# For Delta Metrics
#######

def get_delta_ndcg(batch_ideally_sorted_stds, batch_stds_sorted_via_preds, label_type=LABEL_TYPE.MultiLabel, gpu=False):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_stds: the standard labels sorted in a descending order
    :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_ideally_sorted_stds, label_type=label_type, gpu=gpu)

    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_stds_sorted_via_preds
    else:
        raise NotImplementedError

    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_stds_sorted_via_preds.size(1)).type(torch.cuda.FloatTensor) if gpu \
                                            else torch.arange(batch_stds_sorted_via_preds.size(1))
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


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
