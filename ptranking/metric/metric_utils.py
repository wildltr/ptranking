#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""
import re

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc.adhoc_metric import torch_dcg_at_k

#######
# For Delta Metrics
#######

def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_ideal_rankings: the standard labels sorted in a descending order
    :param batch_predicted_rankings: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, label_type=label_type, device=device)

    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_predict_rankings
    else:
        raise NotImplementedError

    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_predict_rankings.size(1), dtype=torch.float, device=device)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


def metric_results_to_string(list_scores=None, list_cutoffs=None, split_str=', ', metric='nDCG'):
    """
    Convert metric results to a string representation
    :param list_scores:
    :param list_cutoffs:
    :param split_str:
    :return:
    """
    list_str = []
    for i in range(len(list_scores)):
        list_str.append(metric + '@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key, reverse=True)

def test_sort():
    tmp_list = ['net_params_epoch_2.pkl', 'net_params_epoch_34.pkl', 'net_params_epoch_8.pkl']
    print(sort_nicely(tmp_list))
    print(tmp_list)


def get_opt_model(list_model_names):
    sort_nicely(list_model_names)
    return list_model_names[0]
