#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Given the neural ranker, compute nDCG values.
"""

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc_metric import torch_nDCG_at_k, torch_nDCG_at_ks

def ndcg_at_k(ranker=None, test_data=None, k=10, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
    and at least one relevant document. Or there will be errors.
    '''
    sum_ndcg_at_k = torch.zeros(1)
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if batch_labels.size(1) < k: continue # skip the query if the number of associated documents is smaller than k

        if gpu: batch_ranking = batch_ranking.to(device)
        batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_k = torch_nDCG_at_k(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                          batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                          k = k, label_type=label_type)

        sum_ndcg_at_k += torch.squeeze(batch_ndcg_at_k) # default batch_size=1 due to testing data
        cnt += 1

    avg_ndcg_at_k = sum_ndcg_at_k/cnt
    return  avg_ndcg_at_k

def ndcg_at_ks(ranker=None, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if gpu: batch_ranking = batch_ranking.to(device)
        batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                            batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                            ks=ks, label_type=label_type)

        # default batch_size=1 due to testing data
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.squeeze(batch_ndcg_at_ks, dim=0))
        cnt += 1

    avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
    return avg_ndcg_at_ks
