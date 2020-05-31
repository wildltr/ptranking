#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

from org.archive.metric.adhoc_metric import torch_nDCG_at_k, torch_nDCG_at_ks, EMD_at_k

from org.archive.l2r_global import global_gpu as gpu, global_device as device


def ndcg_at_k(ranker=None, test_data=None, k=10, multi_level_rele=True, batch_mode=True):
    '''
    There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
    and at least one relevant document. Or there will be errors.
    '''
    sum_ndcg_at_k = torch.zeros(1)
    cnt = torch.zeros(1)
    for qid, batch_ranking, batch_label in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
        std_labels = torch.squeeze(batch_label)               # remove batch dimension for evaluation
        if std_labels.size(0) < k: continue	                       # skip the query if the number of associated documents is smaller than k

        if gpu: batch_ranking = batch_ranking.to(device)

        if batch_mode:
            batch_rele_preds = ranker.predict(batch_ranking)
            rele_preds = torch.squeeze(batch_rele_preds)
        else:
            rele_preds = ranker.predict(torch.squeeze(batch_ranking))
            rele_preds = torch.squeeze(rele_preds)

        if gpu: rele_preds = rele_preds.cpu()

        _, sorted_inds = torch.sort(rele_preds, descending=True)

        sys_sorted_labels = std_labels[sorted_inds]
        ideal_sorted_labels, _ = torch.sort(std_labels, descending=True)

        ndcg_at_k = torch_nDCG_at_k(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, k=k, multi_level_rele=multi_level_rele)
        sum_ndcg_at_k += ndcg_at_k
        cnt += 1

    avg_ndcg_at_k = sum_ndcg_at_k/cnt
    return  avg_ndcg_at_k


def ndcg_at_ks(ranker=None, test_data=None, ks=[1, 5, 10], multi_level_rele=True, batch_mode=True):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)
    for qid, batch_ranking, batch_label in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if gpu: batch_ranking = batch_ranking.to(device)

        if batch_mode:
            batch_rele_preds = ranker.predict(batch_ranking)
            rele_preds = torch.squeeze(batch_rele_preds)
        else:
            rele_preds = ranker.predict(torch.squeeze(batch_ranking))
            rele_preds = torch.squeeze(rele_preds)

        if gpu: rele_preds = rele_preds.cpu()

        std_labels = torch.squeeze(batch_label)

        _, sorted_inds = torch.sort(rele_preds, descending=True)

        sys_sorted_labels      = std_labels[sorted_inds]
        ideal_sorted_labels, _ = torch.sort(std_labels, descending=True)

        ndcg_at_ks = torch_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=multi_level_rele)
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks)
        cnt += 1

    avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
    return avg_ndcg_at_ks


def emd_at_k(ranker=None, test_Qs=None, k=10, TL_AF=None, multi_level_rele=True):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    assert 'S'==TL_AF or 'ST'==TL_AF

    sum_emd = 0.0
    cnt = 0
    for entry in test_Qs:
        tor_test_ranking, tor_test_std_label_vec = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0)  # remove the size 1 of dim=0 from loader itself
        if tor_test_std_label_vec.size(0) < k:
            continue

        if gpu:
            tor_test_ranking = tor_test_ranking.to(device)
            tor_rele_pred = ranker(tor_test_ranking)
            tor_rele_pred = torch.squeeze(tor_rele_pred)
            tor_rele_pred = tor_rele_pred.cpu()
        else:
            tor_rele_pred = ranker(tor_test_ranking)
            tor_rele_pred = torch.squeeze(tor_rele_pred)

        ideal_desc_labels, ideal_sorted_inds = torch.sort(tor_test_std_label_vec, descending=True)
        sys_corresponding_scores = tor_rele_pred[ideal_sorted_inds]

        tor_max_rele_level = torch.max(ideal_desc_labels)
        sys_corresponding_scores = sys_corresponding_scores * tor_max_rele_level

        emd_v = EMD_at_k(k=k, ideal_desc_labels=ideal_desc_labels[0:k].numpy(), sys_corresponding_scores=sys_corresponding_scores[0:k].numpy())
        sum_emd += emd_v

        cnt += 1

    avg_emd = sum_emd/cnt
    return avg_emd  # averaged value

if __name__ == '__main__':
    #1 test idcg_std
    import numpy as np

    batch_mapped_S = torch.from_numpy(np.asarray([2.0, 3, 3, 1])).view(1, -1).type(torch.FloatTensor)
    #print(idcg_std(batch_mapped_S))