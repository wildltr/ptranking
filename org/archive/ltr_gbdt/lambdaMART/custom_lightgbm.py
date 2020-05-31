#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 19/06/19 | https://y-research.github.io

"""Description

"""

import numpy as np

from org.archive.ltr_adhoc.listwise.lambdaMART.custom_util import per_query_gradient_hessian_lambda, per_query_gradient_hessian_listnet

def lightgbm_test_custom_obj(preds, train_data):
    #print(type(preds))
    #print(type(train_data))
    #print(train_data.shape)

    labels = train_data.get_label()
    group = train_data.get_group()

    #print(type(preds), preds.shape)
    #print(type(labels), labels.shape)
    #print(type(group), group.shape)
    #print()
    #print('labels', labels.shape)
    #print('group', group.shape)
    #print('group-sum', np.sum(group))

    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)

    return grad, hess




def lightgbm_custom_obj_ranknet(preds, train_data, first_order=False, constant_hessian=1.0):
    '''
    The traditional ranknet
    :param preds:      numpy.ndarray of shape (size_data, )
    :param train_data:
    :return:
    '''
    all_labels = train_data.get_label() # numpy.ndarray of shape (size_data, )
    group = train_data.get_group()      # numpy.ndarray of shape (num_queries, )

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group.astype(np.int):
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query  = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='All', epsilon=1.0, weighting=False)

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def lightgbm_custom_obj_ranknet_sharp(preds, train_data, first_order=False, constant_hessian=1.0):
    '''
    The traditional ranknet
    :param preds:      numpy.ndarray of shape (size_data, )
    :param train_data:
    :return:
    '''
    all_labels = train_data.get_label() # numpy.ndarray of shape (size_data, )
    group = train_data.get_group()      # numpy.ndarray of shape (num_queries, )

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group.astype(np.int):
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query  = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='NoTies', epsilon=1.0, weighting=True, weighting_type='DeltaGain')

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def lightgbm_custom_obj_lambdarank(preds, train_data, first_order=False, constant_hessian=1.0):
    '''
    :param preds:      numpy.ndarray of shape (size_data, )
    :param train_data:
    :return:
    '''
    all_labels = train_data.get_label()  # numpy.ndarray of shape (size_data, )
    group = train_data.get_group()       # numpy.ndarray of shape (num_queries, )

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group.astype(np.int):
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='NoTies', epsilon=1.0, weighting=True, weighting_type='DeltaNDCG')

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def lightgbm_custom_obj_listnet(preds, train_data, first_order=False, constant_hessian=1.0):
    '''
    :param preds:      numpy.ndarray of shape (size_data, )
    :param train_data:
    :return:
    '''
    all_labels = train_data.get_label()  # numpy.ndarray of shape (size_data, )
    group = train_data.get_group()       # numpy.ndarray of shape (num_queries, )

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group.astype(np.int):
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_listnet(preds=preds_per_query, labels=labels_per_query, gain_type='Power', first_order=first_order) # Power, Label

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess



import torch
from org.archive.metric.adhoc_metric import torch_nDCG_at_k
def custom_eval_ndcg(preds, train_data):
    all_labels = train_data.get_label()  # numpy.ndarray of shape (size_data, )
    group = train_data.get_group()  # numpy.ndarray of shape (num_queries, )

    sum_ndcg_at_k = torch.zeros(1)
    cnt = torch.zeros(1)

    tor_all_std_labels, tor_all_preds = torch.from_numpy(all_labels.astype(np.float32)), torch.from_numpy(preds.astype(np.float32))
    #tor_all_std_labels, tor_all_preds = tor_all_std_labels.double(), tor_all_preds.double()
    #print(tor_all_std_labels)
    #print(tor_all_preds)

    head = 0
    group = group.astype(np.int).tolist()
    for gr in group:
        tor_per_query_std_labels = tor_all_std_labels[head:head+gr]
        tor_per_query_preds = tor_all_preds[head:head+gr]
        head += gr

        _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

        sys_sorted_labels = tor_per_query_std_labels[tor_sorted_inds]
        ideal_sorted_labels, _ = torch.sort(tor_per_query_std_labels, descending=True)
        #print(ideal_sorted_labels)

        ndcg_at_k = torch_nDCG_at_k(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, k=5, multi_level_rele=True)
        #print(ndcg_at_ks)

        sum_ndcg_at_k = torch.add(sum_ndcg_at_k, ndcg_at_k)
        cnt += 1

    tor_avg_ndcg_at_k = sum_ndcg_at_k / cnt
    avg_ndcg_at_k = tor_avg_ndcg_at_k.data.numpy()

    return 'myndcg', avg_ndcg_at_k, True


if __name__ == '__main__':
    # test mat_delta_ndcg

    """
    
    ideally_sorted_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    labels_sorted_via_preds = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])

    '''
    standard results: delta_ndcg[0, 3] = 0.222
    standard results: delta_ndcg[0, 4] = 0.239
    standard results: delta_ndcg[0, 6] = 0.260
    standard results: delta_ndcg[0, 7] = 0.267
    '''

    mat_delta_ndcg = get_delta_ndcg(ideally_sorted_labels=ideally_sorted_labels, labels_sorted_via_preds=labels_sorted_via_preds)
    print(mat_delta_ndcg[0, :])

    preds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    test_desc_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int)
    # standard result: [0.495, -0.206, -0.104, 0.231, 0.231, -0.033, 0.240, 0.247, -0.051, -0.061]
    grad, hess = per_query_gradient_hessian_lambda(preds=None, labels=labels_sorted_via_preds, test_desc_inds=test_desc_inds, test_preds=preds, lambda_weighting=True)
    print(grad)
    
    """


