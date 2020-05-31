#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

import numpy as np

def triu_indice(k=1, pair_type='NoTies', labels=None):
    '''
    Get unique document pairs being consistent with the specified pair_type. This function is used to avoid duplicate computation.

    All:        pairs including both pairs of documents across different relevance levels and
                pairs of documents having the same relevance level.
    NoTies:   the pairs consisting of two documents of the same relevance level are removed
    No00:     the pairs consisting of two non-relevant documents are removed

    :param batch_mats: [batch, m, m]
    :param k: the offset w.r.t. the diagonal line: k=0 means including the diagonal line, k=1 means upper triangular part without the diagonal line
    :return:
    '''
    #assert pair_type in PAIR_TYPE

    m = len(labels) # the number of documents
    if pair_type == 'All':
        row_inds, col_inds = np.triu_indices(m, k=k)

    elif pair_type == 'No00':
        row_inds, col_inds = np.triu_indices(m, k=k)

        pairs = [e for e in zip(row_inds, col_inds) if not (0==labels[e[0]] and 0==labels[e[1]])] # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == '00': # the pairs consisting of two non-relevant documents
        row_inds, col_inds = np.triu_indices(m, k=k)

        pairs = [e for e in zip(row_inds, col_inds) if (0 == labels[e[0]] and 0 == labels[e[1]])]  # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == 'NoTies':
        row_inds, col_inds = np.triu_indices(m, k=k)

        pairs = [e for e in zip(row_inds, col_inds) if labels[e[0]]!=labels[e[1]]]  # remove pairs of documents of the same level
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    else:
        raise NotImplementedError

    return row_inds, col_inds

def sigmoid(x, epsilon=1.0):
    sigm = 1. / (1. + np.exp(-x*epsilon))
    return sigm


def ideal_dcg(ideally_sorted_labels):
    '''

    :param ideally_sorted_labels:
    :return:
    '''
    gains = np.power(2.0, ideally_sorted_labels) - 1.0
    ranks = np.arange(len(ideally_sorted_labels)) + 1.0

    discounts = np.log2(1.0 + ranks)
    ideal_dcg = np.sum(gains / discounts)

    return ideal_dcg

def get_delta_gains(labels_sorted_via_preds):
    gains = np.power(2.0, labels_sorted_via_preds) - 1.0
    gain_diffs = np.expand_dims(gains, axis=1) - np.expand_dims(gains, axis=0)
    delta_gain = np.abs(gain_diffs)  # absolute delta gains w.r.t. pairwise swapping

    return delta_gain

def get_delta_ndcg(ideally_sorted_labels, labels_sorted_via_preds):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    '''
    idcg = ideal_dcg(ideally_sorted_labels) # ideal discount cumulative gains

    gains = np.power(2.0, labels_sorted_via_preds) - 1.0
    n_gains = gains / idcg                  # normalised gains
    ng_diffs = np.expand_dims(n_gains, axis=1) - np.expand_dims(n_gains, axis=0)

    ranks = np.arange(len(labels_sorted_via_preds)) + 1.0
    dists = 1.0 / np.log2(ranks + 1.0)      # discount co-efficients
    dists_diffs = np.expand_dims(dists, axis=1) - np.expand_dims(dists, axis=0)
    mat_delta_ndcg = np.abs(ng_diffs) * np.abs(dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return mat_delta_ndcg


WEIGHTING_TYPE = ['DeltaNDCG', 'DeltaGain']

def per_query_gradient_hessian_lambda(preds=None, labels=None, first_order=False, weighting=False, weighting_type='DeltaNDCG', pair_type='NoTies', epsilon=1.0):
    '''
    Compute the corresponding gradient & hessian
    cf. LightGBM https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp
    cf. XGBoost  https://github.com/dmlc/xgboost/blob/master/src/objective/rank_obj.cc

    :param preds:  1-dimension predicted scores
    :param labels: 1-dimension ground truth
    :return:
    '''
    desc_inds = np.flip(np.argsort(preds)) # indice that sort the preds in a descending order

    system_sorted_preds  = preds[desc_inds]
    labels_sorted_via_preds = labels[desc_inds]

    row_inds, col_inds = triu_indice(labels=labels_sorted_via_preds, k=1, pair_type=pair_type)

    # prediction difference
    mat_s_ij = np.expand_dims(system_sorted_preds, axis=1) - np.expand_dims(system_sorted_preds, axis=0)
    # S_ij in {-1, 0, 1} is the standard indicator
    mat_S_ij = np.expand_dims(labels_sorted_via_preds, axis=1) - np.expand_dims(labels_sorted_via_preds, axis=0)
    mat_S_ij = np.clip(mat_S_ij, a_min=-1.0, a_max=1.0)

    num_docs, num_pairs = len(labels), len(row_inds)
    if first_order:
        grad = np.zeros((num_docs,))
    else:
        grad, hess = np.zeros((num_docs,)), np.zeros((num_docs,))

    if weighting and weighting in WEIGHTING_TYPE:
        if weighting_type == 'DeltaNDCG':
            ideally_sorted_labels = np.flip(np.sort(labels))
            mat_weights = get_delta_ndcg(ideally_sorted_labels=ideally_sorted_labels, labels_sorted_via_preds=labels_sorted_via_preds)

        elif weighting_type == 'DeltaGain':
            mat_weights = get_delta_gains(labels_sorted_via_preds=labels_sorted_via_preds)

    for i in range(num_pairs): # iterate over pairs
        r, c = row_inds[i], col_inds[i]
        s_ij = mat_s_ij[r, c]
        S_ij = mat_S_ij[r, c]

        lambda_ij = epsilon*(sigmoid(s_ij, epsilon=epsilon) - 0.5*(1.0+S_ij)) # gradient w.r.t. s_i
        if weighting and weighting in WEIGHTING_TYPE: lambda_ij *= mat_weights[r, c] # delta metric variance

        lambda_ji = - lambda_ij # gradient w.r.t. s_j

        grad[desc_inds[r]] += lambda_ij  # desc_inds[r] denotes the original index of the document currently being at r-th position after a full-descending-ordering by predictions
        grad[desc_inds[c]] += lambda_ji

        if not first_order: # 2nd order hessian
            lambda_ij_2order = np.power(epsilon, 2.0) * sigmoid(s_ij) * (1.0-sigmoid(s_ij))
            lambda_ij_2order = np.maximum(lambda_ij_2order, 1e-16) # trick as XGBoost https://github.com/dmlc/xgboost/blob/master/src/objective/rank_obj.cc
            if weighting and weighting in WEIGHTING_TYPE: lambda_ij_2order *= mat_weights[r, c]

            lambda_ji_2order = -lambda_ij_2order

            hess[desc_inds[r]] += lambda_ij_2order
            hess[desc_inds[c]] += lambda_ji_2order

    if first_order:
        return grad, None
    else:
        return grad, hess


def _softmax(x):
    ''' A numberically stable way '''
    x = x - np.max(x)
    exp_x = np.exp(x)
    res = exp_x/np.sum(exp_x)

    return res

GAIN_TYPE = ['Power', 'Label']

def per_query_gradient_hessian_listnet(preds=None, labels=None, gain_type='Power', first_order=False):
    '''
    Compute the corresponding gradient & hessian
    cf. LightGBM https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp
    cf. XGBoost  https://github.com/dmlc/xgboost/blob/master/src/objective/rank_obj.cc

    :param preds:  1-dimension predicted scores
    :param labels: 1-dimension ground truth
    :return:
    '''
    assert gain_type in GAIN_TYPE

    if 'Power' == gain_type:
        gains = np.power(2.0, labels) - 1.0
    elif 'Label' == gain_type:
        gains = labels

    p_pred, p_truth = _softmax(preds), _softmax(gains)

    grad = p_pred - p_truth
    hess = None if first_order else p_pred * (1.0-p_pred)

    return grad, hess