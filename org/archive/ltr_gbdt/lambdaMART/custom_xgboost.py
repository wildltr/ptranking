#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 19/06/21 | https://y-research.github.io

"""Description

"""

import numpy as np

from xgboost import DMatrix

from org.archive.ltr_adhoc.listwise.lambdaMART.custom_util import per_query_gradient_hessian_lambda, per_query_gradient_hessian_listnet

class GDMatrix(DMatrix):
    """
    Wrapper DMatrix of for getting group information
    """
    def __init__(self, data, label=None, group=None, missing=None, weight=None, silent=False,
                 feature_names=None, feature_types=None, nthread=None):
        super(GDMatrix, self).__init__(data=data, label=label, missing=missing, weight=weight, silent=silent,
                                       feature_names=feature_names, feature_types=feature_types, nthread=nthread)
        self.group = group

    def get_group(self):
        return self.group



def xgboost_logregobj(preds, dtrain, x=None):
    labels = dtrain.get_label()

    #print(dtrain)
    #print('labels', labels.shape)
    #print(dtrain.num_col())
    #print(dtrain.num_row())

    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)

    return grad, hess


def xgboost_custom_obj_ranknet(preds, dtrain, first_order=False, constant_hessian=1.0):
    '''
    The traditional ranknet
    :param preds:  numpy.ndarray of shape (size_data, )
    :param dtrain: class xgboost.DMatrix
    :return:
    '''
    all_labels = dtrain.get_label() # numpy.ndarray of shape (size_data, )
    group = dtrain.get_group() # list

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group:
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query  = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='All', epsilon=1.0, weighting=False)

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def xgboost_custom_obj_ranknet_sharp(preds, dtrain, first_order=False, constant_hessian=1.0):
    '''
    The traditional ranknet
    :param preds:   numpy.ndarray of shape (size_data, )
    :param dtrain:  class xgboost.DMatrix
    :return:
    '''
    all_labels = dtrain.get_label() # numpy.ndarray of shape (size_data, )
    group = dtrain.get_group()

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group:
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query  = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='NoTies', epsilon=1.0, weighting=True, weighting_type='DeltaGain')

        all_grad[head:head + num_docs_per_query] = grad_per_query

        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def xgboost_custom_obj_lambdarank(preds, dtrain, first_order=False, constant_hessian=1.0):
    '''
    :param preds:   numpy.ndarray of shape (size_data, )
    :param dtrain:  class xgboost.DMatrix
    :return:
    '''
    all_labels = dtrain.get_label()  # numpy.ndarray of shape (size_data, )
    group = dtrain.get_group()

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group:
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_lambda(preds=preds_per_query, labels=labels_per_query, first_order=first_order, pair_type='NoTies', epsilon=1.0, weighting=True, weighting_type='DeltaNDCG')

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess


def xgboost_custom_obj_listnet(preds, dtrain, first_order=False, constant_hessian=1.0):
    '''
    :param preds:  numpy.ndarray of shape (size_data, )
    :param dtrain: class xgboost.DMatrix
    :return:
    '''
    all_labels = dtrain.get_label()  # numpy.ndarray of shape (size_data, )
    group = dtrain.get_group()

    size_data = len(all_labels)
    if first_order:
        all_grad, all_hess = np.zeros((size_data,)), np.full((size_data,), fill_value=constant_hessian)
    else:
        all_grad, all_hess = np.zeros((size_data,)), np.zeros((size_data,))

    head = 0
    for num_docs_per_query in group:
        labels_per_query = all_labels[head:head + num_docs_per_query]
        preds_per_query = preds[head:head + num_docs_per_query]

        grad_per_query, hess_per_query = per_query_gradient_hessian_listnet(preds=preds_per_query, labels=labels_per_query, gain_type='Power', first_order=first_order) # Power, Label

        all_grad[head:head + num_docs_per_query] = grad_per_query
        if not first_order: all_hess[head:head + num_docs_per_query] = hess_per_query

        head += num_docs_per_query

    return all_grad, all_hess