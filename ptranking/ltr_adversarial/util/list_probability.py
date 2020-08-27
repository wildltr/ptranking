#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

from ptranking.ltr_adhoc.listwise.listmle import apply_LogCumsumExp


"""
Plackett_Luce
"""

def ranking_prob_Plackett_Luce(batch_preds):
    batch_log_prob = log_ranking_prob_Plackett_Luce(batch_preds)
    batch_prob = torch.exp(batch_log_prob)

    return batch_prob


def log_ranking_prob_Plackett_Luce(batch_preds):
    assert 2 == len(batch_preds.size())

    batch_logcumsumexps = apply_LogCumsumExp(batch_preds)
    batch_log_prob = torch.sum(batch_preds - batch_logcumsumexps, dim=1)

    return batch_log_prob


"""
Bradley_Terry
"""

def ranking_prob_Bradley_Terry(batch_preds):
    batch_log_ranking_prob = log_ranking_prob_Bradley_Terry(batch_preds)
    batch_BT_ranking_prob = torch.exp(batch_log_ranking_prob)

    return batch_BT_ranking_prob


def log_ranking_prob_Bradley_Terry(batch_preds):
    '''
    :param batch_preds: [batch_size, list_size]
    :return:
    '''
    assert 2 == len(batch_preds.size())

    max_v = torch.max(batch_preds)
    new_batch_preds = torch.exp(batch_preds - max_v)

    batch_numerators = torch.unsqueeze(new_batch_preds, dim=2).repeat(1, 1, batch_preds.size(1))

    batch_denominaotrs = torch.unsqueeze(new_batch_preds, dim=2) + torch.unsqueeze(new_batch_preds, dim=1)

    batch_BT_probs = batch_numerators / batch_denominaotrs

    batch_log_ranking_prob = torch.sum(torch.sum(torch.triu(torch.log(batch_BT_probs), diagonal=1), dim=2), dim=1)

    return batch_log_ranking_prob
