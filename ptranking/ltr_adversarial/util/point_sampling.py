#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

def generate_true_docs(qid, truth_label, num_samples, dict_true_inds=None):
    '''
    :param qid:
    :param truth_label: it is fine if including '-1'
    :param num_samples:
    :param dict_true_inds:
    :return:
    '''
    if dict_true_inds is not None and qid in dict_true_inds:
        true_inds, size_unique = dict_true_inds[qid]

        if num_samples is None or num_samples>size_unique:
            num_samples = int(0.5*size_unique)
            if num_samples < 1:
                num_samples = 1

        rand_inds = torch.multinomial(torch.ones(size_unique), num_samples, replacement=False)
        return true_inds[rand_inds]
    else:
        # [z, n] If input has n dimensions, then the resulting indices tensor out is of size (z×n), where z is the total number of non-zero elements in the input tensor.
        true_inds = torch.nonzero(torch.gt(truth_label, 0), as_tuple=False)

        size_unique = true_inds.size(0)
        true_inds = true_inds[:, 0]
        if dict_true_inds is not None: #buffer
            dict_true_inds[qid] = (true_inds, size_unique)

        if num_samples is None or num_samples>size_unique:
            num_samples = int(0.5*size_unique)
            if num_samples < 1:
                num_samples = 1

        rand_inds = torch.multinomial(torch.ones(size_unique), num_samples, replacement=False)
        return true_inds[rand_inds]