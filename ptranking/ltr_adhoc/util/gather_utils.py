#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch

import numpy as np


#######
# For Pair Extraction
#######

PAIR_TYPE = ['All', 'NoTies', 'No00', '00', 'Inversion']

def torch_batch_triu(batch_mats=None, k=0, pair_type='All', batch_std_labels=None, gpu=False, device=None):
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
    assert batch_mats.size(1) == batch_mats.size(2)
    assert pair_type in PAIR_TYPE

    m = batch_mats.size(1) # the number of documents

    if pair_type == 'All':
        row_inds, col_inds = np.triu_indices(m, k=k)

    elif pair_type == 'No00':
        assert batch_std_labels.size(0) == 1

        row_inds, col_inds = np.triu_indices(m, k=k)
        std_labels = torch.squeeze(batch_std_labels, 0)
        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()

        pairs = [e for e in zip(row_inds, col_inds) if not (0 == labels[e[0]] and 0 == labels[e[1]])]  # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == 'NoTies':
        assert batch_std_labels.size(0) == 1
        std_labels = torch.squeeze(batch_std_labels, 0)

        row_inds, col_inds = np.triu_indices(m, k=k)

        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()
        pairs = [e for e in zip(row_inds, col_inds) if labels[e[0]]!=labels[e[1]]]  # remove pairs of documents of the same level
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    #tor_row_inds = torch.LongTensor(row_inds).to(device) if gpu else torch.LongTensor(row_inds)
    #tor_col_inds = torch.LongTensor(col_inds).to(device) if gpu else torch.LongTensor(col_inds)
    tor_row_inds = torch.LongTensor(row_inds, device)
    tor_col_inds = torch.LongTensor(col_inds, device)
    batch_triu = batch_mats[:, tor_row_inds, tor_col_inds]

    return batch_triu # shape: [batch_size, number of pairs]


def torch_triu_indice(k=0, pair_type='All', batch_label=None, gpu=False, device=None):
    '''
    Get unique document pairs being consistent with the specified pair_type. This function is used to avoid duplicate computation.

    All:        pairs including both pairs of documents across different relevance levels and
                pairs of documents having the same relevance level.
    NoTies:     the pairs consisting of two documents of the same relevance level are removed
    No00:       the pairs consisting of two non-relevant documents are removed
    Inversion:  the pairs that are inverted order, i.e., the 1st doc is less relevant than the 2nd doc

    :param batch_mats: [batch, m, m]
    :param k: the offset w.r.t. the diagonal line: k=0 means including the diagonal line, k=1 means upper triangular part without the diagonal line
    :return:
    '''
    assert pair_type in PAIR_TYPE

    m = batch_label.size(1) # the number of documents
    if pair_type == 'All':
        row_inds, col_inds = np.triu_indices(m, k=k)

    elif pair_type == 'No00':
        assert batch_label.size(0) == 1

        row_inds, col_inds = np.triu_indices(m, k=k)
        std_labels = torch.squeeze(batch_label, 0)
        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()

        pairs = [e for e in zip(row_inds, col_inds) if not (0==labels[e[0]] and 0==labels[e[1]])] # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == '00': # the pairs consisting of two non-relevant documents
        assert batch_label.size(0) == 1

        row_inds, col_inds = np.triu_indices(m, k=k)
        std_labels = torch.squeeze(batch_label, 0)
        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()

        pairs = [e for e in zip(row_inds, col_inds) if (0 == labels[e[0]] and 0 == labels[e[1]])]  # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == 'NoTies':
        assert batch_label.size(0) == 1
        std_labels = torch.squeeze(batch_label, 0)

        row_inds, col_inds = np.triu_indices(m, k=k)

        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()
        pairs = [e for e in zip(row_inds, col_inds) if labels[e[0]]!=labels[e[1]]]  # remove pairs of documents of the same level
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == 'Inversion':
        assert batch_label.size(0) == 1
        std_labels = torch.squeeze(batch_label, 0)

        row_inds, col_inds = np.triu_indices(m, k=k)

        labels = std_labels.cpu().numpy() if gpu else std_labels.data.numpy()
        pairs = [e for e in zip(row_inds, col_inds) if labels[e[0]] < labels[e[1]]]  # remove pairs of documents of the same level
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    else:
        raise NotImplementedError

    #tor_row_inds = torch.LongTensor(row_inds).to(device) if gpu else torch.LongTensor(row_inds)
    #tor_col_inds = torch.LongTensor(col_inds).to(device) if gpu else torch.LongTensor(col_inds)
    tor_row_inds = torch.LongTensor(row_inds, device)
    tor_col_inds = torch.LongTensor(col_inds, device)
    #batch_triu = batch_mats[:, tor_row_inds, tor_col_inds]

    return tor_row_inds, tor_col_inds  # shape: [number of pairs]

