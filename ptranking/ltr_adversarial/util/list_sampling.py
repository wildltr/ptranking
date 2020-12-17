#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch
from torch.nn import functional as F


EPS = 1e-20


def gumbel_softmax(logits, samples_per_query, temperature=1.0, cuda=False, cuda_device=None):
    '''

    :param logits: [1, ranking_size]
    :param num_samples_per_query: number of stochastic rankings to generate
    :param temperature:
    :return:
    '''
    assert 1 == logits.size(0) and 2 == len(logits.size())

    unif = torch.rand(samples_per_query, logits.size(1)) # [num_samples_per_query, ranking_size]
    if cuda: unif = unif.to(cuda_device)

    gumbel = -torch.log(-torch.log(unif + EPS) + EPS) # Sample from gumbel distribution

    logit = (logits + gumbel) / temperature

    y = F.softmax(logit, dim=1)

    # i.e., #return F.softmax(logit, dim=1)
    return y

def sample_ranking_PL_gumbel_softmax(batch_preds, num_sample_ranking=1, only_indices=True, temperature=1.0, gpu=False, device=None):
    '''
    Sample a ranking based stochastic Plackett-Luce model, where gumble noise is added
    @param batch_preds: [1, ranking_size] vector of relevance predictions for documents associated with the same query
    @param num_sample_ranking: number of rankings to sample
    @param only_indices: only return the indices or not
    @return:
    '''
    if num_sample_ranking > 1:
        target_batch_preds = batch_preds.expand(num_sample_ranking, -1)
    else:
        target_batch_preds = batch_preds

    unif = torch.rand(target_batch_preds.size())  # [num_samples_per_query, ranking_size]
    if gpu: unif = unif.to(device)

    gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

    if only_indices:
        batch_logits = target_batch_preds + gumbel
        _, batch_indices = torch.sort(batch_logits, dim=1, descending=True)
        return batch_indices
    else:
        if 1.0 == temperature:
            batch_logits = target_batch_preds + gumbel
        else:
            batch_logits = (target_batch_preds + gumbel) / temperature

        batch_logits_sorted, batch_indices = torch.sort(batch_logits, dim=1, descending=True)
        return batch_indices, batch_logits_sorted


def arg_shuffle_ties(target_batch_stds, descending=True, gpu=False, device=None):
    ''' Shuffle ties, and return the corresponding indice '''
    batch_size, ranking_size = target_batch_stds.size()
    if batch_size > 1:
        list_rperms = []
        for _ in range(batch_size):
            list_rperms.append(torch.randperm(ranking_size))
        batch_rperms = torch.stack(list_rperms, dim=0)
    else:
        batch_rperms = torch.randperm(ranking_size).view(1, -1)

    if gpu: batch_rperms = batch_rperms.to(device)

    shuffled_target_batch_stds = torch.gather(target_batch_stds, dim=1, index=batch_rperms)
    batch_sorted_inds = torch.argsort(shuffled_target_batch_stds, descending=descending)
    batch_shuffle_ties_inds = torch.gather(batch_rperms, dim=1, index=batch_sorted_inds)

    return batch_shuffle_ties_inds
