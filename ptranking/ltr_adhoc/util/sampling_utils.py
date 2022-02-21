#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

EPS = 1e-20

#######
# Module for listwise sampling
# We note that: each time we only sample one ranking per query since multiple times of sampling just corresonds to more epoches
#######

def arg_shuffle_ties(batch_rankings, descending=True, device=None):
    '''Shuffle ties, and return the corresponding indice '''
    batch_size, ranking_size = batch_rankings.size()
    if batch_size > 1:
        list_rperms = []
        for _ in range(batch_size):
            list_rperms.append(torch.randperm(ranking_size, device=device))
        batch_rperms = torch.stack(list_rperms, dim=0)
    else:
        batch_rperms = torch.randperm(ranking_size, device=device).view(1, -1)

    batch_shuffled_rankings = torch.gather(batch_rankings, dim=1, index=batch_rperms)
    batch_desc_inds = torch.argsort(batch_shuffled_rankings, descending=descending)
    batch_shuffle_ties_inds = torch.gather(batch_rperms, dim=1, index=batch_desc_inds)

    return batch_shuffle_ties_inds


def sample_ranking_PL(batch_preds, only_indices=True, temperature=1.0):
    '''
    Sample one ranking per query based on Plackett-Luce model
    @param batch_preds: [batch_size, ranking_size] each row denotes the relevance predictions for documents associated with the same query
    @param only_indices: only return the indices or not
    '''
    if torch.isnan(batch_preds).any(): # checking is needed for later PL model
        print('batch_preds', batch_preds)
        print('Including NaN error.')

    if 1.0 != temperature:
        target_batch_preds = torch.div(batch_preds, temperature)
    else:
        target_batch_preds = batch_preds

    batch_m, _ = torch.max(target_batch_preds, dim=1, keepdim=True)  # a transformation aiming for higher stability when computing softmax() with exp()
    m_target_batch_preds = target_batch_preds - batch_m
    batch_exps = torch.exp(m_target_batch_preds)
    batch_sample_inds = torch.multinomial(batch_exps, replacement=False, num_samples=batch_preds.size(1))

    if only_indices:
        return batch_sample_inds
    else:
        # sort batch_preds according to the sample order
        # w.r.t. top-k, we need the remaining part, but we don't consider the orders among the remaining parts
        batch_preds_in_sample_order = torch.gather(batch_preds, dim=1, index=batch_sample_inds)
        return batch_sample_inds, batch_preds_in_sample_order


def sample_ranking_PL_gumbel_softmax(batch_preds, only_indices=True, temperature=1.0, device=None):
    '''
    Sample a ranking based stochastic Plackett-Luce model, where gumble noise is added
    @param batch_preds: [batch_size, ranking_size] each row denotes the relevance predictions for documents associated with the same query
    @param only_indices: only return the indices or not
    '''
    unif = torch.rand(batch_preds.size(), device=device)  # [batch_size, ranking_size]

    gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

    if only_indices:
        batch_logits = batch_preds + gumbel
        _, batch_sample_inds = torch.sort(batch_logits, dim=1, descending=True)
        return batch_sample_inds
    else:
        if 1.0 == temperature:
            batch_logits = batch_preds + gumbel
        else:
            batch_logits = (batch_preds + gumbel) / temperature

        batch_logits_in_sample_order, batch_sample_inds = torch.sort(batch_logits, dim=1, descending=True)
        return batch_sample_inds, batch_logits_in_sample_order

######
#
######

def unique_count(std_labels, descending=True):
    asc_std_labels, _ = torch.sort(std_labels)
    uni_elements, inds = torch.unique(asc_std_labels, sorted=True, return_inverse=True)
    asc_uni_cnts = torch.stack([(asc_std_labels == e).sum() for e in uni_elements])

    if descending:
        des_uni_cnts = torch.flip(asc_uni_cnts, dims=[0])
        return des_uni_cnts
    else:
        return asc_uni_cnts


def batch_global_unique_count(batch_std_labels, max_rele_lavel, descending=True, gpu=False):
    '''  '''
    batch_asc_std_labels, _ = torch.sort(batch_std_labels, dim=1)
    # default ascending order
    global_uni_elements = torch.arange(max_rele_lavel+1).type(torch.cuda.FloatTensor) if gpu else torch.arange(max_rele_lavel+1).type(torch.FloatTensor)

    asc_uni_cnts = torch.cat([(batch_asc_std_labels == e).sum(dim=1, keepdim=True) for e in global_uni_elements], dim=1) # row-wise count per element

    if descending:
        des_uni_cnts = torch.flip(asc_uni_cnts, dims=[1])
        return des_uni_cnts
    else:
        return asc_uni_cnts

def uniform_rand_per_label(uni_cnts, device='cpu'):
    """ can be compatible with batch """
    num_unis = uni_cnts.size(0)  # number of unique elements
    inner_rand_inds = (torch.rand(num_unis) * uni_cnts.type(torch.FloatTensor)).type(torch.LongTensor)  # random index w.r.t each interval
    begs = torch.cumsum(torch.cat([torch.tensor([0.], dtype=torch.long, device=device), uni_cnts[0:num_unis - 1]]), dim=0)  # begin positions of each interval within the same vector
    # print('begin positions', begs)
    rand_inds_per_label = begs + inner_rand_inds
    # print('random index', rand_inds_per_label)  # random index tensor([ 0,  1,  3,  6, 10]) ([0, 2, 3, 5, 8])

    return rand_inds_per_label


def sample_per_label(batch_rankings, batch_stds):
    assert 1 == batch_stds.size(0)

    std_labels = torch.squeeze(batch_stds)
    des_uni_cnts = unique_count(std_labels)
    rand_inds_per_label = uniform_rand_per_label(des_uni_cnts)

    sp_batch_rankings = batch_rankings[:, rand_inds_per_label, :]
    sp_batch_stds = batch_stds[:, rand_inds_per_label]

    return sp_batch_rankings, sp_batch_stds


if __name__ == '__main__':
    #1
    #cuda = 'cuda:0'
    cuda = 'cpu'
    target_batch_stds = torch.randn(size=(3, 5), device=cuda)

    batch_shuffle_ties_inds = arg_shuffle_ties(target_batch_stds=target_batch_stds, descending=True, device=cuda)
    print('batch_shuffle_ties_inds', batch_shuffle_ties_inds)

    '''
    std_labels = tensor([3, 3, 2, 1, 1, 1, 0, 0, 0, 0])
    des_uni_cnts = unique_count(std_labels)
    print('des_uni_cnts', des_uni_cnts)

    batch_global_des_uni_cnts = batch_global_unique_count(std_labels.view(2, -1), max_rele_lavel=4)
    print(std_labels.view(2, -1))
    print('batch_global_des_uni_cnts', batch_global_des_uni_cnts)

    rand_inds_per_label = uniform_rand_per_label(des_uni_cnts)
    print('random elements', std_labels[rand_inds_per_label])
    '''
