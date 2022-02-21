#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch
import torch.distributions as tdist

import numpy as np

from ptranking.utils.pytorch.pt_extensions import Gaussian_Integral_0_Inf
apply_Gaussian_Integral_0_Inf = Gaussian_Integral_0_Inf.apply

#from org.l2r_global import global_gpu as gpu, global_device as device

#tensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor
tensor = torch.FloatTensor
tor_zero = tensor([0.0])

# todo support GPU
# todo dubble check the case of including '-1'

def get_weighted_clipped_pos_diffs(qid, sorted_std_labels, global_buffer=None):
    '''
    Get total true pairs based on explicit labels.
    In particular, the difference values are discounted based on positions.
    '''
    num_pos, num_explicit, num_neg_unk, num_unk, num_unique_labels = global_buffer[qid]
    mat_diffs = torch.unsqueeze(sorted_std_labels, dim=1) - torch.unsqueeze(sorted_std_labels, dim=0)
    pos_diffs = torch.where(mat_diffs < 0, tor_zero, mat_diffs)

    clipped_pos_diffs = pos_diffs[0:num_pos, 0:num_explicit]

    total_true_pairs = torch.nonzero(clipped_pos_diffs, as_tuple=False).size(0)

    r_discounts = torch.arange(num_explicit).type(tensor)
    r_discounts = torch.log2(2.0 + r_discounts)
    r_discounts = torch.unsqueeze(r_discounts, dim=0)

    c_discounts = torch.arange(num_pos).type(tensor)
    c_discounts = torch.log2(2.0 + c_discounts)
    c_discounts = torch.unsqueeze(c_discounts, dim=1)

    weighted_clipped_pos_diffs = clipped_pos_diffs / r_discounts
    weighted_clipped_pos_diffs = weighted_clipped_pos_diffs / c_discounts

    return weighted_clipped_pos_diffs, total_true_pairs, num_explicit


def generate_true_pairs(qid, sorted_std_labels, num_pairs, dict_diff=None, global_buffer=None):
    assert dict_diff is not None
    if qid in dict_diff:
        weighted_clipped_pos_diffs, total_true_pairs, total_items = dict_diff[qid]
        #valid_num = min(num_pairs, total_true_pairs)
        res = torch.multinomial(weighted_clipped_pos_diffs.view(1, -1), num_pairs, replacement=True)
        res = torch.squeeze(res)

        #head_inds = res // total_items
        head_inds = torch.div(res, total_items, rounding_mode='floor') # Equivalent to the // operator
        tail_inds = res % total_items
        return head_inds, tail_inds
    else:
        weighted_clipped_pos_diffs, total_true_pairs, total_items = get_weighted_clipped_pos_diffs(qid=qid,
                                                    sorted_std_labels=sorted_std_labels, global_buffer=global_buffer)
        dict_diff[qid] = weighted_clipped_pos_diffs, total_true_pairs, total_items

        #valid_num = min(num_pairs, total_true_pairs)
        res = torch.multinomial(weighted_clipped_pos_diffs.view(1, -1), num_pairs, replacement=True)
        res = torch.squeeze(res)

        #head_inds = res // total_items
        head_inds = torch.div(res, total_items, rounding_mode='floor')
        tail_inds = res % total_items
        return head_inds, tail_inds

def sample_pairs_gaussian(point_vals=None, num_pairs=None, sigma=None):
    mat_means = torch.unsqueeze(point_vals, dim=1) - torch.unsqueeze(point_vals, dim=0)
    mat_probs = apply_Gaussian_Integral_0_Inf(mat_means, np.sqrt(2.0)*sigma)

    head_inds, tail_inds = sample_points_Bernoulli(mat_probs, num_pairs)
    return head_inds, tail_inds


def sample_pairs_BT(point_vals=None, num_pairs=None):
    ''' The probability of observing a pair of ordered documents is formulated based on Bradley-Terry model, i.e., p(d_i > d_j)=1/(1+exp(-delta(s_i - s_j))) '''
    # the rank information is not taken into account, and all pairs are treated equally.

    #total_items = point_vals.size(0)
    mat_diffs = torch.unsqueeze(point_vals, dim=1) - torch.unsqueeze(point_vals, dim=0)
    mat_bt_probs = torch.sigmoid(mat_diffs) # default delta=1.0

    """
    B = tdist.Binomial(1, mat_bt_probs.view(1, -1))
    b_res = B.sample()
    num_unique_pairs = torch.nonzero(b_res).size(0)
    if num_unique_pairs < num_pairs:
        res = torch.multinomial(b_res, num_pairs, replacement=True)
    else:
        res = torch.multinomial(b_res, num_pairs, replacement=False)

    res = torch.squeeze(res)
    head_inds = res / total_items
    tail_inds = res % total_items
    """
    head_inds, tail_inds = sample_points_Bernoulli(mat_bt_probs, num_pairs)
    return head_inds, tail_inds

def sample_points_Bernoulli(mat_probs, num_pairs):
    total_items = mat_probs.size(0)
    B = tdist.Binomial(1, mat_probs.view(1, -1))
    b_res = B.sample()

    res = torch.multinomial(b_res, num_pairs, replacement=True)
    res = torch.squeeze(res)
    #head_inds = res // total_items
    head_inds = torch.div(res, total_items, rounding_mode='floor')
    tail_inds = res % total_items

    return head_inds, tail_inds


def test_generate_true_pairs():
    std_labels = torch.from_numpy(
        np.asarray([4.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # pos_labels = torch.nonzero(std_labels)
    # print('pos_labels', pos_labels.size())

    num_pos = torch.nonzero(std_labels).size(0)
    num_elements = std_labels.size(0)
    # print('num_elements', num_elements)

    mat_diffs = torch.unsqueeze(std_labels, dim=1) - torch.unsqueeze(std_labels, dim=0)
    # print(mat_diffs)

    pos_diffs = torch.where(mat_diffs < 0, tor_zero, mat_diffs)
    # print('pos_diffs', pos_diffs)

    clipped_pos_diffs = pos_diffs[0:num_pos, :]
    # print('clipped_pos_diffs', clipped_pos_diffs)

    total_true_pairs = torch.nonzero(clipped_pos_diffs).size(0)
    print('n', total_true_pairs)

    adopted_true_pairs = 20
    k = min(adopted_true_pairs, total_true_pairs)
    # print('k', k)

    # way-1
    r_discounts = torch.arange(num_elements).type(tensor)
    # print('r_discounts', r_discounts)
    r_discounts = torch.log2(2.0 + r_discounts)
    # r_discounts = 2.0 + r_discounts
    r_discounts = torch.unsqueeze(r_discounts, dim=0)
    # print('r_discounts', r_discounts)

    c_discounts = torch.arange(num_pos).type(tensor)
    # print('c_discounts', c_discounts)
    c_discounts = torch.log2(2.0 + c_discounts)
    # c_discounts = 2.0 + c_discounts
    c_discounts = torch.unsqueeze(c_discounts, dim=1)
    # print('c_discounts', c_discounts)

    weighted_clipped_pos_diffs = clipped_pos_diffs / r_discounts
    weighted_clipped_pos_diffs = weighted_clipped_pos_diffs / c_discounts

    # way-2
    # reversed_clipped_pos_diffs = 1.0/clipped_pos_diffs
    # weighted_clipped_pos_diffs = torch.where(clipped_pos_diffs>0, reversed_clipped_pos_diffs, clipped_pos_diffs)

    print('weighted_clipped_pos_diffs', weighted_clipped_pos_diffs)

    res = torch.multinomial(weighted_clipped_pos_diffs.view(1, -1), k, replacement=False)
    print('res', res)

    res = torch.squeeze(res)
    print(res)

    # row indices
    row_inds = res / num_elements
    print('row_inds', row_inds)
    col_inds = res % num_elements
    print('col_inds', col_inds)

    selected_elements = clipped_pos_diffs[row_inds, col_inds]
    print('selected_elements', selected_elements)

    # print(get_weighted_clipped_pos_diffs(std_labels)[0])
    head_inds, tail_inds = generate_true_pairs(sorted_std_labels=std_labels, num_pairs=5, qid='00')
    selected_elements = clipped_pos_diffs[head_inds, tail_inds]
    print('selected_elements', selected_elements)


def test_sample_pairs_gaussian():
    point_vals = torch.from_numpy(np.asarray([4.0, 3.0, 3.0], dtype=np.float32))
    head_inds, tail_inds = sample_pairs_gaussian(point_vals=point_vals, num_pairs=10, sigma=1.2)
    print('row_inds', head_inds)
    print('col_inds', tail_inds)


def test_sample_pairs_BT():
    point_vals = torch.from_numpy(np.asarray([4.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    #sample_pairs_BT(point_vals=point_vals, num_pairs=10)
    head_inds, tail_inds = sample_pairs_BT(point_vals=point_vals, num_pairs=10)
    print('row_inds', head_inds)
    print('col_inds', tail_inds)


def log_gaussian(observed_xs, mus, variances):
    '''
    :param observed_xs: a number of observed values. Each value x_i follows a specific normal distribution, N(x_i|mu_i, v_i)
    :param mus:
    :param variances:
    :return:
    '''

    log_likelihoods = -0.5 * (tensor(1, observed_xs.size(0)).fill_(np.log(2.0*np.pi)) + torch.log(variances) + (observed_xs-mus)**2/variances)



if __name__ == '__main__':
    #1
    #test_generate_true_pairs()

    #2
    test_sample_pairs_gaussian()

    #3
    #test_sample_pairs_BT()
