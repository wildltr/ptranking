
import torch
from torch.autograd import Variable

import numpy as np

from org.archive.eval.metric import rele_gain

from org.archive.utils.pytorch.pt_extensions import Power

""" Extended torch functions """
power = Power.apply

def tor_stable_softmax_bp(histogram, base=None):
    max_v, _ = torch.max(histogram, dim=1, keepdim=True)  # a transformation aiming for higher stability when computing softmax() with exp()
    hist = histogram - max_v
    hist_exped = power(hist, Variable(torch.Tensor([base]).float()))
    probs = torch.div(hist_exped, torch.sum(hist_exped, dim=1, keepdim=True))
    return probs

def tor_stable_softmax_e(histogram):
    max_v, _ = torch.max(histogram, dim=1, keepdim=True)  # a transformation aiming for higher stability when computing softmax() with exp()
    hist = histogram - max_v
    hist_exped = torch.exp(hist)
    probs = torch.div(hist_exped, torch.sum(hist_exped, dim=1, keepdim=True))
    return probs

def tor_sum_norm(histogram):
    probs = torch.div(histogram, torch.sum(histogram, dim=1, keepdim=True))
    return probs

def cost_mat_abs(tor_batch_std_label_vec):
    ''' Take the absolute difference between rank positions as the moving cost '''
    size_ranking = tor_batch_std_label_vec.size(1)
    cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float32)
    for i in range(size_ranking):
        for j in range(size_ranking):
            cost_mat[i, j] = np.abs(i - j)
    return cost_mat

def cost_mat_pow(tor_batch_std_label_vec, pow=2):
    ''' Take the exponent of the absolute difference between rank positions as the moving cost '''
    size_ranking = tor_batch_std_label_vec.size(1)
    cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float32)
    for i in range(size_ranking):
        for j in range(size_ranking):
            cost_mat[i, j] = (i - j) ** pow
    return cost_mat

def cost_mat_group(cpu_tor_batch_std_label_vec, non_rele_gap=100.0, var_penalty=0.01, gain_base=4.0):
    """
    Take into account the group information among documents, namely whether two documents are of the same standard relevance degree

    @param non_rele_gap the gap between a relevant document and an irrelevant document
    @param var_penalty variance penalty
    @param gain_base the base for computing gain value
    """
    size_ranking = cpu_tor_batch_std_label_vec.size(1)
    std_label_vec = cpu_tor_batch_std_label_vec[0, :].numpy()

    cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float32)
    for i in range(size_ranking):
        i_rele_level = std_label_vec[i]
        for j in range(size_ranking):
            if i==j:
                cost_mat[i, j] = 0
            else:
                j_rele_level = std_label_vec[j]

                if i_rele_level == j_rele_level:
                    cost_mat[i, j] = var_penalty
                else:
                    cost_mat[i, j] = np.abs(rele_gain(i_rele_level, gain_base=gain_base) - rele_gain(j_rele_level, gain_base=gain_base))

                    if 0 == i_rele_level or 0 == j_rele_level:  #enforce the margin between relevance and non-relevance
                        cost_mat[i, j] += non_rele_gap

    return cost_mat


def get_cost_mat(tor_batch_std_label_vec, wass_para_dict=None):
    cost_type = wass_para_dict['cost_type']
    if cost_type == 'CostAbs':
        cost_mat = cost_mat_abs(tor_batch_std_label_vec)

    elif cost_type == 'CostSquare':
        cost_mat = cost_mat_pow(tor_batch_std_label_vec)

    elif cost_type == 'Group':
        gain_base, non_rele_gap, var_penalty = wass_para_dict['gain_base'], wass_para_dict['non_rele_gap'], wass_para_dict['var_penalty']
        cost_mat = cost_mat_group(tor_batch_std_label_vec, non_rele_gap=non_rele_gap, var_penalty=var_penalty, gain_base=gain_base)
    else:
        raise NotImplementedError

    tor_cost_mat = torch.from_numpy(cost_mat).type(torch.FloatTensor)
    return tor_cost_mat, cost_mat


def get_normalized_distributions(tor_batch_std_label_vec=None, tor_batch_prediction=None, wass_dict_std_dists=None, qid=None, wass_para_dict=None, TL_AF=None):
    smooth_type, norm_type = wass_para_dict['smooth_type'], wass_para_dict['norm_type']

    if 'ST' == smooth_type:
        if wass_dict_std_dists is not None:
            if qid in wass_dict_std_dists:  # target distributions
                tor_batch_std_dist = wass_dict_std_dists[qid]
            else:
                tor_batch_smoothed_vec = tor_batch_std_label_vec
                tor_batch_std_dist = tor_stable_softmax_e(tor_batch_smoothed_vec)
                wass_dict_std_dists[qid] = tor_batch_std_dist
        else:
            tor_batch_smoothed_vec = tor_batch_std_label_vec
            tor_batch_std_dist = tor_stable_softmax_e(tor_batch_smoothed_vec)

        #tor_batch_prediction = torch.squeeze(tor_batch_prediction, dim=2)

        if 'S' == TL_AF or 'ST' == TL_AF:  # map to the same relevance level
            tor_max_rele_level = torch.max(tor_batch_std_label_vec)
            tor_batch_prediction = tor_batch_prediction * tor_max_rele_level

        tor_batch_smoothed_pred = tor_batch_prediction

        if 'BothST' == norm_type:
            dists_pred = tor_stable_softmax_e(tor_batch_smoothed_pred)  # predicted distributions through transformation

        elif 'PredSum' == norm_type:
            dists_pred = tor_sum_norm(tor_batch_smoothed_pred)
        else:
            raise NotImplementedError

    elif 'Gain' == smooth_type:
        tor_gain_base = Variable(torch.Tensor([2.0]).float())

        tor_batch_prediction = torch.squeeze(tor_batch_prediction, dim=2)
        if 'S' == TL_AF or 'ST' == TL_AF:  # map to the same relevance level
            tor_max_rele_level = torch.max(tor_batch_std_label_vec)
            tor_batch_prediction = tor_batch_prediction * tor_max_rele_level

        tor_batch_gain_vec_pred = power(tor_batch_prediction, tor_gain_base) - 1.0
        tor_batch_gain_vec_std = torch.pow(tor_gain_base, tor_batch_std_label_vec) - 1.0

        if 'Expt' == norm_type:
            gain_expt_base = wass_para_dict['gain_expt_base']
            dists_pred = tor_stable_softmax_bp(tor_batch_gain_vec_pred, base=gain_expt_base)  # predicted distributions through transformation

            if wass_dict_std_dists is not None:
                if qid in wass_dict_std_dists:  # target distributions
                    tor_batch_std_dist = wass_dict_std_dists[qid]
                else:
                    tor_batch_std_dist = tor_stable_softmax_bp(tor_batch_gain_vec_std, base=gain_expt_base)
                    wass_dict_std_dists[qid] = tor_batch_std_dist
            else:
                tor_batch_std_dist = tor_stable_softmax_bp(tor_batch_gain_vec_std, base=gain_expt_base)

        elif 'Sum' == norm_type:
            dists_pred = tor_sum_norm(tor_batch_gain_vec_pred)  # predicted distributions through transformation

            if wass_dict_std_dists is not None:
                if qid in wass_dict_std_dists:  # target distributions
                    tor_batch_std_dist = wass_dict_std_dists[qid]
                else:
                    tor_batch_std_dist = tor_sum_norm(tor_batch_gain_vec_std)
                    wass_dict_std_dists[qid] = tor_batch_std_dist
            else:
                tor_batch_std_dist = tor_sum_norm(tor_batch_gain_vec_std)
        else:
            raise NotImplementedError

    elif 'Raw' == smooth_type:
        if 'BothPL' == norm_type:
            if wass_dict_std_dists is not None:
                if qid in wass_dict_std_dists:  # target distributions
                    tor_batch_std_dist = wass_dict_std_dists[qid]
                else:
                    tor_batch_smoothed_vec = tor_batch_std_label_vec
                    tor_batch_std_dist = tor_stable_softmax_e(tor_batch_smoothed_vec)
                    wass_dict_std_dists[qid] = tor_batch_std_dist
            else:
                tor_batch_smoothed_vec = tor_batch_std_label_vec
                tor_batch_std_dist = tor_stable_softmax_e(tor_batch_smoothed_vec)

            tor_batch_prediction = torch.squeeze(tor_batch_prediction, dim=2)
            tor_batch_smoothed_pred = tor_batch_prediction
            if 'S' == TL_AF or 'ST' == TL_AF:  # map to the same relevance level
                tor_max_rele_level = torch.max(tor_batch_std_label_vec)
                tor_batch_smoothed_pred = tor_batch_prediction * tor_max_rele_level

            dists_pred = tor_stable_softmax_e(tor_batch_smoothed_pred)  # predicted distributions through transformation

        else: # complete raw
            if wass_dict_std_dists is not None:
                if qid in wass_dict_std_dists:  # target distributions
                    tor_batch_std_dist = wass_dict_std_dists[qid]
                else:
                    tor_batch_std_dist = torch.div(tor_batch_std_label_vec, torch.sum(tor_batch_std_label_vec, dim=1, keepdim=True))
                    wass_dict_std_dists[qid] = tor_batch_std_dist
            else:
                tor_batch_std_dist = torch.div(tor_batch_std_label_vec, torch.sum(tor_batch_std_label_vec, dim=1, keepdim=True))

            if 'ST' == TL_AF:
                dists_pred = torch.squeeze(tor_batch_prediction, dim=2)
            else:
                print('Raw is only consistent with ST setting. ')
                raise NotImplementedError
    else:
        raise NotImplementedError

    return tor_batch_std_dist, dists_pred