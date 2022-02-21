
import math
import numpy as np

import torch

# aNDCG denotes \alpha_nDCG
SRD_METRIC = ['aNDCG', 'nERR-IA']

################
# \alpha_nDCG@k
################
def torch_alpha_dcg_at_k(sorted_q_doc_rele_mat, k=None, alpha=0.5, device='cpu'):
    valid_max_cutoff = sorted_q_doc_rele_mat.size(1)
    cutoff = min(valid_max_cutoff, k)

    target_q_doc_rele_mat = sorted_q_doc_rele_mat[:, 0:cutoff]
    prior_rele_mat = torch.zeros_like(target_q_doc_rele_mat)
    prior_rele_mat[:, 1:cutoff] = target_q_doc_rele_mat[:, 0:cutoff-1] # the times of being covered for 1st doc should be zero
    #prior_cover_cnts = torch.cumsum(prior_rele_mat, dim=1).view(sorted_q_doc_rele_mat.size(0), -1) # TODO is the view needed?
    #print('prior_rele_mat', prior_rele_mat.size())
    prior_cover_cnts = torch.cumsum(prior_rele_mat, dim=1)
    #print('prior_cover_cnts', prior_cover_cnts.size())

    denominators = torch.log2(torch.arange(cutoff, dtype=torch.float, device=device) + 2.0).view(1, -1)
    discounted_gains = torch.pow((1.0 - alpha), prior_cover_cnts) * target_q_doc_rele_mat / denominators
    discounted_global_gains = torch.sum(discounted_gains, dim=0)
    alpha_DCG_at_k = torch.sum(discounted_global_gains)

    return alpha_DCG_at_k

def torch_alpha_ndcg_at_k(sys_q_doc_rele_mat, ideal_q_doc_rele_mat, k=None, alpha=0.5, device='cpu'):
    sys_alpha_DCG_at_k = torch_alpha_dcg_at_k(sorted_q_doc_rele_mat=sys_q_doc_rele_mat, k=k, alpha=alpha, device=device)
    ideal_alpha_DCG_at_k = torch_alpha_dcg_at_k(sorted_q_doc_rele_mat=ideal_q_doc_rele_mat, k=k, alpha=alpha, device=device)

    if ideal_alpha_DCG_at_k > 0:
        alpha_nDCG_at_k = sys_alpha_DCG_at_k/ideal_alpha_DCG_at_k
    else:
        alpha_nDCG_at_k = torch.tensor([0.], device=device)

    return alpha_nDCG_at_k

def torch_alpha_dcg_at_ks(sorted_q_doc_rele_mat, max_cutoff=None, alpha=0.5, device='cpu'):
    target_q_doc_rele_mat = sorted_q_doc_rele_mat[:, 0:max_cutoff]
    prior_rele_mat = torch.zeros_like(target_q_doc_rele_mat)
    prior_rele_mat[:, 1:max_cutoff] = target_q_doc_rele_mat[:, 0:max_cutoff-1]  # the times of being covered for 1st doc should be zero
    #prior_cover_cnts = torch.cumsum(prior_rele_mat, dim=1).view(sorted_q_doc_rele_mat.size(0), -1)
    prior_cover_cnts = torch.cumsum(prior_rele_mat, dim=1)

    denominators = torch.log2(torch.arange(max_cutoff, dtype=torch.float, device=device) + 2.0)

    discounted_gains = torch.pow((1.0 - alpha), prior_cover_cnts) * target_q_doc_rele_mat / denominators
    discounted_global_gains = torch.sum(discounted_gains, dim=0, keepdim=True)
    alpha_DCG_at_ks = torch.cumsum(discounted_global_gains, dim=1).view(1, -1) # dcg w.r.t. each position
    return alpha_DCG_at_ks

def torch_alpha_ndcg_at_ks(sys_q_doc_rele_mat, ideal_q_doc_rele_mat, ks=None, alpha=0.5, device='cpu'):
    valid_max_cutoff = sys_q_doc_rele_mat.size(1)
    used_ks = [k for k in ks if k <= valid_max_cutoff] if valid_max_cutoff < max(ks) else ks
    max_cutoff = max(used_ks)

    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    sys_alpha_dcgs = torch_alpha_dcg_at_ks(sys_q_doc_rele_mat, max_cutoff=max_cutoff, alpha=alpha, device=device)
    sys_alpha_dcg_at_ks = sys_alpha_dcgs[:, inds]  # get cumulative gains at specified rank positions
    ideal_alpha_dcgs = torch_alpha_dcg_at_ks(ideal_q_doc_rele_mat, max_cutoff=max_cutoff, alpha=alpha, device=device)
    ideal_alpha_dcg_at_ks = ideal_alpha_dcgs[:, inds]

    #print('sys_alpha_dcg_at_ks', sys_alpha_dcg_at_ks)
    #print('ideal_alpha_dcg_at_ks', ideal_alpha_dcg_at_ks)

    alpha_ndcg_at_ks = sys_alpha_dcg_at_ks / ideal_alpha_dcg_at_ks

    if torch.count_nonzero(ideal_alpha_dcg_at_ks) < len(used_ks):
        zero_mask = ideal_alpha_dcg_at_ks <= 0
        alpha_ndcg_at_ks[zero_mask] = 0.

    if valid_max_cutoff < max(ks):
        padded_ndcg_at_ks = torch.zeros(sys_q_doc_rele_mat.size(0), len(ks))
        padded_ndcg_at_ks[:, 0:len(used_ks)] = alpha_ndcg_at_ks
        return padded_ndcg_at_ks
    else:
        return alpha_ndcg_at_ks


def alpha_dcg_at_k(sorted_docs, q_doc_subtopics, k=None, alpha=0.5):
    alpha_DCG_at_k = 0.0
    subtopics = np.zeros(20)
    for i in range(k):
        gg = 0.0 # global gain
        if sorted_docs[i] not in q_doc_subtopics:
            continue

        covered_subtopics = q_doc_subtopics[sorted_docs[i]]
        if len(covered_subtopics) > 0:
            for subtopic in covered_subtopics:
                gg += (1 - alpha) ** subtopics[int(subtopic) - 1]
                subtopics[int(subtopic) - 1] += 1

        alpha_DCG_at_k += gg / math.log(i + 2, 2)

    return alpha_DCG_at_k


def alpha_ndcg_at_k(sys_sorted_docs, ideal_sorted_docs, k=None, q_doc_subtopics=None, alpha=0.5):
    sys_alpha_DCG_at_k = alpha_dcg_at_k(sorted_docs=sys_sorted_docs, q_doc_subtopics=q_doc_subtopics, k=k, alpha=alpha)
    ideal_alpha_DCG_at_k = alpha_dcg_at_k(sorted_docs=ideal_sorted_docs, q_doc_subtopics=q_doc_subtopics, k=k, alpha=alpha)
    res_alpha_DCG_at_k = sys_alpha_DCG_at_k / ideal_alpha_DCG_at_k
    return res_alpha_DCG_at_k

####
# delta-\alpha_nDCG when swapping a pair of documents
####
def get_div_ideal_ranking(pool_docs, q_doc_subtopics, alpha=0.5):
    ideal_ranking = []
    set_candidate_docs = set(pool_docs)
    subtopic_cover_cnts = np.zeros(20)
    while(len(set_candidate_docs) > 0):
        max_gg = -1.0 # a minus value indicates that there always be a selected document though the gg value is zero
        ideal_doc = None
        for can_doc in set_candidate_docs: # selecting the ideal doc for next position in a greedy manner
            gg = 0.0 # global gain
            if can_doc in q_doc_subtopics:
                covered_subtopics = q_doc_subtopics[can_doc]
                if len(covered_subtopics) > 0:
                    for subtopic in covered_subtopics:
                        gg += (1 - alpha) ** subtopic_cover_cnts[int(subtopic) - 1]

            if gg > max_gg:
                max_gg = gg
                ideal_doc = can_doc

        if ideal_doc in q_doc_subtopics: # update the times of being covered in prior positions
            covered_subtopics = q_doc_subtopics[ideal_doc]
            if len(covered_subtopics) > 0:
                for subtopic in covered_subtopics:
                    subtopic_cover_cnts[int(subtopic) - 1] += 1

        ideal_ranking.append(ideal_doc)
        set_candidate_docs.remove(ideal_doc)

    return ideal_ranking

def get_delta_alpha_dcg(ideal_q_doc_rele_mat=None, sys_q_doc_rele_mat=None, alpha=0.5, device='cpu', normalization=True):
    '''
    Get the delta-nDCG w.r.t. pairwise swapping of the currently predicted order.
    @param ideal_q_doc_rele_mat: the standard labels sorted in an ideal order
    @param sys_q_doc_rele_mat: the standard labels sorted based on the corresponding predictions
    @param alpha:
    @param device:
    @return:
    '''
    num_subtopics, ranking_size = sys_q_doc_rele_mat.size()

    if normalization:
        ideal_alpha_DCG = torch_alpha_dcg_at_k(sorted_q_doc_rele_mat=ideal_q_doc_rele_mat,
                                               k=ranking_size, alpha=alpha, device=device)

    prior_rele_mat = torch.zeros_like(sys_q_doc_rele_mat)
    prior_rele_mat[:, 1:ranking_size] = sys_q_doc_rele_mat[:, 0:ranking_size - 1]  # the times of being covered for 1st doc should be zero
    prior_cover_cnts = torch.cumsum(prior_rele_mat, dim=1)
    subtopic_user_focus = torch.pow((1.0 - alpha), prior_cover_cnts)

    subtopic_gains = torch.pow(2.0, sys_q_doc_rele_mat) - 1.0
    subtopic_gain_diffs = torch.unsqueeze(subtopic_gains, dim=2) - torch.unsqueeze(subtopic_gains, dim=1)

    ranks = torch.arange(ranking_size, dtype=torch.float, device=device)
    rank_discounts = 1.0 / torch.log2(ranks + 2.0)  # discount co-efficients

    subtopic_user_focus_1st = torch.unsqueeze(subtopic_user_focus, dim=2).expand(-1, -1, ranking_size)
    rank_discounts_1st = rank_discounts.view(1, -1, 1)
    subtopic_coffs_1st = rank_discounts_1st * subtopic_user_focus_1st

    subtopic_user_focus_2nd = torch.unsqueeze(subtopic_user_focus, dim=1).expand(-1, ranking_size, -1)
    rank_discounts_2nd = rank_discounts.view(1, 1, -1)
    subtopic_coffs_2nd = rank_discounts_2nd * subtopic_user_focus_2nd

    # absolute changes w.r.t. pairwise swapping
    delta_alpha_DCG = torch.abs(torch.sum(subtopic_gain_diffs*subtopic_coffs_1st, dim=0) - torch.sum(subtopic_gain_diffs*subtopic_coffs_2nd, dim=0))

    if normalization:
        return delta_alpha_DCG/ideal_alpha_DCG
    else:
        return delta_alpha_DCG

################
# ERR-IA@k & nERR-IA@k
################

def torch_rankwise_err_ia(sorted_q_doc_rele_mat, max_label=None, k=10, point=True, device='cpu'):
    assert max_label is not None # it is either query-level or corpus-level
    num_subtopics = sorted_q_doc_rele_mat.size(0)
    valid_max_cutoff = sorted_q_doc_rele_mat.size(1)
    cutoff = min(valid_max_cutoff, k)

    target_q_doc_rele_mat = sorted_q_doc_rele_mat[:, 0:cutoff]

    t2 = torch.tensor([2.0], dtype=torch.float, device=device)
    satis_pros = (torch.pow(t2, target_q_doc_rele_mat) - 1.0) / torch.pow(t2, max_label)
    unsatis_pros = torch.ones_like(target_q_doc_rele_mat, device=device) - satis_pros
    cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
    cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros, device=device)
    cascad_unsatis_pros[:, 1:cutoff] = cum_unsatis_pros[:, 0:cutoff - 1]

    non_zero_inds = torch.nonzero(torch.sum(target_q_doc_rele_mat, dim=1))
    zero_metric_value = False if non_zero_inds.size(0) > 0 else True

    if zero_metric_value:
        return torch.zeros(1, device=device), zero_metric_value # since no relevant documents within the list
    else:
        pos_rows = non_zero_inds[:, 0]

    reciprocal_ranks = 1.0 / (torch.arange(cutoff, dtype=torch.float, device=device).view(1, -1) + 1.0)
    expt_satis_ranks = satis_pros[pos_rows, :] * cascad_unsatis_pros[pos_rows, :] * reciprocal_ranks

    if point: # a specific position
        err_ia = torch.sum(expt_satis_ranks, dim=(1, 0))
        return err_ia/num_subtopics, zero_metric_value
    else:
        rankwise_err_ia = torch.cumsum(expt_satis_ranks, dim=1)
        rankwise_err_ia = torch.sum(rankwise_err_ia, dim=0)
        return rankwise_err_ia/num_subtopics, zero_metric_value

def torch_err_ia_at_k(sorted_q_doc_rele_mat, max_label=None, k=10, device='cpu'):
    err_ia_at_k, _ = torch_rankwise_err_ia(sorted_q_doc_rele_mat=sorted_q_doc_rele_mat, max_label=max_label, k=k, point=True, device=device)
    return err_ia_at_k

def torch_err_ia_at_ks(sorted_q_doc_rele_mat, max_label=None, ks=None, device='cpu'):
    valid_max_cutoff = sorted_q_doc_rele_mat.size(1)
    need_padding = True if valid_max_cutoff < max(ks) else False
    used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks
    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)

    rankwise_err_ia, zero_metric_value = torch_rankwise_err_ia(sorted_q_doc_rele_mat=sorted_q_doc_rele_mat, point=False,
                                                               max_label=max_label, k=max_cutoff, device=device)
    if zero_metric_value:
        return torch.zeros(len(ks), device=device)
    else:
        err_ia_at_ks = rankwise_err_ia[inds]
        if need_padding:
            padded_err_ia_at_ks = torch.zeros(len(ks), device=device)
            padded_err_ia_at_ks[0:len(used_ks)] = err_ia_at_ks
            return padded_err_ia_at_ks
        else:
            return err_ia_at_ks

def torch_nerr_ia_at_k(sys_q_doc_rele_mat, ideal_q_doc_rele_mat, max_label=None, k=10, device='cpu'):
    valid_max_cutoff = sys_q_doc_rele_mat.size(1)
    cutoff = min(valid_max_cutoff, k)

    sys_err_ia_at_k, zero_metric_value = torch_rankwise_err_ia(sorted_q_doc_rele_mat=sys_q_doc_rele_mat, point=True,
                                                               max_label=max_label, k=cutoff, device=device)
    if zero_metric_value:
        return sys_err_ia_at_k
    else:
        ideal_err_ia_at_k, _ = torch_rankwise_err_ia(sorted_q_doc_rele_mat=ideal_q_doc_rele_mat, max_label=max_label,
                                                     k=cutoff, point=True, device=device)
        if ideal_err_ia_at_k > 0:
            nerr_ia_at_k = sys_err_ia_at_k/ideal_err_ia_at_k
        else:
            nerr_ia_at_k = torch.tensor([0.], device=device)

        return nerr_ia_at_k

def torch_nerr_ia_at_ks(sys_q_doc_rele_mat, ideal_q_doc_rele_mat, max_label=None, ks=None, device='cpu'):
    valid_max_cutoff = sys_q_doc_rele_mat.size(1)
    need_padding = True if valid_max_cutoff < max(ks) else False
    used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks
    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)

    sys_rankwise_err_ia, zero_metric_value = torch_rankwise_err_ia(sorted_q_doc_rele_mat=sys_q_doc_rele_mat, point=False,
                                                                   max_label=max_label, k=max_cutoff, device=device)
    if zero_metric_value:
        return sys_rankwise_err_ia
    else:
        ideal_rankwise_err_ia, _ = torch_rankwise_err_ia(sorted_q_doc_rele_mat=ideal_q_doc_rele_mat, max_label=max_label,
                                                         k=max_cutoff, point=False, device=device)
        rankwise_nerr_ia = sys_rankwise_err_ia/ideal_rankwise_err_ia

        if torch.count_nonzero(ideal_rankwise_err_ia) < max_cutoff:
            zero_mask = ideal_rankwise_err_ia <= 0
            rankwise_nerr_ia[zero_mask] = 0.

        nerr_ia_at_ks = rankwise_nerr_ia[inds]
        if need_padding:
            padded_nerr_ia_at_ks = torch.zeros(len(ks), device=device)
            padded_nerr_ia_at_ks[0:len(used_ks)] = nerr_ia_at_ks
            return padded_nerr_ia_at_ks
        else:
            return nerr_ia_at_ks
