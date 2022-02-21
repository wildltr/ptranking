
import torch
import torch.nn.functional as F

def get_diff_normal(batch_mus, batch_vars, batch_cocos=None):
    '''
    The difference of two normal random variables is another normal random variable. In particular, we consider two
    cases: (1) correlated (2) independent.
    @param batch_mus: the predicted mean
    @param batch_vars: the predicted variance
    @param batch_cocos: the predicted correlation coefficient in [-1, 1], which is formulated as the cosine-similarity of corresponding vectors.
    @return: the mean, variance of the result normal variable.
    '''
    # mu_i - mu_j
    batch_pairsub_mus = torch.unsqueeze(batch_mus, dim=2) - torch.unsqueeze(batch_mus, dim=1)

    # variance w.r.t. S_i - S_j, which is equal to: (1)sigma^2_i + sigma^2_j - \rou_ij*sigma_i*sigma_j  (2) sigma^2_i + sigma^2_j
    if batch_cocos is not None:
        batch_std_vars = torch.pow(batch_vars, .5)
        batch_pairsub_vars = torch.unsqueeze(batch_vars, dim=2) + torch.unsqueeze(batch_vars, dim=1) - \
                             batch_cocos * torch.bmm(torch.unsqueeze(batch_std_vars, dim=2),
                                                     torch.unsqueeze(batch_std_vars, dim=1))
    else:
        batch_pairsub_vars = torch.unsqueeze(batch_vars, dim=2) + torch.unsqueeze(batch_vars, dim=1)

    return batch_pairsub_mus, batch_pairsub_vars

def get_diff_normal_resort(batch_mus, batch_vars, batch_cocos=None, batch_resort_inds=None):
    '''
    Compared with get_diff_normal(), resort is conducted first.
    '''
    batch_resorted_mus = torch.gather(batch_mus, dim=1, index=batch_resort_inds)
    batch_resorted_vars = torch.gather(batch_vars, dim=1, index=batch_resort_inds)
    if batch_cocos is not None:
        num_docs = batch_cocos.size(1)
        batch_cocos_1 = torch.gather(batch_cocos, dim=2,
                                     index=torch.unsqueeze(batch_resort_inds, dim=1).expand(-1, num_docs, -1))
        batch_resorted_cocos = torch.gather(batch_cocos_1, dim=1,
                                     index=torch.unsqueeze(batch_resort_inds, dim=2).expand(-1, -1, num_docs))
    else:
        batch_resorted_cocos = None

    return get_diff_normal(batch_mus=batch_resorted_mus, batch_vars=batch_resorted_vars,
                           batch_cocos=batch_resorted_cocos)

def resort_normal_matrix(batch_mus, batch_vars, batch_resort_inds=None):
    num_elements = batch_mus.size(1)
    # resort mus from two dimensions
    batch_mus_1 = torch.gather(batch_mus, dim=2,
                               index=torch.unsqueeze(batch_resort_inds, dim=1).expand(-1, num_elements, -1))
    batch_resorted_mus = torch.gather(batch_mus_1, dim=1,
                                      index=torch.unsqueeze(batch_resort_inds, dim=2).expand(-1, -1, num_elements))
    # resort vars from two dimensions
    batch_vars_1 = torch.gather(batch_vars, dim=2,
                                index=torch.unsqueeze(batch_resort_inds, dim=1).expand(-1, num_elements, -1))
    batch_resorted_vars = torch.gather(batch_vars_1, dim=1,
                                       index=torch.unsqueeze(batch_resort_inds, dim=2).expand(-1, -1, num_elements))

    return batch_resorted_mus, batch_resorted_vars


def get_expected_rank(batch_mus, batch_vars, batch_cocos=None, return_pairsub_paras=False, return_cdf=False):
    if batch_cocos is None:
        batch_pairsub_mus, batch_pairsub_vars = get_diff_normal(batch_mus=batch_mus, batch_vars=batch_vars)
    else:
        batch_pairsub_mus, batch_pairsub_vars = get_diff_normal(batch_mus=batch_mus, batch_vars=batch_vars,
                                                                batch_cocos=batch_cocos)
    ''' expected ranks '''
    # \Phi(0)$
    batch_Phi0 = 0.5 * torch.erfc(batch_pairsub_mus / torch.sqrt(2 * batch_pairsub_vars))
    # remove diagonal entries
    batch_Phi0_subdiag = torch.triu(batch_Phi0, diagonal=1) + torch.tril(batch_Phi0, diagonal=-1)
    batch_expt_ranks = torch.sum(batch_Phi0_subdiag, dim=2) + 1.0

    if return_pairsub_paras:
        return batch_expt_ranks, batch_pairsub_mus, batch_pairsub_vars
    elif return_cdf:
        return batch_expt_ranks, batch_Phi0_subdiag
    else:
        return batch_expt_ranks

def get_expected_rank_const(batch_mus, const_var, return_pairsub_paras=False, return_cdf=False):

    # f_ij, i.e., mean difference
    batch_pairsub_mus = torch.unsqueeze(batch_mus, dim=2) - torch.unsqueeze(batch_mus, dim=1)
    # variance w.r.t. s_i - s_j, which is equal to sigma^2_i + sigma^2_j
    pairsub_vars = 2 * const_var ** 2

    ''' expected ranks '''
    # \Phi(0)$
    batch_Phi0 = 0.5 * torch.erfc(batch_pairsub_mus / torch.sqrt(2 * pairsub_vars))
    # remove diagonal entries
    batch_Phi0_subdiag = torch.triu(batch_Phi0, diagonal=1) + torch.tril(batch_Phi0, diagonal=-1)
    batch_expt_ranks = torch.sum(batch_Phi0_subdiag, dim=2) + 1.0

    if return_pairsub_paras:
        return batch_expt_ranks, batch_pairsub_mus, pairsub_vars
    elif return_cdf:
        return batch_expt_ranks, batch_Phi0_subdiag
    else:
        return batch_expt_ranks

##########
# negative log-likelihood
##########

def neg_log_likelihood(batch_pairsub_mus, batch_pairsub_vars, top_k=None, device=None):
    '''
    Compute the negative log-likelihood w.r.t. rankings, where the likelihood is formulated as the joint probability of
    consistent pairwise comparisons.
    @param batch_pairsub_mus: mean w.r.t. a pair comparison
    @param batch_pairsub_vars: variance w.r.t. a pair comparison
    @return:
    '''
    batch_full_erfc = torch.erfc(batch_pairsub_mus / torch.sqrt(2 * batch_pairsub_vars))

    if top_k is None:
        # use the triu-part of pairwise probabilities w.r.t. d_i > d_j, and using the trick: log(1.0) is zero
        batch_p_ij_triu = 1.0 - 0.5 * torch.triu(batch_full_erfc, diagonal=1)
        # batch_neg_log_probs = - torch.log(triu_probs) # facing the issue of nan due to overflow
        batch_neg_log_probs = F.binary_cross_entropy(input=batch_p_ij_triu, reduction='none',
                                                     target=torch.ones_like(batch_p_ij_triu, device=device))
    else:  # the part to keep will be 1, otherwise 0
        keep_mask = torch.triu(torch.ones_like(batch_pairsub_vars), diagonal=1)
        keep_mask[:, top_k:, :] = 0.0  # without considering pairs beneath position-k
        batch_p_ij_triu_top_k = 1 - batch_full_erfc * keep_mask * 0.5
        # batch_neg_log_probs = - torch.log(1 - batch_full_erfc * keep_mask * 0.5)  # using the trick: log(1.0) is zero
        batch_neg_log_probs = F.binary_cross_entropy(input=batch_p_ij_triu_top_k, reduction='none',
                                                     target=torch.ones_like(batch_p_ij_triu_top_k, device=device))

    return batch_neg_log_probs # with a shape of [batch_size, ranking_size, ranking_size]

def neg_log_likelihood_explicit(batch_pairsub_mus, std_var, top_k=None, device=None):
    '''
    Compute the negative log-likelihood w.r.t. rankings, where the likelihood is formulated as the joint probability of
    consistent pairwise comparisons.
    @param batch_pairsub_mus: mean w.r.t. a pair comparison
    @param batch_pairsub_vars: variance w.r.t. a pair comparison
    @return:
    '''
    batch_full_erfc = torch.erfc(batch_pairsub_mus / torch.sqrt(torch.tensor([2 * std_var ** 2], device=device)))

    if top_k is None:
        # use the triu-part of pairwise probabilities w.r.t. d_i > d_j, and using the trick: log(1.0) is zero
        batch_p_ij_triu = 1.0 - 0.5 * torch.triu(batch_full_erfc, diagonal=1)
        # batch_neg_log_probs = - torch.log(triu_probs) # facing the issue of nan due to overflow
        batch_neg_log_probs = F.binary_cross_entropy(input=batch_p_ij_triu, reduction='none',
                                                     target=torch.ones_like(batch_p_ij_triu, device=device))
    else:  # the part to keep will be 1, otherwise 0
        keep_mask = torch.triu(torch.ones_like(batch_pairsub_mus), diagonal=1)
        keep_mask[:, top_k:, :] = 0.0  # without considering pairs beneath position-k
        batch_p_ij_triu_top_k = 1 - batch_full_erfc * keep_mask * 0.5
        # batch_neg_log_probs = - torch.log(1 - batch_full_erfc * keep_mask * 0.5)  # using the trick: log(1.0) is zero
        batch_neg_log_probs = F.binary_cross_entropy(input=batch_p_ij_triu_top_k, reduction='none',
                                                     target=torch.ones_like(batch_p_ij_triu_top_k, device=device))

    return batch_neg_log_probs # with a shape of [batch_size, ranking_size, ranking_size]