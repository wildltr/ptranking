

import torch

from ptranking.metric.metric_utils import torch_dcg_at_k
from ptranking.metric.adhoc.adhoc_metric import torch_rankwise_err

"""
Using the evaluation metric as the optimization objective, which relies on the differentiable rank position.
"""

def precision_as_opt_objective(top_k=None, batch_smooth_ranks=None, batch_std_labels=None,
                               presort=False, opt_ideal=False, device=None):
    '''
    Precision expectation maximization.
    @param top_k: only use the top-k results if not None
    @param batch_std_labels:
    @param presort: whether the standard labels are already sorted in descending order or not
    @param opt_ideal: optimise the ideal ranking or sort results each time
    @return:
    '''
    '''
    According to the derivation of differentiable ranks, they are inversely correlated w.r.t. the scoring function.
    Thus should be used as the denominator. For being used as the numerator, i.e., 
    {batch_precision = torch.sum(batch_ascend_expt_ranks/batch_natural_ranks*top_k_labels, dim=1)/k},
    we need to revise the related formulations.
    '''
    ranking_size = batch_std_labels.size(1)
    batch_bi_std_labels = torch.clamp(batch_std_labels, min=0, max=1) # use binary labels
    # 1-dimension vector as batch via broadcasting
    batch_natural_ranks = torch.arange(ranking_size, dtype=torch.float, device=device).view(1, -1) + 1.0
    #batch_expt_ranks = get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos)

    if opt_ideal: # TODO adding dynamic shuffle is better, otherwise it will always be one order
        assert presort is True
        if top_k is None: # using cutoff
            batch_precision = torch.sum(batch_natural_ranks/batch_smooth_ranks*batch_bi_std_labels, dim=1)/ranking_size
        else:
            batch_precision = torch.sum(batch_natural_ranks[:, 0:top_k]/batch_smooth_ranks[:, 0:top_k]
                                        *batch_bi_std_labels[:, 0:top_k], dim=1) / top_k

        precision_loss = -torch.sum(batch_precision)
        zero_metric_value = False # there should be no zero value due to pre-filtering, i.e., min_rele=1
        return precision_loss, zero_metric_value

    else: # recommended than opt_ideal, due to the nature of dynamic ranking
        '''
        Sort the predicted ranks in a ascending natural order (i.e., 1, 2, 3, ..., n),
        the returned indices can be used to sort other vectors following the predicted order
        '''
        batch_ascend_expt_ranks, sort_indices = torch.sort(batch_smooth_ranks, dim=1, descending=False)
        # sort labels according to the expected ranks
        batch_sys_std_labels = torch.gather(batch_bi_std_labels, dim=1, index=sort_indices)

        if top_k is None: # using cutoff
            batch_precision = torch.sum(batch_natural_ranks / batch_ascend_expt_ranks * batch_sys_std_labels,
                                        dim=1) / ranking_size
            precision_loss = -torch.sum(batch_precision)
            return precision_loss, False
        else:
            top_k_labels = batch_sys_std_labels[:, 0:top_k]
            non_zero_inds = torch.nonzero(torch.sum(top_k_labels, dim=1))
            zero_metric_value = False if non_zero_inds.size(0) > 0 else True
            if zero_metric_value:
                return None, zero_metric_value
            else:
                pos_inds = non_zero_inds[:, 0]
                batch_precision = torch.sum(batch_natural_ranks[:, 0:top_k]
                                            / batch_ascend_expt_ranks[pos_inds, 0:top_k]
                                            * top_k_labels[pos_inds, :], dim=1) / top_k
                precision_loss = -torch.sum(batch_precision)
                return precision_loss, zero_metric_value


def AP_as_opt_objective(top_k=None, batch_smooth_ranks=None, batch_std_labels=None,
                        presort=False, opt_ideal=False, device=None):
    ranking_size = batch_std_labels.size(1)
    batch_bi_std_labels = torch.clamp(batch_std_labels, min=0, max=1) # use binary labels
    # 1-dimension vector as batch via broadcasting
    batch_natural_ranks = torch.arange(ranking_size, dtype=torch.float, device=device).view(1, -1) + 1.0

    #batch_expt_ranks = get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos)

    if opt_ideal: # TODO adding dynamic shuffle is better, otherwise it will always be one order
        assert presort is True
        if top_k is None: # using cutoff
            # an alternative formulation
            #batch_cumsum = torch.cumsum(batch_bi_std_labels, dim=1)
            #batch_ap = torch.sum(batch_cumsum / batch_expt_ranks * batch_bi_std_labels, dim=1) / torch.sum(batch_bi_std_labels, dim=1)

            # adopted formulation for being consistent w.r.t. the differentiable formulation of precision
            batch_cumsum = torch.cumsum(batch_natural_ranks / batch_smooth_ranks, dim=1)
            batch_rankwise_pre = batch_cumsum / batch_natural_ranks
            batch_ap = torch.sum(batch_rankwise_pre*batch_bi_std_labels, dim=1) / torch.sum(batch_bi_std_labels, dim=1)
            ap_loss = - torch.sum(batch_ap)
            return ap_loss, False
        else:
            top_k_labels = batch_bi_std_labels[:, 0:top_k]
            # an alternative formulation
            #batch_cumsum = torch.cumsum(top_k_labels, dim=1)
            #batch_ap = torch.sum(batch_cumsum / batch_expt_ranks[:, 0:top_k] * top_k_labels, dim=1) / torch.sum(top_k_labels, dim=1)

            # adopted formulation for being consistent w.r.t. the differentiable formulation of precision
            batch_cumsum = torch.cumsum(batch_natural_ranks/batch_smooth_ranks, dim=1)
            batch_rankwise_pre = batch_cumsum/batch_natural_ranks
            batch_ap = torch.sum(batch_rankwise_pre[:, 0:top_k] * top_k_labels, dim=1) / torch.sum(top_k_labels, dim=1)
            ap_loss = - torch.sum(batch_ap)
            return ap_loss, False
    else:
        '''
        Sort the predicted ranks in a ascending natural order (i.e., 1, 2, 3, ..., n),
        the returned indices can be used to sort other vectors following the predicted order
        '''
        batch_ascend_expt_ranks, sort_indices = torch.sort(batch_smooth_ranks, dim=1, descending=False)
        # sort labels according to the expected ranks
        batch_sys_std_labels = torch.gather(batch_bi_std_labels, dim=1, index=sort_indices)

        if top_k is None: # using cutoff
            batch_cumsum = torch.cumsum(batch_sys_std_labels, dim=1)
            batch_ap = torch.sum(batch_cumsum / batch_ascend_expt_ranks * batch_sys_std_labels,
                                 dim=1) / torch.sum(batch_sys_std_labels, dim=1)
            ap_loss = - torch.sum(batch_ap)
            return ap_loss, False
        else:
            top_k_labels = batch_sys_std_labels[:, 0:top_k]
            non_zero_inds = torch.nonzero(torch.sum(top_k_labels, dim=1))
            zero_metric_value = False if non_zero_inds.size(0) > 0 else True

            if zero_metric_value: # all zero values for the batch
                return None, zero_metric_value
            else:
                pos_inds = non_zero_inds[:, 0]

                # an alternative formulation
                #batch_cumsum = torch.cumsum(top_k_labels, dim=1)
                #batch_ap = torch.sum(batch_cumsum[pos_inds, 0:top_k] / batch_ascend_expt_ranks[pos_inds, 0:top_k]
                #                     * top_k_labels[pos_inds, :], dim=1) / torch.sum(top_k_labels[pos_inds, :], dim=1)

                # adopted formulation for being consistent w.r.t. the differentiable formulation of precision
                batch_cumsum = torch.cumsum(batch_natural_ranks/batch_ascend_expt_ranks, dim=1)
                batch_rankwise_pre = batch_cumsum/batch_natural_ranks
                batch_ap = torch.sum(batch_rankwise_pre[pos_inds, 0:top_k] * top_k_labels[pos_inds, :],
                                     dim=1) / torch.sum(top_k_labels[pos_inds, :], dim=1)
                ap_loss = - torch.sum(batch_ap)
                return ap_loss, False


def nERR_as_opt_objective(top_k=None, batch_smooth_ranks=None, batch_std_labels=None,
                          device=None, opt_ideal=True, presort=False):
    ranking_size = batch_std_labels.size(1)
    max_label = torch.max(batch_std_labels)

    assert presort is True
    k = ranking_size if top_k is None else top_k
    batch_ideal_err = torch_rankwise_err(batch_std_labels, max_label=max_label, k=k, point=True, device=device)

    #batch_expt_ranks = get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos)

    if opt_ideal:
        used_batch_expt_ranks = batch_smooth_ranks
        used_batch_labels = batch_std_labels
    else:
        '''
        Sort the predicted ranks in a ascending natural order (i.e., 1, 2, 3, ..., n),
        the returned indices can be used to sort other vectors following the predicted order
        '''
        batch_ascend_expt_ranks, sort_indices = torch.sort(batch_smooth_ranks, dim=1, descending=False)
        # sort labels according to the expected ranks
        batch_sys_std_labels = torch.gather(batch_std_labels, dim=1, index=sort_indices)

        used_batch_expt_ranks = batch_ascend_expt_ranks
        used_batch_labels = batch_sys_std_labels

    if top_k is None:
        expt_ranks = 1.0 / used_batch_expt_ranks
        satis_pros = (torch.pow(2.0, used_batch_labels) - 1.0) / torch.pow(2.0, max_label)
        unsatis_pros = torch.ones_like(used_batch_labels) - satis_pros
        cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
        cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros)
        cascad_unsatis_pros[:, 1:ranking_size] = cum_unsatis_pros[:, 0:ranking_size - 1]

        expt_satis_ranks = expt_ranks * satis_pros * cascad_unsatis_pros
        batch_err = torch.sum(expt_satis_ranks, dim=1)
        batch_nerr = batch_err / batch_ideal_err
        nerr_loss = -torch.sum(batch_nerr)
        return nerr_loss, False
    else:
        top_k_labels = used_batch_labels[:, 0:top_k]
        if opt_ideal:
            pos_rows = torch.arange(top_k_labels.size(0), dtype=torch.long) # all rows
        else:
            non_zero_inds = torch.nonzero(torch.sum(top_k_labels, dim=1))
            zero_metric_value = False if non_zero_inds.size(0) > 0 else True
            if zero_metric_value:
                return None, zero_metric_value  # should not be optimized due to no useful training signal
            else:
                pos_rows = non_zero_inds[:, 0]

        expt_ranks = 1.0 / used_batch_expt_ranks[pos_rows, 0:top_k]
        satis_pros = (torch.pow(2.0, top_k_labels) - 1.0) / torch.pow(2.0, max_label)
        unsatis_pros = torch.ones_like(top_k_labels) - satis_pros
        cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
        cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros)
        cascad_unsatis_pros[:, 1:top_k] = cum_unsatis_pros[:, 0:top_k-1]

        expt_satis_ranks = expt_ranks * satis_pros[pos_rows, :] * cascad_unsatis_pros[pos_rows, :]
        batch_err = torch.sum(expt_satis_ranks, dim=1)
        batch_nerr = batch_err / batch_ideal_err[pos_rows, :]
        nerr_loss = -torch.sum(batch_nerr)
        return nerr_loss, False


def nDCG_as_opt_objective(top_k=None, batch_smooth_ranks=None, batch_std_labels=None,
                          label_type=None, device=None, opt_ideal=True, presort=False):
    assert presort is True # otherwise re-sorting is required
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_std_labels, label_type=label_type, device=device)
    #batch_expt_ranks = get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos)

    if opt_ideal: # tie-shuffling is not needed, since expected ranks remain the same, so as to other metrics.
        batch_gains = torch.pow(2.0, batch_std_labels) - 1.0
        batch_dists = 1.0 / torch.log2(batch_smooth_ranks + 1.0)  # discount co-efficients
    else:
        '''
        Sort the predicted ranks in a ascending natural order (i.e., 1, 2, 3, ..., n),
        the returned indices can be used to sort other vectors following the predicted order
        '''
        batch_ascend_expt_ranks, batch_resort_inds = torch.sort(batch_smooth_ranks, dim=1, descending=False)
        # sort labels according to the expected ranks
        batch_sys_std_labels = torch.gather(batch_std_labels, dim=1, index=batch_resort_inds)
        batch_gains = torch.pow(2.0, batch_sys_std_labels) - 1.0
        batch_dists = 1.0 / torch.log2(batch_ascend_expt_ranks + 1.0)  # discount co-efficients

    if top_k is None:
        batch_dcgs = batch_gains * batch_dists
        batch_expt_nDCG = torch.sum(batch_dcgs / batch_idcgs, dim=1)
        batch_loss = - torch.sum(batch_expt_nDCG)
        zero_metric_value = False
        return batch_loss, zero_metric_value
    else:
        if opt_ideal:
            batch_dcgs = batch_dists[:, 0:top_k] * batch_gains[:, 0:top_k]
            zero_metric_value = False
            batch_expt_nDCG_k = torch.sum(batch_dcgs / batch_idcgs[:, 0:top_k], dim=1)
            batch_loss = - torch.sum(batch_expt_nDCG_k)
            return batch_loss, zero_metric_value
        else:
            top_batch_gains = batch_gains[:, 0:top_k]
            non_zero_inds = torch.nonzero(torch.sum(top_batch_gains, dim=1))
            zero_metric_value = False if non_zero_inds.size(0) > 0 else True
            if zero_metric_value:
                return None, zero_metric_value # should not be optimized due to no useful training signal
            else:
                pos_inds = non_zero_inds[:, 0]
                batch_dcgs = batch_dists[pos_inds, 0:top_k] * top_batch_gains[pos_inds, :]
                batch_expt_nDCG_k = torch.sum(batch_dcgs / batch_idcgs[pos_inds, 0:top_k], dim=1)
                batch_loss = - torch.sum(batch_expt_nDCG_k)
                return batch_loss, zero_metric_value

