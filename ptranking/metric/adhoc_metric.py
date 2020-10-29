#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
The commonly used IR evaluation metrics, such as AP (average precision), nDCG and ERR
"""

import torch
import numpy as np
from ptranking.ltr_global import global_gpu as gpu, global_device as device, tensor

# todo string_key for multi_level_rele: 1> more clear and be able to extend;

""" Precision """

def torch_precision_at_ks(batch_sys_sorted_labels, ks=None):
	'''	Precision at ks
	:param sys_sorted_labels: [batch_size, ranking_size] system's predicted ltr_adhoc of labels in a descending order
	:param ks: cutoff values
	:return: [batch_size, len(ks)]
	'''
	valid_max_cutoff = batch_sys_sorted_labels.size(1)
	need_padding = True if valid_max_cutoff < max(ks) else False
	used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks

	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)

	batch_sys_sorted_labels = batch_sys_sorted_labels[:, 0:max_cutoff]
	batch_bi_sys_sorted_labels = torch.clamp(batch_sys_sorted_labels, min=0, max=1) # binary
	batch_sys_cumsum_reles = torch.cumsum(batch_bi_sys_sorted_labels, dim=1)

	batch_ranks = (torch.arange(max_cutoff).type(tensor).expand_as(batch_sys_cumsum_reles) + 1.0)

	batch_sys_rankwise_precision = batch_sys_cumsum_reles / batch_ranks
	batch_sys_p_at_ks = batch_sys_rankwise_precision[:, inds]
	if need_padding:
		padded_p_at_ks = torch.zeros(batch_sys_sorted_labels.size(0), len(ks))
		padded_p_at_ks[:, 0:len(used_ks)] = batch_sys_p_at_ks
		return padded_p_at_ks
	else:
		return batch_sys_p_at_ks

""" Average Precision """

def torch_ap_at_ks(batch_sys_sorted_labels, batch_ideal_sorted_labels, ks=None):
	'''
	AP(average precision) at ks (i.e., different cutoff values)
	:param ideal_sorted_labels: [batch_size, ranking_size] the ideal ltr_adhoc of labels
	:param sys_sorted_labels: [batch_size, ranking_size] system's predicted ltr_adhoc of labels in a descending order
	:param ks:
	:return: [batch_size, len(ks)]
	'''
	valid_max_cutoff = batch_sys_sorted_labels.size(1)
	need_padding = True if valid_max_cutoff < max(ks) else False
	used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks
	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)

	batch_sys_sorted_labels = batch_sys_sorted_labels[:, 0:max_cutoff]
	batch_bi_sys_sorted_labels = torch.clamp(batch_sys_sorted_labels, min=0, max=1) # binary
	batch_sys_cumsum_reles = torch.cumsum(batch_bi_sys_sorted_labels, dim=1)

	batch_ranks = (torch.arange(max_cutoff).type(tensor).expand_as(batch_sys_cumsum_reles) + 1.0)

	batch_sys_rankwise_precision = batch_sys_cumsum_reles / batch_ranks # rank-wise precision
	batch_sys_cumsum_precision = torch.cumsum(batch_sys_rankwise_precision * batch_bi_sys_sorted_labels, dim=1) # exclude precisions of which the corresponding documents are not relevant

	batch_std_cumsum_reles = torch.cumsum(batch_ideal_sorted_labels, dim=1)
	batch_sys_rankwise_ap = batch_sys_cumsum_precision / batch_std_cumsum_reles[:, 0:max_cutoff]
	batch_sys_ap_at_ks = batch_sys_rankwise_ap[:, inds]

	if need_padding:
		padded_ap_at_ks = torch.zeros(batch_sys_sorted_labels.size(0), len(ks))
		padded_ap_at_ks[:, 0:len(used_ks)] = batch_sys_ap_at_ks
		return padded_ap_at_ks
	else:
		return batch_sys_ap_at_ks


""" NERR """

def torch_rankwise_err(batch_sorted_labels, max_label=None, k=10, point=True):
	assert batch_sorted_labels.size(1) >= k
	assert max_label is not None # it is either query-level or corpus-level

	batch_labels = batch_sorted_labels[:, 0:k]
	batch_satis_probs = (torch.pow(2.0, batch_labels) - 1.0) / torch.pow(2.0, max_label)

	batch_unsatis_probs = torch.ones_like(batch_labels) - batch_satis_probs
	batch_cum_unsatis_probs = torch.cumprod(batch_unsatis_probs, dim=1)

	batch_ranks = torch.arange(k).type(tensor).expand_as(batch_labels) + 1.0
	batch_expt_ranks = 1.0 / batch_ranks

	batch_cascad_unsatis_probs = torch.ones_like(batch_expt_ranks)
	batch_cascad_unsatis_probs[:, 1:k] = batch_cum_unsatis_probs[:, 0:k-1]

	batch_expt_satis_ranks = batch_expt_ranks * batch_satis_probs * batch_cascad_unsatis_probs  # w.r.t. all rank positions

	if point: # a specific position
		batch_err_at_k = torch.sum(batch_expt_satis_ranks, dim=1)
		return batch_err_at_k
	else:
		batch_rankwise_err = torch.cumsum(batch_expt_satis_ranks, dim=1)
		return batch_rankwise_err

def torch_nerr_at_k(batch_sys_sorted_labels, batch_ideal_sorted_labels, k=None, multi_level_rele=True):
	valid_max_cutoff = batch_sys_sorted_labels.size(1)
	cutoff = max(valid_max_cutoff, k)

	if multi_level_rele:
		max_label = torch.max(batch_ideal_sorted_labels)
		batch_sys_err_at_k = torch_rankwise_err(batch_sys_sorted_labels, max_label=max_label, k=cutoff, point=True)
		batch_ideal_err_at_k = torch_rankwise_err(batch_ideal_sorted_labels, max_label=max_label, k=cutoff, point=True)
		batch_nerr_at_k = batch_sys_err_at_k / batch_ideal_err_at_k
		return batch_nerr_at_k
	else:
		raise NotImplementedError

def torch_nerr_at_ks(batch_sys_sorted_labels, batch_ideal_sorted_labels, ks=None, multi_level_rele=True):
	'''
	:param sys_sorted_labels: [batch_size, ranking_size] the standard labels sorted in descending order according to predicted relevance scores
	:param ks:
	:param multi_level_rele:
	:return: [batch_size, len(ks)]
	'''
	valid_max_cutoff = batch_sys_sorted_labels.size(1)
	need_padding = True if valid_max_cutoff < max(ks) else False
	used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks

	max_label = torch.max(batch_ideal_sorted_labels)
	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)

	if multi_level_rele:
		batch_sys_rankwise_err = torch_rankwise_err(batch_sys_sorted_labels, max_label=max_label, k=max_cutoff, point=False)
		batch_ideal_rankwise_err = torch_rankwise_err(batch_ideal_sorted_labels, max_label=max_label, k=max_cutoff, point=False)
		batch_rankwise_nerr = batch_sys_rankwise_err/batch_ideal_rankwise_err
		batch_nerr_at_ks = batch_rankwise_nerr[:, inds]
		if need_padding:
			padded_nerr_at_ks = torch.zeros(batch_sys_sorted_labels.size(0), len(ks))
			padded_nerr_at_ks[:, 0:len(used_ks)] = batch_nerr_at_ks
			return padded_nerr_at_ks
		else:
			return batch_nerr_at_ks
	else:
		raise NotImplementedError


""" nDCG """

def torch_dcg_at_k(batch_sorted_labels, cutoff=None, multi_level_rele=True):
	'''
	ICML-nDCG, which places stronger emphasis on retrieving relevant documents
	:param batch_sorted_labels: [batch_size, ranking_size] a batch of ranked labels (either standard or predicted by a system)
	:param cutoff: the cutoff position
	:param multi_level_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
	:return: [batch_size, 1] cumulative gains for each rank position
	'''
	if cutoff is None: # using whole list
		cutoff = batch_sorted_labels.size(1)

	if multi_level_rele:    #the common case with multi-level labels
		batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
	else: # the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)
		batch_numerators = batch_sorted_labels[:, 0:cutoff]

	batch_discounts = torch.log2(torch.arange(cutoff).type(tensor).expand_as(batch_numerators) + 2.0)
	batch_dcg_at_k = torch.sum(batch_numerators/batch_discounts, dim=1, keepdim=True)
	return batch_dcg_at_k

def torch_dcg_at_ks(batch_sorted_labels, max_cutoff, multi_level_rele=True):
	'''
	:param batch_sorted_labels: [batch_size, ranking_size] ranked labels (either standard or predicted by a system)
	:param max_cutoff: the maximum cutoff value
	:param multi_level_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
	:return: [batch_size, max_cutoff] cumulative gains for each rank position
	'''
	if multi_level_rele:    #the common case with multi-level labels
		batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:max_cutoff]) - 1.0
	else: # the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)
		batch_numerators = batch_sorted_labels[:, 0:max_cutoff]

	batch_discounts = torch.log2(torch.arange(max_cutoff).type(tensor).expand_as(batch_numerators) + 2.0)
	batch_dcg_at_ks = torch.cumsum(batch_numerators/batch_discounts, dim=1)   # dcg w.r.t. each position
	return batch_dcg_at_ks

def torch_nDCG_at_k(batch_sys_sorted_labels, batch_ideal_sorted_labels, k=None, multi_level_rele=True):
	batch_sys_dcg_at_k = torch_dcg_at_k(batch_sys_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)  # only using the cumulative gain at the final rank position
	batch_ideal_dcg_at_k = torch_dcg_at_k(batch_ideal_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)
	batch_ndcg_at_k = batch_sys_dcg_at_k / batch_ideal_dcg_at_k
	return batch_ndcg_at_k

def torch_nDCG_at_ks(batch_sys_sorted_labels, batch_ideal_sorted_labels, ks=None, multi_level_rele=True):
	valid_max_cutoff = batch_sys_sorted_labels.size(1)
	used_ks = [k for k in ks if k<=valid_max_cutoff] if valid_max_cutoff < max(ks) else ks

	inds = torch.from_numpy(np.asarray(used_ks) - 1)
	batch_sys_dcgs = torch_dcg_at_ks(batch_sys_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
	batch_sys_dcg_at_ks = batch_sys_dcgs[:, inds]  # get cumulative gains at specified rank positions
	batch_ideal_dcgs = torch_dcg_at_ks(batch_ideal_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
	batch_ideal_dcg_at_ks = batch_ideal_dcgs[:, inds]

	batch_ndcg_at_ks = batch_sys_dcg_at_ks / batch_ideal_dcg_at_ks

	if valid_max_cutoff < max(ks):
		padded_ndcg_at_ks = torch.zeros(batch_sys_sorted_labels.size(0), len(ks))
		padded_ndcg_at_ks[:, 0:len(used_ks)] = batch_ndcg_at_ks
		return padded_ndcg_at_ks
	else:
		return batch_ndcg_at_ks



""" Kendall'tau Coefficient """
def torch_kendall_tau(sys_ranking, natural_ascending_as_reference = True):
	'''
	$\tau = 1.0 - \frac{2S(\pi, \delta)}{N(N-1)/2}$, cf. 2006-Automatic Evaluation of Information Ordering: Kendall’s Tau
	The tie issue is not considered within this version.
	The current implementation is just counting the inversion number, then normalized by n(n-1)/2. The underlying assumption is that the reference ltr_adhoc is the ideal ltr_adhoc, say labels are ordered in a descending order.
	:param sys_ranking: system's ltr_adhoc, whose entries can be predicted values, labels, etc.
	:return:
	'''
	assert 1 == len(sys_ranking.size()) # one-dimension vector

	ranking_size = sys_ranking.size(0)
	pair_diffs = sys_ranking.view(-1, 1) - sys_ranking.view(1, -1)

	if natural_ascending_as_reference:
		bi_pair_diffs = torch.clamp(pair_diffs, min=0, max=1)
		bi_pair_diffs_triu1 = torch.triu(bi_pair_diffs, diagonal=1)
		#print('bi_pair_diffs_triu1\n', bi_pair_diffs_triu1)

		tau = 1.0 - 4 * torch.sum(bi_pair_diffs_triu1) / (ranking_size*(ranking_size-1))

	else: # i.e., natural descending as the reference
		bi_pair_diffs = torch.clamp(pair_diffs, min=-1, max=0)
		bi_pair_diffs_triu1 = torch.triu(bi_pair_diffs, diagonal=1)
		#print('bi_pair_diffs_triu1\n', bi_pair_diffs_triu1)
		print('total discordant: ', 2*torch.sum(bi_pair_diffs_triu1))

		tau = 1.0 + 4 * torch.sum(bi_pair_diffs_triu1) / (ranking_size*(ranking_size-1))

	return tau

def rele_gain(rele_level, gain_base=2.0):
	gain = np.power(gain_base, rele_level) - 1.0
	return gain

def np_metric_at_ks(ranker=None, test_Qs=None, ks=[1, 5, 10], multi_level_rele=True, max_rele_level=None):
	'''
	There is no check based on the assumption (say light_filtering() is called)
	that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
	Or there will be errors.
	'''
	cnt = 0
	sum_ndcg_at_ks = torch.zeros(len(ks))
	sum_err_at_ks = torch.zeros(len(ks))
	sum_ap_at_ks = torch.zeros(len(ks))
	sum_p_at_ks = torch.zeros(len(ks))

	list_ndcg_at_ks_per_q = []
	list_err_at_ks_per_q = []
	list_ap_at_ks_per_q = []
	list_p_at_ks_per_q = []

	for entry in test_Qs:
		tor_test_ranking, tor_test_std_label_vec = entry[1], torch.squeeze(entry[2], dim=0)  # remove the size 1 of dim=0 from loader itself

		if gpu:
			tor_rele_pred = ranker.predict(tor_test_ranking.to(device))
			tor_rele_pred = torch.squeeze(tor_rele_pred)
			tor_rele_pred = tor_rele_pred.cpu()
		else:
			tor_rele_pred = ranker.predict(tor_test_ranking)
			tor_rele_pred = torch.squeeze(tor_rele_pred)

		_, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)

		sys_sorted_labels = tor_test_std_label_vec[tor_sorted_inds]
		ideal_sorted_labels, _ = torch.sort(tor_test_std_label_vec, descending=True)

		ndcg_at_ks_per_query = torch_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=multi_level_rele)
		sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks_per_query)
		list_ndcg_at_ks_per_q.append(ndcg_at_ks_per_query.numpy())

		err_at_ks_per_query = torch_nerr_at_ks(sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=multi_level_rele)
		sum_err_at_ks = torch.add(sum_err_at_ks, err_at_ks_per_query)
		list_err_at_ks_per_q.append(err_at_ks_per_query.numpy())

		ap_at_ks_per_query = torch_ap_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks)
		sum_ap_at_ks = torch.add(sum_ap_at_ks, ap_at_ks_per_query)
		list_ap_at_ks_per_q.append(ap_at_ks_per_query.numpy())

		p_at_ks_per_query = torch_precision_at_ks(sys_sorted_labels=sys_sorted_labels, ks=ks)
		sum_p_at_ks = torch.add(sum_p_at_ks, p_at_ks_per_query)
		list_p_at_ks_per_q.append(p_at_ks_per_query.numpy())

		cnt += 1

	ndcg_at_ks = sum_ndcg_at_ks/cnt
	err_at_ks = sum_err_at_ks/cnt
	ap_at_ks = sum_ap_at_ks / cnt
	p_at_ks = sum_p_at_ks/cnt

	return ndcg_at_ks.numpy(), err_at_ks.numpy(), ap_at_ks.numpy(), p_at_ks.numpy(), list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q


def np_stable_softmax_e(histogram):
	histogram = np.asarray(histogram, dtype=np.float64)
	max_v, _ = np.max(histogram, dim=0)  # a transformation aiming for higher stability when computing softmax() with exp()
	hist = histogram - max_v
	hist_exped = np.exp(hist)
	probs = np.divide(hist_exped, np.sum(hist_exped, dim=0))
	return probs


def eval_cost_mat_group(sorted_std_labels, group_div_cost=np.e, margin_to_non_rele=100.0, rele_gain_base=4.0):
	size_ranking = len(sorted_std_labels)
	cost_mat = np.zeros(shape=(size_ranking, size_ranking), dtype=np.float64)

	for i in range(size_ranking):
		i_rele_level = sorted_std_labels[i]
		for j in range(size_ranking):
			if i==j:
				cost_mat[i, j] = 0
			else:
				j_rele_level = sorted_std_labels[j]

				if i_rele_level == j_rele_level:
					cost_mat[i, j] = group_div_cost
				else:
					cost_mat[i, j] = np.abs(rele_gain(i_rele_level, gain_base=rele_gain_base) - rele_gain(j_rele_level, gain_base=rele_gain_base))
					if 0 == i_rele_level or 0 == j_rele_level:
						cost_mat[i, j] += margin_to_non_rele

	return cost_mat
