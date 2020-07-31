#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import ot
import numpy as np

import torch

from org.archive.l2r_global import global_gpu as gpu, global_device as device, tensor

""" Evaluation Metrics """

def rele_gain(rele_level, gain_base=2.0):
	gain = np.power(gain_base, rele_level) - 1.0
	return gain

def torch_ideal_dcg(batch_sorted_labels, gpu=False):
	'''
	:param sorted_labels: [batch, ranking_size]
	:return: [batch, 1]
	'''
	batch_gains = torch.pow(2.0, batch_sorted_labels) - 1.0
	batch_ranks = torch.arange(batch_sorted_labels.size(1))

	batch_discounts = torch.log2(2.0 + batch_ranks.type(torch.cuda.FloatTensor)) if gpu else torch.log2(2.0 + batch_ranks.type(torch.FloatTensor))
	batch_ideal_dcg = torch.sum(batch_gains / batch_discounts, dim=1, keepdim=True)

	return batch_ideal_dcg

""" Precision """
def torch_p_at_ks(sys_sorted_labels, ks=None):
	'''	Precision at ks
	:param sys_sorted_labels: system's predicted ltr_adhoc of labels in a descending order
	:param ks:
	:return:
	'''
	valid_max = sys_sorted_labels.size(0)
	used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)

	ranks = (torch.arange(max_cutoff) + 1.0)

	sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
	bi_sys_sorted_labels = torch.clamp(sys_sorted_labels, min=0, max=1) # binary
	sys_cumsum_reles = torch.cumsum(bi_sys_sorted_labels, dim=0)

	sys_rankwise_precision = sys_cumsum_reles / ranks
	sys_p_at_ks = sys_rankwise_precision[inds]
	if valid_max < max(ks):
		padded_p_at_ks = torch.zeros(len(ks))
		padded_p_at_ks[0:len(used_ks)] = sys_p_at_ks
		return padded_p_at_ks
	else:
		return sys_p_at_ks

""" Average Precision """
def torch_ap_at_ks(sys_sorted_labels, ideal_sorted_labels, ks=None):
	'''
	AP(average precision) at ks (i.e., different cutoff values)
	:param ideal_sorted_labels: the ideal ltr_adhoc of labels
	:param sys_sorted_labels: system's predicted ltr_adhoc of labels in a descending order
	:param ks:
	:return:
	'''

	valid_max = sys_sorted_labels.size(0)
	used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks
	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)

	ranks = (torch.arange(max_cutoff) + 1.0)

	sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
	bi_sys_sorted_labels = torch.clamp(sys_sorted_labels, min=0, max=1) # binary
	sys_cumsum_reles = torch.cumsum(bi_sys_sorted_labels, dim=0)

	sys_rankwise_precision = sys_cumsum_reles / ranks # rank-wise precision
	sys_cumsum_precision = torch.cumsum(sys_rankwise_precision * bi_sys_sorted_labels, dim=0) # exclude precisions of which the corresponding documents are not relevant

	std_cumsum_reles = torch.cumsum(ideal_sorted_labels, dim=0)
	sys_poswise_ap = sys_cumsum_precision / std_cumsum_reles[0:max_cutoff]
	sys_ap_at_ks = sys_poswise_ap[inds]

	if valid_max < max(ks):
		padded_ap_at_ks = torch.zeros(len(ks))
		padded_ap_at_ks[0:len(used_ks)] = sys_ap_at_ks
		return padded_ap_at_ks
	else:
		return sys_ap_at_ks


""" NERR """

def torch_ideal_err(sorted_labels, k=10, point=True, gpu=False):
	assert sorted_labels.size(0) >= k

	max_label = torch.max(sorted_labels)

	labels = sorted_labels[0:k]
	satis_pros = (torch.pow(2.0, labels) - 1.0) / torch.pow(2.0, max_label)

	unsatis_pros = torch.ones_like(labels) - satis_pros
	cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=0)

	if gpu:
		ranks = torch.arange(k).type(tensor) + 1.0
		expt_ranks = 1.0 / ranks
	else:
		ranks = torch.arange(k) + 1.0
		expt_ranks = 1.0 / ranks

	cascad_unsatis_pros = ranks
	cascad_unsatis_pros[1:k] = cum_unsatis_pros[0:k-1]

	expt_satis_ranks = expt_ranks * satis_pros * cascad_unsatis_pros  # w.r.t. all rank positions

	if point: # a specific position
		ideal_err = torch.sum(expt_satis_ranks, dim=0)
		return ideal_err

	else:
		ideal_err_at_ks = torch.cumsum(expt_satis_ranks, dim=0)
		return ideal_err_at_ks


def torch_batch_ideal_err(batch_sorted_labels, k=10, gpu=False, point=True):
	assert batch_sorted_labels.size(1) > k

	batch_max = torch.max(batch_sorted_labels, dim=1)

	batch_labels = batch_sorted_labels[:, 0:k]
	batch_satis_pros = (torch.pow(2.0, batch_labels) - 1.0) / torch.pow(2.0, batch_max)

	batch_unsatis_pros = torch.ones(batch_labels) - batch_satis_pros
	batch_cum_unsatis_pros = torch.cumprod(batch_unsatis_pros, dim=1)

	positions = torch.arange(k) + 1.0
	positions = positions.view(1, -1)
	positions = torch.repeat_interleave(positions, batch_sorted_labels.size(0), dim=0)

	batch_expt_ranks = 1.0 / positions

	cascad_unsatis_pros = positions
	cascad_unsatis_pros[:, 1:k] = batch_cum_unsatis_pros[:, 0:k-1]

	expt_satis_ranks = batch_expt_ranks * batch_satis_pros * cascad_unsatis_pros  # w.r.t. all rank positions

	if point:
		batch_errs = torch.sum(expt_satis_ranks, dim=1)
		return batch_errs
	else:
		batch_err_at_ks = torch.cumsum(expt_satis_ranks, dim=1)
		return batch_err_at_ks


def torch_nerr_at_ks(sys_sorted_labels, ideal_sorted_labels, ks=None, multi_level_rele=True):
	'''
	:param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
	:param ks:
	:param multi_level_rele:
	:return:
	'''
	valid_max = sys_sorted_labels.size(0)
	used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

	max_cutoff = max(used_ks)
	inds = torch.from_numpy(np.asarray(used_ks) - 1)
	if multi_level_rele:
		positions = torch.arange(max_cutoff) + 1.0
		expt_ranks = 1.0 / positions    # expected stop positions

		tor_max_rele = torch.max(sys_sorted_labels)
		satis_pros = (torch.pow(2.0, sys_sorted_labels[0:max_cutoff]) - 1.0)/torch.pow(2.0, tor_max_rele)
		non_satis_pros = torch.ones(max_cutoff) - satis_pros
		cum_non_satis_pros = torch.cumprod(non_satis_pros, dim=0)

		cascad_non_satis_pros = positions
		cascad_non_satis_pros[1:max_cutoff] = cum_non_satis_pros[0:max_cutoff-1]
		expt_satis_ranks = expt_ranks * satis_pros * cascad_non_satis_pros  # w.r.t. all rank positions

		err_at_ks = torch.cumsum(expt_satis_ranks, dim=0)
		#print(err_at_ks)

		ideal_err_at_ks = torch_ideal_err(ideal_sorted_labels, k=max_cutoff, point=False)
		tmp_nerr_at_ks = err_at_ks/ideal_err_at_ks

		nerr_at_ks = tmp_nerr_at_ks[inds]
		if valid_max < max(ks):
			padded_nerr_at_ks = torch.zeros(len(ks))
			padded_nerr_at_ks[0:len(used_ks)] = nerr_at_ks
			return padded_nerr_at_ks
		else:
			return nerr_at_ks
	else:
		raise NotImplementedError



""" nDCG """
def torch_discounted_cumu_gain_at_k(sorted_labels, cutoff, multi_level_rele=True):
	'''
	ICML-nDCG, which places stronger emphasis on retrieving relevant documents
	:param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
	:param max_cutoff: the maximum rank position to be considered
	:param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
	:return: cumulative gains for each rank position
	'''
	if multi_level_rele:    #the common case with multi-level labels
		nums = torch.pow(2.0, sorted_labels[0:cutoff]) - 1.0
	else:
		nums = sorted_labels[0:cutoff]  #the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)

	denoms = torch.log2(torch.arange(cutoff).type(torch.FloatTensor) + 2.0)   #discounting factor
	dited_cumu_gain = torch.sum(nums/denoms)   # discounted cumulative gain value

	return dited_cumu_gain

def torch_discounted_cumu_gain_at_ks(sorted_labels, max_cutoff, multi_level_rele=True):
	'''
	ICML-nDCG, which places stronger emphasis on retrieving relevant documents
	:param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
	:param max_cutoff: the maximum rank position to be considered
	:param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
	:return: cumulative gains for each rank position
	'''

	if multi_level_rele:    #the common case with multi-level labels
		nums = torch.pow(2.0, sorted_labels[0:max_cutoff]) - 1.0
	else:
		nums = sorted_labels[0:max_cutoff]  #the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)

	denoms = torch.log2(torch.arange(max_cutoff).type(torch.FloatTensor) + 2.0)   #discounting factor
	dited_cumu_gains = torch.cumsum(nums/denoms, dim=0)   # discounted cumulative gain value w.r.t. each position

	return dited_cumu_gains

def torch_nDCG_at_k(sys_sorted_labels, ideal_sorted_labels, k=None, multi_level_rele=True):
	sys_dited_cg_at_k = torch_discounted_cumu_gain_at_k(sys_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)  # only using the cumulative gain at the final rank position
	ideal_dited_cg_at_k = torch_discounted_cumu_gain_at_k(ideal_sorted_labels, cutoff=k, multi_level_rele=multi_level_rele)
	ndcg_at_k = sys_dited_cg_at_k / ideal_dited_cg_at_k
	return ndcg_at_k

def torch_nDCG_at_ks(sys_sorted_labels, ideal_sorted_labels, ks=None, multi_level_rele=True):
	valid_max = sys_sorted_labels.size(0)
	used_ks = [k for k in ks if k<=valid_max] if valid_max < max(ks) else ks

	inds = torch.from_numpy(np.asarray(used_ks) - 1)
	sys_dited_cgs = torch_discounted_cumu_gain_at_ks(sys_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
	sys_dited_cg_at_ks = sys_dited_cgs[inds]  # get cumulative gains at specified rank positions
	ideal_dited_cgs = torch_discounted_cumu_gain_at_ks(ideal_sorted_labels, max_cutoff=max(used_ks), multi_level_rele=multi_level_rele)
	ideal_dited_cg_at_ks = ideal_dited_cgs[inds]

	ndcg_at_ks = sys_dited_cg_at_ks / ideal_dited_cg_at_ks

	if valid_max < max(ks):
		padded_ndcg_at_ks = torch.zeros(len(ks))
		padded_ndcg_at_ks[0:len(used_ks)] = ndcg_at_ks
		return padded_ndcg_at_ks
	else:
		return ndcg_at_ks



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

		p_at_ks_per_query = torch_p_at_ks(sys_sorted_labels=sys_sorted_labels, ks=ks)
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


def EMD_at_k(k, ideal_desc_labels, sys_corresponding_scores, group_div_cost=np.e, margin_to_non_rele=100.0, rele_gain_base=4.0):
	if k>len(ideal_desc_labels):
		return 0.0

	cost_mat = eval_cost_mat_group(ideal_desc_labels, group_div_cost=group_div_cost, margin_to_non_rele=margin_to_non_rele, rele_gain_base=rele_gain_base)

	ideal_histogram = np_stable_softmax_e(ideal_desc_labels)
	sys_historgram = np_stable_softmax_e(sys_corresponding_scores)

	# %% EMD
	G0 = ot.emd(a=sys_historgram, b=ideal_histogram, M=cost_mat)
	emd_value = np.sum(G0 * cost_mat)

	return emd_value
