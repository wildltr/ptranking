#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""
import torch

from org.archive.eval.metric import tor_nDCG_at_k, tor_nDCG_at_ks, EMD_at_k

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

def idcg_std(sorted_labels):
	'''
	nums = np.power(2, sorted_labels) - 1.0
	denoms = np.log2(np.arange(len(sorted_labels)) + 2)
	idcgs = np.sum(nums/denoms, axis=1)
	return idcgs
	'''
	nums = torch.pow(2.0, sorted_labels) - 1.0
	a_range = torch.arange(sorted_labels.size(1), dtype=torch.float).to(device) if gpu else torch.arange(sorted_labels.size(1), dtype=torch.float)
	denoms = torch.log2(2.0 + a_range)
	idcgs = torch.sum(nums / denoms, dim=1)

	return idcgs

def tor_ndcg_at_k(ranker=None, test_Qs=None, k=10, multi_level_rele=True, query_aware=False, dict_query_cnts=None):
	'''
	There is no check based on the assumption (say light_filtering() is called)
	that each test instance Q includes at least k documents, and at least one relevant document.
	Or there will be errors.
	'''
	sum_ndcg_at_k = torch.zeros(1)
	cnt = torch.zeros(1)
	for entry in test_Qs:
		tor_test_ranking, tor_test_std_label_vec, qid = entry[0], torch.squeeze(entry[1], dim=0), entry[2][0]  # remove the size 1 of dim=0 from loader itself

		if tor_test_std_label_vec.size(0) < k: continue	# skip the query if the number of associated documents is smaller than k
		if gpu:
			if query_aware:
				tor_rele_pred = ranker.predict(tor_test_ranking.to(device), query_context=dict_query_cnts[qid])
			else:
				tor_rele_pred = ranker.predict(tor_test_ranking.to(device))

			tor_rele_pred = torch.squeeze(tor_rele_pred)
			tor_rele_pred = tor_rele_pred.cpu()
		else:
			if query_aware:
				tor_rele_pred = ranker.predict(tor_test_ranking, query_context=dict_query_cnts[qid])
			else:
				tor_rele_pred = ranker.predict(tor_test_ranking)
			tor_rele_pred = torch.squeeze(tor_rele_pred)

		_, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)

		sys_sorted_labels = tor_test_std_label_vec[tor_sorted_inds]
		ideal_sorted_labels, _ = torch.sort(tor_test_std_label_vec, descending=True)

		ndcg_at_k = tor_nDCG_at_k(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, k=k, multi_level_rele=multi_level_rele)
		sum_ndcg_at_k += ndcg_at_k
		cnt += 1

	avg_ndcg_at_k = sum_ndcg_at_k/cnt
	return  avg_ndcg_at_k


def tor_ndcg_at_ks(ranker=None, test_Qs=None, ks=[1, 5, 10], multi_level_rele=True, query_aware=False, dict_query_cnts=None):
	'''
	There is no check based on the assumption (say light_filtering() is called)
	that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
	Or there will be errors.
	'''
	sum_ndcg_at_ks = torch.zeros(len(ks))
	cnt = torch.zeros(1)
	for entry in test_Qs:
		tor_test_ranking, tor_test_std_label_vec, qid = entry[0], torch.squeeze(entry[1], dim=0), entry[2][0]  # remove the size 1 of dim=0 from loader itself

		if gpu:
			if query_aware:
				tor_rele_pred = ranker.predict(tor_test_ranking.to(device), query_context=dict_query_cnts[qid])
			else:
				tor_rele_pred = ranker.predict(tor_test_ranking.to(device))

			tor_rele_pred = torch.squeeze(tor_rele_pred)
			tor_rele_pred = tor_rele_pred.cpu()
		else:
			if query_aware:
				tor_rele_pred = ranker.predict(tor_test_ranking, query_context=dict_query_cnts[qid])
			else:
				tor_rele_pred = ranker.predict(tor_test_ranking)

			tor_rele_pred = torch.squeeze(tor_rele_pred)

		_, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)

		sys_sorted_labels = tor_test_std_label_vec[tor_sorted_inds]
		ideal_sorted_labels, _ = torch.sort(tor_test_std_label_vec, descending=True)

		ndcg_at_ks = tor_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=multi_level_rele)
		sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks)
		cnt += 1

	avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
	return avg_ndcg_at_ks


def emd_at_k(ranker=None, test_Qs=None, k=10, TL_AF=None, multi_level_rele=True):
	'''
	There is no check based on the assumption (say light_filtering() is called)
	that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
	Or there will be errors.
	'''
	assert 'S'==TL_AF or 'ST'==TL_AF

	sum_emd = 0.0
	cnt = 0
	for entry in test_Qs:
		tor_test_ranking, tor_test_std_label_vec = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0)  # remove the size 1 of dim=0 from loader itself
		if tor_test_std_label_vec.size(0) < k:
			continue

		if gpu:
			tor_test_ranking = tor_test_ranking.to(device)
			tor_rele_pred = ranker(tor_test_ranking)
			tor_rele_pred = torch.squeeze(tor_rele_pred)
			tor_rele_pred = tor_rele_pred.cpu()
		else:
			tor_rele_pred = ranker(tor_test_ranking)
			tor_rele_pred = torch.squeeze(tor_rele_pred)

		ideal_desc_labels, ideal_sorted_inds = torch.sort(tor_test_std_label_vec, descending=True)
		sys_corresponding_scores = tor_rele_pred[ideal_sorted_inds]

		tor_max_rele_level = torch.max(ideal_desc_labels)
		sys_corresponding_scores = sys_corresponding_scores * tor_max_rele_level

		emd_v = EMD_at_k(k=k, ideal_desc_labels=ideal_desc_labels[0:k].numpy(), sys_corresponding_scores=sys_corresponding_scores[0:k].numpy())
		sum_emd += emd_v

		cnt += 1

	avg_emd = sum_emd/cnt
	return avg_emd  # averaged value