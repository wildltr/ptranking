#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch

from org.archive.ranker.transform import tor_batch_triu
from org.archive.ranker.ranker import AbstractNeuralRanker
from org.archive.eval.eval_utils import idcg_std

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

def lambdaRank_loss_function(batch_preds=None, batch_stds=None):
	'''
	This method will impose explicit bias to highly ranked documents that are essentially ties
	:param batch_preds:
	:param batch_stds:
	:return:
	'''
	batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
	batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

	batch_pred_diffs = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
	batch_pred_pairwise_cmps = tor_batch_triu(batch_pred_diffs, k=1)

	tmp_batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
	std_ones = torch.ones(tmp_batch_std_diffs.size())
	std_minus_ones = std_ones - 2.0
	batch_std_diffs = torch.where(tmp_batch_std_diffs > 0, std_ones, tmp_batch_std_diffs)
	batch_std_diffs = torch.where(batch_std_diffs < 0, std_minus_ones, batch_std_diffs)
	batch_std_pairwise_cmps = tor_batch_triu(batch_std_diffs, k=1)

	batch_1st_part = (1.0 - batch_std_pairwise_cmps) * batch_pred_pairwise_cmps * 0.5  # cf. the equation in page-3
	batch_2nd_part = torch.log(torch.exp(-batch_pred_pairwise_cmps) + 1.0)  # cf. the equation in page-3

	''' delta nDCG '''
	batch_idcgs = idcg_std(batch_stds)  # use original input ideal ranking
	batch_idcgs = torch.unsqueeze(batch_idcgs, 1)

	batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
	batch_n_gains = batch_gains / batch_idcgs  # normalised gains
	batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)
	batch_std_ranks = torch.arange(batch_stds_sorted_via_preds.size(1), dtype=torch.float).to(device) if gpu else torch.arange(batch_stds_sorted_via_preds.size(1), dtype=torch.float)
	batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients
	batch_dists = torch.unsqueeze(batch_dists, dim=0)
	batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
	batch_ndg_diffs_abs = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping
	batch_delta_ndcg = tor_batch_triu(batch_ndg_diffs_abs, k=1)

	#weighting with delta-nDCG
	batch_loss = torch.sum((batch_1st_part + batch_2nd_part) * batch_delta_ndcg)

	return batch_loss



class LambdaRank(AbstractNeuralRanker):
	'''
	Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
	Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
	'''

	def __init__(self, ranking_function=None):
		super(LambdaRank, self).__init__(id='LambdaRank', ranking_function=ranking_function)

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = lambdaRank_loss_function(batch_preds, batch_stds)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss