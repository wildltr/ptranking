#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch

from org.archive.ranker.ranker import AbstractNeuralRanker
from org.archive.ranker.transform import tor_batch_triu

def ranknet_loss_function(batch_preds=None, batch_stds=None, explicit_weighting=False):
	'''
	From RankNet to LambdaRank to LambdaMART: An Overview
	:param batch_preds: [batch, ranking_size]
	:param batch_stds: [batch, ranking_size]
	:return:
	'''
	batch_pred_diffs = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
	batch_pred_pairwise_cmps = tor_batch_triu(batch_pred_diffs, k=1) # k should be 1, thus avoids self-comparison

	tmp_batch_std_diffs = torch.unsqueeze(batch_stds, dim=2) - torch.unsqueeze(batch_stds, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
	std_ones = torch.ones(tmp_batch_std_diffs.size())
	std_minus_ones = std_ones - 2.0
	batch_std_diffs = torch.where(tmp_batch_std_diffs > 0, std_ones, tmp_batch_std_diffs)
	batch_std_diffs = torch.where(batch_std_diffs < 0, std_minus_ones, batch_std_diffs)
	batch_std_pairwise_cmps = tor_batch_triu(batch_std_diffs, k=1)  # k should be 1, thus avoids self-comparison

	batch_1st_part = (1.0 - batch_std_pairwise_cmps) * batch_pred_pairwise_cmps * 0.5   # cf. the equation in page-3
	batch_2nd_part = torch.log(torch.exp(-batch_pred_pairwise_cmps) + 1.0)    # cf. the equation in page-3

	# todo
	'''
	if explicit_weighting:
		cost_mat = cost_mat_group(batch_stds, margin_of_non_rele_2_rele=100.0, inner_group_divergence_cost=np.e, gain_base=4.0)
		tor_cost_mat = torch.from_numpy(cost_mat).type(torch.FloatTensor)
		tor_cost_mat = torch.unsqueeze(tor_cost_mat, 0)
		batch_weights = tor_batch_triu(tor_cost_mat, k=1)
		batch_cost = torch.sum((batch_1st_part + batch_2nd_part)*batch_weights)
	else:
	'''
	batch_loss = torch.sum(batch_1st_part + batch_2nd_part)

	return batch_loss


class RankNet(AbstractNeuralRanker):
	'''
	Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
	Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
	'''

	def __init__(self, ranking_function=None, explicit_weighting=False):
		super(RankNet, self).__init__(id='RankNet', ranking_function=ranking_function)
		self.explicit_weighting = explicit_weighting
		if self.explicit_weighting:
			raise NotImplementedError

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = ranknet_loss_function(batch_preds, batch_stds)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss


""" 

class ContextRankNet(AbstractContextNeuralRanker):
	'''
	Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
	Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
	'''

	def __init__(self, in_para_dict=None, cnt_para_dict=None, com_para_dict=None, explicit_weighting=False):
		super(ContextRankNet, self).__init__(in_para_dict, cnt_para_dict, com_para_dict, id='RankNet')
		self.explicit_weighting = explicit_weighting
		if self.explicit_weighting:
			raise NotImplementedError

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = ranknet_loss_function(batch_preds, batch_stds)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss

"""