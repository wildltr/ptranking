#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
"""

import torch
import torch.nn.functional as F

from ptranking.base.adhoc_ranker import AdhocNeuralRanker

class ListNet(AdhocNeuralRanker):
	'''
	Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
	Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
	'''
	def __init__(self, sf_para_dict=None, gpu=False, device=None):
		super(ListNet, self).__init__(id='ListNet', sf_para_dict=sf_para_dict, gpu=gpu, device=device)

	def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
		'''
		The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
		@param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
		@param kwargs:
		@return:
		'''
		'''
		#- deprecated way -#
		batch_top1_pros_pred = F.softmax(batch_preds, dim=1)
		batch_top1_pros_std = F.softmax(batch_stds, dim=1)
		batch_loss = torch.sum(-torch.sum(batch_top1_pros_std * torch.log(batch_top1_pros_pred), dim=1))
		'''

		# todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
		# c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
		batch_loss = torch.sum(-torch.sum(F.softmax(batch_std_labels, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss
