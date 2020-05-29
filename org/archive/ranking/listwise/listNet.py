#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn.functional as F

from org.archive.ranker.ranker import AbstractNeuralRanker

class ListNet(AbstractNeuralRanker):
	'''
	Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
	Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
	'''

	def __init__(self, ranking_function=None):
		super(ListNet, self).__init__(id='ListNet', ranking_function=ranking_function)

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		#The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy
		batch_top1_pros_pred = F.softmax(batch_preds, dim=1)
		batch_top1_pros_std = F.softmax(batch_stds, dim=1)
		batch_loss = torch.sum(-torch.sum(batch_top1_pros_std * torch.log(batch_top1_pros_pred), dim=1))

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss