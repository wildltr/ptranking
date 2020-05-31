#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn.functional as F

from org.archive.base.ranker import NeuralRanker

class ListNet(NeuralRanker):
	'''
	Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
	Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
	'''

	def __init__(self, sf_para_dict=None, sampling=False):
		super(ListNet, self).__init__(id='ListNet', sf_para_dict=sf_para_dict)

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.

		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
		:return:
		'''

		'''
		#- deprecated way -#
		batch_top1_pros_pred = F.softmax(batch_preds, dim=1)
		batch_top1_pros_std = F.softmax(batch_stds, dim=1)
		batch_loss = torch.sum(-torch.sum(batch_top1_pros_std * torch.log(batch_top1_pros_pred), dim=1))
		'''

		# todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
		# c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
		batch_loss = torch.sum(-torch.sum(F.softmax(batch_stds, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss