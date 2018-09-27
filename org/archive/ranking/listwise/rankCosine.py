#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn as nn

from org.archive.ranker.ranker import AbstractNeuralRanker

cos = nn.CosineSimilarity(dim=1)

class RankCosine(AbstractNeuralRanker):
	'''
	Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008.
	Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.
	'''

	def __init__(self, f_para_dict):
		super(RankCosine, self).__init__(f_para_dict)

	def inner_train(self, batch_preds, batch_stds):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = torch.sum((1.0 - cos(batch_preds, batch_stds)) / 0.5)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss