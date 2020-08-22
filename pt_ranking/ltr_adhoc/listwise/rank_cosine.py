#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch
import torch.nn as nn

from pt_ranking.base.ranker import NeuralRanker

cos = nn.CosineSimilarity(dim=1)

class RankCosine(NeuralRanker):
	'''
	Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008.
	Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.
	'''

	def __init__(self, sf_para_dict=None):
		super(RankCosine, self).__init__(id='RankCosine', sf_para_dict=sf_para_dict)

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
		:return:
		'''
		batch_loss = torch.sum((1.0 - cos(batch_preds, batch_stds)) / 0.5)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss
