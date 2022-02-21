#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008.
Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.
"""

import torch
import torch.nn as nn

from ptranking.base.adhoc_ranker import AdhocNeuralRanker

cos = nn.CosineSimilarity(dim=1)

class RankCosine(AdhocNeuralRanker):
	'''
	Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008.
	Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.
	'''
	def __init__(self, sf_para_dict=None, gpu=False, device=None):
		super(RankCosine, self).__init__(id='RankCosine', sf_para_dict=sf_para_dict, gpu=gpu, device=device)

	def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
		'''
		@param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
		@param kwargs:
		@return:
		'''
		batch_loss = torch.sum((1.0 - cos(batch_preds, batch_std_labels)) / 0.5)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss
