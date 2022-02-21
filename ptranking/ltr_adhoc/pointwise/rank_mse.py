#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Viewing the prediction of relevance as a conventional regression problem.
"""

import torch
import torch.nn.functional as F

from ptranking.base.adhoc_ranker import AdhocNeuralRanker

def rankMSE_loss_function(relevance_preds=None, std_labels=None):
	'''
	Ranking loss based on mean square error TODO adjust output scale w.r.t. output layer activation function
	@param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
	@param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
	@return:
	'''
	_batch_loss = F.mse_loss(relevance_preds, std_labels, reduction='none')
	batch_loss = torch.mean(torch.sum(_batch_loss, dim=1))
	return batch_loss

class RankMSE(AdhocNeuralRanker):
	def __init__(self, sf_para_dict=None, gpu=False, device=None):
		super(RankMSE, self).__init__(id='RankMSE', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
		#self.TL_AF = self.get_tl_af()

	def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
		:param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
		:return:
		'''
		batch_loss = rankMSE_loss_function(batch_preds, batch_std_labels)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss
