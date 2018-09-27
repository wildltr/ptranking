#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 27/09/2018 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn as nn

from org.archive.ranker.ranker import AbstractNeuralRanker

mse = nn.MSELoss()

def rankMSE_loss_function(batch_preds=None, batch_stds=None, TL_AF=None):
	'''
	Ranking loss based on mean square error
	:param batch_preds:
	:param batch_stds:
	:return:
	'''
	if 'S' == TL_AF or 'ST' == TL_AF:  # map to the same relevance level
		tor_max_rele_level = torch.max(batch_stds)
		batch_preds = batch_preds * tor_max_rele_level

	batch_loss = mse(batch_preds, batch_stds)
	return batch_loss

class RankMSE(AbstractNeuralRanker):
	def __init__(self, f_para_dict):
		super(RankMSE, self).__init__(f_para_dict)
		self.TL_AF = f_para_dict['TL_AF']

	def inner_train(self, batch_preds, batch_stds):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = rankMSE_loss_function(batch_preds, batch_stds, TL_AF=self.TL_AF)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss