#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
{
author = {Taylor, Michael and Guiver, John and Robertson, Stephen and Minka, Tom},
title = {SoftRank: Optimizing Non-Smooth Rank Metrics},
year = {2008},
}
"""

from itertools import product

import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.metric_utils import torch_dcg_at_k
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter

class SoftRank(AdhocNeuralRanker):
	'''
	author = {Taylor, Michael and Guiver, John and Robertson, Stephen and Minka, Tom},
	title = {SoftRank: Optimizing Non-Smooth Rank Metrics},
	'''
	def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
		super(SoftRank, self).__init__(id='SoftRank', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
		delta = model_para_dict['delta']
		self.delta = torch.tensor([delta], device=self.device)
		self.top_k = model_para_dict['top_k']
		self.metric = model_para_dict['metric']

	def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
		'''
		@param batch_preds: [batch, ranking_size] each row represents the mean predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
		@param kwargs:
		@return:
		'''
		assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking
		assert 'nDCG' == self.metric  # TODO support more metrics
		assert LABEL_TYPE.MultiLabel == kwargs['label_type']  # other types are not considered yet
		label_type = kwargs['label_type']

		batch_mus = batch_preds

		''' expected ranks '''
		# f_ij, i.e., mean difference
		batch_pairsub_mus = torch.unsqueeze(batch_mus, dim=2) - torch.unsqueeze(batch_mus, dim=1)
		# variance w.r.t. s_i - s_j, which is equal to sigma^2_i + sigma^2_j
		pairsub_vars = 2 * self.delta**2
		# \Phi(0)$
		batch_Phi0 = 0.5 * torch.erfc(batch_pairsub_mus / torch.sqrt(2 * pairsub_vars))
		# remove diagonal entries
		batch_Phi0_subdiag = torch.triu(batch_Phi0, diagonal=1) + torch.tril(batch_Phi0, diagonal=-1)
		batch_expt_ranks = torch.sum(batch_Phi0_subdiag, dim=2) + 1.0

		batch_gains = torch.pow(2.0, batch_std_labels) - 1.0
		batch_dists = 1.0 / torch.log2(batch_expt_ranks + 1.0)  # discount co-efficients
		batch_idcgs = torch_dcg_at_k(batch_rankings=batch_std_labels, label_type=label_type, device=self.device)

		#TODO check the effect of removing batch_idcgs
		if self.top_k is None:
			batch_dcgs = batch_dists * batch_gains
			batch_expt_nDCG = torch.sum(batch_dcgs/batch_idcgs, dim=1)
			batch_loss = - torch.sum(batch_expt_nDCG)
		else:
			k = min(self.top_k, batch_std_labels.size(1))
			batch_dcgs = batch_dists[:, 0:k] * batch_gains[:, 0:k]
			batch_expt_nDCG_k = torch.sum(batch_dcgs/batch_idcgs, dim=1)
			batch_loss = - torch.sum(batch_expt_nDCG_k)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss


###### Parameter of SoftRank ######

class SoftRankParameter(ModelParameter):
	''' Parameter class for SoftRank '''
	def __init__(self, debug=False, para_json=None):
		super(SoftRankParameter, self).__init__(model_id='SoftRank', para_json=para_json)
		self.debug = debug

	def default_para_dict(self):
		"""
		Default parameter setting for SoftRank
		:return:
		"""
		self.soft_para_dict = dict(model_id=self.model_id, delta=2.0, metric='nDCG', top_k=None)
		return self.soft_para_dict

	def to_para_string(self, log=False, given_para_dict=None):
		"""
		String identifier of parameters
		:param log:
		:param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
		:return:
		"""
		# using specified para-dict or inner para-dict
		soft_para_dict = given_para_dict if given_para_dict is not None else self.soft_para_dict

		s1, s2 = (':', '\n') if log else ('_', '_')

		metric, delta, top_k = soft_para_dict['metric'], soft_para_dict['delta'], soft_para_dict['top_k']
		if top_k is not None:
			softrank_para_str = s1.join([metric, str(top_k), 'Delta', '{:,g}'.format(delta)])
		else:
			softrank_para_str = s1.join([metric, 'Delta', '{:,g}'.format(delta)])

		return softrank_para_str

	def grid_search(self):
		"""
		Iterator of parameter settings for SoftRank
		"""
		if self.use_json:
			choice_topk = self.json_dict['top_k']
			choice_delta = self.json_dict['delta']
			choice_metric = self.json_dict['metric']
		else:
			choice_delta = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0
			choice_metric = ['nDCG']  # 'nDCG'
			choice_topk = [None] if self.debug else [None]

		for delta, top_k, metric in product(choice_delta, choice_topk, choice_metric):
			self.soft_para_dict = dict(model_id=self.model_id, delta=delta, top_k=top_k, metric=metric)
			yield self.soft_para_dict
