#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
ListMLE: Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm.
In Proceedings of the 25th ICML. 1192–1199.
"""

import torch

from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.sampling_utils import arg_shuffle_ties

class LogCumsumExp(torch.autograd.Function):
	'''
	The PyTorch OP corresponding to the operation: log{ |sum_k^m{ exp{pred_k} } }
	'''
	@staticmethod
	def forward(ctx, input):
		'''
		In the forward pass we receive a context object and a Tensor containing the input;
		we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
		Specifically, ctx is a context object that can be used to stash information for backward computation.
		You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
		:param ctx:
		:param input: i.e., batch_preds of [batch, ranking_size], each row represents the relevance predictions for documents within a ltr_adhoc
		:return: [batch, ranking_size], each row represents the log_cumsum_exp value
		'''

		m, _ = torch.max(input, dim=1, keepdim=True)    #a transformation aiming for higher stability when computing softmax() with exp()
		y = input - m
		y = torch.exp(y)
		y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])    #row-wise cumulative sum, from tail to head
		fd_output = torch.log(y_cumsum_t2h) + m # corresponding to the '-m' operation

		ctx.save_for_backward(input, fd_output)

		return fd_output


	@staticmethod
	def backward(ctx, grad_output):
		'''
		In the backward pass we receive the context object and
		a Tensor containing the gradient of the loss with respect to the output produced during the forward pass (i.e., forward's output).
		We can retrieve cached data from the context object, and
		must compute and return the gradient of the loss with respect to the input to the forward function.
		Namely, grad_output is the gradient of the loss w.r.t. forward's output. Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
		Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output, i.e., the gradient of the loss w.r.t. forward's input
		:param ctx:
		:param grad_output:
		:return:
		'''

		input, fd_output = ctx.saved_tensors
		#chain rule
		bk_output = grad_output * (torch.exp(input) * torch.cumsum(torch.exp(-fd_output), dim=1))

		return bk_output


apply_LogCumsumExp = LogCumsumExp.apply


class ListMLE(NeuralRanker):
	'''
	ListMLE: Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm.
	In Proceedings of the 25th ICML. 1192–1199.
	'''
	def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
		super(ListMLE, self).__init__(id='ListMLE', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
		self.samples_per_query = model_para_dict['samples_per_query']

	def inner_train(self, batch_preds, batch_stds, **kwargs):
		if self.samples_per_query > 1:  # different from repeat(), using expand(), there is not new memory allocation
			expd_batch_stds = batch_stds.expand(self.samples_per_query, -1)
			expd_batch_preds = batch_preds.expand(self.samples_per_query, -1)
		else:
			expd_batch_preds, expd_batch_stds = batch_preds, batch_stds

		# shuffle per epoch rather than using the same order for a query
		batch_shuffle_ties_inds = arg_shuffle_ties(target_batch_stds=expd_batch_stds, descending=True, gpu=self.gpu,
		                                           device=self.device)
		target_batch_preds = torch.gather(expd_batch_preds, dim=1, index=batch_shuffle_ties_inds)

		batch_logcumsumexps = apply_LogCumsumExp(target_batch_preds)
		batch_loss = torch.sum(batch_logcumsumexps - target_batch_preds)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss

###### Parameter of ListMLE ######

class ListMLEParameter(ModelParameter):
	''' Parameter class for ListMLE '''
	def __init__(self, debug=False, para_json=None):
		super(ListMLEParameter, self).__init__(model_id='ListMLE', para_json=para_json)
		self.debug = debug

	def default_para_dict(self):
		"""
        Default parameter setting for ListMLE
        """
		self.listmle_para_dict = dict(model_id=self.model_id, samples_per_query=1.0)
		return self.listmle_para_dict

	def to_para_string(self, log=False, given_para_dict=None):
		"""
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
		# using specified para-dict or inner para-dict
		listmle_para_dict = given_para_dict if given_para_dict is not None else self.listmle_para_dict

		s1, s2 = (':', '\n') if log else ('_', '_')
		listmle_para_str = s1.join(['SP', str(listmle_para_dict['samples_per_query'])])
		return listmle_para_str

	def grid_search(self):
		"""
        Iterator of parameter settings for ListMLE
        """
		if self.use_json:
			choice_samples_per_query = self.json_dict['samples_per_query']
		else:
			choice_samples_per_query = [1] if self.debug else [1, 5, 10]

		for samples_per_query in choice_samples_per_query:
			self.listmle_para_dict = dict(model_id=self.model_id, samples_per_query=samples_per_query)
			yield self.listmle_para_dict
