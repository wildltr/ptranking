#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch

from org.archive.eval.eval_utils import idcg_std
from org.archive.ranker.ranker import AbstractNeuralRanker

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

tor_zero = torch.Tensor([0.0]).to(device) if gpu else torch.Tensor([0.0])

def tor_get_approximated_ranks(batch_pred_diffs=None, alpha=None):
	''' Equation-11 in the paper '''
	batch_indicators = torch.where(batch_pred_diffs < 0, 1.0 / (1.0 + torch.exp(alpha * batch_pred_diffs)), tor_zero)  # w.r.t. negative Sxy
	batch_tmps = torch.exp(torch.mul(batch_pred_diffs, -alpha))
	batch_indicators = torch.where(batch_pred_diffs > 0, torch.div(batch_tmps, batch_tmps + 1.0), batch_indicators)  # w.r.t. positive Sxy
	batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

	return batch_hat_pis

class ApproxNDCG_OP(torch.autograd.Function):
	DEFAULT_ALPHA = 50

	@staticmethod
	def forward(ctx, input, batch_std_labels):
		'''
		In the forward pass we receive a context object and a Tensor containing the input;
		we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
		Specifically, ctx is a context object that can be used to stash information for backward computation.
		You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
		:param ctx:
		:param input: [batch, ranking_size], each row represents the relevance predictions for documents within a ranking
		:return: [batch, ranking_size], each row value represents the approximated nDCG metric value
		'''
		alpha = ApproxNDCG_OP.DEFAULT_ALPHA

		batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)        #computing pairwise differences, i.e., Sij or Sxy

		## batch_pred_diffs_minus_alphaed_exped = torch.exp(torch.mul(batch_pred_diffs, -alpha))
		## batch_hat_pis = torch.sum(torch.div(batch_pred_diffs_minus_alphaed_exped, batch_pred_diffs_minus_alphaed_exped+1.0), dim=2) + 0.5 #get approximated rank positions, i.e., hat_pi(x)

		''' stable version of the above two lines '''
		batch_hat_pis = tor_get_approximated_ranks(batch_pred_diffs, alpha=alpha)

		# used for later back propagation
		bp_batch_exp_alphaed_diffs = torch.where(batch_pred_diffs<0, torch.exp(alpha*batch_pred_diffs), tor_zero) # negative values
		bp_batch_exp_alphaed_diffs = torch.where(batch_pred_diffs>0, torch.exp(-alpha*batch_pred_diffs), bp_batch_exp_alphaed_diffs) # positive values


		batch_gains = torch.pow(2.0, batch_std_labels) - 1.0
		sorted_labels, _ = torch.sort(batch_std_labels, dim=1, descending=True)                 #for optimal ranking based on standard labels
		batch_idcgs = idcg_std(sorted_labels)                                                   # ideal dcg given standard labels

		batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
		batch_ndcg = torch.div(batch_dcg, batch_idcgs)

		ctx.save_for_backward(batch_hat_pis, batch_pred_diffs, batch_idcgs, batch_gains, bp_batch_exp_alphaed_diffs)

		return batch_ndcg


	@staticmethod
	def backward(ctx, grad_output):
		'''
		In the backward pass we receive the context object and
		a Tensor containing the gradient of the loss with respect to the output produced during the forward pass (i.e., forward's output).
		We can retrieve cached data from the context object, and
		must compute and return the gradient of the loss with respect to the input to the forward function.
		Namely, grad_output is the gradient of the loss w.r.t. forward's output. Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
		Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output, i.e., the gradient of the loss w.r.t. forward's input

		i: the i-th rank position
		Si: the relevance prediction w.r.t. the document at the i-th rank position
		Sj: the relevance prediction w.r.t. the document at the j-th rank position
		Sij: the difference between Si and Sj
		:param ctx:
		:param grad_output:
		:return:
		'''
		alpha = ApproxNDCG_OP.DEFAULT_ALPHA
		batch_hat_pis, batch_pred_diffs, batch_idcgs, batch_gains, bp_batch_exp_alphaed_diffs = ctx.saved_tensors

		# the coefficient, which includes ln2, alpha, gain value, (1+hat_pi), pow((log_2_{1+hat_pi} ), 2)
		log_base = torch.tensor([2.0]).to(device) if gpu else torch.tensor([2.0])
		batch_coeff = (alpha/torch.log(log_base))*(batch_gains/((batch_hat_pis + 1.0) * torch.pow(torch.log2(batch_hat_pis + 1.0), 2.0)))  #coefficient part
		#here there is no difference between 'minus-alpha' and 'alpha'
		batch_gradient_Sijs = torch.div(bp_batch_exp_alphaed_diffs, torch.pow((1.0 + bp_batch_exp_alphaed_diffs), 2.0)) # gradients w.r.t. Sij, i.e., main part of delta(hat_pi(d_i))/delta(s_i)
		batch_weighted_sum_gts_i2js = batch_coeff * torch.sum(batch_gradient_Sijs, dim=2)   #sum_{i}_{delta(hat_pi(d_i))/delta(s_j)}
		batch_weighted_sum_gts_js2i = torch.squeeze(torch.bmm(torch.unsqueeze(batch_coeff, dim=1), batch_gradient_Sijs), dim=1) #sum_{j}_{delta(hat_pi(d_j))/delta(s_i)}
		batch_gradient2Sis = torch.div((batch_weighted_sum_gts_i2js - batch_weighted_sum_gts_js2i), torch.unsqueeze(batch_idcgs, dim=1))    #normalization coefficent

		#chain rule
		grad_output.unsqueeze_(1)
		target_gradients = grad_output * batch_gradient2Sis
		target_gradients.unsqueeze_(2)

		#梯度的顺序和forward形参的顺序要对应。
		# it is a must that keeping the same number w.r.t. the input of forward function
		return target_gradients, None


apply_ApproxNDCG_OP = ApproxNDCG_OP.apply

def approxNDCG_loss_function(batch_preds=None, batch_stds=None):
	batch_approx_nDCG = apply_ApproxNDCG_OP(batch_preds, batch_stds)
	batch_loss = -torch.mean(batch_approx_nDCG)
	return batch_loss

class AppoxNDCG(AbstractNeuralRanker):
	'''
	Tao Qin, Tie-Yan Liu, and Hang Li. 2010.
	A general approximation framework for direct optimization of information retrieval measures.
	Journal of Information Retrieval 13, 4 (2010), 375–397.
	'''

	def __init__(self, f_para_dict):
		super(AppoxNDCG, self).__init__(f_para_dict)

	def inner_train(self, batch_preds, batch_stds):
		'''
		:param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ranking
		:param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ranking
		:return:
		'''
		batch_loss = approxNDCG_loss_function(batch_preds, batch_stds)

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss