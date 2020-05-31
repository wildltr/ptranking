#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import numpy as np

import torch
import torch.utils.data
from torch.autograd import Function

""" The following functions builds upon the work 'https://github.com/t-vi/pytorch-tvmisc' by Thomas Viehmann (cf. https://lernapparat.de/tv) """

#todo Note: {Tor_WassLoss | Tor_WassLossSta} currently does not support the case of batch_size>1, which is to be solved.

class WassersteinLossVanilla(Function):
	def __init__(self, cost, lam=1e-3, sinkhorn_iter=20):
		'''
		:param cost: cost = matrix M = distance matrix
		:param lam: lam = lambda of type float > 0
		:param sinkhorn_iter: sinkhorn_iter > 0
		diagonal cost should be 0
		'''
		super(WassersteinLossVanilla, self).__init__()

		self.cost = cost
		self.lam = lam
		self.sinkhorn_iter = sinkhorn_iter
		self.na = cost.size(0)
		self.nb = cost.size(1)
		self.K = torch.exp(-self.cost / self.lam)
		self.KM = self.cost * self.K
		self.stored_grad = None

	def forward(self, pred, target):
		"""
		pred: Batch * K: K = # mass points
		target: Batch * L: L = # mass points
		"""
		assert pred.size(1) == self.na
		assert target.size(1) == self.nb

		nbatch = pred.size(0)
		# guessing: diag(v) K diag(u)
		u = self.cost.new(nbatch, self.na).fill_(1.0 / self.na)

		for i in range(self.sinkhorn_iter): # has checked twice, and there should be no problem
			#left-matrix-mul equals to batch right-matrix-mul (i.e., torch.bmm())
			v = target / (torch.mm(u, self.K.t()))  # double check K vs. K.t() here and next line
			u = pred / (torch.mm(v, self.K))

			# print ("stability at it",i, "u",(u!=u).sum(),u.max(),"v", (v!=v).sum(), v.max())
			if (u != u).sum() > 0 or (v != v).sum() > 0 or u.max() > 1e9 or v.max() > 1e9:  # u!=u is a test for NaN...
				# we have reached the machine precision come back to previous solution and quit loop
				#raise Exception(str(('Warning: numerical errrors', i + 1, "u", (u != u).sum(), u.max(), "v", (v != v).sum(), v.max())))
				break

		loss = (u * torch.mm(v, self.KM.t())).mean(0).sum()  # double check KM vs KM.t()...
		grad = self.lam * u.log() / nbatch  # check whether u needs to be transformed

		#what is the rational underlying the following two lines? reduce variance like IRGAN
		''' alternative way due to the error w.r.t. expand_as() '''
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		'''
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)  # does this help over only once?
		'''

		self.stored_grad = grad

		dist = self.cost.new((loss,))
		return dist

	def backward(self, grad_output):
		# print (grad_output.size(), self.stored_grad.size())
		return self.stored_grad * grad_output[0], None



class Y_WassersteinLossVanilla(Function):
	@staticmethod
	def forward(ctx, pred, target, cost, lam=1e-3, sh_num_iter=20):
		"""
		pred: Batch * K: K = # mass points
		target: Batch * L: L = # mass points
		"""
		na, nb = cost.size(0), cost.size(1)
		assert pred.size(1) == na and target.size(1) == nb
		K = torch.exp(-cost / lam)
		KM = cost * K

		nbatch = pred.size(0)
		# guessing: diag(v) K diag(u)
		u = cost.new(nbatch, na).fill_(1.0 / na)

		for i in range(sh_num_iter): # has checked twice, and there should be no problem
			#left-matrix-mul equals to batch right-matrix-mul (i.e., torch.bmm())
			v = target / (torch.mm(u, K.t()))  # double check K vs. K.t() here and next line
			u = pred / (torch.mm(v, K))

			# print ("stability at it",i, "u",(u!=u).sum(),u.max(),"v", (v!=v).sum(), v.max())
			if (u != u).sum() > 0 or (v != v).sum() > 0 or u.max() > 1e9 or v.max() > 1e9:  # u!=u is a test for NaN...
				# we have reached the machine precision come back to previous solution and quit loop
				#raise Exception(str(('Warning: numerical errrors', i + 1, "u", (u != u).sum(), u.max(), "v", (v != v).sum(), v.max())))
				break

		loss = (u * torch.mm(v, KM.t())).mean(0).sum()  # double check KM vs KM.t()...
		grad = lam * u.log() / nbatch  # check whether u needs to be transformed

		#what is the rational underlying the following two lines? reduce variance like IRGAN
		''' alternative way due to the error w.r.t. expand_as() '''
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		'''
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)  # does this help over only once?
		'''

		ctx.save_for_backward(grad)

		dist = cost.new((loss,))
		return dist

	@staticmethod
	def backward(ctx, grad_output):
		cur_grad = ctx.saved_tensors[0]
		bk_output = cur_grad * grad_output[0]
		return bk_output, None, None, None, None



class WassersteinLossStab(Function):
	def __init__(self, cost, lam=1e-3, sinkhorn_iter=20):
		super(WassersteinLossStab, self).__init__()

		# cost = matrix M = distance matrix
		# lam = lambda of type float > 0
		# sinkhorn_iter > 0
		# diagonal cost should be 0
		self.cost = cost
		self.lam = lam
		self.sinkhorn_iter = sinkhorn_iter
		self.na = cost.size(0)
		self.nb = cost.size(1)
		self.K = torch.exp(-self.cost / self.lam)
		self.KM = self.cost * self.K
		self.stored_grad = None

	def forward(self, pred, target):
		"""
		pred: Batch * K: K = # mass points
		target: Batch * L: L = # mass points
		"""
		assert pred.size(1) == self.na
		assert target.size(1) == self.nb

		batch_size = pred.size(0)

		log_a, log_b = torch.log(pred), torch.log(target)
		log_u = self.cost.new(batch_size, self.na).fill_(-np.log(self.na))
		log_v = self.cost.new(batch_size, self.nb).fill_(-np.log(self.nb))

		for i in range(self.sinkhorn_iter):
			log_u_max = torch.max(log_u, dim=1, keepdim=True)[0]
			u_stab = torch.exp(log_u - log_u_max)
			log_v = log_b - torch.log(torch.mm(self.K.t(), u_stab.t()).t()) - log_u_max
			log_v_max = torch.max(log_v, dim=1, keepdim=True)[0]
			v_stab = torch.exp(log_v - log_v_max)
			log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max
			#error prompted due to the usage of expand_as()
			'''
			log_u_max = torch.max(log_u, dim=1)[0]
			u_stab = torch.exp(log_u - log_u_max.expand_as(log_u))
			log_v = log_b - torch.log(torch.mm(self.K.t(), u_stab.t()).t()) - log_u_max.expand_as(log_v)
			log_v_max = torch.max(log_v, dim=1)[0]
			v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
			log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)
			'''

		#alternative way due to expand_as()
		log_v_max = torch.max(log_v, dim=1, keepdim=True)[0]
		v_stab = torch.exp(log_v - log_v_max)
		logcostpart1 = torch.log(torch.mm(self.KM, v_stab.t()).t()) + log_v_max
		'''
		log_v_max = torch.max(log_v, dim=1)[0]
		v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
		logcostpart1 = torch.log(torch.mm(self.KM, v_stab.t()).t()) + log_v_max.expand_as(log_u)
		'''

		wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
		grad = log_u * self.lam

		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		'''
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)  # does this help over only once?
		'''

		grad = grad / batch_size
		self.stored_grad = grad

		return self.cost.new((wnorm,))

	def backward(self, grad_output):
		# print (grad_output.size(), self.stored_grad.size())
		# print (self.stored_grad, grad_output)
		res = grad_output.new()
		res.resize_as_(self.stored_grad).copy_(self.stored_grad)
		if grad_output[0] != 1:
			res.mul_(grad_output[0])
		return res, None


class Y_WassersteinLossStab(Function):

	@staticmethod
	def forward(ctx, pred, target, cost, lam, sh_num_iter):
		"""
		pred: Batch * K: K = # mass points
		target: Batch * L: L = # mass points
		"""
		na, nb = cost.size(0), cost.size(1)
		assert pred.size(1) == na and target.size(1) == nb

		K = torch.exp(-cost / lam)
		KM = cost * K

		batch_size = pred.size(0)

		log_a, log_b = torch.log(pred), torch.log(target)
		log_u = cost.new(batch_size, na).fill_(-np.log(na))
		log_v = cost.new(batch_size, nb).fill_(-np.log(nb))

		for i in range(sh_num_iter):
			log_u_max = torch.max(log_u, dim=1, keepdim=True)[0]
			u_stab = torch.exp(log_u - log_u_max)
			log_v = log_b - torch.log(torch.mm(K.t(), u_stab.t()).t()) - log_u_max
			log_v_max = torch.max(log_v, dim=1, keepdim=True)[0]
			v_stab = torch.exp(log_v - log_v_max)
			log_u = log_a - torch.log(torch.mm(K, v_stab.t()).t()) - log_v_max
			#error prompted due to the usage of expand_as()
			'''
			log_u_max = torch.max(log_u, dim=1)[0]
			u_stab = torch.exp(log_u - log_u_max.expand_as(log_u))
			log_v = log_b - torch.log(torch.mm(self.K.t(), u_stab.t()).t()) - log_u_max.expand_as(log_v)
			log_v_max = torch.max(log_v, dim=1)[0]
			v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
			log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)
			'''

		#alternative way due to expand_as()
		log_v_max = torch.max(log_v, dim=1, keepdim=True)[0]
		v_stab = torch.exp(log_v - log_v_max)
		logcostpart1 = torch.log(torch.mm(KM, v_stab.t()).t()) + log_v_max
		'''
		log_v_max = torch.max(log_v, dim=1)[0]
		v_stab = torch.exp(log_v - log_v_max.expand_as(log_v))
		logcostpart1 = torch.log(torch.mm(self.KM, v_stab.t()).t()) + log_v_max.expand_as(log_u)
		'''

		wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
		grad = log_u * lam

		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		grad = grad - torch.mean(grad, dim=1, keepdim=True)
		'''
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)
		grad = grad - torch.mean(grad, dim=1).expand_as(grad)  # does this help over only once?
		'''

		grad = grad / batch_size
		ctx.save_for_backward(grad)

		return cost.new((wnorm,))

	@staticmethod
	def backward(ctx, grad_output):
		cur_grad = ctx.saved_tensors[0]

		res = grad_output.new()
		res.resize_as_(cur_grad).copy_(cur_grad)
		if grad_output[0] != 1:
			res.mul_(grad_output[0])

		return res, None, None, None, None


def tor_sinkhorn(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
	# seems to explode terribly fast with 32 bit floats...
	if a is None:
		a = M.new(M.size(0)).fill_(1 / M.size(0))
	if b is None:
		b = M.new(M.size(0)).fill_(1 / M.size(1))

	# init data
	Nini = a.size(0)
	Nfin = b.size(0)

	if log:
		log = {'err': []}

	# we assume that no distances are null except those of the diagonal of distances
	u = M.new(Nfin).fill_(1 / Nfin)
	v = M.new(Nfin).fill_(1 / Nfin)

	uprev = M.new(Nini).zero_()
	vprev = M.new(Nini).zero_()

	K = torch.exp(-M / reg)

	print("K", K.size())

	Kp = K / (a[:, None].expand_as(K))
	cpt = 0
	err = 1
	while (err > stopThr and cpt < numItermax):
		Kt_dot_u = torch.mv(K.t(), u)
		if (Kt_dot_u == 0).sum() > 0 or (u != u).sum() > 0 or (v != v).sum() > 0:  # u!=u is a test for NaN...
			print('Warning: numerical errrors')	# we have reached the machine precision, come back to previous solution and quit loop
			if cpt != 0:
				u = uprev
				v = vprev
			break

		uprev = u
		vprev = v

		v = b / Kt_dot_u
		u = 1. / torch.mv(Kp, v)

		if cpt % 10 == 0:# we can speed up the process by checking for the error only all the 10th iterations
			transp = (u[:, None].expand_as(K)) * K * (v[None, :].expand_as(K))
			err = torch.dist(transp.sum(0), b) ** 2
			if log:
				log['err'].append(err)

			if verbose:
				if cpt % 200 == 0: print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))

		cpt = cpt + 1

	if log:
		log['u'] = u
		log['v'] = v
	# print 'err=',err,' cpt=',cpt
	if log:
		return (u[:, None].expand_as(K)) * K * (v[None, :].expand_as(K)), log
	else:
		return (u[:, None].expand_as(K)) * K * (v[None, :].expand_as(K))


def tor_sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9, warmstart=None, verbose=False, print_period=20, log=False):
	if a is None:
		a = M.new(M.size(0)).fill_(1 / M.size(0))
	if b is None:
		b = M.new(M.size(0)).fill_(1 / M.size(1))

	# init data
	na = a.size(0)
	nb = b.size(0)

	if log: log = {'err': []}

	# we assume that no distances are null except those of the diagonal of distances
	if warmstart is None:
		alpha, beta = M.new(na).zero_(), M.new(nb).zero_()
	else:
		alpha, beta = warmstart

	u, v = M.new(na).fill_(1 / na), M.new(nb).fill_(1 / nb)
	uprev, vprev = M.new(na).zero_(), M.new(nb).zero_()

	def get_K(alpha, beta):
		"""log space computation"""
		return torch.exp(-(M - alpha[:, None].expand_as(M) - beta[None, :].expand_as(M)) / reg)

	def get_Gamma(alpha, beta, u, v):
		"""log space gamma computation"""
		return torch.exp(-(M - alpha[:, None].expand_as(M) - beta[None, :].expand_as(M)) / reg + torch.log(u)[:, None].expand_as(M) + torch.log(v)[None, :].expand_as(M))

	K = get_K(alpha, beta)
	transp = K
	loop = True
	cpt = 0
	err = 1
	while loop:

		if u.abs().max() > tau or v.abs().max() > tau:
			alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
			u, v = M.new(na).fill_(1 / na), M.new(nb).fill_(1 / nb)
			K = get_K(alpha, beta)

		uprev = u
		vprev = v

		Kt_dot_u = torch.mv(K.t(), u)
		v = b / Kt_dot_u
		u = a / torch.mv(K, v)

		if cpt % print_period == 0:
			# we can speed up the process by checking for the error only all the 10th iterations
			transp = get_Gamma(alpha, beta, u, v)
			err = torch.dist(transp.sum(0), b) ** 2

			if log: log['err'].append(err)

			if verbose:
				if cpt % (print_period * 20) == 0:
					print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))

		if err <= stopThr:
			loop = False

		if cpt >= numItermax:
			loop = False

		if (Kt_dot_u == 0).sum() > 0 or (u != u).sum() > 0 or (v != v).sum() > 0:  # u!=u is a test for NaN...
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print('Warning: numerical errrors')
			if cpt != 0:
				u = uprev
				v = vprev
			break

		cpt = cpt + 1
	# print 'err=',err,' cpt=',cpt
	if log:
		log['logu'] = alpha / reg + torch.log(u)
		log['logv'] = beta / reg + torch.log(v)
		log['alpha'] = alpha + reg * torch.log(u)
		log['beta'] = beta + reg * torch.log(v)
		log['warmstart'] = (log['alpha'], log['beta'])
		return get_Gamma(alpha, beta, u, v), log
	else:
		return get_Gamma(alpha, beta, u, v)