#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

from ptranking.base.ranker import NeuralRanker

class AdversarialPlayer(NeuralRanker):
	'''
	An adversarial player, which is used as a component of AdversarialMachine
	'''

	def __init__(self,  id=None, sf_para_dict=None, opt='Adam', lr = 1e-3, weight_decay=1e-3, gpu=False, device=None):
		super(AdversarialPlayer, self).__init__(id=id, sf_para_dict=sf_para_dict, opt=opt, lr=lr,
		                                        weight_decay=weight_decay, gpu=gpu, device=device)
