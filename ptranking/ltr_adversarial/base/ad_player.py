#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ptranking.base.adhoc_ranker import AdhocNeuralRanker

class AdversarialPlayer(AdhocNeuralRanker):
	'''
	An adversarial player, which is used as a component of AdversarialMachine
	'''
	def __init__(self,  id=None, sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
		super(AdversarialPlayer, self).__init__(id=id, sf_para_dict=sf_para_dict,
												weight_decay=weight_decay, gpu=gpu, device=device)
