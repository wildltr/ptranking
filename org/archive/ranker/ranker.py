#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""
import os

import torch
import torch.optim as optim

from org.archive.ranker.neural_f import NeuralFunction

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

class AbstractNeuralRanker():

	def __init__(self, f_para_dict=None):
		self.f_para_dict = f_para_dict

		self.neural_f = self.ini_neural_f(self.f_para_dict)
		self.optimizer = optim.Adam(self.neural_f.parameters(), lr=1e-4, weight_decay=1e-4)  # use regularization

	def ini_neural_f(self, f_para_dict):
		torch.manual_seed(seed=L2R_GLOBAL.l2r_seed)
		neural_f = NeuralFunction(f_para_dict)
		if gpu: neural_f = neural_f.to(device)
		return neural_f

	def reset_parameters(self):
		self.neural_f = self.ini_neural_f(self.f_para_dict)
		self.optimizer = optim.Adam(self.neural_f.parameters(), lr=1e-4, weight_decay=1e-4)  # use regularization

	def train(self, batch_rankings, batch_stds):
		self.neural_f.train(mode=True)  #training mode
		batch_preds = self.inner_predict(batch_rankings)
		return self.inner_train(batch_preds, batch_stds)

	def inner_train(self, batch_preds, batch_stds):
		pass

	def predict(self, batch_rankings):
		self.neural_f.eval()  # evaluation mode
		batch_preds = self.inner_predict(batch_rankings)
		return batch_preds

	def inner_predict(self, batch_rankings):
		batch_preds = self.neural_f(batch_rankings)
		batch_preds = torch.squeeze(batch_preds, dim=2)  # -> [batch, ranking_size]
		return batch_preds

	def save_model(self, dir, name):
		if not os.path.exists(dir):
			os.makedirs(dir)
		torch.save(self.neural_f.state_dict(), dir + name)

	def load_model(self, file_model):
		self.neural_f.load_state_dict(torch.load(file_model))