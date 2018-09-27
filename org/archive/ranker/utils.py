#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hai-Tao Yu on 23/08/2018

"""Description

"""

import random
import numpy as np

from operator import itemgetter

import torch


def pl_sampling(point_pros=None, point_exps=None, top_k=None):
	'''
	Sampling a ranking based on the Plackett Luce model
	:param point_pros: the entry value is: exp(f(x)) / sum_{i}{exp(f(xi))}
	:param point_exps: the entry value is: exp(f(x))
	:param top_k: approximating the ranking probability by just considering the selection of the top-k documents
	:return: the indices corresponding to the sequentially selected documents,
	'''
	ranking_size = len(point_pros)
	indices = np.random.choice(a=ranking_size, size=ranking_size, replace=False, p=point_pros)

	if top_k is not None:
		top_k = min(ranking_size, top_k)
	else:
		top_k = ranking_size

	if ranking_size == top_k:
		top_k = ranking_size-1

	pl_pro = 1.0
	sum_exp_vec = sum(point_exps)

	with np.errstate(divide='raise'):
		for i in range(top_k):
			try:
				p = point_exps[indices[i]]/sum_exp_vec
			except Exception as e:  #RuntimeWarning: divide by zero
				print(point_pros)
				print(point_exps)
				break

			pl_pro *= p
			sum_exp_vec -= point_exps[indices[i]]

	return indices, pl_pro


def shuffle_order_sampling(std_label_vec=None, reverse=True):
	'''
	generating sample rankings based on standard relevance labels.
	To deal with the existence of ties of labels, shuffle-operation is used.
	:param std_label_vec:
	:return:
	'''
	list_labels = []
	for i in range(len(std_label_vec)):
		list_labels.append((i, std_label_vec[i]))

	random.shuffle(list_labels)
	sorted_labels = sorted(list_labels, key=itemgetter(1), reverse=reverse)
	indices = [e[0] for e in sorted_labels]
	indices = np.asarray(indices)
	return indices


def torch_shuffle_order_sampling(tor_std_label_vec=None, reverse=True, gpu_open=False):
	'''
	generating sample rankings based on standard relevance labels.
	To deal with the existence of ties of labels, shuffle-operation is used.
	:param std_label_vec:
	:return:
	'''
	if gpu_open:
		std_label_vec = tor_std_label_vec.cpu().numpy()
	else:
		std_label_vec = tor_std_label_vec.data.numpy()

	list_labels = []
	for i in range(len(std_label_vec)):
		list_labels.append((i, std_label_vec[i]))

	random.shuffle(list_labels)
	sorted_labels = sorted(list_labels, key=itemgetter(1), reverse=reverse)
	indices = [e[0] for e in sorted_labels]
	indices = np.asarray(indices, dtype=np.int)
	return torch.from_numpy(indices)