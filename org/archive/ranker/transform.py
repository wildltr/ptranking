#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 21/08/2018

"""Description

"""

import numpy as np

import torch

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device

def tor_triu(mat=None, k=0):
	'''
	:param mat: [m, m]
	:param k: the offset w.r.t. the diagonal line
	:return:
	'''
	assert mat.size(0) == mat.size(1)

	m = mat.size(1)
	row_inds, col_inds = np.triu_indices(m, k=k)
	tor_row_inds = torch.LongTensor(row_inds).to(device) if gpu else torch.LongTensor(row_inds)
	tor_col_inds = torch.LongTensor(col_inds).to(device) if gpu else torch.LongTensor(col_inds)
	triu = mat[tor_row_inds, tor_col_inds]

	return triu

def tor_batch_triu(batch_mats=None, k=0):
	'''
	:param batch_mats: [batch, m, m]
	:param k: the offset w.r.t. the diagonal line For k=0 means including the diagonal line, k=1 means upper triangular part without the diagonal line
	:return:
	'''
	assert batch_mats.size(1) == batch_mats.size(2)

	m = batch_mats.size(1)
	row_inds, col_inds = np.triu_indices(m, k=k)
	tor_row_inds = torch.LongTensor(row_inds).to(device) if gpu else torch.LongTensor(row_inds)
	tor_col_inds = torch.LongTensor(col_inds).to(device) if gpu else torch.LongTensor(col_inds)
	batch_triu = batch_mats[:, tor_row_inds, tor_col_inds]

	return batch_triu


def tor_triu_indices_group_mask(batch_std_labels=None):
	size_batch = batch_std_labels.size(0)
	size_ranking = batch_std_labels.size(1)
	np_batch_std_labels = batch_std_labels.cpu().numpy() if gpu else batch_std_labels.numpy()

	list_row_inds = list()
	list_col_inds = list()
	for i in range(size_batch):
		all_row_inds, all_col_inds = np.tril_indices(size_ranking, k=0)

		zero_head = np.asarray([0])
		_, cnt_array = np.unique(np_batch_std_labels[i, :], return_counts=True)
		offset_array = np.cumsum(np.hstack((zero_head, cnt_array)))
		for j, cnt in enumerate(cnt_array):
			row_inds, col_inds = np.triu_indices(cnt, k=1)
			row_inds += offset_array[j]
			col_inds += offset_array[j]

			all_row_inds, all_col_inds = np.hstack((all_row_inds, row_inds)), np.hstack((all_col_inds, col_inds))

		list_row_inds.append(all_row_inds)
		list_col_inds.append(all_col_inds)

	return list_row_inds, list_col_inds

def tor_triu_indices_group_select(batch_std_labels=None):
	size_batch = batch_std_labels.size(0)
	np_batch_std_labels = batch_std_labels.cpu().numpy() if gpu else batch_std_labels.numpy()
	#print(np_batch_std_labels)

	list_list_submats = list()
	for i in range(size_batch):
		list_submats = list()  # number of rows, row_inds, col_inds
		_, ascending_cnt_array = np.unique(np_batch_std_labels[i, :], return_counts=True)
		descending_cnt_array = np.flip(ascending_cnt_array, axis=0) # due to the fact that the standard labels are already sorted in descending order
		#print('descending_cnt_array', descending_cnt_array)

		''' because cnt_array is in ascending order that differs from the standard labels' order'''
		flipped_offset_array = np.cumsum(ascending_cnt_array)
		offset_array = np.flip(flipped_offset_array, axis=0) # necessary, due to the fact that the standard labels are already sorted in descending order
		#print('offset_array', offset_array)

		row_begin = 0
		col_begin = 0
		size_row = 0
		for j in range(1, len(offset_array)):
			cnt = descending_cnt_array[j-1]
			size_row += cnt
			col_begin += cnt
			submat = (cnt, np.repeat(row_begin+np.arange(cnt), offset_array[j]) , np.tile(col_begin + np.arange(offset_array[j]), cnt) )    # number of rows, row_inds, col_inds
			#print('submat:\t', submat)
			row_begin += cnt
			list_submats.append(submat)

		list_list_submats.append((size_row, list_submats))

	return list_list_submats







