#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 27/09/2018 | https://y-research.github.io

"""Description

"""

import os

import torch.utils.data as torch_data

from org.archive.data import data_ms
from org.archive.data.data_ms import Listwise_Dataset

def light_filtering(ori_Qs=None, min_docs=None, min_rele=1):
    list_Qs = []
    for Q in ori_Qs:
        if Q.feature_mat.shape[0] < min_docs:           # skip queries with documents that are fewer the pre-specified min_docs
            continue
        if (Q.std_label_vec > 0).sum() < min_rele:      # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            continue

        list_Qs.append(Q)

    return list_Qs


def get_data_loader(original_file=None, has_comment=None, query_level_scale=None,
                    min_docs=None, min_rele=None, need_pre_sampling=None, sample_times_per_q=None, shuffle = False, batch_size = 1, unknown_as_zero=False, binary_rele=False, pytorch_support=True):
    if pytorch_support:
        source_buffer = data_ms.get_qm_file_buffer(in_file=original_file, has_comment=has_comment, query_level_scale=query_level_scale, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

        tor_file_buffer = data_ms.get_tor_file_buffer(source_buffer=source_buffer, need_pre_sampling=need_pre_sampling, samples_per_q=sample_times_per_q)
        if os.path.exists(tor_file_buffer):
            target_data = Listwise_Dataset(Qs=None, source_buffer=source_buffer, need_pre_sampling=need_pre_sampling, samples_per_q=sample_times_per_q, tor_file_buffer=tor_file_buffer)
            data_loader = torch_data.DataLoader(target_data, shuffle=shuffle, batch_size=batch_size)
        else:
            original_Qs, source_buffer = data_ms.load_ms_data_qm(in_file=original_file, has_comment=has_comment, query_level_scale=query_level_scale, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)
            filtered_Qs = light_filtering(original_Qs, min_docs=min_docs, min_rele=min_rele)
            target_data = Listwise_Dataset(Qs=filtered_Qs, source_buffer=source_buffer, need_pre_sampling=need_pre_sampling, samples_per_q=sample_times_per_q)
            data_loader = torch_data.DataLoader(target_data, shuffle=shuffle, batch_size=batch_size)

        return data_loader

    else:   #application case: lambdaMART
        original_Qs, source_buffer = data_ms.load_ms_data_qm(in_file=original_file, has_comment=has_comment, query_level_scale=query_level_scale, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)
        filtered_Qs = light_filtering(original_Qs, min_docs=min_docs, min_rele=min_rele)
        target_data = Listwise_Dataset(Qs=filtered_Qs, source_buffer=source_buffer, need_pre_sampling=need_pre_sampling, samples_per_q=sample_times_per_q, pytorch_support=pytorch_support)
        data_loader = torch_data.DataLoader(target_data, shuffle=shuffle, batch_size=batch_size)

        return data_loader

