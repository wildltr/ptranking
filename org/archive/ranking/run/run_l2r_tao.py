#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 27/09/2018 | https://y-research.github.io

"""Description

"""
import numpy as np
import torch
import getpass

from org.archive.l2r_global import L2R_GLOBAL
from org.archive.ranking.run.l2r import point_run, grid_run

""" Reproducible experiments """
np.random.seed(seed=L2R_GLOBAL.l2r_seed)
torch.manual_seed(seed=L2R_GLOBAL.l2r_seed)

""" Begin of Personal Setting """
def load_gpu(use_gpu_if_available=True, pc=None, expected_device_id=0):   #pc: mbox-f3, mbox-f1
    if use_gpu_if_available:
        global_gpu = torch.cuda.is_available()
        global_device = gpu_device(pc=pc, expected_device_id=expected_device_id)
    else:
        global_gpu = False
        global_device = "cpu"

    return global_gpu, global_device

def gpu_device(pc=None, expected_device_id=0):
    if 'mbox-f3' == pc:
        global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif 'mbox-f1' == pc:
        global_device = torch.device("cuda:{}".format(expected_device_id) if torch.cuda.is_available() else "cpu")
    else:
        raise NotImplementedError

    return global_device

def get_output_dir(pc):
	if 'mbox-f3' == pc:
		root_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
	elif 'mbox-f1' == pc:
		root_output = '/home/haitao/Workbench/project_output/output_L2R/listwise/'
	elif 'imac' == pc:
		root_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
	else:
		raise NotImplementedError
	return root_output

def get_data_dir(dataset, pc):
	if 'mbox-f3' == pc:
		dataset_root_letor = '/home/dl-box/WorkBench/Datasets/L2R/'
		dataset_root_mslr = dataset_root_letor
	elif 'mbox-f1' == pc:
		dataset_root_letor = '/home/haitao/Workbench/Datasets/L2R/'
		dataset_root_mslr = dataset_root_letor
	elif 'imac' == pc:
		dataset_root_letor = '/Users/dryuhaitao/WorkBench/Corpus/'
		dataset_root_mslr = '/Users/dryuhaitao/WorkBench/Corpus/Learning2Rank/'

	## query-document pair graded, [0-4] ##
	if dataset == 'MSLRWEB10K':
		dir_data = dataset_root_mslr + 'MSLR-WEB10K/'
	elif dataset == 'MSLRWEB30K':
		dir_data = dataset_root_mslr + 'MSLR-WEB30K/'

	## query-document pair graded ##
	elif dataset == 'MQ2007_super':
		dir_data = dataset_root_letor + 'LETOR4.0/MQ2007/'
	elif dataset == 'MQ2008_super':
		dir_data = dataset_root_letor + 'LETOR4.0/MQ2008/'

	elif dataset == 'MQ2007_list':
		dir_data = dataset_root_letor + 'LETOR4.0/MQ2007-list/'
	elif dataset == 'MQ2008_list':
		dir_data = dataset_root_letor + 'LETOR4.0/MQ2008-list/'
	else:
		raise NotImplementedError

	return dir_data

use_gpu_if_available = False
expected_device_id = 0

is_mbox_f1 = True if getpass.getuser()=='haitao' else False
is_mbox_f3 = True if getpass.getuser()=='dl-box' else False
if is_mbox_f1 or is_mbox_f3:
    pc = 'mbox-f1' if is_mbox_f1 else 'mbox-f3'
else:
    pc = 'imac'

def get_in_out_dir(data):
	dir_output = get_output_dir(pc)
	dir_data = get_data_dir(data, pc)
	return dir_data, dir_output

L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device = load_gpu(use_gpu_if_available=use_gpu_if_available, pc=pc, expected_device_id=expected_device_id)

""" End of Personal Setting """


if __name__ == '__main__':
	"""
	>>> model names <<<
	pointwise:      RankMSE
	pairwise:       RankNet | LambdaRank
	listwise:       ListNet | ListMLE | RankCosine | ApproxNDCG

	>>> data <<<
	MQ2007_super | MQ2008_super | MSLRWEB10K | MSLRWEB30K
	"""

	print('-- ranking -- Kind notice: you are using * {} *'.format(L2R_GLOBAL.global_device))

	data = 'MQ2007_super'
	dir_data, dir_output = get_in_out_dir(data=data)

	grid_search = False

	if grid_search:
		to_run_models = ['RankNet_PairWeighting']
		# to_run_models = ['ListNet', 'ListMLE', 'ApproxNDCG']

		for model in to_run_models:
			grid_run(data=data, model=model, dir_data=dir_data, dir_output=dir_output)
	else:
		point_run(data=data, model='RankNet', dir_data=dir_data, dir_output=dir_output)