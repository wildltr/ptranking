#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 27/09/2018 | https://y-research.github.io

"""Description

"""
import numpy as np
import torch

from org.archive.l2r_global import L2R_GLOBAL
from org.archive.ranking.run.l2r import point_run, grid_run

""" GPU acceleration if expected """
L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device = False, 'cpu'

""" Reproducible experiments """
np.random.seed(seed=L2R_GLOBAL.l2r_seed)
torch.manual_seed(seed=L2R_GLOBAL.l2r_seed)



if __name__ == '__main__':
	"""
    >>> Supported ranking models <<<
    Pointwise:      RankMSE
    Pairwise:       RankNet | LambdaRank
    Listwise:       ListNet | ListMLE | RankCosine | ApproxNDCG | LambdaMART | WassRank

    >>> Supported datasets <<<
    MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi | MSLRWEB10K | MSLRWEB30K | Yahoo_L2R_Set_1 (TBA) | Yahoo_L2R_Set_1 (TBA)
    """

	dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'

	data = 'MQ2007_super'
	dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'

	grid_search = False

	if grid_search:
		to_run_models = ['RankNet_PairWeighting']
		# to_run_models = ['ListNet', 'ListMLE', 'ApproxNDCG']

		for model in to_run_models:
			grid_run(data=data, model=model, dir_data=dir_data, dir_output=dir_output)
	else:
		point_run(data=data, model='ApproxNDCG', dir_data=dir_data, dir_output=dir_output)