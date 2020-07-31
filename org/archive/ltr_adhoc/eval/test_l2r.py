#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""
import torch

import numpy as np

from org.archive.l2r_global import l2r_seed
from org.archive.ltr_adhoc.eval.l2r import L2REvaluator

np.random.seed(seed=l2r_seed)
torch.manual_seed(seed=l2r_seed)



if __name__ == '__main__':
	"""
	
	>>> Learning-to-Rank Models <<<
    (1) Optimization based on Empirical Risk Minimization
    -----------------------------------------------------------------------------------------
    | Pointwise | RankMSE                                                                   |
    -----------------------------------------------------------------------------------------
    | Pairwise  | RankNet                                                                   |
    -----------------------------------------------------------------------------------------
    | Listwise  | LambdaRank % ListNet % ListMLE % RankCosine %  ApproxNDCG %  WassRank     |
    |           | STListNet                                                                 |
    -----------------------------------------------------------------------------------------   
    

	>>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_L2R | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_L2R | Istella_S % Istella % Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

	''' selected dataset '''
	#data_id = 'MQ2007_Super'

	''' location of the adopted data '''
	#dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'
	#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'

	#data_id = 'Istella_X'
	#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_X/'

	data_id = 'Istella'
	dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella/'

	#data_id = 'Istella_S'
	#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_S/'

	''' output directory '''
	#dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
	dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'

	'''
    with grid_search(), we can (1) test different models in one run; (2) test the hyper-parameters of a specific model in one run
    '''

	grid_search = False


	evaluator = L2REvaluator()

	if grid_search:

		to_run_models = ['ListNet', 'ListMLE']

		for model_id in to_run_models:
			evaluator.grid_run(model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)

	else:
		model_id = 'ListNet'
		evaluator.default_run(model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)
