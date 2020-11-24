#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
A simple script for testing either in-built methods or newly added methods
"""

import torch
import numpy as np

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

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
    |           | STListNet  % LambdaLoss                                                   |
    -----------------------------------------------------------------------------------------   
    

	>>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S % Istella % Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

	cuda = None				# the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

	debug = True            # in a debug mode, we just check whether the model can operate

	config_with_json = False # specify configuration with json files or not

	models_to_run = [
		#'RankMSE',
		#'RankNet',
		#'LambdaRank',
		#'ListNet',
		#'ListMLE',
		#'RankCosine',
		#'ApproxNDCG',
		'WassRank',
		'STListNet',
		'LambdaLoss'
	]

	evaluator = LTREvaluator(cuda=cuda)

	if config_with_json: # specify configuration with json files
		# the directory of json files
		#dir_json = '/Users/dryuhaitao/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
		#dir_json = '/Users/solar/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
		dir_json = '/home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'

		for model_id in models_to_run:
			evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json)

	else: # specify configuration manually
		''' Selected dataset '''
		data_id = 'MQ2008_Super'

		''' By grid_search, we can explore the effects of different hyper-parameters of a model '''
		grid_search = False

		''' Location of the adopted data '''
		dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2008/'
		#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'
		#dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'

		#data_id = 'Istella_X'
		#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_X/'

		#data_id = 'Istella'
		#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella/'

		#data_id = 'Istella_S'
		#dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_S/'

		''' Output directory '''
		dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
		#dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
		#dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'

		for model_id in models_to_run:
			evaluator.run(debug=debug, grid_search=grid_search,
			              model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)
