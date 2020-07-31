#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import numpy as np

from org.archive.l2r_global import l2r_seed
from org.archive.ltr_tree.lambdamart.lambdaMART import LambdaMARTEvaluator

np.random.seed(seed=l2r_seed)


if __name__ == '__main__':

    """
    >>> Learning-to-Rank Models <<<
    
    (3) Tree-based Model (provided by LightGBM & XGBoost)
    -----------------------------------------------------------------------------------------
    | LambdaMART(L)  % LambdaMART(X)                                                        |
    -----------------------------------------------------------------------------------------
    
    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_L2R | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_L2R | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------
    
    """

    ''' selected dataset & location of the adopted data '''
    #data_id = 'MQ2008_Super'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'

    data_id  = 'Istella_S'
    dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'

    ''' output directory '''
    #dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
    dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'

    '''
    with grid_search(), we can (1) test different models in one run; (2) test the hyper-parameters of a specific model in one run
    '''

    evaluator = LambdaMARTEvaluator()

    evaluator.default_run(data_id=data_id, dir_data=dir_data, dir_output=dir_output)