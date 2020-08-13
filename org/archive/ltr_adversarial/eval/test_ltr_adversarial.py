#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch

import numpy as np

from org.archive.ltr_global import ltr_seed
from org.archive.ltr_adversarial.eval.ltr_adversarial import AdLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)


if __name__ == '__main__':

    """
    >>> Learning-to-Rank Models <<< 
    
    (2) Adversarial Optimization
    -----------------------------------------------------------------------------------------
    | Pointwise | IRGAN_Point                                                              |
    -----------------------------------------------------------------------------------------
    | Pairwise  | IRGAN_Pair                                                               |
    -----------------------------------------------------------------------------------------
    | Listwise  | IRGAN_List                                                               |
    -----------------------------------------------------------------------------------------
    

    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------
    | IRGAN_MQ2008_Semi                                                                      |
    -----------------------------------------------------------------------------------------

    """

    ''' selected dataset '''
    data_id = 'MQ2008_Super'

    ''' location of the adopted data '''
    #dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'
    dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
    #dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'

    ''' output directory '''
    #dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
    dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
    #dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'

    debug = True  # with a debug mode, we can make a quick test, e.g., check whether the model can operate or not

    grid_search = False # with grid_search, we can explore the effects of different hyper-parameters of a model

    evaluator = AdLTREvaluator()

    to_run_models = ['IRGAN_Pair']

    for model_id in to_run_models:
        evaluator.run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output, grid_search=grid_search)
