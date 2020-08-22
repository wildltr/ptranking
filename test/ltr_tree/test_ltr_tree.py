#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import numpy as np

from pt_ranking.ltr_global import ltr_seed
from pt_ranking.ltr_tree import TreeLTREvaluator

np.random.seed(seed=ltr_seed)


if __name__ == '__main__':

    """
    >>> Tree-based Learning-to-Rank Models <<<
    
    (3) Tree-based Model
    -----------------------------------------------------------------------------------------
    | LightGBMLambdaMART                                                                    |
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
    
    """

    ''' selected dataset & location of the adopted data '''
    data_id = 'MQ2008_Super'

    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
    dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'

    #data_id  = 'Istella_S'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'

    ''' output directory '''
    #dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
    #dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
    dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'

    debug = True  # with a debug mode, we can make a quick test, e.g., check whether the model can operate or not

    grid_search = False  # with grid_search, we can explore the effects of different hyper-parameters of a model

    evaluator = TreeLTREvaluator()
    evaluator.run(debug=debug, model_id='LightGBMLambdaMART',
                  data_id=data_id, dir_data=dir_data, dir_output=dir_output, grid_search=grid_search)
