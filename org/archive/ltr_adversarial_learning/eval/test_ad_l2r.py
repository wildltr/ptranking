#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

from org.archive.ltr_adversarial_learning.eval.ad_l2r import AdL2REvaluator






if __name__ == '__main__':
    """
                            >>> Learning-to-Rank Models <<< 
    
    (2) Adversarial Optimization
    -----------------------------------------------------------------------------------------
    | Pointwise | IR_GAN_Point                                                              |
    -----------------------------------------------------------------------------------------
    | Pairwise  | IR_GAN_Pair                                                               |
    -----------------------------------------------------------------------------------------
    | Listwise  | IR_GAN_List                                                               |
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
    | IRGAN_Adhoc_Semi                                                                      |
    -----------------------------------------------------------------------------------------

    """

    # selected dataset
    data_id = 'MQ2007_Super'

    # location of the adopted data
    #dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'
    dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'

    # output directory
    #dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
    dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'

    grid_search = False

    evaluator = AdL2REvaluator()

    if grid_search:
        to_run_models = ['ListNet', 'ListMLE', 'ApproxNDCG']
        for model in to_run_models:
            pass
        #evaluator.grid_run(data=data, model=model, dir_data=dir_data, dir_output=dir_output)
    else:
        model_id = 'IR_GAN_Point'
        evaluator.default_run(model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)