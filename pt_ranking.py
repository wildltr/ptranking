#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""


from org.archive.utils.args.argsUtil import ArgsUtil

from org.archive.ltr_adhoc.eval.l2r import L2REvaluator
from org.archive.ltr_tree.lambdamart.lambdaMART import LambdaMARTEvaluator
from org.archive.ltr_adversarial_learning.eval.ad_l2r import AdL2REvaluator


"""
Example command line usage:

python pt_ranking.py -data MQ2007_Super -dir_data /home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/ -dir_output /home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/ -model ListMLE

"""

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
    
    (2) Adversarial Optimization
    -----------------------------------------------------------------------------------------
    | Pointwise | IR_GAN_Point                                                              |
    -----------------------------------------------------------------------------------------
    | Pairwise  | IR_GAN_Pair                                                               |
    -----------------------------------------------------------------------------------------
    | Listwise  | IR_GAN_List                                                               |
    -----------------------------------------------------------------------------------------
    
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

    print('Started PT_Ranking ...')

    args_obj = ArgsUtil(given_root='./')
    l2r_args = args_obj.get_l2r_args()

    if 'IR_GAN' in l2r_args.model:

        evaluator = AdL2REvaluator()
        evaluator.default_run(model_id=l2r_args.model, data_id=l2r_args.data_id, dir_data=l2r_args.dir_data, dir_output=l2r_args.dir_output)

    elif 'LambdaMART' in l2r_args.model:

        evaluator = LambdaMARTEvaluator(engine=l2r_args.engine)

        evaluator.default_run(l2r_args.data_id, dir_data=l2r_args.dir_data, dir_output=l2r_args.dir_output)

    elif '' == l2r_args.framework:

        evaluator = L2REvaluator()
        evaluator.default_run(model_id=l2r_args.model, data_id=l2r_args.data_id, dir_data=l2r_args.dir_data, dir_output=l2r_args.dir_output)

    else:
        raise NotImplementedError

