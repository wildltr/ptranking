#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""


from pt_ranking.utils.args.argsUtil import ArgsUtil

from pt_ranking.ltr_adhoc.eval.ltr import LTREvaluator, LTR_ADHOC_MODEL
from pt_ranking.ltr_tree.eval.ltr_tree import TreeLTREvaluator, LTR_TREE_MODEL
from pt_ranking.ltr_adversarial.eval.ltr_adversarial import AdLTREvaluator, LTR_ADVERSARIAL_MODEL


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
    |           | STListNet  % LambdaLoss                                                   |
    -----------------------------------------------------------------------------------------   
    
    (2) Adversarial Optimization
    -----------------------------------------------------------------------------------------
    | Pointwise | IRGAN_Point                                                               |
    -----------------------------------------------------------------------------------------
    | Pairwise  | IRGAN_Pair                                                                |
    -----------------------------------------------------------------------------------------
    | Listwise  | IRGAN_List                                                                |
    -----------------------------------------------------------------------------------------
    
    (3) Tree-based Model (provided by LightGBM)
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

    print('Started PT_Ranking ...')

    args_obj = ArgsUtil(given_root='./')
    l2r_args = args_obj.get_l2r_args()

    if l2r_args.model in LTR_ADHOC_MODEL:
        evaluator = LTREvaluator()

    elif l2r_args.model in LTR_ADVERSARIAL_MODEL:
        evaluator = AdLTREvaluator()

    elif l2r_args.model in LTR_TREE_MODEL:
        evaluator = TreeLTREvaluator()
    else:
        raise NotImplementedError

    evaluator.run(debug=True, model_id=l2r_args.model, data_id=l2r_args.data_id, dir_data=l2r_args.dir_data, dir_output=l2r_args.dir_output, grid_search=False)

