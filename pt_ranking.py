#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import sys
from ptranking.utils.args.argsUtil import ArgsUtil

from ptranking.ltr_adhoc.eval.ltr import LTREvaluator, LTR_ADHOC_MODEL
from ptranking.ltr_tree.eval.ltr_tree import TreeLTREvaluator, LTR_TREE_MODEL
from ptranking.ltr_adversarial.eval.ltr_adversarial import AdLTREvaluator, LTR_ADVERSARIAL_MODEL


"""
The command line usage:

(1) Without using GPU
python pt_ranking.py -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/

(2) Using GPU
python pt_ranking.py -cuda 0 -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/

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

    args_obj = ArgsUtil(given_root='./')
    l2r_args = args_obj.get_l2r_args()

    if l2r_args.model in LTR_ADHOC_MODEL:
        evaluator = LTREvaluator(cuda=l2r_args.cuda)

    elif l2r_args.model in LTR_ADVERSARIAL_MODEL:
        evaluator = AdLTREvaluator(cuda=l2r_args.cuda)

    elif l2r_args.model in LTR_TREE_MODEL:
        evaluator = TreeLTREvaluator()

    else:
        args_obj.args_parser.print_help()
        sys.exit()

    print('Started evaluation with pt_ranking !')
    evaluator.run(model_id=l2r_args.model, dir_json=l2r_args.dir_json, debug=l2r_args.debug, config_with_json=True)
    print('Finished evaluation with pt_ranking !')

