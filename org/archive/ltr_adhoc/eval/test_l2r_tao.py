#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""
import os
import getpass
import numpy as np

import torch

from org.archive.data import data_utils

from org.archive.l2r_global import l2r_seed
from org.archive.ltr_adhoc.eval.l2r import L2REvaluator

""" Reproducible experiments """
np.random.seed(seed=l2r_seed)
torch.manual_seed(seed=l2r_seed)

""" Begin of Personal Setting """
def load_gpu(use_gpu_if_available=True, pc=None, expected_device_id=0):   #pc: mbox-f3, mbox-f1
    if use_gpu_if_available:
        global_gpu = torch.cuda.is_available()
        global_device = gpu_device(pc=pc, expected_device_id=expected_device_id)
    else:
        global_gpu = False
        global_device = "cpu"

    return global_gpu, global_device

def gpu_device(pc=None, expected_device_id=0):
    if 'mbox-f3' == pc:
        global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    elif 'mbox-f1' == pc:
        global_device = torch.device("cuda:{}".format(expected_device_id) if torch.cuda.is_available() else "cpu")

    else:
        raise NotImplementedError

    return global_device

def get_output_dir(pc):
    if 'mbox-f3' == pc:
        root_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'

    elif 'mbox-f1' == pc:
        root_output = '/home/haitao/Workbench/project_output/output_L2R/listwise/'

    elif 'imac' == pc:
        root_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'

    else:
        raise NotImplementedError

    return root_output

def get_dir_data(data_id, pc):
    if 'mbox-f3' == pc:
        dataset_root_letor = '/home/dl-box/WorkBench/Datasets/L2R/'
        dataset_root_mslr = dataset_root_letor
        dataset_root_yahoo = '/home/dl-box/WorkBench/Datasets/L2R/'

    elif 'mbox-f1' == pc:
        dataset_root_letor = '/home/haitao/Workbench/Datasets/L2R/'
        dataset_root_mslr = dataset_root_letor
        dataset_root_yahoo = '/home/haitao/WorkBench/Datasets/L2R/'

    elif 'imac' == pc:
        dataset_root_letor = '/Users/dryuhaitao/WorkBench/Corpus/'
        dataset_root_mslr = '/Users/dryuhaitao/WorkBench/Corpus/Learning2Rank/'

    ## query-document pair graded, [0-4] ##
    if data_id == 'MSLRWEB10K':
        dir_data = dataset_root_mslr + 'MSLR-WEB10K/'
    elif data_id == 'MSLRWEB30K':
        dir_data = dataset_root_mslr + 'MSLR-WEB30K/'

    elif data_id == 'Set1':
        #dir_data =  + 'Yahoo_L2R_Set_1/'
        dir_data = os.path.join(dataset_root_yahoo, 'Yahoo_L2R_Set_1')
    elif data_id == 'Set2':
        dir_data = dataset_root_yahoo + 'Yahoo_L2R_Set_2/'

    elif data_id == '5FoldSet1':
        dir_data = dataset_root_yahoo + 'Yahoo_L2R_Set_1_5Fold/'

    elif data_id == '5FoldSet2':
        dir_data = dataset_root_yahoo + 'Yahoo_L2R_Set_2_5Fold/'

    ## query-document pair graded ##
    elif data_id == 'MQ2007_Super':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2007/'
    elif data_id == 'MQ2008_Super':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2008/'

    elif data_id == 'MQ2007_List':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2007-list/'
    elif data_id == 'MQ2008_List':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2008-list/'

    elif data_id == 'MQ2007_Semi':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2007-semi/'
    elif data_id == 'MQ2008_Semi':
        dir_data = dataset_root_letor + 'LETOR4.0/MQ2008-semi/'

    elif data_id == 'IRGAN_Adhoc_Semi':
        dir_data = dataset_root_letor + 'IRGAN_Adhoc_Semi/'

    else:
        raise NotImplementedError

    return dir_data

def get_pc():
    is_mbox_f1 = True if getpass.getuser() == 'haitao' else False
    is_mbox_f3 = True if getpass.getuser() == 'dl-box' else False
    if is_mbox_f1 or is_mbox_f3:
        pc = 'mbox-f1' if is_mbox_f1 else 'mbox-f3'
    else:
        pc = 'imac'

    return pc

def get_in_out_dir(data_id, pc):
    dir_output = get_output_dir(pc)
    dir_data   = get_dir_data(data_id, pc)
    return dir_data, dir_output

pc = get_pc()
#expected_device_id = 0
#use_gpu_if_available = True
#global_gpu, global_device = load_gpu(use_gpu_if_available=use_gpu_if_available, pc=pc, expected_device_id=expected_device_id)
#L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device = global_gpu, global_device

""" End of Personal Setting """


if __name__ == '__main__':
    """
                                    >>> Learning-to-Rank Models <<<
    -----------------------------------------------------------------------------------------
    | Pointwise | RankMSE                                                                   |
    |           | PointMDNs (TBA)                                                           |
    -----------------------------------------------------------------------------------------
    | Pairwise  | RankNet                                                                   |
    |           | RankNet_Sharp % PairMDNs (TBA)                                            |
    -----------------------------------------------------------------------------------------
    | Listwise  | LambdaRank % ListNet % ListMLE % RankCosine %  ApproxNDCG %  LambdaMART   |
    |           | WassRank%  OTRank%  ListMDNs (TBA)%  KOTRank%  TriRank%  EMDRank%         |
    |           | LambdaRank_Sharp %  ApproxNDCG_Sharp                                      |
    |           | Virtual_P % Virtual_AP % Virtual_KT % Virtual_NDCG % Virtual_NERR         |
    |           | Virtual_Lambda                                                            |
    |           | STListNet                                                            |
    -----------------------------------------------------------------------------------------   

                                    >>> Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_L2R | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
     
    """

    ### The parameters that are commonly & explicitly specified ###
    debug = True

    grid_search = False

    #data_id = 'MSLRWEB10K'
    #data_id = 'MSLRWEB30K'
    data_id = 'MQ2007_Super'
    #data_id = 'MQ2007_list'
    #data_id = 'MQ2008_Semi'

    #data_id = 'Set1'
    #data_id = 'Set2'

    query_aware = False

    # testing the effect of partially masking ground-truth labels with a specified ratio
    semi_context = False
    if semi_context:
        assert not data_id in data_utils.MSLETOR_SEMI
        mask_ratio = 0.5
        mask_type  = 'rand_mask_rele'
    else:
        mask_ratio = None
        mask_type  = None


    dir_data, dir_output = get_in_out_dir(data_id=data_id, pc=pc)
    unknown_as_zero = True if data_id in data_utils.MSLETOR_SEMI else False
    binary_rele     = True if data_id in data_utils.MSLETOR_SEMI else False

    presort = True # a default setting

    data_dict = dict(data_id=data_id, dir_data=dir_data, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele, presort=presort)

    evaluator = L2REvaluator()
    eval_dict = dict(debug=debug, grid_search=grid_search, query_aware=query_aware, dir_output=dir_output, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)

    # ['RankMSE', 'RankNet', 'LambdaRank', 'ListNet', 'ListMLE', 'RankCosine', 'ApproxNDCG', 'WassRank']

    if grid_search:
        #to_run_models = ['RankMSE', 'RankNet', 'LambdaRank', 'ListNet', 'ListMLE', 'RankCosine', 'ApproxNDCG', 'WassRank', 'MagicRank']
        #to_run_models = ['MagicRank'] # todo 'ListMLE', 'WassRank'

        to_run_models = ['STListNet'] # Virtual_P % Virtual_AP % Virtual_KT % Virtual_NDCG % Virtual_NERR, Virtual_Lambda
        for model_id in to_run_models:
            evaluator.grid_run(model_id=model_id, data_dict=data_dict, eval_dict=eval_dict)

    else:
        data_dict['sample_rankings_per_q'] = 1 # the number of sample rankings per query for training that are generaed by shuffling ties

        #to_run_models = ['RankMSE', 'RankNet', 'LambdaRank', 'ListNet', 'ListMLE', 'RankCosine', 'ApproxNDCG', 'WassRank']
        to_run_models = ['RankNet'] #
        #to_run_models = ['ListNet']
        for model_id in to_run_models:
            evaluator.default_run(model_id=model_id, data_id=data_id, dir_data=data_dict['dir_data'], dir_output=data_dict['dir_output'])
