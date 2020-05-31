#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/01 | https://y-research.github.io

"""Description

"""


import getpass

from org.archive.data import data_utils
from org.archive.ltr_adversarial_learning.eval.ad_l2r import AdL2REvaluator
from org.archive.ltr_adhoc.eval.test_l2r_tao import get_output_dir, get_dir_data

""" Begin of Personal Setting """

use_gpu_if_available = False
expected_device_id = 0

is_mbox_f1 = True if getpass.getuser()=='haitao' else False
is_mbox_f3 = True if getpass.getuser()=='dl-box' else False
if is_mbox_f1 or is_mbox_f3:
    pc = 'mbox-f1' if is_mbox_f1 else 'mbox-f3'
else:
    pc = 'imac'

def get_output_dir(pc):
    if 'mbox-f3' == pc:
        root_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Ad_l2r/'
    elif 'mbox-f1' == pc:
        root_output = '/home/haitao/Workbench/project_output/output_L2R/Ad_l2r/'
    elif 'imac' == pc:
        root_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Ad_l2r/'
    else:
        raise NotImplementedError
    return root_output

def get_in_out_dir(data_id, pc):
    dir_output = get_output_dir(pc)
    dir_data   = get_dir_data(data_id, pc)
    return dir_data, dir_output


""" End of Personal Setting """


if __name__ == '__main__':
    """
    >>> model names <<<
    Pointwise:      IR_GAN_Point  |  IR_WGAN_Point  |  IR_FGAN_Point      |  IR_SWGAN_Point
    Pairwise:       IR_GAN_Pair   |  IR_WGAN_Pair   |  IR_FGAN_Pair
    Listwise:       IR_GAN_List   |  IR_WGAN_List   |  IR_FGAN_List (TBA) | IR_GMAN_List

    >>> data <<<
    MQ2007_Semi | MQ2008_Semi | MQ2007_Super | MQ2008_Super | MSLRWEB10K | MSLRWEB30K
    """

    debug = False

    grid_search = True

    #data_id = 'MSLRWEB30K'
    #data_id = 'MQ2007_Super'
    #data_id = 'MQ2008_Semi'
    data_id = 'IRGAN_Adhoc_Semi'

    dir_data, dir_output = get_in_out_dir(data_id=data_id, pc=pc)

    '''
    It denotes how to make use of semi-supervised datasets
    True:   Keep '-1' labels
    False:  Convert '-1' as zero, use it as supervised dataset
    '''

    semi_context = False
    if semi_context:
        assert not data_id in data_utils.MSLETOR_SEMI
        mask_ratio = 0.1
        mask_type  = 'rand_mask_rele'
    else:
        mask_ratio = None
        mask_type  = None

    '''
    For semi-data,
    The settings of {unknown_as_zero & binary_rele } only take effects on the training data of MSLETOR_SEMI

    For supervised data, 
    The setting of {binary_rele} takes effects on train, vali, test datasets.    
    '''

    unknown_as_zero = False #True if data_id in data_utils.MSLETOR_SEMI else False
    presort = False # since it is not necessary

    if data_id in data_utils.MSLETOR_SEMI:
        binary_rele = True # it binarizes train, vali, test, say for a consistent comparison with irgan paper
    else:
        binary_rele = False

    if binary_rele and data_id in data_utils.MSLETOR_SEMI:
        unknown_as_zero = True # required for binarization

    query_aware = False
    data_dict = dict(data_id=data_id, dir_data=dir_data, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele, presort=presort)

    evaluator = AdL2REvaluator()
    eval_dict = dict(debug=debug, grid_search=grid_search, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type, query_aware=query_aware, dir_output=dir_output)

    if grid_search:
        to_run_models = ['IR_FGAN_Point']

        # todo for IR_GAN_Point, 'S' seems to be more stable

        for model_id in to_run_models:
            evaluator.grid_run(model_id=model_id, data_dict=data_dict, eval_dict=eval_dict)

    else:
        data_dict['sample_rankings_per_q'] = 1  # the number of sample rankings per query for training that are generaed by shuffling ties

        #ad_point_run(debug=debug, model='IR_SWGAN_Point', query_aware=query_aware, data=data, dir_data=dir_data, dir_output=dir_output, binary_rele=binary_rele, unknown_as_zero=unknown_as_zero)
        to_run_models = ['IR_GAN_List']  #
        for model_id in to_run_models:
            evaluator.point_run(model_id=model_id, data_dict=data_dict, eval_dict=eval_dict)