#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/21 | https://y-research.github.io

"""Description
The following implementation builds upon the library of XGBoost: https://github.com/dmlc/xgboost
"""

import os
import sys
import pickle
import datetime
import numpy as np
from itertools import product

import torch

from org.archive.eval.metric import tor_nDCG_at_ks
from org.archive.data.data_ms import load_data_xgboost
from org.archive.ranking.run.l2r import to_output_str

import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file


def load_group_data(file_group):
    group = []
    with open(file_group, "r") as f:
        data = f.readlines()
        for line in data:
            group.append(int(line.split("\n")[0]))
    return group


def update_output_setting(para_dict=None):
    dataset, model, do_validation, root_output = para_dict['dataset'], para_dict['model'], para_dict['do_validation'], para_dict['dir_output']
    grid_search, min_docs, min_rele = para_dict['grid_search'], para_dict['min_docs'], para_dict['min_rele']
    eta, gamma, min_child_weight, max_depth, tree_method = para_dict['eta'], para_dict['gamma'], para_dict['min_child_weight'], \
                                                           para_dict['max_depth'], para_dict['tree_method']
    lm_para_str = '_'.join(['{:,g}'.format(eta), '{:,g}'.format(gamma), '{:,g}'.format(min_child_weight), '{:,g}'.format(max_depth), tree_method])

    print(' '.join(['Start {} on {} for ranking >>>'.format(model, dataset)]))

    if grid_search:
        root_output = root_output + '_'.join(['grid', model]) + '/'
        if not os.path.exists(root_output):
            os.makedirs(root_output)

    para_setting_str = '_'.join(['Vd', str(do_validation), 'Md', str(min_docs), 'Mr', str(min_rele), lm_para_str])
    file_prefix = '_'.join([model, dataset, para_setting_str])

    model_output = root_output + file_prefix + '/'  # model-specific outputs

    if not os.path.exists(model_output):
        os.makedirs(model_output)
    return model_output


def cal_nDCG_at_ks(all_std_labels=None, all_preds=None, group=None, ks=[1, 3, 5, 10]):
    #print(type(all_std_labels))

    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)

    tor_all_std_labels, tor_all_preds = torch.from_numpy(all_std_labels.astype(np.float32)), torch.from_numpy(all_preds.astype(np.float32))
    #tor_all_std_labels, tor_all_preds = tor_all_std_labels.double(), tor_all_preds.double()
    #print(tor_all_std_labels)
    #print(tor_all_preds)
    head = 0
    for gr in group:
        tor_per_query_std_labels = tor_all_std_labels[head:head+gr]
        tor_per_query_preds = tor_all_preds[head:head+gr]
        head += gr

        _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

        sys_sorted_labels = tor_per_query_std_labels[tor_sorted_inds]
        ideal_sorted_labels, _ = torch.sort(tor_per_query_std_labels, descending=True)
        #print(ideal_sorted_labels)

        ndcg_at_ks = tor_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=True)
        #print(ndcg_at_ks)

        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks)
        cnt += 1

    tor_avg_ndcg_at_ks = sum_ndcg_at_ks / cnt
    avg_ndcg_at_ks = tor_avg_ndcg_at_ks.data.numpy()
    return avg_ndcg_at_ks


def cv_eval_lambdaMART_in_XGBoost(para_dict=None):
    # common parameters across different models
    debug, dataset, dir_data, model = para_dict['debug'], para_dict['dataset'], para_dict['dir_data'], para_dict['model']
    min_docs, min_rele, cutoffs = para_dict['min_docs'], para_dict['min_rele'], para_dict['cutoffs']
    do_validation, validation_k, do_log = para_dict['do_validation'], para_dict['validation_k'], para_dict['do_log']
    eta, gamma, min_child_weight, max_depth, tree_method = para_dict['eta'], para_dict['gamma'], para_dict['min_child_weight'], para_dict['max_depth'], para_dict['tree_method']

    if debug:
        fold_num = 2
    else:
        fold_num = 5

    model_output = update_output_setting(para_dict=para_dict)
    if do_log: # open log file
        sys.stdout = open(model_output + 'log.txt', "w")

    time_begin = datetime.datetime.now()        # timing
    l2r_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
    for fold_k in range(1, fold_num + 1):
        print('\nFold-', fold_k)  # fold-wise data preparation plus certain light filtering

        dir_fold_k = dir_data + 'Fold' + str(fold_k) + '/'
        ori_file_train, ori_file_vali, ori_file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'

        file_train_data, file_train_group = load_data_xgboost(ori_file_train, min_docs=min_docs, min_rele=min_rele, dataset=dataset)
        file_vali_data, file_vali_group = load_data_xgboost(ori_file_vali, min_docs=min_docs, min_rele=min_rele, dataset=dataset)
        file_test_data, file_test_group = load_data_xgboost(ori_file_test, min_docs=min_docs, min_rele=min_rele, dataset=dataset)

        x_train, y_train = load_svmlight_file(file_train_data)
        group_train = load_group_data(file_train_group)
        train_dmatrix = DMatrix(x_train, y_train)
        train_dmatrix.set_group(group_train)

        if do_validation:
            x_valid, y_valid = load_svmlight_file(file_vali_data)
            group_valid = load_group_data(file_vali_group)
            valid_dmatrix = DMatrix(x_valid, y_valid)
            valid_dmatrix.set_group(group_valid)

        x_test, y_test = load_svmlight_file(file_test_data)
        group_test = load_group_data(file_test_group)
        test_dmatrix = DMatrix(x_test)

        """ possible settings of params """
        # params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 6}

        # ndcg
        # params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 6}
        #params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric': 'ndcg@10'}

        params = {'objective': 'rank:ndcg', 'eta': eta, 'gamma': gamma, 'min_child_weight': min_child_weight, 'max_depth': max_depth, 'eval_metric': 'ndcg@10-', 'tree_method': tree_method}    # if idealDCG=0, then 0

        # map
        # params = {'objective': 'rank:map', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 6}

        if do_validation:
            fold_xgb_model = xgb.train(params, train_dmatrix, num_boost_round=500, evals=[(valid_dmatrix, 'validation')])
        else:
            fold_xgb_model = xgb.train(params, train_dmatrix, num_boost_round=500)

        fold_checkpoint = '-'.join(['Fold', str(fold_k)])   # buffer model
        save_dir = model_output + fold_checkpoint + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir+'_'.join(['fold', str(fold_k), 'model'])+'.dat', 'wb') as model_file:
            pickle.dump(fold_xgb_model, model_file)

        pred = fold_xgb_model.predict(test_dmatrix) # fold-wise performance
        fold_avg_ndcg_at_ks = cal_nDCG_at_ks(all_std_labels=y_test, all_preds=pred, group=group_test, ks=cutoffs)
        performance_list = [model + ' Fold-' + str(fold_k)]
        for i, co in enumerate(cutoffs):
            performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_avg_ndcg_at_ks[i]))
        performance_str = '\t'.join(performance_list)
        print('\n\t', performance_str)

        l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_avg_ndcg_at_ks)  # sum for later cv-performance

    time_end = datetime.datetime.now()  # overall timing
    elapsed_time_str = str(time_end - time_begin)
    print('Elapsed time:\t', elapsed_time_str + "\n")

    print()  # begin to print either cv or average performance
    l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
    if do_validation:
        eval_prefix = str(fold_num)+'-fold cross validation scores:'
    else:
        eval_prefix = str(fold_num) + '-fold average scores:'

    print(model, eval_prefix, to_output_str(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))

    return l2r_cv_avg_scores


def log_max(dir_output=None, max_cv_avg_scores=None, para_dict=None, cutoffs=None, dataset=None):
    model = para_dict['model']
    with open(file=dir_output + '_'.join(['grid', model]) + '/' + dataset + '_max.txt', mode='w') as max_writer:
        eta, gamma, min_child_weight, max_depth, tree_method = para_dict['eta'], para_dict['gamma'], para_dict['min_child_weight'], para_dict['max_depth'], para_dict['tree_method']
        para_str = '\n'.join(['eta: '+'{:,g}'.format(eta), 'gamma: '+'{:,g}'.format(gamma), 'min_child_weight: '+'{:,g}'.format(min_child_weight), 'max_depth: '+'{:,g}'.format(max_depth), 'tree_method: '+tree_method])
        max_writer.write(para_str + '\n')
        max_writer.write(to_output_str(max_cv_avg_scores, cutoffs))


def grid_run_lambdaMART(data=None, dir_data=None, dir_output=None, debug=False):
    do_log = False if debug else True

    ''' settings that are rarely changed '''
    validation_k = 10
    min_docs = 10
    cutoffs = [1, 3, 5, 10, 20, 50]

    ''' setting w.r.t. data-preprocess '''

    ''' setting w.r.t. train  '''
    choice_validation = [True]  # True, False

    """ setting w.r.t. LambdaMART """
    choice_eta = [0.1]   if debug else [0.1] # learning_rate, range: [0,1], step size shrinkage used in update to prevents overfitting
    choice_gamma = [0.1] if debug else [0.0] # range: [0,∞] Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.

    choice_min_child_weight = [1.0] if debug else [100]
    # range: [0,∞] Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
    # then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node.
    # The larger min_child_weight is, the more conservative the algorithm will be.

    choice_max_depth = [6]        if debug else  [8]  # 6, 12, 20 range: [0,∞] Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
    choice_tree_method = ['auto', 'exact'] if debug else ['auto', 'exact', 'hist']  # auto, exact

    max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
    k_index = cutoffs.index(validation_k)
    max_para_dict = None

    for vd in choice_validation:
        for eta, gamma, min_child_weight, max_depth, tree_method in product(choice_eta, choice_gamma, choice_min_child_weight, choice_max_depth, choice_tree_method):
                        para_dict = dict(grid_search=True, debug=debug, dataset=data, dir_data=dir_data, dir_output=dir_output,
                                         model='LambdaMART', min_docs=min_docs, min_rele=1, cutoffs=cutoffs,
                                         do_validation=vd, validation_k=validation_k, do_log=do_log,
                                         eta=eta, gamma=gamma, min_child_weight=min_child_weight, max_depth=max_depth, tree_method=tree_method)

                        curr_cv_avg_scores = cv_eval_lambdaMART_in_XGBoost(para_dict=para_dict)
                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_para_dict = curr_cv_avg_scores, para_dict

    #record optimal setting
    log_max(dir_output=dir_output, max_cv_avg_scores=max_cv_avg_scores, para_dict=max_para_dict, cutoffs=cutoffs, dataset=data)


def point_run_lambdaMART(data=None, dir_data=None, dir_output=None, debug=False):
    do_log = False if debug else True

    para_dict = dict(debug=debug, dataset=data, dir_data=dir_data, dir_output=dir_output, model='LambdaMART',
                     min_docs=10, min_rele=1, cutoffs=[1, 3, 5, 10, 20, 50], do_validation=True, validation_k=10, do_log=do_log, grid_search=False,
                     eta=0.1, gamma=0.0, min_child_weight=100, max_depth=8, tree_method='exact')

    cv_eval_lambdaMART_in_XGBoost(para_dict=para_dict)


from org.archive.ranking.run.test_l2r_tao import get_data_dir
if __name__ == '__main__':
    """
    >>> Supported datasets <<<
    MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi | MSLRWEB10K | MSLRWEB30K | Yahoo_L2R_Set_1 (TBA) | Yahoo_L2R_Set_1 (TBA)
    """

    data = 'MSLRWEB30K'

    dir_data = get_data_dir(data, pc='mbox-f3')
    dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'

    debug = False

    grid_search = True

    if grid_search:
        grid_run_lambdaMART(data=data, dir_data=dir_data, dir_output=dir_output, debug=debug)
    else:
        point_run_lambdaMART(data=data, dir_data=dir_data, dir_output=dir_output, debug=debug)