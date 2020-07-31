#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import datetime
import numpy as np
from sklearn.datasets import load_svmlight_file

import torch

import lightgbm as lgb
from lightgbm import Dataset

from org.archive.data import data_utils
from org.archive.utils.bigdata.BigPickle import pickle_save
from org.archive.data.data_utils import load_letor_data_as_libsvm_data, YAHOO_L2R, ISTELLA_L2R
from org.archive.metric.adhoc_metric import torch_nDCG_at_ks, torch_nerr_at_ks, torch_ap_at_ks, torch_p_at_ks

"""
The different implementations of LambdaMART are included for a fair comparison, which are provided by the following library:

- LightGBM: https://github.com/microsoft/LightGBM
"""


class LambdaMARTEvaluator():
    """
    The class for evaluating LambdaMART building upon different Gradient Boosting Engines.
    """

    def __init__(self, engine='LightGBM'):
        self.engine = engine

    def load_group_data(self, file_group, use_np=False):
        """  """
        if use_np:
            return np.loadtxt(file_group)
        else:
            group = []
            with open(file_group, "r") as f:
                data = f.readlines()
                for line in data:
                    group.append(int(line.split("\n")[0]))
            return group

    def update_output_setting(self, para_dict=None, data_dict=None, eval_dict=None):
        data_id, model, do_validation, root_output = data_dict['data_id'], eval_dict['model'], eval_dict['do_validation'], eval_dict['dir_output']
        grid_search, min_docs, min_rele = eval_dict['grid_search'], data_dict['min_docs'], data_dict['min_rele']

        print(' '.join(['Start {} on {} >>>'.format(model, data_id)]))

        if grid_search:
            root_output = root_output + '_'.join(['grid', model]) + '/'
            if not os.path.exists(root_output):
                os.makedirs(root_output)

        para_setting_str = '_'.join(['Vd', str(do_validation), 'Md', str(min_docs), 'Mr', str(min_rele)])

        file_prefix = '_'.join([model, data_id, para_setting_str])

        if eval_dict is not None and eval_dict['semi_context']:
            mask_ratio = eval_dict['mask_ratio']
            semi_ratio_str = '_'.join(['Semi', 'Ratio', '{:,g}'.format(mask_ratio)])
            file_prefix = '_'.join([file_prefix, semi_ratio_str])

        model_output = root_output + file_prefix + '/'  # model-specific outputs

        if not os.path.exists(model_output):
            os.makedirs(model_output)
        return model_output

    def result_to_str(self, list_scores=None, list_cutoffs=None, split_str=', ', metric_str=None):
        list_str = []
        for i in range(len(list_scores)):
            list_str.append('{}@{}:{:.4f}'.format(metric_str, list_cutoffs[i], list_scores[i]))
        return split_str.join(list_str)


    def cal_metric_at_ks(self, model, all_std_labels=None, all_preds=None, group=None, ks=[1, 3, 5, 10]):

        cnt = torch.zeros(1)

        sum_ndcg_at_ks = torch.zeros(len(ks))
        sum_nerr_at_ks = torch.zeros(len(ks))
        sum_ap_at_ks = torch.zeros(len(ks))
        sum_p_at_ks = torch.zeros(len(ks))

        list_ndcg_at_ks_per_q = []
        list_err_at_ks_per_q = []
        list_ap_at_ks_per_q = []
        list_p_at_ks_per_q = []

        tor_all_std_labels, tor_all_preds = torch.from_numpy(all_std_labels.astype(np.float32)), torch.from_numpy(all_preds.astype(np.float32))
        #tor_all_std_labels, tor_all_preds = tor_all_std_labels.double(), tor_all_preds.double()
        #print(tor_all_std_labels)
        #print(tor_all_preds)

        head = 0
        if model.endswith('LightGBM'): group = group.astype(np.int).tolist()
        for gr in group:
            tor_per_query_std_labels = tor_all_std_labels[head:head+gr]
            tor_per_query_preds = tor_all_preds[head:head+gr]
            head += gr

            _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

            sys_sorted_labels = tor_per_query_std_labels[tor_sorted_inds]
            ideal_sorted_labels, _ = torch.sort(tor_per_query_std_labels, descending=True)
            #print(ideal_sorted_labels)

            ndcg_at_ks = torch_nDCG_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=True)
            list_ndcg_at_ks_per_q.append(ndcg_at_ks.numpy())

            nerr_at_ks = torch_nerr_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks, multi_level_rele=True)
            list_err_at_ks_per_q.append(nerr_at_ks.numpy())

            ap_at_ks = torch_ap_at_ks(sys_sorted_labels=sys_sorted_labels, ideal_sorted_labels=ideal_sorted_labels, ks=ks)
            list_ap_at_ks_per_q.append(ap_at_ks.numpy())

            p_at_ks = torch_p_at_ks(sys_sorted_labels=sys_sorted_labels, ks=ks)
            list_p_at_ks_per_q.append(p_at_ks.numpy())

            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks)
            sum_nerr_at_ks = torch.add(sum_nerr_at_ks, nerr_at_ks)
            sum_ap_at_ks   = torch.add(sum_ap_at_ks, ap_at_ks)
            sum_p_at_ks    = torch.add(sum_p_at_ks, p_at_ks)
            cnt += 1

        tor_avg_ndcg_at_ks = sum_ndcg_at_ks / cnt
        avg_ndcg_at_ks = tor_avg_ndcg_at_ks.data.numpy()

        tor_avg_nerr_at_ks = sum_nerr_at_ks / cnt
        avg_nerr_at_ks = tor_avg_nerr_at_ks.data.numpy()

        tor_avg_ap_at_ks = sum_ap_at_ks / cnt
        avg_ap_at_ks = tor_avg_ap_at_ks.data.numpy()

        tor_avg_p_at_ks = sum_p_at_ks / cnt
        avg_p_at_ks = tor_avg_p_at_ks.data.numpy()

        return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks, list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q

    def get_paras_LightGBM(self, para_dict=None, eval_dict=None):

        debug, grid_search = eval_dict['debug'], eval_dict['grid_search']

        if debug:
            params = {'learning_rate': 0.1}
        elif grid_search:
            raise NotImplementedError
        else:
            # default setting according to LightGBM experiments
            params = {'boosting_type': 'gbdt',  # ltr_gbdt, dart
                      'objective': 'lambdarank',
                      'metric': 'ndcg',
                      'learning_rate': 0.05,
                      'num_leaves': 400,
                      'num_trees': 1000,
                      'num_threads': 16,
                      'min_data_in_leaf': 50,
                      'min_sum_hessian_in_leaf': 200,
                      #'lambdamart_norm':False,
                      #'is_training_metric':True,
                      'verbosity': -1}

        return params


    def eval_lambdaMART_lightGBM(self, fold_k, file_train, file_vali, file_test, para_dict=None, save_dir=None, data_dict=None, eval_dict=None):
        ''' '''
        data_id, do_validation = data_dict['data_id'], eval_dict['do_validation']

        file_train_data, file_train_group = load_letor_data_as_libsvm_data(file_train, train=True, data_dict=data_dict, eval_dict=eval_dict)
        file_test_data, file_test_group   = load_letor_data_as_libsvm_data(file_test, data_dict=data_dict, eval_dict=eval_dict)

        x_train, y_train = load_svmlight_file(file_train_data)
        group_train = np.loadtxt(file_train_group)
        train_set = Dataset(data=x_train, label=y_train, group=group_train)

        if do_validation:
            file_vali_data, file_vali_group   = load_letor_data_as_libsvm_data(file_vali, data_dict=data_dict, eval_dict=eval_dict)
            x_valid, y_valid = load_svmlight_file(file_vali_data)
            group_valid = np.loadtxt(file_vali_group)
            valid_set = Dataset(data=x_valid, label=y_valid, group=group_valid)

        x_test, y_test = load_svmlight_file(file_test_data)
        group_test = np.loadtxt(file_test_group)
        #test_set = Dataset(data=x_test, label=y_test, group=group_test)

        params = self.get_paras_LightGBM(para_dict=para_dict, eval_dict=eval_dict)

        if do_validation:
            gbm = lgb.train(params=params, train_set=train_set, valid_sets=[valid_set], verbose_eval=10, early_stopping_rounds=100)
        else:
            gbm = lgb.train(params=params, train_set=train_set, verbose_eval=10, num_boost_round=100)

        if data_id in YAHOO_L2R:
            model_file = save_dir + 'model.txt'
        else:
            model_file = save_dir+'_'.join(['fold', str(fold_k), 'model'])+'.txt'

        gbm.save_model(model_file)

        y_pred = gbm.predict(x_test)  # fold-wise prediction

        return y_test, group_test, y_pred

    def cv_eval(self, para_dict=None, data_dict=None, eval_dict=None):
        ''' Evaluation based on k-fold cross validation if multiple folds exist '''

        debug, data_id, dir_data, model = eval_dict['debug'], data_dict['data_id'], data_dict['dir_data'], eval_dict['model']
        cutoffs, do_validation, do_log = eval_dict['cutoffs'], eval_dict['do_validation'], eval_dict['do_log']

        # required setting to be consistent with the dataset
        fold_num = 2 if debug else 5
        if data_id in YAHOO_L2R: fold_num = 1
        if eval_dict['plot']: fold_num = 1
        if data_dict['data_id'] == 'Istella': assert eval_dict['do_vali'] is not True # since there is no validation data

        model_output = self.update_output_setting(para_dict=para_dict, data_dict=data_dict, eval_dict=eval_dict)
        if do_log: sys.stdout = open(model_output + 'log.txt', "w")

        time_begin = datetime.datetime.now()        # timing
        l2r_cv_avg_ndcg_scores = np.zeros(len(cutoffs))  # fold average
        l2r_cv_avg_nerr_scores = np.zeros(len(cutoffs))  # fold average
        l2r_cv_avg_ap_scores = np.zeros(len(cutoffs))  # fold average
        l2r_cv_avg_p_scores = np.zeros(len(cutoffs))  # fold average

        list_all_fold_ndcg_at_ks_per_q = []
        list_all_fold_err_at_ks_per_q = []
        list_all_fold_ap_at_ks_per_q = []
        list_all_fold_p_at_ks_per_q = []

        for fold_k in range(1, fold_num + 1):
            if data_id in YAHOO_L2R:
                data_prefix = dir_data + data_id.lower() + '.'
                ori_file_train, ori_file_vali, ori_file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'
            elif data_id in ISTELLA_L2R:
                data_prefix = dir_data + data_id + '/'
                if data_id == 'Istella_X' or data_id=='Istella_S':
                    ori_file_train, ori_file_vali, ori_file_test = data_prefix + 'train.txt', data_prefix + 'vali.txt', data_prefix + 'test.txt'
                else:
                    ori_file_train, ori_file_test = data_prefix + 'train.txt', data_prefix + 'test.txt'
            else:
                print('\nFold-', fold_k)            # fold-wise data preparation plus certain light filtering
                dir_fold_k = dir_data + 'Fold' + str(fold_k) + '/'
                ori_file_train, ori_file_vali, ori_file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'

            if data_id in YAHOO_L2R:
                save_dir = model_output
            else:
                fold_checkpoint = '-'.join(['Fold', str(fold_k)])   # buffer model
                save_dir = model_output + fold_checkpoint + '/'

            if not os.path.exists(save_dir): os.makedirs(save_dir)

            y_test, group_test, y_pred = self.eval_lambdaMART_lightGBM(fold_k=fold_k, file_train=ori_file_train, file_vali=ori_file_vali,
                                                                       file_test=ori_file_test, para_dict=para_dict, save_dir=save_dir, data_dict=data_dict, eval_dict=eval_dict)

            fold_avg_ndcg_at_ks, fold_avg_nerr_at_ks, fold_avg_ap_at_ks, fold_avg_p_at_ks,\
            list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q = self.cal_metric_at_ks(model=model, all_std_labels=y_test, all_preds=y_pred, group=group_test, ks=cutoffs)

            performance_list = [model] if data_id in YAHOO_L2R else [model + ' Fold-' + str(fold_k)]

            for i, co in enumerate(cutoffs):
                performance_list.append('\nnDCG@{}:{:.4f}'.format(co, fold_avg_ndcg_at_ks[i]))
            for i, co in enumerate(cutoffs):
                performance_list.append('\nnERR@{}:{:.4f}'.format(co, fold_avg_nerr_at_ks[i]))
            for i, co in enumerate(cutoffs):
                performance_list.append('\nMAP@{}:{:.4f}'.format(co, fold_avg_ap_at_ks[i]))
            for i, co in enumerate(cutoffs):
                performance_list.append('\nP@{}:{:.4f}'.format(co, fold_avg_p_at_ks[i]))

            performance_str = '\t'.join(performance_list)
            print('\n\t', performance_str)

            l2r_cv_avg_ndcg_scores = np.add(l2r_cv_avg_ndcg_scores, fold_avg_ndcg_at_ks)  # sum for later cv-performance
            l2r_cv_avg_nerr_scores = np.add(l2r_cv_avg_nerr_scores, fold_avg_nerr_at_ks)  # sum for later cv-performance
            l2r_cv_avg_ap_scores   = np.add(l2r_cv_avg_ap_scores, fold_avg_ap_at_ks)  # sum for later cv-performance
            l2r_cv_avg_p_scores    = np.add(l2r_cv_avg_p_scores, fold_avg_p_at_ks)  # sum for later cv-performance

            list_all_fold_ndcg_at_ks_per_q.extend(list_ndcg_at_ks_per_q)
            list_all_fold_err_at_ks_per_q.extend(list_err_at_ks_per_q)
            list_all_fold_ap_at_ks_per_q.extend(list_ap_at_ks_per_q)
            list_all_fold_p_at_ks_per_q.extend(list_p_at_ks_per_q)

        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - time_begin)
        print('Elapsed time:\t', elapsed_time_str + "\n")

        print()  # begin to print either cv or average performance
        l2r_cv_avg_ndcg_scores = np.divide(l2r_cv_avg_ndcg_scores, fold_num)
        l2r_cv_avg_nerr_scores = np.divide(l2r_cv_avg_nerr_scores, fold_num)
        l2r_cv_avg_ap_scores   = np.divide(l2r_cv_avg_ap_scores, fold_num)
        l2r_cv_avg_p_scores    = np.divide(l2r_cv_avg_p_scores, fold_num)

        if do_validation:
            eval_prefix = str(fold_num)+'-fold cross validation scores:'
        else:
            eval_prefix = str(fold_num) + '-fold average scores:'

        print(model, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_ndcg_scores, list_cutoffs=cutoffs, metric_str='nDCG'))
        print(model, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_nerr_scores, list_cutoffs=cutoffs, metric_str='nERR'))
        print(model, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_ap_scores, list_cutoffs=cutoffs, metric_str='MAP'))
        print(model, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_p_scores, list_cutoffs=cutoffs, metric_str='P'))

        all_fold_ndcg_at_ks_per_q = np.vstack(list_all_fold_ndcg_at_ks_per_q)
        all_fold_err_at_ks_per_q = np.vstack(list_all_fold_err_at_ks_per_q)
        all_fold_ap_at_ks_per_q = np.vstack(list_all_fold_ap_at_ks_per_q)
        all_fold_p_at_ks_per_q = np.vstack(list_all_fold_p_at_ks_per_q)

        pickle_save(all_fold_ndcg_at_ks_per_q, file=model_output + '_'.join([data_id, model, 'all_fold_ndcg_at_ks_per_q.np']))
        pickle_save(all_fold_err_at_ks_per_q, file=model_output + '_'.join([data_id, model, 'all_fold_err_at_ks_per_q.np']))
        pickle_save(all_fold_ap_at_ks_per_q, file=model_output + '_'.join([data_id, model, 'all_fold_ap_at_ks_per_q.np']))
        pickle_save(all_fold_p_at_ks_per_q, file=model_output + '_'.join([data_id, model, 'all_fold_p_at_ks_per_q.np']))

        return l2r_cv_avg_ndcg_scores, l2r_cv_avg_nerr_scores, l2r_cv_avg_ap_scores, l2r_cv_avg_p_scores


    def log_max(self, dir_output=None, max_cv_avg_scores=None, para_dict=None, cutoffs=None, dataset=None):
        model = para_dict['model']
        with open(file=dir_output + '_'.join(['grid', model]) + '/' + dataset + '_max.txt', mode='w') as max_writer:
            eta, gamma, min_child_weight, max_depth, tree_method = para_dict['eta'], para_dict['gamma'], para_dict['min_child_weight'], para_dict['max_depth'], para_dict['tree_method']
            para_str = '\n'.join(['eta: '+'{:,g}'.format(eta), 'gamma: '+'{:,g}'.format(gamma), 'min_child_weight: '+'{:,g}'.format(min_child_weight), 'max_depth: '+'{:,g}'.format(max_depth), 'tree_method: '+tree_method])
            max_writer.write(para_str + '\n')
            max_writer.write(self.result_to_str(max_cv_avg_scores, cutoffs))

    def get_default_dicts(self, data_id, dir_data=None, dir_output=None):
        '''  '''
        debug = False

        grid_search = False

        semi_context = False # testing the effect of partially masking ground-truth labels with a specified ratio
        if semi_context:
            assert not data_id in data_utils.MSLETOR_SEMI
            mask_ratio = 0.5
            mask_type = 'rand_mask_rele'
        else:
            mask_ratio = None
            mask_type = None

        unknown_as_zero = True if data_id in data_utils.MSLETOR_SEMI else False

        binary_rele = False  # using the original values
        presort = False  # a default setting, since there is not need for tree-based models

        data_dict = dict(data_id=data_id, dir_data=dir_data, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele,
                         presort=presort, sample_rankings_per_q=1)
        scale_data, scaler_id, scaler_level = data_utils.get_default_scaler_setting(data_id=data_id)
        # more data settings that are rarely changed
        data_dict.update(dict(min_docs=10, min_rele=1, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level))
        data_meta = data_utils.get_data_meta(data_id=data_id)
        data_dict.update(data_meta)

        # more evaluation settings that are rarely changed
        do_validation = False if data_id == 'Istella' else True
        eval_dict = dict(debug=debug, grid_search=grid_search, do_validation=do_validation, dir_output=dir_output, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)

        do_log = False if eval_dict['debug'] else False
        eval_dict.update(dict(model='_'.join(['LambdaMART', self.engine]), cutoffs=[1, 3, 5, 10, 20], validation_k=10, plot=False, do_log=do_log))

        return data_dict, eval_dict

    def default_run(self, data_id=None, dir_data=None, dir_output=None):

        ''' common setting w.r.t. datasets & evaluation'''
        data_dict, eval_dict = self.get_default_dicts(data_id=data_id, dir_data=dir_data, dir_output=dir_output)

        '''
        default setting according to LightGBM experiments
        params = {'boosting_type': 'gbdt',  # ltr_gbdt, dart
                  'objective': 'lambdarank',
                  'metric': 'ndcg',
                  'learning_rate': 0.05,
                  'num_leaves': 400,
                  'num_trees': 1000,
                  'num_threads': 16,
                  'min_data_in_leaf': 50,
                  'min_sum_hessian_in_leaf': 200,
                  #'lambdamart_norm':False,
                  #'is_training_metric':True,
                  'verbosity': -1}            
        '''
        para_dict = dict()

        self.cv_eval(para_dict=para_dict, data_dict=data_dict, eval_dict=eval_dict)