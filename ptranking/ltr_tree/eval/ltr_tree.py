#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np

import torch

from ptranking.utils.bigdata.BigPickle import pickle_save
from ptranking.data.data_utils import YAHOO_LTR, ISTELLA_LTR, MSLETOR, MSLRWEB
from ptranking.ltr_tree.eval.tree_parameter import TreeDataSetting, TreeEvalSetting
from ptranking.ltr_tree.lambdamart.lightgbm_lambdaMART import LightGBMLambdaMART, LightGBMLambdaMARTParameter
from ptranking.metric.adhoc_metric import torch_nDCG_at_ks, torch_nerr_at_ks, torch_ap_at_ks, torch_p_at_ks
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

LTR_TREE_MODEL = ['LightGBMLambdaMART']


class TreeLTREvaluator(LTREvaluator):
    """
    The class for evaluating different tree-based learning to rank methods.
    """
    def __init__(self, id='TreeLTR'):
        super(TreeLTREvaluator, self).__init__(id=id)

    def display_information(self, data_dict):
        """
        Display some information.
        :param data_dict:
        :return:
        """
        print(' '.join(['\nStart {} on {} >>>'.format(self.model_parameter.model_id, data_dict['data_id'])]))

    def setup_output(self, data_dict=None, eval_dict=None):
        """
        Determine the output.
        :param data_dict:
        :param eval_dict:
        :return:
        """
        dir_output, grid_search, mask_label = eval_dict['dir_output'], eval_dict['grid_search'],\
                                                 eval_dict['mask_label']

        #print(' '.join(['Start {} on {} >>>'.format(self.model_parameter.model_id, data_id)]))

        if grid_search:
            output_root = dir_output + '_'.join(['grid', self.model_parameter.get_identifier()]) + '/'
        else:
            output_root = dir_output + self.model_parameter.get_identifier() + '/'

        data_eval_str = '_'.join([self.data_setting.to_data_setting_string(), self.eval_setting.to_eval_setting_string()])

        if mask_label:
            data_eval_str = '_'.join([data_eval_str, 'MaskLabel', 'Ratio', '{:,g}'.format(eval_dict['mask_ratio'])])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                data_eval_str = '_'.join([data_eval_str, 'QS', data_dict['scaler_id']])
            else:
                data_eval_str = '_'.join([data_eval_str, 'DS', data_dict['scaler_id']])

        output_root = output_root + data_eval_str + '/' + self.model_parameter.to_para_string() + '/'  # run-specific outputs
        return output_root

    def result_to_str(self, list_scores=None, list_cutoffs=None, split_str=', ', metric_str=None):
        """
        Convert metric results to a string
        :param list_scores:
        :param list_cutoffs:
        :param split_str:
        :param metric_str:
        :return:
        """
        list_str = []
        for i in range(len(list_scores)):
            list_str.append('{}@{}:{:.4f}'.format(metric_str, list_cutoffs[i], list_scores[i]))
        return split_str.join(list_str)

    def cal_metric_at_ks(self, model_id, all_std_labels=None, all_preds=None, group=None, ks=[1, 3, 5, 10]):
        """
        Compute metric values with different cutoff values
        :param model:
        :param all_std_labels:
        :param all_preds:
        :param group:
        :param ks:
        :return:
        """
        cnt = torch.zeros(1)

        sum_ndcg_at_ks = torch.zeros(len(ks))
        sum_nerr_at_ks = torch.zeros(len(ks))
        sum_ap_at_ks = torch.zeros(len(ks))
        sum_p_at_ks = torch.zeros(len(ks))

        list_ndcg_at_ks_per_q = []
        list_err_at_ks_per_q = []
        list_ap_at_ks_per_q = []
        list_p_at_ks_per_q = []

        tor_all_std_labels, tor_all_preds = \
            torch.from_numpy(all_std_labels.astype(np.float32)), torch.from_numpy(all_preds.astype(np.float32))

        head = 0
        if model_id.startswith('LightGBM'): group = group.astype(np.int).tolist()
        for gr in group:
            tor_per_query_std_labels = tor_all_std_labels[head:head+gr]
            tor_per_query_preds = tor_all_preds[head:head+gr]
            head += gr

            _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

            sys_sorted_labels = tor_per_query_std_labels[tor_sorted_inds]
            ideal_sorted_labels, _ = torch.sort(tor_per_query_std_labels, descending=True)

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

        return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks,\
               list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q

    def setup_eval(self, data_dict, eval_dict):
        """
        Perform some checks, and revise some setting due to the debug mode
        :param data_dict:
        :param eval_dict:
        :return:
        """
        # required setting to be consistent with the dataset
        if data_dict['data_id'] == 'Istella':
            assert eval_dict['do_validation'] is not True  # since there is no validation data

        self.output_root = self.setup_output(data_dict=data_dict, eval_dict=eval_dict)
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

        self.save_model_dir = self.output_root
        if eval_dict['do_log']: sys.stdout = open(self.output_root + 'log.txt', "w")


    def update_save_model_dir(self, data_dict=None, fold_k=None):
        """
        Update the directory for saving model file when there are multiple folds
        :param data_dict:
        :param fold_k:
        :return:
        """
        if data_dict['data_id'] in MSLETOR or data_dict['data_id'] in MSLRWEB:
            self.save_model_dir = self.output_root + '-'.join(['Fold', str(fold_k)]) + '/'
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)

    def kfold_cv_eval(self, data_dict=None, eval_dict=None, model_para_dict=None):
        """
        Evaluation based on k-fold cross validation if multiple folds exist
        :param data_dict:
        :param eval_dict:
        :param model_para_dict:
        :return:
        """
        self.display_information(data_dict=data_dict)
        self.setup_eval(data_dict=data_dict, eval_dict=eval_dict)
        model_id, data_id = self.model_parameter.model_id, data_dict['data_id']

        fold_num = data_dict['fold_num'] # updated due to the debug mode
        cutoffs, do_validation = eval_dict['cutoffs'], eval_dict['do_validation']

        tree_ranker = globals()[model_id](model_para_dict)

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
            # determine the file paths
            file_train, file_vali, file_test = self.determine_files(data_dict=data_dict, fold_k=fold_k)

            self.update_save_model_dir(data_dict=data_dict, fold_k=fold_k)

            y_test, group_test, y_pred = tree_ranker.run(fold_k=fold_k, file_train=file_train, file_vali=file_vali,
                                                         file_test=file_test, data_dict=data_dict, eval_dict=eval_dict,
                                                         save_model_dir=self.save_model_dir)

            fold_avg_ndcg_at_ks, fold_avg_nerr_at_ks, fold_avg_ap_at_ks, fold_avg_p_at_ks,\
            list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q = \
                                    self.cal_metric_at_ks(model_id=model_id, all_std_labels=y_test, all_preds=y_pred,
                                                          group=group_test, ks=cutoffs)

            performance_list = [model_id] if data_id in YAHOO_LTR or data_id in ISTELLA_LTR else [model_id + ' Fold-' + str(fold_k)]

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

        print(model_id, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_ndcg_scores, list_cutoffs=cutoffs, metric_str='nDCG'))
        print(model_id, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_nerr_scores, list_cutoffs=cutoffs, metric_str='nERR'))
        print(model_id, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_ap_scores, list_cutoffs=cutoffs, metric_str='MAP'))
        print(model_id, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_p_scores, list_cutoffs=cutoffs, metric_str='P'))

        all_fold_ndcg_at_ks_per_q = np.vstack(list_all_fold_ndcg_at_ks_per_q)
        all_fold_err_at_ks_per_q = np.vstack(list_all_fold_err_at_ks_per_q)
        all_fold_ap_at_ks_per_q = np.vstack(list_all_fold_ap_at_ks_per_q)
        all_fold_p_at_ks_per_q = np.vstack(list_all_fold_p_at_ks_per_q)

        pickle_save(all_fold_ndcg_at_ks_per_q, file=self.output_root + '_'.join([data_id, model_id, 'all_fold_ndcg_at_ks_per_q.np']))
        pickle_save(all_fold_err_at_ks_per_q, file=self.output_root + '_'.join([data_id, model_id, 'all_fold_err_at_ks_per_q.np']))
        pickle_save(all_fold_ap_at_ks_per_q, file=self.output_root + '_'.join([data_id, model_id, 'all_fold_ap_at_ks_per_q.np']))
        pickle_save(all_fold_p_at_ks_per_q, file=self.output_root + '_'.join([data_id, model_id, 'all_fold_p_at_ks_per_q.np']))

        return l2r_cv_avg_ndcg_scores, l2r_cv_avg_nerr_scores, l2r_cv_avg_ap_scores, l2r_cv_avg_p_scores

    def set_data_setting(self, debug=False, data_id=None, dir_data=None, tree_data_json=None):
        if tree_data_json is not None:
            self.data_setting = TreeDataSetting(tree_data_json=tree_data_json)
        else:
            self.data_setting = TreeDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, debug=False, dir_output=None, tree_eval_json=None):
        if tree_eval_json is not None:
            self.eval_setting = TreeEvalSetting(debug=debug, tree_eval_json=tree_eval_json)
        else:
            self.eval_setting = TreeEvalSetting(debug=debug, dir_output=dir_output)

    def set_model_setting(self, debug=False, model_id=None, para_json=None):
        if para_json is not None:
            self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
        else:
            self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def point_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None):
        """
        Perform one-time run based on given setting.
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        self.set_model_setting(debug=debug, model_id=model_id)

        self.kfold_cv_eval(data_dict=self.get_default_data_setting(), eval_dict=self.get_default_eval_setting(),
                           model_para_dict=self.get_default_model_setting())


    def grid_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None, dir_json=None):
        """
        Run based on grid-search.
        """
        if dir_json is not None:
            tree_eval_json = dir_json + 'TreeEvalSetting.json'
            tree_data_json = dir_json + 'TreeDataSetting.json'
            para_json = dir_json + model_id + "Parameter.json"

            self.set_eval_setting(debug=debug, tree_eval_json=tree_eval_json)
            self.set_data_setting(tree_data_json=tree_data_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_model_setting(debug=debug, model_id=model_id)

        ''' select the best setting through grid search '''
        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                    for model_para_dict in self.iterate_model_setting():
                        self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict)
