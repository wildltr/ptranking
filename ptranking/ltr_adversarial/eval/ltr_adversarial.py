#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import datetime
import numpy as np
#from tensorboardX import SummaryWriter

from ptranking.utils.bigdata.BigPickle import pickle_save

from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

from ptranking.ltr_adversarial.listwise.irgan_list   import IRGAN_List

from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.ltr_adhoc.eval.eval_utils import ndcg_at_ks, ndcg_at_k
from ptranking.ltr_adversarial.eval.ad_parameter import AdDataSetting, AdEvalSetting, AdScoringFunctionParameter
from ptranking.ltr_global import global_gpu as gpu

LTR_ADVERSARIAL_MODEL = ['IRGAN_Point', 'IRGAN_Pair', 'IRGAN_List']

class AdLTREvaluator(LTREvaluator):
    """
    The class for evaluating different adversarial adversarial ltr methods.
    """
    def __init__(self, id='AD_LTR'):
        super(AdLTREvaluator, self).__init__(id=id)

    def check_consistency(self, data_dict, eval_dict):
        """
        Display some information.
        :param data_dict:
        :param eval_dict:
        :return:
        """
        assert 1 == data_dict['sample_rankings_per_q']  # the required setting w.r.t. adversarial LTR
        if data_dict['data_id'] == 'Istella': assert eval_dict['do_validation'] is not True  # since there is no validation data

    def get_ad_machine(self, eval_dict=None, data_dict=None, sf_para_dict=None, ad_para_dict=None):
        """
        Initialize the adversarial model correspondingly.
        :param eval_dict:
        :param data_dict:
        :param sf_para_dict:
        :param ad_para_dict:
        :return:
        """
        model_id = ad_para_dict['model_id']
        if model_id in ['IRGAN_Point', 'IRGAN_Pair']:
            ad_machine = globals()[model_id](eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict,
                                             temperature=ad_para_dict['temperature'],
                                             d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                             ad_training_order=ad_para_dict['ad_training_order'])
        elif model_id == 'IRGAN_List':
            ad_machine = IRGAN_List(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict,
                                     temperature=ad_para_dict['temperature'],
                                     d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                     samples_per_query=ad_para_dict['samples_per_query'], top_k=ad_para_dict['top_k'],
                                     ad_training_order=ad_para_dict['ad_training_order'], PL=ad_para_dict['PL'],
                                     shuffle_ties=ad_para_dict['shuffle_ties'], repTrick=ad_para_dict['repTrick'],
                                     dropLog=ad_para_dict['dropLog'])
        else:
            raise NotImplementedError

        return ad_machine

    def ad_cv_eval(self, data_dict=None, eval_dict=None, ad_para_dict=None, sf_para_dict=None):
        """
        Adversarial training and evaluation
        :param data_dict:
        :param eval_dict:
        :param ad_para_dict:
        :param sf_para_dict:
        :return:
        """
        self.check_consistency(data_dict, eval_dict)
        self.display_information(data_dict, model_para_dict=ad_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict=ad_para_dict)

        model_id = ad_para_dict['model_id']

        if sf_para_dict['id'] == 'ffnns':
            sf_para_dict['ffnns'].update(dict(num_features=data_dict['num_features']))
        else:
            raise NotImplementedError

        ad_machine = self.get_ad_machine(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)

        time_begin = datetime.datetime.now()  # timing
        g_l2r_cv_avg_scores, d_l2r_cv_avg_scores = np.zeros(len(self.cutoffs)), np.zeros(len(self.cutoffs))  # fold average

        for fold_k in range(1, self.fold_num + 1):
            dict_buffer = dict()  # for buffering frequently used objs
            ad_machine.reset_generator_discriminator()

            fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])

            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k)

            if self.do_vali: g_fold_optimal_ndcgk, d_fold_optimal_ndcgk= 0.0, 0.0
            if self.do_summary:
                list_epoch_loss = [] # not used yet
                g_list_fold_k_train_eval_track, g_list_fold_k_test_eval_track, g_list_fold_k_vali_eval_track = [], [], []
                d_list_fold_k_train_eval_track, d_list_fold_k_test_eval_track, d_list_fold_k_vali_eval_track = [], [], []

            for _ in range(10):
                ad_machine.burn_in(train_data=train_data)


            for epoch_k in range(1, self.epochs + 1):

                if model_id == 'IR_GMAN_List':
                    stop_training = ad_machine.mini_max_train(train_data=train_data, generator=ad_machine.generator,
                                              pool_discriminator=ad_machine.pool_discriminator, dict_buffer=dict_buffer)

                    g_ranker = ad_machine.get_generator()
                    d_ranker = ad_machine.pool_discriminator[0]
                else:
                    stop_training = ad_machine.mini_max_train(train_data=train_data, generator=ad_machine.generator,
                                              discriminator=ad_machine.discriminator, dict_buffer=dict_buffer)

                    g_ranker = ad_machine.get_generator()
                    d_ranker = ad_machine.get_discriminator()

                if stop_training:
                    print('training is failed !')
                    break

                if (self.do_summary or self.do_vali) and (epoch_k % self.log_step == 0 or epoch_k == 1):  # stepwise check
                    if self.do_vali:
                        g_vali_eval_tmp = ndcg_at_k(ranker=g_ranker, test_data=vali_data, k=self.vali_k, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
                        d_vali_eval_tmp = ndcg_at_k(ranker=d_ranker, test_data=vali_data, k=self.vali_k, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
                        g_vali_eval_v, d_vali_eval_v = g_vali_eval_tmp.data.numpy(), d_vali_eval_tmp.data.numpy()

                        if epoch_k > 1:
                            g_buffer, g_tmp_metric_val, g_tmp_epoch = \
                                self.per_epoch_validation(ranker=g_ranker, curr_metric_val=g_vali_eval_v,
                                                          fold_optimal_metric_val=g_fold_optimal_ndcgk, curr_epoch=epoch_k,
                                                          id_str='G', fold_optimal_checkpoint=fold_optimal_checkpoint)
                            # observe better performance
                            if g_buffer: g_fold_optimal_ndcgk, g_fold_optimal_epoch_val = g_tmp_metric_val, g_tmp_epoch

                            d_buffer, d_tmp_metric_val, d_tmp_epoch = \
                                self.per_epoch_validation(ranker=d_ranker, curr_metric_val=d_vali_eval_v,
                                                          fold_optimal_metric_val=d_fold_optimal_ndcgk, curr_epoch=epoch_k,
                                                          id_str='D', fold_optimal_checkpoint=fold_optimal_checkpoint)
                            if d_buffer: d_fold_optimal_ndcgk, d_fold_optimal_epoch_val = d_tmp_metric_val, d_tmp_epoch

                    if self.do_summary: # summarize per-step performance w.r.t. train, test
                        self.per_epoch_summary_step1(ranker=g_ranker, train_data=train_data, test_data=test_data,
                                                     list_fold_k_train_eval_track=g_list_fold_k_train_eval_track,
                                                     list_fold_k_test_eval_track=g_list_fold_k_test_eval_track,
                                                     vali_eval_v=g_vali_eval_v,
                                                     list_fold_k_vali_eval_track=g_list_fold_k_vali_eval_track)

                        self.per_epoch_summary_step1(ranker=d_ranker, train_data=train_data, test_data=test_data,
                                                     list_fold_k_train_eval_track=d_list_fold_k_train_eval_track,
                                                     list_fold_k_test_eval_track=d_list_fold_k_test_eval_track,
                                                     vali_eval_v=d_vali_eval_v,
                                                     list_fold_k_vali_eval_track=d_list_fold_k_vali_eval_track)

            if self.do_summary:
                self.per_epoch_summary_step2(id_str='G', fold_k=fold_k,
                                             list_fold_k_train_eval_track=g_list_fold_k_train_eval_track,
                                             list_fold_k_test_eval_track=g_list_fold_k_test_eval_track,
                                             list_epoch_loss=list_epoch_loss,
                                             list_fold_k_vali_eval_track=g_list_fold_k_vali_eval_track)

                self.per_epoch_summary_step2(id_str='D', fold_k=fold_k,
                                             list_fold_k_train_eval_track=d_list_fold_k_train_eval_track,
                                             list_fold_k_test_eval_track=d_list_fold_k_test_eval_track,
                                             list_epoch_loss=list_epoch_loss,
                                             list_fold_k_vali_eval_track=d_list_fold_k_vali_eval_track)

            if self.do_vali: # using the fold-wise optimal model for later testing based on validation data #
                    g_buffered_model = '_'.join(['net_params_epoch', str(g_fold_optimal_epoch_val), 'G']) + '.pkl'
                    g_ranker.load(self.dir_run + fold_optimal_checkpoint + '/' + g_buffered_model)
                    g_fold_optimal_ranker = g_ranker

                    d_buffered_model = '_'.join(['net_params_epoch', str(d_fold_optimal_epoch_val), 'D']) + '.pkl'
                    d_ranker.load(self.dir_run + fold_optimal_checkpoint + '/' + d_buffered_model)
                    d_fold_optimal_ranker = d_ranker

            else: # using default G # buffer the model after a fixed number of training-epoches if no validation is deployed
                g_ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k), 'G']) + '.pkl')
                g_fold_optimal_ranker = g_ranker

                d_ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k), 'D']) + '.pkl')
                d_fold_optimal_ranker = d_ranker

            g_torch_fold_ndcg_ks = ndcg_at_ks(ranker=g_fold_optimal_ranker, test_data=test_data, ks=self.cutoffs, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
            g_fold_ndcg_ks = g_torch_fold_ndcg_ks.data.numpy()

            d_torch_fold_ndcg_ks = ndcg_at_ks(ranker=d_fold_optimal_ranker, test_data=test_data, ks=self.cutoffs, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
            d_fold_ndcg_ks = d_torch_fold_ndcg_ks.data.numpy()

            performance_list = [' Fold-' + str(fold_k)]  # fold-wise performance
            performance_list.append('Generator')
            for i, co in enumerate(self.cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, g_fold_ndcg_ks[i]))

            performance_list.append('\nDiscriminator')
            for i, co in enumerate(self.cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, d_fold_ndcg_ks[i]))

            performance_str = '\t'.join(performance_list)
            print('\t', performance_str)

            g_l2r_cv_avg_scores = np.add(g_l2r_cv_avg_scores, g_fold_ndcg_ks)  # sum for later cv-performance
            d_l2r_cv_avg_scores = np.add(d_l2r_cv_avg_scores, d_fold_ndcg_ks)

        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - time_begin)
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        # begin to print either cv or average performance
        g_l2r_cv_avg_scores = np.divide(g_l2r_cv_avg_scores, self.fold_num)
        d_l2r_cv_avg_scores = np.divide(d_l2r_cv_avg_scores, self.fold_num)

        if self.do_vali:
            eval_prefix = str(self.fold_num) + '-fold cross validation scores:'
        else:
            eval_prefix = str(self.fold_num) + '-fold average scores:'

        print('Generator', eval_prefix, metric_results_to_string(list_scores=g_l2r_cv_avg_scores, list_cutoffs=self.cutoffs))
        print('Discriminator', eval_prefix, metric_results_to_string(list_scores=d_l2r_cv_avg_scores, list_cutoffs=self.cutoffs))


    def per_epoch_validation(self, ranker, curr_metric_val, fold_optimal_metric_val, curr_epoch, id_str, fold_optimal_checkpoint):
        info_str = ' '.join([str(curr_epoch), ' ', id_str, ' - nDCG@k - ', str(curr_metric_val)])

        if (curr_metric_val > fold_optimal_metric_val) or \
                (curr_epoch == self.epochs and curr_metric_val == fold_optimal_metric_val):  # we need at least a reference, in case all zero
            print('\t', info_str)
            fold_optimal_metric_val = curr_metric_val
            fold_optimal_epoch_val = curr_epoch
            ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/',
                              name='_'.join(['net_params_epoch', str(curr_epoch), id_str]) + '.pkl')  # buffer currently optimal model

            return True, fold_optimal_metric_val, fold_optimal_epoch_val
        else:
            print('\t\t', info_str)
            return False, None, None

    def per_epoch_summary_step1(self, ranker, train_data, list_fold_k_train_eval_track,
                          test_data, list_fold_k_test_eval_track, vali_eval_v, list_fold_k_vali_eval_track):

        fold_k_epoch_k_train_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=train_data, ks=self.cutoffs, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
        np_fold_k_epoch_k_train_ndcg_ks = fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
        list_fold_k_train_eval_track.append(np_fold_k_epoch_k_train_ndcg_ks)

        fold_k_epoch_k_test_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=test_data, ks=self.cutoffs, multi_level_rele=self.data_setting.data_dict['multi_level_rele'], batch_mode=True)
        np_fold_k_epoch_k_test_ndcg_ks = fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
        list_fold_k_test_eval_track.append(np_fold_k_epoch_k_test_ndcg_ks)

        #fold_k_epoch_k_loss = torch_fold_k_epoch_k_loss.cpu().numpy() if gpu else torch_fold_k_epoch_k_loss.data.numpy()
        #list_epoch_loss.append(fold_k_epoch_k_loss)

        if self.do_vali: list_fold_k_vali_eval_track.append(vali_eval_v)

    def per_epoch_summary_step2(self, id_str, fold_k, list_fold_k_train_eval_track, list_fold_k_test_eval_track, list_epoch_loss, list_fold_k_vali_eval_track):
        sy_prefix = '_'.join(['Fold', str(fold_k)])

        fold_k_train_eval = np.vstack(list_fold_k_train_eval_track)
        fold_k_test_eval = np.vstack(list_fold_k_test_eval_track)
        pickle_save(fold_k_train_eval, file=self.dir_run + '_'.join([sy_prefix, id_str, 'train_eval.np']))
        pickle_save(fold_k_test_eval, file=self.dir_run + '_'.join([sy_prefix, id_str, 'test_eval.np']))

        '''
        fold_k_epoch_loss = np.hstack(list_epoch_loss)
        pickle_save((fold_k_epoch_loss, train_data.__len__()),
                    file=self.dir_run + '_'.join([sy_prefix, id_str, 'epoch_loss.np']))
        '''

        if self.do_vali:
            fold_k_vali_eval = np.hstack(list_fold_k_vali_eval_track)
            pickle_save(fold_k_vali_eval, file=self.dir_run + '_'.join([sy_prefix, id_str, 'vali_eval.np']))

    def set_data_setting(self, debug=False, data_id=None, dir_data=None):
        self.data_setting = AdDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, debug=False, dir_output=None):
        self.eval_setting = AdEvalSetting(debug=debug, dir_output=dir_output)

    def set_scoring_function_setting(self, debug=None, data_dict=None):
        self.sf_parameter = AdScoringFunctionParameter(debug=debug, data_dict=data_dict)

    def iterate_scoring_function_setting(self):
        return self.sf_parameter.grid_search()

    def set_model_setting(self, debug=False, model_id=None):
        self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def grid_run(self, debug=True, model_id=None, data_id=None, dir_data=None, dir_output=None):
        """
        Perform adversarial learning-to-rank based on grid search of optimal parameter setting
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)

        self.set_scoring_function_setting(debug=debug)

        self.set_model_setting(debug=debug, model_id=model_id)

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                for sf_para_dict in self.iterate_scoring_function_setting():
                    for ad_para_dict in self.iterate_model_setting():
                        self.ad_cv_eval(data_dict=data_dict, eval_dict=eval_dict,
                                        sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)


    def point_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None):
        """

        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """

        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()

        self.set_scoring_function_setting(debug=debug, data_dict=data_dict)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        ad_model_para_dict = self.get_default_model_setting()

        self.ad_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                        ad_para_dict=ad_model_para_dict)
