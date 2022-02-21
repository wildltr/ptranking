#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
A general framework for evaluating adversarial learning-to-rank methods.
"""

import datetime
import numpy as np
#from tensorboardX import SummaryWriter

from ptranking.utils.bigdata.BigPickle import pickle_save

from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

from ptranking.data.data_utils import MSLETOR_SEMI
from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.ltr_adversarial.eval.ad_parameter import AdDataSetting, AdEvalSetting, AdScoringFunctionParameter

from ptranking.ltr_adversarial.pointwise.irgan_point import IRGAN_Point, IRGAN_PointParameter
from ptranking.ltr_adversarial.pointwise.irfgan_point import IRFGAN_Point, IRFGAN_PointParameter
from ptranking.ltr_adversarial.pairwise.irgan_pair import IRGAN_Pair, IRGAN_PairParameter
from ptranking.ltr_adversarial.pairwise.irfgan_pair import IRFGAN_PairParameter, IRFGAN_Pair
from ptranking.ltr_adversarial.listwise.irgan_list import IRGAN_List, IRGAN_ListParameter
from ptranking.ltr_adversarial.listwise.irfgan_list import IRFGAN_List, IRFGAN_ListParameter

LTR_ADVERSARIAL_MODEL = ['IRGAN_Point', 'IRGAN_Pair', 'IRGAN_List',
                         'IRFGAN_Point', 'IRFGAN_Pair', 'IRFGAN_List']

class AdLTREvaluator(LTREvaluator):
    """
    The class for evaluating different adversarial adversarial ltr methods.
    """
    def __init__(self, frame_id=LTRFRAME_TYPE.Adversarial, cuda=None):
        super(AdLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

    def check_consistency(self, data_dict, eval_dict, sf_para_dict):
        """
        Check whether the settings are reasonable in the context of adversarial learning-to-rank
        """
        ''' Part-1: data loading '''
        assert 1 == data_dict['train_rough_batch_size']  # the required setting w.r.t. adversarial LTR

        if data_dict['data_id'] == 'Istella':
            assert eval_dict['do_validation'] is not True  # since there is no validation data

        if data_dict['data_id'] in MSLETOR_SEMI:
            assert data_dict['unknown_as_zero'] is not True  # use original data

        if data_dict['scale_data']:
            scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
            assert not scaler_level == 'DATASET'  # not supported setting

        assert data_dict['validation_presort']  # Rule of thumb, as validation and test data are for metric-performance
        assert data_dict['test_presort']  # Rule of thumb, as validation and test data are for metric-performance

        ''' Part-2: evaluation setting '''
        if eval_dict['mask_label']:  # True is aimed to use supervised data to mimic semi-supervised data by masking
            assert not data_dict['data_id'] in MSLETOR_SEMI

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
        if model_id in ['IRGAN_Point', 'IRGAN_Pair', 'IRGAN_List', 'IRFGAN_Point', 'IRFGAN_Pair', 'IRFGAN_List']:
            ad_machine = globals()[model_id](eval_dict=eval_dict, data_dict=data_dict, gpu=self.gpu, device=self.device,
                                             sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)
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
        self.display_information(data_dict, model_para_dict=ad_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict=sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict=ad_para_dict)

        model_id = ad_para_dict['model_id']
        fold_num = data_dict['fold_num']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, do_summary = eval_dict['do_validation'], eval_dict['do_summary']

        ad_machine = self.get_ad_machine(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)

        time_begin = datetime.datetime.now()  # timing
        g_l2r_cv_avg_scores, d_l2r_cv_avg_scores = np.zeros(len(cutoffs)), np.zeros(len(cutoffs))  # fold average

        '''
        Dataset-level buffering of frequently used information
        1> e.g., number of positive documents per-query
        '''
        global_buffer = dict() # refresh for each model instance

        for fold_k in range(1, fold_num + 1):
            ad_machine.reset_generator_discriminator()

            fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k)

            # update due to new train_data
            ad_machine.fill_global_buffer(train_data, dict_buffer=global_buffer)

            if do_vali: g_fold_optimal_ndcgk, d_fold_optimal_ndcgk= 0.0, 0.0
            if do_summary:
                list_epoch_loss = [] # not used yet
                g_list_fold_k_train_eval_track, g_list_fold_k_test_eval_track, g_list_fold_k_vali_eval_track = [], [], []
                d_list_fold_k_train_eval_track, d_list_fold_k_test_eval_track, d_list_fold_k_vali_eval_track = [], [], []

            for _ in range(10):
                ad_machine.burn_in(train_data=train_data)

            for epoch_k in range(1, epochs + 1):
                if model_id == 'IR_GMAN_List':
                    stop_training = ad_machine.mini_max_train(train_data=train_data, generator=ad_machine.generator,
                                              pool_discriminator=ad_machine.pool_discriminator, global_buffer=global_buffer)

                    g_ranker = ad_machine.get_generator()
                    d_ranker = ad_machine.pool_discriminator[0]
                else:
                    stop_training = ad_machine.mini_max_train(train_data=train_data, generator=ad_machine.generator,
                                              discriminator=ad_machine.discriminator, global_buffer=global_buffer)

                    g_ranker = ad_machine.get_generator()
                    d_ranker = ad_machine.get_discriminator()

                if stop_training:
                    print('training is failed !')
                    break

                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:
                        g_vali_eval_tmp = g_ranker.ndcg_at_k(test_data=vali_data, k=vali_k, label_type=self.data_setting.data_dict['label_type'])
                        d_vali_eval_tmp = d_ranker.ndcg_at_k(test_data=vali_data, k=vali_k, label_type=self.data_setting.data_dict['label_type'])
                        g_vali_eval_v, d_vali_eval_v = g_vali_eval_tmp.data.numpy(), d_vali_eval_tmp.data.numpy()

                        if epoch_k > 1:
                            g_buffer, g_tmp_metric_val, g_tmp_epoch = \
                                self.per_epoch_validation(ranker=g_ranker, curr_metric_val=g_vali_eval_v,
                                                          fold_optimal_metric_val=g_fold_optimal_ndcgk, curr_epoch=epoch_k,
                                                          id_str='G', fold_optimal_checkpoint=fold_optimal_checkpoint, epochs=epochs)
                            # observe better performance
                            if g_buffer: g_fold_optimal_ndcgk, g_fold_optimal_epoch_val = g_tmp_metric_val, g_tmp_epoch

                            d_buffer, d_tmp_metric_val, d_tmp_epoch = \
                                self.per_epoch_validation(ranker=d_ranker, curr_metric_val=d_vali_eval_v,
                                                          fold_optimal_metric_val=d_fold_optimal_ndcgk, curr_epoch=epoch_k,
                                                          id_str='D', fold_optimal_checkpoint=fold_optimal_checkpoint, epochs=epochs)
                            if d_buffer: d_fold_optimal_ndcgk, d_fold_optimal_epoch_val = d_tmp_metric_val, d_tmp_epoch

                    if do_summary: # summarize per-step performance w.r.t. train, test
                        self.per_epoch_summary_step1(ranker=g_ranker, train_data=train_data, test_data=test_data,
                                                     list_fold_k_train_eval_track=g_list_fold_k_train_eval_track,
                                                     list_fold_k_test_eval_track=g_list_fold_k_test_eval_track,
                                                     vali_eval_v=g_vali_eval_v,
                                                     list_fold_k_vali_eval_track=g_list_fold_k_vali_eval_track,
                                                     cutoffs=cutoffs, do_vali=do_vali)

                        self.per_epoch_summary_step1(ranker=d_ranker, train_data=train_data, test_data=test_data,
                                                     list_fold_k_train_eval_track=d_list_fold_k_train_eval_track,
                                                     list_fold_k_test_eval_track=d_list_fold_k_test_eval_track,
                                                     vali_eval_v=d_vali_eval_v,
                                                     list_fold_k_vali_eval_track=d_list_fold_k_vali_eval_track,
                                                     cutoffs=cutoffs, do_vali=do_vali)

            if do_summary:
                self.per_epoch_summary_step2(id_str='G', fold_k=fold_k,
                                             list_fold_k_train_eval_track=g_list_fold_k_train_eval_track,
                                             list_fold_k_test_eval_track=g_list_fold_k_test_eval_track,
                                             do_vali=do_vali,
                                             list_fold_k_vali_eval_track=g_list_fold_k_vali_eval_track)

                self.per_epoch_summary_step2(id_str='D', fold_k=fold_k,
                                             list_fold_k_train_eval_track=d_list_fold_k_train_eval_track,
                                             list_fold_k_test_eval_track=d_list_fold_k_test_eval_track,
                                             do_vali=do_vali,
                                             list_fold_k_vali_eval_track=d_list_fold_k_vali_eval_track)

            if do_vali: # using the fold-wise optimal model for later testing based on validation data #
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

            g_torch_fold_ndcg_ks = g_fold_optimal_ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, label_type=self.data_setting.data_dict['label_type'])
            g_fold_ndcg_ks = g_torch_fold_ndcg_ks.data.numpy()

            d_torch_fold_ndcg_ks = d_fold_optimal_ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, label_type=self.data_setting.data_dict['label_type'])
            d_fold_ndcg_ks = d_torch_fold_ndcg_ks.data.numpy()

            performance_list = [' Fold-' + str(fold_k)]  # fold-wise performance
            performance_list.append('Generator')
            for i, co in enumerate(cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, g_fold_ndcg_ks[i]))

            performance_list.append('\nDiscriminator')
            for i, co in enumerate(cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, d_fold_ndcg_ks[i]))

            performance_str = '\t'.join(performance_list)
            print('\t', performance_str)

            g_l2r_cv_avg_scores = np.add(g_l2r_cv_avg_scores, g_fold_ndcg_ks)  # sum for later cv-performance
            d_l2r_cv_avg_scores = np.add(d_l2r_cv_avg_scores, d_fold_ndcg_ks)

        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - time_begin)
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        # begin to print either cv or average performance
        g_l2r_cv_avg_scores = np.divide(g_l2r_cv_avg_scores, fold_num)
        d_l2r_cv_avg_scores = np.divide(d_l2r_cv_avg_scores, fold_num)

        if do_vali:
            eval_prefix = str(fold_num) + '-fold cross validation scores:'
        else:
            eval_prefix = str(fold_num) + '-fold average scores:'

        print('Generator', eval_prefix, metric_results_to_string(list_scores=g_l2r_cv_avg_scores, list_cutoffs=cutoffs))
        print('Discriminator', eval_prefix, metric_results_to_string(list_scores=d_l2r_cv_avg_scores, list_cutoffs=cutoffs))


    def per_epoch_validation(self, ranker, curr_metric_val, fold_optimal_metric_val, curr_epoch, id_str, fold_optimal_checkpoint, epochs):
        info_str = ' '.join([str(curr_epoch), ' ', id_str, ' - nDCG@k - ', str(curr_metric_val)])

        if (curr_metric_val > fold_optimal_metric_val) or \
                (curr_epoch == epochs and curr_metric_val == fold_optimal_metric_val):  # we need at least a reference, in case all zero
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
                          test_data, list_fold_k_test_eval_track, vali_eval_v, list_fold_k_vali_eval_track, cutoffs, do_vali):

        fold_k_epoch_k_train_ndcg_ks = ranker.ndcg_at_ks(test_data=train_data, ks=cutoffs, label_type=self.data_setting.data_dict['label_type'])
        np_fold_k_epoch_k_train_ndcg_ks = fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
        list_fold_k_train_eval_track.append(np_fold_k_epoch_k_train_ndcg_ks)

        fold_k_epoch_k_test_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, label_type=self.data_setting.data_dict['label_type'])
        np_fold_k_epoch_k_test_ndcg_ks = fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
        list_fold_k_test_eval_track.append(np_fold_k_epoch_k_test_ndcg_ks)

        #fold_k_epoch_k_loss = torch_fold_k_epoch_k_loss.cpu().numpy() if gpu else torch_fold_k_epoch_k_loss.data.numpy()
        #list_epoch_loss.append(fold_k_epoch_k_loss)

        if do_vali: list_fold_k_vali_eval_track.append(vali_eval_v)

    def per_epoch_summary_step2(self, id_str, fold_k, list_fold_k_train_eval_track, list_fold_k_test_eval_track, do_vali, list_fold_k_vali_eval_track):
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

        if do_vali:
            fold_k_vali_eval = np.hstack(list_fold_k_vali_eval_track)
            pickle_save(fold_k_vali_eval, file=self.dir_run + '_'.join([sy_prefix, id_str, 'vali_eval.np']))

    def set_data_setting(self, debug=False, data_id=None, dir_data=None, ad_data_json=None):
        if ad_data_json is not None:
            self.data_setting = AdDataSetting(ad_data_json=ad_data_json)
        else:
            self.data_setting = AdDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, debug=False, dir_output=None, ad_eval_json=None):
        if ad_eval_json is not None:
            self.eval_setting = AdEvalSetting(debug=debug, ad_eval_json=ad_eval_json)
        else:
            self.eval_setting = AdEvalSetting(debug=debug, dir_output=dir_output)

    def set_scoring_function_setting(self, debug=None, sf_id=None, sf_json=None):
        if sf_json is not None:
            self.sf_parameter = AdScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = AdScoringFunctionParameter(debug=debug, sf_id=sf_id)

    def iterate_scoring_function_setting(self):
        return self.sf_parameter.grid_search()

    def set_model_setting(self, debug=False, model_id=None, para_json=None):
        if para_json is not None:
            self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
        else:
            self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def grid_run(self, debug=True, model_id=None, data_id=None, dir_data=None, dir_output=None, dir_json=None):
        """
        Perform adversarial learning-to-rank based on grid search of optimal parameter setting
        """
        if dir_json is not None:
            ad_data_eval_sf_json = dir_json + 'Ad_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "Parameter.json"
            self.set_eval_setting(debug=debug, ad_eval_json=ad_data_eval_sf_json)
            self.set_data_setting(ad_data_json=ad_data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=ad_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
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


    def point_run(self, debug=False, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None):
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

        self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        ad_model_para_dict = self.get_default_model_setting()

        self.ad_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                        ad_para_dict=ad_model_para_dict)

    def run(self, debug=False, model_id=None, sf_id=None, config_with_json=None, dir_json=None,
            data_id=None, dir_data=None, dir_output=None, grid_search=False):
        if config_with_json:
            assert dir_json is not None
            self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            assert sf_id in ['pointsf', 'listsf']
            if not model_id.endswith('List'):
                assert sf_id == 'pointsf'

            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, sf_id=sf_id,
                              data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, sf_id=sf_id,
                               data_id=data_id, dir_data=dir_data, dir_output=dir_output)
