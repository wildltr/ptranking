#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/01 | https://y-research.github.io

"""Description

"""

import os
import sys
import datetime
import numpy as np
from itertools import product
#from tensorboardX import SummaryWriter

from org.archive.data import data_utils

from org.archive.ltr_adhoc.eval.grid_utils import get_sf_ID
from org.archive.utils.bigdata.BigPickle import pickle_save

from org.archive.ltr_adversarial_learning.pointwise.point_ir_gan import Point_IR_GAN, get_point_irgan_paras_str
from org.archive.ltr_adversarial_learning.pairwise.pair_ir_gan   import Pair_IR_GAN, get_pair_irgan_paras_str
from org.archive.ltr_adversarial_learning.listwise.list_ir_gan   import List_IR_GAN, get_list_irgan_paras_str

from org.archive.ltr_adversarial_learning.pointwise.point_ir_wgan import Point_IR_WGAN
from org.archive.ltr_adversarial_learning.pairwise.pair_ir_wgan   import Pair_IR_WGAN
from org.archive.ltr_adversarial_learning.listwise.list_ir_wgan   import List_IR_WGAN

from org.archive.ltr_adversarial_learning.pointwise.point_ir_fgan import Point_IR_FGAN
from org.archive.ltr_adversarial_learning.pairwise.pair_ir_fgan   import Pair_IR_FGAN
from org.archive.ltr_adversarial_learning.listwise.list_ir_fgan   import List_IR_FGAN

from org.archive.ltr_adversarial_learning.listwise.list_ir_gman import List_IR_GMAN

from org.archive.ltr_adversarial_learning.pointwise.point_ir_swgan import Point_IR_SW_GAN

from org.archive.l2r_global import global_gpu as gpu
from org.archive.ltr_adhoc.eval.l2r import L2REvaluator
from org.archive.ltr_adhoc.eval.grid_utils import sf_grid
from org.archive.ltr_adversarial_learning.eval.grid_utils import ad_list_grid, ad_eval_grid


class AdL2REvaluator(L2REvaluator):
    """
    The class for evaluating different adversarial ltr_adhoc methods.
    """
    def __init__(self, id='AD_LTR'):
        super(AdL2REvaluator, self).__init__(id=id)

    def pre_check(self, data_dict, eval_dict, sf_para_dict):
        assert 1 == data_dict['sample_rankings_per_q']  # required setting w.r.t. adversarial L2R

    def setup_output(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):
        ''' Setting of the output directory '''

        model_id = model_para_dict['model_id']
        grid_search, do_vali, dir_output, query_aware = eval_dict['grid_search'], eval_dict['do_vali'], eval_dict['dir_output'], eval_dict['query_aware']
        semi_context = eval_dict['semi_context']

        dir_root = self.update_dir_root(grid_search, dir_output, model_para_dict)

        eval_dict['dir_root'] = dir_root
        if not os.path.exists(dir_root): os.makedirs(dir_root)

        sf_str = get_sf_ID(sf_para_dict=sf_para_dict)
        data_eval_str = self.get_data_eval_str(data_dict, eval_dict)

        if semi_context:
            data_eval_str = '_'.join([data_eval_str, 'Semi', 'Ratio', '{:,g}'.format(eval_dict['mask_ratio'])])

        if query_aware:
            file_prefix = '_'.join([model_id, 'QA', 'SF', sf_str, data_eval_str])
        else:
            file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join([file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join([file_prefix, 'DS', data_dict['scaler_id']])
        else:
            file_prefix = '_'.join([file_prefix, 'QS', 'BN'])

        ''' w.r.t. adversarial hyper-parameters '''
        if model_id == 'IR_GAN_List':
            model_para_str = get_list_irgan_paras_str(model_para_dict=model_para_dict, log=False)
            file_prefix = '_'.join([file_prefix, model_para_str])

        elif model_id == 'IR_GAN_Point':
            model_para_str = get_point_irgan_paras_str(model_para_dict=model_para_dict, log=False)
            file_prefix = '_'.join([file_prefix, model_para_str])

        elif model_id == 'IR_GAN_Pair':
            model_para_str = get_pair_irgan_paras_str(model_para_dict=model_para_dict, log=False)
            file_prefix = '_'.join([file_prefix, model_para_str])
        else:
            pass

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run


    def get_ad_machine(self, eval_dict=None, data_dict=None, sf_para_dict=None, ad_para_dict=None):
        model_id = ad_para_dict['model_id']

        if model_id == 'IR_GAN_Point':  # pointwise
            ad_machine = Point_IR_GAN(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict,
                                      temperature=ad_para_dict['temperature'],
                                      d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                      ad_training_order=ad_para_dict['ad_training_order'])

        elif model_id == 'IR_GAN_Pair':
            ad_machine = Pair_IR_GAN(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict,
                                     temperature=ad_para_dict['temperature'], d_epoches=ad_para_dict['d_epoches'],
                                     g_epoches=ad_para_dict['g_epoches'], ad_training_order=ad_para_dict['ad_training_order'])

        elif model_id == 'IR_GAN_List':
            ad_machine = List_IR_GAN(eval_dict=eval_dict, data_dict=data_dict, sf_para_dict=sf_para_dict,
                                     temperature=ad_para_dict['temperature'],
                                     d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                     samples_per_query=ad_para_dict['samples_per_query'], top_k=ad_para_dict['top_k'],
                                     ad_training_order=ad_para_dict['ad_training_order'], PL=ad_para_dict['PL'],
                                     shuffle_ties=ad_para_dict['shuffle_ties'],  repTrick=ad_para_dict['repTrick'],
                                     dropLog=ad_para_dict['dropLog'])

        elif model_id == 'IR_GMAN_List':
            ad_machine = List_IR_GMAN(sf_para_dict=sf_para_dict, temperature=ad_para_dict['temperature'],
                                     d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                     samples_per_query=ad_para_dict['samples_per_query'], top_k=ad_para_dict['top_k'],
                                     ad_training_order=ad_para_dict['ad_training_order'], PL=ad_para_dict['PL'],
                                     shuffle_ties=ad_para_dict['shuffle_ties'],  repTrick=ad_para_dict['repTrick'],
                                     dropLog=ad_para_dict['dropLog'])


        elif model_id == 'IR_WGAN_Point':
            ad_machine = Point_IR_WGAN(sf_para_dict=sf_para_dict, temperature=ad_para_dict['temperature'],
                                      d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                      ad_training_order=ad_para_dict['ad_training_order'])

        elif model_id == 'IR_WGAN_Pair':
            ad_machine = Pair_IR_WGAN(rf_para_dict=sf_para_dict)

        elif model_id == 'IR_WGAN_List':
            ad_machine = List_IR_WGAN(sf_para_dict=sf_para_dict, temperature=ad_para_dict['temperature'],
                                     d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                     samples_per_query=ad_para_dict['samples_per_query'], top_k=ad_para_dict['top_k'],
                                     ad_training_order=ad_para_dict['ad_training_order'], PL=ad_para_dict['PL'],
                                     shuffle_ties=ad_para_dict['shuffle_ties'])

        elif model_id == 'IR_FGAN_Point':
            ad_machine = Point_IR_FGAN(sf_para_dict=sf_para_dict)
        elif model_id == 'IR_FGAN_Pair':
            ad_machine = Pair_IR_FGAN(sf_para_dict=sf_para_dict)
        elif model_id == 'IR_FGAN_List':
            ad_machine = List_IR_FGAN(sf_para_dict=sf_para_dict)

        elif model_id == 'IR_SWGAN_Point':
            ad_machine = Point_IR_SW_GAN(sf_para_dict=sf_para_dict, temperature=ad_para_dict['temperature'],
                                      d_epoches=ad_para_dict['d_epoches'], g_epoches=ad_para_dict['g_epoches'],
                                      ad_training_order=ad_para_dict['ad_training_order'])

        else:
            raise NotImplementedError

        return ad_machine


    def cv_eval(self, data_dict=None, eval_dict=None, ad_para_dict=None, sf_para_dict=None):

        self.ad_cv_eval(data_dict=data_dict, eval_dict=eval_dict, ad_para_dict=ad_para_dict, sf_para_dict=sf_para_dict)


    def ad_cv_eval(self, data_dict=None, eval_dict=None, ad_para_dict=None, sf_para_dict=None):

        self.pre_check(data_dict, eval_dict, sf_para_dict)
        self.show_infor(data_dict, model_para_dict=ad_para_dict)
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

            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k, model_para_dict=ad_para_dict)

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
                        g_vali_eval_tmp = self.get_ndcg_at_k(ranker=g_ranker, test_data=vali_data, k=self.vali_k)
                        d_vali_eval_tmp = self.get_ndcg_at_k(ranker=d_ranker, test_data=vali_data, k=self.vali_k)
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

            g_torch_fold_ndcg_ks = self.get_ndcg_at_ks(ranker=g_fold_optimal_ranker, test_data=test_data, ks=self.cutoffs)
            g_fold_ndcg_ks = g_torch_fold_ndcg_ks.data.numpy()

            d_torch_fold_ndcg_ks = self.get_ndcg_at_ks(ranker=d_fold_optimal_ranker, test_data=test_data, ks=self.cutoffs)
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

        print('Generator', eval_prefix, self.result_to_str(list_scores=g_l2r_cv_avg_scores, list_cutoffs=self.cutoffs))
        print('Discriminator', eval_prefix, self.result_to_str(list_scores=d_l2r_cv_avg_scores, list_cutoffs=self.cutoffs))


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

        fold_k_epoch_k_train_ndcg_ks = self.get_ndcg_at_ks(ranker=ranker, test_data=train_data, ks=self.cutoffs)
        np_fold_k_epoch_k_train_ndcg_ks = fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
        list_fold_k_train_eval_track.append(np_fold_k_epoch_k_train_ndcg_ks)

        fold_k_epoch_k_test_ndcg_ks = self.get_ndcg_at_ks(ranker=ranker, test_data=test_data, ks=self.cutoffs)
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


    def grid_run(self, model_id=None, data_dict=None, eval_dict=None):
        ''' perform adversarial learning-to-rank based on grid search of optimal parameter setting '''

        debug = eval_dict['debug']
        #query_aware = eval_dict['query_aware']
        do_log = True if debug else True

        # more data settings that are rarely changed
        data_dict.update(dict(max_docs='All', min_docs=1, min_rele=1))

        # more evaluation settings that are rarely changed
        vali_k = 5
        cutoffs = [1, 3, 5, 10, 20, 50]
        eval_dict.update(dict(vali_k=vali_k, cutoffs=cutoffs, do_log=do_log, log_step=2, do_summary=False, loss_guided=False))

        choice_scale_data, choice_scaler_id, choice_scaler_level = self.get_scaler_setting(data_id=data_dict['data_id'], grid_search=True)

        ''' setting w.r.t. train  '''
        choice_validation = [False] if debug else [False]  # True, False
        choice_epoch = [20] if debug else [100]
        choice_sample_rankings_per_q = [1] if debug else [1]  # this should be 1 for adversarial learning-to-rank

        #choice_semi_context = [True] if debug else [True]
        #choice_presort      = [True] if debug else [True]

        choice_semi_context = [False] if debug else [False]
        choice_presort      = [False] if debug else [False]

        if data_dict['data_id'] in data_utils.MSLETOR_SEMI:
            choice_binary_rele  = [True] if debug else [True]
        else:
            choice_binary_rele  = [False] if debug else [False]

        #choice_binary_rele  = [False] if debug else [False]


        '''
        """ [1] setting w.r.t. ltr_adhoc function on in-depth exploration of neural network setting for l2r"""
        choice_apply_BN = [False] if debug else [True]  # True, False
        choice_apply_RD = [False] if debug else [False]  # True, False

        choice_layers = [3]     if debug else [2,4,6,8,10,15,20,25,30,35,40,45,50]  # 1, 2, 3, 4
        choice_hd_hn_af = ['S'] if debug else ['R']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
        choice_tl_af = ['S']    if debug else ['R']  # activation function for the last layer, sigmoid is suggested due to zero-prediction
        choice_hd_hn_tl_af = ['CE'] if debug else ['CE', 'R', 'LR', 'S'] # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
        choice_apply_tl_af = [True]  # True, False
        '''

        """ [2] setting w.r.t. ltr_adhoc function on virtual functions """
        choice_apply_BN = [False] if debug else [False]  # True, False
        choice_apply_RD = [False] if debug else [False]  # True, False

        choice_layers = [5] if debug else [5]  # 1, 2, 3, 4
        choice_hd_hn_af = ['R', 'T'] if debug else ['R']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
        choice_tl_af = ['S'] if debug else ['S']  # activation function for the last layer, sigmoid is suggested due to zero-prediction

        #choice_hd_hn_tl_af = ['S', 'CE'] if debug else ['CE']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
        choice_hd_hn_tl_af = None

        choice_apply_tl_af = [True]  # True, False todo now it is a in-built setting w.r.t. specific models


        """ adversarial hyper-parameters """
        ## (1) common across point, pair, list ##
        choice_samples_per_query = [5]
        choice_ad_training_order = ['DG'] # GD for irganlist DG for point/pair

        #choice_mask_ratios = None
        choice_mask_ratios = [0.2] if debug else [0.2, 0.4, 0.6, 0.8]  # 0.5, 1.0
        choice_mask_type = ['rand_mask_rele'] if debug else ['rand_mask_rele']

        ''' IRGAN specific '''
        choice_temperatures = [0.5] if debug else [0.5]  # 0.5, 1.0

        ''' Adversarial training: discriminator-epoches vs. generator-epoches '''
        choice_d_g_epoches = [(1, 1)] if debug else [(1, 1)]

        ## pair-specific
        choice_losstype_d = ['svm']

        ## list-specific ##
        choice_top_k = [5]
        choice_shuffle_ties = [False] # todo should be True, otherwise it is probable that the used stdandard ltr_adhoc is in fact not truth ltr_adhoc.
        choice_PL = [True] # discriminator formulation
        choice_repTrick = [False] # for generator
        choice_dropLog = [True] # drop log of discriminator when optimise generator

        for sample_rankings_per_q, binary_rele, presort in product(choice_sample_rankings_per_q, choice_binary_rele, choice_presort):
            for scale_data, scaler_id, scaler_level in product(choice_scale_data, choice_scaler_id, choice_scaler_level):
                '''
                corresponding to query-level context. But of point and pair models, there will be the case of a single document where we can not perform FBN, say a error like this
                Expected more than 1 value per channel when training, got input size torch.Size([1, 46])
                or 
                RuntimeError: weight should contain 5 elements not 46
                '''
                if model_id.endswith('Point') or model_id.endswith('Pair'):
                    FBN = False
                else:
                    #FBN = False if scale_data else True
                    FBN = False

                for partail_eval_dict in ad_eval_grid(choice_validation=choice_validation, choice_epoch=choice_epoch, choice_semi_context=choice_semi_context, choice_mask_ratios=choice_mask_ratios, choice_mask_type=choice_mask_type):
                    eval_dict.update(partail_eval_dict)

                    for sf_para_dict in sf_grid(FBN=FBN, query_aware=False, choice_cnt_strs=None, choice_layers=choice_layers, choice_hd_hn_af=choice_hd_hn_af, choice_tl_af=choice_tl_af,
                                                choice_hd_hn_tl_af=choice_hd_hn_tl_af, choice_apply_tl_af=choice_apply_tl_af, choice_apply_BN=choice_apply_BN, choice_apply_RD=choice_apply_RD):

                        data_dict.update(dict(sample_rankings_per_q=sample_rankings_per_q, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level, binary_rele=binary_rele, presort=presort))

                        for d_g_epoches, samples_per_query, ad_training_order in product(choice_d_g_epoches, choice_samples_per_query, choice_ad_training_order):
                            d_epoches, g_epoches = d_g_epoches

                            if model_id.startswith('IR_GAN') or model_id.startswith('IR_WGAN'):
                                for temperature in choice_temperatures:
                                    ad_para_dict = dict(model_id=model_id, d_epoches=d_epoches, g_epoches=g_epoches, samples_per_query=samples_per_query, temperature=temperature, ad_training_order=ad_training_order)

                                    if model_id == 'IR_GAN_List':
                                        for list_para_dict in ad_list_grid(model_id=model_id, choice_top_k=choice_top_k, choice_shuffle_ties=choice_shuffle_ties, choice_PL=choice_PL, choice_repTrick=choice_repTrick, choice_dropLog=choice_dropLog):
                                            ad_para_dict.update(list_para_dict)
                                            self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)

                                    elif model_id == 'IR_WGAN_List':
                                        for list_para_dict in ad_list_grid(model_id=model_id, choice_top_k=choice_top_k, choice_shuffle_ties=choice_shuffle_ties, choice_PL=choice_PL):
                                            ad_para_dict.update(list_para_dict)
                                            self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)

                                    elif model_id.endswith('Pair'):
                                        for loss_type_d in choice_losstype_d:
                                            ad_para_dict.update(dict(loss_type=loss_type_d))
                                            self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)
                                    else:
                                        self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)
                            else:
                                ad_para_dict = dict(model_id=model_id, d_epoches=d_epoches, g_epoches=g_epoches)
                                self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)


    def point_run(self, model_id=None, data_dict=None, eval_dict=None):

        debug = eval_dict['debug']
        do_log = True if eval_dict['debug'] else True

        epochs = 20 if debug else 100

        scale_data, scaler_id, scaler_level = self.get_scaler_setting(data_id=data_dict['data_id'])

        #FBN = False if scale_data else True
        FBN = False # leads to error like batchnorm.py"

        # more data settings that are rarely changed
        data_dict.update(dict(max_docs='All', min_docs=10, min_rele=1, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level))
        #data_dict.update(dict(max_docs='All', min_docs=None, min_rele=None, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level))

        # checking loss variation
        # do_vali, do_summary = False, True
        do_vali, do_summary = False, False
        #do_vali, do_summary = True, True
        log_step = 1

        # more evaluation settings that are rarely changed
        eval_dict.update(dict(cutoffs=[1, 3, 5, 10, 20], do_vali=do_vali, vali_k=5, do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False, epochs=epochs))

        sf_para_dict = dict()

        if eval_dict['query_aware']:  # to be deprecated
            pass

        elif model_id.endswith('MDNs'):
            pass

        elif model_id.endswith('MCNs'):
            pass

        else:
            sf_para_dict['id'] = 'ffnns'
            ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True, BN=False, RD=False, FBN=FBN)
            sf_para_dict['ffnns'] = ffnns_para_dict


        temperature = 0.2
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG'
        #ad_training_order = 'GD'

        if model_id.endswith('List'):
            ad_para_dict = dict(model_id=model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                temperature=temperature, ad_training_order=ad_training_order, samples_per_query=10,
                                top_k=2, shuffle_ties=False, PL=True, repTrick=False, dropLog=True)
        else:
            ad_para_dict = dict(model_id=model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                temperature=temperature, ad_training_order=ad_training_order, loss_type='svm')

        if 'IR_GAN_Point' == model_id or 'IR_WGAN_Point' == model_id:
            self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)

        else:
            self.cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, ad_para_dict=ad_para_dict)