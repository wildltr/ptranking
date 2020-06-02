#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import os
import sys
import copy
import datetime
import numpy as np
from pathlib import Path
from itertools import product

import torch
#from tensorboardX import SummaryWriter

from org.archive.data import data_utils
from org.archive.data.data_utils import L2RDataset
from org.archive.ltr_adhoc.eval.eval_utils import ndcg_at_ks, ndcg_at_k
from org.archive.utils.bigdata.BigPickle import pickle_save, pickle_load
from org.archive.ltr_adhoc.eval.grid_utils import sf_grid, get_sf_ID, eval_grid

from org.archive.ltr_adhoc.pointwise.rank_mse          import RankMSE
from org.archive.ltr_adhoc.pairwise.ranknet           import RankNet
from org.archive.ltr_adhoc.listwise.lambdarank        import LambdaRank, lambda_para_iterator, get_lambda_para_str, get_default_lambda_para_dict
from org.archive.ltr_adhoc.listwise.rank_cosine        import RankCosine
from org.archive.ltr_adhoc.listwise.listnet           import ListNet
from org.archive.ltr_adhoc.listwise.st_listnet         import STListNet
from org.archive.ltr_adhoc.listwise.listmle           import ListMLE
from org.archive.ltr_adhoc.listwise.approxNDCG        import AppoxNDCG, get_apxndcg_paras_str
from org.archive.ltr_adhoc.listwise.wassrank.wassRank import WassRank, wassrank_para_iterator, get_wass_para_str


from org.archive.l2r_global import global_gpu as gpu, global_device as device, tensor


class L2REvaluator():
    """
    The class for evaluating different ltr_adhoc methods.
    """
    def __init__(self, id='LTR'):
        self.id = id
        # self.permutation_train = False # comparing the effect of using different relevance labels

    def show_infor(self, data_dict, model_para_dict):
        ''' Some tip information '''
        if gpu: print('-- GPU({}) is launched --'.format(device))

        if model_para_dict['model_id'].startswith('Virtual'):
            print(' '.join(['\nStart {}-{} on {} for ltr_adhoc >>>'.format(model_para_dict['model_id'], model_para_dict['metric'], data_dict['data_id'])]))
        else:
           print(' '.join(['\nStart {} on {} for ltr_adhoc >>>'.format(model_para_dict['model_id'], data_dict['data_id'])]))

    @staticmethod
    def get_scaler_setting(data_id, grid_search=False):
        # todo-as-note: setting w.r.t. data-preprocess
        ''' According to {Introducing {LETOR} 4.0 Datasets}, "QueryLevelNorm version: Conduct query level normalization based on data in MIN version. This data can be directly used for learning. We further provide 5 fold partitions of this version for cross fold validation".
         --> Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}
         --> But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is ## not conducted yet##.
         --> For {Yahoo_L2R_Set_1 | Yahoo_L2R_Set_1 }, the query-level normalization is already conducted.
        '''

        if grid_search:
            if data_id in data_utils.MSLRWEB:
                choice_scale_data   = [True]             # True, False
                choice_scaler_id    = ['StandardScaler']  # ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
                choice_scaler_level = ['QUERY']        # SCALER_LEVEL = ['QUERY', 'DATASET']
            else:
                choice_scale_data   = [False]
                choice_scaler_id    = [None]
                choice_scaler_level = [None]

            return choice_scale_data, choice_scaler_id, choice_scaler_level
        else:
            if data_id in data_utils.MSLRWEB:
                scale_data   = True
                scaler_id    = 'StandardScaler'  # ['MinMaxScaler', 'StandardScaler']
                scaler_level = 'QUERY'  # SCALER_LEVEL = ['QUERY', 'DATASET']
            else:
                scale_data   = False
                scaler_id    = None
                scaler_level = None

            return scale_data, scaler_id, scaler_level


    def pre_check(self, data_dict, eval_dict, sf_para_dict):
        if data_dict['sample_rankings_per_q']>1:
            if eval_dict['query_aware']:
                assert sf_para_dict['in_para_dict']   == False
                assert sf_para_dict['cnt_para_dict']  == False
                assert sf_para_dict['com_para_dict']  == False
            else:
                assert sf_para_dict['one_fits_all']['BN'] == False


    def get_files(self, data_dict, fold_k=1):
        ''' Load files which are prepared as k-fold validation format '''
        if data_dict['data_id'] in data_utils.YAHOO_L2R:
            file_train, file_vali, file_test = os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.train.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.valid.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.test.txt')
        else:
            print('Fold-', fold_k)
            fold_k_dir = data_dict['dir_data'] + 'Fold' + str(fold_k) + '/'
            file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        return file_train, file_vali, file_test


    def load_data(self, eval_dict, data_dict, fold_k, model_para_dict=None):
        ''' Load dataset '''
        file_train, file_vali, file_test = self.get_files(data_dict, fold_k=fold_k)

        sample_rankings_per_q = data_dict['sample_rankings_per_q']

        if model_para_dict is not None and 'em_label' in model_para_dict and model_para_dict['em_label']:
            train_data = L2RDataset(train=True, file=file_train, sample_rankings_per_q=sample_rankings_per_q,
                                    shuffle=True, data_dict=data_dict, hot=True)
        else:
            if eval_dict['semi_context']: # enabling masking data as required
                train_data = L2RDataset(train=True, file=file_train, sample_rankings_per_q=sample_rankings_per_q,
                                        shuffle=True, data_dict=data_dict, eval_dict=eval_dict)
            else:
                train_data = L2RDataset(train=True, file=file_train, sample_rankings_per_q=sample_rankings_per_q,
                                        shuffle=True, data_dict=data_dict)

        if data_dict['data_id'] in data_utils.MSLETOR_SEMI or eval_dict['semi_context']:
            tmp_data_dict = copy.deepcopy(data_dict)
            tmp_data_dict.update(dict(unknown_as_zero=False))
            data_dict = tmp_data_dict

        vali_data = None
        if data_dict['scale_data'] and 'DATASET' == data_dict['scaler_level']:
            train_scaler = train_data.get_scaler()

            test_data = L2RDataset(train=False, file=file_test, sample_rankings_per_q=sample_rankings_per_q,
                                   shuffle=False, data_dict=data_dict, given_scaler=train_scaler)

            if eval_dict['do_vali'] or eval_dict['do_summary']:
                vali_data = L2RDataset(train=False, file=file_vali, sample_rankings_per_q=sample_rankings_per_q,
                                       shuffle=False, data_dict=data_dict, given_scaler=train_scaler)
        else:
            test_data  = L2RDataset(train=False, file=file_test, sample_rankings_per_q=sample_rankings_per_q,
                                    shuffle=False, data_dict=data_dict)

            if eval_dict['do_vali'] or eval_dict['do_summary']:
                vali_data = L2RDataset(train=False, file=file_vali, sample_rankings_per_q=sample_rankings_per_q,
                                       shuffle=False, data_dict=data_dict)

        return train_data, test_data, vali_data

    def ini_ranker(self, sf_para_dict, model_para_dict, **kwargs):
        ''' Initialize a ranker given the specified settings'''
        model_id = model_para_dict['model_id']

        if model_id in 'RankMSE':  # - pointwise -
            ranker = RankMSE(sf_para_dict=sf_para_dict)

        elif model_id == 'RankNet':  # -- pairwise --
            ranker = RankNet(sf_para_dict=sf_para_dict)

        elif model_id == 'LambdaRank':  # -- listwise --
            ranker = LambdaRank(sf_para_dict=sf_para_dict, lambda_para_dict=model_para_dict)

        elif model_id == 'ListNet':
            ranker = ListNet(sf_para_dict=sf_para_dict)

        elif model_id == 'STListNet':
            ranker = STListNet(sf_para_dict=sf_para_dict, temperature=1.0)

        elif model_id == 'ListMLE':
            ranker = ListMLE(sf_para_dict=sf_para_dict)

        elif model_id == 'RankCosine':
            ranker = RankCosine(sf_para_dict=sf_para_dict)

        elif model_id == 'ApproxNDCG':
            ranker = AppoxNDCG(sf_para_dict=sf_para_dict, apxNDCG_para_dict=model_para_dict)

        elif model_id in ['WassRank', 'WassRankSP']:
            if model_id.endswith('SP'):
                ranker = WassRank(sf_para_dict=sf_para_dict, wass_para_dict=model_para_dict, dict_cost_mats=kwargs['dict_cost_mats'], dict_std_dists=kwargs['dict_std_dists'], sampling=True)
            else:
                ranker = WassRank(sf_para_dict=sf_para_dict, wass_para_dict=model_para_dict, dict_cost_mats=kwargs['dict_cost_mats'], dict_std_dists=kwargs['dict_std_dists'])

        else:
            raise NotImplementedError

        return ranker


    def load_ranker(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):
        ''' Load a ranker given the specified settings'''

        model_id = model_para_dict['model_id']
        if model_id in ['WassRank', 'WassRankSP']:
            ranker = self.ini_ranker(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict, dict_cost_mats=self.dict_cost_mats, dict_std_dists=self.dict_std_dists)

        else:
            ranker = self.ini_ranker(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict)

        return ranker

    def update_dir_root(self, grid_search, dir_output, model_para_dict):
        ''' Update the output directory when evaluating a particular ranker '''

        #date_prefix = datetime.datetime.now().strftime('%Y_%m_%d') # which may cause multiple directories for the same model due to date differences

        model_id = model_para_dict['model_id']
        if model_id.startswith('Virtual'):
            model_id = '_'.join([model_id, model_para_dict['metric']])

        if grid_search:
            dir_root = dir_output + '_'.join(['gpu', 'grid', model_id]) + '/' if gpu else dir_output + '_'.join(['grid', model_id]) + '/'
        else:
            dir_root = dir_output

        return dir_root


    def get_data_eval_str(self, data_dict=None, eval_dict=None, log=False):
        '''
        (1) Get the identifier string; (2) Get the log string of settings
        :param data_dict:
        :param eval_dict:
        :param log:
        :return:
        '''
        s1, s2 = (':', '\n') if log else ('_', '_')

        data_id, binary_rele = data_dict['data_id'], data_dict['binary_rele']
        max_docs, min_docs, min_rele, sample_rankings_per_q = data_dict['max_docs'], data_dict['min_docs'], data_dict['min_rele'], data_dict['sample_rankings_per_q']
        do_vali, epochs = eval_dict['do_vali'], eval_dict['epochs']

        data_str = s2.join([s1.join(['data_id', data_id]),
                            s1.join(['max_docs', str(max_docs)]),
                            s1.join(['min_docs', str(min_docs)]),
                            s1.join(['min_rele',  str(min_rele)]),
                            s1.join(['sample_times_per_q', str(sample_rankings_per_q)])]) if log \
                   else s1.join([data_id, 'MaD', str(max_docs), 'MiD', str(min_docs), 'MiR', str(min_rele), 'S', str(sample_rankings_per_q)])

        eval_str = s2.join([s1.join(['epochs', str(epochs)]),
                            s1.join(['do_vali', str(do_vali)])]) if log \
                   else s1.join(['EP', str(epochs), 'V', str(do_vali)])

        if binary_rele:
            bi_str = s1.join(['binary_rele', str(binary_rele)]) if log else 'BiRele'
            data_str = s2.join([data_str, bi_str])

        data_eval_str = s2.join([data_str, eval_str])

        return data_eval_str


    def setup_output(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):
        ''' Update output directory '''
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
            if model_id.startswith('Virtual'):
                file_prefix = '_'.join([model_id, model_para_dict['metric'], 'SF', sf_str, data_eval_str])
            else:
                file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join([file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join([file_prefix, 'DS', data_dict['scaler_id']])
        else:
            file_prefix = '_'.join([file_prefix, 'QS', 'BN'])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        if model_id == 'WassRank' or model_id == 'KOTRank':
            wass_paras_str = get_wass_para_str(ot_para_dict=model_para_dict)
            dir_run = dir_run + wass_paras_str + '/'

        elif model_id == 'ApproxNDCG':
            apxNDCG_paras_str = get_apxndcg_paras_str(model_para_dict=model_para_dict)
            dir_run = dir_run + apxNDCG_paras_str + '/'

        elif model_id == 'LambdaRank':
            lambda_paras_str = get_lambda_para_str(lambda_para_dict=model_para_dict)
            dir_run = dir_run + lambda_paras_str + '/'

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run


    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        ''' Summarize the evaluation setting '''
        data_meta = data_utils.get_data_meta(data_id=data_dict['data_id'])
        if eval_dict['debug']: data_meta['fold_num'], eval_dict['epochs'] = 2, 20  # for quick check
        if eval_dict['do_summary']: data_meta['fold_num'] = 1 # using the Fold1 only

        if data_dict['data_id']== 'IRGAN_Adhoc_Semi': data_meta['fold_num'] = 1

        data_dict.update(data_meta)

        self.data_dict = data_dict
        self.eval_dict = eval_dict

        if sf_para_dict['id'] == 'ffnns':
            sf_para_dict['ffnns'].update(dict(num_features=data_dict['num_features']))
        else:
            raise NotImplementedError

        self.dir_run  = self.setup_output(data_dict, eval_dict, sf_para_dict, model_para_dict)

        # for quick access of common evaluation settings
        self.fold_num = data_dict['fold_num']
        self.epochs, self.loss_guided              = eval_dict['epochs'], eval_dict['loss_guided']
        self.vali_k, self.log_step, self.cutoffs   = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        self.do_vali, self.do_summary, self.do_log = eval_dict['do_vali'], eval_dict['do_summary'], eval_dict['do_log']

        if self.do_log: sys.stdout = open(self.dir_run + 'log.txt', "w")
        #if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')

        #self.query_aware, self.dict_query_cnts = False, None
        #if eval_dict['query_aware']: # query aware setting, Note: this ony permits evaluation over a single dataset
        #    self.query_aware, self.loaded_query_cnts, self.dict_query_cnts = True, False, None

    def get_log_para_str(self, model_para_dict):
        ''' Get the log string of a particular ranker '''
        model_id = model_para_dict['model_id']

        if model_id == 'WassRank':
            para_setting_str = get_wass_para_str(ot_para_dict=model_para_dict, log=True)

        elif model_id == 'ApproxNDCG':
            para_setting_str = get_apxndcg_paras_str(model_para_dict=model_para_dict, log=True)

        elif model_id == 'LambdaRank':
            para_setting_str = get_lambda_para_str(lambda_para_dict=model_para_dict, log=True)

        else:
            para_setting_str = ''

        return para_setting_str


    def log_max(self, data_dict=None, max_cv_avg_scores=None, sf_para_dict=None,  eval_dict=None, log_para_str=None):
        ''' Log the best performance across grid search and the corresponding setting '''
        dir_root, cutoffs = eval_dict['dir_root'], eval_dict['cutoffs']
        data_id = data_dict['data_id']

        sf_str = get_sf_ID(sf_para_dict=sf_para_dict, log=True)

        data_eval_str = self.get_data_eval_str(data_dict, eval_dict, log=True)

        if eval_dict['query_aware']:
            sf_str = '\n'.join(['query_aware: True', sf_str])

        with open(file=dir_root + '/' + data_id + '_max.txt', mode='w') as max_writer:
            max_writer.write('\n\n'.join([data_eval_str, sf_str, log_para_str, self.result_to_str(max_cv_avg_scores, cutoffs)]))


    def train_ranker(self, ranker, train_data, model_para_dict=None, epoch_k=None, reranking=False):
        '''	One-epoch train of the given ranker '''
        epoch_loss = tensor([0.0])

        if 'em_label' in model_para_dict and model_para_dict['em_label']:
            for qid, batch_rankings, batch_stds, torch_batch_stds_hot, batch_cnts in train_data.iter_hot():  # _, [batch, ranking_size, num_features], [batch, ranking_size]
                if gpu: batch_rankings, batch_stds, torch_batch_stds_hot, batch_cnts = batch_rankings.to(device), batch_stds.to(device), torch_batch_stds_hot.to(device), batch_cnts.to(device)

                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid, torch_batch_stds_hot=torch_batch_stds_hot, batch_cnts=batch_cnts, epoch_k=epoch_k)

                if stop_training:
                    break
                else:
                    epoch_loss += batch_loss.item()

        else:
            for qid, batch_rankings, batch_stds in train_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
                if gpu: batch_rankings, batch_stds = batch_rankings.to(device), batch_stds.to(device)

                if reranking:
                    # in case the standard labels of the initial retrieval are all zeros providing no optimization information. Meanwhile, some models (e.g., lambdaRank) may fail to train
                    if torch.nonzero(batch_stds).size(0) <= 0:
                        continue

                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid, epoch_k=epoch_k)

                if stop_training:
                    break
                else:
                    epoch_loss += batch_loss.item()

        return epoch_loss, stop_training

    def get_ndcg_at_k(self, ranker=None, test_data=None, k=None, batch_mode=True):

        return ndcg_at_k(ranker=ranker, test_data=test_data, k=k, multi_level_rele=self.data_dict['multi_level_rele'], batch_mode=batch_mode)

    def get_ndcg_at_ks(self, ranker=None, test_data=None, ks=None, batch_mode=True):

        return ndcg_at_ks(ranker=ranker, test_data=test_data, ks=ks, multi_level_rele=self.data_dict['multi_level_rele'], batch_mode=batch_mode)


    def kfold_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        self.pre_check(data_dict, eval_dict, sf_para_dict)
        self.show_infor(data_dict, model_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict)

        model_id = model_para_dict['model_id']

        ranker   = self.load_ranker(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)

        time_begin = datetime.datetime.now()            # timing
        l2r_cv_avg_scores = np.zeros(len(self.cutoffs)) # fold average

        for fold_k in range(1, self.fold_num + 1): # evaluation over k-fold data
            ranker.reset_parameters()              # reset with the same random initialization

            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k, model_para_dict)

            if self.do_vali: fold_optimal_ndcgk = 0.0
            if self.do_summary: list_epoch_loss, list_fold_k_train_eval_track, list_fold_k_test_eval_track, list_fold_k_vali_eval_track = [], [], [], []
            if not self.do_vali and self.loss_guided: first_round, threshold_epoch_loss = True, tensor([10000000.0])

            #if self.query_aware and (not self.loaded_query_cnts): ## todo the query_aware setting will be disabled ##
            #    self.loaded_query_cnts = True
            #    self.load_query_context(eval_dict, cnt_str=sf_para_dict['cnt_str'], train_data=train_data, test_data=test_data, vali_data=vali_data)

            for epoch_k in range(1, self.epochs + 1):
                torch_fold_k_epoch_k_loss, stop_training = self.train_ranker(ranker=ranker, train_data=train_data, model_para_dict=model_para_dict, epoch_k=epoch_k)

                ranker.scheduler.step()  # adaptive learning rate with step_size=40, gamma=0.5

                if stop_training:
                    print('training is failed !')
                    break

                if (self.do_summary or self.do_vali) and (epoch_k % self.log_step == 0 or epoch_k == 1):  # stepwise check
                    if self.do_vali:     # per-step validation score
                        vali_eval_tmp = self.get_ndcg_at_k(ranker=ranker, test_data=vali_data, k=self.vali_k)
                        vali_eval_v = vali_eval_tmp.data.numpy()
                        if epoch_k > 1:  # further validation comparison
                            curr_vali_ndcg = vali_eval_v
                            if (curr_vali_ndcg > fold_optimal_ndcgk) or (epoch_k == self.epochs and curr_vali_ndcg == fold_optimal_ndcgk):  # we need at least a reference, in case all zero
                                print('\t', epoch_k, '- nDCG@{} - '.format(self.vali_k), curr_vali_ndcg)
                                fold_optimal_ndcgk = curr_vali_ndcg
                                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                                fold_optimal_epoch_val = epoch_k
                                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')  # buffer currently optimal model
                            else:
                                print('\t\t', epoch_k, '- nDCG@{} - '.format(self.vali_k), curr_vali_ndcg)

                    if self.do_summary:  # summarize per-step performance w.r.t. train, test
                        fold_k_epoch_k_train_ndcg_ks = self.get_ndcg_at_ks(ranker=ranker, test_data=train_data, ks=self.cutoffs)
                        np_fold_k_epoch_k_train_ndcg_ks = fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
                        list_fold_k_train_eval_track.append(np_fold_k_epoch_k_train_ndcg_ks)

                        fold_k_epoch_k_test_ndcg_ks  = self.get_ndcg_at_ks(ranker=ranker, test_data=test_data, ks=self.cutoffs)
                        np_fold_k_epoch_k_test_ndcg_ks  = fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
                        list_fold_k_test_eval_track.append(np_fold_k_epoch_k_test_ndcg_ks)

                        fold_k_epoch_k_loss = torch_fold_k_epoch_k_loss.cpu().numpy() if gpu else torch_fold_k_epoch_k_loss.data.numpy()
                        list_epoch_loss.append(fold_k_epoch_k_loss)

                        if self.do_vali:
                            list_fold_k_vali_eval_track.append(vali_eval_v)


                elif self.loss_guided:  # stopping check via epoch-loss
                    if first_round and torch_fold_k_epoch_k_loss >= threshold_epoch_loss:
                        print('Bad threshold: ', torch_fold_k_epoch_k_loss, threshold_epoch_loss)

                    if torch_fold_k_epoch_k_loss < threshold_epoch_loss:
                        first_round = False
                        print('\tFold-', str(fold_k), ' Epoch-', str(epoch_k), 'Loss: ', torch_fold_k_epoch_k_loss)
                        threshold_epoch_loss = torch_fold_k_epoch_k_loss
                    else:
                        print('\tStopped according epoch-loss!', torch_fold_k_epoch_k_loss, threshold_epoch_loss)
                        break

            if self.do_summary:  # track
                sy_prefix = '_'.join(['Fold', str(fold_k)])
                fold_k_train_eval = np.vstack(list_fold_k_train_eval_track)
                fold_k_test_eval  = np.vstack(list_fold_k_test_eval_track)
                pickle_save(fold_k_train_eval, file=self.dir_run + '_'.join([sy_prefix, 'train_eval.np']))
                pickle_save(fold_k_test_eval, file=self.dir_run + '_'.join([sy_prefix, 'test_eval.np']))

                fold_k_epoch_loss = np.hstack(list_epoch_loss)
                pickle_save((fold_k_epoch_loss, train_data.__len__()), file=self.dir_run + '_'.join([sy_prefix, 'epoch_loss.np']))
                if self.do_vali:
                    fold_k_vali_eval = np.hstack(list_fold_k_vali_eval_track)
                    pickle_save(fold_k_vali_eval, file=self.dir_run + '_'.join([sy_prefix, 'vali_eval.np']))

            if self.do_vali: # using the fold-wise optimal model for later testing based on validation data
                buffered_model = '_'.join(['net_params_epoch', str(fold_optimal_epoch_val)]) + '.pkl'
                ranker.load(self.dir_run + fold_optimal_checkpoint + '/' + buffered_model)
                fold_optimal_ranker = ranker
            else:            # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
                fold_optimal_ranker = ranker

            torch_fold_ndcg_ks = self.get_ndcg_at_ks(ranker=fold_optimal_ranker, test_data=test_data, ks=self.cutoffs)
            fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()

            performance_list = [model_id + ' Fold-' + str(fold_k)]      # fold-wise performance
            for i, co in enumerate(self.cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_ndcg_ks[i]))
            performance_str = '\t'.join(performance_list)
            print('\t', performance_str)

            l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks) # sum for later cv-performance

        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - time_begin)
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, self.fold_num)
        eval_prefix = str(self.fold_num) + '-fold cross validation scores:' if self.do_vali else str(self.fold_num) + '-fold average scores:'
        print(model_id, eval_prefix, self.result_to_str(list_scores=l2r_cv_avg_scores, list_cutoffs=self.cutoffs))  # print either cv or average performance

        return l2r_cv_avg_scores

    def basic_train(self, ranker, eval_dict, train_data=None, test_data=None, vali_data=None):
        ranker.reset_parameters()  # reset with the same random initialization

        assert train_data is not None
        assert test_data  is not None

        list_losses = []
        list_train_ndcgs = []
        list_test_ndcgs = []

        epochs, cutoffs = eval_dict['epochs'], eval_dict['cutoffs']

        for i in range(epochs):
            epoch_loss = torch.zeros(1).to(device) if gpu else torch.zeros(1)
            for qid, batch_rankings, batch_stds in train_data:
                if gpu: batch_rankings, batch_stds = batch_rankings.to(device), batch_stds.to(device)
                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid)
                epoch_loss += batch_loss.item()

            np_epoch_loss = epoch_loss.cpu().numpy() if gpu else epoch_loss.data.numpy()
            list_losses.append(np_epoch_loss)

            test_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=test_data, ks=cutoffs, multi_level_rele=True)
            np_test_ndcg_ks = test_ndcg_ks.data.numpy()
            list_test_ndcgs.append(np_test_ndcg_ks)

            train_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=train_data, ks=cutoffs, multi_level_rele=True)
            np_train_ndcg_ks = train_ndcg_ks.data.numpy()
            list_train_ndcgs.append(np_train_ndcg_ks)

        test_ndcgs = np.vstack(list_test_ndcgs)
        train_ndcgs = np.vstack(list_train_ndcgs)

        return list_losses, train_ndcgs, test_ndcgs


    def result_to_str(self, list_scores=None, list_cutoffs=None, split_str=', '):
        list_str = []
        for i in range(len(list_scores)):
            list_str.append('nDCG@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
        return split_str.join(list_str)

    def get_sorted_labels(self, max_rele_level):
        ar = np.arange(start=0, stop=max_rele_level + 1)
        #print(ar)
        ar[::-1].sort()

        return ar

    def parse_virtual_rank_id(self, id):
        metric = id.split('_')[1]
        return metric


    def get_default_para_dict(self, model_id):
        if model_id in ['WassRank', 'WassRankSP']: # EOTLossSta | WassLossSta ## p1 | p2 | eg | dg| ddg
            self.dict_cost_mats, self.dict_std_dists = dict(), dict()  # global buffering across a number of runs with different model parameters
            w_para_dict = dict(model_id=model_id, mode='WassLossSta', sh_itr=10, lam=0.1, cost_type='eg', smooth_type='ST',
                               norm_type='BothST', non_rele_gap=10, var_penalty=np.e, gain_base=4)
            return w_para_dict

        elif model_id == 'LambdaRank':
            lambda_para_dict = get_default_lambda_para_dict()
            return lambda_para_dict

        elif model_id == 'ApproxNDCG':
            apxNDCG_dict = dict(model_id=model_id, alpha=10)
            return apxNDCG_dict

        else:
            return dict(model_id=model_id)

    @staticmethod
    def get_default_dicts(data_id, dir_data=None, dir_output=None):
        debug = False
        grid_search = False
        query_aware = False

        # testing the effect of partially masking ground-truth labels with a specified ratio
        semi_context = False
        if semi_context:
            assert not data_id in data_utils.MSLETOR_SEMI
            mask_ratio = 0.5
            mask_type = 'rand_mask_rele'
        else:
            mask_ratio = None
            mask_type = None

        unknown_as_zero = True if data_id in data_utils.MSLETOR_SEMI else False

        binary_rele = False  # using the original values
        presort = True  # a default setting

        data_dict = dict(data_id=data_id, dir_data=dir_data, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele,
                         presort=presort, sample_rankings_per_q=1)

        eval_dict = dict(debug=debug, grid_search=grid_search, query_aware=query_aware, dir_output=dir_output,
                         semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)

        do_log = False if eval_dict['debug'] else True

        scale_data, scaler_id, scaler_level = L2REvaluator.get_scaler_setting(data_id=data_dict['data_id'])

        # more data settings that are rarely changed
        data_dict.update(dict(max_docs='All', min_docs=10, min_rele=1,
                              scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level))

        # checking loss variation
        do_vali, do_summary = False, False
        # do_vali, do_summary = True, False
        # do_vali, do_summary = True, True
        log_step = 2

        # more evaluation settings that are rarely changed
        eval_dict.update(dict(cutoffs=[1, 3, 5, 10, 20], do_vali=do_vali, vali_k=5, do_summary=do_summary,
                              do_log=do_log, log_step=log_step, loss_guided=False, epochs=10))

        return data_dict, eval_dict

    @staticmethod
    def get_default_sf_para_dict(data_dict, eval_dict):
        ##- setting w.r.t. the scoring function -##
        FBN = False if data_dict['scale_data'] else True
        sf_para_dict = dict()

        if eval_dict['query_aware']:  # to be deprecated
            sf_para_dict['id'] = 'ScoringFunction_CAFFNNs'
            # sf_para_dict['cnt_str'] = 'max_mean_var'
            in_para_dict = dict(num_layers=3, HD_AF='CE', HN_AF='CE', TL_AF='CE', apply_tl_af=True, BN=True, RD=False,
                                FBN=FBN)
            # cnt_para_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='R', apply_tl_af=True, BN=True, RD=False)
            cnt_para_dict = None
            com_para_dict = dict(num_layers=3, HD_AF='CE', HN_AF='CE', TL_AF='CE', apply_tl_af=True, BN=True, RD=False)
            sf_para_dict['in_para_dict'] = in_para_dict
            sf_para_dict['cnt_para_dict'] = cnt_para_dict
            sf_para_dict['com_para_dict'] = com_para_dict

        else:
            sf_para_dict['id'] = 'ffnns'
            ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True,
                                   BN=True, RD=False, FBN=FBN)
            sf_para_dict['ffnns'] = ffnns_para_dict

        return sf_para_dict


    def default_run(self, model_id=None, data_id=None, dir_data=None, dir_output=None):
        '''
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        '''

        data_dict, eval_dict = L2REvaluator.get_default_dicts(data_id=data_id, dir_data=dir_data, dir_output=dir_output)

        # model-specific parameter dict #
        model_para_dict = self.get_default_para_dict(model_id=model_id)

        sf_para_dict = L2REvaluator.get_default_sf_para_dict(data_dict=data_dict, eval_dict=eval_dict)

        self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)



    def grid_run(self, model_id=None, data_id=None, dir_data=None, dir_output=None):
        ''' Perform learning-to-rank based on grid search of optimal parameter setting '''

        ''' common setting w.r.t. datasets & evaluation'''
        debug = False

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

        unknown_as_zero = True if data_id in data_utils.MSLETOR_SEMI else False
        binary_rele     = True if data_id in data_utils.MSLETOR_SEMI else False

        presort = True # a default setting

        data_dict = dict(data_id=data_id, dir_data=dir_data, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele, presort=presort)

        eval_dict = dict(debug=debug, grid_search=True, query_aware=query_aware, dir_output=dir_output, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)

        debug = eval_dict['debug']
        query_aware = eval_dict['query_aware']
        do_log = False if debug else True

        # more data settings that are rarely changed
        data_dict.update(dict(max_docs='All', min_docs=10, min_rele=1))

        # more evaluation settings that are rarely changed
        vali_k = 5
        cutoffs = [1, 3, 5, 10, 20, 50]
        eval_dict.update(dict(vali_k=vali_k, cutoffs=cutoffs, do_log=do_log, log_step=2, do_summary=False, loss_guided=False))

        choice_scale_data, choice_scaler_id, choice_scaler_level = self.get_scaler_setting(data_id=data_dict['data_id'], grid_search=True)

        ''' setting w.r.t. train  '''
        choice_validation = [False] if debug else [True]  # True, False
        choice_epoch =      [20] if debug else [100]
        choice_sample_rankings_per_q =    [1] if debug else [1]  # number of sample rankings per query

        #choice_semi_context = [True] if debug else [True]
        #choice_presort      = [True] if debug else [True]

        choice_semi_context = [False] if debug else [False]
        choice_presort      = [True] if debug else [True]

        #choice_mask_ratios = None
        choice_mask_ratios = [0.2] if debug else [0.2, 0.4, 0.6, 0.8]  # 0.5, 1.0
        choice_mask_type = ['rand_mask_rele'] if debug else ['rand_mask_rele']

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
        choice_apply_BN = [False] if debug else [True]  # True, False
        choice_apply_RD = [False] if debug else [False]  # True, False

        choice_layers = [3]     if debug else [3]  # 1, 2, 3, 4
        choice_hd_hn_af = ['S'] if debug else ['R']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
        choice_tl_af = ['S']    if debug else ['R']  # activation function for the last layer, sigmoid is suggested due to zero-prediction
        choice_hd_hn_tl_af = ['R', 'CE'] if debug else ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S'] # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
        choice_apply_tl_af = [True]  # True, False


        ''' query-aware setting '''
        in_choice_layers = [2]     if debug else [3]  # 1, 2, 3, 4
        in_choice_hd_hn_af = ['R'] if debug else ['R']
        in_choice_tl_af = ['S']    if debug else ['S']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers

        cnt_choice_layers = [2]     if debug else [3]  # 1, 2, 3, 4
        cnt_choice_hd_hn_af = ['R'] if debug else ['R']
        cnt_choice_tl_af = ['S']    if debug else ['S']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers

        com_choice_layers = [2]     if debug else [3]  # 1, 2, 3, 4
        com_choice_hd_hn_af = ['R'] if debug else ['R']
        com_choice_tl_af = ['S']  # sigmoid is suggested due to zero-prediction

        #choice_cnt_strs = ['max_mean_var'] if debug else get_all_cnt_strs()
        choice_cnt_strs = None

        """ setting w.r.t.  ApproxNDCG """
        apxNDCG_choice_alpha = [100] if debug else [10]  # 100, 150, 200

        ### -WassRank- ###
        self.dict_cost_mats, self.dict_std_dists = dict(), dict()  # global buffering across a number of runs with different model parameters

        ''' select the best setting thourgh grid search '''
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_model_para_dict = None, None, None

        for partail_eval_dict in eval_grid(choice_validation=choice_validation, choice_epoch=choice_epoch, choice_semi_context=choice_semi_context, choice_mask_ratios=choice_mask_ratios, choice_mask_type=choice_mask_type):
            eval_dict.update(partail_eval_dict)

            for scale_data, scaler_id, scaler_level, presort, sample_rankings_per_q in product(choice_scale_data, choice_scaler_id, choice_scaler_level, choice_presort, choice_sample_rankings_per_q):
                FBN = False if scale_data else True

                for sf_para_dict in sf_grid(FBN=FBN, query_aware=query_aware, choice_cnt_strs=choice_cnt_strs,
                                            choice_layers=choice_layers, choice_hd_hn_af=choice_hd_hn_af, choice_tl_af=choice_tl_af,
                                            choice_hd_hn_tl_af=choice_hd_hn_tl_af, choice_apply_tl_af=choice_apply_tl_af,
                                            in_choice_layers=in_choice_layers, in_choice_hd_hn_af=in_choice_hd_hn_af,
                                            in_choice_tl_af=in_choice_tl_af,
                                            cnt_choice_layers=cnt_choice_layers, choice_apply_BN=choice_apply_BN, choice_apply_RD=choice_apply_RD,
                                            cnt_choice_hd_hn_af=cnt_choice_hd_hn_af, cnt_choice_tl_af=cnt_choice_tl_af,
                                            com_choice_layers=com_choice_layers,
                                            com_choice_hd_hn_af=com_choice_hd_hn_af, com_choice_tl_af=com_choice_tl_af):

                    model_id = '_'.join([model_id, 'QAware', sf_para_dict['cnt_str']]) if query_aware else model_id

                    data_dict.update(dict(presort=presort, sample_rankings_per_q=sample_rankings_per_q, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level))

                    if model_id == 'WassRank':
                        for w_para_dict in wassrank_para_iterator():
                            curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, model_para_dict=w_para_dict)

                            if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                                max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_model_para_dict, = curr_cv_avg_scores, sf_para_dict, eval_dict, w_para_dict

                    elif model_id == 'ApproxNDCG':
                        for alpha in apxNDCG_choice_alpha:
                            apxNDCG_dict = dict(model_id='ApproxNDCG', alpha=alpha)
                            curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, model_para_dict=apxNDCG_dict)

                            if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                                max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_model_para_dict = curr_cv_avg_scores, sf_para_dict, eval_dict, apxNDCG_dict

                    elif model_id == 'LambdaRank':
                        for lambda_para_dict in lambda_para_iterator(debug):
                            curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, model_para_dict=lambda_para_dict)

                            if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                                max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_model_para_dict = curr_cv_avg_scores, sf_para_dict, eval_dict, lambda_para_dict

                    else:  # other traditional methods
                        model_para_dict = dict(model_id=model_id)
                        curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict, model_para_dict=model_para_dict)

                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_eval_dict, max_sf_para_dict, max_model_para_dict = curr_cv_avg_scores, eval_dict, sf_para_dict, model_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, max_cv_avg_scores=max_cv_avg_scores, sf_para_dict=max_sf_para_dict, eval_dict=max_eval_dict,
                     log_para_str=self.get_log_para_str(max_model_para_dict))
