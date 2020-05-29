#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 17/06/2018

"""Description

"""

import os
import sys
import datetime
import warnings
import numpy as np
from pathlib import Path
from itertools import product

import torch

from tensorboardX import SummaryWriter

from org.archive.utils.bigdata.BigPickle import pickle_save, pickle_load
from org.archive.data import data_ms
from org.archive.data.data_utils import get_data_loader

from org.archive.eval.eval_utils import tor_ndcg_at_ks, tor_ndcg_at_k

from org.archive.ranker.nn_blocks import get_rf_str
from org.archive.ranker.ranker import VanillaFFNNs, ContextAwareFFNNs, get_all_cnt_strs, distill_context

from org.archive.ranking.pointwise.rankMSE import RankMSE
from org.archive.ranking.pairwise.rankNet import RankNet
from org.archive.ranking.pairwise.lambdaRank import LambdaRank
from org.archive.ranking.listwise.rankCosine import RankCosine
from org.archive.ranking.listwise.listNet import ListNet
from org.archive.ranking.listwise.listMLE import ListMLE
from org.archive.ranking.listwise.appoxNDCG import AppoxNDCG
from org.archive.ranking.listwise.appoxNDCG import ApproxNDCG_OP
from org.archive.ranking.listwise.wassrank.wassRank import WassRank


from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device



""" Utils """

def get_wass_para_str(wass_para_dict):
    cost_type, smooth_type, norm_type = wass_para_dict['cost_type'], wass_para_dict['smooth_type'], wass_para_dict['norm_type']
    if 'ST' == smooth_type:
        smooth_str = '_'.join(['ST', norm_type])
    else:
        raise NotImplementedError

    if cost_type.startswith('Group'):
        gain_base, non_rele_gap, var_penalty = wass_para_dict['gain_base'], wass_para_dict['non_rele_gap'],  wass_para_dict['var_penalty']
        wass_ct_str = '_'.join([cost_type, '{:,g}'.format(non_rele_gap), '{:,g}'.format(gain_base), '{:,g}'.format(var_penalty)])
    else:
        wass_ct_str = cost_type

    wass_mode, sh_itr, lam = wass_para_dict['mode'], wass_para_dict['sh_itr'], wass_para_dict['lam']
    wass_paras_str = '_'.join([str(wass_ct_str), str(wass_mode), 'Lambda', '{:,g}'.format(lam), 'ShIter', str(sh_itr), smooth_str])

    return wass_paras_str

def to_output_str(list_scores=None, list_cutoffs=None, split_str=', '):
    list_str = []
    for i in range(len(list_scores)):
        list_str.append('nDCG@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)

def get_rf_ID(rf_para_dict=None):
    if rf_para_dict['query_aware']:
        in_para_dict, cnt_para_dict, com_para_dict = rf_para_dict['in_para_dict'], rf_para_dict['cnt_para_dict'], rf_para_dict['com_para_dict']
        in_num_layers, in_HD_AF, in_HN_AF, in_TL_AF = in_para_dict['num_layers'], in_para_dict['HD_AF'], in_para_dict['HN_AF'], in_para_dict['TL_AF']
        in_rf_str = get_rf_str(in_num_layers, in_HD_AF, in_HN_AF, in_TL_AF)

        cnt_num_layers, cnt_HD_AF, cnt_HN_AF, cnt_TL_AF = cnt_para_dict['num_layers'], cnt_para_dict['HD_AF'], cnt_para_dict['HN_AF'], cnt_para_dict['TL_AF']
        cnt_rf_str = get_rf_str(cnt_num_layers, cnt_HD_AF, cnt_HN_AF, cnt_TL_AF)

        com_num_layers, com_HD_AF, com_HN_AF, com_TL_AF = com_para_dict['num_layers'], com_para_dict['HD_AF'], com_para_dict['HN_AF'], com_para_dict['TL_AF']
        com_rf_str = get_rf_str(com_num_layers, com_HD_AF, com_HN_AF, com_TL_AF)

        rf_str = '_'.join([in_rf_str, cnt_rf_str, com_rf_str])

    else:
        if 'MDN' in rf_para_dict and rf_para_dict['MDN']:
            one_fits_all_dict = rf_para_dict['mu_para_dict']
        else:
            one_fits_all_dict = rf_para_dict['one_fits_all']

        num_layers, HD_AF, HN_AF, TL_AF = one_fits_all_dict['num_layers'], one_fits_all_dict['HD_AF'], one_fits_all_dict['HN_AF'], one_fits_all_dict['TL_AF']
        rf_str = get_rf_str(num_layers, HD_AF, HN_AF, TL_AF)

    return rf_str


def update_output_setting(common_para_dict=None, apxNDCG_para_dict=None, wass_para_dict=None, rf_para_dict=None):
    dataset, model, do_validation, root_output = common_para_dict['dataset'], common_para_dict['model'], common_para_dict['do_validation'], common_para_dict['dir_output']
    grid_search, min_docs, min_rele = common_para_dict['grid_search'], common_para_dict['min_docs'], common_para_dict['min_rele']
    num_overall_epochs, sample_times_per_q = common_para_dict['num_overall_epochs'], common_para_dict['sample_times_per_q']
    binary_rele = common_para_dict['binary_rele']

    print(' '.join(['\nStart {} on {} for ranking >>>'.format(model, dataset)]))

    if grid_search:
        if gpu:
            root_output = root_output + '_'.join(['gpu', 'grid', model]) + '/'
        else:
            root_output = root_output + '_'.join(['grid', model]) + '/'

        if not os.path.exists(root_output):
            os.makedirs(root_output)

    rf_str = get_rf_ID(rf_para_dict=rf_para_dict)
    para_setting_str = '_'.join(['RF', rf_str, 'Ep', str(num_overall_epochs), 'St', str(sample_times_per_q), 'Vd', str(do_validation), 'Md', str(min_docs), 'Mr', str(min_rele)])
    if binary_rele:
        file_prefix = '_'.join([model, dataset, 'BiRele', para_setting_str])
    else:
        file_prefix = '_'.join([model, dataset, para_setting_str])

    model_output = root_output + file_prefix + '/'  # model-specific outputs
    if model == 'WassRank':
        wass_paras_str = get_wass_para_str(wass_para_dict=wass_para_dict)
        model_output = model_output + wass_paras_str + '/'

    elif model == 'ApproxNDCG':
        apxNDCG_paras_str = '_'.join(['Alpha', str(apxNDCG_para_dict['apxNDCG_alpha'])])
        model_output = model_output + apxNDCG_paras_str + '/'

    if not os.path.exists(model_output):
        os.makedirs(model_output)
    return model_output


def log_max(cutoffs=None, max_cv_avg_scores=None, common_para_dict=None, apxNDCG_para_dict=None, wass_para_dict=None, rf_para_dict=None):
    dataset, model, do_validation, root_output = common_para_dict['dataset'], common_para_dict['model'], common_para_dict['do_validation'], common_para_dict['dir_output']
    min_docs, min_rele = common_para_dict['min_docs'], common_para_dict['min_rele']
    num_overall_epochs = common_para_dict['num_overall_epochs']
    binary_rele = common_para_dict['binary_rele']
    query_aware = rf_para_dict['query_aware']

    if gpu:
        root_output = root_output + '_'.join(['gpu', 'grid', model]) + '/'
    else:
        root_output = root_output + '_'.join(['grid', model]) + '/'

    rf_str = get_rf_ID(rf_para_dict=rf_para_dict)

    with open(file=root_output + '/' + dataset +'_max.txt', mode='w') as max_writer:
        para_setting_str = 'query_aware: ' + str(rf_para_dict['query_aware'])
        if rf_para_dict['query_aware']:
            para_setting_str = '\n'.join(['cnt_str: '+rf_para_dict['cnt_str'], para_setting_str])

        para_setting_str = '\n'.join([para_setting_str, 'RF: '+rf_str, 'Epoches: '+str(num_overall_epochs), 'Validation: '+str(do_validation), 'Min docs per train ranking: '+str(min_docs), 'Min rele docs per train ranking: '+str(min_rele)])

        para_setting_str = '\n'.join(['model: ' + model, 'dataset: ' + dataset, 'Binarize standard labels: ' + str(binary_rele), para_setting_str])

        if model == 'WassRank':
            cost_type, smooth_type, norm_type = wass_para_dict['cost_type'], wass_para_dict['smooth_type'], wass_para_dict['norm_type']
            wass_ct_str = '\n'.join(['smooth_type:'+smooth_type, 'norm_type: '+norm_type])
            if cost_type.startswith('Group'):
                gain_base, non_rele_gap, var_penalty = wass_para_dict['gain_base'], wass_para_dict['non_rele_gap'], wass_para_dict['var_penalty']
                wass_ct_str = '\n'.join([wass_ct_str, 'cost_type: '+cost_type, 'non_rele_gap: '+'{:,g}'.format(non_rele_gap), 'gain_base: '+'{:,g}'.format(gain_base), 'var_penalty: '+'{:,g}'.format(var_penalty)])
            else:
                wass_ct_str = '\n'.join([wass_ct_str, 'cost_type: '+cost_type])

            wass_mode, sh_itr, lam = wass_para_dict['mode'], wass_para_dict['sh_itr'], wass_para_dict['lam']
            wass_paras_str = '\n'.join([wass_ct_str, 'wass_mode: '+str(wass_mode), 'Lambda: '+'{:,g}'.format(lam), 'ShIter: '+str(sh_itr)])

            para_setting_str = '\n'.join([wass_paras_str, para_setting_str])

        elif model == 'ApproxNDCG':
            apxNDCG_paras_str = '_'.join(['Alpha', str(apxNDCG_para_dict['apxNDCG_alpha'])])
            para_setting_str = '\n'.join([apxNDCG_paras_str, para_setting_str])

        max_writer.write(para_setting_str + '\n\n')
        max_writer.write(to_output_str(max_cv_avg_scores, cutoffs))


def get_ranker(model, rf, **kwargs):
    if model.startswith('RankMSE'):  #pointwise
        ranker = RankMSE(ranking_function=rf)
    elif model.startswith('RankNet'):    #pairwise
        ranker = RankNet(ranking_function=rf)
    elif model.startswith('LambdaRank'):
        ranker = LambdaRank(ranking_function=rf)
    elif model.startswith('ListNet'):    #listwise
        ranker = ListNet(ranking_function=rf)
    elif model.startswith('ListMLE'):
        ranker = ListMLE(ranking_function=rf)
    elif model.startswith('RankCosine'):
        ranker = RankCosine(ranking_function=rf)
    elif model.startswith('ApproxNDCG'):
        ranker = AppoxNDCG(ranking_function=rf)
    elif model.startswith('WassRank'):
        ranker = WassRank(ranking_function=rf, wass_para_dict=kwargs['wass_para_dict'], dict_cost_mats=kwargs['dict_cost_mats'], dict_std_dists=kwargs['dict_std_dists'])
    else:
        raise NotImplementedError

    return ranker

def get_ranking_function(rf_para_dict=None, quasiMDNs=False):
    num_features = rf_para_dict['num_features']

    if rf_para_dict['query_aware']:
        in_para_dict = rf_para_dict['in_para_dict']
        cnt_para_dict = rf_para_dict['cnt_para_dict']
        com_para_dict = rf_para_dict['com_para_dict']

        in_para_dict['num_features'] = num_features
        in_para_dict['out_dim'] = 100

        cnt = len(rf_para_dict['cnt_str'].split('_'))
        cnt_para_dict['num_features'] = num_features * cnt
        cnt_para_dict['out_dim'] = 100

        com_para_dict['num_features'] = cnt_para_dict['out_dim']

        rf = ContextAwareFFNNs(in_para_dict=in_para_dict, cnt_para_dict=cnt_para_dict, com_para_dict=com_para_dict, cnt_str=rf_para_dict['cnt_str'])
        return rf

    else:
        one_fits_all_dict = rf_para_dict['one_fits_all']
        one_fits_all_dict['num_features'] = num_features
        rf = VanillaFFNNs(para_dict=one_fits_all_dict)

        return rf


def train_ranker(ranker, train_data, query_aware=False, **kwargs):
    '''	One-epoch train of the given ranker '''
    epoch_loss = torch.zeros(1).to(device) if gpu else torch.zeros(1)

    if query_aware:
        dict_query_cnts = kwargs['dict_query_cnts']
        for entry in train_data:
            tor_batch_rankings, tor_batch_stds, qid = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0), entry[2][0]  # remove the size 1 of dim=0 from loader itself

            if ranker.id == 'WassRank': cpu_tor_batch_std_label_vec = tor_batch_stds
            if gpu: tor_batch_rankings, tor_batch_stds = tor_batch_rankings.to(device), tor_batch_stds.to(device)

            query_context = None if (dict_query_cnts is None) else dict_query_cnts[qid]

            if ranker.id == 'WassRank':
                batch_loss = ranker.train(tor_batch_rankings, tor_batch_stds, qid=qid, cpu_tor_batch_std_label_vec=cpu_tor_batch_std_label_vec, query_context=query_context)
            else:
                batch_loss = ranker.train(tor_batch_rankings, tor_batch_stds, query_context=query_context)

            epoch_loss += batch_loss.item()
    else:
        for entry in train_data:
            tor_batch_rankings, tor_batch_stds, qid = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0), entry[2][0]  # remove the size 1 of dim=0 from loader itself

            if ranker.id == 'WassRank': cpu_tor_batch_std_label_vec = tor_batch_stds
            if gpu: tor_batch_rankings, tor_batch_stds = tor_batch_rankings.to(device), tor_batch_stds.to(device)

            if ranker.id == 'WassRank':
                batch_loss = ranker.train(tor_batch_rankings, tor_batch_stds, qid=qid, cpu_tor_batch_std_label_vec=cpu_tor_batch_std_label_vec)
            else:
                batch_loss = ranker.train(tor_batch_rankings, tor_batch_stds)

            epoch_loss += batch_loss.item()

    return epoch_loss


def kind_check(common_para_dict):
    dataset = common_para_dict['dataset']

    semi_datasets = ['MQ2007_semi', 'MQ2008_semi']

    if dataset in semi_datasets:
        assert True == common_para_dict['unknown_as_zero']
        s_key = 'binary_rele'
        mes = '{} is a sensitive setting w.r.t. {}, do you really mean {}'.format(s_key, dataset, common_para_dict[s_key])
        warnings.warn(message=mes)
    else:
        assert False == common_para_dict['unknown_as_zero']


def load_query_context(dir_data=None, cnt_str=None, train_data_loader=None, test_data_loader=None, vali_data_loader=None):
    buffered_file = dir_data + 'Buffer/' + cnt_str + '.cnt'
    if os.path.exists(buffered_file):
        print('Loaded ', buffered_file)
        return pickle_load(buffered_file)
    else:
        parent_dir = Path(buffered_file).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        dict_query_cnts = dict()

        for entry in train_data_loader:
            tor_batch_rankings, tor_batch_stds, qid = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0),  entry[2][0]  # remove the size 1 of dim=0 from loader itself
            if gpu: tor_batch_rankings, tor_batch_stds = tor_batch_rankings.to(device), tor_batch_stds.to(device)
            #print('tor_batch_rankings', tor_batch_rankings.size())
            batch_cnts = distill_context(tor_batch_rankings, cnt_str=cnt_str)
            dict_query_cnts[qid] = batch_cnts

        for entry in test_data_loader:
            tor_batch_rankings, tor_batch_stds, qid = entry[0], torch.squeeze(entry[1], dim=0),  entry[2][0]  # remove the size 1 of dim=0 from loader itself
            if gpu: tor_batch_rankings, tor_batch_stds = tor_batch_rankings.to(device), tor_batch_stds.to(device)

            #print('tor_batch_rankings', tor_batch_rankings.size())
            batch_cnts = distill_context(tor_batch_rankings, cnt_str=cnt_str)
            dict_query_cnts[qid] = batch_cnts

        for entry in vali_data_loader:
            tor_batch_rankings, tor_batch_stds, qid = entry[0], torch.squeeze(entry[1], dim=0),  entry[2][0]  # remove the size 1 of dim=0 from loader itself
            if gpu: tor_batch_rankings, tor_batch_stds = tor_batch_rankings.to(device), tor_batch_stds.to(device)

            batch_cnts = distill_context(tor_batch_rankings, cnt_str=cnt_str)
            dict_query_cnts[qid] = batch_cnts

        #
        pickle_save(dict_query_cnts, file=buffered_file)

        return dict_query_cnts


def cv_eval_l2r(common_para_dict=None, apxNDCG_para_dict=None, wass_para_dict=None, dict_cost_mats=None, dict_std_dists=None, rf_para_dict=None):
    if L2R_GLOBAL.global_gpu: print('-- GPU({}) is launched --'.format(L2R_GLOBAL.global_device))

    kind_check(common_para_dict)

    if common_para_dict['debug']:
        fold_num, num_overall_epochs = 2, 10
        common_para_dict['num_overall_epochs'] = num_overall_epochs
    else:
        fold_num = 5

    # common parameters across different models
    dataset, dir_data, model = common_para_dict['dataset'], common_para_dict['dir_data'], common_para_dict['model']
    query_level_scale, binary_rele, unknown_as_zero = common_para_dict['query_level_scale'], common_para_dict['binary_rele'], common_para_dict['unknown_as_zero']
    min_docs, min_rele, cutoffs = common_para_dict['min_docs'], common_para_dict['min_rele'], common_para_dict['cutoffs']
    do_validation, validation_k, do_summary, do_log = common_para_dict['do_validation'], common_para_dict['validation_k'], common_para_dict['do_summary'], common_para_dict['do_log']
    log_every, grid_search, use_epoch_loss = common_para_dict['log_every'], common_para_dict['grid_search'], common_para_dict['use_epoch_loss']
    num_overall_epochs, sample_times_per_q = common_para_dict['num_overall_epochs'], common_para_dict['sample_times_per_q']

    if dataset.endswith('_list'): assert sample_times_per_q == 1  # pre-check 1, where the standard ranking of each query is constant

    num_features, has_comment, multi_level_rele, max_rele_level = data_ms.get_data_meta(dataset=dataset)
    rf_para_dict['num_features'] = num_features

    model_output = update_output_setting(common_para_dict=common_para_dict, apxNDCG_para_dict=apxNDCG_para_dict, wass_para_dict=wass_para_dict, rf_para_dict=rf_para_dict)

    if do_log: # open log file
        sys.stdout = open(model_output + 'log.txt', "w")
    if do_summary:
        summary_writer = SummaryWriter(model_output + 'summary')
        avg_vali_eval_track = 0
        avg_test_eval_track = 0
        list_eoch_loss = []

    rf = get_ranking_function(rf_para_dict=rf_para_dict) # the object of ranking function

    if model.startswith('WassRank'):
        if wass_para_dict['mode'].startswith('Tor'):
            assert 1 == sample_times_per_q # todo Note: {Tor_WassLoss | Tor_WassLossSta} currently does not support the case of batch_size>1, which is to be solved.
        if (not grid_search): dict_cost_mats, dict_std_dists = dict(), dict()

        ranker = get_ranker(model, rf=rf, wass_para_dict=wass_para_dict, dict_cost_mats=dict_cost_mats, dict_std_dists=dict_std_dists)

    else:
        ranker = get_ranker(model, rf=rf)

    time_begin = datetime.datetime.now()        # timing
    l2r_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
    loaded_query_cnts, dict_query_cnts = False, None # query aware setting
    for fold_k in range(1, fold_num + 1):
        ranker.reset_parameters()   # reset with the same random initialization

        print('Fold-', fold_k)  # fold-wise data preparation plus certain light filtering
        dir_fold_k = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'

        train_data_loader = get_data_loader(original_file=file_train, has_comment=has_comment, query_level_scale=query_level_scale,
                                            min_docs=min_docs, min_rele=min_rele, need_pre_sampling=True, sample_times_per_q=sample_times_per_q, shuffle = True, batch_size = 1,
                                            unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

        test_data_loader = get_data_loader(original_file=file_test, has_comment=has_comment, query_level_scale=query_level_scale,
                                           min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1, binary_rele=binary_rele)
        if do_validation or do_summary:
            vali_data_loader = get_data_loader(original_file=file_vali, has_comment=has_comment, query_level_scale=query_level_scale,
                                               min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1, binary_rele=binary_rele)
            if do_validation:
                fold_optimal_ndcgk = 0.0
            if do_summary:
                # fold_k_batchloss_track = []
                fold_k_vali_eval_track = []
                fold_k_test_eval_track = []

        if not do_validation and use_epoch_loss:
            use_epoch_loss = True
            first_round = True
            threshold_epoch_loss = torch.from_numpy(np.asarray([10000000.0])).type(torch.FloatTensor).to(device)
        else:
            use_epoch_loss = False

        if rf_para_dict['query_aware'] and (not loaded_query_cnts):
            dict_query_cnts = load_query_context(dir_data=dir_data, cnt_str=rf_para_dict['cnt_str'],
                                                 train_data_loader=train_data_loader, test_data_loader=test_data_loader, vali_data_loader=vali_data_loader)
            loaded_query_cnts = True

        for epoch_k in range(1, num_overall_epochs + 1):
            tor_fold_k_epoch_k_loss = train_ranker(ranker=ranker, train_data=train_data_loader, query_aware=rf_para_dict['query_aware'], dict_query_cnts=dict_query_cnts)

            if (do_summary or do_validation) and (epoch_k % log_every == 0 or epoch_k == 1):    #stepwise check

                vali_eval_tmp = tor_ndcg_at_k(ranker=ranker, test_Qs=vali_data_loader, k=validation_k, multi_level_rele=multi_level_rele, query_aware=rf_para_dict['query_aware'], dict_query_cnts=dict_query_cnts)
                vali_eval_v = vali_eval_tmp.data.numpy()

                if do_summary:
                    fold_k_epoch_k_loss = tor_fold_k_epoch_k_loss.cpu().numpy() if gpu else tor_fold_k_epoch_k_loss.data.numpy()

                    print('-'.join(['\tFold', str(fold_k), 'Epoch', str(epoch_k)]), fold_k_epoch_k_loss)
                    list_eoch_loss.append(fold_k_epoch_k_loss)

                    fold_k_vali_eval_track.append(vali_eval_v)

                    test_eval_v = tor_ndcg_at_k(ranker=ranker, test_Qs=test_data_loader, k=validation_k, multi_level_rele=multi_level_rele, query_aware=rf_para_dict['query_aware'], dict_query_cnts=dict_query_cnts)
                    fold_k_test_eval_track.append(test_eval_v.data.numpy())

                if do_validation and epoch_k > 1:   # validation
                    curr_vali_ndcg = vali_eval_v
                    if (curr_vali_ndcg > fold_optimal_ndcgk) or (epoch_k == num_overall_epochs and curr_vali_ndcg == fold_optimal_ndcgk):   # we need at least a reference, in case all zero
                        print('\t', epoch_k, '- nDCG@k - ', curr_vali_ndcg)
                        fold_optimal_ndcgk = curr_vali_ndcg
                        fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                        fold_optimal_epoch_val = epoch_k
                        ranker.save_model(dir=model_output + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl') # buffer currently optimal model
                    else:
                        print('\t\t', epoch_k, '- nDCG@k - ', curr_vali_ndcg)

            elif use_epoch_loss:
                # stopping check via epoch-loss
                if first_round and tor_fold_k_epoch_k_loss >= threshold_epoch_loss:
                    print('Bad threshold: ', tor_fold_k_epoch_k_loss, threshold_epoch_loss)

                if tor_fold_k_epoch_k_loss < threshold_epoch_loss:
                    first_round = False
                    print('\tFold-', str(fold_k), ' Epoch-', str(epoch_k), 'Loss: ', tor_fold_k_epoch_k_loss)
                    threshold_epoch_loss = tor_fold_k_epoch_k_loss
                else:
                    print('\tStopped according epoch-loss!', tor_fold_k_epoch_k_loss, threshold_epoch_loss)
                    break

        if do_summary:  #track
            avg_vali_eval_track += np.asarray(fold_k_vali_eval_track)
            avg_test_eval_track += np.asarray(fold_k_test_eval_track)

        if do_validation:   # using the fold-wise optimal model for later testing based on validation data #
            buffered_model = '_'.join(['net_params_epoch', str(fold_optimal_epoch_val)]) + '.pkl'
            ranker.load_model(model_output + fold_optimal_checkpoint + '/' + buffered_model)
            fold_optimal_ranker = ranker

        else:   # buffer the model after a fixed number of training-epoches if no validation is deployed
            fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
            ranker.save_model(dir=model_output + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
            fold_optimal_ranker = ranker

        tor_fold_ndcg_ks = tor_ndcg_at_ks(ranker=fold_optimal_ranker, test_Qs=test_data_loader, ks=cutoffs, multi_level_rele=multi_level_rele, query_aware=rf_para_dict['query_aware'], dict_query_cnts=dict_query_cnts)
        fold_ndcg_ks = tor_fold_ndcg_ks.data.numpy()

        performance_list = [model + ' Fold-' + str(fold_k)] # fold-wise performance
        for i, co in enumerate(cutoffs):
            performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_ndcg_ks[i]))
        performance_str = '\t'.join(performance_list)
        print('\t', performance_str)

        l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks) #sum for later cv-performance

    time_end = datetime.datetime.now()  # overall timing
    elapsed_time_str = str(time_end - time_begin)
    print('Elapsed time:\t', elapsed_time_str + "\n")

    if do_summary:  #track
        avg_vali_eval_track /= fold_num
        avg_test_eval_track /= fold_num
        track_i = 0
        for epoch_k in range(1, num_overall_epochs + 1):
            if 0 == (epoch_k % log_every):
                summary_writer.add_scalar('Vali/Eval', avg_vali_eval_track[track_i], epoch_k)
                summary_writer.add_scalar('Test/Eval', avg_test_eval_track[track_i], epoch_k)
                track_i += 1
        summary_writer.close()

        pickle_save(avg_vali_eval_track, file=model_output + 'avg_vali_eval_track.np')
        pickle_save(avg_test_eval_track, file=model_output + 'avg_test_eval_track.np')
        pickle_save(list_eoch_loss, file=model_output + 'list_eoch_loss')


    print() # begin to print either cv or average performance
    l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
    if do_validation:
        eval_prefix = str(fold_num)+'-fold cross validation scores:'
    else:
        eval_prefix = str(fold_num) + '-fold average scores:'

    print(model, eval_prefix, to_output_str(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))

    return l2r_cv_avg_scores


def get_rf_itr(query_aware=None,
               choice_layers=None, choice_hd_hn_af=None, choice_tl_af=None, choice_apply_tl_af=None,
               in_choice_layers=None, in_choice_hd_hn_af=None, in_choice_tl_af=None,
               cnt_choice_layers=None, cnt_choice_hd_hn_af=None, cnt_choice_tl_af=None,
               com_choice_layers=None, com_choice_hd_hn_af=None, com_choice_tl_af=None,
               choice_cnt_strs=None):
    ''' get the iterator w.r.t. ranking function '''

    rf_itr = list()
    if query_aware:
        for in_num_layers, in_hd_hn_af, in_tl_af in product(in_choice_layers, in_choice_hd_hn_af, in_choice_tl_af):
            in_para_dict = dict(num_layers=in_num_layers, HD_AF=in_hd_hn_af, HN_AF=in_hd_hn_af, TL_AF=in_tl_af, apply_tl_af=True)

            for cnt_num_layers, cnt_hd_hn_af, cnt_tl_af in product(cnt_choice_layers, cnt_choice_hd_hn_af, cnt_choice_tl_af):
                cnt_para_dict = dict(num_layers=cnt_num_layers, HD_AF=cnt_hd_hn_af, HN_AF=cnt_hd_hn_af, TL_AF=cnt_tl_af, apply_tl_af=True)

                for com_num_layers, com_hd_hn_af, com_tl_af in product(com_choice_layers, com_choice_hd_hn_af, com_choice_tl_af):
                    com_para_dict = dict(num_layers=com_num_layers, HD_AF=com_hd_hn_af, HN_AF=com_hd_hn_af, TL_AF=com_tl_af, apply_tl_af=True)

                    for cnt_str in choice_cnt_strs:
                        rf_para_dict = dict()
                        rf_para_dict['query_aware'] = query_aware
                        rf_para_dict['cnt_str'] = cnt_str
                        rf_para_dict['in_para_dict'] = in_para_dict
                        rf_para_dict['cnt_para_dict'] = cnt_para_dict
                        rf_para_dict['com_para_dict'] = com_para_dict
                        rf_itr.append(rf_para_dict)
    else:
        for num_layers, hd_hn_af, tl_af, apply_tl_af in product(choice_layers, choice_hd_hn_af, choice_tl_af, choice_apply_tl_af):
            one_fits_all_dict = dict(num_layers=num_layers, HD_AF=hd_hn_af, HN_AF=hd_hn_af, TL_AF=tl_af, apply_tl_af=apply_tl_af)

            rf_para_dict = dict()
            rf_para_dict['query_aware'] = query_aware
            rf_para_dict['one_fits_all'] = one_fits_all_dict
            rf_itr.append(rf_para_dict)

    return rf_itr


def grid_run(debug=False, data=None, dir_data=None, dir_output=None, model=None, query_aware=False,
             do_summary = False, use_epoch_loss = False, unknown_as_zero = False):
    ''' perform learning-to-rank based on grid search of optimal parameter setting '''

    do_log = False if debug else True

    ''' settings that are rarely changed '''
    validation_k = 10
    min_docs = 10
    cutoffs = [1, 3, 5, 10, 20, 50]

    ''' setting w.r.t. data-preprocess '''
    ''' According to {Introducing {LETOR} 4.0 Datasets}, "QueryLevelNorm version: Conduct query level normalization based on data in MIN version. This data can be directly used for learning. We further provide 5 fold partitions of this version for cross fold validation". 
     --> Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}
     --> But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is ## not conducted yet##.
     --> For {Yahoo_L2R_Set_1 | Yahoo_L2R_Set_1 }, the query-level normalization is already conducted.
    '''
    query_level_scale = True if data.startswith('MSLRWEB') else False
    choice_binary_rele = [False] # True, False

    ''' setting w.r.t. train  '''
    choice_validation = [True] # True, False
    choice_epoch = [150]
    choice_samples = [1] # number of sample rankings per query

    """ setting w.r.t. ranking function """
    choice_layers = [2]     if debug else [1, 2, 4, 6, 8, 10]	#1, 2, 3, 4
    choice_hd_hn_af = ['S'] if debug else ['R'] # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
    choice_tl_af = ['S']    if debug else ['S'] # activation function for the last layer, sigmoid is suggested due to zero-prediction
    choice_apply_tl_af = [True] # True, False

    ''' query-aware setting '''
    in_choice_layers = [2]  if debug else [3]  # 1, 2, 3, 4
    in_choice_hd_hn_af = ['R'] if debug else ['R']
    in_choice_tl_af = ['S']    if debug else ['S']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers

    cnt_choice_layers = [2] if debug else [3]  # 1, 2, 3, 4
    cnt_choice_hd_hn_af = ['R'] if debug else ['R']
    cnt_choice_tl_af = ['S']   if debug else ['S']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers

    com_choice_layers = [2] if debug else [3]  # 1, 2, 3, 4
    com_choice_hd_hn_af = ['R'] if debug else ['R']
    com_choice_tl_af = ['S'] # sigmoid is suggested due to zero-prediction

    choice_cnt_strs = ['max_mean_var'] if debug else get_all_cnt_strs()


    """ setting w.r.t.  ApproxNDCG """
    apxNDCG_choice_alpha = [100] if debug else [50] # 100, 150, 200


    """ setting w.r.t. WassRank """
    ''' setting w.r.t. train '''
    wass_choice_mode = ['WassLossSta']   # WassLoss | WassLossSta | Tor_WassLoss | Tor_WassLossSta # todo Note: {Tor_WassLoss | Tor_WassLossSta} currently does not support the case of batch_size>1, which is to be solved.
    wass_choice_itr = [50]  # number of iterations w.r.t. sink-horn operation
    wass_choice_lam = [0.1] # 0.01 | 1e-3 | 1e-1 | 10  regularization parameter

    wass_cost_type = ['Group']  # 'CostAbs', 'CostSquare', 'Group'
    # member parameters of 'Group' include margin, div, group-base
    wass_choice_non_rele_gap = [100]    # the gap between a relevant document and an irrelevant document
    wass_choice_var_penalty = [np.e]    # variance penalty
    wass_choice_group_base = [4]    # the base for computing gain value

    wass_choice_smooth = ['ST'] # 'ST', i.e., ST: softmax | Gain, namely the way on how to get the normalized distribution histograms
    wass_choice_norm_pl = ['BothST']  # 'BothST': use ST for both prediction and standard labels

    dict_cost_mats, dict_std_dists = dict(), dict() # for buffering


    ''' select the best setting thourgh grid search '''
    max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
    k_index = cutoffs.index(validation_k)

    max_common_para_dict, max_wass_para_dict, max_apxNDCG_para_dict, max_rf_para_dict = None, None, None, None

    rf_itr = get_rf_itr(query_aware=query_aware,
                        choice_layers=choice_layers, choice_hd_hn_af=choice_hd_hn_af, choice_tl_af=choice_tl_af, choice_apply_tl_af=choice_apply_tl_af,
                        in_choice_layers=in_choice_layers, in_choice_hd_hn_af=in_choice_hd_hn_af, in_choice_tl_af=in_choice_tl_af,
                        cnt_choice_layers=cnt_choice_layers, cnt_choice_hd_hn_af=cnt_choice_hd_hn_af, cnt_choice_tl_af=cnt_choice_tl_af,
                        com_choice_layers=com_choice_layers, com_choice_hd_hn_af=com_choice_hd_hn_af, com_choice_tl_af=com_choice_tl_af,
                        choice_cnt_strs=choice_cnt_strs)

    for binary_rele in choice_binary_rele:
        for vd, num_epochs, sample_times_per_q in product(choice_validation, choice_epoch, choice_samples):
            for rf_para_dict in rf_itr:
                model_id = '_'.join([model, 'QAware', rf_para_dict['cnt_str']]) if query_aware else model

                common_para_dict = dict(debug=debug, dataset=data, dir_data=dir_data, dir_output=dir_output,
                                        query_level_scale=query_level_scale, binary_rele=binary_rele, unknown_as_zero=unknown_as_zero,
                                        model=model_id, min_docs=min_docs, min_rele=1, cutoffs=cutoffs,
                                        grid_search=True, do_validation=vd, validation_k=validation_k,
                                        do_summary=do_summary, do_log=do_log, log_every=2, use_epoch_loss=use_epoch_loss,
                                        num_overall_epochs=num_epochs, sample_times_per_q=sample_times_per_q)

                if model_id.startswith('WassRank'):
                    for mode, wsss_lambda, sinkhorn_itr in product(wass_choice_mode, wass_choice_lam, wass_choice_itr):
                        for wass_smooth, norm in product(wass_choice_smooth, wass_choice_norm_pl):
                            for cost_type in wass_cost_type:
                                if cost_type.startswith('Group'):
                                    for non_rele_gap, var_penalty, group_base in product(wass_choice_non_rele_gap, wass_choice_var_penalty, wass_choice_group_base):
                                        w_para_dict = dict(mode=mode, sh_itr=sinkhorn_itr, lam=wsss_lambda, cost_type=cost_type, smooth_type=wass_smooth, norm_type=norm,
                                                           gain_base=group_base, non_rele_gap=non_rele_gap, var_penalty=var_penalty)
                                        curr_cv_avg_scores = cv_eval_l2r(common_para_dict=common_para_dict, wass_para_dict=w_para_dict, dict_cost_mats=dict_cost_mats, dict_std_dists=dict_std_dists, rf_para_dict=rf_para_dict)
                                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                                            max_cv_avg_scores, max_common_para_dict, max_wass_para_dict, max_rf_para_dict = curr_cv_avg_scores, common_para_dict, w_para_dict, rf_para_dict
                                else:
                                    raise NotImplementedError
                elif model_id.startswith('ApproxNDCG'):
                    for alpha in apxNDCG_choice_alpha:
                        ApproxNDCG_OP.DEFAULT_ALPHA = alpha
                        apxNDCG_dict = dict(apxNDCG_alpha=alpha)
                        curr_cv_avg_scores = cv_eval_l2r(common_para_dict=common_para_dict, apxNDCG_para_dict=apxNDCG_dict, rf_para_dict=rf_para_dict)
                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_common_para_dict, max_apxNDCG_para_dict, max_rf_para_dict = curr_cv_avg_scores, common_para_dict, apxNDCG_dict, rf_para_dict

                else:  # other traditional methods
                    curr_cv_avg_scores = cv_eval_l2r(common_para_dict=common_para_dict, rf_para_dict=rf_para_dict)
                    if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                        max_cv_avg_scores, max_common_para_dict, max_rf_para_dict = curr_cv_avg_scores, common_para_dict, rf_para_dict

    #log max setting
    log_max(cutoffs=cutoffs, max_cv_avg_scores=max_cv_avg_scores, common_para_dict=max_common_para_dict, apxNDCG_para_dict=max_apxNDCG_para_dict, wass_para_dict=max_wass_para_dict, rf_para_dict=max_rf_para_dict)

def point_run(debug=False, model=None, query_aware=False, data=None, dir_data=None, dir_output=None,
              query_level_scale = False, binary_rele = False, unknown_as_zero = False, do_summary = False, sample_times_per_q=1):
    do_log = False if debug else True

    rf_para_dict = dict()
    rf_para_dict['query_aware'] = query_aware

    if query_aware:
        rf_para_dict['cnt_str'] = 'max_mean_var'
        in_para_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='R', apply_tl_af=True)
        cnt_para_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='R', apply_tl_af=True)
        com_para_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='R', apply_tl_af=True)
        rf_para_dict['in_para_dict'] = in_para_dict
        rf_para_dict['cnt_para_dict'] = cnt_para_dict
        rf_para_dict['com_para_dict'] = com_para_dict

    elif model.endswith('MDNs'):
        rf_para_dict['MDNs'] = True
        one_fits_all_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='', apply_tl_af=True, sep=False)
        rf_para_dict['one_fits_all'] = one_fits_all_dict

    else:
        one_fits_all_dict = dict(num_layers=3, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True)
        rf_para_dict['one_fits_all'] = one_fits_all_dict

    common_para_dict = dict(debug=debug, dataset=data, dir_data=dir_data, dir_output=dir_output,
                            query_level_scale=query_level_scale, binary_rele=binary_rele, unknown_as_zero=unknown_as_zero,
                            model=model, min_docs=10, min_rele=1, cutoffs=[1, 3, 5, 10, 20, 50],
                            do_validation=True, validation_k=10, do_summary=do_summary, do_log=do_log, log_every=2,
                            grid_search=False, use_epoch_loss=False, num_overall_epochs=200, sample_times_per_q=sample_times_per_q)

    if model.startswith('WassRank'):
        w_para_dict = dict(mode='WassLossSta', sh_itr=50, lam=10, cost_type='Group', smooth_type='ST', norm_type='BothST',
                           non_rele_gap=100, var_penalty=np.e, gain_base=4)
        cv_eval_l2r(common_para_dict=common_para_dict, wass_para_dict=w_para_dict, rf_para_dict=rf_para_dict)

    elif model.startswith('ApproxNDCG'):
        alpha = 100
        ApproxNDCG_OP.DEFAULT_ALPHA = alpha
        apxNDCG_dict = dict(apxNDCG_alpha=alpha)

        cv_eval_l2r(common_para_dict=common_para_dict, apxNDCG_para_dict=apxNDCG_dict, rf_para_dict=rf_para_dict)

    else:
        cv_eval_l2r(common_para_dict=common_para_dict, rf_para_dict=rf_para_dict)
