#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 17/06/2018

"""Description

"""

import os
import sys
import datetime
import numpy as np

import torch

from tensorboardX import SummaryWriter

from org.archive.utils.bigdata.BigPickle import pickle_save
from org.archive.utils.pytorch.extensions import encode_RK

from org.archive.data import data_ms
from org.archive.data.data_utils import get_data_loader

from org.archive.eval.eval_utils import tor_ndcg_at_ks, tor_ndcg_at_k

from org.archive.ranking.pointwise.rankMSE import RankMSE
from org.archive.ranking.pairwise.rankNet import RankNet
from org.archive.ranking.pairwise.lambdaRank import LambdaRank
from org.archive.ranking.listwise.rankCosine import RankCosine
from org.archive.ranking.listwise.listNet import ListNet
from org.archive.ranking.listwise.listMLE import ListMLE
from org.archive.ranking.listwise.appoxNDCG import AppoxNDCG
from org.archive.ranking.listwise.appoxNDCG import ApproxNDCG_OP

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
		gain_base, margin_cost, div_cost = wass_para_dict['gain_base'], wass_para_dict['margin_cost'],  wass_para_dict['div_cost']
		wass_ct_str = '_'.join([cost_type, '{:,g}'.format(margin_cost), '{:,g}'.format(gain_base), '{:,g}'.format(div_cost)])
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

def update_output_setting(common_para_dict=None, apxNDCG_para_dict=None, wass_para_dict=None):
	dataset, model, do_validation, root_output = common_para_dict['dataset'], common_para_dict['model'], common_para_dict['do_validation'], common_para_dict['dir_output']
	grid_search, min_docs, min_rele = common_para_dict['grid_search'], common_para_dict['min_docs'], common_para_dict['min_rele']
	num_overall_epochs, sample_times_per_q = common_para_dict['num_overall_epochs'], common_para_dict['sample_times_per_q']
	num_layers, HD_AF, HN_AF, TL_AF = common_para_dict['num_layers'], common_para_dict['HD_AF'], common_para_dict['HN_AF'], common_para_dict['TL_AF']
	scorer_type = common_para_dict['scorer_type']

	print(' '.join(['Start {} on {} for ranking with {} >>>'.format(model, dataset, scorer_type)]))

	if grid_search:
		if gpu:
			root_output = root_output + '_'.join(['gpu', 'grid', model]) + '/'
		else:
			root_output = root_output + '_'.join(['grid', model]) + '/'

		if not os.path.exists(root_output):
			os.makedirs(root_output)

	af_str = '.'.join([HD_AF, HN_AF, TL_AF])
	para_setting_str = '_'.join(['Hi', str(num_layers), 'Af', af_str, 'Ep', str(num_overall_epochs), 'St', str(sample_times_per_q), 'Vd', str(do_validation), 'Md', str(min_docs), 'Mr', str(min_rele)])
	file_prefix = '_'.join([model, scorer_type, dataset, para_setting_str])

	model_output = root_output + file_prefix + '/'  # model-specific outputs
	if model == 'ListWass':
		wass_paras_str = get_wass_para_str(wass_para_dict=wass_para_dict)
		model_output = model_output + wass_paras_str + '/'

	elif model == 'ApproxNDCG':
		apxNDCG_paras_str = '_'.join(['Alpha', str(apxNDCG_para_dict['apxNDCG_alpha'])])
		model_output = model_output + apxNDCG_paras_str + '/'

	if not os.path.exists(model_output):
		os.makedirs(model_output)
	return model_output

def get_ranker(model, f_para_dict):
	if model == 'RankMSE':  #pointwise
		ranker = RankMSE(f_para_dict)
	elif model == 'RankNet':    #pairwise
		ranker = RankNet(f_para_dict)
	elif model == 'LambdaRank':
		ranker = LambdaRank(f_para_dict)
	elif model == 'ListNet':    #listwise
		ranker = ListNet(f_para_dict)
	elif model == 'ListMLE':
		ranker = ListMLE(f_para_dict)
	elif model == 'RankCosine':
		ranker = RankCosine(f_para_dict)
	elif model == 'ApproxNDCG':
		ranker = AppoxNDCG(f_para_dict)
	else:
		raise NotImplementedError

	return ranker

def train_ranker(ranker, train_data):
	'''	One-epoch train of the given ranker '''
	epoch_loss = torch.zeros(1).to(device) if gpu else torch.zeros(1)
	for entry in train_data:
		tor_batch_rankings, tor_batch_stds = torch.squeeze(entry[0], dim=0), torch.squeeze(entry[1], dim=0)  # remove the size 1 of dim=0 from loader itself
		if gpu: tor_batch_rankings, tor_batch_std_label_vec = tor_batch_rankings.to(device), tor_batch_std_label_vec.to(device)

		batch_loss = ranker.train(tor_batch_rankings, tor_batch_stds)
		epoch_loss += batch_loss.item()

	return epoch_loss

def cv_eval_listwise(common_para_dict=None, apxNDCG_para_dict=None, wass_para_dict=None, wass_dict_cost_mats=None, wass_dict_std_dists=None):
	# common parameters across different models
	debug, dataset, dir_data, model = common_para_dict['debug'], common_para_dict['dataset'], common_para_dict['dir_data'], common_para_dict['model']
	min_docs, min_rele, cutoffs = common_para_dict['min_docs'], common_para_dict['min_rele'], common_para_dict['cutoffs']
	do_validation, validation_k, do_summary, do_log = common_para_dict['do_validation'], common_para_dict['validation_k'], common_para_dict['do_summary'], common_para_dict['do_log']
	log_every, grid_search, use_epoch_loss = common_para_dict['log_every'], common_para_dict['grid_search'], common_para_dict['use_epoch_loss']
	num_overall_epochs, sample_times_per_q = common_para_dict['num_overall_epochs'], common_para_dict['sample_times_per_q']
	num_layers, HD_AF, HN_AF, TL_AF = common_para_dict['num_layers'], common_para_dict['HD_AF'], common_para_dict['HN_AF'], common_para_dict['TL_AF']
	scorer_type = common_para_dict['scorer_type']

	#assert min_docs >= cutoffs[-1]  # pre-check 2, ensure possible evaluation
	if dataset.endswith('_list'): assert sample_times_per_q == 1  # pre-check 1, where the standard ranking of each query is constant

	if debug:
		fold_num, num_overall_epochs = 2, 10
		common_para_dict['num_overall_epochs'] = num_overall_epochs
	else:
		fold_num = 5

	num_features, has_comment, query_level_scale, multi_level_rele, _ = data_ms.get_data_meta(dataset=dataset)
	model_output = update_output_setting(common_para_dict=common_para_dict, apxNDCG_para_dict=apxNDCG_para_dict, wass_para_dict=wass_para_dict)
	if do_log: # open log file
		sys.stdout = open(model_output + 'log.txt', "w")
	if do_summary:
		summary_writer = SummaryWriter(model_output + 'summary')
		avg_vali_eval_track = 0
		avg_test_eval_track = 0
		list_eoch_loss = []

	if (not grid_search) and 'PointWassRank' == model:
		wass_dict_cost_mats, wass_dict_std_dists = dict(), dict()

	F2R = True if scorer_type == 'F2R' else False
	f_para_dict = dict(num_features=num_features, num_layers=num_layers, HD_AF=HD_AF, HN_AF=HN_AF, TL_AF=TL_AF, F2R=F2R)
	ranker = get_ranker(model=model, f_para_dict=f_para_dict)

	time_begin = datetime.datetime.now()        # timing
	l2r_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
	for fold_k in range(1, fold_num + 1):
		ranker.reset_parameters()   # reset with the same random initialization

		print('Fold-', fold_k)  # fold-wise data preparation plus certain light filtering
		dir_fold_k = dir_data + 'Fold' + str(fold_k) + '/'
		file_train, file_vali, file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'

		train_data_loader = get_data_loader(original_file=file_train, has_comment=has_comment, query_level_scale=query_level_scale,
											min_docs=min_docs, min_rele=min_rele, need_pre_sampling=True, sample_times_per_q=sample_times_per_q, shuffle = True, batch_size = 1)

		test_data_loader = get_data_loader(original_file=file_test, has_comment=has_comment, query_level_scale=query_level_scale,
										   min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1)
		if do_validation or do_summary:
			vali_data_loader = get_data_loader(original_file=file_vali, has_comment=has_comment, query_level_scale=query_level_scale,
											   min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1)
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

		for epoch_k in range(1, num_overall_epochs + 1):
			tor_fold_k_epoch_k_loss = train_ranker(ranker=ranker, train_data=train_data_loader)

			if (do_summary or do_validation) and (epoch_k % log_every == 0 or epoch_k == 1):    #stepwise check

				vali_eval_tmp = tor_ndcg_at_k(ranker=ranker, test_Qs=vali_data_loader, k=validation_k, multi_level_rele=multi_level_rele)
				vali_eval_v = vali_eval_tmp.data.numpy()

				if do_summary:
					fold_k_epoch_k_loss = tor_fold_k_epoch_k_loss.cpu().numpy() if gpu else tor_fold_k_epoch_k_loss.data.numpy()

					print('-'.join(['\tFold', str(fold_k), 'Epoch', str(epoch_k)]), fold_k_epoch_k_loss)
					list_eoch_loss.append(fold_k_epoch_k_loss)

					fold_k_vali_eval_track.append(vali_eval_v)

					test_eval_v = tor_ndcg_at_k(ranker=ranker, test_Qs=test_data_loader, k=validation_k, multi_level_rele=multi_level_rele)
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

				#if wass_eval:
				#	wd = wass_eval_at_k(ranker=nr_ranker, test_Qs=test_data_loader, k=20, TL_AF=TL_AF)
				#	print('\t\t', epoch_k, '- WD@k - ', wd)

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

		tor_fold_ndcg_ks = tor_ndcg_at_ks(ranker=fold_optimal_ranker, test_Qs=test_data_loader, ks=cutoffs, multi_level_rele=multi_level_rele)
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


def grid_run(model=None, data=None, dir_data=None, dir_output=None):
	vd = True
	do_summary = False
	use_epoch_loss = False
	validation_k = 10
	min_docs = 10
	cutoffs = [1, 3, 5, 10, 20]

	""" common hyper-parameters """
	choice_scorer_type = ['Com']  # 'Com', 'F2R', '', ''
	choice_epoch = [300]
	choice_layers = [3]	#1, 2, 3, 4
	choice_samples = [1]
	choice_hd_hn_af = ['S'] # 'R6' | 'RK' | 'S'
	choice_tl_af = ['S']

	""" ApxNDCG-specific hyper-parameters """
	apxNDCG_choice_alpha = [50, 100, 150, 200]

	""" wass-specific hyper-parameters """
	wass_choice_mode = ['Tor_WassLossSta']   # WassLoss | WassLossSta | Tor_WassLossSta
	wass_cost_type = ['Group']  # 'CostAbs', 'CostSquare', 'Group'pull
	# member parameters of 'Group' include margin, div, group-base
	wass_choice_margin = [100]
	wass_choice_div = [np.e]
	wass_choice_group_base = [4]

	wass_choice_lam = np.asarray([10, 20, 30, 1, 0.1, 0.01], dtype=np.float32)	# 1e-3 | 1e-1 | 10
	wass_choice_lam = torch.from_numpy(wass_choice_lam).type(torch.FloatTensor).to(device)

	wass_choice_itr = np.asarray([50], dtype=np.int)
	# wass_choice_itr = torch.from_numpy(wass_choice_itr).type(torch.FloatTensor).to(device)

	wass_choice_smooth = ['ST'] # 'ST', i.e., softmax
	wass_choice_norm_pl = ['BothST']  # 'BothST', 'PredSum'

	wass_dict_cost_mats, wass_dict_std_dists = dict(), dict()

	for num_epoch in choice_epoch:
		for num_layers in choice_layers:
			for num_sample in choice_samples:
				for hd_hn_af in choice_hd_hn_af:
					if hd_hn_af=='RK': hd_hn_af = encode_RK(k=data_ms.get_max_rele_level(data))

					for tl_af in choice_tl_af:
						if tl_af == 'RK': tl_af = encode_RK(k=data_ms.get_max_rele_level(data))

						for scorer_type in choice_scorer_type:
							common_para_dict = dict(debug=False, grid_search=True, dataset=data, dir_data=dir_data, dir_output=dir_output, model=model, min_docs=min_docs, min_rele=1, cutoffs=cutoffs,
							                        do_validation=vd, validation_k=validation_k, do_summary=do_summary, do_log=True, log_every=2, use_epoch_loss=use_epoch_loss,
							                        num_overall_epochs=num_epoch, sample_times_per_q=num_sample, num_layers=num_layers,
							                        HD_AF=hd_hn_af, HN_AF=hd_hn_af, TL_AF=tl_af, scorer_type=scorer_type)
							if model == 'ListWass':
								for mode in wass_choice_mode:
									for cost_type in wass_cost_type:
										if cost_type.startswith('Group'):
											for margin in wass_choice_margin:
												for div in wass_choice_div:
													for group_base in wass_choice_group_base:
														for lam_i, lam in enumerate(wass_choice_lam):
															for itr_j, itr in enumerate(wass_choice_itr):
																for smooth in wass_choice_smooth:
																	if 'ST' == smooth:
																		for norm in wass_choice_norm_pl:
																			w_para_dict = dict(mode=mode, sh_itr=itr, lam=lam, cost_type=cost_type, smooth_type=smooth, norm_type=norm,
																			                   gain_base=group_base, margin_cost=margin, div_cost=div)
																			cv_eval_listwise(common_para_dict=common_para_dict, wass_para_dict=w_para_dict,
																							 wass_dict_cost_mats=wass_dict_cost_mats, wass_dict_std_dists=wass_dict_std_dists)
																	else:
																		raise NotImplementedError
										else:
											raise NotImplementedError

							elif model == 'ApproxNDCG':
								for alpha in apxNDCG_choice_alpha:
									ApproxNDCG_OP.DEFAULT_ALPHA = alpha
									apxNDCG_dict = dict(apxNDCG_alpha=alpha)
									cv_eval_listwise(common_para_dict=common_para_dict, apxNDCG_para_dict=apxNDCG_dict)

							else: # other traditional methods
								cv_eval_listwise(common_para_dict=common_para_dict)


def point_run(model=None, data=None, dir_data=None, dir_output=None):
	do_log = False
	do_summary = False
	scorer_type = 'Com' # 'Com', 'F2R'
	hd_hn_af = 'R6' # 'R6' | 'RK' | 'S'
	tl_af = 'S'

	if hd_hn_af == 'RK': hd_hn_af = encode_RK(k=data_ms.get_max_rele_level(data))
	if tl_af == 'RK': tl_af = encode_RK(k=data_ms.get_max_rele_level(data))

	common_para_dict = dict(debug=True, dataset=data, dir_data=dir_data, dir_output=dir_output, model=model, min_docs=10, min_rele=1, cutoffs=[1, 3, 5, 10, 20, 50],
	                        do_validation=True, validation_k=10, do_summary=do_summary, do_log=do_log, log_every=2, grid_search=False, use_epoch_loss=False,
	                        num_overall_epochs=200, sample_times_per_q=1, num_layers=3, HD_AF=hd_hn_af, HN_AF=hd_hn_af, TL_AF=tl_af, scorer_type=scorer_type)
	if model == 'PointWassRank':
		apxNDCG_dict = None
		w_para_dict = dict(mode='Tor_WassLossSta', sh_itr=50, lam=10, cost_type='Group', smooth_type='ST', norm_type='BothST',
		                             gain_base=4, margin_cost=100, div_cost=np.e)
	else:
		w_para_dict, apxNDCG_dict = None, None
		if model == 'ApproxNDCG':
			alpha = 100
			ApproxNDCG_OP.DEFAULT_ALPHA = alpha
			apxNDCG_dict = dict(apxNDCG_alpha=alpha)

	cv_eval_listwise(common_para_dict=common_para_dict, wass_para_dict=w_para_dict, apxNDCG_para_dict=apxNDCG_dict)