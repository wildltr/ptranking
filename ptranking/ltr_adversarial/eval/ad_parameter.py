#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
from itertools import product

from ptranking.eval.parameter import ScoringFunctionParameter
from ptranking.data import get_default_scaler_setting, MSLETOR_SEMI, get_data_meta

class AdScoringFunctionParameter(ScoringFunctionParameter):
	"""  """
	def __init__(self, debug=False, data_dict=None):
		super(AdScoringFunctionParameter, self).__init__(debug=debug, data_dict=data_dict)

	def default_para_dict(self):
		"""
		A default setting of the hyper-parameters of the stump neural scoring function for adversarial ltr.
		:return:
		"""
		ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True, BN=False, RD=False,
							   FBN=False) # FBN = True leads to error like batchnorm.py"
		sf_para_dict = dict()
		sf_para_dict['id'] = 'ffnns'
		sf_para_dict['ffnns'] = ffnns_para_dict

		self.sf_para_dict=sf_para_dict
		return self.sf_para_dict

	def grid_search(self):
		"""
		Iterator of settinging of the hyper-parameters of the stump neural scoring function for adversarial ltr
		:param debug:
		:return:
		"""
		choice_apply_BN = [False] if self.debug else [False]  # True, False
		choice_apply_RD = [False] if self.debug else [False]  # True, False

		choice_layers = [3] if self.debug else [3]  # 1, 2, 3, 4
		choice_hd_hn_af = ['S'] if self.debug else ['R']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
		choice_tl_af = ['S'] if self.debug else ['R']  # activation function for the last layer, sigmoid is suggested due to zero-prediction
		choice_hd_hn_tl_af = None

		choice_apply_tl_af = [True]  # True, False

		if choice_hd_hn_tl_af is not None:
			for BN, RD, num_layers, af, apply_tl_af in product(choice_apply_BN, choice_apply_RD, choice_layers,
															   choice_hd_hn_tl_af, choice_apply_tl_af):
				ffnns_para_dict = dict(FBN=False, BN=BN, RD=RD, num_layers=num_layers, HD_AF=af, HN_AF=af, TL_AF=af,
									   apply_tl_af=apply_tl_af)
				sf_para_dict = dict()
				sf_para_dict['id'] = 'ffnns'
				sf_para_dict['ffnns'] = ffnns_para_dict

				self.sf_para_dict = sf_para_dict
				yield sf_para_dict
		else:
			for BN, RD, num_layers, hd_hn_af, tl_af, apply_tl_af in product(choice_apply_BN, choice_apply_RD,
																			choice_layers, choice_hd_hn_af,
																			choice_tl_af, choice_apply_tl_af):
				ffnns_para_dict = dict(FBN=False, BN=BN, RD=RD, num_layers=num_layers, HD_AF=hd_hn_af, HN_AF=hd_hn_af,
									   TL_AF=tl_af, apply_tl_af=apply_tl_af)
				sf_para_dict = dict()
				sf_para_dict['id'] = 'ffnns'
				sf_para_dict['ffnns'] = ffnns_para_dict

				self.sf_para_dict = sf_para_dict
				yield sf_para_dict


class AdEvalSetting():
	"""
	Class object for evaluation settings w.r.t. adversarial training, etc.
	"""
	def __init__(self, debug=False, dir_output=None):
		self.debug = debug
		self.dir_output = dir_output

	def to_eval_setting_string(self, log=False):
		"""
		String identifier of eval-setting
		:param log:
		:return:
		"""
		eval_dict = self.eval_dict
		s1, s2 = (':', '\n') if log else ('_', '_')

		do_vali, epochs = eval_dict['do_validation'], eval_dict['epochs']

		eval_string = s2.join([s1.join(['epochs', str(epochs)]), s1.join(['do_validation', str(do_vali)])]) if log \
			else s1.join(['EP', str(epochs), 'V', str(do_vali)])

		return eval_string

	def default_setting(self):
		"""
		A default setting for evaluation when performing adversarial ltr
		:param debug:
		:param data_id:
		:param dir_output:
		:return:
		"""
		do_log = False if self.debug else True
		do_validation, do_summary = False, False
		log_step = 2
		epochs = 50
		vali_k = 5

		'''on the usage of mask_label
		(1) given a supervised dataset, True means that mask a supervised data to mimic unsupervised data
		(2) given an unsupervised dataset, this setting is not supported, since it is already an unsupervised data
		'''
		mask_label = False
		if mask_label:
			assert not self.data_id in MSLETOR_SEMI
			mask_ratio = 0.1
			mask_type = 'rand_mask_rele'
		else:
			mask_ratio = None
			mask_type = None

		# more evaluation settings that are rarely changed
		self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output,
						 cutoffs=[1, 3, 5, 10, 20], do_validation=do_validation, vali_k=vali_k,
						 do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False, epochs=epochs,
						 mask_label=mask_label, mask_ratio=mask_ratio, mask_type=mask_type)

		return self.eval_dict

	def grid_search(self):
		"""
		Iterator of settings for evaluation when performing adversarial ltr
		:param debug:
		:param dir_output:
		:return:
		"""
		''' common settings without grid-search '''
		vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]

		do_log = False if self.debug else True
		common_eval_dict = dict(debug=self.debug, grid_search=True, dir_output=self.dir_output,
						 vali_k=vali_k, cutoffs=cutoffs, do_log=do_log, log_step=2, do_summary=False, loss_guided=False)

		''' some settings for grid-search '''
		choice_validation = [False] if self.debug else [True]  # True, False
		choice_epoch = [20] if self.debug else [100]
		choice_mask_label = [False] if self.debug else [False]
		choice_mask_ratios = [0.2] if self.debug else [0.2, 0.4, 0.6, 0.8]  # 0.5, 1.0
		choice_mask_type = ['rand_mask_rele'] if self.debug else ['rand_mask_rele']

		for do_validation, num_epochs, mask_label in product(choice_validation, choice_epoch, choice_mask_label):
			if mask_label:
				for mask_ratio, mask_type in product(choice_mask_ratios, choice_mask_type):
					self.eval_dict = dict(do_validation=do_validation, epochs=num_epochs, mask_label=mask_label,
					                      mask_ratio=mask_ratio, mask_type=mask_type)
					self.eval_dict.update(common_eval_dict)
					yield self.eval_dict
			else:
				self.eval_dict =  dict(do_validation=do_validation, epochs=num_epochs, mask_label=mask_label)
				self.eval_dict.update(common_eval_dict)
				yield self.eval_dict



class AdDataSetting():
	"""
	Class object for data settings w.r.t. data loading and pre-process w.r.t. adversarial optimization
	"""
	def __init__(self, debug=False, data_id=None, dir_data=None):
		self.debug = debug
		self.data_id  = data_id
		self.dir_data = dir_data

	def to_data_setting_string(self, log=False):
		"""
		String identifier of data-setting
		:param log:
		:return:
		"""
		data_dict = self.data_dict
		s1, s2 = (':', '\n') if log else ('_', '_')

		data_id, binary_rele = data_dict['data_id'], data_dict['binary_rele']
		min_docs, min_rele, sample_rankings_per_q = data_dict['min_docs'], data_dict['min_rele'],\
													data_dict['sample_rankings_per_q']

		setting_string = s2.join([s1.join(['data_id', data_id]),
							s1.join(['min_docs', str(min_docs)]),
							s1.join(['min_rele', str(min_rele)]),
							s1.join(['sample_times_per_q', str(sample_rankings_per_q)])]) if log \
			else s1.join([data_id, 'MiD', str(min_docs), 'MiR', str(min_rele), 'S', str(sample_rankings_per_q)])


		if binary_rele:
			bi_str = s1.join(['binary_rele', str(binary_rele)]) if log else 'BiRele'
			setting_string = s2.join([setting_string, bi_str])

		return setting_string

	def default_setting(self):
		"""
		A default setting for data loading when performing adversarial ltr
		:return:
		"""
		unknown_as_zero = False
		binary_rele = False  # using the original values
		presort = False  # a default setting

		scale_data, scaler_id, scaler_level = get_default_scaler_setting(data_id=self.data_id)

		# more data settings that are rarely changed
		self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
						 sample_rankings_per_q=1, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele,
						 presort=presort, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		self.data_dict.update(data_meta)

		return self.data_dict

	def grid_search(self):
		"""
		Iterator of settings for data loading when performing adversarial ltr
		:param debug:
		:param data_id:
		:param dir_data:
		:return:
		"""
		''' common settings without grid-search '''
		binary_rele, unknown_as_zero = False, False
		common_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
								unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		common_data_dict.update(data_meta)

		''' some settings for grid-search '''
		choice_presort = [True] if self.debug else [True]
		choice_sample_rankings_per_q = [1] if self.debug else [1]  # number of sample rankings per query
		choice_scale_data, choice_scaler_id, choice_scaler_level = get_default_scaler_setting(data_id=self.data_id, grid_search=True)

		for scale_data, scaler_id, scaler_level, presort, sample_rankings_per_q in product(choice_scale_data,
																						   choice_scaler_id,
																						   choice_scaler_level,
																						   choice_presort,
																						   choice_sample_rankings_per_q):

			self.data_dict = dict(presort=presort, sample_rankings_per_q=sample_rankings_per_q,
								  scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)
			self.data_dict.update(common_data_dict)
			yield self.data_dict
