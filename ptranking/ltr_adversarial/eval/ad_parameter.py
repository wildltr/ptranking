#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import json
from itertools import product

from ptranking.eval.parameter import ScoringFunctionParameter
from ptranking.data.data_utils import get_default_scaler_setting, MSLETOR_SEMI, get_data_meta

class AdScoringFunctionParameter(ScoringFunctionParameter):
	"""  """
	def __init__(self, debug=False, data_dict=None, sf_json=None):
		super(AdScoringFunctionParameter, self).__init__(debug=debug, data_dict=data_dict, sf_json=sf_json)

	def default_para_dict(self):
		"""
		A default setting of the hyper-parameters of the stump neural scoring function for adversarial ltr.
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
		"""
		if self.sf_json is not None:
			with open(self.sf_json) as json_file:
				json_dict = json.load(json_file)
				choice_BN = json_dict['BN']
				choice_RD = json_dict['RD']
				choice_layers = json_dict['layers']
				choice_apply_tl_af = json_dict['apply_tl_af']
				choice_hd_hn_tl_af = json_dict['hd_hn_tl_af']
		else:
			choice_BN = [False] if self.debug else [False]  # True, False
			choice_RD = [False] if self.debug else [False]  # True, False
			choice_layers = [3] if self.debug else [5]  # 1, 2, 3, 4
			choice_hd_hn_tl_af = ['S'] if self.debug else ['S']
			choice_apply_tl_af = [True]  # True, False

		for BN, RD, num_layers, af, apply_tl_af in product(
				choice_BN, choice_RD, choice_layers, choice_hd_hn_tl_af, choice_apply_tl_af):
			ffnns_para_dict = dict(
				FBN=False, BN=BN, RD=RD, num_layers=num_layers, HD_AF=af, HN_AF=af, TL_AF=af, apply_tl_af=apply_tl_af)
			sf_para_dict = dict()
			sf_para_dict['id'] = 'ffnns'
			sf_para_dict['ffnns'] = ffnns_para_dict
			self.sf_para_dict = sf_para_dict
			yield sf_para_dict


class AdEvalSetting():
	"""
	Class object for evaluation settings w.r.t. adversarial training, etc.
	"""
	def __init__(self, debug=False, dir_output=None, ad_eval_json=None):
		self.debug = debug
		if ad_eval_json is not None:
			self.ad_eval_json = ad_eval_json
			with open(self.ad_eval_json) as json_file:
				self.json_dict = json.load(json_file)
		else:
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
		epochs = 100
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
						 cutoffs=[1, 3, 5, 10, 20, 50], do_validation=do_validation, vali_k=vali_k,
						 do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False, epochs=epochs,
						 mask_label=mask_label, mask_ratio=mask_ratio, mask_type=mask_type)

		return self.eval_dict

	def grid_search(self):
		"""
		Iterator of settings for evaluation when performing adversarial ltr
		"""
		if self.ad_eval_json is not None:  # using json file
			dir_output = self.json_dict['dir_output']
			epochs = 5 if self.debug else self.json_dict['epochs']
			do_validation, vali_k = self.json_dict['do_validation'], self.json_dict['vali_k']
			cutoffs = self.json_dict['cutoffs']
			do_log, log_step = self.json_dict['do_log'], self.json_dict['log_step']
			do_summary = self.json_dict['do_summary']
			loss_guided = self.json_dict['loss_guided']
			mask_label = self.json_dict['mask']['mask_label']
			choice_mask_type = self.json_dict['mask']['mask_type']
			choice_mask_ratio = self.json_dict['mask']['mask_ratio']

			base_dict = dict(debug=False, grid_search=True, dir_output=dir_output)
		else:
			base_dict = dict(debug=self.debug, grid_search=True, dir_output=self.dir_output)
			epochs = 20 if self.debug else 100
			do_validation = False if self.debug else True  # True, False
			vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
			do_log = False if self.debug else True
			log_step = 2
			do_summary, loss_guided = False, False

			mask_label = False if self.debug else False
			choice_mask_type = ['rand_mask_rele']
			choice_mask_ratio = [0.2]

		self.eval_dict = dict(epochs=epochs, do_validation=do_validation, vali_k=vali_k, cutoffs=cutoffs,
							  do_log=do_log, log_step=log_step, do_summary=do_summary, loss_guided=loss_guided,
							  mask_label=mask_label)
		self.eval_dict.update(base_dict)

		if mask_label:
			for mask_type, mask_ratio in product(choice_mask_type, choice_mask_ratio):
				mask_dict = dict(mask_type=mask_type, mask_ratio=mask_ratio)
				self.eval_dict.update(mask_dict)
				yield self.eval_dict
		else:
			yield self.eval_dict


class AdDataSetting():
	"""
	Class object for data settings w.r.t. data loading and pre-process w.r.t. adversarial optimization
	"""
	def __init__(self, debug=False, data_id=None, dir_data=None, ad_data_json=None):
		if ad_data_json is not None:
			self.ad_data_json = ad_data_json
			with open(self.ad_data_json) as json_file:
				self.json_dict = json.load(json_file)
			self.data_id = self.json_dict["data_id"]
		else:
			self.debug = debug
			self.dir_data = dir_data
			self.data_id = data_id

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
		"""
		if self.ad_data_json is not None:  # using json file
			choice_presort = self.json_dict['presort']
			choice_min_docs = self.json_dict['min_docs']
			choice_min_rele = self.json_dict['min_rele']
			choice_binary_rele = self.json_dict['binary_rele']
			choice_unknown_as_zero = self.json_dict['unknown_as_zero']
			choice_sample_rankings_per_q = self.json_dict['sample_rankings_per_q']
			base_data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"])
		else:
			choice_min_docs = [10]
			choice_min_rele = [1]
			choice_presort = [True]
			choice_binary_rele = [False]
			choice_unknown_as_zero = [True]
			choice_sample_rankings_per_q = [1]  # number of sample rankings per query

			base_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		base_data_dict.update(data_meta)

		choice_scale_data, choice_scaler_id, choice_scaler_level = get_default_scaler_setting(data_id=self.data_id,
																							  grid_search=True)

		for min_docs, min_rele, sample_rankings_per_q in product(choice_min_docs, choice_min_rele,
																 choice_sample_rankings_per_q):
			threshold_dict = dict(min_docs=min_docs, min_rele=min_rele, sample_rankings_per_q=sample_rankings_per_q)

			for binary_rele, unknown_as_zero, presort in product(choice_binary_rele, choice_unknown_as_zero,
																 choice_presort):
				custom_dict = dict(binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, presort=presort)

				for scale_data, scaler_id, scaler_level in product(choice_scale_data, choice_scaler_id,
																   choice_scaler_level):
					scale_dict = dict(scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

					self.data_dict = dict()
					self.data_dict.update(base_data_dict)
					self.data_dict.update(threshold_dict)
					self.data_dict.update(custom_dict)
					self.data_dict.update(scale_dict)
					yield self.data_dict
