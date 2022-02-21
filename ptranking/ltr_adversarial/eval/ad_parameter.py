#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
The class of Parameter is designed as a wrapper of parameters of a model, a neural scoring function, etc.
For data loading and evaluation-related setting, the corresponding classes are DataSetting and EvalSetting.
The following classes are specially designed for adversarial ltr models, since the settings may differ a lot.
"""

import json
from itertools import product

from ptranking.ltr_adhoc.eval.parameter import EvalSetting, DataSetting, ScoringFunctionParameter
from ptranking.data.data_utils import get_scaler_setting, MSLETOR_SEMI, get_data_meta

class AdScoringFunctionParameter(ScoringFunctionParameter):
	"""  """
	def __init__(self, debug=False, sf_id=None, sf_json=None):
		super(AdScoringFunctionParameter, self).__init__(debug=debug, sf_id=sf_id, sf_json=sf_json)

	def default_pointsf_para_dict(self):
		"""
		A default setting of the hyper-parameters of the stump neural scoring function for adversarial ltr.
		"""
		self.sf_para_dict = dict()
		self.sf_para_dict['sf_id'] = self.sf_id
		self.sf_para_dict['opt'] = 'Adam'  # Adam | RMS | Adagrad
		self.sf_para_dict['lr'] = 0.001  # learning rate

		pointsf_para_dict = dict(num_layers=5, AF='R', TL_AF='R', apply_tl_af=True,
								 BN=False, bn_type='BN', bn_affine=True)
		self.sf_para_dict[self.sf_id] = pointsf_para_dict

		return self.sf_para_dict

	def default_listsf_para_dict(self):
		''' Not supported due to the inherent sampling mechanism '''
		return NotImplementedError


class AdEvalSetting(EvalSetting):
	"""
	Class object for evaluation settings w.r.t. adversarial training, etc.
	"""
	def __init__(self, debug=False, dir_output=None, ad_eval_json=None):
		super(AdEvalSetting, self).__init__(debug=debug, dir_output=dir_output, eval_json=ad_eval_json)

	def load_para_json(self, para_json):
		with open(para_json) as json_file:
			json_dict = json.load(json_file)["AdEvalSetting"]
		return json_dict

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
		do_validation, do_summary = True, False
		log_step = 1
		epochs = 10 if self.debug else 50
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
		if self.use_json:
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
			log_step = 1
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


class AdDataSetting(DataSetting):
	"""
	Class object for data settings w.r.t. data loading and pre-process w.r.t. adversarial optimization
	"""
	def __init__(self, debug=False, data_id=None, dir_data=None, ad_data_json=None):
		super(AdDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data, data_json=ad_data_json)

	def load_para_json(self, para_json):
		with open(para_json) as json_file:
			json_dict = json.load(json_file)["AdDataSetting"]
		return json_dict

	def to_data_setting_string(self, log=False):
		"""
		String identifier of data-setting
		:param log:
		:return:
		"""
		data_dict = self.data_dict
		s1, s2 = (':', '\n') if log else ('_', '_')

		data_id, binary_rele = data_dict['data_id'], data_dict['binary_rele']
		min_docs, min_rele, train_rough_batch_size, train_presort = data_dict['min_docs'], data_dict['min_rele'],\
													data_dict['train_rough_batch_size'], data_dict['train_presort']

		setting_string = s2.join([s1.join(['data_id', data_id]),
							s1.join(['min_docs', str(min_docs)]),
							s1.join(['min_rele', str(min_rele)]),
							s1.join(['TrBat', str(train_rough_batch_size)])]) if log \
			else s1.join([data_id, 'MiD', str(min_docs), 'MiR', str(min_rele), 'TrBat', str(train_rough_batch_size)])

		if train_presort:
			tr_presort_str = s1.join(['train_presort', str(train_presort)]) if log else 'TrPresort'
			setting_string = s2.join([setting_string, tr_presort_str])

		if binary_rele:
			bi_str = s1.join(['binary_rele', str(binary_rele)]) if log else 'BiRele'
			setting_string = s2.join([setting_string, bi_str])

		return setting_string

	def default_setting(self):
		"""
		A default setting for data loading when performing adversarial ltr
		"""
		unknown_as_zero = False
		binary_rele = False  # using the original values
		train_presort, validation_presort, test_presort = True, True, True
		train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 1, 100, 100
		scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id)

		# more data settings that are rarely changed
		self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
				unknown_as_zero=unknown_as_zero, binary_rele=binary_rele, train_presort=train_presort,
				validation_presort=validation_presort, test_presort=test_presort,
			    train_rough_batch_size=train_rough_batch_size, validation_rough_batch_size=validation_rough_batch_size,
			    test_rough_batch_size=test_rough_batch_size,
			    scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		if self.debug: data_meta['fold_num'] = 2
		self.data_dict.update(data_meta)

		return self.data_dict

	def grid_search(self):
		"""
		Iterator of settings for data loading when performing adversarial ltr
		"""
		if self.use_json:
			scaler_id = self.json_dict['scaler_id']
			choice_min_docs = self.json_dict['min_docs']
			choice_min_rele = self.json_dict['min_rele']
			choice_binary_rele = self.json_dict['binary_rele']
			choice_unknown_as_zero = self.json_dict['unknown_as_zero']
			base_data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"],
								  train_presort=True, test_presort=True, validation_presort=True,
								  train_rough_batch_size=1, validation_rough_batch_size=100, test_rough_batch_size=100)
		else:
			scaler_id = None
			choice_min_docs = [10]
			choice_min_rele = [1]
			choice_binary_rele = [False]
			choice_unknown_as_zero = [False]
			base_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data,
								  train_presort=True, test_presort=True, validation_presort=True,
								  train_rough_batch_size=1, validation_rough_batch_size=100, test_rough_batch_size=100)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		base_data_dict.update(data_meta)

		scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id, scaler_id=scaler_id)

		for min_docs, min_rele in product(choice_min_docs, choice_min_rele):
			threshold_dict = dict(min_docs=min_docs, min_rele=min_rele)

			for binary_rele, unknown_as_zero in product(choice_binary_rele, choice_unknown_as_zero):
				custom_dict = dict(binary_rele=binary_rele, unknown_as_zero=unknown_as_zero)
				scale_dict = dict(scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

				self.data_dict = dict()
				self.data_dict.update(base_data_dict)
				self.data_dict.update(threshold_dict)
				self.data_dict.update(custom_dict)
				self.data_dict.update(scale_dict)
				yield self.data_dict
