#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
The class of Parameter is designed as a wrapper of parameters of a model, a neural scoring function, etc.
For data loading and evaluation-related setting, the corresponding classes are DataSetting and EvalSetting.
"""

import json
from itertools import product

from ptranking.base.neural_utils import get_sf_str
from ptranking.data.data_utils import get_scaler_setting, get_data_meta

class Parameter(object):
    """
    An abstract class for parameter
    """
    def __init__(self):
        pass

    def grid_search(self): # Iterator of parameter setting via grid-search
        pass

    def load_para_json(self, para_json):
        """ load json file of parameter setting """
        with open(para_json) as json_file:
            json_dict = json.load(json_file)
        return json_dict


class ModelParameter(Parameter):
    """
    A simple class for model parameter
    """
    def __init__(self, model_id=None, para_json=None):
        super(ModelParameter, self).__init__()
        self.model_id = model_id
        if para_json is None:
            self.use_json = False
        else:
            self.use_json = True
            self.json_dict = self.load_para_json(para_json=para_json)

    def default_para_dict(self):
        """
        The default parameter setting.
        :return:
        """
        return dict(model_id=self.model_id)

    def to_para_string(self, log=False, given_para_dict=None):
        """
        The string identifier of parameters
        :return:
        """
        return ''

    def grid_search(self):
        """
        Iterator of parameter setting for grid-search
        :return:
        """
        yield dict(model_id=self.model_id)


class ScoringFunctionParameter(ModelParameter):
    """
    The parameter class w.r.t. a neural scoring fuction
    """
    def __init__(self, debug=False, data_dict=None, model_id='ffnns', sf_json=None):
        super(ScoringFunctionParameter, self).__init__(model_id=model_id, para_json=sf_json)
        self.debug = debug
        self.data_dict = data_dict

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["SFParameter"]
        return json_dict

    def default_para_dict(self):
        """
        A default setting of the hyper-parameters of the stump neural scoring function.
        :param data_dict:
        :return:
        """
        assert self.data_dict is not None
        FBN = False if self.data_dict['scale_data'] else True # for feature normalization

        # feed-forward neural networks
        ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True, BN=True, RD=False, FBN=FBN)

        sf_para_dict = dict()
        sf_para_dict['id'] = self.model_id
        sf_para_dict[self.model_id] = ffnns_para_dict

        self.sf_para_dict = sf_para_dict
        return sf_para_dict

    def grid_search(self, data_dict=None):
        """
        Iterator of hyper-parameters of the stump neural scoring function.
        """
        assert data_dict is not None
        FBN = False if data_dict['scale_data'] else True  # for feature normalization
        if self.use_json:
            choice_BN = self.json_dict['BN']
            choice_RD = self.json_dict['RD']
            choice_layers = self.json_dict['layers']
            choice_apply_tl_af = self.json_dict['apply_tl_af']
            choice_hd_hn_tl_af = self.json_dict['hd_hn_tl_af']
        else:
            choice_BN = [False] if self.debug else [True]  # True, False
            choice_RD = [False] if self.debug else [False]  # True, False
            choice_layers = [3]     if self.debug else [5]  # 1, 2, 3, 4
            choice_hd_hn_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S'] # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_apply_tl_af = [True]  # True, False

        for BN, RD, num_layers, af, apply_tl_af in product(choice_BN, choice_RD, choice_layers,
                                                           choice_hd_hn_tl_af, choice_apply_tl_af):
            ffnns_para_dict = dict(
                FBN=FBN, BN=BN, RD=RD, num_layers=num_layers, HD_AF=af, HN_AF=af, TL_AF=af, apply_tl_af=apply_tl_af)
            sf_para_dict = dict()
            sf_para_dict['id'] = 'ffnns'
            sf_para_dict['ffnns'] = ffnns_para_dict
            self.sf_para_dict = sf_para_dict
            yield sf_para_dict


    def to_para_string(self, log=False):
        ''' Get the identifier of scoring function '''
        s1, s2 = (':', '\n') if log else ('_', '_')
        sf_para_dict = self.sf_para_dict

        if sf_para_dict['id'] in ['ScoringFunction_MDNs', 'ScoringFunction_QMDNs']:
            nn_para_dict = sf_para_dict['mu_para_dict']
        else:
            nn_para_dict = sf_para_dict['ffnns']

        num_layers, HD_AF, HN_AF, TL_AF, BN, RD, FBN = nn_para_dict['num_layers'], nn_para_dict['HD_AF'], \
                                                       nn_para_dict['HN_AF'], nn_para_dict['TL_AF'], \
                                                       nn_para_dict['BN'], nn_para_dict['RD'], nn_para_dict['FBN']

        if not nn_para_dict['apply_tl_af']: TL_AF = 'No'

        if log:
            rf_str = s2.join([s1.join(['FeatureBN', str(FBN)]), s1.join(['BN', str(BN)]),
                              s1.join(['num_layers', str(num_layers)]), s1.join(['RD', str(RD)]),
                              s1.join(['HD_AF', HD_AF]), s1.join(['HN_AF', HN_AF]), s1.join(['TL_AF', TL_AF])])
        else:
            if num_layers > 1 or nn_para_dict['apply_tl_af']:
                rf_str = get_sf_str(num_layers, HD_AF, HN_AF, TL_AF)
            elif 1 == num_layers and nn_para_dict['apply_tl_af'] is not True:
                rf_str = get_sf_str(num_layers, HD_AF+'(N)', HN_AF, TL_AF)

            if BN:  rf_str += '_BN'
            if RD:  rf_str += '_RD'
            if FBN: rf_str += '_FBN'

        return rf_str


class EvalSetting(Parameter):
    """
    Class object for evaluation settings w.r.t. training, etc.
    """
    def __init__(self, debug=False, dir_output=None, eval_json=None):
        self.debug = debug
        if eval_json is None:
            self.use_json = False
            self.dir_output = dir_output
        else:
            self.use_json = True
            self.json_dict = self.load_para_json(para_json=eval_json)
            self.dir_output = self.json_dict["dir_output"]

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["EvalSetting"]
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
        A default setting for evaluation
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        do_log = False if self.debug else True
        do_validation, do_summary = True, False  # checking loss variation
        log_step = 2
        epochs = 20 if self.debug else 100
        vali_k = 5

        ''' setting for exploring the impact of randomly removing some ground-truth labels '''
        mask_label = False
        mask_type = 'rand_mask_all'
        mask_ratio = 0.2

        # more evaluation settings that are rarely changed
        self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output,
                              cutoffs=[1, 3, 5, 10, 20, 50], do_validation=do_validation, vali_k=vali_k,
                              do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False, epochs=epochs,
                              mask_label=mask_label, mask_type=mask_type, mask_ratio=mask_ratio)

        return self.eval_dict

    def set_validation_k_and_cutoffs(self, vali_k=None, cutoffs=None):
        self.eval_dict['vali_k'] = vali_k
        self.eval_dict['cutoffs'] = cutoffs

    def check_consistence(self, vali_k=None, cutoffs=None):
        return (self.eval_dict['vali_k'] == vali_k) and (self.eval_dict['cutoffs'] == cutoffs)

    def grid_search(self):
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            epochs = 5 if self.debug else self.json_dict['epochs'] # debug is added for a quick check
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
            choice_mask_type = ['rand_mask_all']
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


class DataSetting(Parameter):
    """
    Class object for data settings w.r.t. data loading and pre-process.
    """
    def __init__(self, debug=False, data_id=None, dir_data=None, data_json=None):
        self.debug = debug

        if data_json is None:
            self.use_json = False
            self.data_id = data_id
            self.dir_data = dir_data
        else:
            self.use_json = True
            self.json_dict = self.load_para_json(para_json=data_json)
            self.data_id = self.json_dict["data_id"]
            self.dir_data = self.json_dict["dir_data"]

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["DataSetting"]
        return json_dict

    def to_data_setting_string(self, log=False):
        """
        String identifier of data-setting
        """
        data_dict = self.data_dict
        s1, s2 = (':', '\n') if log else ('_', '_')

        data_id, binary_rele = data_dict['data_id'], data_dict['binary_rele']
        min_docs, min_rele, train_batch_size, train_presort = data_dict['min_docs'], data_dict['min_rele'], \
                                                              data_dict['train_batch_size'], data_dict['train_presort']

        setting_string = s2.join([s1.join(['data_id', data_id]),
                                  s1.join(['min_docs', str(min_docs)]),
                                  s1.join(['min_rele', str(min_rele)]),
                                  s1.join(['TrBat', str(train_batch_size)])]) if log \
            else s1.join([data_id, 'MiD', str(min_docs), 'MiR', str(min_rele), 'TrBat', str(train_batch_size)])

        if train_presort:
            tr_presort_str = s1.join(['train_presort', str(train_presort)]) if log else 'TrPresort'
            setting_string = s2.join([setting_string, tr_presort_str])

        if binary_rele:
            bi_str = s1.join(['binary_rele', str(binary_rele)]) if log else 'BiRele'
            setting_string = s2.join([setting_string, bi_str])

        return setting_string

    def default_setting(self):
        """
        A default setting for data loading
        :return:
        """
        scaler_id = None
        unknown_as_zero = False # using original labels, e.g., w.r.t. semi-supervised dataset
        binary_rele = False  # using original labels
        train_presort, validation_presort, test_presort = True, True, True
        train_batch_size, validation_batch_size, test_batch_size = 1, 1, 1
        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id, scaler_id=scaler_id)

        # more data settings that are rarely changed
        self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
                              scale_data = scale_data, scaler_id = scaler_id, scaler_level = scaler_level,
                              train_presort=train_presort, validation_presort=validation_presort, test_presort=test_presort,
                              train_batch_size=train_batch_size, validation_batch_size=validation_batch_size,
                              test_batch_size=test_batch_size, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

        data_meta = get_data_meta(data_id=self.data_id) # add meta-information
        self.data_dict.update(data_meta)

        return self.data_dict

    def grid_search(self):
        if self.use_json:
            scaler_id = self.json_dict['scaler_id']
            choice_min_docs = self.json_dict['min_docs']
            choice_min_rele = self.json_dict['min_rele']
            choice_binary_rele = self.json_dict['binary_rele']
            choice_unknown_as_zero = self.json_dict['unknown_as_zero']
            choice_train_presort = self.json_dict['train_presort']
            choice_train_batch_size = self.json_dict['train_batch_size']
            # hard-coding for rarely changed settings
            base_data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"], test_presort=True,
                                  validation_presort=True, validation_batch_size=1, test_batch_size=1)
        else:
            scaler_id = None
            choice_min_docs = [10]
            choice_min_rele = [1]
            choice_binary_rele = [False]
            choice_unknown_as_zero = [False]
            choice_train_presort = [True]
            choice_train_batch_size = [1] # number of sample rankings per query

            base_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, test_presort=True,
                                  validation_presort=True, validation_batch_size=1, test_batch_size=1)

        data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
        base_data_dict.update(data_meta)

        choice_scale_data, choice_scaler_id, choice_scaler_level = \
            get_scaler_setting(data_id=self.data_id, grid_search=True, scaler_id=scaler_id)

        for min_docs, min_rele, train_batch_size in product(choice_min_docs, choice_min_rele, choice_train_batch_size):
            threshold_dict = dict(min_docs=min_docs, min_rele=min_rele, train_batch_size=train_batch_size)

            for binary_rele, unknown_as_zero, train_presort in product(choice_binary_rele, choice_unknown_as_zero, choice_train_presort):
                custom_dict = dict(binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, train_presort=train_presort)

                for scale_data, _scaler_id, scaler_level in product(choice_scale_data, choice_scaler_id, choice_scaler_level):
                    scale_dict = dict(scale_data=scale_data, scaler_id=_scaler_id, scaler_level=scaler_level)

                    self.data_dict = dict()
                    self.data_dict.update(base_data_dict)
                    self.data_dict.update(threshold_dict)
                    self.data_dict.update(custom_dict)
                    self.data_dict.update(scale_dict)
                    yield self.data_dict
