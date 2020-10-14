#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
from itertools import product

from ptranking.base.neural_utils import get_sf_str
from ptranking.data.data_utils import get_default_scaler_setting, get_data_meta, MSLETOR_SEMI

class Parameter(object):
    """
    An abstract class for parameter
    """
    def __init__(self):
	    pass

    def default_para_dict(self):
        """
        The default parameter setting.
        :return:
        """
        return None

    def to_para_string(self):
        """
        The string identifier of parameters
        :return:
        """
        return None

    def grid_search(self):
        """
        Iterator of parameter setting for grid-search
        :return:
        """
        return None


class ModelParameter(Parameter):
    """
    A simple class for model parameter
    """
    def __init__(self, model_id=None):
        super(ModelParameter, self).__init__()
        self.model_id = model_id

    def default_para_dict(self):
        return dict(model_id=self.model_id)

    def to_para_string(self):
        return ''

    def grid_search(self):
         yield dict(model_id=self.model_id)


class ScoringFunctionParameter(ModelParameter):
    """
    The parameter class w.r.t. a neural scoring fuction
    """
    def __init__(self, debug=False, data_dict=None, model_id='ffnns'):
        super(ScoringFunctionParameter, self).__init__(model_id=model_id)
        self.debug = debug
        self.data_dict = data_dict

    def default_para_dict(self):
        """
        A default setting of the hyper-parameters of the stump neural scoring function.
        :param data_dict:
        :return:
        """
        assert self.data_dict is not None
        FBN = False if self.data_dict['scale_data'] else True # for feature normalization

        # feed-forward neural networks
        ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='R', apply_tl_af=True, BN=True, RD=False, FBN=FBN)

        sf_para_dict = dict()
        sf_para_dict['id'] = self.model_id
        sf_para_dict[self.model_id] = ffnns_para_dict

        self.sf_para_dict = sf_para_dict
        return sf_para_dict

    def grid_search(self, data_dict=None):
        """
        Iterator of hyper-parameters of the stump neural scoring function.
        :param debug:
        :param data_dict:
        :return:
        """
        assert data_dict is not None
        FBN = False if data_dict['scale_data'] else True  # for feature normalization

        choice_apply_BN = [False] if self.debug else [True]  # True, False
        choice_apply_RD = [False] if self.debug else [False]  # True, False

        choice_layers = [3]     if self.debug else [5]  # 1, 2, 3, 4
        choice_hd_hn_af = ['S'] if self.debug else ['R']  # 'R6' | 'RK' | 'S' activation function w.r.t. head hidden layers
        choice_tl_af = ['S']    if self.debug else ['R']  # activation function for the last layer, sigmoid is suggested due to zero-prediction
        choice_hd_hn_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S'] # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
        choice_apply_tl_af = [True]  # True, False

        if choice_hd_hn_tl_af is not None:
            for BN, RD, num_layers, af, apply_tl_af in product(choice_apply_BN, choice_apply_RD, choice_layers,
                                                               choice_hd_hn_tl_af, choice_apply_tl_af):
                ffnns_para_dict = dict(FBN=FBN, BN=BN, RD=RD, num_layers=num_layers, HD_AF=af, HN_AF=af, TL_AF=af,
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
                ffnns_para_dict = dict(FBN=FBN, BN=BN, RD=RD, num_layers=num_layers, HD_AF=hd_hn_af, HN_AF=hd_hn_af,
                                       TL_AF=tl_af, apply_tl_af=apply_tl_af)
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
            rf_str = get_sf_str(num_layers, HD_AF, HN_AF, TL_AF)
            if BN:  rf_str += '_BN'
            if RD:  rf_str += '_RD'
            if FBN: rf_str += '_FBN'

        return rf_str


class EvalSetting():
    """
    Class object for evaluation settings w.r.t. training, etc.
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
        A default setting for evaluation
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        do_log = False if self.debug else True
        do_validation, do_summary = False, False  # checking loss variation
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
        """
        Iterator of settings for evaluation
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

        for do_validation, num_epochs, mask_label in product(choice_validation, choice_epoch, choice_mask_label):
            self.eval_dict = dict(do_validation=do_validation, epochs=num_epochs, mask_label=mask_label)
            self.eval_dict.update(common_eval_dict)
            yield self.eval_dict


class DataSetting():
    """
    Class object for data settings w.r.t. data loading and pre-process.
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
        A default setting for data loading
        :return:
        """
        unknown_as_zero = True if self.data_id in MSLETOR_SEMI else False
        binary_rele = False  # using the original values
        presort = False  # a default setting

        scale_data, scaler_id, scaler_level = get_default_scaler_setting(data_id=self.data_id)

        # more data settings that are rarely changed
        self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
                         sample_rankings_per_q=1, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele,
                         presort=presort, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        data_meta = get_data_meta(data_id=self.data_id) # add meta-information
        self.data_dict.update(data_meta)

        return self.data_dict

    def grid_search(self):
        """
        Iterator of settings for data loading
        :param debug:
        :param data_id:
        :param dir_data:
        :return:
        """
        ''' common settings without grid-search '''
        binary_rele = False
        unknown_as_zero = True if self.data_id in MSLETOR_SEMI else False
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
