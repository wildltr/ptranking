#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product

from ptranking.eval.parameter import DataSetting, EvalSetting
from ptranking.data import get_default_scaler_setting, MSLETOR_SEMI, get_data_meta


class TreeDataSetting(DataSetting):
    """
    Class object for data settings w.r.t. data loading and pre-process w.r.t. tree-based method
    """
    def __init__(self, debug=False, data_id=None, dir_data=None):
        super(TreeDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data)

    def default_setting(self):
        """
        A default setting for data loading when running lambdaMART
        :return:
        """
        unknown_as_zero = True if self.data_id in MSLETOR_SEMI else False # since lambdaMART is a supervised method
        binary_rele = False  # using the original values
        presort = True  # this setting leads to no difference for lambdaMART, but it can be altered to reused buffered data

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
        Iterator of settings for data loading when running lambdaMART
        :param debug:
        :param data_id:
        :param dir_data:
        :return:
        """
        self.data_dict = self.default_setting() # a simple setting since there are few factors to grid-search
        yield self.data_dict


class TreeEvalSetting(EvalSetting):
    """
    Class object for evaluation settings w.r.t. tree-based methods
    """
    def __init__(self, debug=False, dir_output=None):
        super(TreeEvalSetting, self).__init__(debug=debug, dir_output=dir_output)

    def to_eval_setting_string(self, log=False):
        """
        String identifier of eval-setting
        :param log:
        :return:
        """
        eval_dict = self.eval_dict
        s1, s2 = (':', '\n') if log else ('_', '_')

        epochs, do_validation = eval_dict['epochs'], eval_dict['do_validation']
        if do_validation:
            eval_string = s1.join(['EarlyStop', str(epochs)])
        else:
            eval_string = s1.join(['BoostRound', str(epochs)])

        return eval_string

    def default_setting(self):
        """
        A default setting for evaluation
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        do_validation = True if self.debug else True
        do_log = False if self.debug else True
        epochs = 10 if self.debug else 100

        # more evaluation settings that are rarely changed
        self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output, do_log=do_log,
                              cutoffs=[1, 3, 5, 10, 20], do_validation=do_validation, epochs=epochs,
                              mask_label=False)

        return self.eval_dict

    def grid_search(self):
        """
        Iterator of settings for evaluation
        :param debug:
        :param dir_output:
        :return:
        """
        ''' common settings without grid-search '''
        do_log = False if self.debug else True
        common_eval_dict = dict(debug=self.debug, grid_search=True, do_log=do_log, dir_output=self.dir_output,
                                cutoffs=[1, 3, 5, 10, 20])

        ''' some settings for grid-search '''
        choice_validation = [False] if self.debug else [True]  # True, False
        choice_epoch = [20] if self.debug else [100]
        choice_mask_label = [False] if self.debug else [False]

        for vd, num_epochs, mask_label in product(choice_validation, choice_epoch, choice_mask_label):
            self.eval_dict = dict(do_validation=vd, epochs=num_epochs, mask_label=mask_label)
            self.eval_dict.update(common_eval_dict)
            yield self.eval_dict
