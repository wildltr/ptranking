#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from itertools import product

from ptranking.ltr_adhoc.eval.parameter import DataSetting, EvalSetting
from ptranking.data.data_utils import get_scaler_setting, MSLETOR_SEMI, get_data_meta


class TreeDataSetting(DataSetting):
    """
    Class object for data settings w.r.t. data loading and pre-process w.r.t. tree-based method
    """
    def __init__(self, debug=False, data_id=None, dir_data=None, tree_data_json=None):
        super(TreeDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data, data_json=tree_data_json)

    def default_setting(self):
        """
        A default setting for data loading when running lambdaMART
        """
        scaler_id = None
        unknown_as_zero = True if self.data_id in MSLETOR_SEMI else False # since lambdaMART is a supervised method
        binary_rele = False  # using the original values
        train_presort, validation_presort, test_presort = False, False, False
        train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 1, 1, 1

        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id, scaler_id=scaler_id)

        # more data settings that are rarely changed
        self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
                unknown_as_zero=unknown_as_zero, binary_rele=binary_rele, train_presort=train_presort,
                validation_presort=validation_presort, test_presort=test_presort, train_rough_batch_size=train_rough_batch_size,
                validation_rough_batch_size=validation_rough_batch_size, test_rough_batch_size=test_rough_batch_size,
                              scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
        self.data_dict.update(data_meta)

        return self.data_dict


class TreeEvalSetting(EvalSetting):
    """
    Class object for evaluation settings w.r.t. tree-based methods
    """
    def __init__(self, debug=False, dir_output=None, tree_eval_json=None):
        super(TreeEvalSetting, self).__init__(debug=debug, dir_output=dir_output, eval_json=tree_eval_json)

    def to_eval_setting_string(self, log=False):
        """
        String identifier of eval-setting
        :param log:
        :return:
        """
        eval_dict = self.eval_dict
        s1, s2 = (':', '\n') if log else ('_', '_')

        early_stop_or_boost_round, do_validation = eval_dict['early_stop_or_boost_round'], eval_dict['do_validation']
        if do_validation:
            eval_string = s1.join(['EarlyStop', str(early_stop_or_boost_round)])
        else:
            eval_string = s1.join(['BoostRound', str(early_stop_or_boost_round)])

        return eval_string

    def default_setting(self):
        """
        A default setting for evaluation
        """
        do_validation = True if self.debug else True
        do_log = False if self.debug else True
        early_stop_or_boost_round = 10 if self.debug else 200

        # more evaluation settings that are rarely changed
        self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output, do_log=do_log,
                              cutoffs=[1, 3, 5, 10, 20, 50], do_validation=do_validation,
                              mask_label=False, early_stop_or_boost_round=early_stop_or_boost_round)

        return self.eval_dict

    def grid_search(self):
        """
        Iterator of settings for evaluation
        """
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            early_stop_or_boost_round = 20 if self.debug else self.json_dict['early_stop_or_boost_round']
            do_validation = self.json_dict['do_validation']
            cutoffs = self.json_dict['cutoffs']
            do_log = self.json_dict['do_log']
            mask_label = self.json_dict['mask']['mask_label']
            choice_mask_type = self.json_dict['mask']['mask_type']
            choice_mask_ratio = self.json_dict['mask']['mask_ratio']

            base_dict = dict(debug=False, grid_search=True, dir_output=dir_output)
        else:
            base_dict = dict(debug=self.debug, grid_search=True, dir_output=self.dir_output)
            early_stop_or_boost_round = 20 if self.debug else 100
            do_validation = False if self.debug else True  # True, False
            cutoffs = [1, 3, 5, 10, 20, 50]
            do_log = False if self.debug else True

            mask_label = False if self.debug else False
            choice_mask_type = ['rand_mask_all']
            choice_mask_ratio = [0.2]

        self.eval_dict = dict(early_stop_or_boost_round=early_stop_or_boost_round, do_validation=do_validation,
                              cutoffs=cutoffs, do_log=do_log, mask_label=mask_label)
        self.eval_dict.update(base_dict)

        if mask_label:
            for mask_type, mask_ratio in product(choice_mask_type, choice_mask_ratio):
                mask_dict = dict(mask_type=mask_type, mask_ratio=mask_ratio)
                self.eval_dict.update(mask_dict)
                yield self.eval_dict
        else:
            yield self.eval_dict
