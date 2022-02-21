#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
The class of Parameter is designed as a wrapper of parameters of a model, a neural scoring function, etc.
For data loading and evaluation-related setting, the corresponding classes are DataSetting and EvalSetting.
"""

import os
import json
import datetime
import numpy as np
from itertools import product

import torch

from ptranking.metric.metric_utils import sort_nicely
from ptranking.utils.bigdata.BigPickle import pickle_save
from ptranking.data.data_utils import get_scaler_setting, get_data_meta
from ptranking.metric.metric_utils import metric_results_to_string, get_opt_model

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
    def __init__(self, debug=False, sf_id=None, sf_json=None):
        super(ScoringFunctionParameter, self).__init__(para_json=sf_json)
        self.debug = debug
        if self.use_json:
            self.sf_id = self.json_dict['sf_id']
        else:
            self.sf_id = sf_id

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["SFParameter"]
        return json_dict

    def default_para_dict(self):
        if self.sf_id.startswith('pointsf'):
            return self.default_pointsf_para_dict()
        elif self.sf_id.startswith('listsf'):
            return self.default_listsf_para_dict()
        else:
            raise NotImplementedError

    def grid_search(self):
        if self.sf_id.startswith('pointsf'):
            return self.pointsf_grid_search()
        elif self.sf_id.startswith('listsf'):
            return self.listsf_grid_search()
        else:
            raise NotImplementedError

    def to_para_string(self, log=False):
        if self.sf_id.startswith('pointsf'):
            return self.pointsf_to_para_string(log=log)
        elif self.sf_id.startswith('listsf'):
            return self.listsf_to_para_string(log=log)
        else:
            raise NotImplementedError

    def default_pointsf_para_dict(self):
        """
        A default setting of the hyper-parameters of the stump neural scoring function.
        """
        # common default settings for a scoring function based on feed-forward neural networks

        self.sf_para_dict = dict()

        if self.use_json:
            opt = self.json_dict['opt'][0]
            lr = self.json_dict['lr'][0]
            pointsf_json_dict = self.json_dict[self.sf_id]
            num_layers = pointsf_json_dict['layers'][0]
            af = pointsf_json_dict['AF'][0]
            tl_af = pointsf_json_dict['TL_AF'][0]
            apply_tl_af = pointsf_json_dict['apply_tl_af'][0]
            BN = pointsf_json_dict['BN'][0]
            bn_type = pointsf_json_dict['bn_type'][0]
            bn_affine = pointsf_json_dict['bn_affine'][0]

            self.sf_para_dict['opt'] = opt
            self.sf_para_dict['lr'] = lr
            pointsf_para_dict = dict(num_layers=num_layers, AF=af, TL_AF=tl_af, apply_tl_af=apply_tl_af,
                                     BN=BN, bn_type=bn_type, bn_affine=bn_affine)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict
        else:
            self.sf_para_dict['opt'] = 'Adam'  # Adam | RMS | Adagrad
            self.sf_para_dict['lr'] = 0.0001  # learning rate

            pointsf_para_dict = dict(num_layers=5, AF='GE', TL_AF='S', apply_tl_af=True,
                                     BN=True, bn_type='BN', bn_affine=True)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict

        return self.sf_para_dict

    def default_listsf_para_dict(self):
        """
        A default setting of the hyper-parameters of the permutation-equivariant neural scoring function.
        """
        self.sf_para_dict = dict()
        self.sf_para_dict['opt'] = 'Adagrad'  # Adam | RMS | Adagrad
        self.sf_para_dict['lr'] = 0.001  # learning rate

        listsf_para_dict = dict(ff_dims=[128, 256, 512], AF='R', TL_AF='GE', apply_tl_af=False,
                                BN=False, bn_type='BN2', bn_affine=False,
                                n_heads=2, encoder_layers=6, encoder_type='DASALC')  # DASALC, AllRank, AttnDIN
        self.sf_para_dict['sf_id'] = self.sf_id
        self.sf_para_dict[self.sf_id] = listsf_para_dict

        return self.sf_para_dict

    def pointsf_grid_search(self):
        """
        Iterator of hyper-parameters of the stump neural scoring function.
        """
        if self.use_json:
            choice_opt = self.json_dict['opt']
            choice_lr = self.json_dict['lr']
            pointsf_json_dict = self.json_dict[self.sf_id]
            choice_layers = pointsf_json_dict['layers']
            choice_af = pointsf_json_dict['AF']
            choice_tl_af = pointsf_json_dict['TL_AF']
            choice_apply_tl_af = pointsf_json_dict['apply_tl_af']
            choice_BN = pointsf_json_dict['BN']
            choice_bn_type = pointsf_json_dict['bn_type']
            choice_bn_affine = pointsf_json_dict['bn_affine']
        else:
            choice_opt = ['Adam']
            choice_lr = [0.001]
            choice_BN = [True]
            choice_bn_type = ['BN2']
            choice_bn_affine = [False]
            choice_layers = [3]     if self.debug else [5]  # 1, 2, 3, 4
            choice_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S'] # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_apply_tl_af = [True]  # True, False

        for opt, lr in product(choice_opt, choice_lr):
            sf_para_dict = dict()
            sf_para_dict['sf_id'] = self.sf_id
            base_dict = dict(opt=opt, lr=lr)
            sf_para_dict.update(base_dict)

            for num_layers, af, apply_tl_af, BN in product(choice_layers, choice_af, choice_apply_tl_af, choice_BN):
                pointsf_para_dict = dict(num_layers=num_layers, AF=af, apply_tl_af=apply_tl_af, BN=BN)
                if apply_tl_af:
                    for tl_af in choice_tl_af:
                        pointsf_para_dict.update(dict(TL_AF=tl_af))
                        if BN:
                            for bn_type, bn_affine in product(choice_bn_type, choice_bn_affine):
                                bn_dict = dict(bn_type=bn_type, bn_affine=bn_affine)
                                pointsf_para_dict.update(bn_dict)
                                sf_para_dict[self.sf_id] = pointsf_para_dict
                                self.sf_para_dict = sf_para_dict
                                yield sf_para_dict
                        else:
                            sf_para_dict[self.sf_id] = pointsf_para_dict
                            self.sf_para_dict = sf_para_dict
                            yield sf_para_dict
                else:
                    if BN:
                        for bn_type, bn_affine in product(choice_bn_type, choice_bn_affine):
                            bn_dict = dict(bn_type=bn_type, bn_affine=bn_affine)
                            pointsf_para_dict.update(bn_dict)
                            sf_para_dict[self.sf_id] = pointsf_para_dict
                            self.sf_para_dict = sf_para_dict
                            yield sf_para_dict
                    else:
                        sf_para_dict[self.sf_id] = pointsf_para_dict
                        self.sf_para_dict = sf_para_dict
                        yield sf_para_dict

    def listsf_grid_search(self):
        if self.use_json:
            choice_opt = self.json_dict['opt']
            choice_lr = self.json_dict['lr']
            listsf_json_dict = self.json_dict[self.sf_id]
            ff_dims = listsf_json_dict['ff_dims']
            choice_af = listsf_json_dict['AF']
            choice_tl_af = listsf_json_dict['tl_af']
            choice_apply_tl_af = listsf_json_dict['apply_tl_af']
            choice_BN = listsf_json_dict['BN']
            choice_bn_type = listsf_json_dict['bn_type']
            choice_bn_affine = listsf_json_dict['bn_affine']
            choice_n_heads = listsf_json_dict['n_heads']
            choice_encoder_type = listsf_json_dict['encoder_type']
            choice_encoder_layers = listsf_json_dict['encoder_layers']
        else:
            choice_opt = ['Adam']
            choice_lr = [0.001]
            choice_BN = [True]
            choice_bn_type = ['BN2']
            choice_bn_affine = [False]
            ff_dims = [128, 256, 512] if self.debug else [128, 256, 512]  # 1, 2, 3, 4
            choice_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_apply_tl_af = [True]  # True, False
            choice_n_heads = [2]
            choice_encoder_type = ["DASALC"] # DASALC, AllRank
            choice_encoder_layers = [3]

        for opt, lr in product(choice_opt, choice_lr):
            sf_para_dict = dict()
            sf_para_dict['sf_id'] = self.sf_id
            base_dict = dict(opt=opt, lr=lr)
            sf_para_dict.update(base_dict)

            for af, n_heads, encoder_type, encoder_layers, BN, apply_tl_af in product(choice_af, choice_n_heads,
                                          choice_encoder_type, choice_encoder_layers, choice_BN, choice_apply_tl_af):
                listsf_para_dict = dict(AF=af, BN=BN, ff_dims=ff_dims, apply_tl_af=apply_tl_af,
                                        n_heads=n_heads, encoder_type=encoder_type, encoder_layers=encoder_layers)
                if apply_tl_af:
                    for tl_af in choice_tl_af:
                        listsf_para_dict.update(dict(TL_AF=tl_af))
                        if BN:
                            for bn_type, bn_affine in product(choice_bn_type, choice_bn_affine):
                                bn_dict = dict(bn_type=bn_type, bn_affine=bn_affine)
                                listsf_para_dict.update(bn_dict)
                                sf_para_dict[self.sf_id] = listsf_para_dict
                                self.sf_para_dict = sf_para_dict
                                yield sf_para_dict
                        else:
                            sf_para_dict[self.sf_id] = listsf_para_dict
                            self.sf_para_dict = sf_para_dict
                            yield sf_para_dict
                else:
                    if BN:
                        for bn_type, bn_affine in product(choice_bn_type, choice_bn_affine):
                            bn_dict = dict(bn_type=bn_type, bn_affine=bn_affine)
                            listsf_para_dict.update(bn_dict)
                            sf_para_dict[self.sf_id] = listsf_para_dict
                            self.sf_para_dict = sf_para_dict
                            yield sf_para_dict
                    else:
                        sf_para_dict[self.sf_id] = listsf_para_dict
                        self.sf_para_dict = sf_para_dict
                        yield sf_para_dict

    def get_stacked_FFNet_str(self, ff_para_dict=None, point=False, log=False, s1=None, s2=None):
        AF, BN = ff_para_dict['AF'], ff_para_dict['BN']
        bn_type = ff_para_dict['bn_type'] if BN else None
        bn_affine = ff_para_dict['bn_affine'] if BN else None
        TL_AF = ff_para_dict['TL_AF'] if ff_para_dict['apply_tl_af'] else 'No'

        if log:
            sf_str = s2.join([s1.join(['AF', AF]), s1.join(['TL_AF', TL_AF])])
            if BN: sf_str = s2.join([sf_str, s1.join(['bn_type', bn_type]), s1.join(['bn_affine', str(bn_affine)])])
            if point:
                sf_str = s2.join([sf_str, s1.join(['num_layers', str(ff_para_dict['num_layers'])])])
            else:
                ff_dims = ff_para_dict['ff_dims']
                sf_str = s2.join([sf_str, s1.join(['ff_dims', '.'.join([str(x) for x in ff_dims] + [AF])])])
        else:
            num_layers = ff_para_dict['num_layers'] if point else len(ff_para_dict['ff_dims'])
            sf_str = AF + str(num_layers) + TL_AF
            if BN:
                if bn_affine:
                    bn_str = bn_type + '_Affine'
                else:
                    bn_str = bn_type
                sf_str = '_'.join([sf_str, bn_str])

        return sf_str

    def pointsf_to_para_string(self, log=False):
        ''' Get the identifier of scoring function '''
        s1, s2 = (':', '\n') if log else ('_', '_')
        sf_para_dict = self.sf_para_dict[self.sf_id]

        sf_str_1 = self.get_stacked_FFNet_str(ff_para_dict=sf_para_dict, point=True, log=log, s1=s1, s2=s2)

        opt, lr = self.sf_para_dict['opt'], self.sf_para_dict['lr']
        sf_str_3 = s2.join([s1.join(['Opt', opt]), s1.join(['lr', '{:,g}'.format(lr)])]) if log\
                   else '_'.join([opt, '{:,g}'.format(lr)])

        if log:
            sf_str = s2.join([sf_str_1, sf_str_3])
        else:
            sf_str = '_'.join([sf_str_1, sf_str_3])

        return sf_str

    def get_encoder_str(self, ff_para_dict=None, log=False, s1=None, s2=None):
        encoder_type, n_heads, encoder_layers = ff_para_dict['encoder_type'],\
                                                ff_para_dict['n_heads'], ff_para_dict['encoder_layers']
        if log:
            sf_str = s2.join([s1.join(['encoder_type', encoder_type]),
                              s1.join(['n_heads', str(n_heads)]),
                              s1.join(['encoder_layers', str(encoder_layers)])
                              ])
        else:
            sf_str = '_'.join([encoder_type, str(n_heads), 'heads', str(encoder_layers), 'encoder'])

        return sf_str


    def listsf_to_para_string(self, log=False):
        ''' Get the identifier of scoring function '''
        s1, s2 = (':', '\n') if log else ('_', '_')
        sf_para_dict = self.sf_para_dict[self.sf_id]

        sf_str_1 = self.get_stacked_FFNet_str(ff_para_dict=sf_para_dict, point=False, log=log, s1=s1, s2=s2)

        sf_str_2 = self.get_encoder_str(ff_para_dict=sf_para_dict, log=log, s1=s1, s2=s2)

        opt, lr = self.sf_para_dict['opt'], self.sf_para_dict['lr']
        sf_str_3 = s2.join([s1.join(['Opt', opt]), s1.join(['lr', '{:,g}'.format(lr)])]) if log\
                    else '_'.join([opt, '{:,g}'.format(lr)])

        if log:
            sf_str = s2.join([sf_str_1, sf_str_2, sf_str_3])
        else:
            sf_str = '_'.join([sf_str_1, sf_str_2, sf_str_3])

        return sf_str


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
        if do_vali:
            vali_metric, vali_k = eval_dict['vali_metric'], eval_dict['vali_k']
            vali_str = '@'.join([vali_metric, str(vali_k)])
            eval_string = s2.join([s1.join(['epochs', str(epochs)]), s1.join(['validation', vali_str])]) if log \
                          else s1.join(['EP', str(epochs), 'V', vali_str])
        else:
            eval_string = s1.join(['epochs', str(epochs)])

        return eval_string

    def default_setting(self):
        """
        A default setting for evaluation
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            epochs = self.json_dict['epochs'] # debug is added for a quick check
            do_validation = self.json_dict['do_validation']
            vali_k = self.json_dict['vali_k'] if do_validation else None
            vali_metric = self.json_dict['vali_metric'] if do_validation else None

            cutoffs = self.json_dict['cutoffs']
            do_log, log_step = self.json_dict['do_log'], self.json_dict['log_step']
            do_summary = self.json_dict['do_summary']
            loss_guided = self.json_dict['loss_guided']
            mask_label = self.json_dict['mask']['mask_label']
            mask_type = self.json_dict['mask']['mask_type']
            mask_ratio = self.json_dict['mask']['mask_ratio']

            self.eval_dict = dict(debug=False, grid_search=False, dir_output=dir_output,
                                  cutoffs=cutoffs, do_validation=do_validation, vali_k=vali_k, vali_metric=vali_metric,
                                  do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=loss_guided,
                                  epochs=epochs, mask_label=mask_label, mask_type=mask_type, mask_ratio=mask_ratio)
        else:
            do_log = False if self.debug else True
            do_validation, do_summary = True, False  # checking loss variation
            log_step = 1
            epochs = 5 if self.debug else 100
            vali_k = 5 if do_validation else None
            vali_metric = 'nDCG' if do_validation else None

            ''' setting for exploring the impact of randomly removing some ground-truth labels '''
            mask_label = False
            mask_type = 'rand_mask_all'
            mask_ratio = 0.2

            # more evaluation settings that are rarely changed
            self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output,
                                  do_validation=do_validation, vali_k=vali_k, vali_metric=vali_metric,
                                  cutoffs=[1, 3, 5, 10, 20, 50], epochs=epochs,
                                  do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False,
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
            do_validation = self.json_dict['do_validation']
            vali_k = self.json_dict['vali_k'] if do_validation else None
            vali_metric = self.json_dict['vali_metric'] if do_validation else None
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
            epochs = 5 if self.debug else 100
            do_validation = False if self.debug else True  # True, False
            vali_k = 5 if do_validation else None
            vali_metric = 'nDCG' if do_validation else None
            cutoffs = [1, 3, 5, 10, 20, 50]
            do_log = False if self.debug else True
            log_step = 1
            do_summary, loss_guided = False, False

            mask_label = False if self.debug else False
            choice_mask_type = ['rand_mask_all']
            choice_mask_ratio = [0.2]

        self.eval_dict = dict(epochs=epochs, do_validation=do_validation, vali_k=vali_k, vali_metric=vali_metric,
                              cutoffs=cutoffs, do_log=do_log, log_step=log_step, do_summary=do_summary,
                              loss_guided=loss_guided, mask_label=mask_label)
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
        min_docs, min_rele, train_rough_batch_size, train_presort = data_dict['min_docs'], data_dict['min_rele'], \
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
        A default setting for data loading
        :return:
        """
        if self.use_json:
            scaler_id = self.json_dict['scaler_id']
            min_docs = self.json_dict['min_docs'][0]
            min_rele = self.json_dict['min_rele'][0]
            binary_rele = self.json_dict['binary_rele'][0]
            unknown_as_zero = self.json_dict['unknown_as_zero'][0]
            tr_batch_size = self.json_dict['tr_batch_size'][0]  # train_rough_batch_size

            scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id, scaler_id=scaler_id)

            # hard-coding for rarely changed settings
            self.data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"],
                                  train_presort=True, test_presort=True, validation_presort=True,
                                  validation_rough_batch_size=100, test_rough_batch_size=100,
                                  min_docs=min_docs, min_rele=min_rele, train_rough_batch_size=tr_batch_size,
                                  scale_data = scale_data, scaler_id = scaler_id, scaler_level = scaler_level,
                                  unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)
        else:
            unknown_as_zero = False # using original labels, e.g., w.r.t. semi-supervised dataset
            binary_rele = False  # using original labels
            train_presort, validation_presort, test_presort = True, True, True
            #train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 1, 100, 100
            train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 100, 100, 100
            scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id)

            # more data settings that are rarely changed
            self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, min_docs=10, min_rele=1,
                                  scale_data = scale_data, scaler_id = scaler_id, scaler_level = scaler_level,
                                  train_presort=train_presort, validation_presort=validation_presort, test_presort=test_presort,
                                  train_rough_batch_size=train_rough_batch_size, validation_rough_batch_size=validation_rough_batch_size,
                                  test_rough_batch_size=test_rough_batch_size, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

        data_meta = get_data_meta(data_id=self.data_id) # add meta-information

        if self.debug: data_meta['fold_num'] = 2
        self.data_dict.update(data_meta)

        return self.data_dict

    def grid_search(self):
        if self.use_json:
            scaler_id = self.json_dict['scaler_id']
            choice_min_docs = self.json_dict['min_docs']
            choice_min_rele = self.json_dict['min_rele']
            choice_binary_rele = self.json_dict['binary_rele']
            choice_unknown_as_zero = self.json_dict['unknown_as_zero']
            choice_tr_batch_size = self.json_dict['tr_batch_size'] # train_rough_batch_size
            # hard-coding for rarely changed settings
            base_data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"],
                                  train_presort=True, test_presort=True, validation_presort=True,
                                  validation_rough_batch_size=100, test_rough_batch_size=100)
        else:
            scaler_id = None
            choice_min_docs = [10]
            choice_min_rele = [1]
            choice_binary_rele = [False]
            choice_unknown_as_zero = [False]
            choice_tr_batch_size = [100]
            base_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data,
                                  train_presort=True, test_presort=True, validation_presort=True,
                                  validation_rough_batch_size=100, test_rough_batch_size=100)

        data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
        if self.debug: data_meta['fold_num'] = 1
        base_data_dict.update(data_meta)

        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=self.data_id, scaler_id=scaler_id)

        for min_docs, min_rele, tr_batch_size in product(choice_min_docs, choice_min_rele, choice_tr_batch_size):
            threshold_dict = dict(min_docs=min_docs, min_rele=min_rele, train_rough_batch_size=tr_batch_size)

            for binary_rele, unknown_as_zero in product(choice_binary_rele, choice_unknown_as_zero):
                custom_dict = dict(binary_rele=binary_rele, unknown_as_zero=unknown_as_zero)
                scale_dict = dict(scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

                self.data_dict = dict()
                self.data_dict.update(base_data_dict)
                self.data_dict.update(threshold_dict)
                self.data_dict.update(custom_dict)
                self.data_dict.update(scale_dict)
                yield self.data_dict

##########
# Tape-recorder objects for logging during the training, validation processes.
##########

class ValidationTape(object):
    """
    Using a specified metric to perform epoch-wise evaluation over the validation dataset.
    """
    def __init__(self, fold_k, num_epochs, validation_metric, validation_at_k, dir_run):
        self.dir_run = dir_run
        self.num_epochs = num_epochs
        self.optimal_metric_value = 0.0
        self.optimal_epoch_value = None
        self.validation_at_k = validation_at_k
        self.validation_metric = validation_metric
        self.fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])

    def epoch_validation(self, epoch_k, metric_value, ranker):
        if epoch_k > 1: # report and buffer currently optimal model
            if (metric_value > self.optimal_metric_value) \
                    or (epoch_k == self.num_epochs and metric_value == self.optimal_metric_value):
                # we need at least a reference, in case all zero
                print('\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k), metric_value)
                self.optimal_epoch_value = epoch_k
                self.optimal_metric_value = metric_value
                ranker.save(dir=self.dir_run + self.fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
            else:
                print('\t\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k), metric_value)

    def get_optimal_path(self):
        buffered_optimal_model = '_'.join(['net_params_epoch', str(self.optimal_epoch_value)]) + '.pkl'
        path = self.dir_run + self.fold_optimal_checkpoint + '/' + buffered_optimal_model
        return path

    def clear_fold_buffer(self, fold_k):
        subdir = '-'.join(['Fold', str(fold_k)])
        run_fold_k_dir = os.path.join(self.dir_run, subdir)
        fold_k_files = os.listdir(run_fold_k_dir)
        list_model_files = []
        if fold_k_files is not None and len(fold_k_files) > 1:
            for f in fold_k_files:
                if f.endswith('.pkl'):
                    list_model_files.append(f)

            if len(list_model_files) > 1:
                sort_nicely(list_model_files)
            for k in range(1, len(list_model_files)):
                tmp_model_file = list_model_files[k]
                os.remove(os.path.join(run_fold_k_dir, tmp_model_file))


class CVTape(object):
    """
    Using multiple metrics to perform (1) fold-wise evaluation; (2) k-fold averaging
    """
    def __init__(self, model_id, fold_num, cutoffs, do_validation, reproduce=False):
        self.cutoffs = cutoffs
        self.fold_num = fold_num
        self.model_id = model_id
        self.reproduce = reproduce
        self.do_validation = do_validation
        self.ndcg_cv_avg_scores = np.zeros(len(cutoffs))
        self.nerr_cv_avg_scores = np.zeros(len(cutoffs))
        self.ap_cv_avg_scores = np.zeros(len(cutoffs))
        self.p_cv_avg_scores = np.zeros(len(cutoffs))
        self.time_begin = datetime.datetime.now() # timing
        if reproduce:
            self.list_per_q_p = []
            self.list_per_q_ap = []
            self.list_per_q_nerr = []
            self.list_per_q_ndcg = []

    def fold_evaluation(self, ranker, test_data, max_label, fold_k, model_id):
        avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks = \
            ranker.adhoc_performance_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu', max_label=max_label)
        fold_ndcg_ks = avg_ndcg_at_ks.data.numpy()
        fold_nerr_ks = avg_nerr_at_ks.data.numpy()
        fold_ap_ks = avg_ap_at_ks.data.numpy()
        fold_p_ks = avg_p_at_ks.data.numpy()

        self.ndcg_cv_avg_scores = np.add(self.ndcg_cv_avg_scores, fold_ndcg_ks)
        self.nerr_cv_avg_scores = np.add(self.nerr_cv_avg_scores, fold_nerr_ks)
        self.ap_cv_avg_scores = np.add(self.ap_cv_avg_scores, fold_ap_ks)
        self.p_cv_avg_scores = np.add(self.p_cv_avg_scores, fold_p_ks)


        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=fold_ndcg_ks, list_cutoffs=self.cutoffs,
                                                         metric='nDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_nerr_ks, list_cutoffs=self.cutoffs,
                                                         metric='nERR'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_ap_ks, list_cutoffs=self.cutoffs,
                                                         metric='AP'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_p_ks, list_cutoffs=self.cutoffs,
                                                         metric='P'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {}\n\t{}".format(model_id, str(fold_k), metric_string))

    def fold_evaluation_reproduce(self, ranker, test_data, dir_run, max_label, fold_k, model_id, device='cpu'):
        self.dir_run = dir_run
        subdir = '-'.join(['Fold', str(fold_k)])
        run_fold_k_dir = os.path.join(dir_run, subdir)
        fold_k_buffered_model_names = os.listdir(run_fold_k_dir)
        fold_opt_model_name = get_opt_model(fold_k_buffered_model_names)
        fold_opt_model = os.path.join(run_fold_k_dir, fold_opt_model_name)
        ranker.load(file_model=fold_opt_model, device=device)

        avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks, list_per_q_ndcg, list_per_q_nerr, list_per_q_ap,\
        list_per_q_p = ranker.adhoc_performance_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu',
                                                      max_label=max_label, need_per_q=True)
        fold_ndcg_ks = avg_ndcg_at_ks.data.numpy()
        fold_nerr_ks = avg_nerr_at_ks.data.numpy()
        fold_ap_ks = avg_ap_at_ks.data.numpy()
        fold_p_ks = avg_p_at_ks.data.numpy()

        self.list_per_q_p.extend(list_per_q_p)
        self.list_per_q_ap.extend(list_per_q_ap)
        self.list_per_q_nerr.extend(list_per_q_nerr)
        self.list_per_q_ndcg.extend(list_per_q_ndcg)

        self.ndcg_cv_avg_scores = np.add(self.ndcg_cv_avg_scores, fold_ndcg_ks)
        self.nerr_cv_avg_scores = np.add(self.nerr_cv_avg_scores, fold_nerr_ks)
        self.ap_cv_avg_scores = np.add(self.ap_cv_avg_scores, fold_ap_ks)
        self.p_cv_avg_scores = np.add(self.p_cv_avg_scores, fold_p_ks)


        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=fold_ndcg_ks, list_cutoffs=self.cutoffs,
                                                         metric='nDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_nerr_ks, list_cutoffs=self.cutoffs,
                                                         metric='nERR'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_ap_ks, list_cutoffs=self.cutoffs,
                                                         metric='AP'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_p_ks, list_cutoffs=self.cutoffs,
                                                         metric='P'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {}\n\t{}".format(model_id, str(fold_k), metric_string))

    def get_cv_performance(self):
        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - self.time_begin)

        ndcg_cv_avg_scores = np.divide(self.ndcg_cv_avg_scores, self.fold_num)
        nerr_cv_avg_scores = np.divide(self.nerr_cv_avg_scores, self.fold_num)
        ap_cv_avg_scores = np.divide(self.ap_cv_avg_scores, self.fold_num)
        p_cv_avg_scores = np.divide(self.p_cv_avg_scores, self.fold_num)

        eval_prefix = str(self.fold_num) + '-fold cross validation scores:' if self.do_validation \
                      else str(self.fold_num) + '-fold average scores:'

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=ndcg_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='nDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=nerr_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='nERR'))
        list_metric_strs.append(metric_results_to_string(list_scores=ap_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='AP'))
        list_metric_strs.append(metric_results_to_string(list_scores=p_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='P'))
        metric_string = '\n'.join(list_metric_strs)
        print("\n{} {}\n{}".format(self.model_id, eval_prefix, metric_string))
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        if self.reproduce:
            torch_mat_per_q_p = torch.cat(self.list_per_q_p, dim=0)
            torch_mat_per_q_ap = torch.cat(self.list_per_q_ap, dim=0)
            torch_mat_per_q_nerr = torch.cat(self.list_per_q_nerr, dim=0)
            torch_mat_per_q_ndcg = torch.cat(self.list_per_q_ndcg, dim=0)
            #print('torch_mat_per_q_ndcg', torch_mat_per_q_ndcg.size())
            mat_per_q_p = torch_mat_per_q_p.data.numpy()
            mat_per_q_ap = torch_mat_per_q_ap.data.numpy()
            mat_per_q_nerr = torch_mat_per_q_nerr.data.numpy()
            mat_per_q_ndcg = torch_mat_per_q_ndcg.data.numpy()

            pickle_save(target=mat_per_q_p, file=self.dir_run + '_'.join([self.model_id, 'all_fold_p_at_ks_per_q.np']))
            pickle_save(target=mat_per_q_ap,
                        file=self.dir_run + '_'.join([self.model_id, 'all_fold_ap_at_ks_per_q.np']))
            pickle_save(target=mat_per_q_nerr,
                        file=self.dir_run + '_'.join([self.model_id, 'all_fold_nerr_at_ks_per_q.np']))
            pickle_save(target=mat_per_q_ndcg,
                        file=self.dir_run + '_'.join([self.model_id, 'all_fold_ndcg_at_ks_per_q.np']))

        return ndcg_cv_avg_scores

class SummaryTape(object):
    """
    Using multiple metrics to perform epoch-wise evaluation on train-data, validation-data, test-data
    """
    def __init__(self, do_validation, cutoffs, label_type, train_presort, test_presort, gpu):
        self.gpu = gpu
        self.cutoffs = cutoffs
        self.list_epoch_loss = []
        self.label_type = label_type
        self.do_validation = do_validation
        if do_validation: self.list_fold_k_vali_track = []
        self.list_fold_k_train_track, self.list_fold_k_test_track = [], []
        self.train_presort, self.test_presort = train_presort, test_presort

    def epoch_summary(self, ranker, torch_epoch_k_loss, train_data, test_data, vali_metric_value):
        ''' Summary in terms of nDCG '''

        fold_k_epoch_k_loss = torch_epoch_k_loss.cpu().numpy() if self.gpu else torch_epoch_k_loss.data.numpy()
        self.list_epoch_loss.append(fold_k_epoch_k_loss)

        if self.do_vali: self.list_fold_k_vali_track.append(vali_metric_value)

        fold_k_epoch_k_train_ndcg_ks = ranker.ndcg_at_ks(test_data=train_data, ks=self.cutoffs,
                                                 label_type=self.label_type, device='cpu', presort=self.train_presort)
        np_fold_k_epoch_k_train_ndcg_ks = \
            fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
        self.list_fold_k_train_track.append(np_fold_k_epoch_k_train_ndcg_ks)

        fold_k_epoch_k_test_ndcg_ks  = ranker.ndcg_at_ks(test_data=test_data, ks=self.cutoffs,
                                                 label_type=self.label_type, device='cpu', presort=self.test_presort)
        np_fold_k_epoch_k_test_ndcg_ks  = \
            fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
        self.list_fold_k_test_track.append(np_fold_k_epoch_k_test_ndcg_ks)

    def fold_summary(self, fold_k, dir_run, train_data_length):
        sy_prefix = '_'.join(['Fold', str(fold_k)])

        if self.do_validation:
            fold_k_vali_eval = np.hstack(self.list_fold_k_vali_track)
            pickle_save(fold_k_vali_eval, file=dir_run + '_'.join([sy_prefix, 'vali_eval.np']))

        fold_k_train_eval = np.vstack(self.list_fold_k_train_track)
        fold_k_test_eval = np.vstack(self.list_fold_k_test_track)
        pickle_save(fold_k_train_eval, file=dir_run + '_'.join([sy_prefix, 'train_eval.np']))
        pickle_save(fold_k_test_eval, file=dir_run + '_'.join([sy_prefix, 'test_eval.np']))

        fold_k_epoch_loss = np.hstack(self.list_epoch_loss)
        pickle_save((fold_k_epoch_loss, train_data_length), file=dir_run + '_'.join([sy_prefix, 'epoch_loss.np']))

class OptLossTape(object):
    ''' Used when the optimization is guided by the training loss '''
    def __init__(self, gpu):
        self.first_round = True
        self.threshold_epoch_loss = torch.cuda.FloatTensor([10000000.0]) if gpu else torch.FloatTensor([10000000.0])

    def epoch_cmp_loss(self, torch_epoch_k_loss, fold_k, epoch_k):
        early_stopping = False
        if self.first_round and torch_epoch_k_loss >= self.threshold_epoch_loss:
            print('Bad threshold: ', torch_epoch_k_loss, self.threshold_epoch_loss)
        if torch_epoch_k_loss < self.threshold_epoch_loss:
            self.first_round = False
            print('\tFold-', str(fold_k), ' Epoch-', str(epoch_k), 'Loss: ', torch_epoch_k_loss)
            self.threshold_epoch_loss = torch_epoch_k_loss
        else:
            print('\tStopped according epoch-loss!', torch_epoch_k_loss, self.threshold_epoch_loss)
            early_stopping = True

        return early_stopping
