
import os
import json
import datetime
import subprocess
import numpy as np
from itertools import product

import torch

from ptranking.utils.bigdata.BigPickle import pickle_save
from ptranking.metric.metric_utils import metric_results_to_string, get_opt_model
from ptranking.ltr_diversification.util.div_data import get_div_data_meta
from ptranking.ltr_adhoc.eval.parameter import EvalSetting, DataSetting, ScoringFunctionParameter

class DivScoringFunctionParameter(ScoringFunctionParameter):
    """  """
    def __init__(self, debug=False, sf_id=None, sf_json=None):
        super(DivScoringFunctionParameter, self).__init__(debug=debug, sf_id=sf_id, sf_json=sf_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["DivSFParameter"]
        return json_dict

    def default_pointsf_para_dict(self):
        """
        The default setting of the hyper-parameters of the stump neural scoring function.
        """
        self.sf_para_dict = dict()

        if self.use_json:
            opt = self.json_dict['opt'][0]
            lr = self.json_dict['lr'][0]
            pointsf_json_dict = self.json_dict[self.sf_id]
            num_layers = pointsf_json_dict['layers'][0]
            af = pointsf_json_dict['AF'][0]

            apply_tl_af = pointsf_json_dict['apply_tl_af'][0]
            tl_af = pointsf_json_dict['TL_AF'][0] if apply_tl_af else None

            BN = pointsf_json_dict['BN'][0]
            bn_type = pointsf_json_dict['bn_type'][0] if BN else None
            bn_affine = pointsf_json_dict['bn_affine'][0] if BN else None

            self.sf_para_dict['opt'] = opt
            self.sf_para_dict['lr'] = lr
            pointsf_para_dict = dict(num_layers=num_layers, AF=af, TL_AF=tl_af, apply_tl_af=apply_tl_af,
                                     BN=BN, bn_type=bn_type, bn_affine=bn_affine)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict
        else:
            # optimization-specific setting
            self.sf_para_dict['opt'] = 'Adagrad' # Adam | RMS | Adagrad
            self.sf_para_dict['lr'] = 0.001 # learning rate

            # common settings for a scoring function based on feed-forward neural networks
            pointsf_para_dict = dict(num_layers=5, AF='GE', TL_AF='GE', apply_tl_af=False,
                                     BN=True, bn_type='BN', bn_affine=True)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict

        return self.sf_para_dict

    def default_listsf_para_dict(self):
        """
        The default setting of the hyper-parameters of the permutation-equivariant neural scoring function.
        """
        self.sf_para_dict = dict()
        if self.use_json:
            opt = self.json_dict['opt'][0]
            lr = self.json_dict['lr'][0]
            listsf_json_dict = self.json_dict[self.sf_id]

            BN = listsf_json_dict['BN'][0]
            bn_type = listsf_json_dict['bn_type'][0] if BN else None
            bn_affine = listsf_json_dict['bn_affine'][0] if BN else None

            ff_dims = listsf_json_dict['ff_dims']
            af = listsf_json_dict['AF'][0]

            apply_tl_af = listsf_json_dict['apply_tl_af'][0]
            tl_af = listsf_json_dict['TL_AF'][0] if apply_tl_af else None

            n_heads = listsf_json_dict['n_heads'][0]
            encoder_type = listsf_json_dict['encoder_type'][0]
            encoder_layers = listsf_json_dict['encoder_layers'][0]

            self.sf_para_dict['opt'] = opt
            self.sf_para_dict['lr'] = lr
            listsf_para_dict = dict(BN=BN, AF=af, ff_dims=ff_dims, apply_tl_af=apply_tl_af,
                                    n_heads=n_heads, encoder_type=encoder_type, encoder_layers=encoder_layers,
                                    bn_type=bn_type, bn_affine=bn_affine, TL_AF=tl_af)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = listsf_para_dict
        else:
            # optimization-specific setting
            self.sf_para_dict['opt'] = 'Adagrad'  # Adam | RMS | Adagrad
            self.sf_para_dict['lr'] = 0.01  # learning rate
            # DASALC, AllRank, AttnDIN
            listsf_para_dict = dict(encoder_type='AttnDIN', n_heads=6, encoder_layers=6, ff_dims=[256, 128, 64],
                                    AF='R', TL_AF='GE', apply_tl_af=False, BN=True, bn_type='BN', bn_affine=True)
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

            choice_apply_tl_af = pointsf_json_dict['apply_tl_af']
            choice_tl_af = pointsf_json_dict['TL_AF'] if True in choice_apply_tl_af else None

            choice_BN = pointsf_json_dict['BN']
            choice_bn_type = pointsf_json_dict['bn_type'] if True in choice_BN else None
            choice_bn_affine = pointsf_json_dict['bn_affine'] if True in choice_BN else None
        else:
            choice_BN = [True]
            choice_bn_type = ['BN']
            choice_bn_affine = [True]
            choice_layers = [3]     if self.debug else [5]  # 1, 2, 3, 4
            choice_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_apply_tl_af = [True]  # True, False
            choice_opt = ['Adam']
            choice_lr = [0.001]

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

            choice_BN = listsf_json_dict['BN']
            choice_bn_type = listsf_json_dict['bn_type'] if True in choice_BN else None
            choice_bn_affine = listsf_json_dict['bn_affine'] if True in choice_BN else None

            ff_dims = listsf_json_dict['ff_dims']
            choice_af = listsf_json_dict['AF']

            choice_apply_tl_af = listsf_json_dict['apply_tl_af']
            choice_tl_af = listsf_json_dict['TL_AF'] if True in choice_apply_tl_af else None

            choice_n_heads = listsf_json_dict['n_heads']
            choice_encoder_type = listsf_json_dict['encoder_type']
            choice_encoder_layers = listsf_json_dict['encoder_layers']
        else:
            ff_dims = [128, 256, 512] if self.debug else [128, 256, 512]  # 1, 2, 3, 4
            choice_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
            choice_apply_tl_af = [True]  # True, False
            choice_n_heads = [2]
            choice_encoder_type = ["DASALC"] # DASALC, AllRank
            choice_encoder_layers = [3]
            choice_opt = ['Adam']
            choice_lr = [0.001]
            choice_BN = [True]
            choice_bn_type = ['BN']
            choice_bn_affine = [True]

        for opt, lr in product(choice_opt, choice_lr):
            sf_para_dict = dict()
            sf_para_dict['sf_id'] = self.sf_id
            base_dict = dict(opt=opt, lr=lr)
            sf_para_dict.update(base_dict)
            for af, n_heads, encoder_type, encoder_layers, BN, apply_tl_af in product(choice_af, choice_n_heads,
                                          choice_encoder_type, choice_encoder_layers, choice_BN, choice_apply_tl_af):
                listsf_para_dict = dict(BN=BN, AF=af, ff_dims=ff_dims, apply_tl_af=apply_tl_af,
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

    def listsf_to_para_string(self, log=False):
        ''' Get the identifier of scoring function '''
        sf_str = super().listsf_to_para_string(log=log)

        s1, s2 = (':', '\n') if log else ('_', '_')
        if self.sf_id.endswith("co"):
            if log:
                sf_str = s2.join([sf_str, s1.join(['CoVariance', 'True'])])
            else:
                sf_str = '_'.join([sf_str, 'CoCo'])

        return sf_str


class DivEvalSetting(EvalSetting):
    """
    Class object for evaluation settings w.r.t. diversified ranking.
    """
    def __init__(self, debug=False, dir_output=None, div_eval_json=None):
        super(DivEvalSetting, self).__init__(debug=debug, dir_output=dir_output, eval_json=div_eval_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["DivEvalSetting"]
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

        rerank = eval_dict['rerank']
        if rerank:
            rerank_k, rerank_model_id = eval_dict['rerank_k'], eval_dict['rerank_model_id']
            eval_string = s2.join([eval_string, s1.join(['rerank_k', str(rerank_k)]),
                                   s1.join(['rerank_model_id', rerank_model_id])]) if log else \
                          s1.join([eval_string, 'RR', str(rerank_k), rerank_model_id])

        return eval_string

    def default_setting(self):
        """
        A default setting for evaluation when performing diversified ranking.
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            epochs = 5 if self.debug else self.json_dict['epochs']

            do_validation = self.json_dict['do_validation']
            vali_k = self.json_dict['vali_k'] if do_validation else None
            vali_metric = self.json_dict['vali_metric'] if do_validation else None

            cutoffs = self.json_dict['cutoffs']
            do_log = self.json_dict['do_log']
            log_step = self.json_dict['log_step'] if do_log else None
            do_summary = self.json_dict['do_summary']
            loss_guided = self.json_dict['loss_guided']

            rerank = self.json_dict['rerank']
            rerank_k = self.json_dict['rerank_k'] if rerank else None
            rerank_dir = self.json_dict['rerank_dir'] if rerank else None
            rerank_model_id = self.json_dict['rerank_model_id'] if rerank else None
            rerank_model_dir = self.json_dict['rerank_model_dir'] if rerank else None
        else:
            do_log = False if self.debug else True
            do_validation, do_summary = True, False
            cutoffs = [1, 3, 5, 10, 20, 50]
            log_step = 1
            epochs = 5 if self.debug else 500
            vali_k = 5
            vali_metric = 'aNDCG' # nERR-IA, aNDCG
            dir_output = self.dir_output
            loss_guided = False

            rerank = False
            rerank_k = 50 if rerank else None
            rerank_dir = '/Users/iimac/Workbench/CodeBench/Output/DivLTR/Rerank/R_reproduce/Opt_aNDCG/' if rerank\
                else None
            rerank_model_id = 'DivProbRanker' if rerank else None
            rerank_model_dir = '/Users/iimac/Workbench/CodeBench/Output/DivLTR/Rerank/R/DivProbRanker_SF_R3R_BN_Affine_AttnDIN_2_heads_6_encoder_Adagrad_0.01_WT_Div_0912_Implicit_EP_300_V_aNDCG@5/1_SuperSoft_ExpRele_0.01_OptIdeal_10/' if rerank\
                else None

        # more evaluation settings that are rarely changed
        self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=dir_output, epochs=epochs,
                              cutoffs=cutoffs, do_validation=do_validation, vali_metric=vali_metric, vali_k=vali_k,
                              do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=loss_guided,
                              rerank=rerank, rerank_k=rerank_k, rerank_dir=rerank_dir, rerank_model_id=rerank_model_id, rerank_model_dir=rerank_model_dir)
        return self.eval_dict

    def grid_search(self):
        """
        Iterator of settings for evaluation when performing diversified ranking.
        """
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            epochs = 5 if self.debug else self.json_dict['epochs']

            do_validation = self.json_dict['do_validation']
            vali_k = self.json_dict['vali_k'] if do_validation else None
            vali_metric = self.json_dict['vali_metric'] if do_validation else None

            cutoffs = self.json_dict['cutoffs']
            do_log, log_step = self.json_dict['do_log'], self.json_dict['log_step']
            do_summary = self.json_dict['do_summary']
            loss_guided = self.json_dict['loss_guided']

            rerank = self.json_dict['rerank']
            rerank_k = self.json_dict['rerank_k'] if rerank else None
            rerank_dir = self.json_dict['rerank_dir'] if rerank else None
            rerank_model_id = self.json_dict['rerank_model_id'] if rerank else None

            base_dict = dict(debug=False, grid_search=True, dir_output=dir_output)
        else:
            base_dict = dict(debug=self.debug, grid_search=True, dir_output=self.dir_output)
            epochs = 2 if self.debug else 100
            do_validation = False if self.debug else True  # True, False
            vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
            vali_metric = 'aNDCG'
            do_log = False if self.debug else True
            log_step = 1
            do_summary, loss_guided = False, False

            rerank = False
            rerank_k = 20 if rerank else None
            rerank_dir = '' if rerank else None
            rerank_model_id = '' if rerank else None

        self.eval_dict = dict(epochs=epochs, do_validation=do_validation, vali_k=vali_k, cutoffs=cutoffs,
                              vali_metric=vali_metric, do_log=do_log, log_step=log_step,
                              do_summary=do_summary, loss_guided=loss_guided,
                              rerank=rerank, rerank_k=rerank_k, rerank_dir=rerank_dir, rerank_model_id=rerank_model_id)
        self.eval_dict.update(base_dict)

        yield self.eval_dict


class DivDataSetting(DataSetting):
    """
    Class object for data settings w.r.t. data loading and pre-process w.r.t. diversified ranking
    """
    def __init__(self, debug=False, data_id=None, dir_data=None, div_data_json=None):
        super(DivDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data, data_json=div_data_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["DivDataSetting"]
        return json_dict

    def to_data_setting_string(self, log=False):
        """
        String identifier of data-setting
        :param log:
        :return:
        """
        data_dict = self.data_dict
        setting_string, add_noise = data_dict['data_id'], data_dict['add_noise']
        if add_noise:
            std_delta = data_dict['std_delta']
            setting_string = '_'.join([setting_string, 'Gaussian', '{:,g}'.format(std_delta)])

        return setting_string

    def default_setting(self):
        """
        A default setting for data loading when performing diversified ranking
        """
        if self.use_json:
            add_noise = self.json_dict['add_noise'][0]
            std_delta = self.json_dict['std_delta'][0] if add_noise else None
            self.data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"],
                                  add_noise=add_noise, std_delta=std_delta)
        else:
            add_noise = False
            std_delta = 1.0 if add_noise else None
            self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, add_noise=add_noise,std_delta=std_delta)

        div_data_meta = get_div_data_meta(data_id=self.data_id)  # add meta-information
        self.data_dict.update(div_data_meta)
        return self.data_dict

    def grid_search(self):
        """
        Iterator of settings for data loading when performing adversarial ltr
        """
        if self.use_json:
            choice_add_noise = self.json_dict['add_noise']
            choice_std_delta = self.json_dict['std_delta'] if True in choice_add_noise else None
            self.data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"])
        else:
            choice_add_noise = [False]
            choice_std_delta = [1.0] if True in choice_add_noise else None
            self.data_dict = dict(data_id=self.data_id, dir_data=self.dir_data)

        div_data_meta = get_div_data_meta(data_id=self.data_id)  # add meta-information
        self.data_dict.update(div_data_meta)

        for add_noise in choice_add_noise:
            if add_noise:
                for std_delta in choice_std_delta:
                    noise_dict = dict(add_noise=add_noise, std_delta=std_delta)
                    self.data_dict.update(noise_dict)
                    yield self.data_dict
            else:
                noise_dict = dict(add_noise=add_noise, std_delta=None)
                self.data_dict.update(noise_dict)
                yield self.data_dict

##########
# Tape-recorder for logging during the training, validation processes.
##########

class DivCVTape(object):
    """
    Using multiple metrics to perform (1) fold-wise evaluation; (2) k-fold averaging
    """
    def __init__(self, model_id, fold_num, cutoffs, do_validation, reproduce=False):
        self.cutoffs = cutoffs
        self.fold_num = fold_num
        self.model_id = model_id
        self.reproduce = reproduce
        self.do_validation = do_validation
        self.andcg_cv_avg_scores = np.zeros(len(cutoffs))
        self.err_ia_cv_avg_scores = np.zeros(len(cutoffs))
        self.nerr_ia_cv_avg_scores = np.zeros(len(cutoffs))
        self.time_begin = datetime.datetime.now() # timing
        if reproduce:
            self.ndeval_cutoffs = [5, 10, 20]
            self.ndeval_err_ia_cv_avg_scores = np.zeros(3)
            self.ndeval_nerr_ia_cv_avg_scores = np.zeros(3)
            self.ndeval_andcg_cv_avg_scores = np.zeros(3)
            self.list_per_q_andcg = []


    def fold_evaluation(self, ranker, test_data, max_label, fold_k, model_id):
        avg_aNDCG_at_ks, avg_err_ia_at_ks, avg_nerr_ia_at_ks = \
            ranker.srd_performance_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu', max_label=max_label)
        fold_aNDCG_ks = avg_aNDCG_at_ks.data.numpy()
        fold_err_ia_ks = avg_err_ia_at_ks.data.numpy()
        fold_nerr_ia_ks = avg_nerr_ia_at_ks.data.numpy()

        self.andcg_cv_avg_scores = np.add(self.andcg_cv_avg_scores, fold_aNDCG_ks)
        self.err_ia_cv_avg_scores = np.add(self.err_ia_cv_avg_scores, fold_err_ia_ks)
        self.nerr_ia_cv_avg_scores = np.add(self.nerr_ia_cv_avg_scores, fold_nerr_ia_ks)

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=fold_aNDCG_ks, list_cutoffs=self.cutoffs,
                                                         metric='aNDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_err_ia_ks, list_cutoffs=self.cutoffs,
                                                         metric='ERR-IA'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_nerr_ia_ks, list_cutoffs=self.cutoffs,
                                                         metric='nERR-IA'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {}\n\t{}".format(model_id, str(fold_k), metric_string))

    def fold_evaluation_reproduce(self, ranker, test_data, dir_run, max_label, fold_k, model_id):
        self.dir_run = dir_run
        subdir = '-'.join(['Fold', str(fold_k)])
        run_fold_k_dir = os.path.join(dir_run, subdir)
        fold_k_buffered_model_names = os.listdir(run_fold_k_dir)
        fold_opt_model_name = get_opt_model(fold_k_buffered_model_names)
        fold_opt_model = os.path.join(run_fold_k_dir, fold_opt_model_name)
        ranker.load(file_model=fold_opt_model)

        avg_andcg_at_ks, avg_err_ia_at_ks, avg_nerr_ia_at_ks, list_per_q_andcg = \
            ranker.srd_performance_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu', max_label=max_label,
                                         generate_div_run=True, dir=run_fold_k_dir,fold_k=fold_k, need_per_q_andcg=True)

        fold_andcg_ks = avg_andcg_at_ks.data.numpy()
        fold_err_ia_ks = avg_err_ia_at_ks.data.numpy()
        fold_nerr_ia_ks = avg_nerr_ia_at_ks.data.numpy()
        self.list_per_q_andcg.extend(list_per_q_andcg)

        self.andcg_cv_avg_scores = np.add(self.andcg_cv_avg_scores, fold_andcg_ks)
        self.err_ia_cv_avg_scores = np.add(self.err_ia_cv_avg_scores, fold_err_ia_ks)
        self.nerr_ia_cv_avg_scores = np.add(self.nerr_ia_cv_avg_scores, fold_nerr_ia_ks)

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=fold_andcg_ks, list_cutoffs=self.cutoffs,
                                                         metric='aNDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_err_ia_ks, list_cutoffs=self.cutoffs,
                                                         metric='ERR-IA'))
        list_metric_strs.append(metric_results_to_string(list_scores=fold_nerr_ia_ks, list_cutoffs=self.cutoffs,
                                                         metric='nERR-IA'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {}\n\t{}".format(model_id, str(fold_k), metric_string))

        p_ndeval = subprocess.Popen(['../../ptranking/metric/srd/ndeval',
                                     '../../ptranking/metric/srd/WT_Div_0912_Implicit_qrels.txt',
                                    run_fold_k_dir+'/fold_run.txt'], shell=False, stdout=subprocess.PIPE, bufsize=-1)
        output_eval_q = p_ndeval.communicate()
        #print(output_eval_q)
        output_eval_q = output_eval_q[-2].decode().split("\n")[-2]
        output_eval_q = output_eval_q.split(',')
        #print('output_eval_q\n', output_eval_q)
        err_ia_5, err_ia_10, err_ia_20    = float(output_eval_q[2]), float(output_eval_q[3]), float(output_eval_q[4])
        nerr_ia_5, nerr_ia_10, nerr_ia_20 = float(output_eval_q[5]), float(output_eval_q[6]), float(output_eval_q[7])
        andcg_5, andcg_10, andcg_20       = float(output_eval_q[11]), float(output_eval_q[12]), float(output_eval_q[13])

        ndeval_err_ia_ks = np.asarray([err_ia_5, err_ia_10, err_ia_20])
        ndeval_nerr_ia_ks = np.asarray([nerr_ia_5, nerr_ia_10, nerr_ia_20])
        ndeval_andcg_ks = np.asarray([andcg_5, andcg_10, andcg_20])

        self.ndeval_err_ia_cv_avg_scores = np.add(self.ndeval_err_ia_cv_avg_scores, ndeval_err_ia_ks)
        self.ndeval_nerr_ia_cv_avg_scores = np.add(self.ndeval_nerr_ia_cv_avg_scores, ndeval_nerr_ia_ks)
        self.ndeval_andcg_cv_avg_scores = np.add(self.ndeval_andcg_cv_avg_scores, ndeval_andcg_ks)

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=ndeval_andcg_ks, list_cutoffs=self.ndeval_cutoffs,
                                                         metric='aNDCG(ndeval)'))
        list_metric_strs.append(metric_results_to_string(list_scores=ndeval_err_ia_ks, list_cutoffs=self.ndeval_cutoffs,
                                                         metric='ERR-IA(ndeval)'))
        list_metric_strs.append(metric_results_to_string(list_scores=ndeval_nerr_ia_ks, list_cutoffs=self.ndeval_cutoffs,
                                                         metric='nERR-IA(ndeval)'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {} (ndeval)\n\t{}".format(model_id, str(fold_k), metric_string))


    def get_cv_performance(self):
        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - self.time_begin)

        andcg_cv_avg_scores = np.divide(self.andcg_cv_avg_scores, self.fold_num)
        err_ia_cv_avg_scores = np.divide(self.err_ia_cv_avg_scores, self.fold_num)
        nerr_ia_cv_avg_scores = np.divide(self.nerr_ia_cv_avg_scores, self.fold_num)

        eval_prefix = str(self.fold_num) + '-fold cross validation scores:' if self.do_validation \
                      else str(self.fold_num) + '-fold average scores:'

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=andcg_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='aNDCG'))
        list_metric_strs.append(metric_results_to_string(list_scores=err_ia_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='ERR-IA'))
        list_metric_strs.append(metric_results_to_string(list_scores=nerr_ia_cv_avg_scores, list_cutoffs=self.cutoffs,
                                                         metric='nERR-IA'))

        metric_string = '\n'.join(list_metric_strs)
        print("\n{} {}\n{}".format(self.model_id, eval_prefix, metric_string))
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        if self.reproduce:
            ndeval_err_ia_cv_avg_scores = np.divide(self.ndeval_err_ia_cv_avg_scores, self.fold_num)
            ndeval_nerr_ia_cv_avg_scores = np.divide(self.ndeval_nerr_ia_cv_avg_scores, self.fold_num)
            ndeval_andcg_cv_avg_scores = np.divide(self.ndeval_andcg_cv_avg_scores, self.fold_num)

            list_metric_strs = []
            list_metric_strs.append(metric_results_to_string(list_scores=ndeval_andcg_cv_avg_scores,
                                                             list_cutoffs=self.ndeval_cutoffs, metric='aNDCG(ndeval)'))
            list_metric_strs.append(metric_results_to_string(list_scores=ndeval_err_ia_cv_avg_scores,
                                                             list_cutoffs=self.ndeval_cutoffs, metric='ERR-IA(ndeval)'))
            list_metric_strs.append(metric_results_to_string(list_scores=ndeval_nerr_ia_cv_avg_scores,
                                                             list_cutoffs=self.ndeval_cutoffs, metric='nERR-IA(ndeval)'))
            metric_string = '\n'.join(list_metric_strs)
            print("\n{} {}(ndeval)\n{}".format(self.model_id, eval_prefix, metric_string))

            #print(self.list_per_q_andcg)
            torch_mat_per_q_andcg = torch.cat(self.list_per_q_andcg, dim=0)
            #print('torch_mat_per_q_andcg', torch_mat_per_q_andcg.size())
            mat_per_q_andcg = torch_mat_per_q_andcg.data.numpy()
            pickle_save(target=mat_per_q_andcg, file=self.dir_run + '_'.join([self.model_id, 'all_fold_andcg_at_ks_per_q.np']))

        return andcg_cv_avg_scores


class DivSummaryTape(object):
    """
    Using multiple metrics to perform epoch-wise evaluation on train-data, validation-data, test-data
    """
    def __init__(self, do_validation, cutoffs, gpu):
        self.gpu = gpu
        self.cutoffs = cutoffs
        self.list_epoch_loss = []
        if do_validation: self.list_fold_k_vali_track = []
        self.list_fold_k_train_track, self.list_fold_k_test_track = [], []

    def epoch_summary(self, torch_epoch_k_loss, ranker, train_data, vali_data, test_data):
        fold_k_epoch_k_loss = torch_epoch_k_loss.cpu().numpy() if self.gpu else torch_epoch_k_loss.data.numpy()
        self.list_epoch_loss.append(fold_k_epoch_k_loss)

        train_avg_ndcg_at_ks, train_avg_err_ia_at_ks, train_avg_nerr_ia_at_ks = \
            ranker.srd_performance_at_ks(test_data=train_data, ks=self.cutoffs, device='cpu')

        test_avg_ndcg_at_ks, test_avg_err_ia_at_ks, test_avg_nerr_ia_at_ks = \
            ranker.srd_performance_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu')

        vali_avg_ndcg_at_ks, vali_avg_err_ia_at_ks, vali_avg_nerr_ia_at_ks = \
            ranker.srd_performance_at_ks(test_data=vali_data, ks=self.cutoffs, device='cpu')

        # TBA