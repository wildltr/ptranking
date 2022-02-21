#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product

import torch
import torch.nn.functional as F

from ptranking.metric.srd.diversity_metric import SRD_METRIC
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.metric.srd.diversity_metric import get_delta_alpha_dcg
from ptranking.ltr_diversification.base.div_mdn_ranker import DivMDNRanker
from ptranking.ltr_diversification.score_and_sort.daletor import alphaDCG_as_a_loss
from ptranking.ltr_diversification.util.div_lambda_utils import get_prob_pairwise_comp_probs
from ptranking.ltr_diversification.util.prob_utils import get_diff_normal, resort_normal_matrix
from ptranking.ltr_diversification.util.prob_utils import get_expected_rank, get_expected_rank_const

# todo inspiring note:
#  (1) Add variance reduction operation
'''
1> samples_per_query == 1 leads to a better results ? due to expand()
2> utility_gap == True leads to poor results?
3> needs to be well tuned and checked ???
4> Mixture Density Networks for confidence ranking, with a learnable sigma, ? for lambdaRank, mart?
5> expected rank for directly optimising metrics, such as nDCG
6> TODO using a group of child mdns, which are combined as a linear combination.
'''

def alpha_dcg_as_a_loss(top_k=None, batch_mus=None, batch_vars=None, batch_cocos=None, q_doc_rele_mat=None,
                        opt_ideal=True, presort=False, beta=0.5, const=False, const_var=None):
    '''
    Alpha_nDCG as the optimization objective.
    @param top_k:
    @param batch_mus:
    @param batch_vars:
    @param batch_cocos:
    @param q_doc_rele_mat:
    @param opt_ideal:
    @param presort:
    @param beta:
    @return:
    '''
    if const:
        batch_expt_ranks, batch_Phi0_subdiag = \
            get_expected_rank_const(batch_mus=batch_mus, const_var=const_var, return_cdf=True)
    else:
        batch_expt_ranks, batch_Phi0_subdiag = \
            get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos, return_cdf=True)

    if opt_ideal:
        assert presort is True
        used_batch_expt_ranks = batch_expt_ranks
        used_q_doc_rele_mat = q_doc_rele_mat
        used_batch_indicators = batch_Phi0_subdiag  # the diagonal elements are zero
    else:
        batch_ascend_expt_ranks, batch_resort_inds = torch.sort(batch_expt_ranks, dim=1, descending=False)
        used_batch_expt_ranks = batch_ascend_expt_ranks
        used_batch_indicators = torch.gather(batch_Phi0_subdiag, dim=1,
                                index=torch.unsqueeze(batch_resort_inds.expand(batch_Phi0_subdiag.size(0), -1), dim=0))
        used_q_doc_rele_mat = torch.gather(q_doc_rele_mat, dim=1,
                                          index=batch_resort_inds.expand(q_doc_rele_mat.size(0), -1))

    _used_q_doc_rele_mat = torch.unsqueeze(used_q_doc_rele_mat, dim=1)
    # duplicate w.r.t. each subtopic -> [num_subtopics, ranking_size, ranking_size]
    batch_q_doc_rele_mat = _used_q_doc_rele_mat.expand(-1, used_q_doc_rele_mat.size(1), -1)
    prior_cover_cnts = torch.sum(used_batch_indicators * batch_q_doc_rele_mat, dim=2)  # [num_subtopics,num_docs]

    batch_per_subtopic_gains = used_q_doc_rele_mat * torch.pow((1.0 - beta), prior_cover_cnts) \
                               / torch.log2(1.0 + used_batch_expt_ranks)
    batch_global_gains = torch.sum(batch_per_subtopic_gains, dim=1)

    if top_k is None:
        alpha_DCG = torch.sum(batch_global_gains)
    else:
        alpha_DCG = torch.sum(batch_global_gains[0:top_k])

    batch_loss = -alpha_DCG

    return batch_loss

def err_ia_as_a_loss(top_k=None, batch_mus=None, batch_vars=None, batch_cocos=None, q_doc_rele_mat=None,
                     opt_ideal=True, presort=False, max_label=1.0, device=None, const=False, const_var=None):
    '''
    ERR-IA as the optimization objective.
    @param top_k:
    @param batch_mus:
    @param batch_vars:
    @param batch_cocos:
    @param q_doc_rele_mat:
    @param opt_ideal:
    @param presort:
    @return:
    '''
    ranking_size = q_doc_rele_mat.size(1)
    #max_label = torch.max(q_doc_rele_mat)
    t2 = torch.tensor([2.0], dtype=torch.float, device=device)

    #k = ranking_size if top_k is None else top_k
    #if self.norm:
    #    batch_ideal_err = torch_rankwise_err(q_doc_rele_mat, max_label=max_label, k=k, point=True, device=self.device)
    if const:
        batch_expt_ranks = get_expected_rank_const(batch_mus=batch_mus, const_var=const_var, return_cdf=False)
    else:
        batch_expt_ranks = get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos)

    if opt_ideal:
        assert presort is True
        used_batch_expt_ranks = batch_expt_ranks
        used_batch_labels = q_doc_rele_mat
    else:
        '''
        Sort the predicted ranks in a ascending natural order (i.e., 1, 2, 3, ..., n),
        the returned indices can be used to sort other vectors following the predicted order
        '''
        batch_ascend_expt_ranks, sort_indices = torch.sort(batch_expt_ranks, dim=1, descending=False)
        # sort labels according to the expected ranks
        # batch_sys_std_labels = torch.gather(batch_std_labels, dim=1, index=sort_indices)
        batch_sys_std_labels = torch.gather(q_doc_rele_mat, dim=1,index=sort_indices.expand(q_doc_rele_mat.size(0), -1))

        used_batch_expt_ranks = batch_ascend_expt_ranks
        used_batch_labels = batch_sys_std_labels

    if top_k is None:
        expt_ranks = 1.0 / used_batch_expt_ranks
        satis_pros = (torch.pow(t2, used_batch_labels) - 1.0) / torch.pow(t2, max_label)
        unsatis_pros = torch.ones_like(used_batch_labels) - satis_pros
        cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
        cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros)
        cascad_unsatis_pros[:, 1:ranking_size] = cum_unsatis_pros[:, 0:ranking_size - 1]

        expt_satis_ranks = expt_ranks * satis_pros * cascad_unsatis_pros
        batch_err = torch.sum(expt_satis_ranks, dim=1)
        #if self.norm:
        #    batch_nerr = batch_err / batch_ideal_err
        #    nerr_loss = -torch.sum(batch_nerr)
        #else:
        nerr_loss = -torch.sum(batch_err)
        return nerr_loss
    else:
        top_k_labels = used_batch_labels[:, 0:top_k]
        if opt_ideal:
            pos_rows = torch.arange(top_k_labels.size(0), dtype=torch.long)  # all rows
        else:
            non_zero_inds = torch.nonzero(torch.sum(top_k_labels, dim=1))
            zero_metric_value = False if non_zero_inds.size(0) > 0 else True
            if zero_metric_value:
                return None, zero_metric_value  # should not be optimized due to no useful training signal
            else:
                pos_rows = non_zero_inds[:, 0]

        expt_ranks = 1.0 / used_batch_expt_ranks[:, 0:top_k]
        satis_pros = (torch.pow(t2, top_k_labels) - 1.0) / torch.pow(t2, max_label)
        unsatis_pros = torch.ones_like(top_k_labels) - satis_pros
        cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
        cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros)
        cascad_unsatis_pros[:, 1:top_k] = cum_unsatis_pros[:, 0:top_k - 1]

        expt_satis_ranks = satis_pros[pos_rows, :] * cascad_unsatis_pros[pos_rows, :] * expt_ranks
        batch_err = torch.sum(expt_satis_ranks, dim=1)
        #if self.norm:
        #    batch_nerr = batch_err / batch_ideal_err[pos_rows, :]
        #    nerr_loss = -torch.sum(batch_nerr)
        #else:
        nerr_loss = -torch.sum(batch_err)
        return nerr_loss

def prob_lambda_loss(opt_id=None, batch_mus=None, batch_vars=None, batch_cocos=None, q_doc_rele_mat=None,
                     opt_ideal=True, presort=False, beta=0.5, device=None, norm=None):
    if 'PairCLS' == opt_id:
        batch_pairsub_mus, batch_pairsub_vars = get_diff_normal(batch_mus=batch_mus, batch_vars=batch_vars,
                                                                batch_cocos=batch_cocos)

        batch_p_ij, batch_std_p_ij = get_prob_pairwise_comp_probs(batch_pairsub_mus=batch_pairsub_mus,
                                                                  batch_pairsub_vars=batch_pairsub_vars,
                                                                  q_doc_rele_mat=q_doc_rele_mat)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')

        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))  # average across queries
        return batch_loss
    elif 'LambdaPairCLS' == opt_id:
        if opt_ideal:
            assert presort is True
            batch_pairsub_mus, batch_pairsub_vars = get_diff_normal(batch_mus=batch_mus, batch_vars=batch_vars,
                                                                    batch_cocos=batch_cocos)

            batch_p_ij, batch_std_p_ij = get_prob_pairwise_comp_probs(batch_pairsub_mus=batch_pairsub_mus,
                                                                  batch_pairsub_vars=batch_pairsub_vars,
                                                                  q_doc_rele_mat=q_doc_rele_mat)

            _delta_alpha_ndcgs = get_delta_alpha_dcg(ideal_q_doc_rele_mat=q_doc_rele_mat,
                                                     sys_q_doc_rele_mat=q_doc_rele_mat,
                                                     alpha=beta, device=device, normalization=norm)
            batch_delta_alpha_ndcg = torch.unsqueeze(_delta_alpha_ndcgs, dim=0)

            _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                                 target=torch.triu(batch_std_p_ij, diagonal=1),
                                                 weight=torch.triu(batch_delta_alpha_ndcg, diagonal=1),
                                                 reduction='none')
            batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))
            return batch_loss
        else:
            batch_expt_ranks, _batch_pairsub_mus, _batch_pairsub_vars = get_expected_rank(
                batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos, return_pairsub_paras=True)
            # predicted order
            _, batch_resort_inds = torch.sort(batch_expt_ranks, dim=1, descending=False)

            batch_resorted_pairsub_mus, batch_resorted_pairsub_vars = resort_normal_matrix(
                batch_mus=_batch_pairsub_mus, batch_vars=_batch_pairsub_vars, batch_resort_inds=batch_resort_inds)

            sys_q_doc_rele_mat = torch.gather(q_doc_rele_mat, dim=1,
                                              index=batch_resort_inds.expand(q_doc_rele_mat.size(0), -1))

            batch_p_ij, batch_std_p_ij = get_prob_pairwise_comp_probs(batch_pairsub_mus=batch_resorted_pairsub_mus,
                                                                      batch_pairsub_vars=batch_resorted_pairsub_vars,
                                                                      q_doc_rele_mat=sys_q_doc_rele_mat)


            _delta_alpha_ndcgs = get_delta_alpha_dcg(ideal_q_doc_rele_mat=q_doc_rele_mat,
                                                     sys_q_doc_rele_mat=sys_q_doc_rele_mat,
                                                     alpha=beta, device=device, normalization=norm)
            batch_delta_alpha_ndcg = torch.unsqueeze(_delta_alpha_ndcgs, dim=0)

            _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                                 target=torch.triu(batch_std_p_ij, diagonal=1),
                                                 weight=torch.triu(batch_delta_alpha_ndcg, diagonal=1),
                                                 reduction='none')
            batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

            return batch_loss


class DivProbRanker(DivMDNRanker):
    '''
    A family of diversified ranking methods based on Mixture Density Networks (MDN)
    TODO double overall checking, such as the presort parameter setting, beta, etc.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(DivProbRanker, self).__init__(id='DivProbRanker', sf_para_dict=sf_para_dict, gpu=gpu, device=device,
                                            K=model_para_dict['K'], cluster=model_para_dict['cluster'],
                                            sort_id=model_para_dict['sort_id'], limit_delta=model_para_dict['limit_delta'])
        '''
        'PairCLS': pairwise classification
        'LambdaPairCLS': pairwise classification weighted by in the lambda manner
        '''
        self.opt_id = model_para_dict['opt_id']
        assert self.opt_id in ['PairCLS', 'LambdaPairCLS', 'SuperSoft', 'Portfolio']

        self.beta = 0.5 # i.e., alpha in alpha-nDCG
        self.torch_zero = torch.tensor([0.0], device=self.device)

        if 'LambdaPairCLS' == self.opt_id:
            self.opt_ideal = model_para_dict['opt_ideal']
            self.norm = model_para_dict['norm']

        elif 'SuperSoft' == self.opt_id:
            self.norm = False
            self.opt_ideal = model_para_dict['opt_ideal']
            self.top_k = model_para_dict['top_k']
            self.metric = model_para_dict['metric']
            assert self.metric in SRD_METRIC

        elif self.opt_id == "Portfolio": #TODO conducat an exploration of the effectiveness of Portfolio for SRD & LTR
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer

            n_assets, max_weight = 50, 1.0
            #"""
            covmat_sqrt = cp.Parameter((n_assets, n_assets))
            rets = cp.Parameter(n_assets)
            alpha = cp.Parameter(nonneg=True)

            w = cp.Variable(n_assets)
            ret = rets @ w
            risk = cp.sum_squares(covmat_sqrt @ w)
            reg = alpha * (cp.norm(w) ** 2)

            prob = cp.Problem(cp.Maximize(ret - risk - reg),
                              [cp.sum(w) == 1,
                               w >= 0,
                               w <= max_weight
                               ])

            assert prob.is_dpp()
            self.cvxpylayer = CvxpyLayer(prob, parameters=[rets, covmat_sqrt, alpha], variables=[w])
            #"""

    def uniform_eval_setting(self, **kwargs):
        eval_dict = kwargs['eval_dict']
        if 'SuperSoft' == self.opt_id and \
                eval_dict["do_validation"] and not eval_dict['vali_metric']==self.metric:
            eval_dict['vali_metric'] = self.metric

    def div_custom_loss_function(self, batch_mus, batch_vars, q_doc_rele_mat, **kwargs):
        '''
        In the context of SRD, batch_size is commonly 1.
        @param batch_mus: [batch_size, ranking_size] each row represents the mean predictions for documents associated with the same query
        @param batch_vars: [batch_size, ranking_size] each row represents the variance predictions for documents associated with the same query
        @param batch_std_labels: [batch_size, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        # aiming for directly optimising alpha-nDCG over top-k documents
        assert 'presort' in kwargs and kwargs['presort'] is True
        presort = kwargs['presort']
        batch_cocos = kwargs['batch_cocos'] if 'batch_cocos' in kwargs else None

        if 'SuperSoft' == self.opt_id:
            if 'aNDCG' == self.metric:
                batch_loss = alpha_dcg_as_a_loss(top_k=self.top_k, batch_mus=batch_mus, batch_vars=batch_vars,
                                                 batch_cocos=batch_cocos, q_doc_rele_mat=q_doc_rele_mat,
                                                 opt_ideal=self.opt_ideal, presort=presort, beta=self.beta)
            elif 'nERR-IA' == self.metric:
                batch_loss = err_ia_as_a_loss(top_k=self.top_k, batch_mus=batch_mus, batch_vars=batch_vars,
                                              batch_cocos=batch_cocos, q_doc_rele_mat=q_doc_rele_mat,
                                              opt_ideal=self.opt_ideal, presort=presort, max_label=1.0,
                                              device=self.device)

        elif self.opt_id == 'LambdaPairCLS':
            batch_loss = prob_lambda_loss(opt_id=self.opt_id, batch_mus=batch_mus, batch_vars=batch_vars,
                                          batch_cocos=batch_cocos, q_doc_rele_mat=q_doc_rele_mat,
                                          opt_ideal=self.opt_ideal, presort=presort, beta=self.beta,
                                          device=self.device, norm=self.norm)
        elif self.opt_id == 'PairCLS':
            batch_loss = prob_lambda_loss(opt_id=self.opt_id, batch_mus=batch_mus, batch_vars=batch_vars,
                                          batch_cocos=batch_cocos, q_doc_rele_mat=q_doc_rele_mat)

        elif self.opt_id == "Portfolio":
            rets = batch_mus
            n_samples, n_assets = rets.shape
            covmat_sqrt = batch_cocos

            alpha = torch.tensor([0.01], device=self.device)
            gamma_sqrt = torch.tensor([0.1], device=self.device)

            gamma_sqrt_ = gamma_sqrt.repeat((1, n_assets * n_assets)).view(n_samples, n_assets, n_assets)
            alpha_abs = torch.abs(alpha)  # it needs to be nonnegative

            #print('rets', rets.size())
            #print('gamma_sqrt_', gamma_sqrt_.size())
            #print('covmat_sqrt', covmat_sqrt.size())
            #print('alpha_abs', alpha_abs)

            batch_preds = self.cvxpylayer(rets, gamma_sqrt_ * covmat_sqrt, alpha_abs)[0]
            #print('batch_preds', batch_preds)

            #print('q_doc_rele_mat', q_doc_rele_mat.size())
            batch_loss = alphaDCG_as_a_loss(batch_preds=batch_preds, q_doc_rele_mat=q_doc_rele_mat,
                                            rt=10, top_k=10, device=self.device)

        else:
            raise NotImplementedError

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of DivProbRanker ######

class DivProbRankerParameter(ModelParameter):
    ''' Parameter class for DivProbRanker '''
    def __init__(self, debug=False, para_json=None):
        super(DivProbRankerParameter, self).__init__(model_id='DivProbRanker', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """ Default parameter setting for DivProbRanker """
        if self.use_json:
            top_k = self.json_dict['top_k'][0]
            opt_id = self.json_dict['opt_id'][0]
            K = self.json_dict['K'][0]
            cluster = self.json_dict['cluster'][0]
            sort_id = self.json_dict['sort_id'][0]
            limit_delta = self.json_dict['limit_delta'][0]
            opt_ideal = self.json_dict['opt_ideal'][0]
            metric = self.json_dict['metric'][0]
            norm = self.json_dict['norm'][0]
            self.probrank_para_dict = dict(model_id=self.model_id, K=K, cluster=cluster, sort_id=sort_id,
                                           top_k=top_k, opt_id=opt_id, limit_delta=limit_delta, metric=metric,
                                           opt_ideal=opt_ideal, norm=norm)
        else:
            # LambdaPairCLS, PairCLS, SuperSoft; sort_id: ExpRele, RERAR, RiskAware
            self.probrank_para_dict = dict(model_id=self.model_id, K=1, cluster=False, sort_id='ExpRele',
                                           top_k=None, opt_id='SuperSoft', limit_delta=0.01, metric='nERR-IA',
                                           opt_ideal=True, norm=True)
        return self.probrank_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        """
        # using specified para-dict or inner para-dict
        probrank_para_dict = given_para_dict if given_para_dict is not None else self.probrank_para_dict

        s1 = ':' if log else '_'
        K, cluster, opt_id = probrank_para_dict['K'], probrank_para_dict['cluster'], probrank_para_dict['opt_id']
        sort_id, limit_delta = probrank_para_dict['sort_id'], probrank_para_dict['limit_delta']

        if cluster:
            probrank_paras_str = s1.join([str(K), 'CS', opt_id])
        else:
            probrank_paras_str = s1.join([str(K), opt_id])

        probrank_paras_str = s1.join([probrank_paras_str, sort_id])

        if limit_delta is not None:
            probrank_paras_str = s1.join([probrank_paras_str, '{:,g}'.format(limit_delta)])

        if 'LambdaPairCLS' == opt_id:
            norm = probrank_para_dict['norm']
            probrank_paras_str = s1.join([probrank_paras_str, 'Norm']) if norm else probrank_paras_str
            opt_ideal = probrank_para_dict['opt_ideal']
            probrank_paras_str = s1.join([probrank_paras_str, 'OptIdeal']) if opt_ideal else probrank_paras_str
        elif 'SuperSoft' == opt_id:
            opt_ideal = probrank_para_dict['opt_ideal']
            probrank_paras_str = s1.join([probrank_paras_str, 'OptIdeal']) if opt_ideal else probrank_paras_str

            top_k = probrank_para_dict['top_k']
            if top_k is None:
                probrank_paras_str = s1.join([probrank_paras_str, 'Full'])
            else:
                probrank_paras_str = s1.join([probrank_paras_str, str(top_k)])

            metric = probrank_para_dict['metric']
            s1.join([probrank_paras_str, metric])

        return probrank_paras_str

    def grid_search(self):
        """ Iterator of parameter settings for MiDeExpectedUtility """
        if self.use_json:
            choice_topk = self.json_dict['top_k']
            choice_opt_id = self.json_dict['opt_id']
            choice_K = self.json_dict['K']
            choice_cluster = self.json_dict['cluster']
            choice_sort_id = self.json_dict['sort_id']
            choice_limit_delta = self.json_dict['limit_delta']
            choice_opt_ideal = self.json_dict['opt_ideal']
            choice_metric = self.json_dict['metric']
            choice_norm = self.json_dict['norm']
        else:
            choice_topk = [10] if self.debug else [10]
            choice_opt_id = ['SuperSoft'] if self.debug else ['SuperSoft', 'PairCLS', 'LambdaPairCLS']
            choice_K = [5]
            choice_cluster = [False]
            choice_sort_id = ['ExpRele']
            choice_limit_delta = [None, 0.1]
            choice_opt_ideal = [True] if self.debug else [True]
            choice_metric = ['aNDCG']  # 'aNDCG', 'nERR-IA'
            choice_norm = [True] if self.debug else [True]

        for K, cluster, opt_id, sort_id, limit_delta in \
                product(choice_K, choice_cluster, choice_opt_id, choice_sort_id, choice_limit_delta):
            self.probrank_para_dict = dict(model_id=self.model_id, K=K, cluster=cluster, opt_id=opt_id, sort_id=sort_id,
                                           limit_delta=limit_delta)
            if opt_id == 'PairCLS':
                yield self.probrank_para_dict
            elif opt_id == 'LambdaPairCLS': # top-k is not needed, due to the requirement of pairwise swapping
                for opt_ideal, norm in product(choice_opt_ideal, choice_norm):
                    inner_para_dict = dict()
                    inner_para_dict['opt_ideal'] = opt_ideal
                    inner_para_dict['norm'] = norm
                    self.probrank_para_dict.update(inner_para_dict)
                    yield self.probrank_para_dict
            elif opt_id == 'SuperSoft':
                for top_k, metric, opt_ideal in product(choice_topk, choice_metric, choice_opt_ideal):
                    inner_para_dict = dict()
                    inner_para_dict['top_k'] = top_k
                    inner_para_dict['metric'] = metric
                    inner_para_dict['opt_ideal'] = opt_ideal
                    self.probrank_para_dict.update(inner_para_dict)
                    yield self.probrank_para_dict
            else:
                raise NotImplementedError
