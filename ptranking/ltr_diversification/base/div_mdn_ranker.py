
import os
import copy

import torch

from ptranking.base.utils import get_stacked_FFNet
from ptranking.ltr_diversification.util.prob_utils import get_expected_rank
from ptranking.ltr_diversification.util.sim_utils import batch_cosine_similarity
from ptranking.ltr_diversification.base.diversity_ranker import DiversityNeuralRanker
from ptranking.base.list_ranker import MultiheadAttention, PositionwiseFeedForward, Encoder, EncoderLayer

dc = copy.deepcopy

# TODO cluster -> group boosting

SORT_ID = ['ExpRele', 'RERAR', 'RiskAware']

class DivMDNRanker(DiversityNeuralRanker):
    '''
    The diversified neural ranker based on Mixture Density Networks
    '''
    def __init__(self, id='DivMDNRanker', sf_para_dict=None,
                 K=1, cluster=False, sort_id='ExpRele', limit_delta=None, weight_decay=1e-3, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

        self.sf_para_dict = sf_para_dict
        self.sf_id = sf_para_dict['sf_id']

        self.opt, self.lr = sf_para_dict['opt'], sf_para_dict['lr']
        self.weight_decay = weight_decay

        if self.sf_id.startswith('pointsf'):  # corresponding to the concatenation operation, i.e., q_repr + doc_repr + latent_cross
            self.sf_para_dict[self.sf_id]['num_features'] *= 3
        elif self.sf_id.startswith('listsf'):
            self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']

        assert K >= 1
        self.K, self.cluster = K, cluster

        # sort_id: ExpRele, RERAR, RiskAware, reciprocal_expected_rank_as_relevance (RERAR)
        assert sort_id in SORT_ID
        self.sort_id = sort_id
        self.limit_delta = limit_delta
        self.b = 0.1

        if self.cluster: # using a group of independent learners (i.e., scoring functions)
            assert not self.sf_id.endswith("co")  # not supported
            assert self.K >=2
            self.sf_para_dict[self.sf_id]['out_dim'] = 3
        else:
            if 1 == self.K: # mu and sigma
                self.sf_para_dict[self.sf_id]['out_dim'] = 2
            else: # mixing coefficient, mu, sigma per component
                self.sf_para_dict[self.sf_id]['out_dim'] = 3 * self.K

        self.stop_check_freq = 10

    def init(self):
        if self.cluster:
            self.group_sf = []
            if self.sf_id.startswith('pointsf'):
                for i in range(self.K):
                    point_sf_i = self.config_point_neural_scoring_function()
                    self.group_sf.append(point_sf_i)

            elif self.sf_id.startswith('listsf'):
                for i in range(self.K):
                    list_sf_i = self.config_list_neural_scoring_function()
                    self.group_sf.append(list_sf_i)
        else:
            if self.sf_id.startswith('pointsf'):
                self.point_sf = self.config_point_neural_scoring_function()
            elif self.sf_id.startswith('listsf'):
                self.list_sf = self.config_list_neural_scoring_function()

        self.config_optimizer()

    def get_diff_normal(self, batch_mus, batch_vars, batch_cocos=None):
        '''
        The difference of two normal random variables is another normal random variable. In particular, we consider two
        cases: (1) correlated (2) independent.
        @param batch_mus: the predicted mean
        @param batch_vars: the predicted variance
        @param batch_cocos: the predicted correlation coefficient in [-1, 1], which is formulated as the cosine-similarity of corresponding vectors.
        @return: the mean, variance of the result normal variable.
        '''
        # mu_i - mu_j
        batch_pairsub_mus = torch.unsqueeze(batch_mus, dim=2) - torch.unsqueeze(batch_mus, dim=1)

        # variance w.r.t. S_i - S_j, which is equal to: (1)sigma^2_i + sigma^2_j - \rou_ij*sigma_i*sigma_j  (2) sigma^2_i + sigma^2_j
        if batch_cocos is not None:
            batch_std_vars = torch.pow(batch_vars, .5)
            batch_pairsub_vars = torch.unsqueeze(batch_vars, dim=2) + torch.unsqueeze(batch_vars, dim=1) - \
                                 batch_cocos * torch.bmm(torch.unsqueeze(batch_std_vars, dim=2),
                                                         torch.unsqueeze(batch_std_vars, dim=1))
        else:
            batch_pairsub_vars = torch.unsqueeze(batch_vars, dim=2) + torch.unsqueeze(batch_vars, dim=1)

        return batch_pairsub_mus, batch_pairsub_vars

    def ini_listsf(self, num_features=None, n_heads=2, encoder_layers=2, dropout=0.1, encoder_type=None,
                   ff_dims=[256, 128, 64], out_dim=1, AF='R', TL_AF='GE', apply_tl_af=False,
                   BN=True, bn_type=None, bn_affine=False):
        '''
        Initialization the univariate scoring function for diversified ranking.
        '''
        # the input size according to the used dataset
        encoder_num_features, fc_num_features = num_features * 3, num_features * 6

        ''' Component-1: stacked multi-head self-attention (MHSA) blocks '''
        mhsa = MultiheadAttention(hid_dim=encoder_num_features, n_heads=n_heads, dropout=dropout, device=self.device)

        if 'AllRank' == encoder_type:
            fc = PositionwiseFeedForward(encoder_num_features, hid_dim=encoder_num_features, dropout=dropout)
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_num_features, mhsa=dc(mhsa), encoder_type=encoder_type,
                                                 fc=fc, dropout=dropout),
                              num_layers=encoder_layers, encoder_type=encoder_type)

        elif 'DASALC' == encoder_type: # we note that feature normalization strategy is different from AllRank
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_num_features, mhsa=dc(mhsa), encoder_type=encoder_type),
                              num_layers=encoder_layers, encoder_type=encoder_type)

        elif 'AttnDIN' == encoder_type:
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_num_features, mhsa=dc(mhsa), encoder_type=encoder_type),
                              num_layers=encoder_layers, encoder_type=encoder_type)
        else:
            raise NotImplementedError

        ''' Component-2: univariate scoring function '''
        uni_ff_dims = [fc_num_features]
        uni_ff_dims.extend(ff_dims)
        uni_ff_dims.append(out_dim)
        uni_sf = get_stacked_FFNet(ff_dims=uni_ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af,
                                   BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)

        ''' Component-3: stacked feed-forward layers for co-variance prediction '''
        if self.sf_id.endswith("co"):
            co_ff_dims = [fc_num_features]
            co_ff_dims.extend(ff_dims)
            co_ffnns = get_stacked_FFNet(ff_dims=co_ff_dims, AF=AF, apply_tl_af=False, dropout=dropout,
                                         BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)

        if self.gpu:
            encoder = encoder.to(self.device)
            uni_sf = uni_sf.to(self.device)
            if self.sf_id.endswith("co"): co_ffnns = co_ffnns.to(self.device)

        if self.sf_id.endswith("co"):
            list_sf = {'encoder': encoder, 'uni_sf': uni_sf, 'co_ffnns':co_ffnns}
        else:
            list_sf = {'encoder': encoder, 'uni_sf': uni_sf}
        return list_sf

    def get_parameters(self):
        if self.cluster:
            if self.sf_id.startswith('pointsf'):
                return (paras for sf in self.group_sf for paras in sf.parameters())
            elif self.sf_id.startswith('listsf'):
                all_parameters = []
                for list_sf_i in self.group_sf:
                    for paras in list_sf_i['encoder'].parameters():
                        all_parameters.append(paras)
                    for paras in list_sf_i['uni_sf'].parameters():
                        all_parameters.append(paras)

                return all_parameters
        else:
            if self.sf_id.startswith('pointsf'):
                return self.point_sf.parameters()
            elif self.sf_id.startswith('listsf'):
                all_parameters = list(self.list_sf['encoder'].parameters()) + \
                                 list(self.list_sf['uni_sf'].parameters())
                if self.sf_id.endswith("co"): all_parameters += list(self.list_sf['co_ffnns'].parameters())
                return all_parameters

    def eval_mode(self):
        if self.cluster:
            if self.sf_id.startswith('pointsf'):
                for i in range(self.K):
                    self.group_sf[i].eval()
            elif self.sf_id.startswith('listsf'):
                for i in range(self.K):
                    self.group_sf[i]['encoder'].eval()
                    self.group_sf[i]['uni_sf'].eval()
        else:
            if self.sf_id.startswith('pointsf'):
                self.point_sf.eval()
            elif self.sf_id.startswith('listsf'):
                self.list_sf['encoder'].eval()
                self.list_sf['uni_sf'].eval()
                if self.sf_id.endswith("co"): self.list_sf['co_ffnns'].eval()

    def train_mode(self):
        if self.cluster:
            if self.sf_id.startswith('pointsf'):
                for i in range(self.K):
                    self.group_sf[i].train(mode=True)
            elif self.sf_id.startswith('listsf'):
                for i in range(self.K):
                    self.group_sf[i]['encoder'].train(mode=True)
                    self.group_sf[i]['uni_sf'].train(mode=True)
        else:
            if self.sf_id.startswith('pointsf'):
                self.point_sf.train(mode=True)
            elif self.sf_id.startswith('listsf'):
                self.list_sf['encoder'].train(mode=True)
                self.list_sf['uni_sf'].train(mode=True)
                if self.sf_id.endswith("co"): self.list_sf['co_ffnns'].train(mode=True)

    def div_switch_forward(self, q_repr, doc_reprs, sf=None):
        latent_cross_reprs = q_repr * doc_reprs
        cat_1st_reprs = torch.cat((q_repr.expand(doc_reprs.size(0), -1), doc_reprs, latent_cross_reprs), 1)

        if self.sf_id.startswith('pointsf'):
            point_sf = self.point_sf if sf is None else sf
            _batch_pred = point_sf(cat_1st_reprs)
            batch_pred = torch.unsqueeze(_batch_pred, dim=0)
            return batch_pred

        elif self.sf_id.startswith('listsf'):
            list_sf = self.list_sf if sf is None else sf

            if 'AllRank' == self.encoder_type:
                batch_encoder_mappings = list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0))

            elif 'DASALC' == self.encoder_type:
                batch_encoder_mappings = list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0))

            elif 'AttnDIN' == self.encoder_type:
                batch_encoder_mappings = list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0))  # the batch dimension is required
            else:
                raise NotImplementedError

            encoder_mappings = torch.squeeze(batch_encoder_mappings, dim=0)
            cat_2nd_reprs = torch.cat((q_repr.expand(doc_reprs.size(0), -1), doc_reprs, latent_cross_reprs, encoder_mappings), dim=1)
            _batch_pred = self.list_sf['uni_sf'](cat_2nd_reprs)
            batch_pred = torch.unsqueeze(_batch_pred, dim=0)

            if self.sf_id.endswith("co"): # computing the Correlation Coefficient based on listwise embeddings
                batch_cocos = batch_cosine_similarity(self.list_sf['co_ffnns'](torch.unsqueeze(cat_2nd_reprs, dim=0)))
                return batch_pred, batch_cocos
            else:
                return batch_pred


    def div_forward(self, q_repr, doc_reprs):
        '''
        '''
        batch_size = 1
        num_docs = doc_reprs.size(0)

        if self.cluster: # a cluster of scoring functions
            assert not self.sf_id.endswith("co")

            pool_pred = []
            for i in range(self.K):
                sf_i = self.group_sf[i]
                batch_pred_i = self.div_switch_forward( q_repr, doc_reprs, sf_i)
                batch_pred_i = batch_pred_i.view(batch_size, num_docs, -1) # -> [batch_size, num_docs, 3]
                pool_pred.append(batch_pred_i)

            batch_pred = torch.cat(pool_pred, dim=1)
            # aiming for consistence, i.e., [, , 3 * self.K], we view 1st K entries as weights, then mus, then std_vars
            batch_components = batch_pred.view(batch_size, num_docs, -1)

        else: # a single scoring function
            if self.sf_id.endswith("co"):
                batch_pred, batch_cocos = self.div_switch_forward(q_repr, doc_reprs)
            else:
                batch_pred = self.div_switch_forward(q_repr, doc_reprs)

            batch_components = batch_pred.view(batch_size, num_docs, -1)

        if 1 == self.K:
            batch_mus, batch_std_vars = torch.split(batch_components, split_size_or_sections=1, dim=2)
            if self.limit_delta is None:
                batch_vars = torch.exp(batch_std_vars)  # representing sigma^2
            else:
                batch_vars = torch.sigmoid(batch_std_vars) * self.limit_delta

            batch_mus, batch_vars = torch.squeeze(batch_mus, dim=2), torch.squeeze(batch_vars, dim=2)
        else:  # batch_components: [, , 3 * self.K], we view 1st K entries as weights, then mus, then std_vars
            batch_weights, batch_mu_i, batch_std_var_i = torch.split(batch_components, split_size_or_sections=self.K, dim=2)

            if self.limit_delta is None:
                batch_var_i = torch.exp(batch_std_var_i)  # representing sigma^2
            else:
                batch_var_i = torch.sigmoid(batch_std_var_i) * self.limit_delta

            batch_coefficients = torch.softmax(batch_weights, dim=2)
            batch_mus = torch.sum(batch_coefficients * batch_mu_i, dim=2)
            batch_vars = torch.sum(batch_coefficients * batch_var_i, dim=2)

        if self.sf_id.endswith("co"):
            return batch_mus, batch_vars, batch_cocos
        else:
            return batch_mus, batch_vars

    def div_predict(self, q_repr, doc_reprs):
        '''
        The relevance prediction. In the context of diversified ranking, the shape is interpreted as:
        @param q_repr:
        @param doc_reprs:
        @return:
        '''
        if self.sf_id.endswith("co"):
            batch_mus, batch_vars, batch_cocos = self.div_forward(q_repr, doc_reprs)
        else:
            batch_cocos = None
            batch_mus, batch_vars = self.div_forward(q_repr, doc_reprs)

        if 'RERAR' == self.sort_id: # reciprocal_expected_rank_as_relevance (RERAR)
            ''' Expected Ranks '''
            batch_expt_ranks = \
                get_expected_rank(batch_mus=batch_mus, batch_vars=batch_vars, batch_cocos=batch_cocos, return_cdf=False)

            batch_RERAR = 1.0 / batch_expt_ranks
            return batch_RERAR
        elif 'ExpRele' == self.sort_id:
            return batch_mus
        elif 'RiskAware' == self.sort_id: # TODO integrating coco for ranking
            return batch_mus - self.b*batch_vars
        else:
            raise NotImplementedError


    def div_train_op(self, q_repr, doc_reprs, q_doc_rele_mat, **kwargs):
        stop_training = False
        if self.sf_id.endswith("co"):
            batch_mus, batch_sigma_sqs, batch_cocos = self.div_forward(q_repr, doc_reprs)
            kwargs['batch_cocos'] = batch_cocos
            return self.div_custom_loss_function(batch_mus, batch_sigma_sqs, q_doc_rele_mat, **kwargs), stop_training
        else:
            batch_mus, batch_sigma_sqs = self.div_forward(q_repr, doc_reprs)
            return self.div_custom_loss_function(batch_mus, batch_sigma_sqs, q_doc_rele_mat, **kwargs), stop_training


    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        if self.cluster: # ''encoder':encoder, 'uni_sf'
            full_state_dict = dict()
            for i in range(self.K):
                sf_i = self.group_sf[i]
                if self.sf_id.startswith('pointsf'):
                    full_state_dict[str(i)] = sf_i.state_dict()
                elif self.sf_id.startswith('listsf'):
                    full_state_dict[str(i)] = {}
                    full_state_dict[str(i)]['encoder'] = sf_i['encoder'].state_dict()
                    full_state_dict[str(i)]['uni_sf'] = sf_i['uni_sf'].state_dict()

            torch.save(full_state_dict, dir + name)
        else:
            if self.sf_id.startswith('pointsf'):
                torch.save(self.point_sf.state_dict(), dir + name)
            elif self.sf_id.startswith('listsf'):
                full_state_dict = dict()
                full_state_dict['encoder'] = self.list_sf['encoder'].state_dict()
                full_state_dict['uni_sf'] = self.list_sf['uni_sf'].state_dict()
                if self.sf_id.endswith("co"): full_state_dict['co_ffnns'] = self.list_sf['co_ffnns'].state_dict()
                torch.save(full_state_dict, dir + name)

    def load(self, file_model, **kwargs):
        if 'context' in kwargs and kwargs['context']=='cpu_gpu': # loading a gpu-trained model to cpu
            full_state_dict = torch.load(file_model, map_location=torch.device('cpu'))
        else:
            full_state_dict = torch.load(file_model)

        if self.cluster:
            for i in range(self.K):
                sf_i = self.group_sf[i]
                if self.sf_id.startswith('pointsf'):
                    sf_i.load_state_dict(full_state_dict[str(i)])
                elif self.sf_id.startswith('listsf'):
                    sf_i['encoder'].load_state_dict(full_state_dict[str(i)]['encoder'])
                    sf_i['uni_sf'].load_state_dict(full_state_dict[str(i)]['uni_sf'])
        else:
            if self.sf_id.startswith('pointsf'):
                self.point_sf.load_state_dict(torch.load(file_model))
            elif self.sf_id.startswith('listsf'):
                self.list_sf['encoder'].load_state_dict(full_state_dict['encoder'])
                self.list_sf['uni_sf'].load_state_dict(full_state_dict['uni_sf'])
                if self.sf_id.endswith("co"): self.list_sf['co_ffnns'].load_state_dict(full_state_dict['co_ffnns'])
