
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptranking.base.ranker import NeuralRanker
from ptranking.base.utils import get_stacked_FFNet

dc = copy.deepcopy

Encoder_Type = ['DASALC', 'AllRank', 'AttnDIN']
"""
DASALC refers to the work:
Zhen Qin, Le Yan, Honglei Zhuang, Yi Tay, Rama Kumar Pasumarthi, Xuanhui Wang, Mike Bendersky, and Marc Najork.
Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?. In Proceedings of ICLR, 2021.

AllRank refers to the work:
title={Context-Aware Learning to Rank with Self-Attention},
author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
year={2020}

AttnDIN refers to the work:
Rama Kumar Pasumarthi Honglei Zhuang Xuanhui Wang Mike Bendersky Marc Najork
Proceedings of the 2020 ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR 2020)

The subtle differences are that: 
DASALC:
1> {MSHA -> LayerNorm} x L-copies
2> latent corss is deployed
The finally adopted hyper-parameter setting is not clearly specified in the paper.

AttnDIN:
1> compared with DASALC, concatenation is deployed rather than latent corss, and Residual is specified
the finally adopted hyper-parameter setting:
For WEB30K, two self-attention layers with 100 neurons and two heads;
For Istella, two self-attention layers with 200 neurons and two heads for attn-DIN;
The univariate scoring function comprises of an input batch normalization layer, followed by 3 feedforward fully connected layers of sizes [1024, 512, 256] with batch normalization and
ReLU activations.

AllRank:
{LayerNorm -> MSHA -> Residual -> FC -> Residual} x L-copies
The Ablation study is shown in Table 5 of the paper.
"""

def make_clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    The encoder block of the transformer
    """
    def __init__(self, layer, num_layers, encoder_type=None):
        '''
        @param layer: one encoder layer, i.e., self-attention layer
        @param num_layers: number of self-attention layers
        @param encoder_type: type differentiation
        '''
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        self.layers = make_clones(layer, num_layers)
        if 'AllRank' == encoder_type:
            self.norm = LayerNorm(layer.hid_dim)

    def forward(self, x):
        '''
        Forward pass through the encoder block.
        @param x: input with a shape of [batch_size, ranking_size, num_features]
        @return:
        '''
        for layer in self.layers:
            x = layer(x)

        if 'AllRank' == self.encoder_type:
            return self.norm(x)

        elif self.encoder_type in ['AttnDIN', 'DASALC']:
            return x

        else:
            raise NotImplementedError

class EncoderLayer(nn.Module):
    """
    One single encoder block
    """
    def __init__(self, hid_dim, mhsa, encoder_type=None, fc=None, dropout=None):
        super(EncoderLayer, self).__init__()
        self.mhsa = mhsa
        self.hid_dim = hid_dim
        self.encoder_type = encoder_type
        if 'AllRank' == encoder_type:
            self.fc = fc
            self.sublayer_cont = make_clones(SublayerConnection(hid_dim=hid_dim, encoder_type=encoder_type, dropout=dropout), 2)
        elif encoder_type in ['AttnDIN', 'DASALC']:
            self.sublayer_cont = SublayerConnection(hid_dim=hid_dim, encoder_type=encoder_type)

    def forward(self, x):
        '''
        @param x:
        @return:
        '''
        if 'AllRank' == self.encoder_type:
            x = self.sublayer_cont[0](x, self.mhsa)
            return self.sublayer_cont[1](x, self.fc)

        elif self.encoder_type in ['AttnDIN', 'DASALC']:
            # self.sublayer(x, self.mhsa) # being equivalent
            return self.sublayer_cont(x, lambda x: self.mhsa(x))
        else:
            raise NotImplementedError


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, hid_dim, encoder_type=None, dropout=None):
        '''
        @param hid_dim: number of input/output features
        @param dropout: dropout probability
        '''
        super(SublayerConnection, self).__init__()
        self.encoder_type = encoder_type
        self.norm = LayerNorm(hid_dim=hid_dim)
        if 'AllRank' == encoder_type:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        Foward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        @param x: input with a shape of [batch_size, ranking_size, num_features]
        @param sublayer: the layer through which to pass the input prior to applying the sum
        @return: output with a shape of [batch_size, ranking_size, num_features]
        '''
        if 'AllRank' == self.encoder_type:
            return x + self.dropout(sublayer(self.norm(x)))
        elif 'DASALC' == self.encoder_type:
            # residual is not clearly mentioned, which is also not specified in Figure 1 of the paper
            return self.norm(sublayer(x))
        elif 'AttnDIN' == self.encoder_type:
            return self.norm(x + sublayer(x))
        else:
            raise NotImplementedError


class LayerNorm(nn.Module):
    """
    The layer-normalizaton module
    """
    def __init__(self, hid_dim, eps=1e-6):
        '''
        @param hid_dim: shape of normalised features
        @param eps: epsilon for standard deviation
        '''
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(hid_dim))
        self.b_2 = nn.Parameter(torch.zeros(hid_dim))
        self.eps = eps

    def forward(self, x):
        '''
        Forward pass through the layer normalization
        @param x: input shape, i.e., [batch_size, ranking_size, num_features]
        @return: normalized input with a shape of [batch_size, ranking_size, num_features]
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiheadAttention(nn.Module):
    """
    Multi-head attention block.
    """
    def __init__(self, hid_dim, n_heads, dropout=0.1, device=None):
        '''
        :param hid_dim: number of features, i.e., input/output dimensionality
        :param n_heads: number of heads
        :param dropout: dropout probability
        '''
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        # W_q matrix
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # W_k matrix
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # W_v matrix
        self.w_v = nn.Linear(hid_dim, hid_dim)

        '''
        E.g., equation-10 for DASALC
        '''
        self.fc = nn.Linear(hid_dim, hid_dim, bias=True)

        self.do_dropout = nn.Dropout(dropout)

        # scaling
        self.scale = torch.sqrt(torch.tensor([hid_dim // n_heads], dtype=torch.float, device=device))

    def forward(self, batch_rankings):
        '''
        Forward pass through the multi-head attention block.
        :param batch_rankings: [batch_size, ranking_size, num_features]
        :return:
        '''
        bsz = batch_rankings.shape[0]
        Q = self.w_q(batch_rankings)
        K = self.w_k(batch_rankings)
        V = self.w_v(batch_rankings)

        '''
        Here, split {K Q V} into multi-group attentions, thus a 4-dimensional matrix
        '''
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        ''' step-1
        Q * K^T / sqrt(d_k)
        [batch_size, n_heads, ranking_size, num_features_sub_head] 
        * [batch_size, n_heads, num_features_sub_head, ranking_size]
        = [batch_size, n_heads, ranking_size, ranking_size]
        '''
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        ''' step-2
        perform softmax -> dropout
        get attention [batch_size, n_heads, ranking_size, ranking_size]
        '''
        attention = self.do_dropout(torch.softmax(attention, dim=-1))

        ''' step-3
        multipy attention and V, and get the results of multi-heads
        [batch_size, n_heads, ranking_size, ranking_size] * [batch_size, n_heads, ranking_size, num_features_sub_head]
        = [batch_size, n_heads, ranking_size, num_features_sub_head]
        '''
        x = torch.matmul(attention, V)

        # transpose again for later concatenation -> [batch_size, ranking_size, n_heads, num_features_sub_head]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [64,12,6,50] -> [64,12,300]
        # -> [batch_size, ranking_size, num_features]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class PositionwiseFeedForward(nn.Module):
    """
    Fully connected feed-forward block.
    """
    def __init__(self, num_features, hid_dim, dropout=0.1):
        """
        :param num_features: input/output dimensionality
        :param hid_dim: hidden dimensionality
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(num_features, hid_dim)
        self.w2 = nn.Linear(hid_dim, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward block.
        :param x: input with a shape of [batch_size, ranking_size, num_features]
        :return: output with a shape of [batch_size, ranking_size, num_features]
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))


class ListNeuralRanker(NeuralRanker):
    '''
    Permutation equivariant neural ranker
    '''
    def __init__(self, id='ListNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(ListNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']

    def init(self):
        self.list_sf = self.config_list_neural_scoring_function()
        self.config_optimizer()

    def config_list_neural_scoring_function(self):
        list_sf = self.ini_listsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        return list_sf

    def get_parameters(self):
        all_parameters = list(self.list_sf['head_ffnns'].parameters()) +\
                         list(self.list_sf['encoder'].parameters()) +\
                         list(self.list_sf['tail_ffnns'].parameters())
        return all_parameters


    def ini_listsf(self, num_features=None, ff_dims=[128, 256, 512], out_dim=1, AF='R', TL_AF='GE', apply_tl_af=False,
                   BN=True, bn_type=None, bn_affine=False, n_heads=2, encoder_layers=3, dropout=0.1, encoder_type=None):
        '''
        Initialization of a permutation equivariant neural network
        '''
        ''' Component-1: stacked feed-forward layers for initial encoding '''
        head_ff_dims = [num_features]
        head_ff_dims.extend(ff_dims)
        head_ff_dims.append(num_features)

        head_ffnns = get_stacked_FFNet(ff_dims=head_ff_dims, AF=AF, TL_AF=AF, apply_tl_af=True, dropout=dropout,
                                       BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)

        ''' Component-2: stacked multi-head self-attention (MHSA) blocks '''
        encoder_dim = num_features
        mhsa = MultiheadAttention(hid_dim=encoder_dim, n_heads=n_heads, dropout=dropout, device=self.device)

        if 'AllRank' == encoder_type:
            fc = PositionwiseFeedForward(num_features, hid_dim=encoder_dim, dropout=dropout)
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_dim, mhsa=dc(mhsa), encoder_type=encoder_type,
                                                 fc=fc, dropout=dropout),
                              num_layers=encoder_layers, encoder_type=encoder_type)

        elif 'DASALC' == encoder_type: # we note that feature normalization strategy is different from AllRank
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_dim, mhsa=dc(mhsa), encoder_type=encoder_type),
                              num_layers=encoder_layers, encoder_type=encoder_type)

        elif 'AttnDIN' == encoder_type:
            encoder = Encoder(layer=EncoderLayer(hid_dim=encoder_dim, mhsa=dc(mhsa), encoder_type=encoder_type),
                              num_layers=encoder_layers, encoder_type=encoder_type)
        else:
            raise NotImplementedError

        ''' Component-3: stacked feed-forward layers for relevance prediction '''
        tail_ff_dims = [num_features]
        tail_ff_dims.extend(ff_dims)
        tail_ff_dims.append(out_dim)
        tail_ffnns = get_stacked_FFNet(ff_dims=tail_ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af,
                                       BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)

        if self.gpu:
            head_ffnns = head_ffnns.to(self.device)
            encoder = encoder.to(self.device)
            tail_ffnns = tail_ffnns.to(self.device)

        list_sf = {'head_ffnns': head_ffnns, 'encoder': encoder, 'tail_ffnns': tail_ffnns}
        return list_sf

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where the documents associated with the same query are scored jointly.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        if 'AllRank' == self.encoder_type:
            # deploy the same mapping for batch queries
            batch_FC_mappings = self.list_sf['head_ffnns'](batch_q_doc_vectors) # -> the same shape as the output of encoder
            batch_encoder_mappings = self.list_sf['encoder'](batch_FC_mappings)
            batch_preds = self.list_sf['tail_ffnns'](batch_encoder_mappings)

        elif 'DASALC' == self.encoder_type:
            batch_FC_mappings = self.list_sf['head_ffnns'](batch_q_doc_vectors) # -> the same shape as the output of encoder
            batch_encoder_mappings = self.list_sf['encoder'](batch_q_doc_vectors) # the input of encoder differs from DASALC
            latent_cross_mappings = (batch_encoder_mappings + 1.0) * batch_FC_mappings
            batch_preds = self.list_sf['tail_ffnns'](latent_cross_mappings)

        elif 'AttnDIN' == self.encoder_type:
            batch_FC_mappings = self.list_sf['head_ffnns'](batch_q_doc_vectors)  # -> the same shape as the output of encoder
            batch_encoder_mappings = self.list_sf['encoder'](batch_FC_mappings)  # the input of encoder differs from DASALC
            concat_mappings = batch_encoder_mappings + batch_q_doc_vectors
            batch_preds = self.list_sf['tail_ffnns'](concat_mappings)
        else:
            raise NotImplementedError

        batch_pred = torch.squeeze(batch_preds, dim=2)  # [batch, num_docs, 1] -> [batch, num_docs]
        return batch_pred

    def eval_mode(self):
        self.list_sf['head_ffnns'].eval()
        self.list_sf['encoder'].eval()
        self.list_sf['tail_ffnns'].eval()

    def train_mode(self):
        self.list_sf['head_ffnns'].train(mode=True)
        self.list_sf['encoder'].train(mode=True)
        self.list_sf['tail_ffnns'].train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save({"head_ffnns": self.list_sf['head_ffnns'].state_dict(),
                    "encoder": self.list_sf['encoder'].state_dict(),
                    "tail_ffnns": self.list_sf['tail_ffnns'].state_dict()}, dir + name)

    def load(self, file_model):
        checkpoint = torch.load(file_model)
        self.list_sf['head_ffnns'].load_state_dict(checkpoint["head_ffnns"])
        self.list_sf['encoder'].load_state_dict(checkpoint["encoder"])
        self.list_sf['tail_ffnns'].load_state_dict(checkpoint["tail_ffnns"])