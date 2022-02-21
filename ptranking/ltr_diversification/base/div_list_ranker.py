
import os
import copy

import torch

from ptranking.base.utils import get_stacked_FFNet
from ptranking.base.list_ranker import MultiheadAttention, PositionwiseFeedForward, Encoder, EncoderLayer, ListNeuralRanker

dc = copy.deepcopy

class DivListNeuralRanker(ListNeuralRanker):
    '''
    A univariate scoring function for diversified ranking, where listwise information is integrated.
    '''
    def __init__(self, id='DivListNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(DivListNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']

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
        if self.gpu:
            encoder = encoder.to(self.device)
            uni_sf = uni_sf.to(self.device)

        list_sf = {'encoder': encoder, 'uni_sf': uni_sf}
        return list_sf

    def div_forward(self, q_repr, doc_reprs):
        latent_cross_reprs = q_repr * doc_reprs
        # TODO is it OK if using expand as boradcasting?
        cat_1st_reprs = torch.cat((q_repr.expand(doc_reprs.size(0), -1), doc_reprs, latent_cross_reprs), 1)

        if 'AllRank' == self.encoder_type:
            #batch_FC_mappings = self.list_sf['head_ffnns'](cat_reprs)
            batch_encoder_mappings = self.list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0))

        elif 'DASALC' == self.encoder_type:
            #batch_FC_mappings = self.list_sf['head_ffnns'](cat_reprs)
            batch_encoder_mappings = self.list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0))

        elif 'AttnDIN' == self.encoder_type:
            #batch_FC_mappings = self.list_sf['head_ffnns'](cat_reprs)  # -> the same shape as the output of encoder
            batch_encoder_mappings = self.list_sf['encoder'](torch.unsqueeze(cat_1st_reprs, dim=0)) # the batch dimension is required
        else:
            raise NotImplementedError

        encoder_mappings = torch.squeeze(batch_encoder_mappings, dim=0)
        cat_2nd_reprs = torch.cat((q_repr.expand(doc_reprs.size(0), -1), doc_reprs, latent_cross_reprs, encoder_mappings), dim=1)
        batch_preds = self.list_sf['uni_sf'](cat_2nd_reprs)
        batch_preds = batch_preds.view(1, -1)  # [num_docs, 1] -> [batch, ranking_size]

        return batch_preds

    def get_parameters(self):
        all_parameters = list(self.list_sf['encoder'].parameters()) +\
                         list(self.list_sf['uni_sf'].parameters())
        return all_parameters

    def eval_mode(self):
        self.list_sf['encoder'].eval()
        self.list_sf['uni_sf'].eval()

    def train_mode(self):
        self.list_sf['encoder'].train(mode=True)
        self.list_sf['uni_sf'].train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save({"encoder": self.list_sf['encoder'].state_dict(),
                    "uni_sf": self.list_sf['uni_sf'].state_dict()}, dir + name)

    def load(self, file_model, **kwargs):
        checkpoint = torch.load(file_model)
        self.list_sf['encoder'].load_state_dict(checkpoint["encoder"])
        self.list_sf['uni_sf'].load_state_dict(checkpoint["uni_sf"])
