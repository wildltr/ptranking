
import os

import torch

from ptranking.base.ranker import NeuralRanker
from ptranking.base.utils import get_stacked_FFNet

class PointNeuralRanker(NeuralRanker):
    '''
    A one-size-fits-all neural ranker.
    Given the documents associated with the same query, this ranker scores each document independently.
    '''
    def __init__(self, id='PointNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(PointNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        self.config_optimizer()

    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)

        return point_sf

    def get_parameters(self):
        return self.point_sf.parameters()

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=1, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        ff_dims = [num_features]
        for i in range(num_layers):
            ff_dims.append(h_dim)
        ff_dims.append(out_dim)

        point_sf = get_stacked_FFNet(ff_dims=ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af, dropout=dropout,
                                     BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)
        return point_sf


    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()

        _batch_preds = self.point_sf(batch_q_doc_vectors)
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def eval_mode(self):
        self.point_sf.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(torch.load(file_model, map_location=device))

    def get_tl_af(self):
        return self.sf_para_dict[self.sf_para_dict['sf_id']]['TL_AF']
