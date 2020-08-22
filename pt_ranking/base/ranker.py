#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#from torch.nn.init import kaiming_normal_ as nr_init
from torch.nn.init import xavier_normal_ as nr_init

from pt_ranking.base.neural_utils import get_AF, ResidualBlock_FFNNs
from pt_ranking.ltr_global import global_gpu as gpu, global_device as device

'''
1. reset parameters optimizer for cross validataion
'''

'''
-- AbstractNeuralRanker
---- NeuralRanker
---- ContextualNeuralRanker (TBA)
---- MetricRanker (TBA)

'''

class AbstractNeuralRanker():

    def __init__(self, id='AbsRanker'):
        self.id = id

    def config_neural_scoring_function(self):
        pass





class NeuralRanker(AbstractNeuralRanker):
    '''
    A common one-size-fits-all neural ranker
    '''

    def __init__(self, id='NeuralRanker', sf_para_dict=None, opt='Adam', lr = 1e-3, weight_decay=1e-3):
        super(NeuralRanker, self).__init__(id=id)

        self.sf_para_dict = sf_para_dict
        self.sf = self.config_neural_scoring_function()

        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.config_optimizer()

        self.stop_check_freq = 10


    def config_neural_scoring_function(self):
        ffnns = self.ini_ffnns(**self.sf_para_dict['ffnns'])
        if gpu: ffnns = ffnns.to(device)

        return ffnns

    def ini_ffnns(self, num_features=None, h_dim=100, out_dim=1, num_layers=3, HD_AF='R', HN_AF='R', TL_AF='S',
                  apply_tl_af=True, BN=True, RD=False, FBN=False):
        '''
        Initialization of a feed-forward neural network

        :param num_features: the number of dimensions of the input layer todo in_dims
        :param h_dim:        the number of dimensions of a hidden layer
        :param out_dim:      the number of dimensions of the output layer todo out_dims
        :param num_layers:   the number of layers
        :param HD_AF:        the activation function for the first layer
        :param HN_AF:        the activation function for the hidden layer(s)
        :param TL_AF:
        :param apply_tl_af:  the activation function for the output layer
        :param BN:           batch normalization
        :param RD:           each hidden layer is implemented as a residual block
        :param FBN:          perform batch normalization over raw input features enabling learnable normalization
        :return:
        '''
        head_AF, hidden_AF = get_AF(HD_AF), get_AF(HN_AF)  # configurable activation functions
        tail_AF = get_AF(TL_AF) if apply_tl_af else None

        ffnns = nn.Sequential()
        if 1 == num_layers:
            # using in-build batch-normalization layer to normalize raw features, enabling learnable normalization
            if FBN: ffnns.add_module('FeatureBN',
                                     nn.BatchNorm1d(num_features=num_features, momentum=1.0, affine=True,
                                                    track_running_stats=False))

            nr_h1 = nn.Linear(num_features, out_dim)  # 1st layer
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)

            if BN:  # before applying activation
                bn_1 = nn.BatchNorm1d(out_dim, momentum=1.0, affine=True,
                                      track_running_stats=False)  # normalize the result w.r.t. each neural unit before activation
                ffnns.add_module('BN_1', bn_1)

            if apply_tl_af:
                ffnns.add_module('ACT_1', tail_AF)

        else:
            # using in-build batch-normalization layer to normalize raw features, enabling learnable normalization
            if FBN: ffnns.add_module('FeatureBN',
                                     nn.BatchNorm1d(num_features=num_features, momentum=1.0, affine=True,
                                                    track_running_stats=False))

            nr_h1 = nn.Linear(num_features, h_dim)  # 1st layer
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)

            if BN:  # before applying activation
                bn1 = nn.BatchNorm1d(h_dim, momentum=1.0, affine=True, track_running_stats=False)
                ffnns.add_module('BN_1', bn1)

            ffnns.add_module('ACT_1', head_AF)

            if num_layers > 2:  # middle layers if needed
                if RD:
                    for i in range(2, num_layers):
                        ffnns.add_module('_'.join(['RD', str(i)]),
                                         ResidualBlock_FFNNs(dim=h_dim, AF=hidden_AF, BN=BN))

                else:
                    for i in range(2, num_layers):
                        ffnns.add_module('_'.join(['DR', str(i)]), nn.Dropout(0.01))
                        nr_hi = nn.Linear(h_dim, h_dim)
                        nr_init(nr_hi.weight)
                        ffnns.add_module('_'.join(['L', str(i)]), nr_hi)

                        if BN:  # before applying activation
                            bn_i = nn.BatchNorm1d(h_dim, momentum=1.0, affine=True, track_running_stats=False)
                            ffnns.add_module('_'.join(['BN', str(i)]), bn_i)

                        ffnns.add_module('_'.join(['ACT', str(i)]), hidden_AF)

            nr_hn = nn.Linear(h_dim, out_dim)  # relevance prediction layer
            nr_init(nr_hn.weight)
            ffnns.add_module('_'.join(['L', str(num_layers)]), nr_hn)

            if BN:  # before applying activation
                ffnns.add_module('_'.join(['BN', str(num_layers)]),
                                 nn.BatchNorm1d(out_dim, momentum=1.0, affine=True, track_running_stats=False))

            if apply_tl_af:
                ffnns.add_module('_'.join(['ACT', str(num_layers)]), tail_AF)

        # print(ffnns)
        # for w in ffnns.parameters():
        #    print(w.data)

        return ffnns


    def config_optimizer(self):
        if 'Adam'  == self.opt:
            self.optimizer = optim.Adam(self.sf.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.sf.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        else:
            raise NotImplementedError

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)


    def train(self, batch_ranking, batch_label, **kwargs):
        stop_training = False

        batch_pred = self.predict(batch_ranking, train=True)

        if 'epoch_k' in kwargs and kwargs['epoch_k'] % self.stop_check_freq == 0:
            stop_training = self.stop_training(batch_pred)

        return self.inner_train(batch_pred, batch_label, **kwargs), stop_training


    def inner_train(self, batch_pred, batch_label, **kwargs):
        '''
        The ltr loss function to be customized
        :param batch_preds:
        :param batch_label:
        :param kwargs:
        :return:
        '''
        pass


    def stop_training(self, preds):
        ''' stop training if the predictions are all zeros or include nan value(s)'''

        #if torch.nonzero(preds).size(0) <= 0: # todo-as-note: 'preds.byte().any()' seems wrong operation w.r.t. gpu
        if torch.nonzero(preds, as_tuple=False).size(0) <= 0: # due to the UserWarning: This overload of nonzero is deprecated:
            print('All zero error.\n')
            #print('All zero error.\n', preds)
            return True

        if torch.isnan(preds).any():
            print('Including NaN error.')
            #print('Including NaN error.\n', preds)
            return True

        return False

    def forward(self, batch_ranking):
        if 1 == batch_ranking.size(0): # in order to support batch normalization
            batch_output = self.sf(torch.squeeze(batch_ranking, dim=0))
            batch_output = torch.unsqueeze(batch_output, dim=0)
        else:
            batch_output = self.sf(batch_ranking)

        #print('batch_outputs', batch_outputs.size())
        batch_pred = torch.squeeze(batch_output, dim=2)  # -> [batch, ranking_size]

        return batch_pred

    def predict(self, batch_ranking, train=False):
        '''
        In the context of adhoc ranking, the shape is interpreted as:
        :param x: [batch, ranking_size, num_features], which is a common input shape
        :return: [batch, ranking_size, 1]
        '''
        if train:
            self.train_mode()  # training mode for training
        else:
            self.eval_mode()   # evaluation mode for testing

        batch_pred = self.forward(batch_ranking=batch_ranking)
        return batch_pred


    def eval_mode(self):
        self.sf.eval()

    def train_mode(self):
        self.sf.train(mode=True)

    def reset_parameters(self):
        ''' reset parameters, e.g., training from zero with the same setting '''
        self.sf = self.config_neural_scoring_function()
        self.config_optimizer()

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.sf.state_dict(), dir + name)

    def load(self, file_model):
        self.sf.load_state_dict(torch.load(file_model))

    def get_tl_af(self):
        return self.sf_para_dict['ffnns']['TL_AF']
