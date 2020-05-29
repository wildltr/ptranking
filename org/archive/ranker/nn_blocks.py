#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.init import kaiming_normal_ as nr_init
from torch.nn.init import xavier_normal_ as nr_init

from org.archive.utils.pytorch.pt_extensions import is_RK, ini_ReLU_K, Exp


def get_AF(af_str):
    if af_str == 'R':
        return nn.ReLU() #ReLU(x)=max(0,x)
    elif af_str == 'S':
        return nn.Sigmoid()
    elif af_str == 'T':
        return nn.Tanh()
    elif af_str == 'LR':
        return nn.LeakyReLU()   # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
    elif af_str == 'ST':
        return nn.Softmax() # Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1
    elif af_str == 'R6':
        return nn.ReLU6()   #ReLU6(x)=min(max(0,x),6)
    elif is_RK(af_str):
        return ini_ReLU_K(af_str) #ReLU_K(x)=min(max(0,x),k)
    elif af_str == 'E':
        return Exp()
    else:
        raise NotImplementedError

def get_rf_str(num_layers=None, HD_AF=None, HN_AF=None, TL_AF=None, join_str='.'):
    assert num_layers >= 1
    if 1 == num_layers:
        return HD_AF
    elif 2 == num_layers:
        return join_str.join([HD_AF, TL_AF])
    elif 3 == num_layers:
        return join_str.join([HD_AF, HN_AF, TL_AF])
    else:
        h_str = HN_AF + str(num_layers-2)
        return join_str.join([HD_AF, h_str, TL_AF])


class FFNNs(nn.Module):
    '''
    The building block of (deep) feed-forward neural networks.
    '''
    def __init__(self, f_para_dict=None):
        super(FFNNs, self).__init__()
        self.ffnns = self.ini_ffnns(**f_para_dict)

    def ini_ffnns(self, num_features=None, h_dim=100, out_dim=1, num_layers=3, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True):
        head_AF, hidden_AF, tail_AF = get_AF(HD_AF), get_AF(HN_AF), get_AF(TL_AF)   #configurable activation functions

        ffnns = nn.Sequential()
        if 1 == num_layers:
            nr_h1 = nn.Linear(num_features, out_dim)  # 1st layer
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)

            if apply_tl_af:
                ffnns.add_module('ACT_1', tail_AF)
        else:
            nr_h1 = nn.Linear(num_features, h_dim)    # 1st layer
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)
            ffnns.add_module('ACT_1', head_AF)

            if num_layers > 2:           # middle layers if needed
                for i in range(2, num_layers):
                    ffnns.add_module('_'.join(['DR', str(i)]), nn.Dropout(0.01))
                    nr_hi = nn.Linear(h_dim, h_dim)
                    nr_init(nr_hi.weight)
                    ffnns.add_module('_'.join(['L', str(i)]), nr_hi)
                    ffnns.add_module('_'.join(['ACT', str(i)]), hidden_AF)

            nr_hn = nn.Linear(h_dim, out_dim)  #relevance prediction layer
            nr_init(nr_hn.weight)
            ffnns.add_module('_'.join(['L', str(num_layers)]), nr_hn)
            if apply_tl_af:
                ffnns.add_module('_'.join(['ACT', str(num_layers)]), tail_AF)

        return ffnns


    def forward(self, x):
        '''
        :param x: [batch, ranking_size, num_features], the common shape of the input
        :return: [batch, ranking_size, 1]
        '''
        res = self.ffnns(x)
        return res

def tor_transform_to_rank_features(x):
    # batch, ranking_size, num_features = x.size(0), x.size(1), x.size(2)
    ranking_size = x.size(1)
    x.transpose_(1, 2)  # [batch, ranking_size, num_features]  ->  [batch, num_features, ranking_size]

    a = torch.unsqueeze(x, 3)  # [batch, num_features, ranking_size, 1]
    b = torch.unsqueeze(x, 2)  # [batch, num_features, 1, ranking_size]
    c = a - b  # [batch, num_features, ranking_size, ranking_size]

    c_zeros = torch.zeros(c.size())
    c_ones = torch.ones(c.size())
    new_c = torch.where(c > 0, c_ones, c)
    new_c = torch.where(new_c < 0, c_zeros, new_c)

    #idx = torch.arange(0, ranking_size, out=torch.LongTensor())
    #new_c[:, :, idx, idx] = 1

    rank_c = torch.sum(new_c, dim=3) + 1.0 # the "+ 1.0" corresponds to the diagonal entries, i.e., self votes
    rank_c.transpose_(1, 2)  # back to [batch, ranking_size, num_features]

    return rank_c



if __name__ == '__main__':
    #1
    '''
    ffnns = FFNNs(f_para_dict={'num_features':6})
    #print(ffnns)
    paras = ffnns.parameters()
    for p in paras:
        print(p)
    '''
