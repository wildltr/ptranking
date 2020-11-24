#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Some utility components for building a neural ranker
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.init import kaiming_normal_ as nr_init
from torch.nn.init import xavier_normal_ as nr_init


class Vanilla_Sigmoid(torch.autograd.Function):
    ''' The vanilla sigmoid operator with a specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0==sigma else sigma * input

        sigmoid_x = 1. / (1. + torch.exp(-x))

        grad = sigmoid_x * (1. - sigmoid_x) if 1.0==sigma else sigma * sigmoid_x * (1. - sigmoid_x)
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        '''
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad  # chain rule

        return bg, None

#- function: vanilla_sigmoid-#
vanilla_sigmoid = Vanilla_Sigmoid.apply


class Robust_Sigmoid(torch.autograd.Function):
    ''' Aiming for a stable sigmoid operator with specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0, gpu=False):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0==sigma else sigma * input

        torch_half = torch.cuda.FloatTensor([0.5]) if gpu else torch.FloatTensor([0.5])
        sigmoid_x_pos = torch.where(input>0, 1./(1. + torch.exp(-x)), torch_half)

        exp_x = torch.exp(x)
        sigmoid_x = torch.where(input<0, exp_x/(1.+exp_x), sigmoid_x_pos)

        grad = sigmoid_x * (1. - sigmoid_x) if 1.0==sigma else sigma * sigmoid_x * (1. - sigmoid_x)
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        '''
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad # chain rule

        return bg, None, None

#- function: robust_sigmoid-#
robust_sigmoid = Robust_Sigmoid.apply

def get_AF(af_str):
    """
    Given the string identifier, get PyTorch-supported activation function.

    """
    if af_str == 'R':
        return nn.ReLU()         # ReLU(x)=max(0,x)

    elif af_str == 'LR':
        return nn.LeakyReLU()    # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)

    elif af_str == 'RR':
        return nn.RReLU()        # the randomized leaky rectified liner unit function

    elif af_str == 'PR':
        return nn.PRelU()

    elif af_str == 'E':          # ELU(x)=max(0,x)+min(0,α∗(exp(x)−1))
        return nn.ELU()

    elif af_str == 'SE':         # SELU(x)=scale∗(max(0,x)+min(0,α∗(exp(x)−1)))
        return nn.SELU()

    elif af_str == 'CE':         # CELU(x)=max(0,x)+min(0,α∗(exp(x/α)−1))
        return nn.CELU()

    elif af_str == 'GE':
        return nn.GELU()

    elif af_str == 'S':
        return nn.Sigmoid()

    elif af_str == 'SW':
        #return SWISH()
        raise NotImplementedError

    elif af_str == 'T':
        return nn.Tanh()

    elif af_str == 'ST':         # a kind of normalization
        return F.softmax()      # Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1

    else:
        raise NotImplementedError


def get_sf_str(num_layers=None, HD_AF=None, HN_AF=None, TL_AF=None, join_str='.'):
    """
    Get the string identifier of a scoring function by concatenating the string-identifiers of deployed activation functions.
    """
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


def decode_sf_str(file_name):
    b_ind = file_name.index('SF_') + 3
    s_ind = file_name[b_ind:len(file_name)].index('_')
    #print(b_ind, s_ind)
    sf_str = file_name[b_ind:b_ind+s_ind]
    #print(sf_str)
    sf_arr = sf_str.split('.')
    #print(sf_arr)
    le = len(sf_arr)
    assert le > 0

    num_layers, HD_AF, HN_AF, TL_AF = None, None, None, None
    if 1 == le:
        num_layers, HD_AF = 1, sf_arr[0]
    elif 2 == le:
        num_layers, HD_AF, TL_AF = 2, sf_arr[0], sf_arr[1]
    else:
        num_layers, HD_AF, HN_AF, TL_AF = int(''.join(filter(str.isdigit, sf_arr[1])))+2, sf_arr[0], sf_arr[1][0:-1], sf_arr[2]

    #print('--', num_layers, HD_AF, HN_AF, TL_AF)
    return num_layers, HD_AF, HN_AF, TL_AF


class ResidualBlock_FFNNs(nn.Module):
    def __init__(self, dim=100, AF=None, BN=True):
        super(ResidualBlock_FFNNs, self).__init__()

        self.ffnns = nn.Sequential()

        layer_1 = nn.Linear(dim, dim)
        nr_init(layer_1.weight)
        self.ffnns.add_module('L_1', layer_1)
        if BN: self.ffnns.add_module('BN_1', nn.BatchNorm1d(dim, momentum=1.0, affine=True, track_running_stats=False))
        self.ffnns.add_module('ACT_1', AF)

        layer_2 = nn.Linear(dim, dim)
        nr_init(layer_2.weight)
        self.ffnns.add_module('L_2', layer_2)
        if BN: self.ffnns.add_module('BN_2', nn.BatchNorm1d(dim, momentum=1.0, affine=True, track_running_stats=False))

        self.AF = AF

    def forward(self, x):
        residual = x
        res = self.ffnns(x)
        res += residual
        res = self.AF(res)

        return res
