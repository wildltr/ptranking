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

########
# Activation function: sigmoid with an explicit sigma.
########
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

########
# Activation function: sigmoid with an explicit sigma, where the overflow is taken into account.
########
class Robust_Sigmoid(torch.autograd.Function):
    ''' Aiming for a stable sigmoid operator with specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0, device='cpu'):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0==sigma else sigma * input

        torch_half = torch.tensor([0.5], dtype=torch.float, device=device)
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

########
# Instantiate activation function with a short string identifier.
########

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

########
#
########
class ResidualBlock_FFNNs(nn.Module):
    def __init__(self, dim=100, AF=None):
        super(ResidualBlock_FFNNs, self).__init__()

        self.ffnns = nn.Sequential()

        layer_1 = nn.Linear(dim, dim)
        nr_init(layer_1.weight)
        self.ffnns.add_module('L_1', layer_1)
        self.ffnns.add_module('ACT_1', get_AF(af_str=AF))

        layer_2 = nn.Linear(dim, dim)
        nr_init(layer_2.weight)
        self.ffnns.add_module('L_2', layer_2)

        self.AF = get_AF(af_str=AF)

    def forward(self, x):
        residual = x
        res = self.ffnns(x)
        res += residual
        res = self.AF(res)

        return res

########
# Customized batch normalization module, where the 3rd dimension is features.
########

##- Variant-1 -##
class LTRBatchNorm(nn.Module):
    """
    Note: given multiple query instances, the mean and variance are computed across queries.
    """
    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=False):
        '''
        @param num_features: C from an expected input of size (N, C, L) or from input of size (N, L)
        @param momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        @param affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
        @param track_running_stats:
        '''
        super(LTRBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, X):
        '''
        @param X: Expected 2D or 3D input
        @return:
        '''
        if 3 == X.dim():
            return self.bn(X.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return self.bn(X)

##- Variant-2 -##

def ltr_batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # calculate the mean and variance on the feature dimension. Here we need to maintain the shape of `X`,
        # so that the broadcasting operation can be carried out later
        mean = X.mean(dim=1, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=1, keepdim=True)

        # In training mode, the current mean and variance are used for the standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_mean = moving_mean.mean(dim=0, keepdim=True)
        moving_var = (1.0 - momentum) * moving_var + momentum * var
        moving_var = moving_var.mean(dim=0, keepdim=True)

    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

class LTRBatchNorm2(nn.Module):
    '''
    By referring to http://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=batchnorm2d
    Given multiple query instances, the mean and variance are computed at a query level.
    '''
    def __init__(self, num_features, momentum=0.1, affine=True, device=None):
        '''
        In the context of learning-to-rank, batch normalization is conducted at a per-query level, namely across documents associated with the same query.
        @param num_features: the number of features
        @param num_dims: is assumed to be [num_queries, num_docs, num_features]
        '''
        super().__init__()
        shape = (1, 1, num_features)
        # The scale parameter and the shift parameter (model parameters) are initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape, device=device))
        self.beta = nn.Parameter(torch.zeros(shape, device=device))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.ones(shape, device=device)
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(shape, device=device))
            self.bias = nn.Parameter(torch.zeros(shape, device=device))

    def forward(self, X):
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = ltr_batch_norm(X, self.gamma, self.beta, self.moving_mean,
                                                              self.moving_var, eps=1e-5, momentum=self.momentum)

        if self.affine:
            Y = Y * self.weight + self.bias

        return Y

########
# Stacked Feed-forward Network
########

def get_stacked_FFNet(ff_dims=None, AF=None, TL_AF=None, apply_tl_af=False, dropout=0.1,
                      BN=True, bn_type=None, bn_affine=False, device='cpu', split_penultimate_layer=False):
    '''
    Generate one stacked feed-forward network.
    '''
    # '2' refers to the simplest case: num_features, out_dim
    assert ff_dims is not None and len(ff_dims) >= 2

    ff_net = nn.Sequential()
    num_layers = len(ff_dims)
    if num_layers > 2:
        for i in range(1, num_layers-1):
            prior_dim, ff_i_dim = ff_dims[i - 1], ff_dims[i]
            ff_net.add_module('_'.join(['dr', str(i)]), nn.Dropout(dropout))
            nr_hi = nn.Linear(prior_dim, ff_i_dim)
            nr_init(nr_hi.weight)
            ff_net.add_module('_'.join(['ff', str(i + 1)]), nr_hi)

            if BN:  # before applying activation
                if 'BN' == bn_type:
                    bn_i = LTRBatchNorm(ff_i_dim, momentum=0.1, affine=bn_affine, track_running_stats=False)
                elif 'BN2' == bn_type:
                    bn_i = LTRBatchNorm2(ff_i_dim, momentum=0.1, affine=bn_affine, device=device)
                else:
                    raise NotImplementedError

                ff_net.add_module('_'.join(['bn', str(i + 1)]), bn_i)

            ff_net.add_module('_'.join(['act', str(i + 1)]), get_AF(AF))

    # last layer
    penultimate_dim, out_dim = ff_dims[-2], ff_dims[-1]
    nr_hn = nn.Linear(penultimate_dim, out_dim)
    nr_init(nr_hn.weight)

    if split_penultimate_layer:
        tail_net = nn.Sequential()
        tail_net.add_module('_'.join(['ff', str(num_layers)]), nr_hn)
        if apply_tl_af:
            if BN:  # before applying activation
                if 'BN' == bn_type:
                    tail_net.add_module('_'.join(['bn', str(num_layers)]),
                                        LTRBatchNorm(out_dim, momentum=0.1, affine=bn_affine, track_running_stats=False))
                elif 'BN2' == bn_type:
                    tail_net.add_module('_'.join(['bn', str(num_layers)]),
                                        LTRBatchNorm2(out_dim, momentum=0.1, affine=bn_affine, device=device))
                else:
                    raise NotImplementedError

            tail_net.add_module('_'.join(['act', str(num_layers)]), get_AF(TL_AF))

        return ff_net, tail_net
    else:
        ff_net.add_module('_'.join(['ff', str(num_layers)]), nr_hn)

        if apply_tl_af:
            if BN:  # before applying activation
                if 'BN' == bn_type:
                    ff_net.add_module('_'.join(['bn', str(num_layers)]),
                                      LTRBatchNorm(out_dim, momentum=0.1, affine=bn_affine, track_running_stats=False))
                elif 'BN2' == bn_type:
                    ff_net.add_module('_'.join(['bn', str(num_layers)]),
                                      LTRBatchNorm2(out_dim, momentum=0.1, affine=bn_affine, device=device))
                else:
                    raise NotImplementedError

            ff_net.add_module('_'.join(['act', str(num_layers)]), get_AF(TL_AF))

        return ff_net