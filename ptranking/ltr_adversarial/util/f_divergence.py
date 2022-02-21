#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch


def get_f_divergence_functions(f_div_str=None):
    '''
    the activation function is chosen as a monotone increasing function
    '''
    if 'TVar' == f_div_str: # Total variation
        def activation_f(v):
            return 0.5 * torch.tanh(v)

        def conjugate_f(t):
            return t

    elif 'KL' == f_div_str: # Kullback-Leibler
        def activation_f(v):
            return v

        def conjugate_f(t):
            return torch.exp(t-1)

    elif 'RKL' == f_div_str: # Reverse KL
        def activation_f(v):
            return -torch.exp(-v)

        def conjugate_f(t):
            return -1.0 - torch.log(-t)

    elif 'PC' == f_div_str: # Pearson chi-square
        def activation_f(v):
            return v

        def conjugate_f(t):
            return 0.25 * torch.pow(t, exponent=2.0) + t

    elif 'NC' == f_div_str: # Neyman chi-square
        def activation_f(v):
            return 1.0 - torch.exp(-v)

        def conjugate_f(t):
            return 2.0 - 2.0 * torch.sqrt(1.0-t)

    elif 'SH' == f_div_str: # Squared Hellinger
        def activation_f(v):
            return 1.0 - torch.exp(-v)

        def conjugate_f(t):
            return t/(1.0-t)

    elif 'JS' == f_div_str: # Jensen-Shannon
        def activation_f(v):
            return torch.log(torch.tensor(2.0)) - torch.log(1.0 + torch.exp(-v))

        def conjugate_f(t):
            return -torch.log(2.0 - torch.exp(t))

    elif 'JSW' == f_div_str: # Jensen-Shannon-weighted
        def activation_f(v):
            return -math.pi*torch.log(math.pi) - torch.log(1.0+torch.exp(-v))

        def conjugate_f(t):
            return (1.0-math.pi)*torch.log((1.0-math.pi)/(1.0-math.pi*torch.exp(t/math.pi)))

    elif 'GAN' == f_div_str: # GAN
        def activation_f(v):
            return -torch.log(1.0 + torch.exp(-v))

        def conjugate_f(t):
            return -torch.log(1.0 - torch.exp(t))

    return activation_f, conjugate_f
