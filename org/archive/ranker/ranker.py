#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 26/09/2018 | https://y-research.github.io

"""Description

"""
import os
from itertools import combinations

import torch
import torch.optim as optim

from org.archive.ranker.nn_blocks import FFNNs

from org.archive.l2r_global import L2R_GLOBAL
gpu, device = L2R_GLOBAL.global_gpu, L2R_GLOBAL.global_device


class AbstractNeuralRankingFunction():

    def __init__(self, id='name of the neural ranking function'):
        self.id = id
        self.rf = None # the object of ranking function

    def ini(self):
        ''' define the architecture of the ranking function '''
        pass

    def get_parameters(self):
        ''' get the parameters w.r.t. the ranking function '''
        pass

    def eval_mode(self):
        pass

    def train_mode(self):
        pass

    def run(self, batch_rankings, **kwargs):
        '''
        :param batch_rankings: [batch, ranking_size, num_features]
        :return:
        '''
        pass

    def save(self, dir, name):
        pass

    def load(self, file_model):
        pass

    def get_tl_af(self):
        pass


class VanillaFFNNs(AbstractNeuralRankingFunction):
    '''
    A vanilla ranking function based on (deep) feed-forward neural networks.
    '''
    def __init__(self, id='VanillaFFNNs', para_dict=None):
        super(VanillaFFNNs, self).__init__(id=id)
        self.para_dict = para_dict
        self.ini()

    def ini(self):
        torch.manual_seed(seed=L2R_GLOBAL.l2r_seed)
        self.rf = FFNNs(self.para_dict)
        if gpu: self.rf = self.rf.to(device)

    def get_parameters(self):
        return self.rf.parameters()

    def eval_mode(self):
        self.rf.eval()

    def train_mode(self):
        self.rf.train(mode=True)

    def run(self, batch_rankings, **kwargs):
        batch_outputs = self.rf(batch_rankings)
        #print('batch_rankings', batch_rankings.size())
        #print('batch_outputs', batch_outputs.size())
        batch_outputs = torch.squeeze(batch_outputs, dim=2)  # -> [batch, ranking_size]
        return batch_outputs

    def save(self, dir, name):
        torch.save(self.rf.state_dict(), dir + name)

    def load(self, file_model):
        self.rf.load_state_dict(torch.load(file_model))

    def get_tl_af(self):
        return self.para_dict['TL_AF']


def context_(x):
    '''
    get the context representation
    :param x: [batch, ranking_size, num_features]
    :return: [batch, size_context_vector]
    '''
    batch_max, _ = torch.max(x, dim=1)
    #print(batch_max.size())
    batch_mean = torch.mean(x, dim=1)
    #print(batch_mean.size())
    batch_std_var = torch.std(x, dim=1)
    #print(batch_std_var.size())
    #batch_var = torch.var(x, dim=1)
    batch_cnt = torch.cat((batch_max, batch_mean, batch_std_var), dim=1)

    return batch_cnt

def get_context(x, key):
    '''
    get the specified context representation
    :param x: [batch, ranking_size, num_features]
    :return: [batch, size_context_vector]
    '''
    if 'max' == key:
        batch_cnt, _ = torch.max(x, dim=1)
        #print(batch_max.size())
    elif 'mean' == key:
        batch_cnt = torch.mean(x, dim=1)
        #print(batch_mean.size())
    elif 'svar' == key:
        batch_cnt = torch.std(x, dim=1)
        #print(batch_std_var.size())
    elif 'var' == key:
        batch_cnt = torch.var(x, dim=1)

    return batch_cnt

def distill_context(x, cnt_str='max_mean_var'):
    '''
    get the context representation
    :param x: [batch, ranking_size, num_features]
    :return: [batch, size_context_vector]
    '''

    cnt_keys = cnt_str.split('_')
    batch_cnts = get_context(x, cnt_keys[0])
    for k in range(1, len(cnt_keys)):
        batch_cnt = get_context(x, cnt_keys[k])
        batch_cnts = torch.cat((batch_cnts, batch_cnt), dim=1)

    return batch_cnts


def get_all_cnt_strs():
    set_keys = ['mean', 'max', 'svar', 'var']

    all_cnt_strs = []
    for num in range(1, len(set_keys) + 1):
        for comb_keys in combinations(set_keys, num):
            # print(subset)
            if len(comb_keys) > 1:
                all_cnt_strs.append('_'.join(comb_keys))
            else:
                all_cnt_strs.extend(comb_keys)

    #print(all_cnt_strs)
    return all_cnt_strs


class ContextAwareFFNNs(AbstractNeuralRankingFunction):
    '''
    A context-aware ranking function based on (deep) feed-forward neural networks.
    '''
    def __init__(self, id='ContextAwareFFNNs', in_para_dict=None, cnt_para_dict=None, com_para_dict=None, cnt_str=None):
        '''
        :param in_para_dict: project the input vector to an inner representation, say In_rep
        :param cnt_para_dict: project the context vector to an inner representation, say Cnt_rep
        :param com_para_dict: project the combination of In_rep & Cnt_rep to the prediction
        :param id:
        '''
        super(ContextAwareFFNNs, self).__init__(id=id)
        assert in_para_dict['out_dim'] == cnt_para_dict['out_dim'] \
               and com_para_dict['num_features'] == cnt_para_dict['out_dim']

        self.in_para_dict = in_para_dict
        self.cnt_para_dict = cnt_para_dict
        self.com_para_dict = com_para_dict
        self.cnt_str = cnt_str
        self.ini()

    def ini(self):
        torch.manual_seed(seed=L2R_GLOBAL.l2r_seed)
        self.in_proj_f = FFNNs(self.in_para_dict)
        self.cnt_proj_f = FFNNs(self.cnt_para_dict)
        self.com_proj_f = FFNNs(self.com_para_dict)

        if gpu:
            self.in_proj_f = self.in_proj_f.to(device)
            self.cnt_proj_f = self.cnt_proj_f.to(device)
            self.com_proj_f = self.com_proj_f.to(device)

    def get_parameters(self):
        all_params = list(self.in_proj_f.parameters()) + list(self.cnt_proj_f.parameters()) + list(self.com_proj_f.parameters())
        return all_params

    def eval_mode(self):
        self.in_proj_f.eval()  # evaluation mode
        self.cnt_proj_f.eval()  # evaluation mode
        self.com_proj_f.eval()  # evaluation mode

    def train_mode(self):
        self.in_proj_f.train(mode=True)  # training mode
        self.cnt_proj_f.train(mode=True)  # training mode
        self.com_proj_f.train(mode=True)  # training mode

    def run(self, batch_rankings, **kwargs):
        batch_doc_reprs = self.in_proj_f(batch_rankings)  # get projected vectors given raw input document vectors

        if 'query_context' in kwargs and kwargs['query_context'] is not None:
            batch_cnts = kwargs['query_context']
        else:
            batch_cnts = distill_context(batch_rankings, cnt_str=self.cnt_str)

        batch_cnt_reprs = self.cnt_proj_f(batch_cnts)  # get projected vectors given context vectors

        batch_outputs = self.com_proj_f(batch_doc_reprs + batch_cnt_reprs)  # get projected vectors given combined vectors

        batch_outputs = torch.squeeze(batch_outputs, dim=2)  # -> [batch, ranking_size]
        return batch_outputs

    def save(self, dir, name):
        model_dicts = (self.in_proj_f.state_dict(), self.cnt_proj_f.state_dict(), self.com_proj_f.state_dict())
        torch.save(model_dicts, dir + name)

    def load(self, file_model):
        in_dict, cnt_dict, com_dict = torch.load(file_model)
        self.in_proj_f.load_state_dict(in_dict)
        self.cnt_proj_f.load_state_dict(cnt_dict)
        self.com_proj_f.load_state_dict(com_dict)

    def get_tl_af(self):
        return self.com_para_dict['TL_AF']



class AbstractNeuralRanker():

    def __init__(self, id=None, ranking_function=None, opt='Adam', lr = 1e-3, weight_decay=1e-3):
        self.id = id
        self.ranking_function = ranking_function # approximated function for predicting the relevance of documents
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        #self.optimizer = optim.Adam(self.ranking_function.get_parameters(), lr=1e-3, weight_decay=1e-3)  # use regularization
        self.ini_optimizer()

    def ini_optimizer(self):
        if 'Adam' == self.opt:
            self.optimizer = optim.Adam(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        else:
            raise NotImplementedError

    def reset_parameters(self):
        ''' reset parameters, e.g., training from zero with the same setting '''
        self.ranking_function.ini()
        #self.optimizer = optim.Adam(self.ranking_function.get_parameters(), lr=1e-3, weight_decay=1e-3)  # use regularization
        self.ini_optimizer()

    def train(self, batch_rankings, batch_stds, **kwargs):
        #print('batch_rankings', batch_rankings.size())
        batch_preds = self.predict(batch_rankings, train=True, **kwargs)
        #print('batch_preds', batch_preds.size())
        return self.inner_train(batch_preds, batch_stds, **kwargs)

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        pass

    def predict(self, batch_rankings, train=False, **kwargs):
        if train:
            self.ranking_function.train_mode()  # training mode for training
        else:
            self.ranking_function.eval_mode()  # evaluation mode for testing

        batch_preds = self.ranking_function.run(batch_rankings, **kwargs)
        return batch_preds

    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.ranking_function.save(dir=dir, name=name)

    def load_model(self, file_model):
        self.ranking_function.load(file_model)



class MDNsRanker(AbstractNeuralRanker):
    def __init__(self, id=None, ranking_function=None):
        super(MDNsRanker, self).__init__(id=id, ranking_function=ranking_function)

    def train(self, batch_rankings, batch_stds, **kwargs):
        #print('batch_rankings', batch_rankings.size())
        batch_mus, batch_sigmas = self.predict(batch_rankings, train=True, **kwargs)
        #print('batch_preds', batch_preds.size())
        return self.inner_train(batch_mus, batch_stds, batch_sigmas, **kwargs)

    def inner_train(self, batch_mus, batch_stds, batch_sigmas, **kwargs):
        pass

    def predict(self, batch_rankings, train=False, **kwargs):
        if train:
            self.ranking_function.train_mode()  # training mode for training
        else:
            self.ranking_function.eval_mode()  # evaluation mode for testing

        batch_mus, batch_sigmas = self.ranking_function.run(batch_rankings, **kwargs)
        if train:
            return batch_mus, batch_sigmas
        else:
            return batch_mus # regarding mu as the relevance degree

    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.ranking_function.save(dir=dir, name=name)

    def load_model(self, file_model):
        self.ranking_function.load(file_model)