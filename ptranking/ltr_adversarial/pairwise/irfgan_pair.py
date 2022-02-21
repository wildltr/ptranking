#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import copy
import numpy as np
from itertools import product

import torch

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine
from ptranking.ltr_adversarial.pointwise.point_generator import Point_Generator
from ptranking.ltr_adversarial.util.f_divergence import get_f_divergence_functions
from ptranking.ltr_adversarial.pointwise.point_discriminator import Point_Discriminator
from ptranking.data.data_utils import MSLETOR_SEMI
from ptranking.ltr_adversarial.util.pair_sampling import generate_true_pairs, sample_points_Bernoulli

class IRFGAN_Pair(AdversarialMachine):
    ''' '''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, ad_para_dict=None, g_key='BT', sigma=1.0, gpu=False, device=None):
        super(IRFGAN_Pair, self).__init__(eval_dict=eval_dict, data_dict=data_dict, gpu=gpu, device=device)

        self.f_div_id = ad_para_dict['f_div_id']
        self.samples_per_query = ad_para_dict['samples_per_query']
        self.activation_f, self.conjugate_f = get_f_divergence_functions(self.f_div_id)
        self.dict_diff = dict()
        self.tensor = torch.cuda.FloatTensor if self.gpu else torch.FloatTensor

        assert g_key=='BT'
        '''
        Underlying formulation of generation
        (1) BT: the probability of observing a pair of ordered documents is formulated via Bradley-Terry model,
         i.e., p(d_i > d_j)=1/(1+exp(-sigma(s_i - s_j))), the default value of sigma is given as 1.0            
        '''
        self.g_key = g_key
        self.sigma = sigma # only used w.r.t. SR

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        d_sf_para_dict[sf_para_dict['sf_id']]['apply_tl_af'] = False

        self.generator = Point_Generator(sf_para_dict=g_sf_para_dict, gpu=gpu, device=device)
        self.discriminator = Point_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=device)

    def fill_global_buffer(self, train_data, dict_buffer=None):
        ''' Buffer the number of positive documents, and the number of non-positive documents per query '''
        assert self.data_dict['train_presort'] is True  # this is required for efficient truth exampling

        if self.data_dict['data_id'] in MSLETOR_SEMI:
            for entry in train_data:
                qid, _, batch_label = entry[0][0], entry[1], entry[2]
                if not qid in dict_buffer:
                    pos_boolean_mat = torch.gt(batch_label, 0)
                    num_pos = torch.sum(pos_boolean_mat)

                    explicit_boolean_mat = torch.ge(batch_label, 0)
                    num_explicit = torch.sum(explicit_boolean_mat)

                    ranking_size = batch_label.size(1)
                    num_neg_unk = ranking_size - num_pos
                    num_unk = ranking_size - num_explicit

                    num_unique_labels = torch.unique(batch_label).size(0)

                    dict_buffer[qid] = (num_pos, num_explicit, num_neg_unk, num_unk, num_unique_labels)
        else:
            for entry in train_data:
                qid, _, batch_label = entry[0][0], entry[1], entry[2]
                if not qid in dict_buffer:
                    pos_boolean_mat = torch.gt(batch_label, 0)
                    num_pos = torch.sum(pos_boolean_mat)

                    ranking_size = batch_label.size(1)

                    num_explicit = ranking_size
                    num_neg_unk = ranking_size - num_pos
                    num_unk = 0

                    num_unique_labels = torch.unique(batch_label).size(0)

                    dict_buffer[qid] = (num_pos, num_explicit, num_neg_unk, num_unk, num_unique_labels)


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, global_buffer=None):
        '''
        Here it can not use the way of training like irgan-pair (still relying on single documents rather thank pairs),
        since ir-fgan requires to sample with two distributions.
        '''
        stop_training = self.train_discriminator_generator_single_step(train_data=train_data, generator=generator,
                                                               discriminator=discriminator, global_buffer=global_buffer)
        return stop_training


    def train_discriminator_generator_single_step(self, train_data=None, generator=None, discriminator=None,
                                                  global_buffer=None):
        ''' Train both discriminator and generator with a single step per query '''
        stop_training = False
        generator.train_mode()
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0][0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.type(self.tensor)

            sorted_std_labels = torch.squeeze(batch_label, dim=0)

            num_pos, num_explicit, num_neg_unk, num_unk, num_unique_labels = global_buffer[qid]

            if num_unique_labels <2: # check unique values, say all [1, 1, 1] generates no pairs
                continue

            true_head_inds, true_tail_inds = generate_true_pairs(qid=qid, sorted_std_labels=sorted_std_labels,
                             num_pairs=self.samples_per_query, dict_diff=self.dict_diff, global_buffer=global_buffer)

            batch_preds = generator.predict(batch_ranking)  # [batch, size_ranking]

            # todo determine how to activation
            point_preds = torch.squeeze(batch_preds)

            if torch.isnan(point_preds).any():
                print('Including NaN error.')
                stop_training = True
                return stop_training

            #--generate samples
            if 'BT' == self.g_key:
                mat_diffs = torch.unsqueeze(point_preds, dim=1) - torch.unsqueeze(point_preds, dim=0)
                mat_bt_probs = torch.sigmoid(mat_diffs)  # default delta=1.0

                fake_head_inds, fake_tail_inds = sample_points_Bernoulli(mat_bt_probs, num_pairs=self.samples_per_query)
            else:
                raise NotImplementedError
            #--

            # real data and generated data
            true_head_docs = batch_ranking[:, true_head_inds, :]
            true_tail_docs = batch_ranking[:, true_tail_inds, :]
            fake_head_docs = batch_ranking[:, fake_head_inds, :]
            fake_tail_docs = batch_ranking[:, fake_tail_inds, :]

            ''' optimize discriminator '''
            discriminator.train_mode()
            true_head_preds = discriminator.predict(true_head_docs)
            true_tail_preds = discriminator.predict(true_tail_docs)
            true_preds = true_head_preds - true_tail_preds
            fake_head_preds = discriminator.predict(fake_head_docs)
            fake_tail_preds = discriminator.predict(fake_tail_docs)
            fake_preds = fake_head_preds - fake_tail_preds

            dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator
            discriminator.optimizer.zero_grad()
            dis_loss.backward()
            discriminator.optimizer.step()

            ''' optimize generator '''  #
            discriminator.eval_mode()
            d_fake_head_preds = discriminator.predict(fake_head_docs)
            d_fake_tail_preds = discriminator.predict(fake_tail_docs)
            d_fake_preds = self.conjugate_f(self.activation_f(d_fake_head_preds - d_fake_tail_preds))

            if 'BT' == self.g_key:
                log_g_probs = torch.log(mat_bt_probs[fake_head_inds, fake_tail_inds].view(1, -1))
            else:
                raise NotImplementedError

            g_batch_loss = -torch.mean(log_g_probs * d_fake_preds)

            generator.optimizer.zero_grad()
            g_batch_loss.backward()
            generator.optimizer.step()

        # after iteration ove train_data
        return stop_training

    def reset_generator(self):
        self.generator.init()

    def reset_discriminator(self):
        self.discriminator.init()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

###### Parameter of IRFGAN_Pair ######

class IRFGAN_PairParameter(ModelParameter):
    ''' Parameter class for IRFGAN_Pair '''
    def __init__(self, debug=False, para_json=None):
        super(IRFGAN_PairParameter, self).__init__(model_id='IRFGAN_Pair')
        self.debug = debug
        self.para_json = para_json

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_Pair
        :return:
        """
        f_div_id = 'KL'
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG'
        samples_per_query = 5

        self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches, f_div_id=f_div_id,
                                 ad_training_order=ad_training_order, samples_per_query=samples_per_query)
        return self.ad_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        ad_para_dict = given_para_dict if given_para_dict is not None else self.ad_para_dict

        s1 = ':' if log else '_'
        f_div_id = ad_para_dict['f_div_id']

        pair_irfgan_paras_str = f_div_id

        return pair_irfgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Pair
        """
        if self.para_json is not None:
            with open(self.para_json) as json_file:
                json_dict = json.load(json_file)

            choice_samples_per_query = json_dict['samples_per_query']
            #choice_ad_training_order = json_dict['ad_training_order']
            choice_f_div_id = json_dict['f_div']
            '''
            d_g_epoch_strings = json_dict['d_g_epoch']
            choice_d_g_epoch = []
            for d_g_epoch_str in d_g_epoch_strings:
                epoch_arr = d_g_epoch_str.split('-')
                choice_d_g_epoch.append((int(epoch_arr[0]), int(epoch_arr[1])))
            '''
        else:
            choice_samples_per_query = [5]
            #choice_ad_training_order = ['DG']  # GD for irganlist DG for point/pair
            choice_f_div_id = ['KL'] if self.debug else ['KL']  #
            #choice_d_g_epoch = [(1, 1)] if self.debug else [(1, 1)] # discriminator-epoches vs. generator-epoches

        for samples_per_query, f_div_id in product(choice_samples_per_query, choice_f_div_id):
            self.ad_para_dict = dict(model_id=self.model_id, samples_per_query=samples_per_query, f_div_id=f_div_id)

            yield self.ad_para_dict
