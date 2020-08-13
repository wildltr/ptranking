#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import copy
import numpy as np
from itertools import product

import torch

from org.archive.eval.parameter import ModelParameter
from org.archive.ltr_adversarial.base.ad_machine import AdversarialMachine
from org.archive.ltr_adversarial.pointwise.point_generator import Point_Generator
from org.archive.ltr_adversarial.util.f_divergence import get_f_divergence_functions
from org.archive.ltr_adversarial.pointwise.point_discriminator import Point_Discriminator

from org.archive.ltr_adversarial.util.pair_sampling import generate_true_pairs, sample_points_Bernoulli

from org.archive.utils.pytorch.pt_extensions import Gaussian_Integral_0_Inf
apply_Gaussian_Integral_0_Inf = Gaussian_Integral_0_Inf.apply

from org.archive.ltr_global import global_gpu as gpu, tensor

class IRFGAN_Pair(AdversarialMachine):
    ''' '''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, f_div_id='KL', sampling_str='BT', sigma=3.0):
        super(IRFGAN_Pair, self).__init__(eval_dict=eval_dict, data_dict=data_dict)

        self.activation_f, self.conjugate_f = get_f_divergence_functions(f_div_id)
        if sampling_str in ['BT']:
            '''
            (1) BT: the probability of observing a pair of ordered documents is formulated via Bradley-Terry model, i.e., p(d_i > d_j)=1/(1+exp(-sigma(s_i - s_j))), the default value of sigma is given as 1.0            
            '''
            self.sampling_str = sampling_str
            self.sigma = sigma # only used w.r.t. SR
        else:
            '''
            SR is not supported due to time-consuming
            (2) SR: the relevance prediction of a document is regarded as the mean of a Gaussian score distribution with a shared smoothing variance. 
                Thus, the probability of observing a pair of ordered documents is the integral of the difference of two Gaussian random variables, which is itself a Gaussian, cf. Eq-9 within the original paper.
            '''
            raise NotImplementedError

        sf_para_dict['ffnns']['apply_tl_af'] = True

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        d_sf_para_dict['ffnns']['apply_tl_af'] = False

        self.generator = Point_Generator(sf_para_dict=g_sf_para_dict)
        self.discriminator = Point_Discriminator(sf_para_dict=d_sf_para_dict)


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, d_epoches=1, g_epoches=1, dict_buffer=None):

        '''
        Here it can not use the way of training like irgan-pair (still relying on single documents rather thank pairs), since ir-fgan requires to sample with two distributions.
        '''

        self.train_discriminator_generator_single_step(train_data=train_data, generator=generator, discriminator=discriminator, dict_buffer=dict_buffer)

        stop_training = False
        return stop_training


    def train_discriminator_generator_single_step(self, train_data=None, generator=None, discriminator=None, num_samples_per_query=20, dict_buffer=None, **kwargs): # train both discriminator and generator with a single step per query
        for entry in train_data:
            # todo assuming that unknown lable (i.e., -1) is converted to 0

            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if gpu: batch_ranking = batch_ranking.type(tensor)

            tmp_batch_label = torch.squeeze(batch_label, dim=0)
            if torch.unique(tmp_batch_label).size(0) <2: # check unique values, say all [1, 1, 1] generates no pairs
                continue

            true_head_inds, true_tail_inds = generate_true_pairs(sorted_std_labels=torch.sort(tmp_batch_label, descending=True)[0], num_pairs=num_samples_per_query, qid=qid, dict_weighted_clipped_pos_diffs=dict_buffer)

            batch_preds = generator.predict(batch_ranking, train=True)  # [batch, size_ranking]

            # todo determine how to activation
            point_preds = torch.squeeze(batch_preds)

            #--generate samples
            if 'SR' == self.sampling_str:
                mat_means = torch.unsqueeze(point_preds, dim=1) - torch.unsqueeze(point_preds, dim=0)
                mat_probs = apply_Gaussian_Integral_0_Inf(mat_means, np.sqrt(2.0) * self.sigma)

                fake_head_inds, fake_tail_inds = sample_points_Bernoulli(mat_probs, num_pairs=num_samples_per_query)

            elif 'BT' == self.sampling_str:
                mat_diffs = torch.unsqueeze(point_preds, dim=1) - torch.unsqueeze(point_preds, dim=0)
                mat_bt_probs = torch.sigmoid(mat_diffs)  # default delta=1.0

                fake_head_inds, fake_tail_inds = sample_points_Bernoulli(mat_bt_probs, num_pairs=num_samples_per_query)
            else:
                raise NotImplementedError
            #--

            # real data and generated data
            true_head_docs = batch_ranking[:, true_head_inds, :]
            true_tail_docs = batch_ranking[:, true_tail_inds, :]
            fake_head_docs = batch_ranking[:, fake_head_inds, :]
            fake_tail_docs = batch_ranking[:, fake_tail_inds, :]

            ''' optimize discriminator '''
            true_head_preds = discriminator.predict(true_head_docs, train=True)
            true_tail_preds = discriminator.predict(true_tail_docs, train=True)
            true_preds = true_head_preds - true_tail_preds
            fake_head_preds = discriminator.predict(fake_head_docs, train=True)
            fake_tail_preds = discriminator.predict(fake_tail_docs, train=True)
            fake_preds = fake_head_preds - fake_tail_preds

            dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator
            discriminator.optimizer.zero_grad()
            dis_loss.backward()
            discriminator.optimizer.step()

            ''' optimize generator '''  #
            d_fake_head_preds = discriminator.predict(fake_head_docs)
            d_fake_tail_preds = discriminator.predict(fake_tail_docs)
            d_fake_preds = self.conjugate_f(self.activation_f(d_fake_head_preds - d_fake_tail_preds))

            if 'SR' == self.sampling_str:
                log_g_probs = torch.log(mat_probs[fake_head_inds, fake_tail_inds].view(1, -1))

            elif 'BT' == self.sampling_str:
                log_g_probs = torch.log(mat_bt_probs[fake_head_inds, fake_tail_inds].view(1, -1))
            else:
                raise NotImplementedError

            g_batch_loss = -torch.mean(log_g_probs * d_fake_preds)

            generator.optimizer.zero_grad()
            g_batch_loss.backward()
            generator.optimizer.step()

    def reset_generator(self):
        self.generator.reset_parameters()

    def reset_discriminator(self):
        self.discriminator.reset_parameters()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

###### Parameter of IRFGAN_Pair ######

class IRFGAN_PairParameter(ModelParameter):
    ''' Parameter class for IRFGAN_Pair '''
    def __init__(self, debug=False):
        super(IRFGAN_PairParameter, self).__init__(model_id='IRFGAN_Pair')
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_Pair
        :return:
        """
        f_div_id = 'KL'
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG'
        # ad_training_order = 'GD'
        sampling_str = 'BT'

        self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                 f_div_id=f_div_id, ad_training_order=ad_training_order, sampling_str=sampling_str)
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
        d_epoches, g_epoches, f_div_id, ad_training_order = ad_para_dict['d_epoches'], ad_para_dict['g_epoches'],\
                                                       ad_para_dict['f_div_id'], ad_para_dict['ad_training_order']

        pair_irfgan_paras_str = s1.join([str(d_epoches), str(g_epoches), '{:,g}'.format(f_div_id),
                                         ad_training_order])

        return pair_irfgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Pair
        :param debug:
        :return:
        """
        choice_samples_per_query = [5]
        choice_ad_training_order = ['DG']  # GD for irganlist DG for point/pair
        choice_f_div_id = ['KL'] if self.debug else ['KL']  #
        choice_d_g_epoches = [(1, 1)] if self.debug else [(1, 1)] # discriminator-epoches vs. generator-epoches

        for d_g_epoches, samples_per_query, ad_training_order, f_div_id in \
                product(choice_d_g_epoches, choice_samples_per_query, choice_ad_training_order, choice_f_div_id):
            d_epoches, g_epoches = d_g_epoches

            self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                samples_per_query=samples_per_query, f_div_id=f_div_id,
                                ad_training_order=ad_training_order, sampling_str = 'BT')

            yield self.ad_para_dict
