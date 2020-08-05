#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import copy
import numpy as np

import torch

from org.archive.ltr_adversarial_learning.base.ad_machine import AdversarialMachine
from org.archive.ltr_adversarial_learning.pointwise.point_generator import Point_Generator
from org.archive.ltr_adversarial_learning.util.f_divergence import get_f_divergence_functions
from org.archive.ltr_adversarial_learning.pointwise.point_discriminator import Point_Discriminator

from org.archive.ltr_adversarial_learning.util.pair_sampling import generate_true_pairs, sample_points_Bernoulli

from org.archive.utils.pytorch.pt_extensions import Gaussian_Integral_0_Inf
apply_Gaussian_Integral_0_Inf = Gaussian_Integral_0_Inf.apply

from org.archive.l2r_global import global_gpu as gpu, tensor, global_device as device, torch_zero, torch_one, cpu_torch_one, cpu_torch_zero

class Pair_IR_FGAN(AdversarialMachine):
    ''' '''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, f_div_str='KL', sampling_str='BT', sigma=3.0):
        super(Pair_IR_FGAN, self).__init__(eval_dict=eval_dict, data_dict=data_dict)

        self.activation_f, self.conjugate_f = get_f_divergence_functions(f_div_str)
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