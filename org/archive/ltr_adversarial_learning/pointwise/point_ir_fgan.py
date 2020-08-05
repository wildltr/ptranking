#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import copy

import torch
import torch.nn.functional as F

from org.archive.ltr_adversarial_learning.base.ad_machine import AdversarialMachine
from org.archive.ltr_adversarial_learning.util.point_sampling import generate_true_docs
from org.archive.ltr_adversarial_learning.pointwise.point_generator import Point_Generator
from org.archive.ltr_adversarial_learning.util.f_divergence import get_f_divergence_functions
from org.archive.ltr_adversarial_learning.pointwise.point_discriminator import Point_Discriminator

from org.archive.l2r_global import global_gpu as gpu, global_device as device


class Point_IR_FGAN(AdversarialMachine):
    ''''''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, f_div_str='KL', d_epoches=None, g_epoches=None,
                 ad_training_order=None):
        super(Point_IR_FGAN, self).__init__(eval_dict=eval_dict, data_dict=data_dict)

        self.activation_f, self.conjugate_f = get_f_divergence_functions(f_div_str)

        sf_para_dict['ffnns']['apply_tl_af'] = False

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)

        self.generator = Point_Generator(sf_para_dict=g_sf_para_dict)
        self.discriminator = Point_Discriminator(sf_para_dict=d_sf_para_dict)

        self.d_epoches = d_epoches
        self.g_epoches = g_epoches
        self.ad_training_order = ad_training_order
        #self.samples_per_query = samples_per_query


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, d_epoches=1, g_epoches=1, dict_buffer=None, num_samples_per_query=3, single_step=True):
        if single_step:
            self.train_discriminator_generator_single_step(train_data=train_data, generator=generator, discriminator=discriminator, dict_buffer=dict_buffer, num_samples_per_query=num_samples_per_query)

        else:
            if self.ad_training_order == 'DG': # being consistent with the provided code
                for d_epoch in range(self.d_epoches):
                    if d_epoch % 10 == 0:
                        generated_data = self.generate_data(train_data=train_data, generator=generator)

                    self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

                for g_epoch in range(self.g_epoches):
                    stop_training = self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator)  # train generator
                    if stop_training: return stop_training

            else: # being consistent with Algorithms-1 in the paper
                for g_epoch in range(self.g_epoches):
                    stop_training = self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator)  # train generator
                    if stop_training: return stop_training

                for d_epoch in range(self.d_epoches):
                    if d_epoch % 10 == 0:
                        generated_data = self.generate_data(train_data=train_data, generator=generator)

                    self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

        stop_training = False
        return stop_training


    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        for entry in train_data:
            qid, batch_ranking = entry[0], entry[1]

            if qid in generated_data:
                if gpu: batch_ranking = batch_ranking.to(device)

                pos_inds, neg_inds = generated_data[qid]

                true_docs = batch_ranking[0, pos_inds, :]
                fake_docs = batch_ranking[0, neg_inds, :]

                true_preds = discriminator.predict(true_docs, train=True)
                fake_preds = discriminator.predict(fake_docs, train=True)

                dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator

                discriminator.optimizer.zero_grad()
                dis_loss.backward()
                discriminator.optimizer.step()


    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None, **kwargs):

        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]

            used_batch_label = batch_label
            if gpu: batch_ranking = batch_ranking.to(device)

            # [1, ranking_size] -> [ranking_size]
            # [z, n] If input has n dimensions, then the resulting indices tensor out is of size (z×n), where z is the total number of non-zero elements in the input tensor.
            all_pos_inds = torch.gt(torch.squeeze(used_batch_label), 0).nonzero()
            num_pos = all_pos_inds.size()[0]

            if num_pos < 1: continue

            ranking_size = batch_label.size(1)
            if num_pos < ranking_size:
                half = int(ranking_size*0.5)

                if num_pos<= half:
                    num_samples = num_pos
                    #pos_inds = all_pos_inds[:, 0]
                else: # aiming for a balance
                    num_samples = half
                    #pos_inds = all_pos_inds[0:half, 0]

                batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
                batch_prob = F.softmax(torch.squeeze(batch_pred), dim=0)

                #print('num_samples', num_samples)
                #print('batch_prob', batch_prob)

                neg_inds = torch.multinomial(batch_prob, num_samples, replacement=True)

                # todo cpu inds & cuda inds w.r.t. other methods
                #if gpu: pos_inds = pos_inds.to(device)
                #return (pos_inds, neg_inds)

                #true_docs = batch_ranking[0, pos_inds, :]
                fake_docs = batch_ranking[0, neg_inds, :]

                d_fake_preds = discriminator.predict(fake_docs)
                d_fake_preds = self.conjugate_f(self.activation_f(d_fake_preds))

                ger_loss = -torch.mean((torch.log(batch_prob[neg_inds]) * d_fake_preds))

                generator.optimizer.zero_grad()
                ger_loss.backward()
                generator.optimizer.step()
            else:
                continue

        stop_training = False
        return stop_training


    def generate_data(self, train_data=None, generator=None, **kwargs):
        ''' negative sampling based on current generator for training discriminator '''

        generated_data = dict()

        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]

            samples = self.per_query_generation(qid, batch_ranking, batch_label, generator)

            if samples is not None: generated_data[qid] = samples

        return generated_data


    def per_query_generation(self, qid, batch_ranking, batch_label, generator):

        used_batch_label = batch_label

        if gpu: batch_ranking = batch_ranking.to(device)

        # [1, ranking_size] -> [ranking_size]
        # [z, n] If input has n dimensions, then the resulting indices tensor out is of size (z×n), where z is the total number of non-zero elements in the input tensor.
        all_pos_inds = torch.gt(torch.squeeze(used_batch_label), 0).nonzero()
        num_pos = all_pos_inds.size()[0]

        if num_pos < 1:
            return None

        ranking_size = batch_label.size(1)
        if num_pos < ranking_size:
            half = int(ranking_size*0.5)
            if num_pos<= half:
                num_samples = num_pos
                pos_inds = all_pos_inds[:, 0]
            else: # aiming for a balance
                num_samples = half
                pos_inds = all_pos_inds[0:half, 0]

            batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
            batch_prob = F.softmax(torch.squeeze(batch_pred), dim=0)

            #print('num_samples', num_samples)
            #print('batch_prob', batch_prob)

            neg_inds = torch.multinomial(batch_prob, num_samples, replacement=True)

            # todo cpu inds & cuda inds w.r.t. other methods
            #if gpu: pos_inds = pos_inds.to(device)
            return (pos_inds, neg_inds)
        else:
            return None


    def train_discriminator_generator_single_step(self, train_data=None, generator=None, discriminator=None, num_samples_per_query=None, dict_buffer=None, **kwargs): # train both discriminator and generator with a single step per query
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if gpu: batch_ranking = batch_ranking.to(device)

            true_inds = generate_true_docs(qid, torch.squeeze(batch_label), num_samples=num_samples_per_query, dict_true_inds=dict_buffer)

            batch_preds = generator.predict(batch_ranking, train=True)  # [batch, size_ranking]
            batch_probs = F.softmax(torch.squeeze(batch_preds), dim=0)

            fake_inds = torch.multinomial(batch_probs, true_inds.size()[0], replacement=False)

            #real data and generated data
            true_docs = batch_ranking[0, true_inds, :]
            fake_docs = batch_ranking[0, fake_inds, :]
            true_docs = torch.unsqueeze(true_docs, dim=0)
            fake_docs = torch.unsqueeze(fake_docs, dim=0)

            ''' optimize discriminator '''
            true_preds = discriminator.predict(true_docs, train=True)
            fake_preds = discriminator.predict(fake_docs, train=True)

            dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator

            discriminator.optimizer.zero_grad()
            dis_loss.backward()
            discriminator.optimizer.step()

            ''' optimize generator '''  #
            d_fake_preds = discriminator.predict(fake_docs)
            d_fake_preds = self.conjugate_f(self.activation_f(d_fake_preds))

            ger_loss = -torch.mean((torch.log(batch_probs[fake_inds]) * d_fake_preds))

            generator.optimizer.zero_grad()
            ger_loss.backward()
            generator.optimizer.step()

    def reset_generator(self):
        self.generator.reset_parameters()

    def reset_discriminator(self):
        self.discriminator.reset_parameters()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator