#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import copy
import numpy as np
from itertools import product

import torch
import torch.nn.functional as F

from ptranking.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_player import AdversarialPlayer
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine

from ptranking.ltr_global import global_gpu as gpu, global_device as device, torch_zero, torch_one, cpu_torch_one, cpu_torch_zero

class IRGAN_Pair_Generator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None, temperature=None):
        super(IRGAN_Pair_Generator, self).__init__(sf_para_dict=sf_para_dict)
        self.temperature = temperature

    def predict(self, batch_ranking, train=False):
        if train:
            self.train_mode()  # training mode for training
        else:
            self.eval_mode()  # evaluation mode for testing

        batch_pred = self.forward(batch_ranking)

        if self.temperature is not None and 1.0 != self.temperature:
                batch_pred = batch_pred / self.temperature

        return batch_pred


class IRGAN_Pair_Discriminator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None):
        super(IRGAN_Pair_Discriminator, self).__init__(sf_para_dict=sf_para_dict)

    def get_reward(self, pos_docs=None, neg_docs=None, loss_type='svm'):
        ''' used by irgan '''
        batch_pos_preds = self.predict(pos_docs)  # [batch, size_ranking]
        batch_neg_preds = self.predict(neg_docs)  # [batch, size_ranking]

        if 'svm' == loss_type:
            reward = torch.sigmoid(torch.max(torch_zero, torch_one - (batch_pos_preds - batch_neg_preds)))
        elif 'log' == loss_type:
            reward = torch.log(torch.sigmoid(batch_neg_preds - batch_pos_preds))
        else:
            raise NotImplementedError

        return reward



class IRGAN_Pair(AdversarialMachine):
    ''''''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, temperature=0.2, loss_type='svm', d_epoches=None, g_epoches=None, ad_training_order=None):
        '''

        :param eval_dict:
        :param data_dict:
        :param sf_para_dict:
        :param temperature:   according to the description around Eq-10, temperature is deployed, while it is not used within the provided code
        :param loss_type:
        :param d_epoches:
        :param g_epoches:
        :param ad_training_order:
        '''
        super(IRGAN_Pair, self).__init__(eval_dict=eval_dict, data_dict=data_dict)

        sf_para_dict['ffnns']['apply_tl_af'] = True

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        d_sf_para_dict['ffnns']['apply_tl_af'] = False
        #d_sf_para_dict['ffnns']['TL_AF'] = 'S'  # as required by the IRGAN model

        self.generator     = IRGAN_Pair_Generator(sf_para_dict=g_sf_para_dict, temperature=temperature)
        self.discriminator = IRGAN_Pair_Discriminator(sf_para_dict=d_sf_para_dict)

        self.loss_type = loss_type
        self.d_epoches = d_epoches
        self.g_epoches = g_epoches
        self.temperature = temperature
        self.ad_training_order = ad_training_order
        #self.samples_per_query = samples_per_query


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, dict_buffer=None):
        if self.ad_training_order == 'DG':
            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator)

                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

            for g_epoch in range(self.g_epoches):
                self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator)  # train generator

        else:
            for g_epoch in range(self.g_epoches):
                self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator)  # train generator

            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator)

                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

        stop_training = False
        return stop_training


    def generate_data(self, train_data=None, generator=None, **kwargs):
        '''
        negative sampling based on current generator for training discriminator
        todo this is a re-implementation as the released irgan-tensorflow, but it seems that this part of irgan-tensorflow is not consistent with the discription of the paper (i.e., the description below Eq. 7)
        '''
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
        pos_inds = torch.gt(torch.squeeze(used_batch_label), 0).nonzero()
        num_pos = pos_inds.size()[0]

        if num_pos < 1:
            return None

        if num_pos < used_batch_label.size(1):
            '''
            sampling condition: pos_inds<|tor_batch_stds|
            sample numbers: min(|pos_inds|, |total_neg_inds|)
            '''
            batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
            # batch_probs = torch.sigmoid(torch.squeeze(batch_preds))
            batch_prob = F.softmax(torch.squeeze(batch_pred), dim=0)

            # intersection implementation (keeping consistent with the released irgan-tensorflow): remove the positive part, then sample
            reversed_batch_label = torch.where(used_batch_label > 0, cpu_torch_zero, cpu_torch_one)  # negative values
            total_neg_inds = torch.nonzero(torch.squeeze(reversed_batch_label))

            num_samples = min(num_pos, total_neg_inds.size()[0])

            batch_neg_probs = batch_prob[total_neg_inds[:, 0]]
            tmp_neg_inds = torch.multinomial(batch_neg_probs, num_samples, replacement=False)
            sample_neg_inds = total_neg_inds[tmp_neg_inds]  # using the original indices within total_neg_inds rather than batch_neg_probs

            if num_samples < num_pos:
                tmp_pos_inds = torch.multinomial(torch.ones(num_pos), num_samples=num_samples, replacement=False)
                sample_pos_inds = pos_inds[tmp_pos_inds]
            else:
                sample_pos_inds = pos_inds

            # generated_data[qid] = (sample_pos_inds, sample_neg_inds)
            # convert 2-d index matrix into 1-d index matrix
            return (np.squeeze(sample_pos_inds, axis=1), np.squeeze(sample_neg_inds, axis=1))
        else:
            return None


    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        for entry in train_data:
            qid, batch_ranking = entry[0], entry[1]

            if qid in generated_data:
                if gpu: batch_ranking = batch_ranking.to(device)

                pos_inds, neg_inds = generated_data[qid]
                pos_docs = batch_ranking[:, pos_inds, :]
                neg_docs = batch_ranking[:, neg_inds, :]

                batch_pos_preds = discriminator.predict(pos_docs, train=True) # [batch, size_ranking]
                batch_neg_preds = discriminator.predict(neg_docs, train=True) # [batch, size_ranking]

                if 'svm' == self.loss_type:
                    #dis_loss = torch.mean(torch.max(torch.zeros(1), 1.0-(batch_pos_preds-batch_neg_preds)))
                    dis_loss = torch.mean(torch.max(torch_zero, torch_one - (batch_pos_preds - batch_neg_preds)))
                elif 'log' == self.loss_type:
                    dis_loss = -torch.mean(torch.log(torch.sigmoid(batch_pos_preds-batch_neg_preds)))
                else:
                    raise NotImplementedError

                discriminator.optimizer.zero_grad()
                dis_loss.backward()
                discriminator.optimizer.step()


    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None, **kwargs):
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if gpu: batch_ranking = batch_ranking.to(device)

            pos_inds = torch.gt(torch.squeeze(batch_label), 0).nonzero()

            g_preds = generator.predict(batch_ranking, train=True)
            g_probs = torch.sigmoid(torch.squeeze(g_preds))

            neg_inds = torch.multinomial(g_probs, pos_inds.size(0), replacement=True)

            pos_docs = batch_ranking[:, pos_inds[:, 0], :]
            neg_docs = batch_ranking[:, neg_inds, :]

            reward = discriminator.get_reward(pos_docs=pos_docs, neg_docs=neg_docs, loss_type=self.loss_type)

            g_loss = -torch.mean((torch.log(g_probs[neg_inds]) * reward))

            generator.optimizer.zero_grad()
            g_loss.backward()
            generator.optimizer.step()

    def reset_generator(self):
        self.generator.reset_parameters()

    def reset_discriminator(self):
        self.discriminator.reset_parameters()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

###### Parameter of IRGAN_Pair ######

class IRGAN_PairParameter(ModelParameter):
    ''' Parameter class for Pair_IR_GAN '''
    def __init__(self, debug=False):
        super(IRGAN_PairParameter, self).__init__(model_id='IRGAN_Pair')
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_Pair
        :return:
        """
        temperature = 0.2
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG'
        # ad_training_order = 'GD'

        self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                 temperature=temperature, ad_training_order=ad_training_order, loss_type='svm')
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
        d_epoches, g_epoches, temperature, ad_training_order, loss_type = \
            ad_para_dict['d_epoches'], ad_para_dict['g_epoches'], ad_para_dict['temperature'],\
            ad_para_dict['ad_training_order'], ad_para_dict['loss_type']

        pair_irgan_paras_str = s1.join([str(d_epoches), str(g_epoches), '{:,g}'.format(temperature),
                                        ad_training_order, loss_type])

        return pair_irgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Pair
        :param debug:
        :return:
        """
        choice_samples_per_query = [5]
        choice_ad_training_order = ['DG']  # GD for irganlist DG for point/pair
        choice_temperatures = [0.5] if self.debug else [0.5]  # 0.5, 1.0
        choice_d_g_epoches = [(1, 1)] if self.debug else [(1, 1)]  # discriminator-epoches vs. generator-epoches

        choice_losstype_d = ['svm']

        for d_g_epoches, samples_per_query, ad_training_order, temperature, loss_type_d in product(choice_d_g_epoches,
                       choice_samples_per_query, choice_ad_training_order, choice_temperatures, choice_losstype_d):
            d_epoches, g_epoches = d_g_epoches

            self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                     samples_per_query=samples_per_query, temperature=temperature,
                                     ad_training_order=ad_training_order, loss_type=loss_type_d)

            yield self.ad_para_dict
