#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_player import AdversarialPlayer
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine


LAMBDA = 0.5

# Discriminator Loss #
#dis_loss_function = nn.BCELoss() # todo it leads to weird errors which should be explored ?!
dis_loss_function = nn.BCEWithLogitsLoss()


class IRGAN_Point_Generator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None, temperature=0.5, gpu=False, device=None):
        super(IRGAN_Point_Generator, self).__init__(sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.temperature = temperature

    def predict(self, batch_ranking):
        batch_pred = self.forward(batch_ranking)
        if self.temperature is not None and 1.0 != self.temperature:
                batch_pred = batch_pred / self.temperature

        return batch_pred


class IRGAN_Point_Discriminator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None, gpu=False, device=None):
        super(IRGAN_Point_Discriminator, self).__init__(sf_para_dict=sf_para_dict, gpu=gpu, device=device)

    def get_reward(self, batch_ranking):
        ''' used by irgan '''
        reward = (self.predict(batch_ranking) - 0.5) * 2
        reward = torch.squeeze(reward, dim=0)
        return reward



class IRGAN_Point(AdversarialMachine):
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, ad_para_dict=None, gpu=False, device=None):
        '''
        :param ad_training_order: really matters, DG is preferred than GD
        '''
        super(IRGAN_Point, self).__init__(eval_dict=eval_dict, data_dict=data_dict, gpu=gpu, device=device)

        ''' required final layer setting for Point_IR_GAN '''
        # the setting of 'apply_tl_af=False' is due to the later application of softmax function w.r.t. all documents
        # TODO experiments show it is quite important to be True, otherwise will be nan issues.
        assert sf_para_dict[sf_para_dict['sf_id']]['apply_tl_af'] == True # local assignment affects the grid-evaluation

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        #d_sf_para_dict['ffnns']['apply_tl_af'] = True
        d_sf_para_dict[sf_para_dict['sf_id']]['TL_AF'] = 'S' # as required by the IRGAN model

        self.generator = IRGAN_Point_Generator(sf_para_dict=g_sf_para_dict, temperature=ad_para_dict['temperature'], gpu=gpu, device=device)
        self.discriminator = IRGAN_Point_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=device)

        self.d_epoches = ad_para_dict['d_epoches']
        self.g_epoches = ad_para_dict['g_epoches']
        self.temperature = ad_para_dict['temperature']
        self.ad_training_order = ad_para_dict['ad_training_order']
        self.samples_per_query = ad_para_dict['samples_per_query']

    def fill_global_buffer(self, train_data, dict_buffer=None):
        ''' Buffer the number of positive documents per query '''
        assert self.data_dict['train_presort'] is True  # this is required for efficient truth exampling

        for entry in train_data:
            qid, _, batch_label = entry[0], entry[1], entry[2]
            if not qid in dict_buffer:
                boolean_mat = torch.gt(batch_label, 0)
                num_pos = torch.sum(boolean_mat) # number of positive documents
                dict_buffer[qid] = num_pos


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, global_buffer=None):
        if self.ad_training_order == 'DG': # being consistent with the provided code
            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator, global_buffer=global_buffer)
                # train discriminator
                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)

            for g_epoch in range(self.g_epoches):
                stop_training = self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator,
                                                     global_buffer=global_buffer)  # train generator
                if stop_training: return stop_training

        else: # being consistent with Algorithms-1 in the paper
            for g_epoch in range(self.g_epoches): # train generator
                stop_training = self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator,
                                                     global_buffer=global_buffer)
                if stop_training: return stop_training

            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator, global_buffer=global_buffer)

                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

        stop_training = False
        return stop_training


    def generate_data(self, train_data=None, generator=None, global_buffer=None):
        ''' Sampling for training discriminator '''
        generator.eval_mode()

        generated_data = dict()
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking, generator=generator, global_buffer=global_buffer)

            if samples is not None: generated_data[qid] = samples

        return generated_data


    def per_query_generation(self, qid, batch_ranking, generator, global_buffer):
        num_pos = global_buffer[qid]

        if num_pos >= 1:
            valid_num = min(num_pos, self.samples_per_query)
            pos_inds = torch.randperm(num_pos)[0:valid_num] # randomly select positive documents

            if self.gpu: batch_ranking = batch_ranking.to(self.device) # [batch, size_ranking]

            batch_pred = generator.predict(batch_ranking)
            pred_probs = F.softmax(torch.squeeze(batch_pred), dim=0)

            neg_inds = torch.multinomial(pred_probs, valid_num, replacement=True)

            return (pos_inds, neg_inds) # torch.LongTensor as index
        else:
            return None


    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        discriminator.train_mode()
        for entry in train_data:
            qid, batch_ranking = entry[0], entry[1]

            if qid in generated_data:
                if self.gpu: batch_ranking = batch_ranking.to(self.device)

                pos_inds, neg_inds = generated_data[qid]
                pos_docs = batch_ranking[0, pos_inds, :]
                neg_docs = batch_ranking[0, neg_inds, :]

                num = len(pos_inds)
                batch_docs = torch.unsqueeze(torch.cat([pos_docs, neg_docs], dim=0), dim=0)  # [batch=1, num_doc, num_features] w.r.t. one query
                batch_label = torch.unsqueeze(torch.cat((torch.ones(num), torch.zeros(num)), dim=0), dim=0)  # corresponding labels
                if self.gpu: batch_label = batch_label.to(self.device)

                batch_pred = discriminator.predict(batch_docs)

                # based on inbuilt BCELoss, since both generator and discriminator include the sigmoid layer as the last layer
                dis_loss = dis_loss_function(batch_pred, batch_label)

                discriminator.optimizer.zero_grad()
                dis_loss.backward()
                discriminator.optimizer.step()


    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None,
                        global_buffer=None):
        generator.train_mode()
        discriminator.eval_mode()
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.to(self.device)

            num_pos = global_buffer[qid]
            if num_pos < 1: continue

            ranking_inds = torch.arange(batch_ranking.size(1))
            pos_inds = ranking_inds[0:num_pos]

            g_preds = generator.predict(batch_ranking)
            if torch.isnan(g_preds).any():
                print('Including NaN error.')
                print('g_preds', g_preds)
                stop_training = True
                return stop_training

            g_probs = F.softmax(torch.squeeze(g_preds), dim=0)

            prob_IS = g_probs * (1.0 - LAMBDA)
            #prob_IS[pos_inds[:, 0]] = prob_IS[pos_inds[:, 0]] + (LAMBDA / (1.0 * pos_inds.size(0)))
            prob_IS[pos_inds] += (LAMBDA / (1.0 * num_pos))

            choose_inds = torch.multinomial(prob_IS, num_pos * 5, replacement=True)

            choose_IS = g_probs[choose_inds] / prob_IS[choose_inds]
            choose_docs = batch_ranking[0, choose_inds, :]
            choose_reward = discriminator.get_reward(torch.unsqueeze(choose_docs, dim=0))  # => [batch, size_ranking, num_features]

            ''' Eq-22 of the arXiv IRGAN paper '''
            g_loss = -torch.mean((torch.log(g_probs[choose_inds]) * choose_reward * choose_IS))

            # do we really need choose_IS
            #g_loss = -torch.mean((torch.log(g_probs[choose_inds]) * choose_reward))

            generator.optimizer.zero_grad()
            g_loss.backward()
            generator.optimizer.step()

        stop_training = False
        return stop_training

    def reset_generator(self):
        self.generator.init()

    def reset_discriminator(self):
        self.discriminator.init()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator


###### Parameter of IRGAN_Point ######

class IRGAN_PointParameter(ModelParameter):
    ''' Parameter class for IRGAN_Point '''
    def __init__(self, debug=False, para_json=None):
        super(IRGAN_PointParameter, self).__init__(model_id='IRGAN_Point', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """ Default parameter setting for IRGAN_Point """
        temperature = 0.5
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG' # 'GD'
        samples_per_query = 5

        self.ad_para_dict = dict(model_id=self.model_id,
                                 ad_training_order=ad_training_order, d_epoches=d_epoches, g_epoches=g_epoches,
                                 temperature=temperature, samples_per_query=samples_per_query)
        return self.ad_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        """
        # using specified para-dict or inner para-dict
        ad_para_dict = given_para_dict if given_para_dict is not None else self.ad_para_dict

        s1 = ':' if log else '_'
        d_epoches, g_epoches, temperature, ad_training_order, samples_per_query = ad_para_dict['d_epoches'],\
                                          ad_para_dict['g_epoches'], ad_para_dict['temperature'],\
                                          ad_para_dict['ad_training_order'], ad_para_dict['samples_per_query']

        irgan_point_paras_str = s1.join([str(d_epoches), str(g_epoches), '{:,g}'.format(temperature),
                                         ad_training_order, str(samples_per_query)])

        return irgan_point_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Point
        """
        if self.use_json:
            d_g_epoch_strings = self.json_dict['d_g_epoch']
            choice_temperature = self.json_dict['temperature']
            choice_samples_per_query = self.json_dict['samples_per_query']
            choice_ad_training_order = self.json_dict['ad_training_order']
            choice_d_g_epoch = []
            for d_g_epoch_str in d_g_epoch_strings:
                epoch_arr = d_g_epoch_str.split('-')
                choice_d_g_epoch.append((int(epoch_arr[0]), int(epoch_arr[1])))
        else:
            choice_samples_per_query = [5]
            choice_ad_training_order = ['DG']  # GD for irganlist DG for point/pair
            choice_temperature = [0.5] if self.debug else [0.5]  # 0.5, 1.0
            choice_d_g_epoch = [(1, 1)] if self.debug else [(1, 1)] # discriminator-epoches vs. generator-epoches

        for d_g_epoches, samples_per_query, ad_training_order, temperature in \
                product(choice_d_g_epoch, choice_samples_per_query, choice_ad_training_order, choice_temperature):
            d_epoches, g_epoches = d_g_epoches

            self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                samples_per_query=samples_per_query, temperature=temperature,
                                ad_training_order=ad_training_order)

            yield self.ad_para_dict
