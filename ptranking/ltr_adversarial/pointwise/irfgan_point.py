#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import copy
from itertools import product

import torch
import torch.nn.functional as F

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine
from ptranking.ltr_adversarial.pointwise.point_generator import Point_Generator
from ptranking.ltr_adversarial.util.f_divergence import get_f_divergence_functions
from ptranking.ltr_adversarial.pointwise.point_discriminator import Point_Discriminator

class IRFGAN_Point(AdversarialMachine):
    '''  '''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, ad_para_dict=None, gpu=False, device=None):
        super(IRFGAN_Point, self).__init__(eval_dict=eval_dict, data_dict=data_dict, gpu=gpu, device=device)

        self.f_div_id = ad_para_dict['f_div_id']
        ''' muted due to default train_discriminator_generator_single_step() '''
        #self.d_epoches = ad_para_dict['d_epoches']
        #self.g_epoches = ad_para_dict['g_epoches']
        #self.ad_training_order = ad_para_dict['ad_training_order']
        self.samples_per_query = ad_para_dict['samples_per_query']

        self.activation_f, self.conjugate_f = get_f_divergence_functions(self.f_div_id)

        #sf_para_dict['ffnns']['apply_tl_af'] = False
        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)

        self.generator = Point_Generator(sf_para_dict=g_sf_para_dict, gpu=gpu, device=self.device)
        self.discriminator = Point_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=self.device)

    def fill_global_buffer(self, train_data, dict_buffer=None):
        ''' Buffer the number of positive documents per query '''
        assert self.data_dict['train_presort'] is True  # this is required for efficient truth exampling

        for entry in train_data:
            qid, _, batch_label = entry[0], entry[1], entry[2]
            if not qid in dict_buffer:
                boolean_mat = torch.gt(batch_label, 0)
                num_pos = torch.sum(boolean_mat) # number of positive documents
                dict_buffer[qid] = num_pos


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, global_buffer=None, single_step=True):
        if single_step:
            stop_training = self.train_discriminator_generator_single_step(train_data=train_data, generator=generator,
                                                            discriminator=discriminator, global_buffer=global_buffer)
            return stop_training
        else:
            if self.ad_training_order == 'DG': # being consistent with the provided code
                for d_epoch in range(self.d_epoches):
                    if d_epoch % 10 == 0:
                        generated_data = self.generate_data(train_data=train_data, generator=generator,
                                                            global_buffer=global_buffer)

                    self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

                for g_epoch in range(self.g_epoches):
                    stop_training = self.train_generator(train_data=train_data, generator=generator,
                                                         discriminator=discriminator, global_buffer=global_buffer)  # train generator
                    if stop_training: return stop_training

            else: # being consistent with Algorithms-1 in the paper
                for g_epoch in range(self.g_epoches):
                    stop_training = self.train_generator(train_data=train_data, generator=generator,
                                                         discriminator=discriminator, global_buffer=global_buffer)  # train generator
                    if stop_training: return stop_training

                for d_epoch in range(self.d_epoches):
                    if d_epoch % 10 == 0:
                        generated_data = self.generate_data(train_data=train_data, generator=generator,
                                                            global_buffer=global_buffer)

                    self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

            stop_training = False
            return stop_training


    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        discriminator.train_mode()
        for entry in train_data:
            qid, batch_ranking = entry[0], entry[1]

            if qid in generated_data:
                if self.gpu: batch_ranking = batch_ranking.to(self.device)

                pos_inds, neg_inds = generated_data[qid]

                true_docs = batch_ranking[0, pos_inds, :]
                fake_docs = batch_ranking[0, neg_inds, :]

                true_preds = discriminator.predict(true_docs)
                fake_preds = discriminator.predict(fake_docs)

                dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator

                discriminator.optimizer.zero_grad()
                dis_loss.backward()
                discriminator.optimizer.step()


    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None,
                        global_buffer=None):
        generator.train_mode()
        discriminator.eval_mode()
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]

            num_pos = global_buffer[qid]
            if num_pos < 1: continue

            batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
            pred_probs = F.softmax(torch.squeeze(batch_pred), dim=0)

            neg_inds = torch.multinomial(pred_probs, self.samples_per_query, replacement=False)
            fake_docs = batch_ranking[0, neg_inds, :]

            d_fake_preds = discriminator.predict(fake_docs)
            d_fake_preds = self.conjugate_f(self.activation_f(d_fake_preds))

            ger_loss = -torch.mean((torch.log(pred_probs[neg_inds]) * d_fake_preds))

            generator.optimizer.zero_grad()
            ger_loss.backward()
            generator.optimizer.step()

        stop_training = False
        return stop_training


    def generate_data(self, train_data=None, generator=None, global_buffer=None):
        ''' Sampling for training discriminator '''
        generator.eval_mode()

        generated_data = dict()
        for entry in train_data:
            qid, batch_ranking, _ = entry[0], entry[1], entry[2]
            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking, generator=generator,
                                                global_buffer=global_buffer)
            if samples is not None:
                generated_data[qid] = samples

        return generated_data

    def per_query_generation(self, qid, batch_ranking, generator, global_buffer):
        num_pos = global_buffer[qid]

        if num_pos >= 1:
            valid_num = min(num_pos, self.samples_per_query)
            pos_inds = torch.randperm(num_pos)[0:valid_num] # randomly select positive documents

            batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
            pred_probs = F.softmax(torch.squeeze(batch_pred), dim=0)

            neg_inds = torch.multinomial(pred_probs, valid_num, replacement=True)

            return (pos_inds, neg_inds) # torch.LongTensor as index
        else:
            return None


    def train_discriminator_generator_single_step(self, train_data=None, generator=None, discriminator=None,
                                                  global_buffer=None):
        ''' Train both discriminator and generator with a single step per query '''
        generator.train_mode()

        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.to(self.device)

            num_pos = global_buffer[qid]
            if num_pos < 1: continue

            valid_num = min(num_pos, self.samples_per_query)
            true_inds = torch.randperm(num_pos)[0:valid_num]  # randomly select positive documents

            batch_preds = generator.predict(batch_ranking)  # [batch, size_ranking]
            pred_probs = F.softmax(torch.squeeze(batch_preds), dim=0)

            if torch.isnan(pred_probs).any():
                stop_training = True
                return stop_training

            fake_inds = torch.multinomial(pred_probs, valid_num, replacement=False)

            #real data and generated data
            true_docs = batch_ranking[0, true_inds, :]
            fake_docs = batch_ranking[0, fake_inds, :]
            true_docs = torch.unsqueeze(true_docs, dim=0)
            fake_docs = torch.unsqueeze(fake_docs, dim=0)

            ''' optimize discriminator '''
            discriminator.train_mode()
            true_preds = discriminator.predict(true_docs)
            fake_preds = discriminator.predict(fake_docs)

            dis_loss = torch.mean(self.conjugate_f(self.activation_f(fake_preds))) - torch.mean(self.activation_f(true_preds))  # objective to minimize w.r.t. discriminator

            discriminator.optimizer.zero_grad()
            dis_loss.backward()
            discriminator.optimizer.step()

            ''' optimize generator '''  #
            discriminator.eval_mode()
            d_fake_preds = discriminator.predict(fake_docs)
            d_fake_preds = self.conjugate_f(self.activation_f(d_fake_preds))

            ger_loss = -torch.mean((torch.log(pred_probs[fake_inds]) * d_fake_preds))

            generator.optimizer.zero_grad()
            ger_loss.backward()
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

###### Parameter of IRFGAN_Point ######

class IRFGAN_PointParameter(ModelParameter):
    ''' Parameter class for IRFGAN_Point '''
    def __init__(self, debug=False, para_json=None):
        super(IRFGAN_PointParameter, self).__init__(model_id='IRFGAN_Point')
        self.debug = debug
        self.para_json = para_json

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_Point
        :return:
        """
        f_div_id = 'KL'
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG' # 'GD'
        samples_per_query = 5

        self.ad_para_dict = dict(model_id=self.model_id, samples_per_query=samples_per_query, d_epoches=d_epoches,
                                 g_epoches=g_epoches, f_div_id=f_div_id, ad_training_order=ad_training_order)
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

        point_irfgan_paras_str = f_div_id

        return point_irfgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Point
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
