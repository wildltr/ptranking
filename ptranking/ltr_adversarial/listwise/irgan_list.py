#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import json
import copy
from itertools import product

import torch

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine
from ptranking.ltr_adversarial.listwise.list_generator import List_Generator
from ptranking.ltr_adversarial.listwise.list_discriminator import List_Discriminator

from ptranking.ltr_adversarial.util.list_probability import log_ranking_prob_Bradley_Terry, log_ranking_prob_Plackett_Luce

from ptranking.utils.pytorch.pt_extensions import arg_shuffle_ties
from ptranking.ltr_adversarial.util.list_sampling import gumbel_softmax


class IRGAN_List(AdversarialMachine):
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, ad_para_dict=None, optimal_train=False, gpu=False, device=None):
        '''
        :param optimal_train: training with supervised generator or discriminator
        '''
        super(IRGAN_List, self).__init__(eval_dict=eval_dict, data_dict=data_dict, gpu=gpu, device=device)

        g_sf_para_dict = sf_para_dict

        # todo support all setting based Nan checking
        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        d_sf_para_dict[sf_para_dict['sf_id']]['apply_tl_af'] = True
        d_sf_para_dict[sf_para_dict['sf_id']]['TL_AF'] = 'S'

        self.generator = List_Generator(sf_para_dict=g_sf_para_dict, gpu=gpu, device=device)
        self.discriminator = List_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=device)

        self.super_generator = List_Generator(sf_para_dict=g_sf_para_dict, gpu=gpu, device=device)
        self.super_discriminator = List_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=device)

        self.top_k = ad_para_dict['top_k']
        self.d_epoches = ad_para_dict['d_epoches']
        self.g_epoches = ad_para_dict['g_epoches']
        self.temperature = ad_para_dict['temperature']
        self.ad_training_order = ad_para_dict['ad_training_order']
        self.samples_per_query = ad_para_dict['samples_per_query']
        self.PL_discriminator = ad_para_dict['PL_D']
        self.replace_trick_4_generator = ad_para_dict['repTrick']
        self.drop_discriminator_log_4_reward = ad_para_dict['dropLog']
        self.optimal_train = optimal_train

        self.pre_check()


    def pre_check(self):
        if self.eval_dict['mask_label']:
            # it is impossible to generate a standard ranking due to the existance of '-1'
            assert self.top_k is not None


    def burn_in(self, train_data=None):
        if self.optimal_train:
            for entry in train_data:
                qid, batch_ranking = entry[0], entry[1]
                if self.gpu: batch_ranking = batch_ranking.to(self.device)

                g_batch_pred = self.super_generator.predict(batch_ranking)
                g_batch_log_ranking = log_ranking_prob_Plackett_Luce(g_batch_pred)
                g_loss = -torch.mean(g_batch_log_ranking)

                # alternative debugging
                #g_batch_logcumsumexps = apply_LogCumsumExp(g_batch_preds)
                #g_loss = torch.sum(g_batch_logcumsumexps - g_batch_preds)

                self.super_generator.optimizer.zero_grad()
                g_loss.backward()
                self.super_generator.optimizer.step()

                d_batch_pred = self.super_discriminator.predict(batch_ranking)

                if self.PL_discriminator:
                    d_batch_ranking_prob = log_ranking_prob_Plackett_Luce(d_batch_pred)
                else:
                    d_batch_ranking_prob = log_ranking_prob_Bradley_Terry(d_batch_pred)

                d_loss = -torch.mean(d_batch_ranking_prob)  # objective to minimize

                # alternative debugging
                #d_batch_logcumsumexps = apply_LogCumsumExp(d_batch_preds)
                #d_loss = torch.sum(d_batch_logcumsumexps - d_batch_preds)

                self.super_discriminator.optimizer.zero_grad()
                d_loss.backward()
                self.super_discriminator.optimizer.step()

    def fill_global_buffer(self, train_data, dict_buffer=None):
        """ for listwise, no particular global information is required """
        assert self.data_dict['train_presort'] is True

    def mini_max_train(self, train_data=None, generator=None, discriminator=None, global_buffer=None, per_query_ad_training=True):
        if per_query_ad_training:
            self.per_query_mini_max_train(train_data=train_data, generator=generator, discriminator=discriminator,
                                          ad_training_order=self.ad_training_order, top_k=self.top_k,
                                          samples_per_query=self.samples_per_query,
                                          per_query_g_epoch=self.g_epoches, per_query_d_epoch=self.d_epoches,
                                          replace_trick_4_generator=self.replace_trick_4_generator,
                                          PL_discriminator=self.PL_discriminator,
                                          drop_discriminator_log_4_reward=self.drop_discriminator_log_4_reward,
                                          temperature=self.temperature)
        else:
            raise NotImplementedError

        stop_training = False
        return stop_training


    def per_query_mini_max_train(self, train_data=None, generator=None, discriminator=None,
                                 ad_training_order='DG', samples_per_query=5, per_query_g_epoch=3, per_query_d_epoch=3,
                                 top_k=10, # todo does not work for reparameterization due to ltr_adhoc-size mismatch
                                 replace_trick_4_generator=True,  # replace leads to better
                                 PL_discriminator=False, # both can work
                                 replace_trick_4_discriminator=False, # False is a must
                                 drop_discriminator_log_4_reward=True, # both can work
                                 temperature=None):

        g_mod = 2 if per_query_d_epoch > 1 else 1 # regarding the generation

        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.to(self.device)

            if ad_training_order == 'DG':
                # optimising discriminator
                discriminator.train_mode()
                if self.optimal_train:
                    self.super_generator.eval_mode()
                else:
                    generator.eval_mode()

                for d_epoch in range(per_query_d_epoch):
                    if d_epoch % g_mod == 0: # update generated data
                        if self.optimal_train:
                            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking,
                                batch_label=batch_label, pos_and_neg=True, generator=self.super_generator, top_k=top_k,
                                samples_per_query=samples_per_query, temperature=temperature)
                        else:
                            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking,
                                batch_label=batch_label, pos_and_neg=True, generator=generator, top_k=top_k,
                                samples_per_query=samples_per_query, temperature=temperature)

                    if samples is None:
                        continue
                    else:
                        batch_std_sample_ranking, batch_gen_sample_ranking = samples
                        self.train_discriminator(discriminator=discriminator, PL_discriminator=PL_discriminator,
                                                 batch_std_sample_ranking=batch_std_sample_ranking,
                                                 batch_gen_sample_ranking=batch_gen_sample_ranking,
                                                 replace_trick_4_discriminator=replace_trick_4_discriminator)
                # optimising generator
                generator.train_mode()
                if self.optimal_train:
                    self.super_discriminator.eval_mode()
                else:
                    discriminator.eval_mode()

                for _ in range(per_query_g_epoch):
                    if self.optimal_train:
                        self.train_generator(batch_ranking=batch_ranking, generator=generator,
                                             samples_per_query=samples_per_query, top_k=top_k,
                                             discriminator=self.super_discriminator, PL_discriminator=PL_discriminator,
                                             replace_trick_4_generator=replace_trick_4_generator,
                                             drop_discriminator_log_4_reward=drop_discriminator_log_4_reward,
                                             temperature=temperature)
                    else:
                        self.train_generator(batch_ranking=batch_ranking, generator=generator, samples_per_query=samples_per_query, top_k=top_k,
                            discriminator=discriminator, PL_discriminator=PL_discriminator, replace_trick_4_generator=replace_trick_4_generator,
                            drop_discriminator_log_4_reward=drop_discriminator_log_4_reward, temperature=temperature)
            else:
                # optimising generator
                generator.train_mode()
                if self.optimal_train:
                    self.super_discriminator.eval_mode()
                else:
                    discriminator.eval_mode()

                if self.optimal_train:
                    for _ in range(per_query_g_epoch):
                        self.train_generator(batch_ranking=batch_ranking, generator=generator,
                                             samples_per_query=samples_per_query, top_k=top_k, temperature=temperature,
                                             discriminator=self.super_discriminator, PL_discriminator=PL_discriminator,
                                             replace_trick_4_generator=replace_trick_4_generator,
                                             drop_discriminator_log_4_reward=drop_discriminator_log_4_reward)
                else:
                    for _ in range(per_query_g_epoch):
                        self.train_generator(batch_ranking=batch_ranking, generator=generator,
                                             samples_per_query=samples_per_query, top_k=top_k, temperature=temperature,
                                             discriminator=discriminator, PL_discriminator=PL_discriminator,
                                             replace_trick_4_generator=replace_trick_4_generator,
                                             drop_discriminator_log_4_reward=drop_discriminator_log_4_reward)
                # optimising discriminator
                discriminator.train_mode()
                if self.optimal_train:
                    self.super_generator.eval_mode()
                else:
                    generator.eval_mode()

                for d_epoch in range(per_query_d_epoch):
                    if d_epoch % g_mod == 0:
                        if self.optimal_train:
                            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking,
                                batch_label=batch_label, pos_and_neg=True, generator=self.super_generator,
                                temperature=temperature, top_k=top_k, samples_per_query=samples_per_query)
                        else:
                            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking,
                                batch_label=batch_label, pos_and_neg=True, generator=generator,
                                temperature=temperature, samples_per_query=samples_per_query, top_k=top_k)

                    if samples is None: continue
                    else:
                        batch_std_sample_ranking, batch_gen_sample_ranking = samples
                        self.train_discriminator(discriminator=discriminator, PL_discriminator=PL_discriminator,
                                                 batch_std_sample_ranking=batch_std_sample_ranking,
                                                 batch_gen_sample_ranking=batch_gen_sample_ranking,
                                                 replace_trick_4_discriminator=replace_trick_4_discriminator)


    def per_query_generation(self, qid=None, batch_ranking=None, batch_label=None, pos_and_neg=None, generator=None,
                             samples_per_query=None, top_k=None, temperature=None):
        '''
        :param pos_and_neg: corresponding to discriminator optimization or generator optimization
        '''
        g_batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
        batch_gen_stochastic_prob = gumbel_softmax(g_batch_pred, samples_per_query=samples_per_query, temperature=temperature, cuda=self.gpu, cuda_device=self.device)
        sorted_batch_gen_stochastic_probs, batch_gen_sto_sorted_inds = torch.sort(batch_gen_stochastic_prob, dim=1, descending=True)

        if pos_and_neg: # for training discriminator
            used_batch_label = batch_label

            ''' Generate truth-ranking based on shuffling ties
            There is not need to firstly filter out documents of '-1', due to the descending sorting and we only use the top ones
            BTW, the only required condition is: the number of non-minus-one documents is larger than top_k, which builds upon the customized mask_data()
            '''
            per_query_label = torch.squeeze(used_batch_label)
            list_std_sto_sorted_inds = []
            for i in range(samples_per_query):
                shuffle_ties_inds = arg_shuffle_ties(per_query_label, descending=True)
                list_std_sto_sorted_inds.append(shuffle_ties_inds)
            batch_std_sto_sorted_inds = torch.stack(list_std_sto_sorted_inds, dim=0)

            list_pos_ranking, list_neg_ranking = [], []
            if top_k is None:  # using all documents
                for i in range(samples_per_query):
                    pos_inds = batch_std_sto_sorted_inds[i, :]
                    pos_ranking = batch_ranking[0, pos_inds, :]
                    list_pos_ranking.append(pos_ranking)

                    neg_inds = batch_gen_sto_sorted_inds[i, :]
                    neg_ranking = batch_ranking[0, neg_inds, :]
                    list_neg_ranking.append(neg_ranking)
            else:
                for i in range(samples_per_query):
                    pos_inds = batch_std_sto_sorted_inds[i, 0:top_k]
                    pos_ranking = batch_ranking[0, pos_inds, :]  # sampled sublist of documents
                    list_pos_ranking.append(pos_ranking)

                    neg_inds = batch_gen_sto_sorted_inds[i, 0:top_k]
                    neg_ranking = batch_ranking[0, neg_inds, :]
                    list_neg_ranking.append(neg_ranking)

            batch_std_sample_ranking = torch.stack(list_pos_ranking, dim=0)
            batch_gen_sample_ranking = torch.stack(list_neg_ranking, dim=0)

            return batch_std_sample_ranking, batch_gen_sample_ranking

        else: # for training generator
            if top_k is None:
                return sorted_batch_gen_stochastic_probs, batch_gen_sto_sorted_inds
            else:
                list_g_sort_top_preds, list_g_sort_top_inds = [], []  # required to cope with ranking_size mismatch
                for i in range(samples_per_query):
                    neg_inds = batch_gen_sto_sorted_inds[i, 0:top_k]
                    list_g_sort_top_inds.append(neg_inds)

                    top_gen_stochastic_probs = sorted_batch_gen_stochastic_probs[i, 0:top_k]
                    list_g_sort_top_preds.append(top_gen_stochastic_probs)

                top_sorted_batch_gen_stochastic_probs = torch.stack(list_g_sort_top_preds, dim=0)
                return top_sorted_batch_gen_stochastic_probs, list_g_sort_top_inds


    def get_reward(self, reward_d_batch_preds_4_gen_sorted_as_g, PL_discriminator=None, replace_trick_4_generator=None, drop_discriminator_log_4_reward=None):

        if PL_discriminator:
            reward_d_batch_log_ranking_prob_4_gen = log_ranking_prob_Plackett_Luce(reward_d_batch_preds_4_gen_sorted_as_g)
        else:
            reward_d_batch_log_ranking_prob_4_gen = log_ranking_prob_Bradley_Terry(reward_d_batch_preds_4_gen_sorted_as_g)

        if replace_trick_4_generator:
            if drop_discriminator_log_4_reward:
                batch_rewards = -torch.exp(reward_d_batch_log_ranking_prob_4_gen)
            else:
                batch_rewards = -reward_d_batch_log_ranking_prob_4_gen
        else:
            if drop_discriminator_log_4_reward:
                batch_rewards = torch.exp(1.0 - reward_d_batch_log_ranking_prob_4_gen)
            else:
                batch_rewards = torch.log(1.0 - reward_d_batch_log_ranking_prob_4_gen)

        return batch_rewards


    def train_discriminator(self, discriminator, batch_std_sample_ranking=None, batch_gen_sample_ranking=None, PL_discriminator=None, replace_trick_4_discriminator=None):

        d_batch_preds_4_std = discriminator.predict(batch_std_sample_ranking)
        d_batch_preds_4_gen = discriminator.predict(batch_gen_sample_ranking)

        # debugging
        # discriminator.stop_training(d_batch_preds_4_std)
        # discriminator.stop_training(d_batch_preds_4_gen)

        if PL_discriminator:
            d_batch_log_ranking_prob_4_std = log_ranking_prob_Plackett_Luce(d_batch_preds_4_std)
            d_batch_log_ranking_prob_4_gen = log_ranking_prob_Plackett_Luce(d_batch_preds_4_gen)
        else:
            d_batch_log_ranking_prob_4_std = log_ranking_prob_Bradley_Terry(d_batch_preds_4_std)
            d_batch_log_ranking_prob_4_gen = log_ranking_prob_Bradley_Terry(d_batch_preds_4_gen)

        if replace_trick_4_discriminator:  # replace trick
            dis_loss = torch.sum(
                d_batch_log_ranking_prob_4_gen - d_batch_log_ranking_prob_4_std)  # objective to minimize

        else:  # standard cross-entropy loss
            dis_loss = - (torch.sum(d_batch_log_ranking_prob_4_std) + torch.sum(
                torch.log(1.0 - d_batch_log_ranking_prob_4_gen)))

        discriminator.optimizer.zero_grad()
        dis_loss.backward()
        discriminator.optimizer.step()


    def train_generator(self, batch_ranking, generator=None, samples_per_query=None, top_k=None,
                        discriminator=None, PL_discriminator=None, replace_trick_4_generator=None,
                        drop_discriminator_log_4_reward=None, temperature=None):
        # todo the variance issue really maters !
        # note that the ranking_size should be consistent between generator and discriminator
        reward_d_preds_4_gen = discriminator.predict(batch_ranking)

        if top_k is None:
            sorted_batch_gen_stochastic_probs, batch_gen_sto_sorted_inds = self.per_query_generation(
                pos_and_neg=False, generator=generator, batch_ranking=batch_ranking, top_k=top_k,
                samples_per_query=samples_per_query, temperature=temperature)

            list_d_preds = []
            for i in range(samples_per_query):
                # sort according to generator
                reward_d_preds_4_gen_sorted_as_g = reward_d_preds_4_gen[0, batch_gen_sto_sorted_inds[i, :]]
                list_d_preds.append(reward_d_preds_4_gen_sorted_as_g)

            reward_d_batch_preds_4_gen_sorted_as_g = torch.stack(list_d_preds, dim=0)
            g_batch_gen_stochastic_probs = sorted_batch_gen_stochastic_probs

        else:
            top_sorted_batch_gen_stochastic_probs, list_g_sort_top_inds = self.per_query_generation(
                pos_and_neg=False, generator=generator, batch_ranking=batch_ranking, top_k=top_k,
                samples_per_query=samples_per_query, temperature=temperature)

            list_d_top_preds = []
            for i, neg_inds in enumerate(list_g_sort_top_inds):
                top_reward_d_preds_4_gen = reward_d_preds_4_gen[0, neg_inds]  # sampled sublist of documents
                list_d_top_preds.append(top_reward_d_preds_4_gen)

            reward_d_batch_preds_4_gen_sorted_as_g = torch.stack(list_d_top_preds, dim=0)
            g_batch_gen_stochastic_probs = top_sorted_batch_gen_stochastic_probs

        # directly use stochastic predictions
        g_batch_log_ranking_prob_4_gen = log_ranking_prob_Plackett_Luce(g_batch_gen_stochastic_probs)

        ''' discriminator reward '''
        # discriminator.stop_training(reward_d_batch_preds_4_gen_sorted_as_g)
        batch_rewards = self.get_reward(reward_d_batch_preds_4_gen_sorted_as_g,
                                        PL_discriminator=PL_discriminator,
                                        replace_trick_4_generator=replace_trick_4_generator,
                                        drop_discriminator_log_4_reward=drop_discriminator_log_4_reward)
        '''
        weighting w.r.t. the discriminator
        without reward, there is not connection between generator and discriminator
        '''
        g_batch_loss = torch.mean(g_batch_log_ranking_prob_4_gen * batch_rewards.detach())

        generator.optimizer.zero_grad()
        g_batch_loss.backward()
        generator.optimizer.step()


    def reset_generator(self):
        self.generator.init()

    def reset_discriminator(self):
        self.discriminator.init()

    def get_generator(self, super=False):
        #print('super', super)
        return self.super_generator if super else self.generator

    def get_discriminator(self, super=False):
        return self.super_discriminator if super else self.discriminator


###### Parameter of IRGAN_List ######

class IRGAN_ListParameter(ModelParameter):
    ''' Parameter class for IRGAN_List '''
    def __init__(self, debug=False, para_json=None):
        super(IRGAN_ListParameter, self).__init__(model_id='IRGAN_List', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_List
        :return:
        """
        temperature = 0.2
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG' #

        self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                            temperature=temperature, ad_training_order=ad_training_order, samples_per_query=10,
                            top_k=2, PL_D=True, repTrick=False, dropLog=True)

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

        d_epoches, g_epoches, temperature, ad_training_order = ad_para_dict['d_epoches'], ad_para_dict['g_epoches'],\
                                                       ad_para_dict['temperature'], ad_para_dict['ad_training_order']

        prefix = s1.join([str(d_epoches), str(g_epoches), '{:,g}'.format(temperature), ad_training_order])

        top_k, PL_D, repTrick, dropLog = ad_para_dict['top_k'], ad_para_dict['PL_D'], ad_para_dict['repTrick'], \
                                       ad_para_dict['dropLog']

        top_k_str = 'topAll' if top_k is None else 'top' + str(top_k)
        s_str = 'S' + str(ad_para_dict['samples_per_query'])
        df_str = 'PLD' if PL_D else 'BTD'
        prefix = s1.join([prefix, top_k_str, s_str, df_str])

        if repTrick: prefix += '_Rep'
        if dropLog:  prefix += '_DropLog'

        list_irgan_paras_str = prefix
        return list_irgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_List
        """
        if self.use_json:
            choice_samples_per_query = self.json_dict['samples_per_query']
            choice_ad_training_order = self.json_dict['ad_training_order']
            choice_temperature = self.json_dict['temperature']
            d_g_epoch_strings = self.json_dict['d_g_epoch']
            choice_d_g_epoch = []
            for d_g_epoch_str in d_g_epoch_strings:
                epoch_arr = d_g_epoch_str.split('-')
                choice_d_g_epoch.append((int(epoch_arr[0]), int(epoch_arr[1])))

            choice_top_k = self.json_dict['top_k']
            choice_PL_D = self.json_dict['PL_D']
            choice_repTrick = self.json_dict['repTrick']
            choice_dropLog = self.json_dict['dropLog']
        else:
            choice_samples_per_query = [5]
            choice_ad_training_order = ['DG']  # GD for irganlist DG for point/pair
            choice_temperature = [0.5] if self.debug else [0.5]  # 0.5, 1.0
            choice_d_g_epoch = [(1, 1)] if self.debug else [(1, 1)]  # discriminator-epoches vs. generator-epoches

            # settings that are specific to a listwise method#
            choice_top_k = [5]
            choice_PL_D = [True]  # discriminator formulation
            choice_repTrick = [False]  # for generator
            choice_dropLog = [True]  # drop log of discriminator when optimise generator

        for d_g_epoches, samples_per_query, ad_training_order, temperature in product(choice_d_g_epoch,
                                            choice_samples_per_query, choice_ad_training_order, choice_temperature):
            d_epoches, g_epoches = d_g_epoches
            self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                     samples_per_query=samples_per_query, temperature=temperature,
                                     ad_training_order=ad_training_order)

            for top_k, PL_D, repTrick, dropLog in product(choice_top_k, choice_PL_D, choice_repTrick, choice_dropLog):
                ad_list_para_dict = dict()
                ad_list_para_dict['top_k'] = top_k
                ad_list_para_dict['PL_D'] = PL_D
                ad_list_para_dict['repTrick'] = repTrick
                ad_list_para_dict['dropLog'] = dropLog
                self.ad_para_dict.update(ad_list_para_dict)

                yield self.ad_para_dict
