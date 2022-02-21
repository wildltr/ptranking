#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from itertools import product

import torch
import torch.nn.functional as F

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adversarial.base.ad_player import AdversarialPlayer
from ptranking.ltr_adversarial.base.ad_machine import AdversarialMachine

class IRGAN_Pair_Generator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None, temperature=None, gpu=False, device=None):
        super(IRGAN_Pair_Generator, self).__init__(sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.temperature = temperature

    def predict(self, batch_ranking):
        batch_pred = self.forward(batch_ranking)

        if self.temperature is not None and 1.0 != self.temperature:
                batch_pred = batch_pred / self.temperature

        return batch_pred


class IRGAN_Pair_Discriminator(AdversarialPlayer):
    def __init__(self, sf_para_dict=None, gpu=False, device=None):
        super(IRGAN_Pair_Discriminator, self).__init__(sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.torch_one = torch.tensor([1.0], device=self.device)
        self.torch_zero = torch.tensor([0.0], device=self.device)

    def get_reward(self, pos_docs=None, neg_docs=None, loss_type='svm'):
        ''' used by irgan '''
        batch_pos_preds = self.predict(pos_docs)  # [batch, size_ranking]
        batch_neg_preds = self.predict(neg_docs)  # [batch, size_ranking]

        if 'svm' == loss_type:
            reward = torch.sigmoid(torch.max(self.torch_zero, self.torch_one - (batch_pos_preds - batch_neg_preds)))
        elif 'log' == loss_type:
            reward = torch.log(torch.sigmoid(batch_neg_preds - batch_pos_preds))
        else:
            raise NotImplementedError

        return reward



class IRGAN_Pair(AdversarialMachine):
    ''''''
    def __init__(self, eval_dict, data_dict, sf_para_dict=None, ad_para_dict=None, gpu=False, device=None):
        '''
        :param sf_para_dict:
        :param temperature: according to the description around Eq-10, temperature is deployed, while it is not used within the provided code
        '''
        super(IRGAN_Pair, self).__init__(eval_dict=eval_dict, data_dict=data_dict, gpu=gpu, device=device)

        self.torch_one = torch.tensor([1.0], device=self.device)
        self.torch_zero = torch.tensor([0.0], device=self.device)

        sf_para_dict[sf_para_dict['sf_id']]['apply_tl_af'] = True

        g_sf_para_dict = sf_para_dict

        d_sf_para_dict = copy.deepcopy(g_sf_para_dict)
        d_sf_para_dict[sf_para_dict['sf_id']]['apply_tl_af'] = False
        #d_sf_para_dict['ffnns']['TL_AF'] = 'S'  # as required by the IRGAN model

        self.generator     = IRGAN_Pair_Generator(sf_para_dict=g_sf_para_dict, temperature=ad_para_dict['temperature'], gpu=gpu, device=device)
        self.discriminator = IRGAN_Pair_Discriminator(sf_para_dict=d_sf_para_dict, gpu=gpu, device=device)

        self.loss_type = ad_para_dict['loss_type']
        self.d_epoches = ad_para_dict['d_epoches']
        self.g_epoches = ad_para_dict['g_epoches']
        self.temperature = ad_para_dict['temperature']
        self.ad_training_order = ad_para_dict['ad_training_order']
        self.samples_per_query = ad_para_dict['samples_per_query']

    def fill_global_buffer(self, train_data, dict_buffer=None):
        ''' Buffer the number of positive documents, and the number of non-positive documents per query '''
        assert self.data_dict['train_presort'] is True  # this is required for efficient truth exampling

        for entry in train_data:
            qid, _, batch_label = entry[0], entry[1], entry[2]
            if not qid in dict_buffer:
                boolean_mat = torch.gt(batch_label, 0)
                num_pos = torch.sum(boolean_mat)
                ranking_size = batch_label.size(1)
                num_neg_unk = ranking_size - num_pos
                dict_buffer[qid] = (num_pos, num_neg_unk)


    def mini_max_train(self, train_data=None, generator=None, discriminator=None, global_buffer=None):
        if self.ad_training_order == 'DG':
            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator, global_buffer=global_buffer)

                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

            for g_epoch in range(self.g_epoches):
                self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator,
                                     global_buffer=global_buffer)  # train generator

        else:
            for g_epoch in range(self.g_epoches):
                self.train_generator(train_data=train_data, generator=generator, discriminator=discriminator,
                                     global_buffer=global_buffer)  # train generator

            for d_epoch in range(self.d_epoches):
                if d_epoch % 10 == 0:
                    generated_data = self.generate_data(train_data=train_data, generator=generator,
                                                        global_buffer=global_buffer)
                self.train_discriminator(train_data=train_data, generated_data=generated_data, discriminator=discriminator)  # train discriminator

        stop_training = False
        return stop_training


    def generate_data(self, train_data=None, generator=None, global_buffer=None):
        '''
        Sampling for training discriminator
        This is a re-implementation as the released irgan-tensorflow, but it seems that this part of irgan-tensorflow
        is not consistent with the discription of the paper (i.e., the description below Eq. 7)
        '''
        generator.eval_mode()

        generated_data = dict()
        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.to(self.device)
            samples = self.per_query_generation(qid=qid, batch_ranking=batch_ranking, generator=generator,
                                                global_buffer=global_buffer)
            if samples is not None: generated_data[qid] = samples

        return generated_data


    def per_query_generation(self, qid, batch_ranking, generator, global_buffer):
        num_pos, num_neg_unk = global_buffer[qid]
        valid_num = min(num_pos, num_neg_unk, self.samples_per_query)
        if num_pos >= 1 and valid_num >= 1:
            ranking_inds = torch.arange(batch_ranking.size(1))
            '''
            intersection implementation (keeping consistent with the released irgan-tensorflow):
            remove the positive part, then sample to form pairs
            '''
            pos_inds = torch.randperm(num_pos)[0:valid_num]  # randomly select positive documents

            #if self.gpu: batch_ranking = batch_ranking.to(self.device) # [batch, size_ranking]
            batch_pred = generator.predict(batch_ranking)  # [batch, size_ranking]
            pred_probs = F.softmax(torch.squeeze(batch_pred), dim=0)
            neg_unk_probs = pred_probs[num_pos:]
            # sample from negative / unlabelled documents
            inner_neg_inds = torch.multinomial(neg_unk_probs, valid_num, replacement=False)
            neg_inds = ranking_inds[num_pos:][inner_neg_inds]
            # todo with cuda, confirm the possible time issue on indices
            return (pos_inds, neg_inds)
        else:
            return None


    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        discriminator.train_mode()

        for entry in train_data:
            qid, batch_ranking = entry[0], entry[1]

            if qid in generated_data:
                if self.gpu: batch_ranking = batch_ranking.to(self.device)

                pos_inds, neg_inds = generated_data[qid]
                pos_docs = batch_ranking[:, pos_inds, :]
                neg_docs = batch_ranking[:, neg_inds, :]

                batch_pos_preds = discriminator.predict(pos_docs) # [batch, size_ranking]
                batch_neg_preds = discriminator.predict(neg_docs) # [batch, size_ranking]

                if 'svm' == self.loss_type:
                    #dis_loss = torch.mean(torch.max(torch.zeros(1), 1.0-(batch_pos_preds-batch_neg_preds)))
                    dis_loss = torch.mean(torch.max(self.torch_zero, self.torch_one - (batch_pos_preds - batch_neg_preds)))
                elif 'log' == self.loss_type:
                    dis_loss = -torch.mean(torch.log(torch.sigmoid(batch_pos_preds-batch_neg_preds)))
                else:
                    raise NotImplementedError

                discriminator.optimizer.zero_grad()
                dis_loss.backward()
                discriminator.optimizer.step()


    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None, global_buffer=None):
        generator.train_mode()
        discriminator.eval_mode()

        for entry in train_data:
            qid, batch_ranking, batch_label = entry[0], entry[1], entry[2]
            if self.gpu: batch_ranking = batch_ranking.to(self.device)

            num_pos, num_neg_unk = global_buffer[qid]
            valid_num = min(num_pos, num_neg_unk, self.samples_per_query)

            if num_pos < 1 or valid_num < 1:
                continue

            pos_inds = torch.randperm(num_pos)[0:valid_num]  # randomly select positive documents

            g_preds = generator.predict(batch_ranking)
            g_probs = torch.sigmoid(torch.squeeze(g_preds))
            neg_inds = torch.multinomial(g_probs, valid_num, replacement=False)

            pos_docs = batch_ranking[:, pos_inds, :]
            neg_docs = batch_ranking[:, neg_inds, :]

            reward = discriminator.get_reward(pos_docs=pos_docs, neg_docs=neg_docs, loss_type=self.loss_type)

            g_loss = -torch.mean((torch.log(g_probs[neg_inds]) * reward))

            generator.optimizer.zero_grad()
            g_loss.backward()
            generator.optimizer.step()

    def reset_generator(self):
        self.generator.init()

    def reset_discriminator(self):
        self.discriminator.init()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

###### Parameter of IRGAN_Pair ######

class IRGAN_PairParameter(ModelParameter):
    ''' Parameter class for IRGAN_Pair '''
    def __init__(self, debug=False, para_json=None):
        super(IRGAN_PairParameter, self).__init__(model_id='IRGAN_Pair', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for IRGAN_Pair
        :return:
        """
        temperature = 0.5
        d_epoches, g_epoches = 1, 1
        ad_training_order = 'DG' # 'GD'
        samples_per_query = 5

        self.ad_para_dict = dict(model_id=self.model_id, loss_type='svm', d_epoches=d_epoches, g_epoches=g_epoches,
                                 temperature=temperature, ad_training_order=ad_training_order,
                                 samples_per_query=samples_per_query)
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
        d_epoches, g_epoches, temperature, ad_training_order, loss_type, samples_per_query = \
            ad_para_dict['d_epoches'], ad_para_dict['g_epoches'], ad_para_dict['temperature'],\
            ad_para_dict['ad_training_order'], ad_para_dict['loss_type'], ad_para_dict['samples_per_query']

        pair_irgan_paras_str = s1.join([str(d_epoches), str(g_epoches), '{:,g}'.format(temperature),
                                        ad_training_order, loss_type, str(samples_per_query)])

        return pair_irgan_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for IRGAN_Pair
        """
        if self.use_json:
            d_g_epoch_strings = self.json_dict['d_g_epoch']
            choice_losstype_d = self.json_dict['losstype_d']
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
            choice_d_g_epoch = [(1, 1)] if self.debug else [(1, 1)]  # discriminator-epoches vs. generator-epoches
            choice_losstype_d = ['svm']

        for d_g_epoches, samples_per_query, ad_training_order, temperature, loss_type_d in product(choice_d_g_epoch,
                       choice_samples_per_query, choice_ad_training_order, choice_temperature, choice_losstype_d):
            d_epoches, g_epoches = d_g_epoches

            self.ad_para_dict = dict(model_id=self.model_id, d_epoches=d_epoches, g_epoches=g_epoches,
                                     samples_per_query=samples_per_query, temperature=temperature,
                                     ad_training_order=ad_training_order, loss_type=loss_type_d)

            yield self.ad_para_dict
