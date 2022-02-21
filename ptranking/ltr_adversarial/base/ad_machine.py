#!/usr/bin/env python
# -*- coding: utf-8 -*-


class AdversarialMachine():
    '''
    An abstract adversarial learning-to-rank framework
    '''

    def __init__(self, eval_dict=None, data_dict=None, gpu=False, device=None):
        #todo double-check the necessity of these two arguments
        self.eval_dict = eval_dict
        self.data_dict = data_dict
        self.gpu, self.device = gpu, device

    def pre_check(self):
        pass

    def burn_in(self, train_data=None):
        pass

    def mini_max_train(self, train_data=None, generator=None, discriminator=None, d_epoches=1, g_epoches=1, global_buffer=None):
        pass

    def fill_global_buffer(self):
        '''
        Buffer some global information for higher efficiency.
        We note that global information may differ w.r.t. the model
        '''
        pass

    def generate_data(self, train_data=None, generator=None, **kwargs):
        pass

    def train_generator(self, train_data=None, generated_data=None, generator=None, discriminator=None, **kwargs):
        pass

    def train_discriminator(self, train_data=None, generated_data=None, discriminator=None, **kwargs):
        pass

    def reset_generator(self):
        pass

    def reset_discriminator(self):
        pass

    def reset_generator_discriminator(self):
        self.reset_generator()
        self.reset_discriminator()

    def get_generator(self):
        return None

    def get_discriminator(self):
        return None
