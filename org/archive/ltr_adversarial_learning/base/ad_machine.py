#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/02 | https://y-research.github.io

"""Description

"""

class AdversarialMachine():
    '''
    An abstract adversarial learning-to-rank framework
    '''

    def __init__(self, eval_dict=None, data_dict=None):
        self.eval_dict = eval_dict
        self.data_dict = data_dict

    def pre_check(self):
        pass

    def burn_in(self, train_data=None):
        pass

    def mini_max_train(self, train_data=None, generator=None, discriminator=None, d_epoches=1, g_epoches=1, dict_buffer=None):
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