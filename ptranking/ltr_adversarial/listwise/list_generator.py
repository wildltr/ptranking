#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

from ptranking.ltr_adversarial.base.ad_player import AdversarialPlayer

class List_Generator(AdversarialPlayer):
    '''
    A listwise generator
    '''
    def __init__(self, sf_para_dict=None, opt='RMS', gpu=False, device=None):
        super(List_Generator, self).__init__(sf_para_dict=sf_para_dict, opt=opt, gpu=gpu, device=device)
