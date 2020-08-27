#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

from ptranking.ltr_adversarial.base.ad_player import AdversarialPlayer

class List_Discriminator(AdversarialPlayer):
    '''
    A listwise discriminator
    '''
    def __init__(self, sf_para_dict=None, opt='RMS'):
        super(List_Discriminator, self).__init__(sf_para_dict=sf_para_dict, opt=opt)
