#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/04 | https://y-research.github.io

"""Description

"""

from org.archive.ltr_adversarial_learning.base.ad_player import AdversarialPlayer

class List_Discriminator(AdversarialPlayer):
    '''
    A listwise discriminator
    '''
    def __init__(self, sf_para_dict=None, opt='RMS'):
        super(List_Discriminator, self).__init__(sf_para_dict=sf_para_dict, opt=opt)