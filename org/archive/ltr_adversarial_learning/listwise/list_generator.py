#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/11/04 | https://y-research.github.io

"""Description

"""

from org.archive.ltr_adversarial_learning.base.ad_player import AdversarialPlayer

class List_Generator(AdversarialPlayer):
    '''
    A listwise generator
    '''
    def __init__(self, sf_para_dict=None, opt='RMS'):
        super(List_Generator, self).__init__(sf_para_dict=sf_para_dict, opt=opt)