#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""


from org.archive.ltr_adversarial.base.ad_player import AdversarialPlayer

class Point_Generator(AdversarialPlayer):
    '''
    A pointwise generator
    '''
    def __init__(self, sf_para_dict=None):
        super(Point_Generator, self).__init__(sf_para_dict=sf_para_dict)
