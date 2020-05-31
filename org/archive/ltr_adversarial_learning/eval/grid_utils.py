#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/02/28 | https://y-research.github.io

"""Description

"""

from itertools import product

def ad_list_grid(model_id, choice_top_k=None, choice_shuffle_ties=None, choice_PL=None, choice_repTrick=None, choice_dropLog=None):
    ''' get the iterator w.r.t. listwise models '''

    if model_id.startswith('IR_GAN'):
        for top_k, shuffle_ties, PL, repTrick, dropLog in product(choice_top_k, choice_shuffle_ties, choice_PL, choice_repTrick, choice_dropLog):
            ad_list_para_dict = dict()
            ad_list_para_dict['top_k'] = top_k
            ad_list_para_dict['shuffle_ties'] = shuffle_ties
            ad_list_para_dict['PL'] = PL
            ad_list_para_dict['repTrick'] = repTrick
            ad_list_para_dict['dropLog'] = dropLog

            yield ad_list_para_dict

    elif model_id.startswith('IR_WGAN'):
        for top_k, shuffle_ties, PL in product(choice_top_k, choice_shuffle_ties, choice_PL):
            ad_list_para_dict = dict()
            ad_list_para_dict['top_k'] = top_k
            ad_list_para_dict['shuffle_ties'] = shuffle_ties
            ad_list_para_dict['PL'] = PL

            yield ad_list_para_dict

    else:
        raise NotImplementedError


def ad_eval_grid(choice_validation, choice_epoch, choice_semi_context, choice_mask_ratios, choice_mask_type):
    ''' get the iterator w.r.t. evaluation settings '''
    for vd, num_epochs, semi_context in product(choice_validation, choice_epoch, choice_semi_context):
        if semi_context:
            for mask_ratio, mask_type in product(choice_mask_ratios, choice_mask_type):
                yield dict(do_vali=vd, epochs=num_epochs, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)
        else:
            yield dict(do_vali=vd, epochs=num_epochs, semi_context=semi_context)