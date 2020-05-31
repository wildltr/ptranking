#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 19/04/03 | https://y-research.github.io

"""Description

"""

from itertools import product

from org.archive.base.base_utils import get_sf_str

##

def eval_grid(choice_validation, choice_epoch, choice_semi_context, choice_mask_ratios, choice_mask_type):
    ''' get the iterator w.r.t. evaluation settings '''

    for vd, num_epochs, semi_context, mask_type in product(choice_validation, choice_epoch, choice_semi_context, choice_mask_type):
        if semi_context:
            for mask_ratio in choice_mask_ratios:
                yield dict(do_vali=vd, epochs=num_epochs, semi_context=semi_context, mask_ratio=mask_ratio, mask_type=mask_type)
        else:
            yield dict(do_vali=vd, epochs=num_epochs, semi_context=semi_context)

### scoring function ###

def sf_grid(query_aware=None, FBN=False,
            choice_apply_BN=None, choice_apply_RD=None,
            choice_layers=None, choice_hd_hn_af=None, choice_tl_af=None, choice_hd_hn_tl_af=None, choice_apply_tl_af=None,
            in_choice_layers=None, in_choice_hd_hn_af=None, in_choice_tl_af=None,
            cnt_choice_layers=None, cnt_choice_hd_hn_af=None, cnt_choice_tl_af=None,
            com_choice_layers=None, com_choice_hd_hn_af=None, com_choice_tl_af=None,
            choice_cnt_strs=None):
    ''' get the iterator w.r.t. scoring function '''

    if query_aware:
        for in_num_layers, in_hd_hn_af, in_tl_af in product(in_choice_layers, in_choice_hd_hn_af, in_choice_tl_af):
            in_para_dict = dict(FBN=FBN, num_layers=in_num_layers, HD_AF=in_hd_hn_af, HN_AF=in_hd_hn_af, TL_AF=in_tl_af, apply_tl_af=True)

            for cnt_num_layers, cnt_hd_hn_af, cnt_tl_af in product(cnt_choice_layers, cnt_choice_hd_hn_af, cnt_choice_tl_af):
                cnt_para_dict = dict(num_layers=cnt_num_layers, HD_AF=cnt_hd_hn_af, HN_AF=cnt_hd_hn_af, TL_AF=cnt_tl_af, apply_tl_af=True)

                for com_num_layers, com_hd_hn_af, com_tl_af in product(com_choice_layers, com_choice_hd_hn_af, com_choice_tl_af):
                    com_para_dict = dict(num_layers=com_num_layers, HD_AF=com_hd_hn_af, HN_AF=com_hd_hn_af, TL_AF=com_tl_af, apply_tl_af=True)

                    for cnt_str in choice_cnt_strs:
                        sf_para_dict = dict()
                        sf_para_dict['id']      = 'ScoringFunction_CAFFNNs'
                        sf_para_dict['cnt_str'] = cnt_str
                        sf_para_dict['in_para_dict'] = in_para_dict
                        sf_para_dict['cnt_para_dict'] = cnt_para_dict
                        sf_para_dict['com_para_dict'] = com_para_dict

                        yield sf_para_dict
    else:
        if choice_hd_hn_tl_af is not None:
            for BN, RD, num_layers, af, apply_tl_af in product(choice_apply_BN, choice_apply_RD, choice_layers, choice_hd_hn_tl_af, choice_apply_tl_af):
                ffnns_para_dict = dict(FBN=FBN, BN=BN, RD=RD, num_layers=num_layers, HD_AF=af, HN_AF=af, TL_AF=af, apply_tl_af=apply_tl_af)
                sf_para_dict = dict()
                sf_para_dict['id'] = 'ffnns'
                sf_para_dict['ffnns'] = ffnns_para_dict

                yield sf_para_dict
        else:
            for BN, RD, num_layers, hd_hn_af, tl_af, apply_tl_af in product(choice_apply_BN, choice_apply_RD, choice_layers, choice_hd_hn_af, choice_tl_af, choice_apply_tl_af):
                ffnns_para_dict = dict(FBN=FBN, BN=BN, RD=RD, num_layers=num_layers, HD_AF=hd_hn_af, HN_AF=hd_hn_af, TL_AF=tl_af, apply_tl_af=apply_tl_af)
                sf_para_dict = dict()
                sf_para_dict['id'] = 'ffnns'
                sf_para_dict['ffnns'] = ffnns_para_dict

                yield sf_para_dict


### identifier ###

def get_sf_ID(sf_para_dict=None, log=False):
    ''' Get the identifier of scoring function '''
    s1, s2 = (':', '\n') if log else ('_', '_')

    if 'ScoringFunction_CAFFNNs' == sf_para_dict['id']:
        in_para_dict, cnt_para_dict, com_para_dict = sf_para_dict['in_para_dict'], sf_para_dict['cnt_para_dict'], sf_para_dict['com_para_dict']

        in_num_layers, in_HD_AF, in_HN_AF, in_TL_AF, in_BN, in_RD, in_FBN  = in_para_dict['num_layers'],\
                                                                             in_para_dict['HD_AF'], in_para_dict['HN_AF'], in_para_dict['TL_AF'],\
                                                                             in_para_dict['BN'], in_para_dict['RD'], in_para_dict['FBN']
        if not in_para_dict['apply_tl_af']:
            in_TL_AF = 'No'

        if cnt_para_dict is not None:
            cnt_num_layers, cnt_HD_AF, cnt_HN_AF, cnt_TL_AF, cnt_BN, cnt_RD  = cnt_para_dict['num_layers'],\
                                                                               cnt_para_dict['HD_AF'], cnt_para_dict['HN_AF'], cnt_para_dict['TL_AF'],\
                                                                               cnt_para_dict['BN'], cnt_para_dict['RD']
            if not cnt_para_dict['apply_tl_af']:
                cnt_TL_AF = 'No'

        com_num_layers, com_HD_AF, com_HN_AF, com_TL_AF, com_BN, com_RD  = com_para_dict['num_layers'],\
                                                                           com_para_dict['HD_AF'], com_para_dict['HN_AF'], com_para_dict['TL_AF'],\
                                                                           com_para_dict['BN'], com_para_dict['RD']
        if not com_para_dict['apply_tl_af']:
            com_TL_AF = 'No'

        if log:
            in_rf_str = s2.join([s1.join(['FeatureBN', str(in_FBN)]),
                                 s1.join(['BN', str(in_BN)]),
                              s1.join(['num_layers', str(in_num_layers)]),
                              s1.join(['RD', str(in_RD)]),
                              s1.join(['HD_AF', in_HD_AF]), s1.join(['HN_AF', in_HN_AF]), s1.join(['TL_AF', in_TL_AF])])

            if cnt_para_dict is not None:
                cnt_rf_str = s2.join([s1.join(['BN', str(cnt_BN)]),
                                  s1.join(['num_layers', str(cnt_num_layers)]),
                                  s1.join(['RD', str(cnt_RD)]),
                                  s1.join(['HD_AF', cnt_HD_AF]), s1.join(['HN_AF', cnt_HN_AF]), s1.join(['TL_AF', cnt_TL_AF])])

            com_rf_str = s2.join([s1.join(['BN', str(com_BN)]),
                              s1.join(['num_layers', str(com_num_layers)]),
                              s1.join(['RD', str(com_RD)]),
                              s1.join(['HD_AF', com_HD_AF]), s1.join(['HN_AF', com_HN_AF]), s1.join(['TL_AF', com_TL_AF])])
        else:
            in_rf_str = get_sf_str(in_num_layers, in_HD_AF, in_HN_AF, in_TL_AF)
            in_rf_str = s1.join([in_rf_str, 'BN', str(in_BN), 'RD', str(in_RD), 'FBN', str(in_FBN)])

            if cnt_para_dict is not None:
                cnt_rf_str = get_sf_str(cnt_num_layers, cnt_HD_AF, cnt_HN_AF, cnt_TL_AF)
                cnt_rf_str = s1.join([cnt_rf_str, 'BN', str(cnt_BN), 'RD', str(cnt_RD)])

            com_rf_str = get_sf_str(com_num_layers, com_HD_AF, com_HN_AF, com_TL_AF)
            com_rf_str = s1.join([com_rf_str, 'BN', str(com_BN), 'RD', str(com_RD)])

        if cnt_para_dict is not None:
            rf_str = s2.join([in_rf_str, cnt_rf_str, com_rf_str])
        else:
            rf_str = s2.join([in_rf_str, com_rf_str])

    elif 'ScoringFunction_Horn' == sf_para_dict['id']:
        tau, sh_itr = sf_para_dict['tau'], sf_para_dict['sh_itr']
        gru, bidirect, gru_layers = sf_para_dict['gru'], sf_para_dict['bidirect'], sf_para_dict['gru_layers']

        if not log and gru:
            gru_str = s2.join(['GRU', str(bidirect), str(gru_layers)])
        else:
            gru_str = 'NotGru'

        rf_str = s2.join([s1.join(['tau', '{:,g}'.format(tau)]),
                          s1.join(['sh_itr', str(sh_itr)]),
                          s1.join(['gru', str(gru)]),
                          s1.join(['bidirect', str(bidirect)]),
                          s1.join(['gru_layers', str(gru_layers)])]) if log \
                            else s1.join(['Horn', 'Tau', '{:,g}'.format(tau), 'Sh', str(sh_itr), gru_str])
        return rf_str

    else:
        if sf_para_dict['id'] in ['ScoringFunction_MDNs', 'ScoringFunction_QMDNs']:
            nn_para_dict = sf_para_dict['mu_para_dict']
        else:
            nn_para_dict = sf_para_dict['ffnns']

        num_layers, HD_AF, HN_AF, TL_AF, BN, RD, FBN = nn_para_dict['num_layers'],  nn_para_dict['HD_AF'],\
                                                       nn_para_dict['HN_AF'], nn_para_dict['TL_AF'], \
                                                       nn_para_dict['BN'], nn_para_dict['RD'], nn_para_dict['FBN']

        if not nn_para_dict['apply_tl_af']: TL_AF = 'No'

        if log:
            rf_str = s2.join([s1.join(['FeatureBN', str(FBN)]), s1.join(['BN', str(BN)]),
                              s1.join(['num_layers', str(num_layers)]), s1.join(['RD', str(RD)]),
                              s1.join(['HD_AF', HD_AF]), s1.join(['HN_AF', HN_AF]), s1.join(['TL_AF', TL_AF])])
        else:
            rf_str = get_sf_str(num_layers, HD_AF, HN_AF, TL_AF)
            if BN:  rf_str += '_BN'
            if RD:  rf_str += '_RD'
            if FBN: rf_str += '_FBN'

    return rf_str