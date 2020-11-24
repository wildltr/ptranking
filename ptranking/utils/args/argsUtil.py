#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
Class for parsing command-line input
"""

import argparse

class ArgsUtil(object):
    def __init__(self, given_root=None):
        self.args_parser = argparse.ArgumentParser('Run pt_ranking.')
        self.ini_l2r_args(given_root=given_root)

    def ini_l2r_args(self, given_root=None):
        self.given_root = given_root

        ''' gpu '''
        self.args_parser.add_argument('-cuda', type=int, help='gpu id', default=None)

        ''' data '''
        self.args_parser.add_argument('-data_id', type=str, help='the data collection upon which you will perform learning-to-rank.')
        self.args_parser.add_argument('-dir_data', type=str, help='the path where the data locates.')

        ''' output '''
        self.args_parser.add_argument('-dir_output', type=str, help='the output path of the results.')

        ''' model '''
        self.args_parser.add_argument('-model', type=str, help='specify the learning-to-rank method')

        self.args_parser.add_argument('-engine', type=str, help='specify the driving engine for LambdaMART, either XGBoost or LightGBM')

    def update_if_required(self, args):
        return args

    def get_l2r_args(self):
        l2r_args = self.args_parser.parse_args()
        l2r_args = self.update_if_required(l2r_args)
        return l2r_args

