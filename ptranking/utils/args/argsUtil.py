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
        self.args_parser.add_argument('-cuda', type=int, help='specify the gpu id if needed, such as 0 or 1.', default=None)

        ''' model '''
        self.args_parser.add_argument('-model', type=str, help='specify the learning-to-rank method')

        ''' debug '''
        self.args_parser.add_argument("-debug", action="store_true", help="quickly check the setting in a debug mode")

        ''' path of json files specifying the evaluation details '''
        self.args_parser.add_argument('-dir_json', type=str, help='the path of json files specifying the evaluation details.')

    def update_if_required(self, args):
        return args

    def get_l2r_args(self):
        l2r_args = self.args_parser.parse_args()
        l2r_args = self.update_if_required(l2r_args)
        return l2r_args

