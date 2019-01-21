#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/12/03 | https://y-research.github.io

"""Description

"""


from org.archive.utils.args.argsUtil import ArgsUtil
from org.archive.ranking.run.l2r import point_run

"""
Example command line usage:

python ptl2r.py -data MQ2008_super -dir_data /home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/ -dir_output /home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/ -model ListMLE

"""

if __name__ == '__main__':
    print('started ptl2r ...')

    args_obj = ArgsUtil(given_root='./')
    l2r_args = args_obj.get_l2r_args()

    if 'L2R' == l2r_args.framework:

        point_run(model=l2r_args.model, data=l2r_args.data, dir_data=l2r_args.dir_data, dir_output=l2r_args.dir_output, binary_rele=l2r_args.binary)

    else:
        raise NotImplementedError

