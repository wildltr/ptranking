#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import torch

def batch_count(batch_std_labels=None, max_rele_grade=None, descending=False, gpu=False):
    """
    Todo now an api is already provided by pytorch
    :param batch_std_labels:
    :param max_rele_grade:
    :param descending:
    :param gpu:
    :return:
    """
    rele_grades = torch.arange(max_rele_grade+1).type(torch.cuda.FloatTensor) if gpu else torch.arange(max_rele_grade+1).type(torch.FloatTensor)
    if descending: rele_grades, _ = torch.sort(rele_grades, descending=True)

    batch_cnts = torch.stack([(batch_std_labels == g).sum(dim=1) for g in rele_grades])
    batch_cnts = torch.t(batch_cnts)
    return batch_cnts
