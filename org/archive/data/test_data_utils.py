#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 19/05/14 | https://y-research.github.io

"""Description

"""
import numpy as np

from operator import itemgetter

from org.archive.data.data_utils import L2RDataLoader, L2RDataset, YAHOO_L2R, MSLRWEB, MSLETOR_SEMI, MSLETOR, prepare_data_for_ranklib, partition, convert_yahoo_into_5folds

def test_loading():
    # 1
    #file = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/Fold1/train.txt'
    # file = '/home/dl-box/WorkBench/Datasets/L2R/MSLR-WEB10K/Fold1/train.txt'
    train = True

    file = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_S/test.txt'
    #file = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/tmp'
    train = False

    # ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
    data_dict = dict(data_id='Istella', binary_rele=False,
                     min_docs=50, min_rele=1, presort=False,
                     scale_data=False, scaler_id='MinMaxScaler', scaler_level='QUERY')

    ms = L2RDataLoader(train=train, file=file, data_dict=data_dict)
    train_Qs = ms.load_data()

    '''
    list_qids = ['113', '1648', '4126', '987']
    # list_qids = ['151', '166', "29827"]
    for Q in train_Qs:
        qid, doc_reprs, doc_labels = Q
        if qid in list_qids:
            print('qid:\t', qid)
            print('docs:\t', len(doc_labels))
            print(np.unique(doc_labels, return_counts=True))
            print()
    '''

    '''
    for tp in ms.iterate_per_query():
        if tp[0] == '10':
            print(tp[2])
            print(tp[1])
    '''

    #
    sorted_train_Qs = sorted(train_Qs, key=itemgetter(0), reverse=False)
    print(len(sorted_train_Qs))
    for Q in sorted_train_Qs:
        qid, doc_reprs, doc_labels = Q
        print(qid)
        print('\t', doc_labels.shape)
        print('\t', doc_reprs.shape)

        if '669' == qid:
            print('\t', doc_labels)
            print('\t', doc_reprs)


def test_loading_2():
    # file = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/Fold1/train.txt'
    # file = '/home/dl-box/WorkBench/Datasets/L2R/MSLR-WEB10K/Fold1/train.txt'
    file = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_2/set2.train.txt'

    # ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
    '''
    data_dict = dict(data_id='MQ2007_Super', binary_rele=False,
                     min_docs=50, min_rele=1,
                     scale_data=True, scaler_id='MinMaxScaler', scaler_level='QUERY')
    '''

    data_dict = dict(data_id='Set2', binary_rele=False,
                     min_docs=1, min_rele=0,
                     scale_data=False)

    dataset = L2RDataset(file, sample_rankings_per_q=1, shuffle=True, data_dict=data_dict, hot=False)

    print('query size:', dataset.__len__())
    doc_cnt = 0
    for qid, torch_batch_rankings, torch_batch_std_labels in dataset:
        print(qid)
        #print(torch_batch_std_labels)
        #print(torch_batch_rankings)
        doc_cnt += torch_batch_rankings.size(1)

        if qid == '29922':
            print(torch_batch_std_labels)
            np_reprs = torch_batch_rankings.data.numpy()
            print(np_reprs[0, 0, :])
            print()
            print(np_reprs[0, 1, :])
            print()
            print(np_reprs[0, 2, :])

    print('docs:', doc_cnt)

    """
    # hot iteration
    for qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts in dataset.iter_hot():
        qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts = entry
        print(qid)
        print(batch_cnts)
        print(torch_batch_std_labels)
        print(torch_batch_std_hot_labels)
    """


import collections

def enrich(x, num_features=None):
    dict_feats = collections.OrderedDict([(str(i), ':'.join([str(i), '0'])) for i in range(1, num_features+1)])

    feats = x.strip().split(" ")
    for feat in feats:
        dict_feats[feat.split(':')[0]] = feat

    enriched_feats = list(dict_feats.values())
    return ' '.join(enriched_feats)

import pandas as pd

def test_y_loader():
    file = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_2/set2.train.txt'
    df = pd.read_csv(file, names=['rele_truth']) # the column of rele_truth will be updated

    cols = ['rele_truth', 'qid', 'features']
    df[cols] = df.rele_truth.str.split(' ', n=2, expand=True) # split


    print(df.columns)
    print(df.loc[0])
    print()

    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: enrich(x, num_features=700))

    print(df.loc[0])
    print()

    feature_cols = [str(f_index) for f_index in range(1, 700 + 1)]
    df[feature_cols] = df.features.str.split(' ', n=700, expand=True)  # split
    df.drop(columns=['features'], inplace=True)  # remove the feature string column

    print(df.columns)
    print(df.loc[0])
    print()

    for c in range(1, 700 + 2):  # remove keys per column from key:value
        df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

    print(df.columns)
    print(df.loc[0])
    print()

from org.archive.data.data_utils import load_data_xgboost
def test_xgboost():
    ori_file_train = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/Fold1/train.txt'

    file_train_data, file_train_group = load_data_xgboost(ori_file_train, min_docs=10, min_rele=1, data_id='MQ2007_super')


#######################
def get_doc_num(dataset):
    doc_num = 0
    for qid, torch_batch_rankings, torch_batch_std_labels in dataset:
        doc_num += torch_batch_std_labels.size(1)

    return doc_num


def get_min_max_docs(train_dataset, vali_dataset, test_dataset):
    min_doc = 10000000
    max_doc = 0
    sum_rele = 0

    for qid, torch_batch_rankings, torch_batch_std_labels in train_dataset:
        #print('torch_batch_std_labels', torch_batch_std_labels.size())
        doc_num = torch_batch_std_labels.size(1)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (torch_batch_std_labels>0).sum()

    for qid, torch_batch_rankings, torch_batch_std_labels in vali_dataset:
        doc_num = torch_batch_std_labels.size(1)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (torch_batch_std_labels>0).sum()

    for qid, torch_batch_rankings, torch_batch_std_labels in test_dataset:
        doc_num = torch_batch_std_labels.size(1)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (torch_batch_std_labels>0).sum()

    return min_doc, max_doc, sum_rele.data.numpy()


from org.archive.ltr_adhoc.eval.test_l2r_tao import get_in_out_dir

def check_dataset_statistics(data_id, filtering_dumb_queries=False):
    '''

    '''

    if filtering_dumb_queries:
        min_docs, min_rele, presort = 1, 1, False
    else:
        min_docs, min_rele, presort = None, None, False

    dir_data, dir_output = get_in_out_dir(data_id=data_id, pc='mbox-f3')

    if data_id in YAHOO_L2R:
        data_prefix = dir_data + data_id.lower() + '.'
        file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'
        data_dict = dict(data_id=data_id, binary_rele=False, max_docs='All', min_docs=min_docs, min_rele=min_rele, scale_data=False, presort=presort)
    else:
        fold_k = 1
        fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        data_dict = dict(data_id=data_id, binary_rele=False,
                         max_docs='All', min_docs=min_docs, min_rele=min_rele,
                         scale_data=False, scaler_id='StandardScaler', scaler_level='QUERY', presort=presort)

        if data_id in MSLETOR_SEMI:
            if filtering_dumb_queries:
                data_dict.update(dict(unknown_as_zero=True))
                data_dict.update(dict(binary_rele=True))
            else:
                data_dict.update(dict(unknown_as_zero=False))
                data_dict.update(dict(binary_rele=False))

    # common
    train_dataset = L2RDataset(file_train, sample_rankings_per_q=1, shuffle=False, data_dict=data_dict, hot=False)
    vali_dataset = L2RDataset(file_vali, sample_rankings_per_q=1, shuffle=False, data_dict=data_dict, hot=False)
    test_dataset = L2RDataset(file_test, sample_rankings_per_q=1, shuffle=False, data_dict=data_dict, hot=False)

    num_queries = train_dataset.__len__() + vali_dataset.__len__() + test_dataset.__len__()
    print('Total queries:\t', num_queries)
    print(train_dataset.__len__(), vali_dataset.__len__(), test_dataset.__len__())

    print('Test', test_dataset.__len__())


    num_docs = get_doc_num(train_dataset) + get_doc_num(vali_dataset) + get_doc_num(test_dataset)
    print('Total docs:\t', num_docs)

    min_doc, max_doc, sum_rele = get_min_max_docs(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset)
    print('min, max documents per query', min_doc, max_doc)
    print('total relevant documents', sum_rele)
    print('avg rele documents per query', sum_rele * 1.0 / num_queries)
    print('avg documents per query', num_docs * 1.0 / num_queries)


def norm_ms_data_ranklib(data_id):
    dir_data, dir_output = get_in_out_dir(data_id=data_id, pc='mbox-f3')

    assert data_id in MSLRWEB
    fold_num = 5
    for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
        fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        prepare_data_for_ranklib(file_train, min_docs=10, min_rele=1, data_id=data_id)
        prepare_data_for_ranklib(file_vali, min_docs=10, min_rele=1, data_id=data_id)
        prepare_data_for_ranklib(file_test, min_docs=10, min_rele=1, data_id=data_id)

import random

def test_p():
    p = np.random.permutation(12)

    print(p)
    random.shuffle(p)
    print(p)

    lst = p.tolist()
    rest = partition(lst=lst, n=5)
    print(rest)

#from org.archive.data.data_utils import mask_data

def check_masking(data_id, filtering_dumb_queries=False):
    if filtering_dumb_queries:
        min_docs, min_rele, presort = 1, 1, True
    else:
        min_docs, min_rele, presort = None, None, True

    dir_data, dir_output = get_in_out_dir(data_id=data_id, pc='mbox-f3')

    if data_id in YAHOO_L2R:
        data_prefix = dir_data + data_id.lower() + '.'
        file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'
        data_dict = dict(data_id=data_id, binary_rele=False, max_docs='All', min_docs=min_docs, min_rele=min_rele, scale_data=False, presort=presort)
    else:
        fold_k = 1
        fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        data_dict = dict(data_id=data_id, binary_rele=False,
                         max_docs='All', min_docs=min_docs, min_rele=min_rele,
                         scale_data=False, scaler_id='StandardScaler', scaler_level='QUERY', presort=presort)

        if data_id in MSLETOR_SEMI:
            if filtering_dumb_queries:
                data_dict.update(dict(unknown_as_zero=True))
                data_dict.update(dict(binary_rele=True))
            else:
                data_dict.update(dict(unknown_as_zero=False))
                data_dict.update(dict(binary_rele=False))

    # common
    train_dataset = L2RDataset(train=True, file=file_train, sample_rankings_per_q=1, shuffle=False, data_dict=data_dict, hot=False)

    mask_data(mask_ratio=0.4, train_data=train_dataset, presort=presort)

import torch
def check_gt_gather():
    mat = torch.rand(size=(2, 5))
    print('mat', mat)


    res1 = mat > 0.5
    print('res1', res1)

    res2 = res1.nonzero()
    print('res2', res2)

torch_zero = torch.FloatTensor([0.0])
def check_mask_rele(mask_ratio=0.4):
    mat = torch.randint(size=(1, 20), low=-2, high=3)

    mat = torch.squeeze(mat,dim=0)
    print('mat', mat.size(), mat)

    all_rele_inds = torch.gt(mat, torch_zero).nonzero()
    print('all_rele_inds', all_rele_inds.size(), all_rele_inds)
    num_rele = all_rele_inds.size()[0]
    print('num_rele', num_rele)

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)
    print('mask_inds', mask_inds)

    rele_inds_to_mask = all_rele_inds[mask_inds, 0]
    print('rele_inds_to_mask', rele_inds_to_mask)


def check_np_mask_rele(mask_ratio=0.4):

    mat = np.random.randint(low=-2, high=3, size=20)
    print('mat', mat.shape, mat)

    all_rele_inds = np.greater(mat, 0).nonzero()[0] # due to one-dimension
    print('all_rele_inds', all_rele_inds)
    num_rele = all_rele_inds.shape[0]
    print('num_rele', num_rele)

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)
    print('mask_inds', mask_inds)

    rele_inds_to_mask = all_rele_inds[mask_inds]
    print('rele_inds_to_mask', rele_inds_to_mask)



def check_file():
    #file = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/Istella_S/test.txt'
    #file = '/home/dl-box/ダウンロード/sample/test.txt'
    file = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/tmp'

    i = 0
    with open(file=file) as reader:
        for line in reader.readlines():
            print(line)
            #arrs = line.strip().split(' ')
            arrs = line.split(' ')
            print(arrs)
            print(len(arrs))
            i+=1

            if i>10:
                break




if __name__ == '__main__':
    #
    test_loading()

    #check_file()

    #2
    #test_loading_2()

    #3
    #test_y_loader()

    #4
    #test_xgboost()

    #5
    #check_dataset_statistics(data_id='MQ2008_Semi', filtering_dumb_queries=False)

    #check_dataset_statistics(data_id='5FoldSet2')

    '''
    #### MQ2007_Super
    ----no filtering
    Total queries:	 1692
    Total docs:	 69623
    min, max documents per query 6 147
    total relevant documents 17991
    avg rele documents per query 10.632978723404255
    avg documents per query 41.1483451536643
    
    ----filtering-dumb
    Total queries:	 1455
    Total docs:	 59570
    min, max documents per query 6 147
    total relevant documents 17991
    avg rele documents per query 12.364948453608248
    avg documents per query 40.94158075601374
    
    #### MSLRWEB30K
    ----no filtering
    Total queries:	 31531
    Total docs:	 3771125
    min, max documents per query 1 1251
    total relevant documents 1830173
    avg rele documents per query 58.04360787796137
    avg documents per query 119.60055183787384

    ----filtering-dumb
    Total queries:	 30295
    Total docs:	 3749144
    min, max documents per query 10 1251
    total relevant documents 1829610
    avg rele documents per query 60.39313418055785
    avg documents per query 123.75454695494307
    
    #Set1
    Total queries:	 21713
    Total docs:	 652096
    min, max documents per query 10 139
    total relevant documents 485569
    avg rele documents per query 22.363054391378437
    avg documents per query 30.032515083129923
    
    #####   MQ2008_Semi     ####
    
    ----filtering-dumb
    check_dataset_statistics(data_id='MQ2008_Semi')
    
    Total queries:	 564
    339 120 105
    Test 105
    Total docs:	 386244
    min, max documents per query 6 1827
    total relevant documents 2932
    avg rele documents per query 5.198581560283688
    avg documents per query 684.8297872340426
    
    ----no filtering
    check_dataset_statistics(data_id='MQ2008_Semi')
    Total queries:	 784
    471 157 156
    Test 156
    Total docs:	 546260
    min, max documents per query 6 1831
    total relevant documents 2932
    avg rele documents per query 3.739795918367347
    avg documents per query 696.7602040816327
    
    '''

    # ranklib
    #norm_ms_data_ranklib(data_id='MSLRWEB30K')

    #test_p()

    #convert_yahoo_into_5folds(data_id='Set1')

    #check_masking(data_id='MQ2007_Super')

    #check_gt_gather()

    #check_mask_rele()

    #check_np_mask_rele()

