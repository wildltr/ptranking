#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Examples on how to use data_utils module
"""

from org.archive.data.data_utils import L2RDataLoader, L2RDataset, YAHOO_L2R, ISTELLA_L2R, MSLRWEB, MSLETOR_SEMI, MSLETOR, prepare_data_for_ranklib, partition, convert_yahoo_into_5folds
import torch

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


def get_min_max_feature(train_dataset, vali_dataset, test_dataset):
    min_f = 0
    max_f = 1000
    for qid, torch_batch_rankings, torch_batch_std_labels in train_dataset:
        mav = torch.max(torch_batch_rankings)
        if torch.isinf(mav):
            print(qid, mav)
        else:
            if mav > max_f: max_f = mav

        miv = torch.min(torch_batch_rankings)
        if miv < min_f: min_f = miv
    print('train', min_f, '\t', max_f)

    min_f = 0
    max_f = 1000
    for qid, torch_batch_rankings, torch_batch_std_labels in vali_dataset:
        mav = torch.max(torch_batch_rankings)
        if mav > max_f: max_f = mav
        miv = torch.min(torch_batch_rankings)
        if miv < min_f: min_f = miv
    print('vali', min_f, '\t', max_f)

    min_f = 0
    max_f = 1000
    for qid, torch_batch_rankings, torch_batch_std_labels in test_dataset:
        mav = torch.max(torch_batch_rankings)
        if mav > max_f: max_f = mav
        miv = torch.min(torch_batch_rankings)
        if miv < min_f: min_f = miv
    print('test', min_f, '\t', max_f)



def check_dataset_statistics(data_id, dir_data, buffer=False):
    '''
    Get the basic statistics on the specified dataset
    '''

    if data_id in YAHOO_L2R:
        data_prefix = dir_data + data_id.lower() + '.'
        file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'

    elif data_id in ISTELLA_L2R:
        data_prefix = dir_data + data_id + '/'

        if data_id == 'Istella_X' or data_id=='Istella_S':
            file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'vali.txt', data_prefix + 'test.txt'
        else:
            file_train, file_test = data_prefix + 'train.txt', data_prefix + 'test.txt'

    else:
        fold_k = 1
        fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

    # common
    if 'Istella' == data_id:
        pass # since there is no vali part
    else:
        train_dataset = L2RDataset(train=True, file=file_train, data_id=data_id, shuffle=False, buffer=buffer)
        vali_dataset =  L2RDataset(train=False, file=file_vali, data_id=data_id, shuffle=False, buffer=buffer)
        test_dataset =  L2RDataset(train=False, file=file_test, data_id=data_id, shuffle=False, buffer=buffer)

        num_queries = train_dataset.__len__() + vali_dataset.__len__() + test_dataset.__len__()
        print('Dataset:\t', data_id)
        print('Total queries:\t', num_queries)
        print('\tTrain:', train_dataset.__len__(), 'Vali:', vali_dataset.__len__(), 'Test:', test_dataset.__len__())

        num_docs = get_doc_num(train_dataset) + get_doc_num(vali_dataset) + get_doc_num(test_dataset)
        print('Total docs:\t', num_docs)

        min_doc, max_doc, sum_rele = get_min_max_docs(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset)
        print('min, max documents per query', min_doc, max_doc)
        print('total relevant documents', sum_rele)
        print('avg rele documents per query', sum_rele * 1.0 / num_queries)
        print('avg documents per query', num_docs * 1.0 / num_queries)

        #print()
        #get_min_max_feature(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset)




if __name__ == '__main__':
    # 1
    data_id  = 'MQ2007_Super'
    dir_data = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2007/'

    ''' results as below
    Total queries:	 1692
        Train: 1017 Vali: 339 Test: 336
    Total docs:	 69623
    min, max documents per query 6 147
    total relevant documents 17991
    avg rele documents per query 10.632978723404255
    avg documents per query 41.1483451536643
    
    '''

    #2
    #data_id  = 'MQ2008_Semi'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008-semi/'

    ''' results as below
    Total queries:	 785
        Train: 472 Vali: 157 Test: 156
    Total docs:	 546260
    min, max documents per query 5 531049
    total relevant documents 2932
    avg rele documents per query 3.735031847133758
    avg documents per query 695.8726114649681
    '''

    #data_id  = 'Istella_S'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'
    '''
    Dataset:	 Istella_S
    Total queries:	 33018
        Train: 19245 Vali: 7211 Test: 6562
    Total docs:	 3408630
    min, max documents per query 3 182
    total relevant documents 388224
    avg rele documents per query 11.757950208976922
    avg documents per query 103.23550790477921
    '''



    check_dataset_statistics(data_id=data_id, dir_data=dir_data)
