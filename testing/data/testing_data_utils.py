#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Examples on how to use data_utils module
"""

from ptranking.data.data_utils import LTRDataset, YAHOO_LTR, ISTELLA_LTR, MSLETOR_SEMI
import torch

def get_doc_num(dataset):
    ''' compute the number of documents in a dataset '''
    doc_num = 0
    for qid, torch_batch_rankings, torch_batch_std_labels in dataset:
        doc_num += torch_batch_std_labels.size(1)

    return doc_num

def get_min_max_docs(train_dataset, vali_dataset, test_dataset, semi_supervised=False):
    ''' get the minimum / maximum number of documents per query '''
    min_doc = 10000000
    max_doc = 0
    sum_rele = 0
    if semi_supervised:
        sum_unknown = 0

    for qid, torch_batch_rankings, torch_batch_std_labels in train_dataset:
        #print('torch_batch_std_labels', torch_batch_std_labels.size())
        doc_num = torch_batch_std_labels.size(1)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (torch_batch_std_labels>0).sum()
        if semi_supervised:
            sum_unknown += (torch_batch_std_labels<0).sum()

    if vali_dataset is not None:
        for qid, torch_batch_rankings, torch_batch_std_labels in vali_dataset:
            doc_num = torch_batch_std_labels.size(1)
            min_doc = min(doc_num, min_doc)
            max_doc = max(max_doc, doc_num)
            sum_rele += (torch_batch_std_labels>0).sum()
            if semi_supervised:
                sum_unknown += (torch_batch_std_labels < 0).sum()

    for qid, torch_batch_rankings, torch_batch_std_labels in test_dataset:
        doc_num = torch_batch_std_labels.size(1)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (torch_batch_std_labels>0).sum()
        if semi_supervised:
            sum_unknown += (torch_batch_std_labels<0).sum()

    if semi_supervised:
        return min_doc, max_doc, sum_rele.data.numpy(), sum_unknown.data.numpy()
    else:
        return min_doc, max_doc, sum_rele.data.numpy()


def get_min_max_feature(train_dataset, vali_dataset, test_dataset):
    ''' get the minimum / maximum feature values in a dataset '''
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
    if data_id in YAHOO_LTR:
        data_prefix = dir_data + data_id.lower() + '.'
        file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'

    elif data_id in ISTELLA_LTR:
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
        train_dataset = LTRDataset(train=True, file=file_train, data_id=data_id, shuffle=False, buffer=buffer)
        test_dataset =  LTRDataset(train=False, file=file_test, data_id=data_id, shuffle=False, buffer=buffer)

        num_queries = train_dataset.__len__() + test_dataset.__len__()
        print('Dataset:\t', data_id)
        print('Total queries:\t', num_queries)
        print('\tTrain:', train_dataset.__len__(), 'Test:', test_dataset.__len__())

        num_docs = get_doc_num(train_dataset) + get_doc_num(test_dataset)
        print('Total docs:\t', num_docs)

        min_doc, max_doc, sum_rele = get_min_max_docs(train_dataset=train_dataset, vali_dataset=None, test_dataset=test_dataset)
    else:
        train_dataset = LTRDataset(train=True, file=file_train, data_id=data_id, shuffle=False, buffer=buffer)
        vali_dataset =  LTRDataset(train=False, file=file_vali, data_id=data_id, shuffle=False, buffer=buffer)
        test_dataset =  LTRDataset(train=False, file=file_test, data_id=data_id, shuffle=False, buffer=buffer)

        num_queries = train_dataset.__len__() + vali_dataset.__len__() + test_dataset.__len__()
        print('Dataset:\t', data_id)
        print('Total queries:\t', num_queries)
        print('\tTrain:', train_dataset.__len__(), 'Vali:', vali_dataset.__len__(), 'Test:', test_dataset.__len__())

        num_docs = get_doc_num(train_dataset) + get_doc_num(vali_dataset) + get_doc_num(test_dataset)
        print('Total docs:\t', num_docs)

        if data_id in MSLETOR_SEMI:
            min_doc, max_doc, sum_rele, sum_unknown = \
                get_min_max_docs(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset, semi_supervised=True)
        else:
            min_doc, max_doc, sum_rele = get_min_max_docs(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset)

    print('min, max documents per query', min_doc, max_doc)
    print('total relevant documents', sum_rele)
    print('avg rele documents per query', sum_rele * 1.0 / num_queries)
    print('avg documents per query', num_docs * 1.0 / num_queries)
    if data_id in MSLETOR_SEMI:
        print('total unlabeled documents', sum_unknown)

        #print()
        #get_min_max_feature(train_dataset=train_dataset, vali_dataset=vali_dataset, test_dataset=test_dataset)




if __name__ == '__main__':
    # 1
    data_id  = 'MQ2007_Super'
    dir_data = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2007/'
    dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'

    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
    ''' results as below
    Total queries:	 1692
        Train: 1017 Vali: 339 Test: 336
    Total docs:	 69623
    min, max documents per query 6 147
    total relevant documents 17991
    avg rele documents per query 10.632978723404255
    avg documents per query 41.1483451536643
    
    '''

    data_id = 'MQ2008_Super'
    dir_data = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2008/'
    check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
    '''
    
    '''

    #2
    data_id  = 'MQ2008_Semi'
    dir_data = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2008-semi/'
    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
    ''' results as below
    Total queries:	 784
        Train: 471 Vali: 157 Test: 156
    Total docs:	 546260
    min, max documents per query 5 531049
    total relevant documents 2932
    avg rele documents per query 3.735031847133758
    avg documents per query 695.8726114649681
    
    Dataset:	 MQ2008_Semi
    Total queries:	 784
        Train: 471 Vali: 157 Test: 156
    Total docs:	 546260
    min, max documents per query 6 1831
    total relevant documents 2932
    avg rele documents per query 3.739795918367347
    avg documents per query 696.7602040816327
    '''

    #data_id  = 'Istella_S'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'
    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
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

    #data_id  = 'Istella_X'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'
    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=True)
    '''
    Dataset:	 Istella_X
    Total queries:	 10000
        Train: 6000 Vali: 2000 Test: 2000
    Total docs:	 26791447
    min, max documents per query 1 4999
    total relevant documents 46371
    avg rele documents per query 4.6371
    avg documents per query 2679.1447
    '''

    #data_id  = 'Istella'
    #dir_data = '/home/dl-box/WorkBench/Datasets/L2R/ISTELLA_L2R/'

    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)

    data_id  = 'MQ2007_List'
    dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2007-list/'
    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
    '''
    Dataset:	 MQ2007_List
    Total queries:	 1692
        Train: 1017 Vali: 339 Test: 336
    Total docs:	 1231351
    min, max documents per query 257 1346
    total relevant documents 1231351
    avg rele documents per query 727.7488179669031
    avg documents per query 727.7488179669031
    '''

    data_id  = 'MQ2008_List'
    dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008-list/'
    #check_dataset_statistics(data_id=data_id, dir_data=dir_data, buffer=False)
    '''
    Dataset:	 MQ2008_List
    Total queries:	 784
        Train: 471 Vali: 157 Test: 156
    Total docs:	 902220
    min, max documents per query 204 1831
    total relevant documents 902220
    avg rele documents per query 1150.7908163265306
    avg documents per query 1150.7908163265306
    '''
