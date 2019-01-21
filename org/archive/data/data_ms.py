#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 23/05/2018

"""Description

"""
import os
import numpy as np
import sklearn.externals.six
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import torch
import torch.utils.data as data

from org.archive.utils.bigdata.BigPickle import pickle_save, pickle_load
from org.archive.utils.numpy.np_extensions import np_arg_shuffle_ties


#from org.archive.ranker.utils import shuffle_order_sampling

## Section-1: loading MS-format Learning-to-rank dataset ##

def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    """Transforms an iterator of lines to an iterator of LETOR rows.
    Each row is represented by a (x, y, qid, comment) tuple.
    Parameters
    ----------
    lines : iterable of lines
        Lines to parse.
    has_targets : bool, optional, i.e., the relevance label
        Whether the file contains targets. If True, will expect the first token
        of every line to be a real representing the sample's target (i.e.
        score). If False, will use -1 as a placeholder for all targets.
    one_indexed : bool, optional, i.e., whether the index of the first feature is 1
        Whether feature ids are one-indexed. If True, will subtract 1 from each
        feature id.
    missing : float, optional
        Placeholder to use if a feature value is not provided for a sample.
    Yields
    ------
    x : array of floats
        Feature vector of the sample.
    y : float
        Target value (score) of the sample, or -1 if no target was parsed.
    qid : object
        Query id of the sample. This is currently guaranteed to be a string.
    comment : str
        Comment accompanying the sample.
    """
    for line in lines:
        if has_comment:
            data, _, comment = line.rstrip().partition('#')
            toks = data.split()
        else:
            toks = line.rstrip().split()

        num_features = 0
        feature_vec = np.repeat(missing, 8)
        std_score = -1.0
        if has_targets:
            std_score = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1

            assert fid >= 0
            while len(feature_vec) <= fid:
                orig = len(feature_vec)
                feature_vec.resize(len(feature_vec) * 2)
                feature_vec[orig:orig * 2] = missing

            feature_vec[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        feature_vec.resize(num_features)

        if has_comment:
            yield (feature_vec, std_score, qid, comment)
        else:
            yield (feature_vec, std_score, qid)



def read_dataset(source, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    """Parses a LETOR dataset from `source`.
    Parameters
    ----------
    source : string or iterable of lines
        String, file, or other file-like object to parse.
    has_targets : bool, optional
        See `iter_lines`.
    one_indexed : bool, optional
        See `iter_lines`.
    missing : float, optional
        See `iter_lines`.
    Returns
    -------
    X : array of arrays of floats
        Feature matrix (see `iter_lines`).
    y : array of floats
        Target vector (see `iter_lines`).
    qids : array of objects
        Query id vector (see `iter_lines`).
    comments : array of strs
        Comment vector (see `iter_lines`).
    """
    if isinstance(source, sklearn.externals.six.string_types):
        source = source.splitlines()

    max_width = 0
    feature_vecs, std_scores, qids = [], [], []
    if has_comment:
        comments = []

    it = iter_lines(source, has_targets=has_targets, one_indexed=one_indexed, missing=missing, has_comment=has_comment)
    if has_comment:
        for f_vec, s, qid, comment in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            comments.append(comment)
            max_width = max(max_width, len(f_vec))
    else:
        for f_vec, s, qid in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            max_width = max(max_width, len(f_vec))

    assert max_width > 0
    all_features_mat = np.ndarray((len(feature_vecs), max_width), dtype=np.float64)
    all_features_mat.fill(missing)
    for i, x in enumerate(feature_vecs):
        all_features_mat[i, :len(x)] = x

    all_labels_vec = np.array(std_scores)

    if has_comment:
        docids = [_parse_docid(comment) for comment in comments]
        #features, std_scores, qids, docids
        return all_features_mat, all_labels_vec, qids, docids
    else:
        # features, std_scores, qids
        return all_features_mat, all_labels_vec, qids

def _parse_docid(comment):
    parts = comment.strip().split()
    return parts[2]

def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]


## For listwise usage ##

class QueryL(object):
    '''
    The collections of query-document pairs w.r.t. a unique query
    '''
    def __init__(self, qid, list_docids=None, list_features=None, list_labels=None):
        self.qid = qid
        self.list_docids = list_docids
        self.list_features = list_features
        self.list_labels = list_labels

class QueryM(object):
    '''
    The collections of query-document pairs w.r.t. a unique query
    '''
    def __init__(self, qid, list_docids=None, feature_mat=None, std_label_vec=None, unknown_as_zero=False, binary_rele=False):
        '''
        :param qid:
        :param list_docids:
        :param feature_mat:
        :param std_label_vec:
        :param unknown_as_zero: corresponds to the semi-supervised case
        '''
        self.qid = qid
        self.list_docids = list_docids
        self.feature_mat = feature_mat
        if binary_rele:
            self.std_label_vec = np.clip(std_label_vec, a_min=0, a_max=1)
        elif unknown_as_zero and not binary_rele:
            self.std_label_vec = np.clip(std_label_vec, a_min=0, a_max=10)
        else:
            self.std_label_vec = std_label_vec


def get_qm_file_buffer(in_file, has_comment=False, query_level_scale=False, scaler_id='MinMax', unknown_as_zero=False, binary_rele=False):
    prefix = in_file[:in_file.find('.txt')]

    if binary_rele:
        prefix = '_'.join([prefix, 'BiRele'])
    if unknown_as_zero:
        prefix = '_'.join([prefix, 'UnkZero'])

    if query_level_scale:
        scale_prefix = '_'.join(['QLevelScale', scaler_id])
        file_buffer = '_'.join([prefix, 'C' + str(has_comment), scale_prefix + '.pkl'])
    else:
        file_buffer = '_'.join([prefix, 'C' + str(has_comment) + '.pkl'])

    file_buffer = file_buffer.replace('Fold', 'BufferedFold')
    return file_buffer

def load_ms_data_qm(in_file, has_comment=False, query_level_scale=False, scaler_id='MinMax', file_buffer=None, unknown_as_zero=False, binary_rele=False):
    '''
    :param in_file:
    :param has_comment:
    :param query_level_scale: perform query-level scaling, say normalization
    :param scaler: MinMaxScaler | RobustScaler
    :param unknown_as_zero: if not labled, regard the relevance degree as zero
    :return:
    '''
    if file_buffer is None:
        file_buffer = get_qm_file_buffer(in_file=in_file, has_comment=has_comment, query_level_scale=query_level_scale, scaler_id=scaler_id, unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)

    if os.path.exists(file_buffer):
        return pickle_load(file_buffer), file_buffer
    else:
        parent_dir = Path(file_buffer).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    if scaler_id == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler_id == 'Robust':
        scaler = RobustScaler()
    else:
        raise NotImplementedError

    list_Qs = []
    with open(in_file) as file_obj:
        dict_data = dict()

        if has_comment:
            all_features_mat, all_labels_vec, qids, docids = read_dataset(file_obj, has_comment=True)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]
                docid = docids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, docid, f_vec))
                else:
                    dict_data[qid] = [(std_s, docid, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))
                list_labels_per_q = tmp[0]
                list_docids_per_q = tmp[1]
                list_features_per_q = tmp[2]
                feature_mat = np.vstack(list_features_per_q)
                if query_level_scale:
                    feature_mat = scaler.fit_transform(feature_mat)

                Q = QueryM(qid=qid, list_docids=list_docids_per_q, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q), unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)
                list_Qs.append(Q)
        else:
            all_features_mat, all_labels_vec, qids = read_dataset(file_obj, has_comment=False)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, f_vec))
                else:
                    dict_data[qid] = [(std_s, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))
                list_labels_per_q = tmp[0]
                list_features_per_q = tmp[1]
                feature_mat = np.vstack(list_features_per_q)
                if query_level_scale:
                    feature_mat = scaler.fit_transform(feature_mat)
                Q = QueryM(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q), unknown_as_zero=unknown_as_zero, binary_rele=binary_rele)
                list_Qs.append(Q)
    #buffer
    pickle_save(list_Qs, file=file_buffer)
    return list_Qs, file_buffer


def get_data_meta(dataset=None):
    ## query-document pair graded, [0-4] ##
    if dataset == 'MSLRWEB10K' or dataset == 'MSLRWEB30K':
        max_rele_level = 4
        multi_level_rele = True
        num_features = 136
        has_comment = False
        #unknown_as_zero = False

    ## query-document pair graded ##
    elif dataset == 'MQ2007_super' or dataset == 'MQ2008_super':
        max_rele_level = 2
        multi_level_rele = True
        num_features = 46
        has_comment = True
        #unknown_as_zero = False

    elif dataset == 'MQ2007_semi' or dataset == 'MQ2008_semi':
        max_rele_level = 2
        multi_level_rele = True
        num_features = 46
        has_comment = True
        #unknown_as_zero = True # if not labled, regard the relevance degree as zero

    ## query-document pair graded, [0-4] ##
    elif dataset == 'Yahoo_L2R_Set_1' or dataset == 'Yahoo_L2R_Set_2':
        '''
        max_rele_level = 4
        multi_level_rele = True
        num_features = -1 # libsvm format, rather than uniform number
        has_comment = False
        unknown_as_zero = True
        '''
        raise NotImplementedError

    else:
        raise NotImplementedError

    return num_features, has_comment, multi_level_rele, max_rele_level


def get_max_rele_level(dataset=None):
    ## query-document pair graded, [0-4] ##
    if dataset == 'MSLRWEB10K':
        max_rele_level = 4

    elif dataset == 'MSLRWEB30K':
        max_rele_level = 4

    ## query-document pair graded ##
    elif dataset == 'MQ2007_super':
        max_rele_level = 2

    elif dataset == 'MQ2008_super':
        max_rele_level = 2

    elif dataset == 'MQ2007_list':
        max_rele_level = 2

    elif dataset == 'MQ2008_list':
        max_rele_level = 2
    else:
        raise NotImplementedError

    return max_rele_level



## pytorch-wrapper ##

def get_tor_file_buffer(source_buffer=None, need_pre_sampling=False, samples_per_q=1, sort_sample_ranking_in_des=True):
    if source_buffer is not None:  # consistent buffer
        if need_pre_sampling:
            suffix = '_'.join(['DesSort', str(sort_sample_ranking_in_des), 'Sp', str(samples_per_q)])
            tor_file_buffer = source_buffer.replace('.pkl', '_' + suffix + '.torch')
        else:
            tor_file_buffer = source_buffer.replace('pkl', 'torch')

    return tor_file_buffer


class Listwise_Dataset(data.Dataset):

    def __init__(self, Qs, source_buffer=None, need_pre_sampling=False, samples_per_q=1, sort_sample_ranking_in_des=True, tor_file_buffer=None, pytorch_support=True):
        '''
        :param Qs: original Query instance consisting of feature matrix & corresponding labels
        :param samples_per_q: the number of sample rankin
        :param des_sorted: the documents are sorted in default descending order with shuffle_order_sampling()
        '''
        if pytorch_support:
            if tor_file_buffer is None:
                tor_file_buffer = get_tor_file_buffer(source_buffer=source_buffer, need_pre_sampling=need_pre_sampling, samples_per_q=samples_per_q, sort_sample_ranking_in_des=sort_sample_ranking_in_des)

            if os.path.exists(tor_file_buffer):
                self.all_Qs = pickle_load(tor_file_buffer)
            else:
                parent_dir = Path(tor_file_buffer).parent
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

                self.all_Qs = []
                if need_pre_sampling:
                    for Q in Qs:
                        list_ranking = []
                        list_std_label_vec = []
                        for _ in range(samples_per_q):
                            #des_inds = shuffle_order_sampling(Q.std_label_vec, reverse=sort_sample_ranking_in_des)  # sampled std-ranking
                            des_inds = np_arg_shuffle_ties(Q.std_label_vec, descending=sort_sample_ranking_in_des) # sampling by shuffling ties
                            list_ranking.append(Q.feature_mat[des_inds])
                            list_std_label_vec.append(Q.std_label_vec[des_inds])

                        batch_rankings = np.stack(list_ranking, axis=0)
                        batch_std_label_vec = np.stack(list_std_label_vec, axis=0)
                        tor_batch_rankings = torch.from_numpy(batch_rankings).type(torch.FloatTensor)
                        tor_batch_std_label_vec = torch.from_numpy(batch_std_label_vec).type(torch.FloatTensor)
                        tor_Q = QueryM(qid=Q.qid, feature_mat=tor_batch_rankings, std_label_vec=tor_batch_std_label_vec)
                        self.all_Qs.append(tor_Q)
                else:
                    for Q in Qs:
                        tor_batch_rankings = torch.from_numpy(Q.feature_mat).type(torch.FloatTensor)
                        tor_batch_std_label_vec = torch.from_numpy(Q.std_label_vec).type(torch.FloatTensor)
                        tor_Q = QueryM(qid=Q.qid, feature_mat=tor_batch_rankings, std_label_vec=tor_batch_std_label_vec)
                        self.all_Qs.append(tor_Q)

                pickle_save(self.all_Qs, file=tor_file_buffer)

        else:	# application case: lambdaMART
            if need_pre_sampling:
                for Q in Qs:
                    list_ranking = []
                    list_std_label_vec = []
                    for _ in range(samples_per_q):
                        #des_inds = shuffle_order_sampling(Q.std_label_vec, reverse=sort_sample_ranking_in_des)  # sampled std-ranking
                        des_inds = np_arg_shuffle_ties(Q.std_label_vec, descending=sort_sample_ranking_in_des)  # sampling by shuffling ties
                        list_ranking.append(Q.feature_mat[des_inds])
                        list_std_label_vec.append(Q.std_label_vec[des_inds])

                    batch_rankings = np.stack(list_ranking, axis=0)
                    batch_std_label_vec = np.stack(list_std_label_vec, axis=0)
                    current_Q = QueryM(qid=Q.qid, feature_mat=batch_rankings, std_label_vec=batch_std_label_vec)
                    self.all_Qs.append(current_Q)
            else:
                self.all_Qs = Qs

    def __getitem__(self, index):
        current_Q = self.all_Qs[index]
        return current_Q.feature_mat, current_Q.std_label_vec, current_Q.qid

    def __len__(self):
        return len(self.all_Qs)


class Listwise_Dataset_Step2(data.Dataset):

    def __init__(self, Qs, qm=False):
        self.tor_Qs = Qs
        self.qm = qm

    def __getitem__(self, index):
        tor_Q = self.tor_Qs[index]

        if self.qm:
            return tor_Q.feature_mat, tor_Q.std_label_vec, tor_Q.qid
        else:
            #tor_std_label_vec, truncated_labels, truncated_ranking, truncated_step_1_pred
            return tor_Q[0], tor_Q[1], tor_Q[2], tor_Q[3]

    def __len__(self):
        return len(self.tor_Qs)

#from ylab.archive.listwise.ranking_listwise import light_filtering

def test_loader():
    dir_data, num_features, has_comment, query_level_scale, multi_level_rele = get_data_meta(dataset='MQ2007_super', pc='imac')
    file_train = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2007/Fold1/train.txt'

    original_train_Qs, train_buffer = load_ms_data_qm(in_file=file_train, has_comment=has_comment, query_level_scale=query_level_scale)
    print(len(original_train_Qs))
    #train_Qs = light_filtering(original_train_Qs, min_docs=10, min_rele=1)
    train_data = Listwise_Dataset(Qs=original_train_Qs, source_buffer=train_buffer, need_pre_sampling=True, samples_per_q=2)
    print(train_data.__len__())
    train_data_loader = data.DataLoader(train_data, shuffle=True, batch_size=1)
    for i, entry in enumerate(train_data_loader):
        if i > 3:
            break
        else:
            batch_rankings, batch_std_label_vec = entry[0], entry[1]
            print(batch_rankings.shape)
            print(batch_std_label_vec.shape)
            print(torch.max(batch_std_label_vec))

    file_vali = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2007/Fold1/vali.txt'
    original_vali_Qs, vali_buffer = load_ms_data_qm(in_file=file_vali, has_comment=has_comment, query_level_scale=query_level_scale)
    print(len(original_vali_Qs))
    # train_Qs = light_filtering(original_train_Qs, min_docs=10, min_rele=1)
    vali_data = Listwise_Dataset(Qs=original_vali_Qs, source_buffer=vali_buffer, need_pre_sampling=False)
    print(vali_data.__len__())
    vali_data_loader = data.DataLoader(vali_data, shuffle=True, batch_size=1)
    for i, entry in enumerate(vali_data_loader):
        if i > 3:
            break
        else:
            batch_rankings, batch_std_label_vec = entry[0], entry[1]
            print(batch_rankings.shape)
            print(batch_std_label_vec.shape)
            print(torch.max(batch_std_label_vec))



## pytorch-wrapper ##

def get_doc_num(list_Qs):
    doc_num = 0
    for Q in list_Qs:
        doc_num += len(Q.std_label_vec)
    return doc_num

def get_min_max_docs(train_Qs, vali_Qs, test_Qs):
    min_doc = 10000000
    max_doc = 0
    sum_rele = 0
    for Q in train_Qs:
        doc_num = len(Q.std_label_vec)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (Q.std_label_vec>0).sum()

    for Q in vali_Qs:
        doc_num = len(Q.std_label_vec)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (Q.std_label_vec > 0).sum()

    for Q in test_Qs:
        doc_num = len(Q.std_label_vec)
        min_doc = min(doc_num, min_doc)
        max_doc = max(max_doc, doc_num)
        sum_rele += (Q.std_label_vec > 0).sum()

    return min_doc, max_doc, sum_rele


def light_filtering(ori_Qs=None, min_docs=None, min_rele=1):
    list_Qs = []
    for Q in ori_Qs:
        if Q.feature_mat.shape[0] < min_docs:           # skip queries with documents that are fewer the pre-specified min_docs
            continue
        if (Q.std_label_vec > 0).sum() < min_rele:      # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            continue

        list_Qs.append(Q)

    return list_Qs

def check_statistics(dataset=None, dir_data=None, fold_num=1):
    assert fold_num == 1 # one fold includes train, vali, test, which is just the total data
    num_features, has_comment, query_level_scale, multi_level_rele, _, _= get_data_meta(dataset=dataset)
    for fold_k in range(1, fold_num + 1):
        print('Fold-', fold_k)  # fold-wise data preparation plus certain light filtering
        dir_fold_k = dir_data + 'Fold' + str(fold_k) + '/'
        file_train, file_vali, file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'

        original_train_Qs, _ = load_ms_data_qm(in_file=file_train, has_comment=has_comment, query_level_scale=query_level_scale)
        original_vali_Qs, _ = load_ms_data_qm(in_file=file_vali, has_comment=has_comment, query_level_scale=query_level_scale)
        original_test_Qs, _ = load_ms_data_qm(in_file=file_test, has_comment=has_comment, query_level_scale=query_level_scale)

        #filtered_Qs = light_filtering(original_test_Qs, min_docs=10, min_rele=1)
        #num_docs = get_doc_num(filtered_Qs)
        #print(num_docs)

    num_queries = len(original_train_Qs) + len(original_vali_Qs) + len(original_test_Qs)
    print('Total queries:\t', num_queries)
    num_docs = get_doc_num(original_train_Qs) + get_doc_num(original_vali_Qs) + get_doc_num(original_test_Qs)
    print('Total docs:\t', num_docs)

    min_doc, max_doc, sum_rele = get_min_max_docs(train_Qs=original_train_Qs, vali_Qs=original_vali_Qs, test_Qs=original_test_Qs)
    print('min, max documents per query', min_doc, max_doc, sum_rele)
    print('avg rele documents per query', sum_rele*1.0/num_queries)
    print('avg documents per query', num_docs * 1.0 / num_queries)



## XGBoost lambdaMART ##
def save_data_xgboost(group_data, output_feature, output_group, group_labels=None, min_docs=None, min_rele=None, dataset=None):
    if len(group_data) == 0:
        return

    if len(group_labels) < min_docs:
        return
    else:
        label_vec = np.asarray(group_labels, dtype=np.int)
        if (label_vec > 0).sum() < min_rele:  # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            return

    output_group.write(str(len(group_data)) + "\n")
    for data in group_data:
        # only include nonzero features
        if dataset == 'MQ2007_super':
            feats = [p for p in data[2:]] # due to mismatch error, i.e., the number of features in train data is different from the number of features in validation
        else:
            feats = [p for p in data[2:] if float(p.split(':')[1]) != 0.0]

        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def get_xgboost_buffer(in_file):
    buffer_prefix = in_file.replace('Fold', 'BufferedFold')
    file_buffered_data = buffer_prefix.replace('txt', 'data')
    file_buffered_group = buffer_prefix.replace('txt', 'group')
    return file_buffered_data, file_buffered_group

def load_data_xgboost(in_file, min_docs=None, min_rele=None, dataset=None):
    file_buffered_data, file_buffered_group = get_xgboost_buffer(in_file)

    if os.path.exists(file_buffered_data):
        return file_buffered_data, file_buffered_group
    else:
        parent_dir = Path(file_buffered_data).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        # run xgboost/demo/rank/trans_data.py
        fi = open(in_file)
        output_feature = open(file_buffered_data, "w")
        output_group = open(file_buffered_group, "w")

        group_data = []
        group_labels = []
        group = ""
        for line in fi:
            if not line:
                break
            if "#" in line:
                line = line[:line.index("#")]
            splits = line.strip().split(" ")
            if splits[1] != group:
                save_data_xgboost(group_data, output_feature, output_group, group_labels=group_labels, min_docs=min_docs, min_rele=min_rele, dataset=dataset)
                group_data = []
                group_labels = []

            group = splits[1]
            group_data.append(splits)
            group_labels.append(splits[0])

        save_data_xgboost(group_data, output_feature, output_group, group_labels=group_labels, min_docs=min_docs, min_rele=min_rele, dataset=dataset)

        fi.close()
        output_feature.close()
        output_group.close()

        return file_buffered_data, file_buffered_group

## XGBoost lambdaMART ##


if __name__ == '__main__':
    #test
    '''
    in_file = '/Users/dryuhaitao/WorkBench/Corpus/Learning2Rank/MSLR-WEB10K/all.txt'
    list_Qs = load_ms_data_qm(in_file=in_file)

    list_doc_nums = []
    for Q in list_Qs:
        assert (len(Q.std_label_vec) == Q.feature_mat.shape[0])
        list_doc_nums.append(len(Q.std_label_vec))

    print('Number of queries:', len(list_doc_nums))
    print('Maximum docs:', max(list_doc_nums))
    print('Minimum docs:', min(list_doc_nums))
    print('Avg docs per query:', sum(list_doc_nums)/len(list_doc_nums))

    """
    Number of queries: 10000
    Maximum docs: 908
    Minimum docs: 1
    Avg docs per query: 120.0192
    """

    '''

    #pl_sampling
    '''
    pros = [0.1, 0.3, 0.4, 0.2]
    result = np.random.choice(a=len(pros), size=len(pros), replace=False, p=pros)
    print(result)
    '''

    #check correctness
    #in_file = '/Users/dryuhaitao/WorkBench/Corpus/Learning2Rank/MSLR-WEB10K/all.txt'
    #list_Qs = load_ms_data_qm(in_file=in_file)

    '''
    in_file = '/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2007-list/Fold1/test.txt'
    list_Qs = load_ms_data_qm(in_file=in_file, has_comment=True)



    Q = list_Qs[len(list_Qs)-1]
    print(Q.qid)
    print(Q.feature_mat.shape)
    print(Q.std_label_vec.shape)
    print(Q.std_label_vec)
    print(Q.feature_mat[0, :])
    print(Q.feature_mat[Q.feature_mat.shape[0]-1, :])
    '''

    #
    #test_loader()

    #
    check_statistics(dataset='MQ2007_super', dir_data='/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/', fold_num=1)
    '''
    Total queries:	 1692
    Total docs:	 69623
    min, max documents per query 6 147
    avg rele documents per query 10.632978723404255
    avg documents per query 41.1483451536643
    '''

    #check_statistics(dataset='MQ2008_super', dir_data='/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/', fold_num=1)
    '''
    Total queries:	 784
    Total docs:	 15211
    min, max documents per query 5 121
    avg rele documents per query 3.739795918367347
    avg documents per query 19.401785714285715
    '''

    #check_statistics(dataset='MSLRWEB10K')
    '''
    Total queries:	 10000
    Total docs:	 1200192
    min, max documents per query 1 908
    '''

    #check_statistics(dataset='MSLRWEB30K')

    '''
    Total queries:	 31531
    Total docs:	 3771125
    min, max documents per query 1 1251
    '''

    #check_statistics(dataset='MQ2007_semi', dir_data='/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007-semi/')
    '''
    Total queries:	 1692
    Total docs:	 770724
    min, max documents per query 10 1268 17991
    avg rele documents per query 10.632978723404255
    avg documents per query 455.51063829787233
    '''

    #check_statistics(dataset='MQ2008_semi', dir_data='/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008-semi/')
    '''
    Total queries:	 784
    Total docs:	 546260
    min, max documents per query 6 1831 2932
    avg rele documents per query 3.739795918367347
    avg documents per query 696.7602040816327
    '''



