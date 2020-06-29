#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import os
import random
import warnings
import collections
import numpy  as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import torch
import torch.utils.data as data

from org.archive.ltr_adhoc.util.bin_utils import batch_count
from org.archive.utils.numpy.np_extensions import np_arg_shuffle_ties
from org.archive.ltr_adhoc.util.one_hot_utils import get_one_hot_reprs
from org.archive.utils.bigdata.BigPickle import pickle_save, pickle_load


## Supported datasets and formats ##

MSLETOR_SEMI  = ['MQ2007_Semi', 'MQ2008_Semi', 'IRGAN_Adhoc_Semi']
MSLETOR_LIST  = ['MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR       = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List', 'IRGAN_Adhoc_Semi']

MSLRWEB       = ['MSLRWEB10K', 'MSLRWEB30K']

YAHOO_L2R     = ['Set1', 'Set2']
YAHOO_L2R_5Fold     = ['5FoldSet1', '5FoldSet2']


ISTELLA_L2R   = ['Istella_S', 'Istella', 'Istella_X']
ISTELLA_MAX = 1000000 # As ISTELLA contain extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000

GLTR_LIBSVM = ['LTR_LibSVM', 'LTR_LibSVM_K']
GLTR_LETOR  = ['LETOR', 'LETOR_K']

"""
GLTR refers to General Learning-to-rank, thus
GLTR_LIBSVM and GLTR_LETOR refer to general learning-to-rank datasets given in the formats of libsvm and LETOR, respectively.

The suffix '_K' indicates that the dataset consists of K folds in order to perform k-fold cross validation.

For GLTR_LIBSVM, it is defined as follows, where features with zero values are not included.
<ground-truth label int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

4 qid:105 2:0.4 8:0.7 50:0.5
1 qid:105 5:0.5 30:0.7 32:0.4 48:0.53
0 qid:210 4:0.9 38:0.01 39:0.5 45:0.7
1 qid:210 1:0.2 8:0.9 31:0.93 40:0.6

The above sample dataset includes two queries, the query “105” has 2 documents, the corresponding ground-truth labels are 4 and 1, respectively.

For GLTR_LETOR, it is defined as follows, where features with zero values are still included and the number of features per row must be the same.
<ground-truth label int> qid:<query_id int> [<feature_id int>:<feature_value float>]

4 qid:105 1:0.4 2:0.7  3:0.5
1 qid:105 1:0.5 2:0.7  3:0.4
0 qid:210 1:0.9 2:0.01 3:0.5
1 qid:210 1:0.2 2:0.9  3:0.93
"""

## supported feature normalization ##
SCALER_LEVEL = ['QUERY', 'DATASET']
SCALER_ID    = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']

## supported ways of masking labels ##
MASK_TYPE = ['rand_mask_all', 'rand_mask_rele']


def get_data_meta(data_id=None):
    '''
    :param data_id:
    :return: get the meta-information corresponding to the specified dataset
    '''

    if data_id in MSLRWEB:
        max_rele_level = 4
        multi_level_rele = True
        num_features = 136
        has_comment = False
        fold_num = 5

    elif data_id in MSLETOR_SUPER:
        max_rele_level = 2
        multi_level_rele = True
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_SEMI:
        max_rele_level = 2
        multi_level_rele = True
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_LIST:
        max_rele_level = None
        multi_level_rele = False
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in YAHOO_L2R:
        max_rele_level = 4
        multi_level_rele = True
        num_features = 700 # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 1

    elif data_id in YAHOO_L2R_5Fold:
        max_rele_level = 4
        multi_level_rele = True
        num_features = 700  # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 5

    elif data_id in ISTELLA_L2R:
        max_rele_level = 4
        multi_level_rele = True
        num_features = 220  # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 1

    else:
        raise NotImplementedError

    data_meta = dict(num_features=num_features, has_comment=has_comment, multi_level_rele=multi_level_rele, max_rele_level=max_rele_level, fold_num=fold_num)
    return data_meta


def enrich(x, num_features=None):
    ''' enrich libsvm with zeros '''
    dict_feats = collections.OrderedDict([(str(i), ':'.join([str(i), '0'])) for i in range(1, num_features+1)])

    feats = x.strip().split(" ")
    for feat in feats:
        dict_feats[feat.split(':')[0]] = feat

    enriched_feats = list(dict_feats.values())
    return ' '.join(enriched_feats)


class L2RDataLoader():
    """
    Loading the specified dataset as: list of tuples consisting of (query_id, all_document_features_as_numpy_tensor, all_labels_as_numpy_tensor)
    """

    def __init__(self, train, file, data_dict=None, buffer=True):
        '''
        :param train: training data or not
        :param file:
        :param data_id:
        :param data_dict:
        :param buffer: buffer the dataset in the format as list of tuples in case of multiple access
        '''

        self.df = None
        self.train = train
        self.file = file
        self.buffer = buffer

        self.data_id   = data_dict['data_id']

        assert self.data_id in MSLETOR or self.data_id in MSLRWEB \
               or self.data_id in YAHOO_L2R or self.data_id in YAHOO_L2R_5Fold \
               or self.data_id in ISTELLA_L2R

        self.data_dict = data_dict
        self.pre_check()

        if self.data_id in YAHOO_L2R or self.data_id in ISTELLA_L2R:
            self.df_file = file[:file.find('.txt')].replace(self.data_id.lower() + '.', 'Buffered' + self.data_id + '/') + '.df'  # the original data file buffer as a dataframe
        else:
            self.df_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '.df' # the original data file buffer as a dataframe

        # plus filtering, such as the queries that includes no relevant documents (stdandard labels), we call it dumb queries
        self.filtering = False

        self.min_rele  = self.data_dict['min_rele']
        if self.min_rele is not None and self.min_rele > 0:
            self.filtering = True
            fi_suffix = '_'.join(['MiR', str(self.min_rele)])
        else:
            fi_suffix = ''

        self.min_docs = self.data_dict['min_docs']
        if self.min_docs is not None and self.min_docs > 0:
            self.filtering = True
            if len(fi_suffix)>0:
                fi_suffix = '_'.join([fi_suffix, 'MiD', str(self.min_docs)])
            else:
                fi_suffix = '_'.join(['MiD', str(self.min_docs)])

        res_suffix = ''
        if self.data_dict['binary_rele']:
            res_suffix += '_B'

        if self.data_dict['unknown_as_zero']:
            res_suffix += '_UO'

        if self.filtering:
            if self.data_id in YAHOO_L2R:
                self.filtered_df_file = file[:file.find('.txt')].replace(self.data_id.lower() + '.', 'Buffered' + self.data_id + '/') + '_' + fi_suffix + res_suffix + '.df'
            else:
                self.filtered_df_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' +fi_suffix + res_suffix +'.df'

        pq_suffix = '_'.join([fi_suffix, 'PerQ']) if len(fi_suffix) > 0 else 'PerQ'

        if self.data_dict['presort']: pq_suffix = '_'.join([pq_suffix, 'PreSort'])

        # plus scaling
        self.scale_data   = self.data_dict['scale_data']
        self.scaler_id    = self.data_dict['scaler_id'] if 'scaler_id' in self.data_dict else None
        self.scaler_level = self.data_dict['scaler_level'] if 'scaler_level' in self.data_dict else None
        if self.scale_data:
            assert self.scaler_id is not None and self.scaler_id in SCALER_ID and self.scaler_level in SCALER_LEVEL

            if 'DATASET' == self.scaler_level:
                pq_suffix = '_'.join([pq_suffix, 'DS', self.scaler_id])
            else:
                pq_suffix = '_'.join([pq_suffix, 'QS', self.scaler_id])

        if self.data_id in YAHOO_L2R:
            self.perquery_file = file[:file.find('.txt')].replace(self.data_id.lower() + '.', 'Buffered' + self.data_id + '/') + '_' + pq_suffix + res_suffix + '.np'
        else:
            self.perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix + res_suffix +'.np'





    def pre_check(self):
        """
        Check whether the settings are consist w.r.t. a particular dataset
        """
        data_id = self.data_dict['data_id']
        '''
        if data_id in MSLETOR_SEMI:
            if self.data_dict['presort'] or self.data_dict['binary_rele']:
                assert True == self.data_dict['unknown_as_zero'] # a must due to '-1' documents

            s_key = 'binary_rele'
            mes = '{} is a sensitive setting w.r.t. {}, do you really mean {}'.format(s_key, data_id, self.data_dict[s_key])
            warnings.warn(message=mes)
        else:
            assert False == self.data_dict['unknown_as_zero']
        '''

        if data_id in MSLETOR_LIST:
            # the dataset, for which the standard ltr_adhoc of each query is unique
            assert self.data_dict['sample_times_per_q'] == 1
        else:
            pass
            # assert 'All' == data_dict['max_docs']


    def load_data(self, given_scaler=None):
        '''
        Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
        :param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
        :return:
        '''
        if self.data_id in MSLETOR:
            self.num_features = 46
        elif self.data_id in MSLRWEB:
            self.num_features = 136
        elif self.data_id in YAHOO_L2R or self.data_id in YAHOO_L2R_5Fold:
            self.num_features = 700
        elif self.data_id in ISTELLA_L2R:
            self.num_features = 220

        if os.path.exists(self.perquery_file):
            list_Qs = pickle_load(self.perquery_file)

            if self.train and self.scale_data and 'DATASET' == self.scaler_level: # fit dataset-level scaler and will be used for validation data and test data
                self.get_filtered_df_file()
                if self.scale_data: self.ini_scaler(joint_transform=False) # since the buffered file is already scaled
                self.df = None                         # the df object is for one-time fiting

            return list_Qs
        else:
            self.get_filtered_df_file()

            if not self.train and self.scale_data and 'DATASET' == self.scaler_level:
                assert given_scaler is not None
                self.scaler = given_scaler
            elif self.scale_data:
                self.ini_scaler(joint_transform=True)

            list_Qs = []
            qids = self.df.qid.unique()
            np.random.shuffle(qids)

            for qid in qids:
                if self.data_dict['presort']:
                    if self.train and self.data_id in MSLETOR_SEMI:
                        if self.data_dict['unknown_as_zero']:
                            qdf = self.df[self.df.qid == qid].sort_values('rele_truth_unk2zero', ascending=False)
                        else:
                            # due to the '-1' documents
                            raise NotImplementedError('Non-supported error!')
                    else:
                        qdf = self.df[self.df.qid == qid].sort_values('rele_truth', ascending=False)
                else:
                    qdf = self.df[self.df.qid == qid]  # non-sorted

                doc_reprs = qdf[self.feature_cols].values
                if self.scale_data:
                    if 'QUERY' == self.scaler_level:
                        if self.data_id in ISTELLA_L2R:
                            # due to the possible extremely large features, e.g., 1.79769313486e+308
                            doc_reprs = self.scaler.fit_transform(np.clip(doc_reprs, a_min=None, a_max=ISTELLA_MAX))
                        else:
                            doc_reprs = self.scaler.fit_transform(doc_reprs)
                    else:
                        doc_reprs = self.scaler.transform(doc_reprs)

                if self.train and self.data_id in MSLETOR_SEMI:
                    if self.data_dict['unknown_as_zero']:
                        if self.data_dict['binary_rele']:
                            doc_labels = qdf['rele_binary'].values
                        else:
                            doc_labels = qdf['rele_truth_unk2zero'].values
                    else:
                        doc_labels = qdf['rele_truth'].values
                else:
                    if self.data_dict['binary_rele']:
                        doc_labels = qdf['rele_binary'].values
                    else:
                        doc_labels = qdf['rele_truth'].values

                # doc_ids    = sorted_qdf['#docid'].values # commented due to rare usage
                list_Qs.append((qid, doc_reprs, doc_labels))

            if self.buffer:
                parent_dir = Path(self.perquery_file).parent
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

                pickle_save(list_Qs, file=self.perquery_file)

            return list_Qs

    def get_df_file(self):
        ''' Load original data file as a dataframe. If buffer exists, load buffered file. '''

        if os.path.exists(self.df_file):
            self.df = pd.read_pickle(self.df_file)
        else:
            if self.data_id in MSLETOR:
                self.df = self.load_LETOR4()
            elif self.data_id in MSLRWEB:
                self.df = self.load_MSLRWEB()
            elif self.data_id in YAHOO_L2R or self.data_id in YAHOO_L2R_5Fold:
                self.df = self.load_YahooL2R()
            elif self.data_id in ISTELLA_L2R:
                self.df = self.load_ISTELLA()
            else:
                raise NotImplementedError

            '''
            if self.buffer:
                parent_dir = Path(self.df_file).parent
                if not os.path.exists(parent_dir): os.makedirs(parent_dir)
                self.df.to_pickle(self.df_file)
            '''

    def get_filtered_df_file(self):
        ''' Perform filtering over the dataframe file '''
        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]

        if self.filtering:
            if os.path.exists(self.filtered_df_file):
                self.df = pd.read_pickle(self.filtered_df_file)
            else:
                self.get_df_file()
                #print('T', self.df.qid.unique())
                self.filter()
                #print('X', self.df.qid.unique())
                #if self.buffer: self.df.to_pickle(self.filtered_df_file)
        else:
            self.get_df_file()



    def load_LETOR4(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[[-2, -3, -5, -6, -8, -9]], axis=1, inplace=True)  # remove redundant keys
        assert self.num_features == len(df.columns) - 5

        for c in range(1, self.num_features+2):           							 # remove keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        df.columns = ['rele_truth', 'qid'] + self.feature_cols + ['#docid', 'inc', 'prob']

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        if self.train and self.data_id in MSLETOR_SEMI:
            if self.data_dict['unknown_as_zero']:
                '''
                # the following way is not correct
                print(df.rele_truth.values)
                df['rele_truth_unk2zero'] = df['rele_truth'].copy(deep=True)
                print(df.rele_truth_unk2zero.values)
                df[df['rele_truth_unk2zero'] < 0] = 0
                print(df.rele_truth_unk2zero.values)
                print(df.rele_truth.values)
                '''

                df['rele_truth_unk2zero'] = df['rele_truth'].apply(lambda x: x if x > 0 else 0)

                if self.data_dict['binary_rele']:
                    df['rele_binary'] = (df['rele_truth_unk2zero'] > 0).astype(np.float32)  # additional binarized column for later filtering
            else:
                assert self.data_dict['binary_rele'] is not True

        else:
            df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

        return df


    def load_MSLRWEB(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True) # remove the line-break
        assert self.num_features == len(df.columns) - 2

        for c in range(1, len(df.columns)):           # remove the keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])


        df.columns = ['rele_truth', 'qid'] + self.feature_cols

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)     # additional binarized column for later filtering

        return df

    def load_ISTELLA(self):
        '''  '''
        df = pd.read_csv(self.file, header=None, lineterminator='\n', names=['all_in_one'])

        df['all_in_one'] = df.all_in_one.str.strip()

        df[['rele_truth', 'qid'] + self.feature_cols] = df.all_in_one.str.split(' ', n=222, expand=True)
        df.drop(columns=['all_in_one'], inplace=True)  # remove the original column of all_in_one

        #print(len(df.columns))
        assert self.num_features == len(df.columns) - 2

        for c in range(1, len(df.columns)):           # remove the keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        #df.columns = ['rele_truth', 'qid'] + self.feature_cols

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)     # additional binarized column for later filtering

        return df


    def load_YahooL2R(self):
        '''  '''
        df = pd.read_csv(self.file, names=['rele_truth'])  # the column of rele_truth will be updated
        #print('Loaded raw txt file.')

        cols = ['rele_truth', 'qid', 'features']
        df[cols] = df.rele_truth.str.split(' ', n=2, expand=True)  # pop-up label & qid

        df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: enrich(x, num_features=self.num_features))

        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]
        df[self.feature_cols] = df.features.str.split(' ', n=self.num_features, expand=True)  # split
        df.drop(columns=['features'], inplace=True)  # remove the feature string column

        #print('Finished spliting...')

        for c in range(1, self.num_features + 2):  # remove keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)     # additional binarized column for later filtering

        #print('Loaded DF')

        return df


    def ini_scaler(self, joint_transform=False):
        assert self.scaler_id in SCALER_ID
        if self.scaler_id == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.scaler_id == 'RobustScaler':
            self.scaler = RobustScaler()
        elif self.scaler_id == 'StandardScaler':
            self.scaler = StandardScaler()

        if self.train and 'DATASET' == self.scaler_level:
            f_mat = self.df[self.feature_cols]
            self.scaler.fit(f_mat)

            if joint_transform: self.df[self.feature_cols] = self.scaler.transform(f_mat)


    def filter(self):
        '''
        filter out dumb queries
        '''
        if self.data_id in MSLETOR_SUPER or self.data_id in MSLRWEB\
                or self.data_id in YAHOO_L2R or self.data_id in YAHOO_L2R_5Fold\
                or self.data_id in ISTELLA_L2R:

            if self.min_rele > 0: self.df = self.df.groupby('qid').filter(lambda s: s.rele_binary.sum() >= self.min_rele)
            #print(self.df.shape)
            if self.min_docs > 1: self.df = self.df.groupby('qid').filter(lambda s: len(s) >= self.min_docs)
            #print(self.df.shape)

        elif self.data_id in MSLETOR_SEMI:
            if not 'rele_binary' in self.df.columns:
                if not 'rele_binary' in self.df.columns:
                    self.df['rele_truth_unk2zero'] = self.df['rele_truth']
                    self.df[self.df['rele_truth_unk2zero'] < 0] = 0

                self.df['rele_binary'] = (self.df['rele_truth_unk2zero'] > 0).astype(np.float32)  # additional binarized column for later filtering

            # since unknown documents (-1) can be a relevant document
            if self.min_rele > 0: self.df = self.df.groupby('qid').filter(lambda s: s.rele_binary.abs().sum() >= self.min_rele)
            #print(self.df.shape)
            if self.min_docs >= 1: self.df = self.df.groupby('qid').filter(lambda s: len(s) >= self.min_docs)
        else:
            raise NotImplementedError



class L2RDataset(data.Dataset):
    '''
    Loading the specified dataset as data.Dataset
    '''

    def __init__(self, train, file, data_id=None, data_dict=None,
                 sample_rankings_per_q=1, shuffle=True, given_scaler=None, hot=False, eval_dict=None, buffer=True):

        assert data_id is not None or data_dict is not None
        if data_dict is None: data_dict = self.get_default_data_dict(data_id=data_id)

        self.train = train

        if data_dict['data_id'] in MSLETOR or data_dict['data_id'] in MSLRWEB \
                or data_dict['data_id'] in YAHOO_L2R or data_dict['data_id'] in YAHOO_L2R_5Fold \
                or data_dict['data_id'] in ISTELLA_L2R \
                or data_dict['data_id'] == 'IRGAN_Adhoc_Semi':

            loader = L2RDataLoader(train=train, file=file, data_dict=data_dict, buffer=buffer)
            #print(loader.perquery_file)
            perquery_file = loader.perquery_file

            if sample_rankings_per_q>1:
                if hot:
                    torch_perquery_file = perquery_file.replace('.np', '_'.join(['SP', str(sample_rankings_per_q), 'Hot', '.torch']))
                else:
                    torch_perquery_file = perquery_file.replace('.np', '_'.join(['SP', str(sample_rankings_per_q), '.torch']))
            else:
                if hot:
                    torch_perquery_file = perquery_file.replace('.np', '_Hot.torch')
                else:
                    torch_perquery_file = perquery_file.replace('.np', '.torch')

            if eval_dict is not None:
                semi_context, mask_ratio, mask_type = eval_dict['semi_context'], eval_dict['mask_ratio'], eval_dict['mask_type']
                if semi_context:
                    semi_ratio_str = '_'.join(['Semi', mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
                    torch_perquery_file = torch_perquery_file.replace('.torch', semi_ratio_str)
            else:
                semi_context = False

            if os.path.exists(torch_perquery_file):
                self.list_torch_Qs = pickle_load(torch_perquery_file)
            else:
                self.list_torch_Qs = []

                list_Qs = loader.load_data(given_scaler=given_scaler)
                list_inds = list(range(len(list_Qs)))
                for ind in list_inds:
                    qid, doc_reprs, doc_labels = list_Qs[ind]

                    if sample_rankings_per_q > 1:
                        assert semi_context is not True # not supported since it is rarely used.

                        list_ranking = []
                        list_labels = []
                        for _ in range(self.sample_rankings_per_q):
                            des_inds = np_arg_shuffle_ties(doc_labels, descending=True)  # sampling by shuffling ties
                            list_ranking.append(doc_reprs[des_inds])
                            list_labels.append(doc_labels[des_inds])

                        batch_rankings = np.stack(list_ranking, axis=0)
                        batch_std_labels = np.stack(list_labels, axis=0)

                        torch_batch_rankings = torch.from_numpy(batch_rankings).type(torch.FloatTensor)
                        torch_batch_std_labels = torch.from_numpy(batch_std_labels).type(torch.FloatTensor)

                    else:
                        torch_batch_rankings = torch.from_numpy(doc_reprs).type(torch.FloatTensor)
                        torch_batch_rankings = torch.unsqueeze(torch_batch_rankings, dim=0)  # a consistent batch dimension of size 1

                        torch_batch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)
                        torch_batch_std_labels = torch.unsqueeze(torch_batch_std_labels, dim=0)

                        if semi_context: # masking
                            if mask_type == 'rand_mask_rele':
                                torch_batch_rankings, torch_batch_std_labels = random_mask_rele_labels(batch_ranking=torch_batch_rankings, batch_label=torch_batch_std_labels, mask_ratio=mask_ratio, mask_value=0, presort=data_dict['presort'])

                            elif mask_type == 'rand_mask_all':
                                masked_res = random_mask_all_labels(batch_ranking=torch_batch_rankings, batch_label=torch_batch_std_labels, mask_ratio=mask_ratio, mask_value=0, presort=data_dict['presort'])
                                if masked_res is not None:
                                    torch_batch_rankings, torch_batch_std_labels = masked_res
                                else:
                                    continue
                            else:
                                raise NotImplementedError

                    if hot:
                        assert semi_context is not True # not supported since it is rarely used.

                        max_rele_level = data_dict['max_rele_level']
                        assert max_rele_level is not None

                        torch_batch_std_hot_labels = get_one_hot_reprs(torch_batch_std_labels)
                        batch_cnts = batch_count(batch_std_labels=torch_batch_std_labels, max_rele_grade=max_rele_level, descending=True)

                        self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts))

                    else:
                        self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels))

                #buffer
                #print('Num of q:', len(self.list_torch_Qs))
                if buffer:
                    parent_dir = Path(self.torch_perquery_file).parent
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)

                    pickle_save(self.list_torch_Qs, torch_perquery_file)
        else:
            raise NotImplementedError

        self.hot     = hot
        self.shuffle = shuffle


    def get_default_data_dict(self, data_id):

        if data_id in MSLRWEB:
            scale_data=True
            scaler_id='StandardScaler'
            scaler_level='QUERY'

        elif data_id in ISTELLA_L2R:
            scale_data=True
            scaler_id='StandardScaler'
            scaler_level='QUERY'

        else:
            scale_data=False
            scaler_id=None
            scaler_level=None

        min_docs = 1
        min_rele = -1 # we note that it includes dumb queries that has no relevant documents

        data_dict = dict(data_id=data_id, max_docs='All', min_docs=min_docs, min_rele=min_rele,
                         sample_times_per_q=1, presort=False, binary_rele=False, unknown_as_zero=False,
                         scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        return data_dict

    def get_scaler(self):
        pass
        # todo dataset-level normalization
        #return self.loader.scaler

    def __len__(self):
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        return qid, torch_batch_rankings, torch_batch_std_labels

    def iter_hot(self):
        list_inds = list(range(len(self.list_torch_Qs)))
        if self.shuffle: random.shuffle(list_inds)

        for ind in list_inds:
            qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts = self.list_torch_Qs[ind]
            yield qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts



## ---------------------------------------------------- ##
""" data-preprocess for xgboost/lightgbm oriented runs """

def save_data_xgboost(group_data, output_feature, output_group, group_labels=None, min_docs=None, min_rele=None, data_id=None):
    if len(group_data) == 0:
        return

    if len(group_labels) < min_docs:
        return
    else:
        label_vec = np.asarray(group_labels, dtype=np.int)
        if (label_vec > 0).sum() < min_rele:  # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            return

    output_group.write(str(len(group_data)) + "\n")
    for data in group_data: # data: segments according to split(' ')
        # only include nonzero features
        if data_id == 'MQ2007_super':
            feats = [p for p in data[2:]] # keep zero-valued features due to mismatch error, i.e., the number of features in train data is different from the number of features in validation
        else:
            feats = [p for p in data[2:] if float(p.split(':')[1]) != 0.0]

        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def get_xgboost_buffer(in_file, data_id=None, eval_dict=None):
    """ get absolute paths of data file and group file """

    if data_id in MSLETOR or data_id in MSLRWEB:
        buffer_prefix       = in_file.replace('Fold', 'BufferedFold')
        file_buffered_data  = buffer_prefix.replace('txt', 'data')
        file_buffered_group = buffer_prefix.replace('txt', 'group')

    elif data_id in YAHOO_L2R:
        buffer_prefix       = in_file[:in_file.find('.txt')].replace(data_id.lower() + '.', 'Buffered' + data_id + '/')
        file_buffered_data  = buffer_prefix  + '.data'
        file_buffered_group = buffer_prefix  + '.group'

    else:
        raise NotImplementedError

    if eval_dict is not None and eval_dict['semi_context']:
        mask_ratio = eval_dict['mask_ratio']
        mask_type = eval_dict['mask_type']
        semi_ratio_str = '_'.join(['Semi', mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
        file_buffered_data = file_buffered_data.replace('.data', '_'+semi_ratio_str+'.data')
        file_buffered_group = file_buffered_group.replace('.group', '_'+semi_ratio_str+'.group')
        print('file_buffered_data', file_buffered_data)
        print('file_buffered_group', file_buffered_group)

    return file_buffered_data, file_buffered_group

def load_data_xgboost(in_file, min_docs=None, min_rele=None, data_id=None):
    file_buffered_data, file_buffered_group = get_xgboost_buffer(in_file, data_id=data_id)

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
                save_data_xgboost(group_data, output_feature, output_group, group_labels=group_labels, min_docs=min_docs, min_rele=min_rele, data_id=data_id)
                group_data = []
                group_labels = []

            group = splits[1]
            group_data.append(splits)
            group_labels.append(splits[0])

        save_data_xgboost(group_data, output_feature, output_group, group_labels=group_labels, min_docs=min_docs, min_rele=min_rele, data_id=data_id)

        fi.close()
        output_feature.close()
        output_group.close()

        return file_buffered_data, file_buffered_group


def save_data_xgboost_np(doc_reprs, doc_labels, output_feature, output_group):
    num_docs = doc_reprs.shape[0]

    output_group.write(str(num_docs) + "\n") # group file

    for i in range(num_docs): # per document only include nonzero features
        feats = doc_reprs[i, :].tolist()
        libsvm_feats = []
        for key, val in enumerate(feats):
            if val != 0.0: libsvm_feats.append(':'.join([str(key+1), str(val)]))

        output_feature.write(str(doc_labels[i]) + " " + " ".join(libsvm_feats) + "\n")


def prepare_data_for_lambdaMART(in_file, train=False, min_docs=None, min_rele=None, data_id=None, eval_dict=None):
    """  """
    if not data_id in MSLRWEB: # for non-five-fold datasets
        assert eval_dict['semi_context'] is not True
        return load_data_xgboost(in_file=in_file, min_docs=min_docs, min_rele=min_rele, data_id=data_id)

    else: # query normalization is performed as a default setting for 'MSLRWEB10K', 'MSLRWEB30K'
        if eval_dict is not None and eval_dict['semi_context'] and train:
            file_buffered_data, file_buffered_group = get_xgboost_buffer(in_file, data_id=data_id, eval_dict=eval_dict)

            if os.path.exists(file_buffered_data):
                return file_buffered_data, file_buffered_group
            else:
                parent_dir = Path(file_buffered_data).parent
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

                output_group = open(file_buffered_group, "w")
                output_feature = open(file_buffered_data, "w")

                data_dict = dict(data_id=data_id, max_docs='All', min_docs=min_docs, min_rele=min_rele, presort=False,
                                 scale_data=True, scaler_id='StandardScaler', scaler_level='QUERY')

                loader = L2RDataLoader(file=in_file, data_dict=data_dict, train=train)
                list_Qs = loader.load_data()

                if 'rand_mask_rele' == eval_dict['mask_type']:
                    for qid, doc_reprs, doc_labels in list_Qs:
                        doc_labels = np_random_mask_rele_labels(batch_label=doc_labels, mask_ratio=eval_dict['mask_ratio'], mask_value=0)
                        if doc_labels is not None:
                            save_data_xgboost_np(doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int), output_feature=output_feature, output_group=output_group)
                else:
                    raise NotImplementedError

                output_group.close()
                output_feature.close()

            return file_buffered_data, file_buffered_group

        else:
            file_buffered_data, file_buffered_group = get_xgboost_buffer(in_file, data_id=data_id)

            if os.path.exists(file_buffered_data):
                return file_buffered_data, file_buffered_group
            else:
                parent_dir = Path(file_buffered_data).parent
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

                output_group = open(file_buffered_group, "w")
                output_feature = open(file_buffered_data, "w")

                data_dict = dict(data_id=data_id, max_docs='All', min_docs=min_docs, min_rele=min_rele, presort=False,
                                 scale_data=True, scaler_id='StandardScaler', scaler_level='QUERY')

                loader = L2RDataLoader(file=in_file, data_dict=data_dict, train=train)
                list_Qs = loader.load_data()

                for qid, doc_reprs, doc_labels in list_Qs:
                    save_data_xgboost_np(doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int), output_feature=output_feature, output_group=output_group)

                output_group.close()
                output_feature.close()

            return file_buffered_data, file_buffered_group

## ---------------------------------------------------- ##

## Convert Yahood l2r data into five folds ##

def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def write_per_query(writer, qid, doc_reprs, doc_labels):
    num_docs = doc_reprs.shape[0]

    for i in range(num_docs):  # per document only include nonzero features
        feats = doc_reprs[i, :].tolist()
        libsvm_feats = []
        for key, val in enumerate(feats):
            # libsvm_feats.append(':'.join([str(key + 1), str(val)]))
            if val != 0.0: libsvm_feats.append(':'.join([str(key + 1), str(val)]))

        writer.write(str(int(doc_labels[i])) + " " + 'qid:'+qid+ ' ' + " ".join(libsvm_feats) + "\n")


def write_all(fold_dict, writer_dict, list_Qs):
    for qid, doc_reprs, doc_labels in list_Qs:
        w_ind = fold_dict[qid]
        writer = writer_dict[w_ind]
        write_per_query(writer, qid, doc_reprs, doc_labels)


def convert_yahoo_into_5folds(data_id):

    if data_id == 'Set1':
        dir_data = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_1/'
        dir_output = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_1_5Fold/'

    elif data_id == 'Set2':
        dir_data = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_2/'
        dir_output = '/home/dl-box/WorkBench/Datasets/L2R/Yahoo_L2R_Set_2_5Fold/'
    else:
        raise NotImplementedError

    if data_id in YAHOO_L2R:
        data_prefix = dir_data + data_id.lower() + '.'
        file_train, file_vali, file_test = data_prefix + 'train.txt', data_prefix + 'valid.txt', data_prefix + 'test.txt'
        #data_dict = dict(data_id=data_id, binary_rele=False, min_docs=1, min_rele=1, scale_data=False)
        data_dict = dict(data_id=data_id, max_docs='All', min_docs=1, min_rele=1,
                         binary_rele=False, scale_data=False, scaler_id='StandardScaler', scaler_level='QUERY')
    else:
        raise NotImplementedError



    train_loader = L2RDataLoader(file=file_train, data_dict=data_dict)
    train_list_Qs = train_loader.load_data()

    vali_loader = L2RDataLoader(file=file_vali, data_dict=data_dict)
    vali_list_Qs = vali_loader.load_data()

    test_loader = L2RDataLoader(file=file_test, data_dict=data_dict)
    test_list_Qs = test_loader.load_data()

    #train_dataset = L2RDataset(file_train, sample_rankings_per_q=1, shuffle=True, data_dict=data_dict, hot=False)
    #vali_dataset = L2RDataset(file_vali, sample_rankings_per_q=1, shuffle=True, data_dict=data_dict, hot=False)
    #test_dataset = L2RDataset(file_test, sample_rankings_per_q=1, shuffle=True, data_dict=data_dict, hot=False)

    # list of qid
    list_qids = []
    for qid, _, _ in train_list_Qs:
        list_qids.append(qid)

    for qid, _, _ in vali_list_Qs:
        list_qids.append(qid)

    for qid, _, _ in test_list_Qs:
        list_qids.append(qid)

    random.shuffle(list_qids)
    five_folds = partition(lst=list_qids, n=5)

    fold_dict = {}
    for ind, fold_arr in enumerate(five_folds):
        for qid in fold_arr:
            fold_dict[qid] = ind+1

    writer_dict = {}
    wr1 = open(dir_output + 'S1.txt', "w")
    wr2 = open(dir_output + 'S2.txt', "w")
    wr3 = open(dir_output + 'S3.txt', "w")
    wr4 = open(dir_output + 'S4.txt', "w")
    wr5 = open(dir_output + 'S5.txt', "w")
    writer_dict[1] = wr1
    writer_dict[2] = wr2
    writer_dict[3] = wr3
    writer_dict[4] = wr4
    writer_dict[5] = wr5

    write_all(fold_dict, writer_dict, train_list_Qs)
    write_all(fold_dict, writer_dict, vali_list_Qs)
    write_all(fold_dict, writer_dict, test_list_Qs)

    wr1.flush()
    wr1.close()

    wr2.flush()
    wr2.close()

    wr3.flush()
    wr3.close()

    wr4.flush()
    wr4.close()

    wr5.flush()
    wr5.close()



##--------------Per Query Feature Normalization for Ranklib --------------------##
def get_ranklib_buffer(in_file, data_id=None):
    """ get absolute paths of data file and group file """
    assert data_id in MSLRWEB
    buffer_prefix       = in_file.replace('Fold', 'BufferedFoldLib')
    file_buffered_data  = buffer_prefix.replace('txt', 'data')

    return file_buffered_data

def save_data_ranklib(qid, doc_reprs, doc_labels, output_feature):
    num_docs = doc_reprs.shape[0]

    for i in range(num_docs): # per document only include nonzero features
        feats = doc_reprs[i, :].tolist()
        libsvm_feats = []
        for key, val in enumerate(feats):
            libsvm_feats.append(':'.join([str(key+1), str(val)]))

        #output_feature.write(str(doc_labels[i]) + " " + " ".join(libsvm_feats) + "\n")
        output_feature.write(str(int(doc_labels[i])) + " " + 'qid:' + qid + ' ' + " ".join(libsvm_feats) + "\n")

def prepare_data_for_ranklib(in_file, min_docs=None, min_rele=None, data_id=None):
    """  """
    # query normalization is performed as a default setting for 'MSLRWEB10K', 'MSLRWEB30K'
    file_buffered_data = get_ranklib_buffer(in_file, data_id=data_id)

    parent_dir = Path(file_buffered_data).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    output_feature = open(file_buffered_data, "w")

    data_dict = dict(data_id=data_id, max_docs='All', min_docs=min_docs, min_rele=min_rele,
                     scale_data=True, scaler_id='StandardScaler', scaler_level='QUERY')

    loader = L2RDataLoader(file=in_file, data_dict=data_dict)
    list_Qs = loader.load_data()

    for qid, doc_reprs, doc_labels in list_Qs:
        save_data_ranklib(qid=qid, doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int), output_feature=output_feature)

    output_feature.close()

#todo-as-note
'''
before, the qid is not added!!!!

'''


#######################
# Masking Application #
#######################

torch_zero = torch.FloatTensor([0.0])
def random_mask_all_labels(batch_ranking, batch_label, mask_ratio, mask_value=0, presort=False):
    '''
    Mask the ground-truth labels with the specified ratio as '0'. Meanwhile, re-sort according to the labels if required.
    :param doc_reprs:
    :param doc_labels:
    :param mask_ratio: the ratio of labels to be masked
    :param mask_value:
    :param presort:
    :return:
    '''

    size_ranking = batch_label.size(1)
    num_to_mask = int(size_ranking*mask_ratio)
    mask_ind = np.random.choice(size_ranking, size=num_to_mask, replace=False)

    batch_label[:, mask_ind] = mask_value

    if torch.gt(batch_label, torch_zero).any(): # whether the masked one includes explicit positive labels
        if presort: # re-sort according to the labels if required
            std_labels = torch.squeeze(batch_label)
            sorted_labels, sorted_inds = torch.sort(std_labels, descending=True)

            batch_label = torch.unsqueeze(sorted_labels, dim=0)
            batch_ranking = batch_ranking[:, sorted_inds, :]

        return batch_ranking, batch_label
    else:
        return None


def random_mask_rele_labels(batch_ranking, batch_label=None, mask_ratio=None, mask_value=0, presort=False):
    '''
    Mask the ground-truth labels with the specified ratio as '0'. Meanwhile, re-sort according to the labels if required.
    :param doc_reprs:
    :param doc_labels:
    :param mask_ratio: the ratio of labels to be masked
    :param mask_value:
    :param presort:
    :return:
    '''

    assert 1 == batch_label.size(0) # todo for larger batch-size, need to per-dimension masking

    # squeeze for easy process
    docs, labels = torch.squeeze(batch_ranking, dim=0), torch.squeeze(batch_label)

    all_rele_inds = torch.gt(labels, torch_zero).nonzero()
    num_rele = all_rele_inds.size()[0]

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)

    rele_inds_to_mask = all_rele_inds[mask_inds, 0] # the 0-column corresponds to original rele index since all_rele_inds.size()=(num_rele, 1)

    batch_label[:, rele_inds_to_mask] = mask_value

    if torch.gt(batch_label, torch_zero).any(): # whether the masked one includes explicit positive labels
        if presort: # re-sort according to the labels if required
            std_labels = torch.squeeze(batch_label)
            sorted_labels, sorted_inds = torch.sort(std_labels, descending=True)

            batch_label = torch.unsqueeze(sorted_labels, dim=0)
            batch_ranking = batch_ranking[:, sorted_inds, :]

        return batch_ranking, batch_label
    else:
        # only supports enough rele labels
        raise NotImplementedError


def np_random_mask_labels(batch_label, mask_ratio, mask_value=0):
    '''
    Mask the ground-truth labels with the specified ratio as '0'.
    '''
    size_ranking = len(batch_label)
    num_to_mask = int(size_ranking*mask_ratio)
    mask_ind = np.random.choice(size_ranking, size=num_to_mask, replace=False)

    batch_label[mask_ind] = mask_value

    if np.greater(batch_label, 0.0).any(): # whether the masked one includes explicit positive labels
        return batch_label
    else:
        return None


def np_random_mask_rele_labels(batch_label, mask_ratio, mask_value=0):
    '''
    Mask the ground-truth labels with the specified ratio as '0'.
    '''
    all_rele_inds = np.greater(batch_label, 0).nonzero()[0] # due to one-dimension
    #print('all_rele_inds', all_rele_inds)
    num_rele = all_rele_inds.shape[0]
    #print('num_rele', num_rele)

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)
    #print('mask_inds', mask_inds)

    rele_inds_to_mask = all_rele_inds[mask_inds]
    #print('rele_inds_to_mask', rele_inds_to_mask)

    batch_label[rele_inds_to_mask] = mask_value

    if np.greater(batch_label, 0.0).any(): # whether the masked one includes explicit positive labels
        return batch_label
    else:
        return None



