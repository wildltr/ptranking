
import os
import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data

from ptranking.utils.bigdata.BigPickle import pickle_save, pickle_load
from ptranking.metric.srd.diversity_metric import get_div_ideal_ranking

'''
Implicit refers to that no subtopic string is used.
Explicit refers to that explicit subtopic strings are used.
'''
TREC_DIV = ['WT_Div_0912_Implicit', 'WT_Div_0912_Explicit']

def get_div_data_meta(data_id=None):
    """ Get the meta-information corresponding to the specified dataset """
    if data_id in TREC_DIV:
        fold_num = 5
        max_label = 1
        num_features = 100
    else:
        raise NotImplementedError

    data_meta = dict(num_features=num_features, fold_num=fold_num, max_label=max_label)
    return data_meta

def to_matrix(perm_docs, q_doc_subtopics):
    num_subtopics = -1
    num_docs = len(perm_docs)

    rele_mat = np.zeros((20, num_docs)) # assuming a temporary maximum-number of subtopics
    for col, doc in enumerate(perm_docs):
        if doc not in q_doc_subtopics:
            continue
        else:
            covered_subtopics = q_doc_subtopics[doc]
            if len(covered_subtopics) == 0:
                continue
            else:
                for subtopic_serial in covered_subtopics:
                    subtopic_id = int(subtopic_serial)
                    row = subtopic_id - 1
                    rele_mat[row, col] = 1.0
                    if subtopic_id > num_subtopics:
                        num_subtopics = subtopic_id

    return rele_mat[0:num_subtopics, :]


class DIVDataset(data.Dataset):
    """
    Loading the specified dataset as data.Dataset, a pytorch format.
    We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    """
    def __init__(self, split_type, list_as_file, data_id=None, data_dict=None, fold_dir=None, presort=True, alpha=0.5,
                 dictQueryRepresentation=None, dictDocumentRepresentation=None, dictQueryPermutaion=None,
                 dictQueryDocumentSubtopics=None, buffer=True, add_noise=False, std_delta=1.0):
        self.presort = presort
        self.add_noise = add_noise
        ''' split-specific settings '''
        self.split_type = split_type
        self.data_id = data_dict['data_id']
        assert presort is True # since it is time-consuming to generate the ideal diversified ranking dynamically.

        if data_dict['data_id'] in TREC_DIV: # supported datasets
            torch_buffer_file = fold_dir.replace('folder', 'Bufferedfolder') + split_type.name
            if self.presort:
                torch_buffer_file = '_'.join([torch_buffer_file, 'presort', '{:,g}'.format(alpha)])
            if self.add_noise:
                torch_buffer_file = '_'.join([torch_buffer_file, 'gaussian', '{:,g}'.format(std_delta)])

            torch_buffer_file += '.torch'

            if os.path.exists(torch_buffer_file):
                print('loading buffered file ...')
                self.list_torch_Qs = pickle_load(torch_buffer_file)
            else:
                self.list_torch_Qs = []
                for qid in list_as_file:
                    np_q_repr = dictQueryRepresentation[str(qid)] # [1, 100]
                    alphaDCG = dictQueryPermutaion[str(qid)]['alphaDCG']
                    q_doc_subtopics = dictQueryDocumentSubtopics[str(qid)]
                    perm_docs = dictQueryPermutaion[str(qid)]['permutation']
                    if self.presort:
                        # print('json-alphaDCG', alphaDCG) # TODO the meaning of json-alphaDCG needs to be confirmed
                        ''' the following comparison shows that the provided permutation of docs is the ideal ranking '''
                        #print('personal-computation for json', alpha_DCG_at_k(sorted_docs=perm_docs, q_doc_subtopics=q_doc_subtopics, k=4, alpha=0.5))
                        perm_docs = get_div_ideal_ranking(pool_docs=perm_docs, q_doc_subtopics=q_doc_subtopics, alpha=alpha)
                        #print('personal-computation for ideal', alpha_DCG_at_k(sorted_docs=perm_docs, q_doc_subtopics=q_doc_subtopics, k=4, alpha=0.5))
                        #print('===')

                    list_doc_reprs = []
                    for doc in perm_docs:
                        list_doc_reprs.append(dictDocumentRepresentation[doc]) # [1, 100]
                    np_doc_reprs = np.vstack(list_doc_reprs) # [permutation_size, 100]

                    q_repr = torch.from_numpy(np_q_repr).type(torch.FloatTensor)
                    doc_reprs = torch.from_numpy(np_doc_reprs).type(torch.FloatTensor)

                    if self.add_noise: # add gaussian noise
                        q_noise = torch.normal(mean=torch.zeros_like(q_repr), std=std_delta)
                        doc_noise = torch.normal(mean=torch.zeros_like(doc_reprs), std=std_delta)
                        q_repr = torch.add(q_repr, q_noise)
                        doc_reprs = torch.add(doc_reprs, doc_noise)

                    np_rele_mat = to_matrix(perm_docs=perm_docs, q_doc_subtopics=q_doc_subtopics)
                    q_doc_rele_mat = torch.from_numpy(np_rele_mat).type(torch.FloatTensor)
                    self.list_torch_Qs.append((qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat))

                #print('Num of q:', len(self.list_torch_Qs))
                if buffer:
                    parent_dir = Path(torch_buffer_file).parent
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)
                    pickle_save(self.list_torch_Qs, torch_buffer_file)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat = self.list_torch_Qs[index]
        return qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat


class RerankDIVDataset(DIVDataset):
    """
    Loading the specified dataset as data.Dataset, a pytorch format.
    We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    """
    def __init__(self, split_type=None, list_as_file=None, data_id=None, data_dict=None, fold_dir=None,
                 presort=True, alpha=0.5, dictQueryRepresentation=None, dictDocumentRepresentation=None,
                 dictQueryPermutaion=None, dictQueryDocumentSubtopics=None, buffer=True,
                 discriminator=None, eval_dict=None, gpu=False, device=None):
        super(RerankDIVDataset, self).__init__(split_type=split_type, list_as_file=list_as_file, data_id=data_id,
                                               data_dict=data_dict, fold_dir=fold_dir, presort=presort, alpha=alpha,
                                               dictQueryRepresentation=dictQueryRepresentation,
                                               dictDocumentRepresentation=dictDocumentRepresentation,
                                               dictQueryPermutaion=dictQueryPermutaion,
                                               dictQueryDocumentSubtopics=dictQueryDocumentSubtopics,
                                               buffer=buffer)
        assert discriminator is not None
        self.discriminator = discriminator
        self.rerank_k = eval_dict['rerank_k']
        self.gpu, self.device = gpu, device

    def __len__(self):
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        if self.discriminator is None: # quick testing
            top_k = 100
            qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat = self.list_torch_Qs[index]
            #print('perm_docs', type(perm_docs))
            #print('q_doc_subtopics', q_doc_subtopics)
            #print('q_doc_rele_mat', q_doc_rele_mat)
            return qid, q_repr, None, doc_reprs[0:top_k, :], alphaDCG, None, q_doc_rele_mat[:, 0:top_k]
        else:
            qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat = self.list_torch_Qs[index]

            top_k_sys_sorted_inds = \
                deploy_1st_stage_div_discriminating(discriminator=self.discriminator, rerank_k=self.rerank_k,
                                                    q_repr=q_repr, doc_reprs=doc_reprs, gpu=self.gpu,device=self.device)

            #TODO
            """
            Though it falls back to the relative order within the original ideal order, the obtained order might not be
            ideal due to the greedy selection where the top-selection matters.
            """
            #print('top_k_sys_sorted_inds', top_k_sys_sorted_inds)
            top_k_relative_ideal_inds, _ = torch.sort(top_k_sys_sorted_inds, descending=False)
            #print('top_k_relative_ideal_inds', top_k_relative_ideal_inds)

            return qid, q_repr, None, doc_reprs[top_k_relative_ideal_inds, :],\
                   alphaDCG, None, q_doc_rele_mat[:, top_k_relative_ideal_inds]


def deploy_1st_stage_div_discriminating(discriminator, rerank_k, q_repr, doc_reprs, gpu, device):
    ''' Perform 1st-stage ranking as a discriminating process. '''

    sys_rele_preds = discriminator.div_predict(q_repr, doc_reprs) # [1, ranking_size]
    if gpu: sys_rele_preds = sys_rele_preds.cpu()

    _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True)  # [1, ranking_size]

    batch_top_k_sys_sorted_inds = sys_sorted_inds[:, 0:rerank_k]

    return torch.squeeze(batch_top_k_sys_sorted_inds)