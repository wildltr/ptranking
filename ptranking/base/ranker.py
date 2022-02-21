#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, unique, auto

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_k, torch_ndcg_at_ks, torch_nerr_at_ks, torch_ap_at_ks,\
    torch_precision_at_ks, torch_nerr_at_k, torch_ap_at_k, torch_precision_at_k
from ptranking.metric.srd.diversity_metric import torch_alpha_ndcg_at_k, torch_alpha_ndcg_at_ks, torch_err_ia_at_k,\
    torch_err_ia_at_ks, torch_nerr_ia_at_k, torch_nerr_ia_at_ks


@unique
class LTRFRAME_TYPE(Enum):
    """ Learning-to-rank frame type """
    GBDT = auto()
    Adhoc = auto()
    Adversarial = auto()
    Probabilistic = auto()
    Diversification = auto()
    X = auto() # Covering capabilities of De-biasing, Fairness


class Evaluator():
    """ An interface with in-built evaluation APIs """

    def ndcg_at_k(self, test_data=None, k=10, label_type=LABEL_TYPE.MultiLabel, presort=False, device='cpu'):
        '''
        Compute nDCG@k with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode() # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)

            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings,
                                              batch_ideal_rankings=batch_ideal_rankings,
                                              k=k, label_type=label_type, device=device)

            sum_ndcg_at_k += torch.sum(batch_ndcg_at_k) # due to batch processing

        avg_ndcg_at_k = sum_ndcg_at_k / num_queries
        return avg_ndcg_at_k

    def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, presort=False, device='cpu'):
        '''
        Compute nDCG with multiple cutoff values with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode() # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, label_type=label_type, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks

    def nerr_at_k(self, test_data=None, k=10, label_type=LABEL_TYPE.MultiLabel, max_label=None, presort=False, device='cpu'):
        '''
        Compute the performance using nERR@k
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_nerr_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_nerr_at_k = torch_nerr_at_k(batch_predict_rankings=batch_predict_rankings,
                                              batch_ideal_rankings=batch_ideal_rankings, max_label=max_label,
                                              k=k, label_type=label_type, device=device)
            sum_nerr_at_k += torch.sum(batch_nerr_at_k)  # due to batch processing

        avg_nerr_at_k = sum_nerr_at_k / num_queries
        return avg_nerr_at_k

    def ap_at_k(self, test_data=None, k=10, presort=False, device='cpu'):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_ap_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ap_at_k = torch_ap_at_k(batch_predict_rankings=batch_predict_rankings,
                                          batch_ideal_rankings=batch_ideal_rankings, k=k, device=device)
            sum_ap_at_k += torch.sum(batch_ap_at_k)  # due to batch processing

        avg_ap_at_k = sum_ap_at_k / num_queries
        return avg_ap_at_k

    def p_at_k(self, test_data=None, k=10, device='cpu'):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_p_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

            batch_p_at_k = torch_precision_at_k(batch_predict_rankings=batch_predict_rankings, k=k, device=device)
            sum_p_at_k += torch.sum(batch_p_at_k)  # due to batch processing

        avg_p_at_k = sum_p_at_k / num_queries
        return avg_p_at_k

    def validation(self, vali_data=None, vali_metric=None, k=5, presort=False, max_label=None, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
        if 'nDCG' == vali_metric:
            return self.ndcg_at_k(test_data=vali_data, k=k, label_type=label_type, presort=presort, device=device)
        elif 'nERR' == vali_metric:
            return self.nerr_at_k(test_data=vali_data, k=k, label_type=label_type,
                                  max_label=max_label, presort=presort, device=device)
        elif 'AP' == vali_metric:
            return self.ap_at_k(test_data=vali_data, k=k, presort=presort, device=device)
        elif 'P' == vali_metric:
            return self.p_at_k(test_data=vali_data, k=k, device=device)
        else:
            raise NotImplementedError

    def adhoc_performance_at_ks(self, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, max_label=None,
                                presort=False, device='cpu', need_per_q=False):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        sum_nerr_at_ks = torch.zeros(len(ks))
        sum_ap_at_ks = torch.zeros(len(ks))
        sum_p_at_ks = torch.zeros(len(ks))

        if need_per_q: list_per_q_p, list_per_q_ap, list_per_q_nerr, list_per_q_ndcg = [], [], [], []

        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, label_type=label_type, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))

            batch_nerr_at_ks = torch_nerr_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings, max_label=max_label,
                                                ks=ks, label_type=label_type, device=device)
            sum_nerr_at_ks = torch.add(sum_nerr_at_ks, torch.sum(batch_nerr_at_ks, dim=0))

            batch_ap_at_ks = torch_ap_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings, ks=ks, device=device)
            sum_ap_at_ks = torch.add(sum_ap_at_ks, torch.sum(batch_ap_at_ks, dim=0))

            batch_p_at_ks = torch_precision_at_ks(batch_predict_rankings=batch_predict_rankings, ks=ks, device=device)
            sum_p_at_ks = torch.add(sum_p_at_ks, torch.sum(batch_p_at_ks, dim=0))

            if need_per_q:
                list_per_q_p.append(batch_p_at_ks)
                list_per_q_ap.append(batch_ap_at_ks)
                list_per_q_nerr.append(batch_nerr_at_ks)
                list_per_q_ndcg.append(batch_ndcg_at_ks)

            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        avg_nerr_at_ks = sum_nerr_at_ks / num_queries
        avg_ap_at_ks = sum_ap_at_ks / num_queries
        avg_p_at_ks = sum_p_at_ks / num_queries

        if need_per_q:
            return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks,\
                   list_per_q_ndcg, list_per_q_nerr, list_per_q_ap, list_per_q_p
        else:
            return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks





    def alpha_ndcg_at_k(self, test_data=None, k=5, device='cpu'):
        '''
        Compute alpha-nDCG@k with the given data
        @param test_data:
        @param k:
        @return:
        '''
        self.eval_mode()
        assert test_data.presort is True

        cnt = torch.zeros(1)
        sum_alpha_nDCG_at_k = torch.zeros(1)
        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in test_data:
            if torch.sum(q_doc_rele_mat) < 1.0: continue # since this instance provides no learning signal
            if q_doc_rele_mat.size(1) < k: continue      # skip the query if the number of associated documents is smaller than k

            if self.gpu: q_repr, doc_reprs = q_repr.to(self.device), doc_reprs.to(self.device)
            sys_rele_preds = self.div_predict(q_repr, doc_reprs) # [1, ranking_size]
            if self.gpu: sys_rele_preds = sys_rele_preds.cpu()

            _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True) # [1, ranking_size]

            ''' the output result will have the same shape as index '''
            sys_q_doc_rele_mat = torch.gather(q_doc_rele_mat, dim=1, index=sys_sorted_inds.expand(q_doc_rele_mat.size(0), -1))
            ''' the alternative way for gather() '''
            #sys_q_doc_rele_mat = q_doc_rele_mat[:, torch.squeeze(sys_sorted_inds, dim=0)]

            ideal_q_doc_rele_mat = q_doc_rele_mat # under the assumption of presort

            alpha_nDCG_at_k = torch_alpha_ndcg_at_k(sys_q_doc_rele_mat=sys_q_doc_rele_mat, k=k, alpha=0.5,
                                                    device=device, ideal_q_doc_rele_mat=ideal_q_doc_rele_mat)

            sum_alpha_nDCG_at_k += alpha_nDCG_at_k  # default batch_size=1 due to testing data
            cnt += 1

        avg_alpha_nDCG_at_k = sum_alpha_nDCG_at_k / cnt
        return avg_alpha_nDCG_at_k

    def alpha_ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], device='cpu'):
        '''
        Compute alpha-nDCG with multiple cutoff values with the given data
        There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
        and at least one relevant document. Or there will be errors.
        '''
        self.eval_mode()
        assert test_data.presort is True

        cnt = torch.zeros(1)
        sum_alpha_nDCG_at_ks = torch.zeros(len(ks))
        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in test_data:
            if torch.sum(q_doc_rele_mat) < 1.0: continue
            if self.gpu: q_repr, doc_reprs = q_repr.to(self.device), doc_reprs.to(self.device)
            sys_rele_preds = self.div_predict(q_repr, doc_reprs)
            if self.gpu: sys_rele_preds = sys_rele_preds.cpu()

            _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True)

            sys_q_doc_rele_mat = torch.gather(q_doc_rele_mat, dim=1, index=sys_sorted_inds.expand(q_doc_rele_mat.size(0), -1))

            ideal_q_doc_rele_mat = q_doc_rele_mat # under the assumption of presort

            alpha_nDCG_at_ks = torch_alpha_ndcg_at_ks(sys_q_doc_rele_mat=sys_q_doc_rele_mat, ks=ks, alpha=0.5,
                                                      ideal_q_doc_rele_mat=ideal_q_doc_rele_mat, device=device)

            sum_alpha_nDCG_at_ks = torch.add(sum_alpha_nDCG_at_ks, torch.squeeze(alpha_nDCG_at_ks, dim=0))
            cnt += 1

        avg_alpha_nDCG_at_ks = sum_alpha_nDCG_at_ks / cnt
        return avg_alpha_nDCG_at_ks

    def err_ia_at_k(self, test_data=None, k=5, max_label=None, device='cpu'):
        '''
        Compute ERR-IA@k with the given data
        @param test_data:
        @param k:
        @return:
        '''
        self.eval_mode()

        cnt = torch.zeros(1)
        sum_err_ia_at_k = torch.zeros(1)
        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in test_data:
            if torch.sum(q_doc_rele_mat) < 1.0: continue # since this instance provides no learning signal
            if q_doc_rele_mat.size(1) < k: continue # skip query if the number of associated documents is smaller than k

            if self.gpu: q_repr, doc_reprs = q_repr.to(self.device), doc_reprs.to(self.device)
            sys_rele_preds = self.div_predict(q_repr, doc_reprs) # [1, ranking_size]
            if self.gpu: sys_rele_preds = sys_rele_preds.cpu()

            _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True) # [1, ranking_size]

            ''' the output result will have the same shape as index '''
            sys_q_doc_rele_mat = \
                torch.gather(q_doc_rele_mat, dim=1, index=sys_sorted_inds.expand(q_doc_rele_mat.size(0), -1))
            ''' the alternative way for gather() '''
            #sys_q_doc_rele_mat = q_doc_rele_mat[:, torch.squeeze(sys_sorted_inds, dim=0)]

            err_ia_at_k = \
                torch_err_ia_at_k(sorted_q_doc_rele_mat=sys_q_doc_rele_mat, max_label=max_label, k=k, device=device)

            sum_err_ia_at_k += err_ia_at_k  # default batch_size=1 due to testing data
            cnt += 1

        avg_err_ia_at_k = sum_err_ia_at_k / cnt
        return avg_err_ia_at_k

    def nerr_ia_at_k(self, test_data=None, k=5, max_label=None, device='cpu'):
        '''
        Compute nERR-IA@k with the given data
        @param test_data:
        @param k:
        @return:
        '''
        self.eval_mode()
        assert test_data.presort is True

        cnt = torch.zeros(1)
        sum_nerr_ia_at_k = torch.zeros(1)
        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in test_data:
            if torch.sum(q_doc_rele_mat) < 1.0: continue # since this instance provides no learning signal
            if q_doc_rele_mat.size(1) < k: continue      # skip the query if the number of associated documents is smaller than k

            if self.gpu: q_repr, doc_reprs = q_repr.to(self.device), doc_reprs.to(self.device)
            sys_rele_preds = self.div_predict(q_repr, doc_reprs) # [1, ranking_size]
            if self.gpu: sys_rele_preds = sys_rele_preds.cpu()

            _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True) # [1, ranking_size]

            ''' the output result will have the same shape as index '''
            sys_q_doc_rele_mat = torch.gather(q_doc_rele_mat, dim=1, index=sys_sorted_inds.expand(q_doc_rele_mat.size(0), -1))
            ''' the alternative way for gather() '''
            #sys_q_doc_rele_mat = q_doc_rele_mat[:, torch.squeeze(sys_sorted_inds, dim=0)]

            ideal_q_doc_rele_mat = q_doc_rele_mat # under the assumption of presort

            nerr_ia_at_k = torch_nerr_ia_at_k(sys_q_doc_rele_mat=sys_q_doc_rele_mat, max_label=max_label,
                                              ideal_q_doc_rele_mat=ideal_q_doc_rele_mat, k=k, device=device)

            sum_nerr_ia_at_k += nerr_ia_at_k  # default batch_size=1 due to testing data
            cnt += 1

        avg_nerr_ia_at_k = sum_nerr_ia_at_k / cnt
        return avg_nerr_ia_at_k

    def srd_performance_at_ks(self, test_data=None, ks=[1, 5, 10], max_label=None, device='cpu',
                              generate_div_run=False, dir=None, fold_k=None, need_per_q_andcg=False):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode
        assert test_data.presort is True

        num_queries = 0
        sum_andcg_at_ks = torch.zeros(len(ks), device=device)
        sum_err_ia_at_ks = torch.zeros(len(ks), device=device)
        sum_nerr_ia_at_ks = torch.zeros(len(ks), device=device)

        if need_per_q_andcg: list_per_q_andcg = []
        if generate_div_run: fold_run = open(dir + '/fold_run.txt', 'w')

        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in test_data:
            if not torch.sum(q_doc_rele_mat) > 0: continue # skip the case of no positive labels
            if self.gpu: q_repr, doc_reprs = q_repr.to(self.device), doc_reprs.to(self.device)
            sys_rele_preds = self.div_predict(q_repr, doc_reprs)
            if self.gpu: sys_rele_preds = sys_rele_preds.cpu()

            _, sys_sorted_inds = torch.sort(sys_rele_preds, dim=1, descending=True)

            if generate_div_run:
                np_sys_sorted_inds = torch.squeeze(sys_sorted_inds).data.numpy()
                num_docs = len(perm_docs)
                for i in range(num_docs):
                    doc = perm_docs[np_sys_sorted_inds[i]]
                    fold_run.write(' '.join([str(qid), 'Q0', doc, str(i + 1), str(num_docs - i), 'Fold'+str(fold_k)+"\n"]))
                    fold_run.flush()

            sys_q_doc_rele_mat = \
                torch.gather(q_doc_rele_mat, dim=1, index=sys_sorted_inds.expand(q_doc_rele_mat.size(0), -1))

            ideal_q_doc_rele_mat = q_doc_rele_mat # under the assumption of presort

            andcg_at_ks = torch_alpha_ndcg_at_ks(sys_q_doc_rele_mat=sys_q_doc_rele_mat, ks=ks, alpha=0.5, device=device,
                                                 ideal_q_doc_rele_mat=ideal_q_doc_rele_mat)
            err_ia_at_ks = torch_err_ia_at_ks(sorted_q_doc_rele_mat=sys_q_doc_rele_mat,
                                              max_label=max_label, ks=ks, device=device)
            nerr_ia_at_ks = torch_nerr_ia_at_ks(sys_q_doc_rele_mat=sys_q_doc_rele_mat,
                                                ideal_q_doc_rele_mat=ideal_q_doc_rele_mat,
                                                max_label=max_label, ks=ks, device=device)

            if need_per_q_andcg: list_per_q_andcg.append(andcg_at_ks)
            sum_andcg_at_ks = torch.add(sum_andcg_at_ks, torch.squeeze(andcg_at_ks, dim=0))
            sum_err_ia_at_ks = torch.add(sum_err_ia_at_ks, err_ia_at_ks)
            sum_nerr_ia_at_ks = torch.add(sum_nerr_ia_at_ks, nerr_ia_at_ks)
            num_queries += 1

        if generate_div_run:
            fold_run.flush()
            fold_run.close()

        avg_andcg_at_ks = sum_andcg_at_ks / num_queries
        avg_err_ia_at_ks = sum_err_ia_at_ks / num_queries
        avg_nerr_ia_at_ks = sum_nerr_ia_at_ks / num_queries

        if need_per_q_andcg:
            return avg_andcg_at_ks, avg_err_ia_at_ks, avg_nerr_ia_at_ks, list_per_q_andcg
        else:
            return avg_andcg_at_ks, avg_err_ia_at_ks, avg_nerr_ia_at_ks



class NeuralRanker(Evaluator):
    """
    NeuralRanker is a class that represents a general learning-to-rank model.
    The component of neural scoring function is usually a common setting for fair comparison.
    Different learning-to-rank models inherit NeuralRanker, but differ in custom_loss_function, which corresponds to a particular loss function.
    """
    def __init__(self, id='AbsRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

        self.sf_para_dict = sf_para_dict
        self.sf_id = sf_para_dict['sf_id']

        self.opt, self.lr = sf_para_dict['opt'], sf_para_dict['lr']
        self.weight_decay = weight_decay

        self.stop_check_freq = 10

    def init(self):
        '''
        Initialise necessary components, such as the scoring function and the optimizer.
        In most cases, it is also used as reset_parameters()
        '''
        pass



    def get_parameters(self):
        '''
        Get the trainable parameters of the scoring function.
        '''
        pass

    def config_optimizer(self):
        '''
        Configure the optimizer correspondingly.
        '''
        if 'Adam' == self.opt:
            self.optimizer = optim.Adam(self.get_parameters(), lr = self.lr, weight_decay = self.weight_decay)
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif 'Adagrad' == self.opt:
            self.optimizer = optim.Adagrad(self.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

    def eval_mode(self):
        pass

    def train_mode(self):
        pass

    def save(self, dir, name):
        pass

    def load(self, file_model, **kwargs):
        pass

    def uniform_eval_setting(self, **kwargs):
        """
        Update the evaluation setting dynamically if necessary in order to achieve consistent evaluation and training.
        E.g., the validation metric should be consistent with the metric being directly optimized.
        @param kwargs:
        """
        pass

    def stop_training(self, batch_preds):
        '''
        Stop training if the predictions are all zeros or include nan value(s)
        '''

        #if torch.nonzero(preds).size(0) <= 0: # todo-as-note: 'preds.byte().any()' seems wrong operation w.r.t. gpu
        if torch.nonzero(batch_preds, as_tuple=False).size(0) <= 0: # due to the UserWarning: This overload of nonzero is deprecated:
            print('All zero error.\n')
            return True

        if torch.isnan(batch_preds).any():
            print('Including NaN error.')
            return True

        return False

    ''' >>>> Adhoc Ranking >>>> '''

    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()

        assert 'label_type' in kwargs and 'presort' in kwargs
        label_type, presort = kwargs['label_type'], kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss, stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)

            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()

        epoch_loss = epoch_loss/num_queries
        return epoch_loss, stop_training

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        stop_training = False
        batch_preds = self.forward(batch_q_doc_vectors)

        if 'epoch_k' in kwargs and kwargs['epoch_k'] % self.stop_check_freq == 0:
            stop_training = self.stop_training(batch_preds)

        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs), stop_training

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        The loss function to be customized
        @param batch_preds: [batch, ranking_size] each row represents the predicted relevance values for documents associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs:
        @return:
        '''
        pass

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function. It can differ from predict(), since the output of forward() is not limited to relevance prediction.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        pass

    def predict(self, batch_q_doc_vectors):
        '''
        The relevance prediction.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_preds = self.forward(batch_q_doc_vectors)
        return batch_preds

    ''' <<<< Adhoc Ranking <<<< '''

    ''' >>>> Diversified Ranking >>>> '''

    def div_train(self, train_data, epoch_k=None):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()

        presort = train_data.presort
        epoch_loss = torch.tensor([0.0], device=self.device)
        for qid, q_repr, perm_docs, doc_reprs, alphaDCG, q_doc_subtopics, q_doc_rele_mat in train_data:
            if torch.sum(q_doc_rele_mat) < 1.0: continue # skip instances that provide no training signal
            if self.gpu: q_repr, doc_reprs, q_doc_rele_mat = q_repr.to(self.device), doc_reprs.to(self.device), q_doc_rele_mat.to(self.device)

            batch_loss, stop_training = self.div_train_op(q_repr, doc_reprs, q_doc_rele_mat, qid=qid, alphaDCG=alphaDCG, epoch_k=epoch_k, presort=presort)

            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()

        len = train_data.__len__()
        epoch_loss = epoch_loss/len
        return epoch_loss, stop_training

    def div_train_op(self, q_repr, doc_reprs, q_doc_rele_mat, **kwargs):
        '''
        Per-query training based on the documents that are associated with the same query.
        '''
        stop_training = False
        batch_pred = self.div_forward(q_repr, doc_reprs)

        if 'epoch_k' in kwargs and kwargs['epoch_k'] % self.stop_check_freq == 0:
            stop_training = self.stop_training(batch_pred)

        return self.div_custom_loss_function(batch_pred, q_doc_rele_mat, **kwargs), stop_training

    def div_custom_loss_function(self, batch_preds, q_doc_rele_mat, **kwargs):
        '''
        The per-query loss function to be customized
        :param batch_preds:
        :param batch_label:
        :param kwargs:
        :return:
        '''
        pass

    def div_forward(self, q_repr, doc_reprs):
        '''
        Forward pass through the scoring function.
        It can differ from predict(), since the output of forward() is not limited to relevance prediction.
        @param batch_ranking:
        @return:
        '''
        pass

    def div_predict(self, q_repr, doc_reprs):
        '''
        The relevance prediction. In the context of adhoc ranking, the shape is interpreted as:
        '''
        batch_pred = self.div_forward(q_repr, doc_reprs)
        return batch_pred

    ''' <<<< Diversified Ranking <<<< '''
