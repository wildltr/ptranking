
import torch
from itertools import product

from ptranking.base.utils import robust_sigmoid
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_diversification.base.diversity_ranker import DiversityNeuralRanker

def get_approx_ranks(batch_preds, rt=None, device=None, q_doc_rele_mat=None):
    ''' get approximated rank positions: Equation-7 in the paper'''
    batch_pred_diffs = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)  # computing pairwise differences, i.e., Sij or Sxy

    batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), rt, device)  # using {-1.0*} may lead to a poor performance when compared with the above way;

    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

    _q_doc_rele_mat = torch.unsqueeze(q_doc_rele_mat, dim=1)
    batch_q_doc_rele_mat = _q_doc_rele_mat.expand(-1, q_doc_rele_mat.size(1), -1) # duplicate w.r.t. each subtopic -> [num_subtopics, ranking_size, ranking_size]
    prior_cover_cnts = torch.sum(batch_indicators * batch_q_doc_rele_mat, dim=2) - q_doc_rele_mat/2.0 # [num_subtopics, num_docs]

    return batch_hat_pis, prior_cover_cnts

def alphaDCG_as_a_loss(batch_preds=None, q_doc_rele_mat=None, rt=10, device=None, alpha=0.5, top_k=10):
    """
    There are two ways to formulate the loss: (1) using the ideal order; (2) using the predicted order (TBA)
    """
    batch_hat_pis, prior_cover_cnts = get_approx_ranks(batch_preds, rt=rt, device=device, q_doc_rele_mat=q_doc_rele_mat)

    batch_per_subtopic_gains = q_doc_rele_mat * torch.pow((1.0-alpha), prior_cover_cnts) / torch.log2(1.0 + batch_hat_pis)
    batch_global_gains = torch.sum(batch_per_subtopic_gains, dim=1)

    if top_k is None:
        alpha_DCG = torch.sum(batch_global_gains)
    else:
        alpha_DCG = torch.sum(batch_global_gains[0:top_k])

    batch_loss = -alpha_DCG
    return batch_loss


class DALETOR(DiversityNeuralRanker):
    """Description
    Le Yan Zhen Qin Rama Kumar Pasumarthi Xuanhui Wang Mike Bendersky.
    Diversification-Aware Learning to Rank using Distributed Representation.
    The Web Conference 2021 (WWW)
    """

    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(DALETOR, self).__init__(id='DALETOR', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.rt = model_para_dict['rt']
        self.top_k = model_para_dict['top_k']

    def div_custom_loss_function(self, batch_preds, q_doc_rele_mat, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        assert 'presort' in kwargs and kwargs['presort'] is True # aiming for directly optimising alpha-nDCG over top-k documents

        batch_loss = alphaDCG_as_a_loss(batch_preds=batch_preds, q_doc_rele_mat=q_doc_rele_mat,
                                        rt=self.rt, top_k=self.top_k, device=self.device)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


###### Parameter of DALETOR ######

class DALETORParameter(ModelParameter):
    ''' Parameter class for DALETOR '''

    def __init__(self, debug=False, para_json=None):
        super(DALETORParameter, self).__init__(model_id='DALETOR', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for DALETOR. Here rt (reversed T) corresponds to 1/T in paper.
        :return:
        """
        if self.use_json:
            top_k = self.json_dict['top_k'][0]
            rt = self.json_dict['rt'][0]  # corresponds to 1/T in paper
            self.DALETOR_para_dict = dict(model_id=self.model_id, rt=rt, top_k=top_k)
        else:
            self.DALETOR_para_dict = dict(model_id=self.model_id, rt=10., top_k=10)

        return self.DALETOR_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        DALETOR_para_dict = given_para_dict if given_para_dict is not None else self.DALETOR_para_dict

        rt, top_k = DALETOR_para_dict['rt'], DALETOR_para_dict['top_k']

        s1 = ':' if log else '_'
        if top_k is None:
            DALETOR_paras_str = s1.join(['rt', str(rt), 'topk', 'Full'])
        else:
            DALETOR_paras_str = s1.join(['rt', str(rt), 'topk', str(top_k)])
        return DALETOR_paras_str

    def grid_search(self):
        """
        Iterator of parameter settings for ApproxNDCG
        """
        if self.use_json:
            choice_rt = self.json_dict['rt'] # corresponds to 1/T in paper
            choice_topk = self.json_dict['top_k'] # the cutoff value of optimising objective alpha-nDCG@k
        else:
            choice_rt = [10.0] if self.debug else [10.0]  # 1.0, 10.0, 50.0, 100.0
            choice_topk = [10] if self.debug else [10]

        for rt, top_k in product(choice_rt, choice_topk):
            self.DALETOR_para_dict = dict(model_id=self.model_id, rt=rt, top_k=top_k)
            yield self.DALETOR_para_dict