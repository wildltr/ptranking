
import torch

from ptranking.base.point_ranker import PointNeuralRanker

class DivPointNeuralRanker(PointNeuralRanker):
    '''
    A one-size-fits-all neural ranker.
    Given the documents associated with the same query, this ranker scores each document independently.
    '''
    def __init__(self, id='DivPointNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(DivPointNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)

    def div_forward(self, q_repr, doc_reprs):
        num_docs, num_features = doc_reprs.size()

        latent_cross_reprs = q_repr * doc_reprs
        #TODO is it OK if using expand as boradcasting?
        cat_reprs = torch.cat((q_repr.expand(doc_reprs.size(0), -1), latent_cross_reprs, doc_reprs), 1)

        _pred = self.point_sf(cat_reprs)
        batch_pred = _pred.view(-1, num_docs) # -> [batch_size, num_docs], where batch_size=1 since only one query

        return batch_pred
