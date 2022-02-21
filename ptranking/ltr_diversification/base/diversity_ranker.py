
from ptranking.ltr_diversification.base.div_list_ranker import DivListNeuralRanker
from ptranking.ltr_diversification.base.div_point_ranker import DivPointNeuralRanker

class DiversityNeuralRanker(DivPointNeuralRanker, DivListNeuralRanker):
    '''
    A combination of PointNeuralRanker & PENeuralRanker
    '''
    def __init__(self, id='DiversityNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

        self.sf_para_dict = sf_para_dict
        self.sf_id = sf_para_dict['sf_id']
        assert self.sf_id in ['pointsf', 'listsf']

        self.opt, self.lr = sf_para_dict['opt'], sf_para_dict['lr']
        self.weight_decay = weight_decay

        self.stop_check_freq = 10

        if 'pointsf' == self.sf_id: # corresponding to the concatenation operation, i.e., q_repr + doc_repr + latent_cross
            self.sf_para_dict[self.sf_id]['num_features'] *= 3
        elif 'listsf' == self.sf_id:
            self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']


    def init(self):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.init(self)
        elif self.sf_id.startswith('listsf'):
            DivListNeuralRanker.init(self)

    def get_parameters(self):
        if self.sf_id.startswith('pointsf'):
            return DivPointNeuralRanker.get_parameters(self)
        elif self.sf_id.startswith('listsf'):
            return DivListNeuralRanker.get_parameters(self)

    def div_forward(self, q_repr, doc_reprs):
        if self.sf_id.startswith('pointsf'):
            return DivPointNeuralRanker.div_forward(self, q_repr, doc_reprs)
        elif self.sf_id.startswith('listsf'):
            return DivListNeuralRanker.div_forward(self, q_repr, doc_reprs)

    def eval_mode(self):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.eval_mode(self)
        elif self.sf_id.startswith('listsf'):
            DivListNeuralRanker.eval_mode(self)

    def div_validation(self, vali_data=None, vali_metric=None, k=5, max_label=None, device='cpu'):
        if 'aNDCG' == vali_metric:
            return self.alpha_ndcg_at_k(test_data=vali_data, k=k, device=device)
        elif 'nERR-IA' == vali_metric: # nERR-IA is better choice than ERR-IA with no normalization
            return self.nerr_ia_at_k(test_data=vali_data, k=k, max_label=max_label, device=device)
        else:
            raise NotImplementedError

    def train_mode(self):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.train_mode(self)
        elif self.sf_id.startswith('listsf'):
            DivListNeuralRanker.train_mode(self)

    def save(self, dir, name):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.save(self, dir=dir, name=name)
        elif self.sf_id.startswith('listsf'):
            DivListNeuralRanker.save(self, dir=dir, name=name)

    def load(self, file_model, **kwargs):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.load(self, file_model=file_model, **kwargs)
        elif self.sf_id.startswith('listsf'):
            DivListNeuralRanker.load(self, file_model=file_model, **kwargs)

    def get_tl_af(self):
        if self.sf_id.startswith('pointsf'):
            DivPointNeuralRanker.get_tl_af(self)
        elif self.sf_id.startswith('listsf'):
            self.sf_para_dict[self.sf_para_dict['sf_id']]['AF']
