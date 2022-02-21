
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.list_ranker import ListNeuralRanker
from ptranking.base.point_ranker import PointNeuralRanker


class AdhocNeuralRanker(PointNeuralRanker, ListNeuralRanker):
    '''
    A combination of PointNeuralRanker & PENeuralRanker
    '''
    def __init__(self, id='AdhocNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

        self.sf_para_dict = sf_para_dict
        self.sf_id = sf_para_dict['sf_id']
        assert self.sf_id in ['pointsf', 'listsf']

        self.opt, self.lr = sf_para_dict['opt'], sf_para_dict['lr']
        self.weight_decay = weight_decay

        self.stop_check_freq = 10

        if 'listsf' == self.sf_id:
            self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']

    def init(self):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.init(self)
        elif 'listsf' == self.sf_id:
            ListNeuralRanker.init(self)

    def get_parameters(self):
        if 'pointsf' == self.sf_id:
            return PointNeuralRanker.get_parameters(self)
        elif 'listsf' == self.sf_id:
            return ListNeuralRanker.get_parameters(self)

    def forward(self, batch_q_doc_vectors):
        if 'pointsf' == self.sf_id:
            return PointNeuralRanker.forward(self, batch_q_doc_vectors)
        elif 'listsf' == self.sf_id:
            return ListNeuralRanker.forward(self, batch_q_doc_vectors)

    def eval_mode(self):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.eval_mode(self)
        elif 'listsf' == self.sf_id:
            ListNeuralRanker.eval_mode(self)

    def train_mode(self):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.train_mode(self)
        elif 'listsf' == self.sf_id:
            ListNeuralRanker.train_mode(self)

    # depreated due to moving to Evaluator
    def _validation(self, vali_data=None, vali_metric=None, k=5, presort=False, max_label=None, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
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

    def save(self, dir, name):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.save(self, dir=dir, name=name)
        elif 'listsf' == self.sf_id:
            ListNeuralRanker.save(self, dir=dir, name=name)

    def load(self, file_model, device):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.load(self, file_model=file_model, device=device)
        elif 'listsf' == self.sf_id:
            ListNeuralRanker.load(self, file_model=file_model, device=device)

    def get_tl_af(self):
        if 'pointsf' == self.sf_id:
            PointNeuralRanker.get_tl_af(self)
        elif 'listsf' == self.sf_id:
            self.sf_para_dict[self.sf_para_dict['sf_id']]['AF']
