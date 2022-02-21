
import torch
import torch.nn.functional as F

from ptranking.base.list_ranker import ListNeuralRanker

class DASALC(ListNeuralRanker):
	'''
	Zhen Qin, Le Yan, Honglei Zhuang, Yi Tay, Rama Kumar Pasumarthi, Xuanhui Wang, Mike Bendersky, and Marc Najork.
	Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?. In Proceedings of ICLR, 2021.
	'''
	def __init__(self, sf_para_dict=None, gpu=False, device=None):
		super(DASALC, self).__init__(id='DASALC', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
		assert 'listsf' == sf_para_dict['sf_id']

	def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
		'''
		The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
		@param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
		@param kwargs:
		@return:
		'''
		#print('batch_preds', batch_preds.size())
		#print('batch_stds', batch_stds.size())


		# todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
		# c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
		batch_loss = torch.sum(-torch.sum(F.softmax(batch_std_labels, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()

		return batch_loss
