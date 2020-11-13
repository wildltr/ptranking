import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import math
import numpy as np


def shuffle_ties(vec, descending=True):
    '''
    namely, randomly permuate ties
    :param vec:
    :param descending: the sorting order w.r.t. the input vec
    :return:
    '''
    if len(vec.size()) > 1:
        raise NotImplementedError
    else:
        length = vec.size()[0]
        perm = torch.randperm(length)
        shuffled_vec, _ = torch.sort(vec[perm], descending=descending)
        return shuffled_vec

def arg_shuffle_ties(vec, descending=True):
    ''' the same as shuffle_ties, but return the corresponding indice '''
    if len(vec.size()) > 1:
        raise NotImplementedError
    else:
        length = vec.size()[0]
        perm = torch.randperm(length)
        sorted_shuffled_vec_inds = torch.argsort(vec[perm], descending=descending)
        shuffle_ties_inds = perm[sorted_shuffled_vec_inds]
        return shuffle_ties_inds

def test_shuffle_ties():
    np_arr = np.asarray([0.8, 0.8, 0.7, 0.7, 0.5, 0.5])
    vec = torch.from_numpy(np_arr)

    shuffle_ties_inds = arg_shuffle_ties(vec=vec)
    print(shuffle_ties_inds)

    print(vec[shuffle_ties_inds])

    print(shuffle_ties(vec=vec))


def plackett_luce_sampling(probs, softmaxed=False):
    '''
    sample a ltr_adhoc based on the Plackett-Luce model
    :param vec: a vector of values, the higher, the more possible the corresponding entry will be sampled
    :return: the indice of the corresponding ltr_adhoc
    '''
    if softmaxed:
        inds = torch.multinomial(probs, probs.size()[0], replacement=False)
    else:
        probs = F.softmax(probs, dim=0)
        inds = torch.multinomial(probs, probs.size()[0], replacement=False)

    return inds


def soft_rank_sampling(loc, covariance_matrix=None, inds_style=True, descending=True):
    '''
    :param loc: mean of the distribution
    :param covariance_matrix: positive-definite covariance matrix
    :param inds_style: true means the indice leading to the ltr_adhoc
    :return:
    '''
    m = MultivariateNormal(loc, covariance_matrix)
    vals = m.sample()
    if inds_style:
        sorted_inds = torch.argsort(vals, descending=descending)
        return sorted_inds
    else:
        vals



class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x




class Power(torch.autograd.Function):
    '''
    d(b^x)/dx = (b^x)ln(b), now torch.pow() allows taking derivatives
    '''

    @staticmethod
    def forward(ctx, input, base):
        input_exped = torch.pow(base, input)
        ctx.save_for_backward(input_exped, base)
        return input_exped

    @staticmethod
    def backward(ctx, grad_output):
        input_exped, base = ctx.saved_tensors
        grad = torch.mul(input_exped, torch.log(base))
        return grad, None # todo ??? chain rule:


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

from scipy.integrate import quad
class Gaussian_Integral_0_Inf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, MU, sigma, gpu):
        '''
        :param ctx:
        :param mu:
        :param sigma: a float value
        :return:
        '''
        #print('MU', MU)
        tmp_MU = MU.detach()
        tmp_MU = tmp_MU.view(1, -1)
        np_MU = tmp_MU.cpu().numpy() if gpu else tmp_MU.numpy()
        #print('np_MU', np_MU)
        np_integrated_probs = [quad(lambda y: ONEOVERSQRT2PI * np.exp(-0.5 * (y-mu/sigma) ** 2) / sigma, 0, np.inf)[0] for mu in np.ravel(np_MU)]
        #print('np_integrated_probs', np_integrated_probs)
        integrated_probs = torch.as_tensor(np_integrated_probs, dtype=MU.dtype)
        integrated_probs = integrated_probs.view(MU.size())
        #print('integrated_probs', integrated_probs)
        ctx.save_for_backward(MU, torch.tensor([sigma]))
        return integrated_probs

    @staticmethod
    def backward(ctx, grad_output):
        mu, sigma = ctx.saved_tensors
        probs = ONEOVERSQRT2PI * torch.exp(-0.5 * (-mu/sigma) ** 2) / sigma  # point gaussian probabilities given mu and sigma
        # chain rule
        bk_output = grad_output * probs
        return bk_output, None, None


def sinkhorn_2D(x, num_iter=5):
    '''
    Sinkhorn (1964) showed that if X is a positive square matrix, there exist positive diagonal matrices D1 and D2 such that D1XD2 is doubly stochastic.
    The method of proof is based on an iterative procedure of alternatively normalizing the rows and columns of X.
    :param x: the given positive square matrix
    '''
    for i in range(num_iter):
        x = torch.div(x, torch.sum(x, dim=1, keepdim=True))
        x = torch.div(x, torch.sum(x, dim=0, keepdim=True))
    return x


def logsumexp_(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def sinkhorn_batch_(batch_x, num_iter=20, eps=1e-10, tau=0.05):
    '''
    Temperature (tau) -controlled Sinkhorn layer.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column normalization.
    -To ensure positivity, the effective input to sinkhorn has to be exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 1967
    :param batch_x: a batch of square matrices, the restriction of 'positive' w.r.t. batch_x is not needed, since the exp() is deployed here.
    :param num_iter: number of sinkhorn iterations (in practice, as little as 20 iterations are needed to achieve decent convergence for N~100)
    :return: A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are converted to 3D tensors with batch_size equals to 1)
    '''
    if tau is not None:
        batch_x = batch_x/tau   # as tau approaches zero(positive), the result is more like a permutation matrix
    for _ in range(num_iter):
        batch_x = batch_x - logsumexp(batch_x, dim=2, keepdim=True) # row normalirzation
        batch_x = batch_x - logsumexp(batch_x, dim=1, keepdim=True) # column normalization

        if (batch_x != batch_x).sum() > 0 or (batch_x != batch_x).sum() > 0 or batch_x.max() > 1e9 or batch_x.max() > 1e9:  # u!=u is a test for NaN...
            break

    return torch.exp(batch_x) + eps # add a small offset 'eps' in order to avoid numerical errors due to exp()



def swish(x):
    return x * torch.sigmoid(x)


class SWISH(nn.Module):
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, x):
        res = x * torch.sigmoid(x)
        return res



class ReLU_K(nn.Hardtanh):
    r"""Applies the element-wise function :math:`\text{ReLU_K}(x) = \min(\max(0,x), k)`

    Args:   inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::
    m = ReLU_K(k=1)
    input = torch.randn(2)
    output = m(input)
    """

    def __init__(self, inplace=False, k=None):
        assert k > 0
        super(ReLU_K, self).__init__(0, k, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

def encode_RK(k):
    assert k is not None
    return ''.join(['R', str(k)])

def is_RK(rk_str):
    return rk_str[0] == 'R' and rk_str[-1].isdigit()

def decode_RK(rk_str):
    return rk_str[0], int(rk_str[1:])

def ini_ReLU_K(rk_str):
    _, k = decode_RK(rk_str=rk_str)
    return ReLU_K(k=k)

def test_reluk():
    input = torch.randn(20)

    m6 = nn.ReLU6()
    output6 = m6(input)
    print(output6)

    mk = ReLU_K(k=1)
    outputk = mk(input)
    print(outputk)

    rk_str = 'R1'
    if is_RK(rk_str):
        mmkk = ini_ReLU_K(rk_str)
        oot = mmkk(input)
        print(oot)

'''
    def ini_std_pl(self, alpha=1.0):
        #stable softmax
        self.prediction_std = self.std_label_vec * alpha
        self.max_v = max(self.prediction_std)
        self.std_point_exps_truncated = np.exp(self.prediction_std-self.max_v)
        self.std_point_pros = self.std_point_exps_truncated/sum(self.std_point_exps_truncated)
'''

def pl_normalize(batch_scores=None):
    '''
    Normalization based on the 'Plackett_Luce' model
    :param batch_scores: [batch, ranking_size]
    :return: the i-th entry represents the probability of being ranked at the i-th position
    '''
    m, _ = torch.max(batch_scores, dim=1, keepdim=True)  # for higher stability
    y = batch_scores - m
    y = torch.exp(y)
    y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dim=1), dim=1), dim=1)  # row-wise cumulative sum, from tail to head
    batch_pros = torch.div(y, y_cumsum_t2h)

    return batch_pros



if __name__ == '__main__':
    #1
    #test_reluk()

    #2
    test_shuffle_ties()
