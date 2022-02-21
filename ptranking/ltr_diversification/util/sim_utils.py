
import torch

def batch_cosine_similarity(x1, x2=None, eps=1e-8):
    '''
    :param x1: [batch_size, num_docs, num_features]
    :param x2: the same shape or None
    :param eps:
    :return:
    '''
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=2, keepdim=True)
    #print('w1', w1.size(), '\n', w1)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=2, keepdim=True)
    batch_numerator = torch.bmm(x1, x2.permute(0, 2, 1))
    batch_denominator = torch.bmm(w1, w2.permute(0, 2, 1)).clamp(min=eps)
    return batch_numerator/batch_denominator