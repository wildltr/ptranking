#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

import numpy as np


def np_shuffle_ties(vec, descending=True):
    '''
    namely, randomly permuate ties
    :param vec:
    :param descending: the sorting order w.r.t. the input vec
    :return:
    '''
    if len(vec.shape) > 1:
        raise NotImplementedError
    else:
        length = vec.shape[0]
        perm = np.random.permutation(length)
        shuffled_vec = sorted(vec[perm], reverse=descending)
        return shuffled_vec

def np_arg_shuffle_ties(vec, descending=True):
    ''' the same as np_shuffle_ties, but return the corresponding indice '''
    if len(vec.shape) > 1:
        raise NotImplementedError
    else:
        length = vec.shape[0]
        perm = np.random.permutation(length)
        if descending:
            sorted_shuffled_vec_inds = np.argsort(-vec[perm])
        else:
            sorted_shuffled_vec_inds = np.argsort(vec[perm])

        shuffle_ties_inds = perm[sorted_shuffled_vec_inds]
        return shuffle_ties_inds

def test_np_shuffle_ties():
    np_arr = np.asarray([0.8, 0.8, 0.7, 0.7, 0.5, 0.5])

    print(np_shuffle_ties(vec=np_arr, descending=True))
    inds = np_arg_shuffle_ties(np_arr)
    print(inds)
    print(np_arr[inds])

def np_softmax(xs):
    ys = xs - np.max(xs)
    exps = np.exp(ys)
    return exps/exps.sum(axis=0)

def np_plackett_luce_sampling(items, probs, softmaxed=False):
    '''
    sample a ltr_adhoc based on the Plackett-Luce model
    :param vec: a vector of values, the higher, the more possible the corresponding entry will be sampled
    :return: the indice of the corresponding ltr_adhoc
    '''
    if softmaxed:
        ranking = np.random.choice(items, size=len(probs), p=probs, replace=False)
    else:
        probs = np_softmax(probs)
        ranking = np.random.choice(items, size=len(probs), p=probs, replace=False)

    return ranking

def test_pl_sampling():
    pass

if __name__ == '__main__':
    #1
    test_np_shuffle_ties()