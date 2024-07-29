"""
This code is distributed under the Apache License Version 2.0. See the LICENSE in the home directory of the project.

Metric calculations from the cute_ranking Python package by Nathan Cooper (ncoop7).
The code was taken from from the cute_ranking package at https://github.com/ncoop57/cute_ranking/blob/main/cute_ranking/core.py in order to avoid downgrading packages.
"""

import numpy as np

def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def r_precision(r):
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])

def precision_at_k(r, k):
    """Calculates the precision at k metric"""
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    val = np.mean(r)
    return max(val, r_precision(r))     # This line was changed to take the best value between precision at k and R-Precision.

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def dcg_at_k(r, k, method=0):
    r = np.asarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

