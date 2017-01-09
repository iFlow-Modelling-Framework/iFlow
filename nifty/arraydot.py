"""
arraydot

Date: 03-01-17
Authors: Y.M. Dijkstra
"""
import numpy as np


def arraydot(a, b):
    '''Dot product over the last dimension of a and last/second-to-last dimension of b. If a and b are maximum 2D,
    this is the same as np.dot. For larger dimensional input, the other dimensions are matched.

    Examples:
    1) If a.shape = (k, l, m, n) and b.shape(k, l, n, j) then
    arraydot(a, b).shape = (k, l, m, j)

    2) If a.shape = (k, l, m, n) and b.shape(k, l, n) then
    arraydot(a, b).shape = (k, l, m)

    3) If a.shape = (k, l, m, n) and b.shape(k, 1, n, j) then       (i.e. second dimension of b is copied l times)
    arraydot(a, b).shape = (k, l, m, j)
    '''
    if len(a.shape) < 3:
        c = np.dot(a, b)
    elif len(a.shape) == 3:
        jb = np.inf
        if b.shape[0] == 1:
            jb = 0
        c = np.zeros(a.shape[:2]+b.shape[2:], dtype=a.dtype)
        for j in range(0, a.shape[0]):
            c[j, Ellipsis] = np.dot(a[j, Ellipsis], b[min(j, jb), Ellipsis])
    elif len(a.shape) == 4:
        jb = np.inf
        kb = np.inf
        if b.shape[0] == 1:
            jb = 0
        if b.shape[1] == 1:
            kb = 0
        c = np.zeros(a.shape[:3]+b.shape[3:], dtype=a.dtype)
        for j in range(0, a.shape[0]):
            for k in range(0, a.shape[1]):
                c[j, k, Ellipsis] = np.dot(a[j, k, Ellipsis], b[min(j, jb), min(k, kb), Ellipsis])
    else:
        raise NotImplementedError('arraydot not implemented for first argument with more than 4 dimensions')

    return c