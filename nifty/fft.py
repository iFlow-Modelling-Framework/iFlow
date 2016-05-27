"""
fft

Date: 22-Feb-16
Authors: Y.M. Dijkstra
"""
import numpy as np


def fft(u, dim):
    N = u.shape[dim]
    c = np.fft.rfft(u, axis=dim)
    c[[slice(None)]*dim+[0]+[Ellipsis]] = c[[slice(None)]*dim+[0]+[Ellipsis]]/N
    c[[slice(None)]*dim+[slice(1, None)]+[Ellipsis]] = 2.*c[[slice(None)]*dim+[slice(1, None)]+[Ellipsis]]/N
    return c
