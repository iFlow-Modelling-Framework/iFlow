"""
fft

Date: 22-Feb-16
Authors: Y.M. Dijkstra
"""
import numpy as np


def invfft(u, dim):
    N = u.shape[dim]
    c = np.real(np.fft.ifft(u, axis=dim)*N)
    return c
