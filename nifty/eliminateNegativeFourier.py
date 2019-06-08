"""
eliminateNegativeFourier

Date: 06-Aug-15
Authors: Y.M. Dijkstra
"""
import numpy as np


def eliminateNegativeFourier(uCoef, dimNo):
    fmax = (uCoef.shape[dimNo]-1)/2
    u = np.empty(uCoef.shape[:dimNo]+(fmax+1,)+uCoef.shape[dimNo+1:], dtype=uCoef.dtype)
    u[(slice(None),)*dimNo+(0,)+(Ellipsis,)] = np.real(uCoef[(slice(None),)*dimNo+(fmax,)+(Ellipsis,)])       # remove imaginary part of subtidal component
    u[(slice(None),)*dimNo+(slice(1, None),)+(Ellipsis,)] = uCoef[(slice(None),)*dimNo+(slice(fmax+1, None),)+(Ellipsis,)]
    u[(slice(None),)*dimNo+(slice(1, None),)+(Ellipsis,)] += np.conjugate(uCoef[(slice(None),)*dimNo+(slice(fmax-1, None, -1),)+(Ellipsis,)])

    return u