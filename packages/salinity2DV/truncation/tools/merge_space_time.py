import numpy as np
from copy import copy


def merge_space_time(G, H):
    """

    Args:
        G: matrix/tensor with space components of the Galerkin coefficients
        H: matirx/tensor with time components of the Galerkin coefficients

    Returns:

    """
    ndims = len(G.shape)-1
    M = G.shape[1:]
    ftot = H.shape[-1]

    Gshape = [1]
    for i in M:
        Gshape.append(i)
        Gshape.append(1)

    Gr = G.reshape(Gshape)
    Hr= H.reshape([1]+[1, ftot]*ndims)
    GH = (Gr*Hr).reshape([1]+[i*ftot for i in M])

    return GH
