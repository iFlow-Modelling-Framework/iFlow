"""
bandedMatrixMultiplication

Date: 20-05-16
Authors: Y.M. Dijkstra
"""
import numpy as np


def bandedMatrixMultiplication(A, B, truncate=False):
    """Multiply two matrices A and B that are both in band matrix format.
    Result is a bandmatrix with minimal bandwidth.

    Parameters:
        A, B (2D array) - matrices in band matrix format. It is required that the second dimension of A and B are the same
        truncate (bool, optional) - truncate the result to have a bandwith equal to the maximum bandwith of A and B

    Returns:
        C (2D array) - band matrix with first dimension the minimum bandwith and second dimension equal to that of A and B
    """

    shape = list(A.shape)
    shape[0] = shape[1]
    Afull = np.zeros(shape, dtype=A.dtype)
    Bfull = np.zeros(shape, dtype=B.dtype)
    bandwdA = int((A.shape[0]-1)/2)
    bandwdB = int((B.shape[0]-1)/2)
    size = shape[1]

    # convert both to full matrices
    Afull[(range(0, size), range(0, size))] += A[(bandwdA, slice(None))+(Ellipsis,)]
    for n in range(1, bandwdA+1):
        Afull[(range(0, size-n), range(n, size))] += A[(bandwdA-n, slice(n,None))]
        Afull[(range(n, size), range(0, size-n))] += A[(bandwdA+n, slice(None, -n))]

    Bfull[(range(0, size), range(0, size))] += B[(bandwdB, slice(None))+(Ellipsis,)]
    for n in range(1, bandwdB+1):
        Bfull[(range(0, size-n), range(n, size))] += B[(bandwdB-n, slice(n,None))]
        Bfull[(range(n, size), range(0, size-n))] += B[(bandwdB+n, slice(None, -n))]

    # multiply
    Cfull = np.dot(Afull, Bfull)

    # convert back to band
    bandwd = 0
    for n in np.arange(size-1, -1, -1):
        if np.any(abs(Cfull[(range(n, size), range(0, size-n))]) > 0):
            bandwd = max(bandwd, n)
        if np.any(abs(Cfull[(range(0, size-n), range(n, size))]) > 0):
            bandwd = max(bandwd, n)

    shape[0] = 2*bandwd+1
    C = np.zeros(shape, Cfull.dtype)
    C[(bandwd, slice(None))] = Cfull[(range(0, size), range(0, size))]
    for n in range(1, bandwd+1):
        C[(bandwd-n, slice(n, None))] = Cfull[(range(0, size-n), range(n, size))]
        C[(bandwd+n, slice(None, -n))] = Cfull[(range(n, size), range(0, size-n))]

    if truncate == True:
        newbandwd = max(bandwdA, bandwdB)
        if newbandwd<=bandwd:
            C = C[bandwd-newbandwd:bandwd+newbandwd+1, :]
        else:
            C = np.concatenate((np.zeros((newbandwd-bandwd, C.shape[1])), C, np.zeros((newbandwd-bandwd, C.shape[1]))), axis=0)

    return C


# # More general version for A, B having arbitrary shape and multiplying over dimensions (startdim, startdim+1).
# # Problem is np.dot, which produces the wrong result
# def bandedMatrixMultiplication(A, B, startdim = 0):
#     shape = list(A.shape)
#     shape[startdim] = shape[startdim+1]
#     Afull = np.zeros(shape)
#     Bfull = np.zeros(shape)
#     bandwd = (A.shape[startdim]-1)/2
#     size = shape[startdim+1]
#
#     # convert both to full matrices
#     Afull[(slice(None)]*startdim +[range(0, size), range(0, size)]+[Ellipsis)] += A[(slice(None)]*startdim +[bandwd, slice(None)]+[Ellipsis)]
#     Bfull[(slice(None)]*startdim +[range(0, size), range(0, size)]+[Ellipsis)] += B[(slice(None)]*startdim +[bandwd, slice(None)]+[Ellipsis)]
#     for n in range(1, bandwd+1):
#         Afull[(slice(None)]*startdim +[range(0, size-n), range(n, size)]+[Ellipsis)] += A[(slice(None)]*startdim +[bandwd-n, slice(n,None)]+[Ellipsis)]
#         Afull[(slice(None)]*startdim +[range(n, size), range(0, size-n)]+[Ellipsis)] += A[(slice(None)]*startdim +[bandwd+n, slice(None, -n)]+[Ellipsis)]
#         Bfull[(slice(None)]*startdim +[range(0, size-n), range(n, size)]+[Ellipsis)] += B[(slice(None)]*startdim +[bandwd-n, slice(n,None)]+[Ellipsis)]
#         Bfull[(slice(None)]*startdim +[range(n, size), range(0, size-n)]+[Ellipsis)] += B[(slice(None)]*startdim +[bandwd+n, slice(None, -n)]+[Ellipsis)]
#
#     # multiply
#     #   First set axes in right place for np.dot
#     Afull = np.swapaxes(Afull, startdim+1, len(A.shape)-1)
#     Afull = np.swapaxes(Afull, startdim, len(A.shape)-2)
#     Bfull = np.swapaxes(Bfull, startdim+1, len(A.shape)-1)
#     Bfull = np.swapaxes(Bfull, startdim, len(A.shape)-2)
#
#     #   Then multiply
#     Cfull = np.dot(Afull, Bfull)
#
#     #   Then put axes back
#     Cfull = np.swapaxes(Cfull, startdim, len(A.shape)-2)
#     Cfull = np.swapaxes(Cfull, startdim+1, len(A.shape)-1)
#
#     # convert back to band
#     bandwd = 0
#     for n in np.arange(Cfull.shape[startdim], -1, -1):
#         if np.any(abs(Cfull[(slice(None)]*startdim+[range(n, Cfull.shape[startdim]), range(0, Cfull.shape[startdim]-n)]+[Ellipsis)]) > 0):
#             bandwd = max(bandwd, n)
#
#     shape[startdim] = 2*bandwd+1
#     C = np.zeros(shape)
#     C[(slice(None)]*startdim+ [bandwd, slice(None)]+[Ellipsis)] = Cfull[(slice(None)]*startdim+[range(0, Cfull.shape[startdim]), range(0, Cfull.shape[startdim])]+[Ellipsis)]
#     for n in range(1, bandwd+1):
#         C[(slice(None)]*startdim+ [bandwd-n, slice(n, None)]+[Ellipsis)] = Cfull[(slice(None)]*startdim+[range(0, Cfull.shape[startdim]-n), range(n, Cfull.shape[startdim])]+[Ellipsis)]
#         C[(slice(None)]*startdim+ [bandwd+n, slice(None, -n)]+[Ellipsis)] = Cfull[(slice(None)]*startdim+[range(n, Cfull.shape[startdim]), range(0, Cfull.shape[startdim]-n)]+[Ellipsis)]
#
#     return C