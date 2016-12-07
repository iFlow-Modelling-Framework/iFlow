"""
availabilitySolver
solve the equation
(b1[n]*a^<n>)_t = F - (sum_{m=0}^{n-1} b1[m]a_c^<m> + b2[m]a_cx^<m>)_t

Date: 05-Dec-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny


def availabilitySolver(b1, G, data):
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    ftot = 2*fmax+1
    OMEGA = data.v('OMEGA')

    A = np.zeros((jmax+1, 1, fmax+1, len(G)+1), dtype=complex)
    Rhs = np.zeros((ftot, len(G)+1), dtype=complex)

    for j in range(0, jmax+1):
        ################################################################################################################
        # 1. solve problem without subtidal part; replace this by setting subtidal part of c^<n> to 1
        ################################################################################################################
        # LHS
        B = convertBandToFull(convertVectorToBand(b1[j, 0, :], 0))
        DB = convertBandToFull(convertVectorToBand(np.arange(0, fmax+1)*1j*OMEGA*b1[j, 0, :], 0))
        D = np.eye(ftot, dtype=complex)
        D[range(0, ftot), range(0, ftot)] = (np.arange(-fmax, fmax+1)*1j*OMEGA).reshape((1, ftot))
    
        M = DB + np.dot(B, D)
        M[fmax, :] = 0              # replace subtidal equation
        M[fmax, fmax] = 1.          # ""

        # RHS
        Rhs[fmax, 0] = 1               # Solve null space s.t. a = a_c

        for m, Gm in enumerate(G):
            Rhs[:, m+1] = np.concatenate((np.zeros(fmax), Gm[j, 0, :]), 0)

        # Solve
        A[j, 0, :, :] = ny.eliminateNegativeFourier(np.linalg.solve(M, Rhs), 0)

    return A


def convertVectorToBand(a, dim):
        bandwidth = 0
        fmax = a.shape[dim]-1
        ftot = 2*fmax+1
        for n in np.arange(fmax, -1, -1):
            if np.any(abs(a[[slice(None)]*dim + [n] + [Ellipsis]]) > 0):
                bandwidth = max(bandwidth, n)

        N = np.zeros([2*bandwidth+1, ftot], dtype=complex)
        N[bandwidth, :] = a[[slice(None)]*dim + [0] + [Ellipsis]]*np.ones([1, ftot])
        for n in range(1, bandwidth+1):
            N[bandwidth+n, :-n] = 0.5*a[[slice(None)]*dim + [n] + [Ellipsis]]*np.ones([1, ftot-n])
            N[bandwidth-n, n:] = 0.5*np.conj(a[[slice(None)]*dim + [n] + [Ellipsis]])*np.ones([1, ftot-n])
        return N

def convertBandToFull(A):
    shape = list(A.shape)
    shape[0] = shape[1]
    Afull = np.zeros(shape, dtype=A.dtype)
    bandwdA = (A.shape[0]-1)/2
    size = shape[1]

    # convert both to full matrices
    Afull[[range(0, size), range(0, size)]] += A[[bandwdA, slice(None)]+[Ellipsis]]
    for n in range(1, bandwdA+1):
        Afull[[range(0, size-n), range(n, size)]] += A[[bandwdA-n, slice(n,None)]]
        Afull[[range(n, size), range(0, size-n)]] += A[[bandwdA+n, slice(None, -n)]]
    return Afull