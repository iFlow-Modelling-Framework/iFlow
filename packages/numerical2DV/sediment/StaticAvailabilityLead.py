"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
import step as st
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from numpy.linalg import svd


class StaticAvailabilityLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module StaticAvailability')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        OMEGA = self.input.v('OMEGA')

        beta = 1
        c0 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        c0_int = ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
        D = np.eye(ftot, dtype=complex)
        D[range(0, ftot), range(0, ftot)] = (np.arange(-fmax, fmax+1)*1j*OMEGA).reshape((1, ftot))
        a = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        for j in range(0, jmax+1):
            C = self.convertBandToFull(self.convertVectorToBand(c0_int[j,0,:], 0))
            B = beta*np.eye(ftot, dtype=complex) + C
            DB = self.convertBandToFull(self.convertVectorToBand(np.arange(0, fmax+1)*1j*OMEGA*c0_int[j,0,:], 0))

            M = DB + np.dot(B, D)
            M[fmax, :] = 0
            M[fmax, fmax] = 1.
            Rhs = np.zeros(fmax*2+1)
            Rhs[fmax] = 1
            a[j, 0, :] = (ny.eliminateNegativeFourier(np.linalg.solve(M, Rhs), 0)).reshape(fmax+1)


        a = a/a[:,:,[0]]
        d = {}
        d['a0'] = a
        import matplotlib.pyplot as plt
        import step as st
        st.configure()
        plt.figure(1, figsize=(1,2))
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        # plt.plot(x,np.abs(a[:, 0, 2]/a[:, 0, 0]))
        plt.plot(x,np.abs(a[:, 0, 0]))
        plt.plot(x,np.abs(a[:, 0, 2]))
        st.show()

        return d

    def convertVectorToBand(self, a, dim):
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

    def convertBandToFull(self, A):
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

    def nullspace(self, A, atol=1e-13, rtol=0):
        """Compute an approximate basis for the nullspace of A.

        The algorithm used by this function is based on the singular value
        decomposition of `A`.

        A : ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
        atol : float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
        rtol : float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.

        If both `atol` and `rtol` are positive, the combined tolerance is the
        maximum of the two; that is::
            tol = max(atol, rtol * smax)
        Singular values smaller than `tol` are considered to be zero.

        ns : ndarray
            If `A` is an array with shape (m, k), then `ns` will be an array
            with shape (k, n), where n is the estimated dimension of the
            nullspace of `A`.  The columns of `ns` are a basis for the
            nullspace; each element in numpy.dot(A, ns) will be approximately
            zero.

        http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
        """

        A = np.atleast_2d(A)
        u, s, vh = svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns