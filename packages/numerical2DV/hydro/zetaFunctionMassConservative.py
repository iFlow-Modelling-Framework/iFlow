"""
zetaFunctionMassConservative

Solve a function
q_t + 1/B (M q_x + F)_x = 0     for x in (0, L)
    q(0) = Fopen
    F(L)q_x(L) = Fclosed

Solved in a mass conservative way, i.e. the solution q_x is always
such that M q_x + F = int q_t/B


Date: 04-Aug-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
from nifty import integrate


def zetaFunctionMassConservative(M, F, Fopen, Fclosed, data, hasMatrix = False):
    # Init
    jmax = data.v('grid', 'maxIndex', 'x')
    fmax = data.v('grid', 'maxIndex', 'f')
    OMEGA = data.v('OMEGA')
    ftot = 2*fmax+1

    x = data.v('grid', 'axis', 'x')
    dx = (x[1:]-x[:-1])*data.v('L')

    B = data.v('B', x=x[:-1]+.5*(x[1:]-x[:-1]))                 # widths between two grid points

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # determine bandwith
        bandwidth = 0
        for n in np.arange(ftot-1, -1, -1):
            for m in np.arange(ftot-1, -1, -1):
                if any(abs(M[:, 0, n, m]) > 0):
                    bandwidth = max(bandwidth, n-m, -n+m)

        # init
        A = np.zeros([2*ftot+2*bandwidth+1, ftot*(jmax+1)], dtype=complex)
        Mdiag = np.zeros([jmax+1, 2*bandwidth+1, ftot], dtype=complex)
        Mdiag[:, bandwidth, :] = np.diagonal(M[:, 0, :, :], 0, 1, 2)
        for n in range(1, bandwidth+1):
            Mdiag[:, bandwidth+n, :-n] = np.diagonal(M[:, 0, :, :], -n, 1, 2)
            Mdiag[:, bandwidth-n, n:] = np.diagonal(M[:, 0, :, :], n, 1, 2)

        # dx vectors
        Bdx_down = (dx[:jmax-1]*B[:jmax-1]).reshape(jmax-1, 1, 1)
        Bdx_up = (dx[1:jmax]*B[1:jmax]).reshape(jmax-1, 1, 1)
        dx_av = 0.5*(dx[1:jmax]+dx[:jmax-1]).reshape(jmax-1, 1)

        # Build matrix
        a = Mdiag[:-2, :, :]/Bdx_down
        b = -Mdiag[1:-1, :, :]/Bdx_down-Mdiag[1:-1, :, :]/Bdx_up
        c = Mdiag[2:, :, :]/Bdx_up
        b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape(1, ftot)*dx_av

        a = np.swapaxes(a, 0, 1)
        b = np.swapaxes(b, 0, 1)
        c = np.swapaxes(c, 0, 1)

        A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] += a.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[2*fmax+1:2*fmax+2*bandwidth+2, ftot:-ftot] += b.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[:2*bandwidth+1, 2*ftot:] += c.reshape(a.shape[0], a.shape[1]*a.shape[2])

        # Boundary conditions
        #   Sea (j=0)
        b = -Mdiag[0, :, :]/(dx[0]*B[0])
        b[bandwidth, :] += 0.5*np.arange(-fmax, ftot-fmax)*1j*OMEGA*dx[0]
        c = Mdiag[1, :, :]/(dx[0]*B[0])

        A[2*fmax+1:2*fmax+2*bandwidth+2, :ftot] += b
        A[:2*bandwidth+1, ftot:2*ftot] += c

        #   Weir (j=jmax)
        A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += Mdiag[-1, :, :]
        bandwidth = bandwidth+ftot
    else:
        A = M
        bandwidth = (A.shape[0]-1)/2

    ##### RIGHT HAND SIDE #####
    nRHS = F.shape[-1]
    zRHS = np.zeros([ftot*(jmax+1), nRHS], dtype=complex)
    for i in range(0, nRHS):
        # forcing on open boundary
        zRHS[:ftot, i] = -np.arange(-fmax, fmax+1)*1j*OMEGA*Fopen[0, 0, :, i]

        # forcing on closed boundary
        zRHS[-ftot:, i] = Fclosed[0, 0, :, i]

        # internal forcing
        bdx_up = (dx[1:]*B[1:]).reshape(jmax-1, 1)
        bdx_down = (dx[:-1]*B[:-1]).reshape(jmax-1, 1)
        bdx0 = dx[0]*B[0]
        zRHS[ftot:-ftot, i] = -((F[2:, 0, :, i] - F[1:-1, 0, :, i])/bdx_up - (F[1:-1, 0, :, i] - F[:-2, 0, :, i])/bdx_down).reshape((jmax-1)*ftot)
        zRHS[:ftot, i] += -(F[1, 0, :, i] - F[0, 0, :, i])/bdx0

    ##### SOLVE #####
    zetax = solve_banded((bandwidth, bandwidth), A, zRHS, overwrite_ab=False, overwrite_b=True)
    zetax = zetax.reshape(jmax+1, 1, ftot, nRHS)

    # integrate to zeta
    zeta = integrate(zetax.reshape((jmax+1, 1, 1, ftot, nRHS)), 'x', 0, np.arange(0, jmax+1), data.slice('grid'), INTMETHOD='INTERPOLSIMPSON')
    zeta = zeta.reshape((jmax+1, 1, ftot, nRHS))

    zetaOpenBoundary = Fopen[0, None, 0, None, :, :]*np.ones([jmax+1, 1, ftot, nRHS], dtype=complex)
    zeta += zetaOpenBoundary

    return zeta, zetax, A

