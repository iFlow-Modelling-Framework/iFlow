"""
zetaFunctionMassConservative

Solve a function
q_t + 1/B (M q_x )_x = -F/B     for x in (0, L)
    q(0) = Fopen
    F(L)q_x(L) = Fclosed

For tidal component n (n=0, residual; n=1: M2; n=2: M4)

Date: 23-01-17
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
from nifty import integrate


def zetaFunctionUncoupled(n, M, F, Fopen, Fclosed, data, hasMatrix = False, reverseBC = False):
    # Init
    jmax = data.v('grid', 'maxIndex', 'x')
    OMEGA = data.v('OMEGA')

    x = data.v('grid', 'axis', 'x')
    dx = (x[1:]-x[:-1])*data.v('L')

    B = data.v('B', x=x[:-1]+.5*(x[1:]-x[:-1]))                 # widths between two grid points

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # init
        A = np.zeros([3, jmax+1], dtype=complex)
        # dx vectors
        Bdx_down = (dx[:jmax-1]*B[:jmax-1])
        Bdx_up = (dx[1:jmax]*B[1:jmax])
        dx_av = 0.5*(dx[1:jmax]+dx[:jmax-1])

        # Build matrix
        a = M[:-2]/Bdx_down
        b = -M[1:-1]/Bdx_down-M[1:-1]/Bdx_up
        c = M[2:]/Bdx_up
        b += n*1j*OMEGA*dx_av

        A[2, :-2] += a
        A[1, 1:-1] += b
        A[0, 2:] += c

        # Boundary conditions
        #   Option 1: force zeta at x=0 and zetax at x=L
        if not reverseBC:
            #   Sea (j=0)
            b = -M[0]/(dx[0]*B[0])
            b += 0.5*n*1j*OMEGA*dx[0]
            c = M[1]/(dx[0]*B[0])

            A[1, 0] += b
            A[0, 1] += c

            #   Weir (j=jmax)
            A[1, -1] += M[-1]
        #   Option 2: force zeta at x=L and zetax at x=0
        else:
            #   Sea (j=0)
            A[1, 0] += M[0]

            #   Weir (j=jmax)
            b = M[-1]/(dx[-1]*B[-1])
            b += 0.5*n*1j*OMEGA*dx[-1]
            a = -M[-2]/(dx[-1]*B[-1])

            A[2, -2] += a
            A[1, -1] += b

    else:
        A = M

    ##### RIGHT HAND SIDE #####
    nRHS = F.shape[-1]
    zRHS = np.zeros([jmax+1, nRHS], dtype=complex)

    if not reverseBC:
        # forcing on open boundary
        zRHS[0, :] = -n*1j*OMEGA*Fopen[0, :]

        # forcing on closed boundary
        zRHS[-1, :] = Fclosed[0, :]

        # internal forcing
        FdivBav = 0.5*(F[1:, :]+F[:-1, :])/B.reshape((jmax, 1))
        zRHS[1:-1, :] = -FdivBav[1:]+FdivBav[:-1]
        zRHS[0, :] += -FdivBav[0]
    else:
        # forcing on open boundary
        zRHS[0, :] = Fopen[0, :]

        # forcing on closed boundary
        zRHS[-1, :] = -n*1j*OMEGA*Fclosed[0, :]

        # internal forcing
        FdivBav = 0.5*(F[1:, :]+F[:-1, :])/B.reshape((jmax, 1))
        zRHS[1:-1, :] = -FdivBav[1:]+FdivBav[:-1]
        zRHS[-1, :] += -FdivBav[-1]

    ##### SOLVE #####
    zetax = solve_banded((1, 1), A, zRHS, overwrite_ab=False, overwrite_b=True)
    zetax = zetax.reshape(jmax+1, nRHS)

    # integrate to zeta
    if not reverseBC:
        zeta = integrate(zetax.reshape((jmax+1, 1, 1, nRHS)), 'x', 0, np.arange(0, jmax+1), data.slice('grid'), INTMETHOD='TRAPEZOIDAL')
        zeta = zeta.reshape((jmax+1, nRHS))

        zeta += Fopen[0, :]
    else:
        zeta = integrate(zetax.reshape((jmax+1, 1, 1, nRHS)), 'x', jmax  , np.arange(0, jmax+1), data.slice('grid'), INTMETHOD='TRAPEZOIDAL')
        zeta = zeta.reshape((jmax+1, nRHS))

        zeta += Fclosed[0, :]

    return zeta, zetax, A

