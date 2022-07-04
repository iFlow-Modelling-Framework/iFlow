"""
sclosureFunction

Date: 11-Jan-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny


def sclosureFunction(Q, AK, F, Fopen, Fclosed, data):
    jmax = data.v('grid', 'maxIndex', 'x')
    nRHS = F.shape[-1]

    A = np.zeros((jmax+1, jmax+1))
    ##### LEFT-HAND SIDE #####
    x = ny.dimensionalAxis(data.slice('grid'), 'x')[:, 0, 0]
    dx = x[1:]-x[:-1]
    A[range(0, jmax), range(0, jmax)] = +AK[:-1]/dx    # main diagonal
    A[range(0, jmax), range(1, jmax+1)] = Q[:-1]-AK[1:]/dx    # upper diagonal

    # BC closed end
    A[-1, -1] = -AK[-1]
    A[-1, 0] = AK[0]

    ##### RIGHT-HAND SIDE #####
    sRHS = np.zeros((jmax+1, nRHS))
    sRHS[:-1, :] = F[:-1, :]
    sRHS[-1, :] = Q[0]*Fopen - Q[-1]*Fclosed + ny.integrate(F, 'x', 0, jmax, data.slice('grid'))

    ##### SOLVE #####
    Sx = np.zeros((jmax+1, 1, 1, nRHS))
    Sx[:, 0, 0, :] = np.linalg.solve(A, sRHS)

    ##### INTEGRATE ######
    # integrate from back to front to make sure that the landward BC is guaranteed
    S = ny.integrate(Sx, 'x', jmax, range(0, jmax+1), data.slice('grid')) + Fclosed

    ##### CORRECTION ######
    # Integration errors may cause the solution to not satisfy the boundary conditions.
    # By the definition of integration here, the landward boundary condition is satisfied, but the seaward condition is not
    # Apply a correction that scales the salinity profile
    #for i in range(0, nRHS):
    #    if S[0, 0, 0, i]-Fclosed[0, i] != 0:
    #        S[:, 0, 0, i] = (Fopen[0, i]-Fclosed[0, i])/(S[0, 0, 0, i]-Fclosed[0, i])*(S[:, 0, 0, i]-Fclosed[0, i])+Fclosed[0, i]

    return S[:, 0, 0, :], Sx[:, 0, 0, :]

