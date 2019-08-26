"""
Uncoupled solver, used by CFunction

Date: 19-12-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
import nifty as ny
from copy import copy


def cFunctionUncoupled(ws, Kv, F, Fsurf, Fbed, data, n, hasMatrix=False):
    ####################################################################################################################
    # Init
    ####################################################################################################################
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')


    # Init Ctd
    nRHS = F.shape[-1]
    cCoef = np.zeros((jmax+1, kmax+1, 1, nRHS), dtype=complex)
    cMatrix = np.zeros((jmax+1, 3, kmax+1), dtype=complex)

    try:
        z = ny.dimensionalAxis(data.slice('grid'), 'z')[:,:,0]
    except:
        z = -np.linspace(0, 1, kmax+1).reshape((1, kmax+1))*np.ones((jmax+1, 1))

    ####################################################################################################################
    # build, save and solve the matrices in every water column
    ####################################################################################################################
    for j in range(0, jmax+1):
        dz = z[j, 1:]-z[j, :-1]
        dz = dz
        dz_down = dz[:-1]
        dz_up = dz[1:]
        dz_av = 0.5*(dz_up+dz_down)

        if not hasMatrix:
            A = np.zeros((3, kmax+1), dtype=complex)

            ##### LEFT HAND SIDE #####
            # Build matrix.
            #  NB. can use general numerical schemes as dz < 0
            a = -0.5*(Kv[j, 0:-2, 0]+Kv[j, 1:-1, 0])/dz_down                                           + ws[j, 0:-2, 0]      # for k-1: from 1..kmax-1
            b = 0.5*(Kv[j, 0:-2, 0]+Kv[j, 1:-1, 0])/dz_down + 0.5*(Kv[j, 2:, 0]+Kv[j, 1:-1, 0])/dz_up - ws[j, 1:-1, 0]        # for k: from 1..kmax-1
            c =                                             - 0.5*(Kv[j, 2:, 0]+Kv[j, 1:-1, 0])/dz_up                       # for k+1: from 1..kmax
            # add inertia later

            # Build matrix k=1..kmax-1
            A[2, :-2] = a
            A[1, 1:-1] = b
            A[0, 2:] = c

            ## surface
            b =   ws[j, 0, 0]-Kv[j, 0, 0]/dz[0]
            c =   +Kv[j, 0, 0]/dz[0]

            A[1, 0] = b
            A[0, 1] = c

            ## bed
            a = -Kv[j, 0, 0]/dz[-1]
            b = Kv[j, 0, 0]/dz[-1]

            A[2, -2] = a
            A[1, -1] = b

            # save matrix
            cMatrix[j, Ellipsis] = copy(A[Ellipsis])
        else:
            A = copy(Kv[j, Ellipsis])

        A[1, 1:-1] += n*1j*OMEGA*dz_av

        ################################################################################################################
        # Right hand side
        ################################################################################################################
        RHS = np.zeros([kmax+1, nRHS], dtype=complex)

        RHS[1:-1, :] = F[j, 1:-1, :]*dz_up.reshape((kmax-1, 1))
        RHS[0, :] = Fsurf[j, 0, :]
        RHS[-1, :] += Fbed[j, 0, :]

        ################################################################################################################
        # Solve
        ################################################################################################################
        cstag = solve_banded((1, 1), A, RHS, overwrite_ab=True, overwrite_b=True)
        cCoef[j, :, 0, :] = cstag.reshape(kmax+1, nRHS)

    return cCoef, cMatrix
