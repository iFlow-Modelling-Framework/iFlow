"""
cFunction

Date: 19-12-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
import nifty as ny


def pFunction(n, ws, Kv, F, Fsurf, Fbed, data, hasMatrix = False):
    ####################################################################################################################
    # Init
    ####################################################################################################################
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')

    # Init Ctd
    nRHS = F.shape[-1]
    cMatrix = np.empty([jmax+1, 3, kmax+1], dtype=complex)
    cCoef = np.zeros([jmax+1, kmax+1, 2*fmax+1, nRHS], dtype=complex)

    ####################################################################################################################
    # build, save and solve the matrices in every water column
    ####################################################################################################################
    try:
        z = ny.dimensionalAxis(data.slice('grid'), 'z')[:,:,0]
    except:
        z = -np.linspace(0, 1, kmax+1).reshape((1, kmax+1))
    dz = z[:, 1:]-z[:, :-1]
    del z
    dz = dz
    dz_down = dz[:, :-1]
    dz_up = dz[:, 1:]

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # Init
        A = np.zeros([jmax+1, 3, kmax+1], dtype=complex)
        # Build matrix.
        #  NB. can use general numerical schemes as dz < 0
        a = - 0.5*(Kv[:, 0:-2]+Kv[:, 1:-1])/dz_down                                     + ws[:, 0:-2]      # for k-1: from 1..kmax-1
        b =   0.5*(Kv[:, 0:-2]+Kv[:, 1:-1])/dz_down + 0.5*(Kv[:, 1:-1]+Kv[:, 2:])/dz_up - ws[:, 1:-1]        # for k: from 1..kmax-1
        c =                                         - 0.5*(Kv[:, 1:-1]+Kv[:, 2:])/dz_up                       # for k+1: from 1..kmax
        b += 0.5*n*1j*OMEGA

        # Build matrix k=1..kmax-1
        A[:, 0, :-2] = c
        A[:, 1, 1:-1] = b
        A[:, 2, 2:] = a

        ## surface
        b =   ws[:, 0]-Kv[:, 0]/dz[:, 0]
        c =   Kv[:, 0]/dz[:, 0]

        A[:, 1, 0] = b
        A[:, 0, 1] = c

        ## bed
        a = Kv[:, -1]/dz[:, -1]
        b = ws[:, -1]-Kv[:, -1]/dz[:, -1]

        # Build matrix k=1..kmax-1
        A[:, 2, -2] = a
        A[:, 1, -1] = b

        ################################################################################################################
        # Right hand side
        ################################################################################################################
        RHS = np.zeros([jmax+1, kmax+1, nRHS], dtype=complex)

        RHS[:, 1:-1, :] = (F[:, 1:-1, :]*dz_up.reshape((jmax+1, kmax-1, 1)))
        RHS[:, 0, :] = Fsurf[:, 0, :]
        RHS[:, -1, :] += -Fbed[:, 0, :]

        ################################################################################################################
        # Solve
        ################################################################################################################
        for j in range(0, jmax+1):
            cstag = solve_banded((1, 1), A[j, Ellipsis], RHS[j, Ellipsis], overwrite_ab=True, overwrite_b=True)
            cCoef[j, :, fmax+n, :] = cstag.reshape(kmax+1, nRHS)



    return cCoef, cMatrix
