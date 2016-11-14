"""
cFunction

Date: 09-11-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
import nifty as ny


def cFunction(ws, Kv, F, Fsurf, Fbed, data, hasMatrix = False):
    # Init
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    ftot = 2*fmax+1

    # determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Kv[:, :, n]) > 0):
            bandwidth = max(bandwidth, n)

    # Init Ctd
    nRHS = F.shape[-1]
    cMatrix = np.empty([jmax+1, 2*ftot+2*bandwidth+1, ftot*(kmax)], dtype=complex)
    cCoef = np.zeros([jmax+1, kmax, ftot, nRHS], dtype=complex)

    # build, save and solve the matrices in every water column
    for j in range(0, jmax+1):
        # print str(j) +' of ' + str(jmax)
        # dz vectors
        z = ny.dimensionalAxis(data.slice('grid'), 'z')[j,:,0]
        dz = z[1:]-z[0:-1]      #runs from 0 to kmax-1
        del z
        dz = dz.reshape((len(dz), 1, 1))
        dz_up = np.zeros(dz.shape)
        dz_down = (dz + dz[[0]+range(0,kmax-1)])/2.
        dz_up[:-1, Ellipsis] = (dz[1:]+dz[:-1])/2
        dz_up[-1, Ellipsis] = dz_up[-2,Ellipsis]

        ##### LEFT HAND SIDE #####
        N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        Ws = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        if not hasMatrix:
            # Init
            A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax)], dtype=complex)

            # Build eddy viscosity matrix blocks
            N[:, bandwidth, :] = Kv[j, :, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                N[:, bandwidth+n, :-n] = 0.5*Kv[j, :, n].reshape(kmax+1, 1)*np.ones([1, ftot-n])
                N[:, bandwidth-n, n:] = 0.5*np.conj(Kv[j, :, n]).reshape(kmax+1, 1)*np.ones([1, ftot-n])

            # Build eddy viscosity matrix blocks
            Ws[:, bandwidth, :] = ws[j, :, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                Ws[:, bandwidth+n, :-n] = 0.5*ws[j, :, n].reshape(kmax+1, 1)*np.ones([1, ftot-n])
                Ws[:, bandwidth-n, n:] = 0.5*np.conj(ws[j, :, n]).reshape(kmax+1, 1)*np.ones([1, ftot-n])

            # Insert BC by setting N, Ws at zero at the boundary
            N[0,Ellipsis] = 0
            N[-1,Ellipsis] = 0

            # Build matrix.
            #  NB. can use general numerical schemes as dz < 0
            Ws_avg = 0.5*(Ws[[0] + range(0,kmax-1), Ellipsis] + Ws[range(0,kmax), Ellipsis])
            Ws_avg[0, Ellipsis] = 0
            a = - N[0:-1, :, :]/dz_down                     + Ws[0:-1, :, :]      # for k-1: from 1..kmax
            b =   N[0:-1, :, :]/dz_down + N[1:, :, :]/dz_up - Ws[1:, :, :]        # for k: from 1..kmax
            c =                         - N[1:, :, :]/dz_up                 # for k+1: from 1..kmax
            b[:, bandwidth, :] += 0.5*(np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*dz.reshape((kmax, 1))

            a = np.swapaxes(a, 0, 1)
            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            # Build matrix k=1..kmax
            A[2*ftot:2*ftot+2*bandwidth+1, :-ftot] = (a.reshape(a.shape[0], a.shape[1]*a.shape[2]))[:,ftot:]
            A[ftot:ftot+2*bandwidth+1, :] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
            A[:2*bandwidth+1, ftot:] = (c.reshape(c.shape[0], c.shape[1]*c.shape[2]))[:,:-ftot]

            # save matrix
            cMatrix[j, Ellipsis] = A[Ellipsis]
            bandwidthA = bandwidth+ftot
        else:
            A = Kv[j, Ellipsis]     # if hasMatrix Kv replaces the role of the matrix in this equation
            bandwidthA = (A.shape[0]-1)/2


        ##### RIGHT HAND SIDE #####
        RHS = np.zeros([ftot*(kmax), nRHS], dtype=complex)

        RHS[:, :] = ((F[j, :-1, :, :]+F[j, 1:, :, :])*0.5*dz).reshape(((F.shape[1]-1)*F.shape[2], nRHS))
        RHS[:ftot, :] = -Fsurf[j, 0, :, :]
        RHS[-ftot:, :] += Fbed[j, 0, :, :]

        ##### SOLVE #####
        cstag = solve_banded((bandwidthA, bandwidthA), A, RHS, overwrite_ab=True, overwrite_b=True)
        cCoef[j, :, :, :] = cstag.reshape(kmax, ftot, nRHS)

        c = np.zeros((jmax+1, kmax+1, ftot, nRHS), dtype=complex)
        c[:,1:-1,:, :] = 0.5*(cCoef[:,:-1,:,:]+cCoef[:,1:,:,:])

        # top and bottom
        for j in range(0, jmax+1):
            N[[0, kmax], bandwidth, :] = Kv[j, [0, kmax], 0].reshape(2, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                N[[0, kmax], bandwidth+n, :-n] = 0.5*Kv[j, [0, kmax], n].reshape(2, 1)*np.ones([1, ftot-n])
                N[[0, kmax], bandwidth-n, n:] = 0.5*np.conj(Kv[j, [0, kmax], n]).reshape(2, 1)*np.ones([1, ftot-n])

            Ws[[0, kmax], bandwidth, :] = ws[j, [0, kmax], 0].reshape(2, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                Ws[[0, kmax], bandwidth+n, :-n] = 0.5*ws[j, [0, kmax], n].reshape(2, 1)*np.ones([1, ftot-n])
                Ws[[0, kmax], bandwidth-n, n:] = 0.5*np.conj(ws[j, [0, kmax], n]).reshape(2, 1)*np.ones([1, ftot-n])

            rhs1 = np.swapaxes(-N[0,Ellipsis]/(0.5*dz[0]), 0, 1)
            c[j,0,:, :] = solve_banded((bandwidth, bandwidth), Ws[0,Ellipsis]-N[0,Ellipsis]/(0.5*dz[0]), rhs1*cCoef[j, 0, :, :]+Fsurf[j,0,:,:], overwrite_ab=True, overwrite_b=True)
            rhs1 = np.swapaxes(N[-1,Ellipsis]/(0.5*dz[-1]), 0, 1)
            c[j,-1,:, :] = solve_banded((bandwidth, bandwidth), Ws[-1,Ellipsis]+N[-1,Ellipsis]/(0.5*dz[-1]), rhs1*cCoef[j, -1, :, :]+Fbed[j,0,:,:], overwrite_ab=True, overwrite_b=True)

    return c, cMatrix
