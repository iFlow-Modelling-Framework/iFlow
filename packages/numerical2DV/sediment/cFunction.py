"""
Solve sediment equation: c_t - (ws(x,z,f)*c + Kv(x,z,f)*c_z)_z = F(x,z,f)
with boundary conditions Kv(x,-H,f)*c_z = Fbed(x,-H,f)
                         ws(x,0,f)c + Kv(x,0,f)*c_z = Fsurf(x,0,f)

Date: 09-01-19
Notes: 9-1-19: updated to include uncoupled solver to speed up computations in some cases
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
import nifty as ny


def cFunction(ws, Kv, F, Fsurf, Fbed, data, hasMatrix = False):
    ####################################################################################################################
    # Init
    ####################################################################################################################
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
    cMatrix = np.empty([jmax + 1, 2 * ftot + 2 * bandwidth + 1, ftot * (kmax + 1)], dtype=complex)
    cCoef = np.zeros([jmax + 1, kmax + 1, ftot, nRHS], dtype=complex)
    cCoef_d = np.zeros([jmax + 1, kmax + 1, ftot, nRHS], dtype=complex)

    ####################################################################################################################
    # If bandwidth=0, revert to a simpler solution method
    ####################################################################################################################
    if bandwidth == 0 or (hasMatrix and Kv.shape[1]==3):
        from cFunctionUncoupled import cFunctionUncoupled
        for n in range(0, fmax+1):
            if np.all(F[:, :, fmax+n, :]==0) and np.all(F[:, :, fmax-n, :]==0) and np.all(Fsurf[:, :, fmax+n, :]==0) and np.all(Fsurf[:, :, fmax-n, :]==0) and np.all(Fbed[:, :, fmax+n, :]==0) and np.all(Fbed[:, :, fmax-n, :]==0):
                cCoef_d[:, :, fmax+n, :] = 0
                cMatrix_d = None
            else:
                # F2 = np.zeros(F.shape[:2]+F.shape[-1], dtype=F.dtype)
                # Fsurf2 = np.zeros(Fsurf.shape[:2]+Fsurf.shape[-1], dtype=Fsurf.dtype)
                # Fbed2 = np.zeros(Fbed.shape[:2]+Fbed.shape[-1], dtype=Fbed.dtype)

                F2 = F[:, :, fmax+n, :] + bool(n)*np.conj(F[:, :, fmax-n, :])
                Fsurf2 = Fsurf[:, :, fmax+n, :] + bool(n)*np.conj(Fsurf[:, :, fmax-n, :])
                Fbed2 = Fbed[:, :, fmax+n, :] + bool(n)*np.conj(Fbed[:, :, fmax-n, :])
                cCoef_d[:, :, [fmax+n], :], cMatrix_d = cFunctionUncoupled(ws, Kv, F2, Fsurf2, Fbed2, data, n, hasMatrix=hasMatrix)
        return cCoef_d, cMatrix_d

    ####################################################################################################################
    # Else, continue computation
    #   Build, save and solve the matrices in every water column
    ####################################################################################################################
    for j in range(0, jmax+1):
        z = ny.dimensionalAxis(data.slice('grid'), 'z')[j, :, 0]
        dz = z[1:]-z[:-1]
        del z
        dz = dz.reshape((len(dz), 1, 1))
        dz_down = dz[:-1, Ellipsis]
        dz_up = dz[1:, Ellipsis]
        dz_av = 0.5*(dz_up+dz_down)

        ##### LEFT HAND SIDE #####
        N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        Ws = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        if not hasMatrix:
            # Init
            A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)

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

            # Build matrix.
            #  NB. can use general numerical schemes as dz < 0
            a = - 0.5*(N[0:-2, :, :]+N[1:-1, :, :])/dz_down                                         + Ws[0:-2, :, :]      # for k-1: from 1..kmax-1
            b =   0.5*(N[0:-2, :, :]+N[1:-1, :, :])/dz_down + 0.5*(N[1:-1, :, :]+N[2:, :, :])/dz_up - Ws[1:-1, :, :]        # for k: from 1..kmax-1
            c =                                             - 0.5*(N[1:-1, :, :]+N[2:, :, :])/dz_up                       # for k+1: from 1..kmax
            b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*dz_av.reshape((kmax-1, 1))

            a = np.swapaxes(a, 0, 1)
            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            # Build matrix k=1..kmax-1
            A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] = (a.reshape(a.shape[0], a.shape[1]*a.shape[2]))
            A[ftot:ftot+2*bandwidth+1, ftot:-ftot] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
            A[:2*bandwidth+1, 2*ftot:] = (c.reshape(c.shape[0], c.shape[1]*c.shape[2]))

            ## surface
            b =   Ws[[0],Ellipsis]-N[[0],Ellipsis]/dz[[0], Ellipsis]
            c =   +N[[0],Ellipsis]/dz[[0]]

            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            A[ftot:ftot+2*bandwidth+1, :ftot] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
            A[:2*bandwidth+1, ftot:2*ftot] = (c.reshape(c.shape[0], c.shape[1]*c.shape[2]))

            ## bed
            a = -N[[-1],Ellipsis]/dz[[-1]]
            b = N[[-1],Ellipsis]/dz[[-1]]

            a = np.swapaxes(a, 0, 1)
            b = np.swapaxes(b, 0, 1)

            # Build matrix k=1..kmax-1
            A[2*ftot:2*ftot+2*bandwidth+1, -2*ftot:-ftot] = (a.reshape(a.shape[0], a.shape[1]*a.shape[2]))
            A[ftot:ftot+2*bandwidth+1, -ftot:] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

            # save matrix
            cMatrix[j, Ellipsis] = A[Ellipsis]
            bandwidthA = bandwidth+ftot
        else:
            A = Kv[j, Ellipsis]     # if hasMatrix Kv replaces the role of the matrix in this equation
            bandwidthA = (A.shape[0]-1)/2

        ################################################################################################################
        # Right hand side
        ################################################################################################################
        RHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)

        RHS[ftot:-ftot, :] = (F[j, 1:-1, Ellipsis]*dz_up).reshape(((F.shape[1]-2)*F.shape[2], nRHS))
        RHS[:ftot, :] = Fsurf[j, 0, Ellipsis]
        RHS[-ftot:, :] += Fbed[j, 0, Ellipsis]

        ################################################################################################################
        # Solve
        ################################################################################################################
        cstag = solve_banded((bandwidthA, bandwidthA), A, RHS, overwrite_ab=True, overwrite_b=True)
        cCoef[j, Ellipsis] = cstag.reshape(kmax+1, ftot, nRHS)



    return cCoef, cMatrix
