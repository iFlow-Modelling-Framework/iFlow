"""
cFunction
Uses integral condition; still has problems with the prediction of the phase

Date: 19-12-2016
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
    cMatrix = np.empty([jmax+1, 2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
    cCoef = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
    cBed = np.zeros([jmax+1, kmax+1, ftot, fmax+1], dtype=complex)
    cSurf = np.zeros([jmax+1, kmax+1, ftot, fmax+1], dtype=complex)

    ####################################################################################################################
    # build, save and solve the matrices in every water column
    ####################################################################################################################
    for j in range(0, jmax+1):
        try:
            z = ny.dimensionalAxis(data.slice('grid'), 'z')[j,:,0]
        except:
            z = -np.linspace(0, 1, kmax+1)
        dz = z[1:]-z[:-1]
        del z
        dz = dz.reshape((len(dz), 1, 1))
        dz_down = dz[:-1,Ellipsis]
        dz_up = dz[1:, Ellipsis]

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
            b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*dz_up.reshape((kmax-1, 1))

            a = np.swapaxes(a, 0, 1)
            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            # Build matrix k=1..kmax-1
            A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] = (a.reshape(a.shape[0], a.shape[1]*a.shape[2]))
            A[ftot:ftot+2*bandwidth+1, ftot:-ftot] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
            A[:2*bandwidth+1, 2*ftot:] = (c.reshape(c.shape[0], c.shape[1]*c.shape[2]))

            ## surface
            b =   Ws[[0],Ellipsis]-N[[0],Ellipsis]/dz[[0], Ellipsis]
            c =   N[[0],Ellipsis]/dz[[0]]

            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            A[ftot:ftot+2*bandwidth+1, :ftot] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
            A[:2*bandwidth+1, ftot:2*ftot] = (c.reshape(c.shape[0], c.shape[1]*c.shape[2]))

            ## bed
            a = N[[-1],Ellipsis]/dz[[-1]]
            b = -N[[-1],Ellipsis]/dz[[-1]]

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
        # Solve for F
        ################################################################################################################
        nRHS_F = [i for i in range(0, nRHS) if (F[j, :, :, i]!=0).any() ]
        if len(nRHS_F) > 0:

            RHS = np.zeros([ftot*(kmax+1), len(nRHS_F)], dtype=complex)
            RHS[ftot:-ftot, :] = (np.rollaxis(F[j, 1:-1, :, nRHS_F], 0, 3)*dz_up).reshape(((F.shape[1]-2)*F.shape[2], len(nRHS_F)))

            csol = solve_banded((bandwidthA, bandwidthA), A, RHS, overwrite_ab=True, overwrite_b=True)
            cCoef[j, :, :, nRHS_F] = np.rollaxis(csol.reshape(kmax+1, ftot, len(nRHS_F)), -1, 0)

        ################################################################################################################
        # Solve for Fbed
        ################################################################################################################
        RHS = np.zeros([ftot*(kmax+1), fmax+1], dtype=complex)
        RHS[-ftot+fmax:, :] = np.eye(fmax+1)

        ## bed
        b = np.zeros(N[[1], :, :].shape)
        b[:, bandwidth, :] = np.ones((1, ftot))
        b = np.swapaxes(b, 0, 1)
        A[2*ftot:2*ftot+2*bandwidth+1, -2*ftot:-ftot] = 0
        A[ftot:ftot+2*bandwidth+1, -ftot:] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        csol = solve_banded((bandwidthA, bandwidthA), A, RHS, overwrite_ab=True, overwrite_b=True)
        cBed[j, :, :, :] = csol.reshape(kmax+1, ftot, fmax+1)

        ################################################################################################################
        # Solve for FSurf
        ################################################################################################################
        RHS = np.zeros([ftot*(kmax+1), fmax+1], dtype=complex)
        RHS[fmax:ftot, :] = np.eye(fmax+1)

        # surface
        b = np.zeros(N[[1], :, :].shape)
        b[:, bandwidth, :] = np.ones((1, ftot))
        b = np.swapaxes(b, 0, 1)
        A[ftot:ftot+2*bandwidth+1, :ftot] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])
        A[:2*bandwidth+1, ftot:2*ftot] = 0
        # bed
        a = N[[-1],Ellipsis]/dz[[-1]]
        b = -N[[-1],Ellipsis]/dz[[-1]]
        a = np.swapaxes(a, 0, 1)
        b = np.swapaxes(b, 0, 1)
        A[2*ftot:2*ftot+2*bandwidth+1, -2*ftot:-ftot] = (a.reshape(a.shape[0], a.shape[1]*a.shape[2]))
        A[ftot:ftot+2*bandwidth+1, -ftot:] = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        csol = solve_banded((bandwidthA, bandwidthA), A, RHS, overwrite_ab=True, overwrite_b=True)
        cSurf[j, :, :, :] = csol.reshape(kmax+1, ftot, fmax+1)

    ####################################################################################################################
    # Scale bed boundary by integral condition: alpha = fac^-1 rhs
    ####################################################################################################################
    cBed = ny.eliminateNegativeFourier(cBed, 2)
    D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1, 1))*np.ones(cBed.shape, dtype=complex)

    fac = np.squeeze(ny.integrate(np.expand_dims(D*cBed, 2), 'z', 0, kmax, data.slice('grid')), 2) + ny.complexAmplitudeProduct(ws[:, [-1], Ellipsis], cBed[:, [-1], Ellipsis], 2)
    rhs = -Fbed[:, :, fmax:]

    # solve integral condition
    alpha = np.linalg.solve(fac, rhs)
    for j in range(0, jmax+1):
        cCoef[j, :, fmax:, :] += np.dot(cBed[j, Ellipsis], alpha[j, 0, :])

    ####################################################################################################################
    # Scale surface boundary by integral condition: alpha = fac^-1 rhs
    ####################################################################################################################
    cSurf = ny.eliminateNegativeFourier(cSurf, 2)
    fac = np.squeeze(ny.integrate(np.expand_dims(D*cSurf, 2), 'z', 0, kmax, data.slice('grid')), 2) + ny.complexAmplitudeProduct(ws[:, [0], Ellipsis], cSurf[:, [0], Ellipsis], 2)
    rhs = Fsurf[:, :, fmax:]

    # solve integral condition
    alpha = np.linalg.solve(fac, rhs)
    for j in range(0, jmax+1):
        cCoef[j, :, fmax:, :] += np.dot(cSurf[j, Ellipsis], alpha[j, 0, :])

    return cCoef, cMatrix

def bandedMatVec(A, x):
    shape = list(A.shape)
    shape[0] = shape[1]
    Afull = np.zeros(shape, dtype=A.dtype)
    bandwdA = (A.shape[0]-1)/2
    size = shape[1]

    Afull[[range(0, size), range(0, size)]] += A[[bandwdA, slice(None)]+[Ellipsis]]
    for n in range(1, bandwdA+1):
        Afull[[range(0, size-n), range(n, size)]] += A[[bandwdA-n, slice(n,None)]]
        Afull[[range(n, size), range(0, size-n)]] += A[[bandwdA+n, slice(None, -n)]]

    return np.dot(Afull, x)

