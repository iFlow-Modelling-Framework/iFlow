"""
uFunction

Date: 07-Aug-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
from nifty.integrate import integrate
import nifty as ny
from bandedMatrixMultiplication import bandedMatrixMultiplication

def uFunctionMomentumConservative(Av, F, Fsurf, Fbed, data, hasMatrix = False):
    # Init
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    ftot = 2*fmax+1
    bottomBC = data.v('BottomBC')

    # determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Av[:, :, n]) > 0):
            bandwidth = max(bandwidth, n)

    if bottomBC == 'PartialSlip':
        sf = data.v('Roughness', range(0, jmax+1), 0, range(0, fmax+1))
        Sf = sfMat(sf)
        SfInv = sfMat(data.v('Roughness', range(0, jmax+1), 0, range(0, fmax+1)), True)

        # set new bandwidth. This is determined by cf*U*N. However in the code this product is truncated to
        # have a bandwidth equal to the maximum bandwidth of cf*U and N. This is the new bandwidth
        bandwidth = max(bandwidth, (Sf.shape[1]-1)/2) # set bandwidth as max of bandwidth of Av and cf|u|

    # Init Ctd
    nRHS = F.shape[-1]
    velocityMatrix = np.empty([jmax+1, 2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
    uzCoef = np.zeros([jmax+1, kmax+1, 1, ftot, ftot], dtype=complex)
    uzFirst = np.zeros([jmax+1, kmax+1, 1, ftot, nRHS], dtype=complex)

    ubedCoef = np.zeros([jmax+1, kmax+1, ftot, ftot], dtype=complex)
    ubedFirst = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)

    # build, save and solve the velocity matrices in every water column
    for j in range(0, jmax+1):
        ##### LEFT HAND SIDE #####
        if not hasMatrix:
            # Init
            A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
            N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
            z = ny.dimensionalAxis(data.slice('grid'), 'z')[j,:,0]
            dz = z[1:]-z[0:-1]
            del z

            # dz vectors
            dz_down = dz[:kmax-1].reshape(kmax-1, 1, 1)
            dz_up = dz[1:kmax].reshape(kmax-1, 1, 1)
            dz_av = 0.5*(dz_down+dz_up)

            # Build eddy viscosity matrix blocks
            N[:, bandwidth, :] = Av[j, :, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                N[:, bandwidth+n, :-n] = 0.5*Av[j, :, n].reshape(kmax+1, 1)*np.ones([1, ftot-n])
                N[:, bandwidth-n, n:] = 0.5*np.conj(Av[j, :, n]).reshape(kmax+1, 1)*np.ones([1, ftot-n])

            # Build matrix. Discretisation: central for second derivative, central for first derivative
            #  NB. can use general numerical schemes as dz < 0
            a = -N[:-2, :, :]/dz_down
            b = N[1:kmax, :, :]/dz_up+N[1:kmax, :, :]/dz_down
            c = -N[2:, :, :]/dz_up
            b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*dz_av.reshape((kmax-1, 1))

            a = np.swapaxes(a, 0, 1)
            b = np.swapaxes(b, 0, 1)
            c = np.swapaxes(c, 0, 1)

            # Build matrix
            A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] = a.reshape(a.shape[0], a.shape[1]*a.shape[2])
            A[2*fmax+1:2*fmax+2*bandwidth+2, ftot:-ftot] = b.reshape(a.shape[0], a.shape[1]*a.shape[2])
            A[0:2*bandwidth+1, 2*ftot:] = c.reshape(a.shape[0], a.shape[1]*a.shape[2])

            # Boundary conditions
            #   Surface (k=0)
            A[2*fmax+1:2*fmax+2*bandwidth+2, :ftot] = N[0, :, :]

            #   Bed (k=kmax)
            A[2*ftot:, -2*ftot:-ftot] = N[kmax-1, :, :]/dz[-1]
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] = -N[kmax, :, :]/dz[-1]
            A[2*fmax+bandwidth+1,None, -ftot:]+= -(np.arange(-fmax, ftot-fmax)*1j*OMEGA*0.5).reshape(1, ftot)*dz[-1]
            if bottomBC in ['PartialSlip']:
                # for partial slip: multiply the equation closest to the bed by Sf.
                D = (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))

                cfUNbed = bandedMatrixMultiplication(Sf[j, Ellipsis], N[kmax, :, :], truncate=True)
                cfUNabovebed = bandedMatrixMultiplication(Sf[j, Ellipsis], N[kmax-1, :, :], truncate=True)
                cfUD = bandedMatrixMultiplication(D, Sf[j, Ellipsis], truncate=True)
                A[2*ftot:, -2*ftot:-ftot] = cfUNabovebed/dz[-1]
                A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] = -cfUNbed/dz[-1]
                A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += -0.5*dz[-1]*cfUD

                DNsmall = bandedMatrixMultiplication(D, N[kmax, :, :], truncate=True)
                DN = np.zeros((2*bandwidth+1, ftot), dtype=complex)
                DN[bandwidth-(DNsmall.shape[0]-1)/2:bandwidth+(DNsmall.shape[0]-1)/2 + 1, :] = DNsmall
                A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += DN

            # save matrix
            velocityMatrix[j, Ellipsis] = A[Ellipsis]
            bandwidthA = bandwidth+ftot
        else:
            A = Av[j, Ellipsis]     # if hasMatrix Av replaces the role of the matrix in this equation
            bandwidthA = (A.shape[0]-1)/2


        ##### RIGHT HAND SIDE #####
        uRHS_zeta = np.zeros([ftot*(kmax+1), ftot], dtype=complex)
        uRHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)

        # implicit forcing from -g zeta_x term; only a contribution at the bed
        uRHS_zeta[-ftot:, :] = np.diag([1]*ftot)
        if bottomBC in ['PartialSlip']:
            # for partial slip: multiply the equation closest to the bed by Sf.
            uRHS_zeta[-ftot:, :] = bandedMatVec(Sf[j, Ellipsis], np.diag([1]*ftot))

        # forcing from other factors
        uRHS[:ftot, :] = Fsurf[j, 0, :, :]
        uRHS[-ftot:, :] = 0.5*(F[j, kmax, :, :]+F[j, kmax-1, :, :])
        uRHS[ftot:-ftot, :] = 0.5*(F[j, 2:, :, :]-F[j, :-2, :, :]).reshape((kmax-1)*ftot, nRHS)
        if bottomBC in ['PartialSlip']:
            # for partial slip: multiply the equation closest to the bed by Sf.
            uRHS[-ftot:, :] = bandedMatVec(Sf[j,Ellipsis], 0.5*(F[j, kmax, :, :]+F[j, kmax-1, :, :]))
            uRHS[-ftot:, :]+= (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((ftot, 1))*Fbed[j, 0, :, :]

        ##### SOLVE #####
        uz = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_zeta, uRHS), 1), overwrite_ab=True, overwrite_b=True)
        uz = uz.reshape(kmax+1, ftot, ftot+nRHS)
        uzCoef[j, :, 0, :, :] = uz[:, :, :ftot]
        uzFirst[j, :, 0, :, :] = uz[:, :, ftot:]

        # add bottom velocity in case of partial/quadratic slip
        if bottomBC in ['PartialSlip']:
            # if matrix is not calculated, Av and N are not available, calculate them here
            if hasMatrix:
                Avbed = data.v('Av', j, kmax, range(0, fmax+1))

                bandwidth = 0
                for n in np.arange(fmax, -1, -1):
                    if abs(Avbed[n]) > 0:
                        bandwidth = max(bandwidth, n)

                N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
                N[-1, bandwidth, :] = Avbed[0]*np.ones([ftot])
                for n in range(1, bandwidth+1):
                    N[-1, bandwidth+n, :-n]= 0.5*Avbed[n]*np.ones([ftot-n])
                    N[-1, bandwidth-n, n:] = 0.5*np.conj(Avbed[n])*np.ones([ftot-n])

            # calculate banded matrix products
            SfInvN = bandedMatrixMultiplication(SfInv[j, Ellipsis], N[-1, Ellipsis], truncate=True)
            SfInvFbed = bandedMatVec(SfInv[j, Ellipsis], Fbed[j, 0,:, :])

            # rewrite to bandmatrix
            bandwd_inv = (SfInvN.shape[0]-1)/2    # note: bandwidth of inverse velocity can be larger than other bandwidth
            Ndiag = np.diag(SfInvN[bandwd_inv, :])
            for n in range(1, bandwd_inv+1):
                Ndiag += np.diag(SfInvN[bandwd_inv+n, :-n], n)
                Ndiag += np.diag(SfInvN[bandwd_inv-n, n:], -n)

            # calculate bed velocity
            ubedCoef[j, :, :, :] = (np.dot(Ndiag, uzCoef[j, -1, 0, :, :])).reshape((1, ftot, ftot))*np.ones([kmax+1, 1, 1])
            ubedFirst[j, :, :, :] = (np.dot(Ndiag, uzFirst[j, -1, 0, :, :])).reshape((1, ftot, nRHS))*np.ones([kmax+1, 1, 1])
            ubedFirst[j, :, :, :] += -SfInvFbed.reshape((1,)+SfInvFbed.shape)*np.ones([kmax+1, 1, 1])

    uCoef = integrate(uzCoef, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'), INTMETHOD='INTERPOLSIMPSON')[:, :, 0, :, :] + ubedCoef
    if 0 in (uzFirst).shape:
        uFirst = uzFirst
    else:
        uFirst = integrate(uzFirst, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'), INTMETHOD='INTERPOLSIMPSON')[:, :, 0, :, :] + ubedFirst

    return uCoef, uFirst, uzCoef[:, :, 0, :, :], uzFirst[:, :, 0, :, :], velocityMatrix

def sfMat(sf, inv = False):
    jmax = sf.shape[0]-1
    fmax = sf.shape[-1]-1
    # maximum index of f grid (fmax+1 grid points incl. 0)
    ftot = 2*fmax+1

    #determine bandwidth
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if any(np.abs(sf[:, n]) > 0):
            bandwidth = max(bandwidth, n)

    # make bandmatrix form
    Mat = np.zeros([jmax+1, 2*bandwidth+1, ftot], dtype=complex)
    Mat[:, bandwidth, :] = sf[:, [0]]*np.ones([1, ftot])
    for n in range(1, bandwidth+1):
        Mat[:, bandwidth+n, :-n] = 0.5*sf[:, [n]]*np.ones([1, ftot-n])
        Mat[:, bandwidth-n, n:] = 0.5*np.conj(sf[:, [n]])*np.ones([1, ftot-n])

    if inv: # if True, compute the inverse of Mat and cf
        Matinv = np.zeros((jmax+1, ftot, ftot), dtype=complex)
        for j in range(0, jmax+1):
            Matinv[j, Ellipsis] = solve_banded((bandwidth, bandwidth), Mat[j, Ellipsis], np.eye(ftot), overwrite_ab=True, overwrite_b=True)

        # convert back to band
        bandwd = 0
        for n in np.arange(ftot-1, -1, -1):
            if np.any(abs(Matinv[(slice(None), range(n, ftot), range(0, ftot-n))]) > 0):
                bandwd = max(bandwd, n)
            if np.any(abs(Matinv[(slice(None),range(0, ftot-n), range(n, ftot))]) > 0):
                bandwd = max(bandwd, n)

        Mat = np.zeros((jmax+1, 2*bandwd+1, ftot), dtype=complex)
        Mat[(slice(None), bandwd, slice(None))] = Matinv[(slice(None), range(0, ftot), range(0, ftot))]
        for n in range(1, bandwd+1):
            Mat[(slice(None), bandwd-n, slice(n, None))] = Matinv[(slice(None), range(0, ftot-n), range(n, ftot))]
            Mat[(slice(None), bandwd+n, slice(None,-n))] = Matinv[(slice(None), range(n, ftot), range(0, ftot-n))]
    return Mat

def bandedMatVec(A, x):
    shape = list(A.shape)
    shape[0] = shape[1]
    Afull = np.zeros(shape, dtype=A.dtype)
    bandwdA = (A.shape[0]-1)/2
    size = shape[1]

    Afull[(range(0, size), range(0, size))] += A[(bandwdA, slice(None))+(Ellipsis,)]
    for n in range(1, bandwdA+1):
        Afull[(range(0, size-n), range(n, size))] += A[(bandwdA-n, slice(n,None))]
        Afull[(range(n, size), range(0, size-n))] += A[(bandwdA+n, slice(None, -n))]

    return np.dot(Afull, x)

