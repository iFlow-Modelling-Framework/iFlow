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

def uFunction(Av, F, Fsurf, Fbed, data, hasMatrix = False):
    # Init
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    ftot = 2*fmax+1
    bottomBC = data.v('BottomBC')

    # determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Av[:, n]) > 0):
            bandwidth = max(bandwidth, n)

    if bottomBC == 'QuadraticSlip':
        cfU = cfUMat(data)
        cfUInv = cfUMat(data, True)

        # set new bandwidth. This is determined by cf*U*N. However in the code this product is truncated to
        # have a bandwidth equal to the maximum bandwidth of cf*U and N. This is the new bandwidth
        bandwidth = max(bandwidth, (cfU.shape[1]-1)/2) # set bandwidth as max of bandwidth of Av and cf|u|

    # Init Ctd
    nRHS = F.shape[-1]
    velocityMatrix = np.empty([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
    uzCoef = np.zeros([kmax+1, 1, ftot, ftot], dtype=complex)
    uzFirst = np.zeros([kmax+1, 1, ftot, nRHS], dtype=complex)

    ubedCoef = np.zeros([kmax+1, ftot, ftot], dtype=complex)
    ubedFirst = np.zeros([kmax+1, ftot, nRHS], dtype=complex)

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # Init
        A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
        N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        dz = (data.v('grid', 'axis', 'z')[1:]-data.v('grid', 'axis', 'z')[0:-1])*data.n('H')

        # dz vectors
        dz_down = dz[:kmax-1].reshape(kmax-1, 1, 1)
        dz_up = dz[1:kmax].reshape(kmax-1, 1, 1)
        dz_av = 0.5*(dz_down+dz_up)

        # Build eddy viscosity matrix blocks
        N[:, bandwidth, :] = Av[:, [0]]*np.ones([1, ftot])
        for n in range(1, bandwidth+1):
            N[:, bandwidth+n, :-n] = 0.5*Av[:, [n]]*np.ones([1, ftot-n])
            N[:, bandwidth-n, n:] = 0.5*np.conj(Av[:, [n]])*np.ones([1, ftot-n])

        # Build matrix. Discretisation: central for second derivative, central for first derivative
        #  NB. can use general numerical schemes as dz < 0
        a = -N[:-2, :, :]/dz_down
        b = N[1:kmax, :, :]/dz_up+N[1:kmax, :, :]/dz_down
        c = -N[2:, :, :]/dz_up
        b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape(1, ftot)*dz_av.reshape(kmax-1,1)

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
        A[2*fmax+bandwidth+1,None, -ftot:] += -(np.arange(-fmax, ftot-fmax)*1j*OMEGA*0.5).reshape((1, ftot))*dz[-1]
        if bottomBC in ['PartialSlip']:
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*N[kmax, :, :]/data.v('Roughness')
        elif bottomBC in ['QuadraticSlip']:
            D = (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))

            cfUNbed = bandedMatrixMultiplication(cfU, N[kmax, :, :], truncate=True)
            cfUNabovebed = bandedMatrixMultiplication(cfU, N[kmax-1, :, :], truncate=True)
            cfUD = bandedMatrixMultiplication(D, cfU, truncate=True)
            A[2*ftot:, -2*ftot:-ftot] = cfUNabovebed/dz[-1]
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] = -cfUNbed/dz[-1]
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += -0.5*dz[-1]*cfUD

            DNsmall = bandedMatrixMultiplication(D, N[kmax, :, :], truncate=True)
            DN = np.zeros((2*bandwidth+1, ftot), dtype=complex)
            DN[bandwidth-(DNsmall.shape[0]-1)/2:bandwidth+(DNsmall.shape[0]-1)/2 + 1, :] = DNsmall
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += DN

        # save matrix
        velocityMatrix[Ellipsis] = A[Ellipsis]
        bandwidthA = bandwidth+ftot
    else:
        A = Av[Ellipsis]     # if hasMatrix Av replaces the role of the matrix in this equation
        bandwidthA = (A.shape[0]-1)/2


    ##### RIGHT HAND SIDE #####
    uRHS_zeta = np.zeros([ftot*(kmax+1), ftot], dtype=complex)
    uRHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)

    # implicit forcing from -g zeta_x term; only a contribution at the bed
    uRHS_zeta[-ftot:, :] = np.diag([1]*ftot)
    if bottomBC in ['QuadraticSlip']:
        uRHS_zeta[-ftot:, :] = bandedMatVec(cfU, np.diag([1]*ftot))

    # forcing from other factors
    uRHS[:ftot, :] = Fsurf[0, :, :]
    uRHS[ftot:-ftot, :] = 0.5*(F[2:, :, :]-F[:-2, :, :]).reshape((kmax-1)*ftot, nRHS)
    uRHS[-ftot:, :] = 0.5*(F[kmax, :, :]+F[kmax-1, :, :])
    if bottomBC in ['PartialSlip']:
        uRHS[-ftot:, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((ftot, 1))*Fbed[0, :, :]/data.v('Roughness')
    if bottomBC in ['QuadraticSlip']:
        uRHS[-ftot:, :] = bandedMatVec(cfU, 0.5*(F[kmax, :, :]+F[kmax-1, :, :]))
        uRHS[-ftot:, :]+= (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((ftot, 1))*Fbed[0, :, :]

    ##### SOLVE #####
    uz = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_zeta, uRHS), 1), overwrite_ab=True, overwrite_b=True)
    uz = uz.reshape(kmax+1, ftot, ftot+nRHS)
    uzCoef[:, 0, :, :] = uz[:, :, :ftot]
    uzFirst[:, 0, :, :] = uz[:, :, ftot:]

    # add bottom velocity in case of partial/quadratic slip
    if bottomBC in ['PartialSlip', 'QuadraticSlip']:
        # if matrix is not calculated, Av and N are not available, calculate them here
        if not 'N' in locals():
            Avbed = data.v('Av', kmax, range(0, fmax+1))

            bandwidth = 0
            for n in np.arange(fmax, -1, -1):
                if abs(Avbed[n]) > 0:
                    bandwidth = max(bandwidth, n)

            N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
            N[-1, bandwidth, :] = Avbed[0]*np.ones([ftot])
            for n in range(1, bandwidth+1):
                N[-1, bandwidth+n, :-n] = 0.5*Avbed[n]*np.ones([ftot])
                N[-1, bandwidth-n, n:] = 0.5*np.conj(Avbed[n])*np.ones([ftot])

        if bottomBC in ['PartialSlip']:
            Ndiag = np.diag(N[-1, bandwidth, :])
            for n in range(1, bandwidth+1):
                Ndiag += np.diag(N[-1, bandwidth+n, :-n], n)
                Ndiag += np.diag(N[-1, bandwidth-n, n:], -n)

            ubedCoef = (np.dot(Ndiag, uzCoef[-1, 0, :, :])/data.v('Roughness')).reshape((1, ftot, ftot))*np.ones([kmax+1, 1, 1])
            ubedFirst = (np.dot(Ndiag, uzFirst[-1, 0, :, :])/data.v('Roughness')).reshape((1, ftot, nRHS))*np.ones([kmax+1, 1, 1])
            ubedFirst += -Fbed[[0], :, :]/data.v('Roughness')*np.ones([kmax+1, 1, 1])
        elif bottomBC in ['QuadraticSlip']:
            cfUInv = cfUMat(data, True)
            cfUN = bandedMatrixMultiplication(cfUInv, N[-1, Ellipsis], truncate=True)
            cfUFbed = bandedMatVec(cfUInv, Fbed[0,:, :])

            bandwidth = (cfUN.shape[0]-1)/2
            Ndiag = np.diag(cfUN[bandwidth, :])
            for n in range(1, bandwidth+1):
                Ndiag += np.diag(cfUN[bandwidth+n, :-n], n)
                Ndiag += np.diag(cfUN[bandwidth-n, n:], -n)

            ubedCoef = (np.dot(Ndiag, uzCoef[-1, 0, :, :])).reshape((1, ftot, ftot))*np.ones([kmax+1, 1, 1])
            ubedFirst = (np.dot(Ndiag, uzFirst[-1, 0, :, :])).reshape((1, ftot, nRHS))*np.ones([kmax+1, 1, 1])
            ubedFirst += -cfUFbed.reshape((1,)+cfUFbed.shape)*np.ones([kmax+1, 1, 1])

    uCoef = integrate(uzCoef, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'))[:,0,:,:] + ubedCoef
    uFirst = integrate(uzFirst, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'))[:,0,:,:]+ ubedFirst

    return uCoef, uFirst, uzCoef[:, 0, :, :], uzFirst[:, 0, :, :], velocityMatrix

def cfUMat(data, inv = False):
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    ftot = 2*fmax+1

    # load data
    cf = data.v('Roughness')
    u0 = data.v('u0', kmax, range(0, fmax+1))

    if u0 is not None:
        # determine |u| from chebyshev approximation
        c = ny.polyApproximation(np.abs, 4)
        uscale = np.max(np.abs(u0))                                   # scale for maximum of velocity
        u2 = ny.complexAmplitudeProduct(u0/uscale, u0/uscale, 0)   # (velocity/uscale)^2
        u4 = ny.complexAmplitudeProduct(u2, u2, 0)

        absu = uscale*(c[0]*np.eye(fmax+1, 1).reshape(fmax+1)+c[2]*u2+c[4]*u4)

        #determine bandwidth
        bandwidth = 0
        for n in np.arange(fmax, -1, -1):
            if abs(absu[n]) > 0:
                bandwidth = max(bandwidth, n)

        # make bandmatrix form
        U = np.zeros([2*bandwidth+1, ftot], dtype=complex)
        U[bandwidth, :] = absu[0]*np.ones([1, ftot])
        for n in range(1, bandwidth+1):
            U[bandwidth+n, :-n] = 0.5*absu[n]*np.ones([1, ftot-n])
            U[bandwidth-n, n:] = 0.5*np.conj(absu[n])*np.ones([1, ftot-n])

        if inv: # if True, compute the inverse of U and cf
            Uinvfull = solve_banded((bandwidth, bandwidth), U, np.eye(ftot), overwrite_ab=True, overwrite_b=True)

            # convert back to band
            bandwd = 0
            for n in np.arange(ftot-1, -1, -1):
                if np.any(abs(Uinvfull[[range(n, ftot), range(0, ftot-n)]]) > 0):
                    bandwd = max(bandwd, n)
                if np.any(abs(Uinvfull[[range(0, ftot-n), range(n, ftot)]]) > 0):
                    bandwd = max(bandwd, n)

            U = np.zeros((2*bandwd+1, ftot), dtype=complex)
            U[[bandwd, slice(None)]] = Uinvfull[[range(0, ftot), range(0, ftot)]]
            for n in range(1, bandwd+1):
                U[[bandwd-n, slice(n, None)]] = Uinvfull[[range(0, ftot-n), range(n, ftot)]]
                U[[bandwd+n, slice(None, -n)]] = Uinvfull[[range(n, ftot), range(0, ftot-n)]]

            cf = 1./cf
    else:
        U = np.ones((1, ftot))

    return U*cf

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

