"""
svarFunction

Date: 11-Jan-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from scipy.linalg import solve_banded


def svarFunction(Kv, F, Fsurf, Fbed, data, hasMatrix=False):
# Init
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    u0z = data.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
    ftot = 2*fmax+1

    # determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Kv[:, :, n]) > 0):
            bandwidth = max(bandwidth, n)

    # Init Ctd
    nRHS = F.shape[-1]
    salinityMatrix = np.empty([jmax+1, 2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
    szCoef = np.zeros([jmax+1, kmax+1, 1, ftot, ftot], dtype=complex)
    szFirst = np.zeros([jmax+1, kmax+1, 1, ftot, nRHS], dtype=complex)

    # build, save and solve the velocity matrices in every water column
    for j in range(0, jmax+1):
        # dz vectors
        dz = (data.v('grid', 'axis', 'z')[0, 1:]-data.v('grid', 'axis', 'z')[0, 0:-1])*data.n('H', j)
        dz = dz.reshape(dz.shape[0])
        dz_down = dz[:kmax-1].reshape(kmax-1, 1, 1)
        dz_up = dz[1:kmax].reshape(kmax-1, 1, 1)
        dz_av = 0.5*(dz_down+dz_up)

        ##### LEFT HAND SIDE #####
        if not hasMatrix:
            # Init
            A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
            N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)

            # Build eddy viscosity matrix blocks
            N[:, bandwidth, :] = Kv[j, :, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
            for n in range(1, bandwidth+1):
                N[:, bandwidth+n, :-n] = 0.5*Kv[j, :, n].reshape(kmax+1, 1)*np.ones([1, ftot])
                N[:, bandwidth-n, n:] = 0.5*np.conj(Kv[j, :, n]).reshape(kmax+1, 1)*np.ones([1, ftot])

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
            A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] = N[-1, :, :]

            # save matrix
            salinityMatrix[j, Ellipsis] = A[Ellipsis]
            bandwidthA = bandwidth+ftot
        else:
            A = Kv[j, Ellipsis]     # if hasMatrix Av replaces the role of the matrix in this equation
            bandwidthA = (A.shape[0]-1)/2


        ##### RIGHT HAND SIDE #####
        #   Implicit part of the forcing uz0*dz*Sx
        u0zdz = u0z[j, 1:-1, :]*dz_av.reshape((kmax-1, 1))

        umat = np.zeros((kmax+1, ftot, ftot), dtype=complex)
        umat[1:-1, range(0, ftot), range(0, ftot)] = u0zdz[:, [0]]*np.ones([1, ftot])
        for n in range(1, fmax+1):
            umat[1:-1, range(n, ftot), range(0, ftot-n)] = 0.5*u0zdz[:, [n]]*np.ones([1, ftot-n]) # lower diag
            umat[1:-1, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(u0zdz[:, [n]])*np.ones([1, ftot-n]) # upper diag

        uRHS_implicit = umat.reshape((ftot*(kmax+1), ftot))

        #   Forcing from other factors
        uRHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)
        uRHS[:ftot, :] = Fsurf[j, 0, :, :]
        uRHS[-ftot:, :] = Fbed[j, 0, :, :]
        uRHS[ftot:-ftot, :] = 0.5*(F[j, 2:, :, :]-F[j, :-2, :, :]).reshape((kmax-1)*ftot, nRHS)

        ##### SOLVE #####
        sz = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_implicit, uRHS), 1), overwrite_ab=True, overwrite_b=True)
        sz = sz.reshape(kmax+1, ftot, ftot+nRHS)
        szCoef[j, :, 0, :, :] = sz[:, :, :ftot]
        szFirst[j, :, 0, :, :] = sz[:, :, ftot:]
        del sz, uRHS_implicit, uRHS

    ##### INTEGRATION CONSTANT #####
    if hasMatrix:
        KvBed = data.v('Av', range(0, jmax+1), [kmax-1, kmax], range(0, fmax+1))
    else:
        KvBed = Kv[:, [kmax-1, kmax], :]
    KvBed = np.concatenate((np.zeros((jmax+1, 2, fmax)), KvBed), 2)

    u0bed = data.v('u0', range(0, jmax+1), [-1], range(0, fmax+1))
    u0bedmat = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
    u0bedmat[:, 0, range(0, ftot), range(0, ftot)] = u0bed[:, 0, [0]]*np.ones([1, ftot])
    for n in range(1, fmax+1):
        u0bedmat[:, 0, range(n, ftot), range(0, ftot-n)] = 0.5*u0bed[:, 0, [n]]*np.ones([1, ftot-n]) # lower diag
        u0bedmat[:, 0, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(u0bed[:, 0, [n]])*np.ones([1, ftot-n]) # upper diag


    Dinv = np.diag((np.arange(-fmax, ftot-fmax)*1j*OMEGA))
    Dinv[fmax, fmax] = np.inf
    Dinv[range(0, ftot), range(0, ftot)] = Dinv[range(0, ftot), range(0, ftot)]**(-1)
    z = ny.dimensionalAxis(data.slice('grid'), 'z')
    dzbed = z[:, [-1], 0] - z[:, [-2], 0]
    sbedCoef = -ny.complexAmplitudeProduct(KvBed[:, [-2], :], szCoef[:, [-2], 0, :, :], 2, includeNegative=True)/dzbed.reshape(dzbed.shape+(1, 1)) + u0bedmat
    sbedCoef = np.dot(Dinv, sbedCoef)
    sbedCoef = np.rollaxis(sbedCoef, 0, 3)
    sbedFirst = (Fbed-ny.complexAmplitudeProduct(KvBed[:, [-2], :], szFirst[:, [-2], 0, :, :], 2, includeNegative=True))/dzbed.reshape(dzbed.shape+(1, 1)) + F[:, [-1], :, :]
    sbedFirst = np.dot(Dinv, sbedFirst)
    sbedFirst = np.rollaxis(sbedFirst, 0, 3)

    ##### INTEGRATION #####
    sCoef = ny.integrate(szCoef, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'))[:, :, 0, :, :] + sbedCoef*np.ones((1, kmax+1, 1, 1))
    sFirst = ny.integrate(szFirst, 'z', kmax, np.arange(0, kmax+1), data.slice('grid'))[:, :, 0, :, :] + sbedFirst*np.ones((1, kmax+1, 1, 1))

    ##### CLOSURE FOR THE DEPTH-AVERAGED TIME-AVERAGED SALINITY VARIATION #####
    ny.integrate(sCoef[:, :, fmax, fmax], 'z', kmax, np.arange(0, kmax+1), data.slice('grid'))

    return sCoef, sFirst, szCoef[:, :, 0, :, :], szFirst[:, :, 0, :, :], salinityMatrix