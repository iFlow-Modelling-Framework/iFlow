"""
svarFunction

Date: 12-01-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from scipy.linalg import solve_banded


def svarFunction(Kv, F, Fsurf, Fbed, data, hasMatrix=False):
    """Solve a function
    Ds - (Kv s_z)_z = -u0_z + F
    subject to
        Kv s_z (-H) = Fbed
        Kv s_z (0) = Fsurf

    The returned solution has a part 'sCoef' and 'szCoef' for the forcing by u_z and a part 'sForced' and 'szForced' for the
    forcing by F, Fbed and Fsurf.

    Args:
        Kv: (ndarray(jmax+1, kmax+1, fmax+1)) - data on eddy diffusivity
            or a salinityMatrix as calculated before by this function.
        F: (ndarray(jmax+1, kmax+1, fmax+1, nRHS)) - interior forcing. nRHS is the number of individual forcing components
        Fsurf: (ndarray(jmax+1, 1, fmax+1, nRHS)) - surface forcing. nRHS is the number of individual forcing components
        Fbed: (ndarray(jmax+1, 1, fmax+1, nRHS)) - surface forcing. nRHS is the number of individual forcing components
        data: (DataContainer) - should at least contain 'grid', 'OMEGA' and 'u0'
        hasMatrix: (bool) - if True then it is assumed that Kv contains a salinityMatrix as calculated by this function before.
                            The matrix is not computed again, which saves time.

    Returns:
        sCoef and szCoef: (ndarray(jmax+1, kmax+1, fmax+1, 1)) the solution s and its vertical derivative for the forcing by u0_z.
                            the final dimension '1' denotes a single forcing component.
        sForced and szForced: (ndarray(jmax+1, kmax+1, fmax+1, nRHS)) the solution s and its vertical derivative for the other forcings.
                            the solution is separated for each RHS term in F, Fsurf and Fbed.
        salinityMatrix: matrix used in computation. Can be reused for subsequent calls of this function.
    """
    # Init
    jmax = data.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    uz = data.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
    u0bed = data.v('u0', range(0, jmax+1), [-1], range(0, fmax+1))
    ftot = 2*fmax+1

    # Determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Kv[:, :, n]) > 0):
            bandwidth = max(bandwidth, n)

    # Init Ctd
    nRHS = F.shape[-1]
    salinityMatrix = np.empty([jmax+1, 2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
    szCoef = np.zeros([jmax+1, kmax+1, fmax+1, 1], dtype=complex)
    szForced = np.zeros([jmax+1, kmax+1, fmax+1, nRHS], dtype=complex)

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
        umat = np.zeros((kmax+1, ftot), dtype=complex)
        umat[1:-1, fmax:ftot] = -uz[j, 1:-1, :]*dz_av.reshape((kmax-1, 1))
        uRHS_implicit = umat.reshape((ftot*(kmax+1), 1))

        #   Forcing from other factors
        uRHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)
        uRHS[fmax:ftot, :] = Fsurf[j, 0, :, :]
        uRHS[-ftot+fmax:, :] = Fbed[j, 0, :, :]
        F_full = np.concatenate((np.zeros((jmax+1, kmax+1, fmax, nRHS)), F), 2)
        uRHS[ftot:-ftot, :] = 0.5*(F_full[j, 2:, :, :]-F_full[j, :-2, :, :]).reshape((kmax-1)*ftot, nRHS)

        ##### SOLVE #####
        sz = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_implicit, uRHS), 1), overwrite_ab=True, overwrite_b=True)
        sz = sz.reshape(kmax+1, ftot, 1+nRHS)
        szCoef[j, :, :, :] = ny.eliminateNegativeFourier(sz[:, :, :1], 1)
        szForced[j, :, :, :] = ny.eliminateNegativeFourier(sz[:, :, 1:], 1)
        del sz, uRHS_implicit, uRHS

    ##### INTEGRATION CONSTANT #####
    SIGMASAL = data.v('SIGMASAL')
    if hasMatrix:
        KvBed = data.v('Av', range(0, jmax+1), [kmax-1, kmax], range(0, fmax+1))/SIGMASAL # TODO: shift to turbulence model
    else:
        KvBed = Kv[:, [kmax-1, kmax], :]

    Dinv = np.diag((np.arange(0, fmax+1)*1j*OMEGA))
    Dinv[0, 0] = np.inf
    Dinv[range(0, fmax+1), range(0, fmax+1)] = Dinv[range(0, fmax+1), range(0, fmax+1)]**(-1)
    z = ny.dimensionalAxis(data.slice('grid'), 'z')
    dzbed = z[:, [-1], 0] - z[:, [-2], 0]
    sbedCoef = -ny.complexAmplitudeProduct(KvBed[:, [-2], :], szCoef[:, [-2], :, :], 2)/dzbed.reshape(dzbed.shape+(1, 1)) - u0bed.reshape(u0bed.shape+(1,))
    sbedCoef = np.dot(Dinv, sbedCoef)
    sbedCoef = np.rollaxis(sbedCoef, 0, 3)
    sbedForced = (Fbed-ny.complexAmplitudeProduct(KvBed[:, [-2], :], szForced[:, [-2], :, :], 2))/dzbed.reshape(dzbed.shape+(1, 1)) + F[:, [-1], :, :]
    sbedForced = np.dot(Dinv, sbedForced)
    sbedForced = np.rollaxis(sbedForced, 0, 3)

    ##### INTEGRATION #####
    sCoef = ny.integrate(szCoef, 'z', kmax, np.arange(0, kmax+1), data.slice('grid')) + sbedCoef*np.ones((1, kmax+1, 1, 1))
    sForced = ny.integrate(szForced, 'z', kmax, np.arange(0, kmax+1), data.slice('grid')) + sbedForced*np.ones((1, kmax+1, 1, 1))

    ##### CLOSURE FOR THE DEPTH-AVERAGED TIME-AVERAGED SALINITY VARIATION #####
    H = data.v('H', range(0, jmax+1)).reshape((jmax+1, 1, 1, 1))
    sCoef[:, :, [0], :] -= (ny.integrate(sCoef[:, :, [0], :], 'z', kmax, 0, data.slice('grid'))/H)*np.ones((1, kmax+1, 1, 1))
    sForced[:, :, [0], :] -= (ny.integrate(sForced[:, :, [0], :], 'z', kmax, 0, data.slice('grid'))/H)*np.ones((1, kmax+1, 1, 1))

    if hasMatrix:
        return sCoef, sForced, szCoef, szForced, Kv
    else:
        return sCoef, sForced, szCoef, szForced, salinityMatrix