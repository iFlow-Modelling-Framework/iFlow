"""
uFunction

Date: 07-Aug-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
from nifty import dynamicImport
from src.util.diagnostics import KnownError
import nifty as ny


def uFunction(Av, F, Fsurf, Fbed, data, hasMatrix = False):
    #timers = [Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer(),Timer(), Timer(), Timer()]

    # Init
    kmax = data.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
    fmax = data.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
    OMEGA = data.v('OMEGA')
    ftot = 2*fmax+1

    # determine bandwidth of eddy viscosity matrix
    bandwidth = 0
    for n in np.arange(fmax, -1, -1):
        if np.any(abs(Av[:, n]) > 0):
            bandwidth = max(bandwidth, n)

    # Init Ctd
    nRHS = F.shape[-1]
    velocityMatrix = np.empty([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # Init
        A = np.zeros([2*ftot+2*bandwidth+1, ftot*(kmax+1)], dtype=complex)
        N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        dz = (data.v('grid', 'axis', 'z')[1:]-data.v('grid', 'axis', 'z')[:-1])*data.n('H')
        dz = dz.reshape(dz.shape[0])

        # dz vectors
        dz_down = dz[:kmax-1].reshape(kmax-1, 1, 1)
        dz_up = dz[1:kmax].reshape(kmax-1, 1, 1)
        dz_av = 0.5*(dz_down+dz_up)

        # Build eddy viscosity matrix blocks
        N[:, bandwidth, :] = Av[:, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
        for n in range(1, bandwidth+1):
            N[:, bandwidth+n, :-n] = 0.5*Av[:, n].reshape(kmax+1, 1)*np.ones([1, ftot-n])
            N[:, bandwidth-n, n:] = 0.5*np.conj(Av[:, n]).reshape(kmax+1, 1)*np.ones([1, ftot-n])

        # Build matrix. Discretisation: central for second derivative, central for first derivative
        #  NB. can use general numerical schemes as dz < 0
        a = -0.5*(N[1:kmax, :, :]+N[:kmax-1, :, :])/(dz_down*dz_av)
        b =  0.5*(N[1:kmax, :, :]+N[2:, :, :])/(dz_up*dz_av)+0.5*(N[1:kmax, :, :]+N[:kmax-1, :, :])/(dz_down*dz_av)
        c = -0.5*(N[1:kmax, :, :]+N[2:, :, :])/(dz_up*dz_av)
        b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*np.ones([kmax-1, 1])

        a = np.swapaxes(a, 0, 1)
        b = np.swapaxes(b, 0, 1)
        c = np.swapaxes(c, 0, 1)

        # Build matrix
        A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] += a.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[2*fmax+1:2*fmax+2*bandwidth+2, ftot:-ftot] += b.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[0:2*bandwidth+1, 2*ftot:] += c.reshape(a.shape[0], a.shape[1]*a.shape[2])

        # Boundary conditions
        #   Surface (k=0)
        A[2*fmax+1:2*fmax+2*bandwidth+2, :ftot] += -N[0, :, :]/dz[0]  # first-order upwind discretisation; on diagonal
        A[0:2*bandwidth+1, ftot:2*ftot] += N[0, :, :]/dz[0]       # first-order upwind discretisation; off diagonal

        #   Bed (k=kmax)
        if data.v('BottomBC') == 'NoSlip':
            A[2*fmax+1+bandwidth, -ftot:] = 1.
        else:
            raise KnownError('BottomBC not implemented')

        # save matrix
        velocityMatrix = A[Ellipsis]
        bandwidthA = bandwidth+ftot
    else:
        A = Av[Ellipsis]     # if hasMatrix Av replaces the role of the matrix in this equation
        bandwidthA = (A.shape[0]-1)/2


    ##### RIGHT HAND SIDE #####
    uRHS_zeta = np.zeros([ftot*(kmax+1), ftot], dtype=complex)
    uRHS = np.zeros([ftot*(kmax+1), nRHS], dtype=complex)

    # implicit forcing from barotropic pressure term
    uRHS_zeta[ftot:-ftot, :] = np.concatenate([np.diag([1]*ftot)]*(kmax-1))

    # forcing from other factors
    uRHS[:ftot, :] = Fsurf[0, :, :]
    uRHS[-ftot:, :] = Fbed[0, :, :]
    uRHS[ftot:-ftot, :] = F[1:-1, :, :].reshape((kmax-1)*ftot, nRHS)


    ##### SOLVE #####
    u = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_zeta, uRHS), 1), overwrite_ab=True, overwrite_b=True)
    u = u.reshape(kmax+1, ftot, ftot+nRHS)
    uCoef = u[:, :, :ftot]
    uFirst = u[:, :, ftot:]

    uzCoef = ny.derivative(uCoef.reshape(kmax+1, 1, ftot, ftot), 'z', data.slice('grid')).reshape(kmax+1, ftot, ftot)
    uzFirst = ny.derivative(uFirst.reshape(kmax+1, 1, ftot, nRHS), 'z', data.slice('grid')).reshape(kmax+1, ftot, nRHS)


    return uCoef, uFirst, uzCoef, uzFirst, velocityMatrix


