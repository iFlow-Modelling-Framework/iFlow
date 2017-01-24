"""
uFunction

Date: 07-Aug-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from scipy.linalg import solve_banded
import nifty as ny


def uFunction((Av, Avz), F, Fsurf, Fbed, data, hasMatrix = False):
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
    velocityMatrix = np.empty([2*ftot+2*bandwidth+1, ftot*kmax], dtype=complex)

    ##### LEFT HAND SIDE #####
    if not hasMatrix:
        # Init
        A = np.zeros([2*ftot+2*bandwidth+1, ftot*kmax], dtype=complex)
        N = np.zeros([kmax+1, 2*bandwidth+1, ftot], dtype=complex)
        dz = (data.v('grid', 'axis', 'z')[1:]-data.v('grid', 'axis', 'z')[:-1])*data.n('H')
        dz = dz.reshape(dz.shape[0])

        # dz vectors
        dz_k = dz[1:kmax-1].reshape(kmax-2, 1, 1)
        dz_down = 0.5*(dz[:kmax-2]+dz[1:kmax-1]).reshape(kmax-2, 1, 1)
        dz_up = 0.5*(dz[2:kmax]+dz[1:kmax-1]).reshape(kmax-2, 1, 1)

        # Build eddy viscosity matrix blocks
        N[:, bandwidth, :] = Av[:, 0].reshape(kmax+1, 1)*np.ones([1, ftot])
        for n in range(1, bandwidth+1):
            N[:, bandwidth+n, :-n] = 0.5*Av[:, n].reshape(kmax+1, 1)*np.ones([1, ftot-n])
            N[:, bandwidth-n, n:] = 0.5*np.conj(Av[:, n]).reshape(kmax+1, 1)*np.ones([1, ftot-n])

        # Build matrix. Discretisation: central for second derivative, central for first derivative
        #  NB. can use general numerical schemes as dz < 0
        a = -N[1:kmax-1, :, :]/(dz_down*dz_k)
        b =  N[2:kmax, :, :]/(dz_up*dz_k)+N[1:kmax-1, :, :]/(dz_down*dz_k)
        c = -N[2:kmax, :, :]/(dz_up*dz_k)
        #b[:, bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA).reshape((1, ftot))*np.ones([kmax-2, 1])

        a = np.swapaxes(a, 0, 1)
        b = np.swapaxes(b, 0, 1)
        c = np.swapaxes(c, 0, 1)

        # Build matrix
        A[2*ftot:2*ftot+2*bandwidth+1, :-2*ftot] += a.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[2*fmax+1:2*fmax+2*bandwidth+2, ftot:-ftot] += b.reshape(a.shape[0], a.shape[1]*a.shape[2])
        A[0:2*bandwidth+1, 2*ftot:] += c.reshape(a.shape[0], a.shape[1]*a.shape[2])

        # Boundary conditions
        #   Surface (k=1)
        b =  N[1, :, :]/(dz[0]*0.5*(dz[0]+dz[1]))
        c = -N[1, :, :]/(dz[0]*0.5*(dz[0]+dz[1]))
        #b[bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA)

        A[2*fmax+1:2*fmax+2*bandwidth+2, :ftot] += b
        A[0:2*bandwidth+1, ftot:2*ftot] += c

        #   Bed (k=kmax)
        a = -N[-2, :, :]/(dz[-1]*0.5*(dz[-1]+dz[-2]))
        b =  N[-2, :, :]/(dz[-1]*0.5*(dz[-1]+dz[-2]))
        #b[bandwidth, :] += (np.arange(-fmax, ftot-fmax)*1j*OMEGA)
        A[2*ftot:, -2*ftot:-ftot] += a
        A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += b

        # |u*|u*
        z0 = data.v('Roughness')
        logfactor = (data.v('KAPPA')/np.log(data.v('ALPHA')+.5*abs(dz[-1])/z0))**2.
        lhs = np.zeros([ftot, ftot], dtype=complex)
        if data.v('u0') is None:
            c = ny.polyApproximation(uuabs, 1)

            lhs[[fmax], :] += -logfactor*c[1]*np.ones([1, ftot])
        else:
            ubed = data.v('u0', [kmax], range(0, fmax+1))*np.log(.5*abs(dz[-1])/z0)
            print ubed
            Uamp = np.sum(abs(ubed))
            ubed_dimless = ubed/Uamp
            ubed_dimless_sq = ny.complexAmplitudeProduct(ubed_dimless, ubed_dimless, 1)
            c = ny.polyApproximation(uuabs, 4)

            lhs[[fmax], :] += -Uamp*logfactor*(c[1] + c[3]*ubed_dimless_sq[:, [0]])*np.ones([1, ftot])
            for n in range(1, fmax+1):
                lhs[[fmax+n], :-n] += -Uamp*logfactor*0.5*c[3]*ubed_dimless_sq[:, [n]]*np.ones([1, ftot-n])
                lhs[[fmax-n], n:] += -Uamp*logfactor*0.5*np.conj(c[3]*ubed_dimless_sq[:, [n]])*np.ones([1, ftot-n])

        A[2*fmax+1:2*fmax+2*bandwidth+2, -ftot:] += lhs[fmax-bandwidth:fmax+bandwidth+1, :]/dz[-1]

        # save matrix
        velocityMatrix = A[Ellipsis]
        bandwidthA = bandwidth+ftot
    else:
        A = Av[Ellipsis]     # if hasMatrix Av replaces the role of the matrix in this equation
        bandwidthA = (A.shape[0]-1)/2


    ##### RIGHT HAND SIDE #####
    uRHS_zeta = np.zeros([ftot*kmax, ftot], dtype=complex)
    uRHS = np.zeros([ftot*kmax, nRHS], dtype=complex)

    # implicit forcing from barotropic pressure term
    uRHS_zeta[ftot:-ftot, :] = np.concatenate([np.diag([1]*ftot)]*(kmax-2))

    # forcing from other factors
    uRHS[:ftot, :] = Fsurf[0, :, :]
    uRHS[-ftot:, :] = Fbed[0, :, :]
    uRHS[ftot:-ftot, :] = F[1:-1, :, :].reshape((kmax-2)*ftot, nRHS)


    ##### SOLVE #####
    u = solve_banded((bandwidthA, bandwidthA), A, np.concatenate((uRHS_zeta, uRHS), 1), overwrite_ab=True, overwrite_b=True)
    u = u.reshape(kmax, ftot, ftot+nRHS)
    uCoef = u[:, :, :ftot]
    uFirst = u[:, :, ftot:]

    #interpolate
    uCoefnew = np.zeros((kmax+1, ftot, ftot),dtype=complex)
    uCoefnew[1:kmax, Ellipsis] = .5*(uCoef[1:,Ellipsis]+uCoef[:-1,Ellipsis])
    z0 = data.v('Roughness')
    dz = (data.v('grid', 'axis', 'z')[1:]-data.v('grid', 'axis', 'z')[:-1])*data.n('H')
    uCoefnew[-1, Ellipsis] = uCoef[-1,Ellipsis]/np.log(.5*abs(dz[-1])/z0)
    uCoefnew[0, Ellipsis] = uCoef[0,Ellipsis]



    return uCoefnew, uFirst, velocityMatrix

def uuabs(u):
    return u*np.abs(u)