# import numpy as np
import nifty as ny
from copy import copy
from .toBandedTri import *


def discretiseOperator(U, order, eqno, varno, shape, x, U_up=None, U_down=None, forceCentral=False):
    timers = [ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer()]
    A  = np.zeros(U.shape)
    B = np.zeros(U.shape)
    C = np.zeros(U.shape)
    D = np.zeros(U.shape)
    E = np.zeros(U.shape)
    x = x.reshape((U.shape[0], 1, 1))

    ####################################################################################################################
    # Interior
    ####################################################################################################################
    timers[0].tic()
    # Reaction terms
    if order==0:
        C = copy(U)

    # Advection terms
    elif order==1 and not forceCentral:
        # Flux vector splitting - first order upwind
        if U.shape[-1]==U.shape[-2]:        # for square matrices
            l, K = np.linalg.eig(U)
            Lambda = np.zeros(K.shape, dtype=complex)
            Lambda[:, range(0, Lambda.shape[1]), range(0, Lambda.shape[2])] = l
            Kinv = np.linalg.inv(K)
            Ap = np.real(np.matmul(np.matmul(K, np.maximum(Lambda,0)), Kinv))
            Am = np.real(np.matmul(np.matmul(K, np.minimum(Lambda,0)), Kinv))
        elif U.shape[-1]<U.shape[-2]:
            Ap = np.zeros(U.shape)
            Am = np.zeros(U.shape)
            for i in range(int(U.shape[-2]/U.shape[-1])):
                subU = U[:, i*U.shape[-1]:(i+1)*U.shape[-1], :]
                checkSubU = copy(subU)
                checkSubU[:, range(0, U.shape[-1]), range(0, U.shape[-1])] = 0
                if np.count_nonzero(checkSubU)==0:     # if the submatrix is a diagonal matrix
                    Ap[:, i*U.shape[-1]:(i+1)*U.shape[-1], :] = np.maximum(subU,0)
                    Am[:, i*U.shape[-1]:(i+1)*U.shape[-1], :] = np.minimum(subU,0)
                else:
                    raise Exception('flux splitting of non-square non-diagonal matrix not implemented')
        else:
            Ap = np.zeros(U.shape)
            Am = np.zeros(U.shape)
            for i in range(int(U.shape[-1]/U.shape[-2])):
                subU = U[:, :, i*U.shape[-2]:(i+1)*U.shape[-2]]
                checkSubU = copy(subU)
                checkSubU[:, range(0, U.shape[-2]), range(0, U.shape[-2])] = 0
                if np.count_nonzero(checkSubU)==0:     # if the submatrix is a diagonal matrix
                    Ap[:, :, i*U.shape[-2]:(i+1)*U.shape[-2]] = np.maximum(subU,0)
                    Am[:, :, i*U.shape[-2]:(i+1)*U.shape[-2]] = np.minimum(subU,0)
                else:
                    raise Exception('flux splitting of non-square non-diagonal matrix not implemented')

        A[2:-2] = 1./(x[3:-1]-x[1:-3])*Ap[2:-2]                         # symmetric upwind scheme Veldman and Lam (2008)
        B[2:-2] = -4./(x[3:-1]-x[1:-3])*Ap[2:-2]
        C[2:-2] = 3./(x[3:-1]-x[1:-3])*Ap[2:-2] - 3./(x[3:-1]-x[1:-3])*Am[2:-2]
        D[2:-2] = 4./(x[3:-1]-x[1:-3])*Am[2:-2]
        E[2:-2] = -1./(x[3:-1]-x[1:-3])*Am[2:-2]

        # boundaries
        A[0] = 0                                                        # ordinary non-equidistant BDF scheme
        B[0] = 0
        C[0] = 3./(3*x[0]-4*x[1]+x[2])*(Am[0]+Ap[0])
        D[0] = -4./(3*x[0]-4*x[1]+x[2])*(Am[0]+Ap[0])
        E[0] = 1./(3*x[0]-4*x[1]+x[2])*(Am[0]+Ap[0])

        # A[1] = 0                                                        # ordinary non-equidistant BDF scheme
        # B[1] = 0
        # C[1] = 3./(3*x[1]-4*x[2]+x[3])*(Am[1]+Ap[1])
        # D[1] = -4./(3*x[1]-4*x[2]+x[3])*(Am[1]+Ap[1])
        # E[1] = 1./(3*x[1]-4*x[2]+x[3])*(Am[1]+Ap[1])
        #
        # A[-2] = 1./(3*x[-2]-4*x[-3]+x[-4])*(Am[-2]+Ap[-2])              # ordinary non-equidistant BDF scheme
        # B[-2] = -4./(3*x[-2]-4*x[-3]+x[-4])*(Am[-2]+Ap[-2])
        # C[-2] = 3./(3*x[-2]-4*x[-3]+x[-4])*(Am[-2]+Ap[-2])
        # D[-2] = 0
        # E[-2] = 0

        # A[1] = 0                                                      # upwind
        # B[1] = -1./(0.5*(x[2]-x[0]))*Ap[1]
        # C[1] =  1./(0.5*(x[2]-x[0]))*Ap[1] -1./(0.5*(x[2]-x[0]))*Am[1]
        # D[1] =  1./(0.5*(x[2]-x[0]))*Am[1]
        # E[1] = 0
        #
        # A[-2] = 0
        # B[-2] = -1./(0.5*(x[-1]-x[-3]))*Ap[-2]
        # C[-2] = 1./(0.5*(x[-1]-x[-3]))*Ap[-2] -1./(0.5*(x[-1]-x[-3]))*Am[-2]
        # D[-2] = 1./(0.5*(x[-1]-x[-3]))*Am[-2]
        # E[-2] = 0

        A[1] = 0                                                        # central scheme
        B[1] = -1./(x[2]-x[0])*(Ap[1]+Am[1])
        C[1] =  0
        D[1] =  1./(x[2]-x[0])*(Ap[1]+Am[1])
        E[1] = 0

        A[-2] = 0
        B[-2] = -1./(x[-1]-x[-3])*(Ap[-2]+Am[-2])
        C[-2] = 0
        D[-2] = 1./(x[-1]-x[-3])*(Ap[-2]+Am[-2])
        E[-2] = 0

        A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*(Am[-1]+Ap[-1])              # ordinary non-equidistant BDF scheme
        B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*(Am[-1]+Ap[-1])
        C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*(Am[-1]+Ap[-1])
        D[-1] = 0
        E[-1] = 0

    elif order==1 and forceCentral:
        # central scheme
        B[1:-1] = -1./(x[2:]-x[:-2])*(U[1:-1])
        D[1:-1] =  1./(x[2:]-x[:-2])*(U[1:-1])

        # A[0] = 0
        # B[0] = 0
        # C[0] = -1./(x[1]-x[0])*(U[0])
        # D[0] = 1./(x[1]-x[0])*(U[0])
        # E[0] = 0
        #
        # A[-1] = 0
        # B[-1] = -1./(x[-1]-x[-2])*(U[-1])
        # C[-1] = 1./(x[-1]-x[-2])*(U[-1])
        # D[-1] = 0
        # E[-1] = 0

        A[0] = 0                                                        # ordinary non-equidistant BDF scheme
        B[0] = 0
        C[0] = 3./(3*x[0]-4*x[1]+x[2])*(U[0])
        D[0] = -4./(3*x[0]-4*x[1]+x[2])*(U[0])
        E[0] = 1./(3*x[0]-4*x[1]+x[2])*(U[0])

        A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*(U[-1])              # ordinary non-equidistant BDF scheme
        B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*(U[-1])
        C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*(U[-1])
        D[-1] = 0
        E[-1] = 0

    # Diffusion terms
    else: #(order==2)   # nb not on the boundaries
        B[1:-1] = (U[1:-1]/(x[1:-1]-x[:-2]))/(0.5*(x[2:]-x[:-2]))
        C[1:-1] = (- U[1:-1]/(x[2:]-x[1:-1]) - U[1:-1]/(x[1:-1]-x[:-2]))/(0.5*(x[2:]-x[:-2]))
        D[1:-1] = (U[1:-1]/(x[2:]-x[1:-1]))/(0.5*(x[2:]-x[:-2]))

    timers[0].toc()
    ####################################################################################################################
    # BCs
    ####################################################################################################################
    timers[1].tic()
    if U_up is not None:
        if order==0:
            A[-1] = 0
            B[-1] = 0
            C[-1] = U_up
            D[-1] = 0
            E[-1] = 0
        elif order==1:
            A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*U_up
            B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*U_up
            C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*U_up
            D[-1] = 0
            E[-1] = 0

    if U_down is not None:
        if order==0:
            A[0] = 0
            B[0] = 0
            C[0] = U_down
            D[0] = 0
            E[0] = 0
        elif order==1:
            A[0] = 0
            B[0] = 0
            C[0] = 3./(3*x[0]-4*x[1]+x[2])*U_down
            D[0] = -4./(3*x[0]-4*x[1]+x[2])*U_down
            E[0] = 1./(3*x[0]-4*x[1]+x[2])*U_down

    timers[1].toc()
    ####################################################################################################################
    # Total matrices
    ####################################################################################################################
    timers[2].tic()
    Atot = np.zeros((U.shape[0], sum(shape), sum(shape)))
    Btot = np.zeros((U.shape[0], sum(shape), sum(shape)))
    Ctot = np.zeros((U.shape[0], sum(shape), sum(shape)))
    Dtot = np.zeros((U.shape[0], sum(shape), sum(shape)))
    Etot = np.zeros((U.shape[0], sum(shape), sum(shape)))

    Atot[:, sum(shape[:eqno]):sum(shape[:eqno+1]), sum(shape[:varno]):sum(shape[:varno+1])] = A
    Btot[:, sum(shape[:eqno]):sum(shape[:eqno+1]), sum(shape[:varno]):sum(shape[:varno+1])] = B
    Ctot[:, sum(shape[:eqno]):sum(shape[:eqno+1]), sum(shape[:varno]):sum(shape[:varno+1])] = C
    Dtot[:, sum(shape[:eqno]):sum(shape[:eqno+1]), sum(shape[:varno]):sum(shape[:varno+1])] = D
    Etot[:, sum(shape[:eqno]):sum(shape[:eqno+1]), sum(shape[:varno]):sum(shape[:varno+1])] = E
    timers[2].toc()

    timers[3].tic()
    Matbanded = toBandedPenta(Atot, Btot, Ctot, Dtot, Etot)
    timers[3].toc()

    # timers[0].disp('build order '+str(order))
    # timers[1].disp('bc')
    # timers[2].disp('large matrix')
    # timers[3].disp('toPenta')

    return Matbanded