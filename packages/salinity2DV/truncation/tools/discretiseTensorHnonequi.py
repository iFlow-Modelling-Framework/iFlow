# import numpy as np
import nifty as ny
from copy import copy
from .toBandedTri import *


def TensorFluxSplitting(U, axis):
    # Flux vector splitting
    Umv = np.moveaxis(U, (1, axis), (-2, -1))

    l, K = np.linalg.eig(Umv)
    if np.sum(abs(np.imag(l)))>1e-14:
        a=0

    Lambda = np.zeros(K.shape, dtype=complex)
    Lambda[:, :, range(0, Lambda.shape[-2]), range(0, Lambda.shape[-1])] = l
    Kinv = np.linalg.inv(K)
    Ap = np.real(np.matmul(np.matmul(K, np.maximum(Lambda,0)), Kinv))
    Am = np.real(np.matmul(np.matmul(K, np.minimum(Lambda,0)), Kinv))

    Ap = np.moveaxis(Ap, (-2, -1), (1, axis))
    Am = np.moveaxis(Am, (-2, -1), (1, axis))

    return Ap, Am

def discretiseTensor(U_tuple, b, order1, axis1, order2, eqno2, varno2, shape, x, U_down=None, U_up=None, postProcess=None, forceCentral=False):
    """
    Compute a banded operator
    Args:
        b:
        order1:
        axis1:
        order2:
        eqno2:
        varno2:
        shape:
        data:
        postProcess: set to a value like 'b' that needs to be used to determine the direction of upwinding, but uses b for actual computation. This is good for postprocessing a decomposition

    Returns:

    """
    timers = [ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer()]
    Ap, Am, U = U_tuple
    M = b.shape[-1]
    jmax = b.shape[0]-1

    if axis1==-1 or axis1==3:
        b = b.reshape((jmax+1, 1, 1, M))
    if axis1==-2 or axis1==2:
        b = b.reshape((jmax+1, 1, M, 1))

    timers[0].tic()
    ####################################################################################################################
    # derivative_order =0 on axis1 and 1 on axis2
    ####################################################################################################################
    if order1==0:
        if order2==0:
            # axis 1
            Ub = np.sum(U*b, axis=axis1)

            # axis 2
            A = np.zeros(Ub.shape)
            B = np.zeros(Ub.shape)
            C = Ub
            D = np.zeros(Ub.shape)
            E = np.zeros(Ub.shape)

        else:   # i.e. order2==1
            if not forceCentral:
                ## Upwind scheme
                # axis 1
                timers[3].tic()
                if postProcess is None:
                    Apb = np.sum(Ap*0.5*(b+np.abs(b))+Am*0.5*(b-np.abs(b)), axis=axis1)
                    Amb = np.sum(Ap*0.5*(b-np.abs(b))+Am*0.5*(b+np.abs(b)), axis=axis1)
                else:
                    Apb = np.sum(Ap*b*0.5*(1+np.sign(postProcess))+Am*b*0.5*(1-np.sign(postProcess)), axis=axis1)
                    Amb = np.sum(Ap*b*0.5*(1-np.sign(postProcess))+Am*b*0.5*(1+np.sign(postProcess)), axis=axis1)
                timers[3].toc()

                A = np.zeros(Apb.shape)
                B = np.zeros(Apb.shape)
                C = np.zeros(Apb.shape)
                D = np.zeros(Apb.shape)
                E = np.zeros(Apb.shape)
                x = x.reshape((Apb.shape[0], 1, 1))

                timers[4].tic()
                # axis 2
                A[2:-2] = 1./(x[3:-1]-x[1:-3])*Apb[2:-2]                            # symmetric upwind scheme Veldman and Lam (2008)
                B[2:-2] = -4./(x[3:-1]-x[1:-3])*Apb[2:-2]
                C[2:-2] = 3./(x[3:-1]-x[1:-3])*Apb[2:-2] - 3./(x[3:-1]-x[1:-3])*Amb[2:-2]
                D[2:-2] = 4./(x[3:-1]-x[1:-3])*Amb[2:-2]
                E[2:-2] = -1./(x[3:-1]-x[1:-3])*Amb[2:-2]

                # boundaries
                 # boundaries
                A[0] = 0                                                # ordinary non-equidistant BDF scheme
                B[0] = 0
                C[0] =  3./(3*x[0]-4*x[1]+x[2])*(Amb[0]+Apb[0])
                D[0] = -4./(3*x[0]-4*x[1]+x[2])*(Amb[0]+Apb[0])
                E[0] =  1./(3*x[0]-4*x[1]+x[2])*(Amb[0]+Apb[0])

                # A[1] = 0                                                # ordinary non-equidistant BDF scheme
                # B[1] = 0
                # C[1] =  3./(3*x[1]-4*x[2]+x[3])*(Amb[1]+Apb[1])
                # D[1] = -4./(3*x[1]-4*x[2]+x[3])*(Amb[1]+Apb[1])
                # E[1] =  1./(3*x[1]-4*x[2]+x[3])*(Amb[1]+Apb[1])
                #
                # A[-2] = 1./(3*x[-2]-4*x[-3]+x[-4])*(Amb[-2]+Apb[-2])
                # B[-2] = -4./(3*x[-2]-4*x[-3]+x[-4])*(Amb[-2]+Apb[-2])
                # C[-2] = 3./(3*x[-2]-4*x[-3]+x[-4])*(Amb[-2]+Apb[-2])
                # D[-2] = 0
                # E[-2] = 0

                A[1] = 0                                                # symmetric upwind scheme Veldman and Lam (2008)
                B[1] = -1./(0.5*(x[2]-x[0]))*Apb[1]
                C[1] =  1./(0.5*(x[2]-x[0]))*Apb[1] -1./(0.5*(x[2]-x[0]))*Amb[1]
                D[1] =  1./(0.5*(x[2]-x[0]))*Amb[1]
                E[1] = 0

                A[-2] = 0
                B[-2] = -1./(0.5*(x[-1]-x[-3]))*Apb[-2]
                C[-2] = 1./(0.5*(x[-1]-x[-3]))*Apb[-2] -1./(0.5*(x[-1]-x[-3]))*Amb[-2]
                D[-2] = 1./(0.5*(x[-1]-x[-3]))*Amb[-2]
                E[-2] = 0

                # A[1] = 0                                              # central scheme
                # B[1] = -1./(x[2]-x[0])*(Apb[1]+Amb[1])
                # C[1] =  0
                # D[1] =  1./(x[2]-x[0])*(Apb[1]+Amb[1])
                # E[1] = 0
                #
                # A[-2] = 0
                # B[-2] = -1./(x[-1]-x[-3])*(Apb[-2]+Amb[-2])
                # C[-2] = 0
                # D[-2] = 1./(x[-1]-x[-3])*(Apb[-2]+Amb[-2])
                # E[-2] = 0

                A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*(Amb[-1]+Apb[-1])
                B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*(Amb[-1]+Apb[-1])
                C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*(Amb[-1]+Apb[-1])
                D[-1] = 0
                E[-1] = 0
                timers[4].toc()
            else:
                ## Central scheme
                Ub = np.sum(U*b, axis=axis1)

                A = np.zeros(Ub.shape)
                B = np.zeros(Ub.shape)
                C = np.zeros(Ub.shape)
                D = np.zeros(Ub.shape)
                E = np.zeros(Ub.shape)
                x = x.reshape((Ub.shape[0], 1, 1))

                B[1:-1] = -1./(x[2:]-x[:-2])*(Ub[1:-1])
                D[1:-1] =  1./(x[2:]-x[:-2])*(Ub[1:-1])

                # A[0] = 0                                                        # ordinary non-equidistant BDF scheme
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
                C[0] = 3./(3*x[0]-4*x[1]+x[2])*(Ub[0])
                D[0] = -4./(3*x[0]-4*x[1]+x[2])*(Ub[0])
                E[0] = 1./(3*x[0]-4*x[1]+x[2])*(Ub[0])

                A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*(Ub[-1])              # ordinary non-equidistant BDF scheme
                B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*(Ub[-1])
                C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*(Ub[-1])
                D[-1] = 0
                E[-1] = 0

            if U_up is not None:
                A[-1] = 1./(3*x[-1]-4*x[-2]+x[-3])*U_up
                B[-1] = -4./(3*x[-1]-4*x[-2]+x[-3])*U_up
                C[-1] = 3./(3*x[-1]-4*x[-2]+x[-3])*U_up
                D[-1] = 0
                E[-1] = 0

            if U_down is not None:
                A[0] = 0
                B[0] = 0
                C[0] =  3./(3*x[0]-4*x[1]+x[2])*U_down
                D[0] = -4./(3*x[0]-4*x[1]+x[2])*U_down
                E[0] =  1./(3*x[0]-4*x[1]+x[2])*U_down




    ####################################################################################################################
    # derivative_order = 1 on axis1 and 0 on axis2
    ####################################################################################################################
    elif order1==1:
        # Ap = Ap*0.5*(np.sign(b)+1) + Am*0.5*(np.sign(b)-1)        # use estimate of the sign of b using previous result; does not work
        # Am = Ap*0.5*(np.sign(b)-1) + Am*0.5*(np.sign(b)+1)
        #
        # # axis 1
        # A = 1./(2.*dx)*Ap
        # B = -4./(2.*dx)*Ap
        # C = 3./(2.*dx)*Ap - 3./(2.*dx)*Am
        # D = 4./(2.*dx)*Am
        # E = -1./(2.*dx)*Am
        #
        # Ub = np.zeros((jmax+1, M, M))
        # Ub[2:-2] = np.sum(A[2:-2]*b[:-4]+B[2:-2]*b[1:-3]+C[2:-2]*b[2:-2]+D[2:-2]*b[3:-1]+E[2:-2]*b[4:], axis=axis1)
        #
        # #       boundaries
        # Ub[0] = np.sum(((Am+Ap)[0]/dx)*(-1./2.*b[2] + 2.*b[1] -3./2.*b[0]), axis=axis1)
        # Ub[1] = np.sum((Ap[1]/dx)*(b[1]-b[0]) + (Am[1]/dx)*(b[2]-b[1]), axis=axis1)
        #
        # Ub[-1] = np.sum(((Am+Ap)[-1]/dx)*(1./2.*b[-3] - 2.*b[-2] +3./2.*b[-1]), axis=axis1)
        # Ub[-2] = np.sum((Ap[-2]/dx)*(b[-2]-b[-3]) + (Am[-2]/dx)*(b[-1]-b[-2]), axis=axis1)

        Ub = np.sum(U*np.gradient(b, x, edge_order=2, axis=0), axis=axis1)

        # axis 2
        A = np.zeros(Ub.shape)
        B = np.zeros(Ub.shape)
        C = copy(Ub)
        D = np.zeros(Ub.shape)
        E = np.zeros(Ub.shape)

        if U_up is not None:
            A[-1] = 0
            B[-1] = 0
            C[-1] = U_up
            D[-1] = 0
            E[-1] = 0

        if U_down is not None:
            A[0] = 0
            B[0] = 0
            C[0] = U_down
            D[0] = 0
            E[0] = 0

    timers[0].toc()
    ####################################################################################################################
    # Total matrices
    ####################################################################################################################
    timers[1].tic()
    Atot = np.zeros((jmax+1, sum(shape), sum(shape)))
    Btot = np.zeros((jmax+1, sum(shape), sum(shape)))
    Ctot = np.zeros((jmax+1, sum(shape), sum(shape)))
    Dtot = np.zeros((jmax+1, sum(shape), sum(shape)))
    Etot = np.zeros((jmax+1, sum(shape), sum(shape)))

    Atot[:, sum(shape[:eqno2]):sum(shape[:eqno2+1]), sum(shape[:varno2]):sum(shape[:varno2+1])] = A
    Btot[:, sum(shape[:eqno2]):sum(shape[:eqno2+1]), sum(shape[:varno2]):sum(shape[:varno2+1])] = B
    Ctot[:, sum(shape[:eqno2]):sum(shape[:eqno2+1]), sum(shape[:varno2]):sum(shape[:varno2+1])] = C
    Dtot[:, sum(shape[:eqno2]):sum(shape[:eqno2+1]), sum(shape[:varno2]):sum(shape[:varno2+1])] = D
    Etot[:, sum(shape[:eqno2]):sum(shape[:eqno2+1]), sum(shape[:varno2]):sum(shape[:varno2+1])] = E
    timers[1].toc()

    timers[2].tic()
    Matbanded = toBandedPenta(Atot, Btot, Ctot, Dtot, Etot)
    timers[2].toc()

    # timers[0].disp('    Derivatives ' + str(order1) + ', '+str(order2))
    # timers[1].disp('    Total matrices')
    # timers[2].disp('    Penta')
    # timers[3].disp('        Preping')
    # timers[4].disp('        Discretisation')

    return Matbanded
