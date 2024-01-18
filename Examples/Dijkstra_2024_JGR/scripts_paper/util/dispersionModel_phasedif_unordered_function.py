import step as st
import scipy.interpolate
import scipy.optimize
import scipy.linalg
import nifty as ny
import numpy as np

def delay_harmonic(Kh, A, ssea, Q0, Qvec, k, N):
    ########################################################################################################################
    # Prep
    #######################################################################################################################
    Tadj0 = ((Q0/(A*Kh))**2*Kh)**(-1)
    L = k*A*Kh/Q0

    Tvec = 10**np.linspace(-1,6,60)*24*3600
    Tadj_est = np.zeros(len(Qvec))
    Tadj_max = np.zeros(len(Qvec))
    Tadj_time = np.zeros((len(Qvec), 1500))
    Tdelay = np.zeros((len(Tvec),len(Qvec)))
    phi = np.zeros((len(Tvec),len(Qvec)))
    magn = np.zeros((len(Tvec),len(Qvec)))
    for qqq, Q1 in enumerate(Qvec):
        alpha = Q1/Q0

        # Tvec = 10**np.linspace(4,5,6)*24*3600

        Tadj_est[qqq] = Tadj0*(1+6/8*alpha**2+5/8*alpha**4+35/64*alpha**6+alpha**8)
        Tadj_max[qqq] = (((Q0-Q1)/(A*Kh))**2*Kh)**(-1)
        print(Tadj0/(3600*24), Tadj_max[qqq]/(3600*24))
        s1 = np.zeros(len(Tvec), dtype=complex)
        s0 = np.zeros(len(Tvec), dtype=complex)
        smax = np.zeros(len(Tvec))
        tmax = np.zeros(len(Tvec))
        for qq,T in enumerate(Tvec):
            ########################################################################################################################
            # Computation
            ########################################################################################################################
            omega = 2*np.pi/T
            Qmat = np.diag(Q0*np.ones(2*N+1)) + .5*np.diag(np.conj(Q1)*np.ones(2*N),1) + .5*np.diag(Q1*np.ones(2*N),-1)
            Dmat = np.diag(np.arange(-N,N+1)*1j*omega)
            AKHmat = A*Kh*np.eye(2*N+1)

            B = np.zeros((2*(2*N+1), 2*(2*N+1)), dtype=complex)
            C = np.zeros((2*(2*N+1), 2*(2*N+1)), dtype=complex)
            B[:2*N+1,:2*N+1] = Qmat
            B[:2*N+1:,2*N+1:] = AKHmat
            B[2*N+1:,:2*N+1] = np.eye(2*N+1)

            C[:2*N+1,:2*N+1] = A*Dmat
            C[2*N+1:,2*N+1:] = np.eye(2*N+1)

            M = np.matmul(np.linalg.inv(B),C)
            l, P = np.linalg.eig(M)
            indices = [i for i in range(0, len(l)) if np.real(l[i])<-1e-19]
            # rows = [N] + list(range(2*N+1,3*N+1)) + list(range(3*N+2,4*N+2))    # pseudo-neumann
            rows = [N] + list(range(0,N)) + list(range(N+1,2*N+1))    # dirichlet
            b = np.zeros(2*N+1)
            b[0] = ssea

            l_neg = l[indices]
            if len(l_neg)!=2*N+1:
                print(l_neg)
            P_neg = P[:, indices]
            P_reduced = P_neg[rows,:]
            c = np.dot(np.linalg.inv(P_reduced), b)
            #
            s00 = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*0))), c)
            s0max = np.max(ny.invfft2(s00,0,1500))
            sigma = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*L))), c)
            s = ny.eliminateNegativeFourier(sigma[:2*N+1],0)/s0max*30
            s0[qq] = s[0]
            s1[qq] = s[1]

            smax[qq] = np.max(ny.invfft2(s,0,1500))
            tmax[qq] = (np.linspace(0,1,1000)[np.argmax(ny.invfft2(s,0,1000))])*T
            # tmax[qq] = (scipy.optimize.fmin(fs, tmax[qq], (-s,), xtol=1e-9, ftol=1e-12, disp=0))%T
            # tmax[qq] = (scipy.optimize.fmin(fs, tmax[qq], (-s,), xtol=1e-9, ftol=1e-12, disp=0)-scipy.optimize.fmin(fs, tmax[qq], (np.asarray([Q0,Q1]),), xtol=1e-6, ftol=1e-8, disp=0))%T
            # print(tmax[qq]/Tadj0)

        ########################################################################################################################
        # time dependent adjustment time
        ########################################################################################################################
        # s_time = ny.invfft2(s,0,1500)
        # Ls = (-1/L*np.log(s_time/30))**(-1)
        # Q = Q0 + Q1*np.cos(2*np.pi*np.linspace(0,1,1500))
        # Tadj_time[qqq,:] = A*Ls/Q

        Tdelay[:,qqq] = (0.5-np.angle(s1)/(2*np.pi))*Tvec
        magn[:,qqq] = abs(s1/s0*Q0/Q1)
        phi[:,qqq] = np.angle(s1)*180/np.pi
    return Tvec, Tdelay, magn, phi

########################################################################################################################

