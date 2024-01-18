import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
import nifty as ny
colours = st.configure()
Ls_contour = 1

"""
Linear dispersion model
Ast = Qs_x +AKh s_xx = 0
with s(0,t)=ssea

Solved assuming Q=Q^0+Q^1(t)
All other parameters constant.
Q^1(t) is assumed periodic -> use FT.
"""

def fs(t, c):
    return np.sum(np.real(c*np.exp(np.arange(len(c))*1j*omega*t)))

def fs_vec(t, c):
    c = c.reshape((len(c), 1))
    t = t.reshape((1,len(t)))
    n = np.arange(len(c)).reshape((len(c),1))
    return np.sum(np.real(c*np.exp(n*1j*omega*t)), axis=0)

########################################################################################################################
# Parameters
########################################################################################################################
Kh0 = 100
A = 1e4
ssea0 = 15
sseavec = [0.1,5,10,13]
Q0 = 100
k = 3.4
N = 10
L = k*A*Kh0/Q0
jmax = 100

########################################################################################################################
# Prep
#######################################################################################################################
Tadj0 = ((Q0/(A*Kh0))**2*Kh0)**(-1)

Tvec = 10**np.linspace(-1,6,60)*24*3600
Tadj_est = np.zeros(len(sseavec))
Tadj_max = np.zeros(len(sseavec))
Tadj_time = np.zeros((len(sseavec), 1500))
Tdelay = np.zeros((len(Tvec),len(sseavec)))
phi = np.zeros((len(Tvec),len(sseavec)))
magn = np.zeros((len(Tvec),len(sseavec)))
for qqq, ssea1 in enumerate(sseavec):
    alpha = ssea1/ssea0
    Kh1 = 0
    
    # Tvec = 10**np.linspace(4,5,6)*24*3600

    # Tadj_est[qqq] = Tadj0*(1+6/8*alpha**2+5/8*alpha**4+35/64*alpha**6+alpha**8)
    # Tadj_max[qqq] = (((Q0-Q1)/(A*Kh))**2*Kh)**(-1)
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
        Qmat = np.diag(Q0*np.ones(2*N+1))
        Dmat = np.diag(np.arange(-N,N+1)*1j*omega)
        AKHmat = A*(np.diag(Kh0*np.ones(2*N+1)) + .5*np.diag(np.conj(Kh1)*np.ones(2*N),1) + .5*np.diag(Kh1*np.ones(2*N),-1))

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
        b[0] = ssea0
        b[N] = 0.5*np.conj(ssea1)
        b[N+1] = 0.5*ssea1

        l_neg = l[indices]
        if len(l_neg)!=2*N+1:
            print(l_neg)
        P_neg = P[:, indices]
        P_reduced = P_neg[rows,:]
        c = np.dot(np.linalg.inv(P_reduced), b)

        s00 = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*0))), c)
        s0max = np.max(ny.invfft2(s00,0,1500))
        sigma = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*L))), c)
        s = ny.eliminateNegativeFourier(sigma[:2*N+1],0)/s0max*30
        s0[qq] = s[0]
        s1[qq] = s[1]

        # smax[qq] = np.max(ny.invfft2(s,0,1500))
        # tmax[qq] = (np.linspace(0,1,1000)[np.argmax(ny.invfft2(s,0,1000))])*T
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

    Tdelay[:,qqq] = (-(np.angle(s1)-np.angle(ssea1))/(2*np.pi))*Tvec
    magn[:,qqq] = s1/s0*ssea0/ssea1
    phi[:,qqq] = np.angle(s1)*180/np.pi

########################################################################################################################
# Plot
########################################################################################################################
### Fig 1 ###
plt.figure(1, figsize=(1,2))
plt.subplot(1,2,1)
plt.plot(Tvec/Tadj0, Tvec/Tadj0, 'k--')
plt.text(0.15, 2, '$T_{delay}>T$', fontsize=6)
plt.fill_between(Tvec/Tadj0, Tvec/Tadj0, 20, color='grey')

plt.plot(Tvec/Tadj0, 0.25*Tvec/Tadj0, '--', color=colours[5])
plt.text(7.5, 3.0, '90 deg', fontsize=6, color=colours[5], rotation=75)

for qqq, Q1 in enumerate(sseavec):
    #plt.plot(Tvec/Tadj_avg[qqq],  Tdelay[:,qqq]/Tadj_avg[qqq], label=r'$\alpha=$'+str(Q1/Q0))
    plt.plot(Tvec/Tadj0,  Tdelay[:,qqq]/Tadj0, label=r'$\alpha=$'+str(Q1/Q0))
plt.legend()
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.ylim(0, 20)
plt.xlim(1e-1,1e5)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

plt.subplot(1,2,2)
plt.plot(Tvec/Tadj0, 0.25*np.ones(len(Tvec)), '--', color=colours[5])
plt.text(100, 0.25*0.98, '90 deg', fontsize=6, color=colours[5], va='top')

for qqq, Q1 in enumerate(sseavec):
    plt.plot(Tvec/Tadj0,  Tdelay[:,qqq]/Tvec)

plt.ylabel('$T_{delay}/T$')
plt.ylim(0, None)
plt.xlim(1e-1,1e5)
plt.xlabel('$T/T_{adj}$')
plt.xscale('log')

### Fig 2 ###
# plt.figure(2, figsize=(2,2))
# plt.subplot(2,1,1)
# Tadj_avg = np.mean(np.sqrt(Tadj_time), axis=-1)**2
# for qqq, Q1 in enumerate(sseavec):
#     p = plt.plot(np.linspace(0, 1, 1500), Tadj_time[qqq], label=r'$\alpha=$'+str(Q1/Q0))
#     plt.plot([0,1], [Tadj_avg[qqq]]*2, '--', color=p[0].get_color())
#
# plt.subplot(2,2,3)
# plt.plot(Tvec/Tadj0, Tvec/Tadj0, 'k--')
# plt.text(0.15, 2, '$T_{delay}>T$', fontsize=6)
# plt.fill_between(Tvec/Tadj0, Tvec/Tadj0, 20, color='grey')
#
# plt.plot(Tvec/Tadj0, 0.25*Tvec/Tadj0, '--', color=colours[5])
# plt.text((Tvec/Tadj0)[int(len(Tvec)/2.2)]*0.7, 0.25*(Tvec/Tadj0)[int(len(Tvec)/2.2)], '90 deg', fontsize=6, color=colours[5], rotation=80)
#
# for qqq, Q1 in enumerate(sseavec):
#     #plt.plot(Tvec/Tadj_avg[qqq],  Tdelay[:,qqq]/Tadj_avg[qqq], label=r'$\alpha=$'+str(Q1/Q0))
#     plt.plot(Tvec/Tadj0,  Tdelay[:,qqq]/Tadj0, label=r'$\alpha=$'+str(Q1/Q0))
# plt.legend()
# plt.ylabel('$T_{delay}/T_{adj,0}$')
# plt.ylim(0, 20)
# plt.xlabel('$T_{forcing}/T_{adj,0}$')
# plt.xscale('log')
#
# plt.subplot(2,2,4)
# plt.plot(Tvec/Tadj0, Tvec/Tadj0, 'k--')
# plt.text(0.15, 2, '$T_{delay}>T$', fontsize=6)
# # plt.fill_between(Tvec/Tadj0, Tvec/Tadj0, 1.5, color='grey')
#
# plt.plot(Tvec/Tadj0, 0.25*Tvec/Tadj0, '--', color=colours[5])
# plt.text((Tvec/Tadj0)[int(len(Tvec)/2.2)]*0.7, 0.25*(Tvec/Tadj0)[int(len(Tvec)/2.2)], '90 deg', fontsize=6, color=colours[5], rotation=80)
#
# for qqq, Q1 in enumerate(sseavec):
#     #plt.plot(Tvec/Tdelay[-1,qqq],  Tdelay[:,qqq]/Tdelay[-1,qqq])
#     plt.plot(Tvec/Tadj_est[qqq],  Tdelay[:,qqq]/Tadj_est[qqq])
#
# plt.ylabel('$T_{delay}/T_{adj}$')
# plt.ylim(0, 5)
# plt.xlabel('$T_{forcing}/T_{adj}$')
# plt.xscale('log')

st.show()

