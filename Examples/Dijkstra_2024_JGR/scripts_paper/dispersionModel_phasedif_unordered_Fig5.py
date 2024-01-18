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
Khvec = [0,0,0,0,0, 1,50,70,80,90]
A = 1e4
ssea = 30
Q0 = 100
Qvec = [1,50,70,80,90,0,0,0,0,0]
k = 3.4
N = 15
L = k*A*Kh0/Q0
jmax = 100
tmax = 500 # for fft in post-processing

########################################################################################################################
# Prep
#######################################################################################################################
Tadj0 = ((Q0/(A*Kh0))**2*Kh0)**(-1)

iso = ssea*np.exp(-k)

Tvec = 10**np.linspace(-1,6.2,60)*24*3600
Tadj_est = np.zeros(len(Qvec))
Tadj_max = np.zeros(len(Qvec))
Tadj_time = np.zeros((len(Qvec), 1500))
Tdelay = np.zeros((len(Tvec),len(Qvec)))
phi = np.zeros((len(Tvec),len(Qvec)))
magn = np.zeros((len(Tvec),len(Qvec)))

s_time = np.zeros((len(Qvec), tmax))
Q_time = np.zeros((len(Qvec), tmax))
Kh_time = np.zeros((len(Qvec), tmax))

for qqq, Q1 in enumerate(Qvec):
    Kh1 = Khvec[qqq]
    s1 = np.zeros(len(Tvec), dtype=complex)
    s0 = np.zeros(len(Tvec), dtype=complex)
    smax = np.zeros(len(Tvec))
    # tmax = np.zeros(len(Tvec))
    for qq,T in enumerate(Tvec):
        ########################################################################################################################
        # Computation
        ########################################################################################################################
        omega = 2*np.pi/T
        Qmat = np.diag(Q0*np.ones(2*N+1)) + .5*np.diag(np.conj(Q1)*np.ones(2*N),1) + .5*np.diag(Q1*np.ones(2*N),-1)
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
        b[0] = ssea

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

    ########################################################################################################################
    # time dependent adjustment time
    ########################################################################################################################
    s_time[qqq,:] = ny.invfft2(s,0,tmax)
    Ls = (-1/L*np.log(s_time/30))**(-1)
    Q_time[qqq,:] = Q0 + Q1*np.cos(2*np.pi*np.linspace(0,1,tmax))
    Kh_time[qqq,:] = Kh0 + Kh1*np.cos(2*np.pi*np.linspace(0,1,tmax))

    if qqq>=5:
        Tdelay[:,qqq] = (-(np.angle(s1)-np.angle(Q1))/(2*np.pi))*Tvec
    else:
        Tdelay[:,qqq] = (0.5-(np.angle(s1)-np.angle(Q1))/(2*np.pi))*Tvec
    magn[:,qqq] = s1/s0*Q0/Q1
    phi[:,qqq] = np.angle(s1)*180/np.pi

########################################################################################################################
# Plot
########################################################################################################################
### Fig 1 ###
plt.figure(1, figsize=(1,2))
plt.subplot(1,2,1)
plt.plot(Tvec/Tadj0, Tvec/Tadj0, 'k--')
# plt.text(0.15, 5, '$T_{delay}>T_{forcing}$', fontsize=5)
plt.fill_between(Tvec/Tadj0, Tvec/Tadj0, 20, color='grey')

plt.plot(Tvec/Tadj0, 0.25*Tvec/Tadj0, '--', color=colours[5])
plt.text(7.5, 3.0, '90 deg', fontsize=6, color=colours[5], rotation=75)

for qqq, Q1 in enumerate(Qvec[:5]):
    #plt.plot(Tvec/Tadj_avg[qqq],  Tdelay[:,qqq]/Tadj_avg[qqq], label=r'$\alpha=$'+str(Q1/Q0))
    plt.plot(Tvec/Tadj0,  Tdelay[:,qqq]/Tadj0, label=r'$|\hat{Q}_1|/Q_0=$'+str(Q1/Q0))
plt.legend(fontsize=5.5)
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.ylim(0, 20)
plt.xlim(1e-1,1e5)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

plt.subplot(1,2,2)
plt.plot(Tvec/Tadj0, Tvec/Tadj0, 'k--')
# plt.text(0.15, 2, '$T_{delay}>T_{forcing}$', fontsize=5)
plt.fill_between(Tvec/Tadj0, Tvec/Tadj0, 20, color='grey')

plt.plot(Tvec/Tadj0, 0.25*Tvec/Tadj0, '--', color=colours[5])
plt.text(6, 2.3, '90 deg', fontsize=6, color=colours[5], rotation=82)

for qqq, Kh1 in enumerate(Khvec[5:]):
    #plt.plot(Tvec/Tadj_avg[qqq],  Tdelay[:,qqq]/Tadj_avg[qqq], label=r'$\alpha=$'+str(Q1/Q0))
    plt.plot(Tvec/Tadj0,  Tdelay[:,qqq+5]/Tadj0, label=r'$|\hat{K}_{h,1}|/K_{h,0}=$'+str(Kh1/Kh0))
plt.legend(fontsize=5.5, ncol=2)
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.ylim(0, 7)
plt.xlim(1e-1,1e5)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

# Tadj_avg = np.mean(np.sqrt(Tadj_time), axis=-1)**2
# for qqq, Q1 in enumerate(Qvec):
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
# for qqq, Q1 in enumerate(Qvec):
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
# for qqq, Q1 in enumerate(Qvec):
#     #plt.plot(Tvec/Tdelay[-1,qqq],  Tdelay[:,qqq]/Tdelay[-1,qqq])
#     plt.plot(Tvec/Tadj_est[qqq],  Tdelay[:,qqq]/Tadj_est[qqq])
#
# plt.ylabel('$T_{delay}/T_{adj}$')
# plt.ylim(0, 5)
# plt.xlabel('$T_{forcing}/T_{adj}$')
# plt.xscale('log')

st.show()

