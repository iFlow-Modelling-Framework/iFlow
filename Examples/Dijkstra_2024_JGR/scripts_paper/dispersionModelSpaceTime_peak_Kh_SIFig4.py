import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
from adjustment.paper.util.dispersionModel_phasedif_unordered_function import delay_harmonic
colours = st.configure()
from adjustment.paper.util.dispersionModelSpaceTime_peak_function import dispersion_peak


########################################################################################################################
## Settings
########################################################################################################################
L =1e5
T = 365*24*3600
jmax = 1500
tmax = 365*3
A0 = 1e4
k = 3.4

Q0_vec = [800,800,800,400,400,400,100,100,100]
Kh0 = 100
ssea0 = 30
amp_vec = [10,100,500,10,100,500,10,100,500]
Pvec = [4,8,16,4,8,16,4,8,16]

########################################################################################################################
## Theory lines
########################################################################################################################
Q0 = Q0_vec[-1]
Tadj_eq = A0**2/Q0**2*Kh0
Qvec = np.asarray([0.01])*Q0
Tvec_theory, Tdelay_theory, _, _ = delay_harmonic(Kh0, A0, ssea0, Q0, Qvec, k, 20)

plt.figure(5, figsize=(1,2))
plt.subplot(1,2,2)
plt.fill_between(10**np.linspace(-1,2,100), 10**np.linspace(-1,2,100), 10.1*np.ones(100), color='grey')
plt.plot(10**np.linspace(-1,2,100),10**np.linspace(-1,2,100), 'k-')
plt.plot(10**np.linspace(-1,2,100),.25*10**np.linspace(-1,2,100), '--', color=colours[5])
plt.text(6, 1.9, '90 deg', fontsize=6, color=colours[5], va='bottom', rotation=70)

for qqq, Q1 in enumerate(Qvec):
    plt.plot(Tvec_theory/Tadj_eq,  Tdelay_theory[:,qqq]/Tadj_eq, color=colours[qqq], label='$|\hat{K}_{h,1}|/K_{h,0}=$'+str(Q1/Q0))
plt.legend(fontsize=5.5)

########################################################################################################################
## Computations
############ ############################################################################################################
x = np.linspace(0, L, jmax)
t = np.linspace(0, T, tmax)
s = np.zeros((jmax, tmax))
dx = x[1]-x[0]
dt = t[1]-t[0]

Kh = Kh0*np.ones((tmax))
ssea = ssea0*np.ones(tmax)

for q, P in enumerate(Pvec):
    print(q)
    Q0 = Q0_vec[q]
    Q = Q0*np.ones(tmax)

    amp = amp_vec[q]
    Kh = Kh0*np.ones((tmax))
    Kh[np.where(abs(t-0.5*T)<=.25*T/P)] += amp*(np.cos((t-.5*T)/T*2*np.pi*P))[np.where(abs(t-0.5*T)<=.25*T/P)]

    x_k = k*A0*Kh0/Q0

    s2, tforcing_matrix, Tadj_matrix, Tdelay = dispersion_peak(x, x_k, t, Q, Kh, ssea, A0, var='Kh')


    ########################################################################################################################
    ## Plots
    ########################################################################################################################
    plt.figure(5, figsize=(1,2))
    if q<3:
        plt.subplot(3,2,1)
        plt.yticks([100,200,600])
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')

    elif q<6:
        plt.subplot(3,2,3)
        plt.yticks([100,200,600])
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')
    else:
        plt.subplot(3,2,5)
        plt.yticks([100,200,600])
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')

    p=plt.plot(t/(3600*24), Kh, color=colours[q])
    ax = plt.gca()
    if q<6:
        ax.set_xticklabels([])
    ax.ticklabel_format(axis='y',useOffset=False, style='plain')

    plt.subplot(1,2,2)
    plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((Tdelay/Tadj_matrix)).flatten(), '.', color=p[0].get_color(), markersize=0.5)
    plt.plot(((tforcing_matrix/Tadj_matrix)[:,int(len(t)/2)]).flatten(),((Tdelay/Tadj_matrix)[:,int(len(t)/2)]).flatten(), 'o', color=p[0].get_color())

plt.subplot(3,2,5)
plt.xlabel('t (d)')
plt.subplot(3,2,3)
plt.ylabel('Kh ($m^2/s$)')

plt.subplot(1,2,2)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.xscale('log')
plt.ylim(0, 10)
plt.xlim(1e-1,1e3)
# plt.legend(fontsize=5)


st.show(hspace=0)
