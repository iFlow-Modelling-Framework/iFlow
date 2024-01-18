import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st

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
ssea_vec = [25,20,10,25,20,10,25,20,10]
amp_vec = [5,10,20,5,10,20,5,10,20]
Pvec = [4,8,24,4,8,24,4,8,24]

########################################################################################################################
## Theory lines
########################################################################################################################
Q0 = Q0_vec[-1]
ssea0=ssea_vec[0]
Tadj_eq = ((Q0/(A0*Kh0))**2*Kh0)**(-1)
Tvec_theory = 10**np.linspace(-1,3,100)*Tadj_eq
om_fun = 2*np.pi/Tvec_theory
x = k*A0*Kh0/Q0

s0 = ssea0*np.exp(-Q0/(A0*Kh0)*x)
r2 = -Q0/(2*A0*Kh0)-np.sqrt((Q0**2-4*1j*om_fun*A0**2*Kh0)/(4*Kh0**2*A0**2))
s1 = np.exp(r2*x)
phi = np.angle(s1)*180/np.pi

set = 0
for i in range(len(Tvec_theory)):
    if phi[i]<0:
        set = 1
    if set==1 and phi[i]>=0:
        phi[:i]+=360
        set = 0
    # phi[:np.max(np.where(phi<0))+1] = phi[:np.max(np.where(phi<0))+1]+360
Tdelay_theory = (phi/360*Tvec_theory)

plt.figure(5, figsize=(1,2))
plt.subplot(1,2,2)
# plt.fill_between(10**np.linspace(-1,2,100), 10**np.linspace(-1,2,100), 10.1*np.ones(100), color='grey')
plt.plot(10**np.linspace(-1,2,100),10**np.linspace(-1,2,100), 'k-')
plt.plot(10**np.linspace(-1,2,100),.25*10**np.linspace(-1,2,100), '--', color=colours[5])
plt.text(6, 1.9, '90 deg', fontsize=6, color=colours[5], va='bottom', rotation=70)

plt.plot(Tvec_theory/Tadj_eq,  Tdelay_theory/Tadj_eq, color=colours[0])
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
    ssea0 = ssea_vec[q]
    amp = amp_vec[q]
    ssea = ssea0*np.ones((tmax))
    ssea[np.where(abs(t-0.5*T)<=.25*T/P)] += amp*(np.cos((t-.5*T)/T*2*np.pi*P))[np.where(abs(t-0.5*T)<=.25*T/P)]

    x_k = k*A0*Kh0/Q0

    s2, tforcing_matrix, Tadj_matrix, Tdelay = dispersion_peak(x, x_k, t, Q, Kh, ssea, A0, var='ssea')


    ########################################################################################################################
    ## Plots
    ########################################################################################################################
    plt.figure(5, figsize=(1,2))
    if q<3:
        plt.subplot(3,2,1)
        plt.yticks([0,10,20,30])
        plt.ylim(0,35)
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')

    elif q<6:
        plt.subplot(3,2,3)
        plt.yticks([0,10,20,30])
        plt.ylim(0,35)
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')
    else:
        plt.subplot(3,2,5)
        plt.yticks([0,10,20,30])
        plt.ylim(0,35)
        plt.text(0, 500, 'Q='+str(int(Q0))+' $m^3/s$', fontsize=6, ha='left')

    p=plt.plot(t/(3600*24), ssea, color=colours[q])
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
plt.ylabel('ssea ($psu$)')

plt.subplot(1,2,2)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.xscale('log')
plt.ylim(0, 6)
plt.xlim(1e-1,1e3)
# plt.legend(fontsize=5)


st.show(hspace=0)
