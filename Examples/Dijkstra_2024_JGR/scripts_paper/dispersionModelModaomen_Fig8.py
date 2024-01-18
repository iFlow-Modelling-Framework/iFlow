import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
import scipy.interpolate
import scipy.optimize
import scipy.linalg
import nifty as ny
import netCDF4 as nc
from adjustment.paper.util.dispersionModel_phasedif_unordered_function import delay_harmonic
from adjustment.paper.util.dispersionModelSpaceTime_peak_function import dispersion_peak
colours = st.configure()

########################################################################################################################
## Settings
########################################################################################################################
L = 1e5
jmax = 400
A0 = 13300
Ac = 1e12#1e5
loc = .5e4#1e4       # location where we analyse the signal
isoline = 0.5
variable ='Kh'

pt_Q = r"Examples/Dijkstra_2024_JGR/ModaomenCase/RiverDischarge.nc"
fh = nc.Dataset(pt_Q, mode='r')
Q = np.asarray(0.3185*(fh.variables['West_river_discharge'][:] + fh.variables['North_river_discharge'][:]))
# Q = 2000*np.ones(len(Q))
t = np.asarray(fh.variables['River_date'][:])

t = t[720:9192]*3600*24
Q = Q[720:9192]
Q = ny.savitzky_golay(Q, 25, 1) # smooth to prevent jumps
tmax = len(t)
T = t[-1]-t[0]

Kh = np.loadtxt(r'Examples/Dijkstra_2024_JGR/ModaomenCase/Kh_fitted.txt')
# Kh = 1000*np.ones((jmax,tmax))
# Kh += 10*np.sin(t*np.pi*2/(10*24*3600)) + 10*np.sin(t*np.pi*2/(2*24*3600))+ 10*np.sin(t*np.pi*2/(4*24*3600))+ 10*np.sin(t*np.pi*2/(20*24*3600))
# Q = 1500*np.ones((tmax))
# Q += 400*(np.tanh((t-np.mean(t))/(3*24*3600))+1) - 400*(np.tanh((t-np.mean(t)-20*24*3600)/(3*24*3600))+1)

########################################################################################################################
## Init
############ ############################################################################################################
x = np.linspace(0, L, jmax)
t = np.linspace(0, T, tmax)
s = np.zeros((jmax, tmax))
x_ind = np.argmin(np.abs(x-loc))
A = A0*np.exp(-x/Ac)
ssea = 30*np.ones(tmax)
dx = x[1]-x[0]
dt = t[1]-t[0]

## S at t=0
s_prev = ssea[0]*np.exp(-Q[0]/(Kh[0]*A)*x)
s[:, 0] = s_prev

########################################################################################################################
## Time integration
########################################################################################################################
for n in range(len(t[:-1])):
    M = np.zeros((3, jmax))
    rhs = np.zeros(jmax)
    M[0,2:] = -0.5*(A[2:]*Kh[n+1]+A[1:-1]*Kh[n+1])/dx**2 - Q[n+1]/dx
    M[1,1:-1] = 0.5*(A[2:]*Kh[n+1]+A[1:-1]*Kh[n+1])/dx**2 + 0.5*(A[:-2]*Kh[n+1]+A[1:-1]*Kh[n+1])/dx**2 + Q[n+1]/dx + A[1:-1]/dt
    M[2,:-2] = -0.5*(A[:-2]*Kh[n+1]+A[1:-1]*Kh[n+1])/dx**2
    rhs[1:-1] = A[1:-1]*s_prev[1:-1]/dt

    M[1,0] = 1
    rhs[0] = ssea[n+1]
    M[1,-1] = 1
    rhs[-1] = 0

    snew = scipy.linalg.solve_banded((1,1),M,rhs)
    s[:, n+1] = snew
    s_prev = snew

########################################################################################################################
## Adjustment time
########################################################################################################################
## Actual adjustment time
Ls_exp = np.zeros(len(t))
Ls = np.zeros(len(t))
k = np.zeros(len(t))
for i in range(len(t)):
    s_interp = scipy.interpolate.interp1d(x, s[:,i]-isoline)
    Ls[i] = scipy.optimize.fsolve(s_interp, 0)
    Ls_exp[i] = Ls[i]/(-np.log(isoline/ssea[i]))
    k[i] = -np.log(s_interp(loc)/ssea[i])
Tadj = A[0]/Q*Ls_exp
# Tadj[:]=Tadj[0]

########################################################################################################################
## Wavelets
########################################################################################################################
s2, tforcing_matrix, Tadj_matrix, TdelayQ, cor_period, aWCT_sgnQ, coi2, period2 = dispersion_peak(x, loc, t, Q, Kh, ssea, A0, var='Q', more_output=True)

_, _, _, TdelayKh, _, aWCT_sgnKh, _, _ = dispersion_peak(x, loc, t, Q, Kh, ssea, A0, var='Kh', more_output=True)

########################################################################################################################
## Theory lines
########################################################################################################################
print('Theoretical lines')
Qvec = np.asarray([0.01, .1, .5, .9])*Q[0]
ktheory = 3.4
Tadj_theory = A[0]**2*Kh[0]/(Q[0]**2)
Tvec_theory, Tdelay_theory, _, _ = delay_harmonic(Kh[0], A[0], ssea[0], Q[0], Qvec, ktheory, 20)

########################################################################################################################
## Max in Q and Kh
########################################################################################################################
maxind_Q = []
maxind_Kh = []
for i in range(1,len(t)-1):
    if Q[i]>Q[i-1] and Q[i]>Q[i+1]:
        maxind_Q.append(i)
    if Kh[i]>Kh[i-1] and Kh[i]>Kh[i+1]:
        maxind_Kh.append(i)

########################################################################################################################
## Plots
########################################################################################################################
print('Plotting')

########################################################################################################################
## FIG 1
########################################################################################################################
plt.figure(1, figsize=(2.8,2))
plt.subplot(3,2,1)
plt.plot(t/(24*3600), Q)
ax = plt.gca()
ax.ticklabel_format(axis='y',useOffset=False, style='plain')
plt.xlabel('t (d)')#, fontsize=6)
plt.ylabel('Q $(m^3/s)$')#, fontsize=6)
plt.ylim(0,None)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

plt.twinx()
plt.plot(t/(24*3600), s[x_ind,:], color=colours[1])
ax1 = plt.gca()
ax1.set_ylabel('s at x=5 km (psu)', color=colours[1])#, fontsize=6)
ax1.tick_params(axis='y', labelcolor=colours[1])
plt.ylim(0, None)

plt.subplot(3,2,2)
plt.plot(t/(24*3600), Kh)
plt.xlabel('t (d)')#, fontsize=6)
plt.ylabel('Kh $(m^2/s)$')#, fontsize=6)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
ax = plt.gca()
ax.ticklabel_format(axis='y',useOffset=False, style='plain')
plt.ylim(0,None)

## Wavelet space ##
plt.subplot(3,2,3)
plt.pcolormesh(t/(3600*24), cor_period/(3600*24),Tadj_matrix/(3600*24), cmap='jet', shading='gouraud')
fig = plt.gcf()
ax = plt.gca()
cbar = plt.colorbar(orientation = 'horizontal', fraction=.1, aspect=30, pad=.3)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(label='$T_{adj,0}$ (d)',size=6)

plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgnQ, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]), np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]), 'k', alpha=0.6)

plt.yscale('log')
plt.ylim(1,None)
plt.xlabel('t (d)')#, fontsize=6)
plt.ylabel('wavelet scale (d)')#, fontsize=6)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

plt.subplot(3,2,4)
plt.pcolormesh(t/(3600*24), cor_period/(3600*24),Tadj_matrix/(3600*24), cmap='jet', shading='gouraud')
fig = plt.gcf()
ax = plt.gca()
cbar = plt.colorbar(orientation = 'horizontal', fraction=.1, aspect=30, pad=.3)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(label='$T_{adj,0}$ (d)',size=6)

plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgnKh, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]), np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]), 'k', alpha=0.6)

plt.yscale('log')
plt.ylim(1,None)
plt.xlabel('t (d)')#, fontsize=6)
plt.ylabel('wavelet scale (d)')#, fontsize=6)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

## DELAY ##
plt.subplot(3,2,5)
# for qqq, Q1 in enumerate(Qvec):
plt.plot(Tvec_theory/Tadj_theory,  Tdelay_theory[:,0]/Tadj_theory, label='$|\hat{Q}_1|/Q_0=$'+str(Qvec[0]/Q[0]))

Tplt = 10**np.linspace(-1,3,100)
plt.fill_between(Tplt, Tplt, 21*np.ones(100), color='grey')
# plt.plot(((tforcing_matrix/Tadj[:,None].T)).flatten(),((Tdelay/Tadj[:,None].T)).flatten(), 'r.')
plt.plot(((tforcing_matrix/Tadj_matrix)[:,:4200]).flatten(),((TdelayQ/Tadj_matrix)[:,:4200]).flatten(), '.', color=colours[0], markersize=0.5, alpha=0.2)
plt.plot(((tforcing_matrix/Tadj_matrix)[:,4200:]).flatten(),((TdelayQ/Tadj_matrix)[:,4200:]).flatten(), '.', color=colours[3], markersize=0.5, alpha=0.2)
# plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((TdelayQ/Tadj_matrix)).flatten(), '.', color=colours[1], markersize=0.5, alpha=0.2)

# plt.plot(((tforcing_matrix/Tadj_matrix)[:, maxind_Q]).flatten(),((TdelayQ/Tadj_matrix)[:, maxind_Q]).flatten(), 'o', color=colours[1], markersize=1)
# plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((Tdelay2/Tadj_matrix)).flatten(), '.', label=r'$t>\frac{1}{2}T$')
plt.plot(10**np.linspace(-1,2,100),10**np.linspace(-1,2,100), 'k-')
plt.plot(10**np.linspace(-1,2,100),.25*10**np.linspace(-1,2,100), '--', color=colours[5])
plt.text(6, 1.9, '90 deg', fontsize=6, color=colours[5], va='bottom', rotation=70)

plt.xlabel('$T_{forcing}/T_{adj,0}$')#, fontsize=6)
plt.ylabel('$T_{delay}/T_{adj,0}$')#, fontsize=6)
plt.xscale('log')
plt.ylim(0, 20)
plt.xlim(1e-1,1e3)
plt.legend(fontsize=5.5)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

plt.subplot(3,2,6)
plt.plot(Tvec_theory/Tadj_theory,  Tdelay_theory[:,0]/Tadj_theory, label='$|\hat{K}_{h,1}|/K_{h,0}=$'+str(Qvec[0]/Q[0]))

Tplt = 10**np.linspace(-1,3,100)
plt.fill_between(Tplt, Tplt, 15*np.ones(100), color='grey')
# plt.plot(((tforcing_matrix/Tadj[:,None].T)).flatten(),((Tdelay/Tadj[:,None].T)).flatten(), 'r.')
plt.plot(((tforcing_matrix/Tadj_matrix)[:,:4200]).flatten(),((TdelayKh/Tadj_matrix)[:,:4200]).flatten(), '.', color=colours[0], markersize=0.5, alpha=0.2)
plt.plot(((tforcing_matrix/Tadj_matrix)[:,4200:]).flatten(),((TdelayKh/Tadj_matrix)[:,4200:]).flatten(), '.', color=colours[3], markersize=0.5, alpha=0.2)
# plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((TdelayKh/Tadj_matrix)).flatten(), '.', color=colours[1], markersize=0.5, alpha=0.2)
# plt.plot(((tforcing_matrix/Tadj_matrix)[:, maxind_Kh]).flatten(),((TdelayKh/Tadj_matrix)[:, maxind_Kh]).flatten(), 'o', color=colours[1], markersize=1)
# plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((Tdelay2/Tadj_matrix)).flatten(), '.', label=r'$t>\frac{1}{2}T$')
plt.plot(10**np.linspace(-1,2,100),10**np.linspace(-1,2,100), 'k-')
plt.plot(10**np.linspace(-1,3,100),np.zeros(100), 'k--')
plt.plot(10**np.linspace(-1,2,100),.25*10**np.linspace(-1,2,100), '--', color=colours[5])
plt.text(6, 1.9, '90 deg', fontsize=6, color=colours[5], va='bottom', rotation=70)
plt.legend(fontsize=5.5)

plt.xlabel('$T_{forcing}/T_{adj,0}$')#, fontsize=6)
plt.ylabel('$T_{delay}/T_{adj,0}$')#, fontsize=6)
plt.xscale('log')
plt.ylim(-10, 6)
plt.xlim(1e-1,1e3)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

plt.figure(2, figsize=(1,2))
## DELAY ##
plt.subplot(1,2,1)
# for qqq, Q1 in enumerate(Qvec):
plt.plot(Tvec_theory/Tadj_theory,  Tdelay_theory[:,0]/Tvec_theory, label='$|\hat{Q}_1|/Q_0=$'+str(Qvec[0]/Q[0]))

Tplt = 10**np.linspace(-1,3,100)
plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((TdelayQ/tforcing_matrix)).flatten(), '.', label=r'$Q$')

plt.xlabel('$T_{forcing}/T_{adj,0}$', fontsize=6)
plt.ylabel('$T_{delay}/T_{adj,0}$', fontsize=6)
plt.xscale('log')
plt.ylim(0, 20)
plt.xlim(1e-1,1e3)
plt.legend(fontsize=5.5)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

plt.subplot(1,2,2)
plt.plot(Tvec_theory/Tadj_theory,  Tdelay_theory[:,0]/Tvec_theory, label='$|\hat{K}_{h,1}|/K_{h,0}=$'+str(Qvec[0]/Q[0]))

plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((TdelayKh/tforcing_matrix)).flatten(), '.')
plt.legend(fontsize=5.5)

plt.xlabel('$T_{forcing}/T_{adj,0}$', fontsize=6)
plt.ylabel('$T_{delay}/T_{adj,0}$', fontsize=6)
plt.xscale('log')
plt.ylim(-10, 6)
plt.xlim(1e-1,1e3)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

st.show()
