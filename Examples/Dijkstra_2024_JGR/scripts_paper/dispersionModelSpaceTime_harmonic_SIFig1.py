import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
import pycwt as wavelet
import scipy.interpolate
import scipy.optimize
import scipy.linalg
import nifty as ny
from adjustment.paper.util.dispersionModel_phasedif_unordered_function import delay_harmonic
colours = st.configure()

########################################################################################################################
## Settings
########################################################################################################################
L =1e5
T = 365*24*3600
jmax = 1500
tmax = 365*3
A0 = 10*1000
Ac = 1e9#1e5
k = 3.4

Q0 = 800
amp = 10
P = 12
Kh0 = 100
ssea0 = 30

# Q0 = 100
# amp = 600
# P = 6
# Kh0 = 100
# ssea0 = 30

########################################################################################################################
## Init
############ ############################################################################################################
x = np.linspace(0, L, jmax)
t = np.linspace(0, T, tmax)
s = np.zeros((jmax, tmax))
dx = x[1]-x[0]
dt = t[1]-t[0]

## Construct Q
Q = Q0*np.ones(tmax)
# Q += amp*(1+np.tanh((t-.4*T)/(T/200))) - amp*(1+np.tanh((t-0.6*T)/(T/200)))
Q += amp*(np.sin((t)/T*2*np.pi*P)) #+ amp*(np.sin((t-.5*T)/T*2*np.pi*5*P)) + amp*(np.sin((t-.5*T)/T*2*np.pi*P/5)) + amp*(np.sin((t-.5*T)/T*2*np.pi*10*P)) + amp*(np.sin((t-.5*T)/T*2*np.pi*P/10))
# Q[np.where(abs(t-0.5*T)<=.25*T/P)] += amp*(np.cos((t-.5*T)/T*2*np.pi*P))[np.where(abs(t-0.5*T)<=.25*T/P)]

# plt.figure(1, figsize=(1,2))
# plt.plot(t, Q)
# st.show()
print([i for i in ny.fft(Q,0)[:5]])
# Q = 200*np.ones(tmax) + amp*(1+np.tanh((t-.48*T)/(T/200))) - amp*(1+np.tanh((t-0.52*T)/(T/200)))
# Q = 200*np.ones(tmax) + amp*(1+np.tanh((t-.495*T)/(T/100))) - amp*(1+np.tanh((t-0.505*T)/(T/100)))
# Q = 200*np.ones(tmax) + amp*(1+np.tanh((t-.45*T)/(T/200))) - amp*(1+np.tanh((t-0.55*T)/(T/200)))

A = A0*np.exp(-x/Ac)
Kh = Kh0*np.ones((jmax, tmax))
ssea = ssea0*np.ones(tmax)

## S at t=0
s_prev = ssea0*np.exp(-Q0/(Kh0*A[0])*x)
s[:, 0] = s_prev

## index corresponding to k for background Q
x_k = k*A0*Kh0/Q0
x_ind = np.argmin(np.abs(x-x_k))
print(x_ind)
########################################################################################################################
## Time integration
########################################################################################################################
for n in range(len(t[:-1])):
    M = np.zeros((3, jmax))
    rhs = np.zeros(jmax)
    M[0,2:] = -0.5*(A[2:]*Kh[2:, n+1]+A[1:-1]*Kh[1:-1, n+1])/dx**2 - Q[n+1]/dx
    M[1,1:-1] = 0.5*(A[2:]*Kh[2:, n+1]+A[1:-1]*Kh[1:-1, n+1])/dx**2 + 0.5*(A[:-2]*Kh[:-2, n+1]+A[1:-1]*Kh[1:-1, n+1])/dx**2 + Q[n+1]/dx + A[1:-1]/dt
    M[2,:-2] = -0.5*(A[:-2]*Kh[:-2, n+1]+A[1:-1]*Kh[1:-1, n+1])/dx**2
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
## Equilibrium adjustment time
Tadj_eq = A0**2*Kh0/Q0**2
Tadj_av = A0**2*Kh0/np.mean(Q)**2

## Actual adjustment time
isoline = ssea0*np.exp(-k)
Ls = np.zeros(len(t))
for i in range(len(t)):
    s_interp = scipy.interpolate.interp1d(x, s[:,i]-isoline)
    Ls[i] = scipy.optimize.fsolve(s_interp, 0)/(-np.log(isoline/ssea[i]))
Tadj = A[0]/Q*Ls

########################################################################################################################
## Wavelets
########################################################################################################################
## Settings
data1 = dict(name='Q', nick='Q', var=Q)
data2 = dict(name='s', nick='s', var=s[x_ind,:])
mother = wavelet.Morlet(6)          # Morlet mother wavelet with m=6
s1 = data1['var']
s2 = data2['var']

slevel = 0.7                       # Significance level
dj = 1/12                           # Twelve sub-octaves per octaves
s0 = -1  # 2 * dt                   # Starting scale, here 6 months
J = -1  # 7 / dj                    # Seven powers of two with dj sub-octaves
n = tmax
# alpha1, _, _ = wavelet.ar1(s1)  # Lag-1 autocorrelation for red noise
# alpha2, _, _ = wavelet.ar1(s2)  # Lag-1 autocorrelation for red noise
alpha1 = np.corrcoef(s1[:-1],s1[1:])[0,1]
alpha2 = np.corrcoef(s2[:-1],s2[1:])[0,1]

## CWT
COILIMITER = 1#1#0.3

std1 = s1.std()
std2 = s2.std()
W1, scales1, freqs1, coi1, _, _ = wavelet.cwt(s1/std1, dt, dj, s0, J, mother)
coi1 = coi1*COILIMITER
signif1, fft_theor1 = wavelet.significance(1.0, dt, scales1, 0, alpha1, significance_level=slevel, wavelet=mother)
W2, scales2, freqs2, coi2, _, _ = wavelet.cwt(s2/std2, dt, dj, s0, J, mother)
coi2 = coi2*COILIMITER
signif2, fft_theor2 = wavelet.significance(1.0, dt, scales2, 0, alpha2, significance_level=slevel, wavelet=mother)

power1 = (np.abs(W1)) ** 2             # Normalized wavelet power spectrum
power2 = (np.abs(W2)) ** 2             # Normalized wavelet power spectrum
power1_sc = power1/(scales1.reshape((len(freqs1),1)))*scales1[0]
power2_sc = power2/(scales2.reshape((len(freqs1),1)))*scales2[0]
period1 = 1/freqs1
period2 = 1/freqs2
sig95_1 = np.ones([1, n]) * signif1[:, None]
sig95_1 = power1 / sig95_1             # Where ratio > 1, power is significant
sig95_2 = np.ones([1, n]) * signif2[:, None]
sig95_2 = power2 / sig95_2             # Where ratio > 1, power is significant

# power1_sc[np.where(sig95_1<1)] = np.nan
# power2_sc[np.where(sig95_2<1)] = np.nan

## XWT
# W12, cross_coi, freq, signif = wavelet.xwt(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True)
W12, cross_coi, freq, signif = wavelet.xwt(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.95, wavelet='morlet', normalize=True)
cross_power = np.abs(W12)**2
cross_sig = np.ones([1, n]) * signif[:, None]
cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
cross_period = 1/freq

# Calculate the wavelet coherence (WTC). The WTC finds regions in time frequency space where the two time seris co-vary, but do not necessarily have high power.
# WCT, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True, cache=True)
WCT, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.95, wavelet='morlet', normalize=True, cache=True)

cor_period = 1 / freq
try:
    cor_sig = np.ones([1, n]) * sig[:, None]
    cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
except:
    cor_sig = np.abs(WCT)
    print('WARNING: WCT significance could not be computed')

angle = 0.5 * np.pi - aWCT
u, v = np.cos(angle), np.sin(angle)

########################################################################################################################
## Select significant points to measure delay
########################################################################################################################
rel_power1 = power1/(np.ones([1, n]) * signif1[:, None])

# Select points
points1 = np.zeros(power1.shape)
points1[1:-1,:] = 0.25*(np.sign(rel_power1[1:-1]-rel_power1[:-2])+1) + 0.25*(np.sign(rel_power1[1:-1]-rel_power1[2:])+1)
# points1 = np.ones(power1.shape)
# points1[np.where(power1_sc<0.001)] = 0
points1[np.where(cross_sig<1)] = 0
points1[np.where(cor_sig<1)] = 0


# points1 = np.zeros(power1.shape)
# points1[np.where(power1_sc>0.001)] = .25
# points1[np.where(cross_sig>1)] = 2*points1[np.where(cross_sig>1)]
# points1[np.where(cor_sig>1)] = 2*points1[np.where(cor_sig>1)]
tmat = t[:,None].T*np.ones((points1.shape[0],1))
# points1[np.where(tmat/(24*3600)<181.5)] = 0
# points1[np.where(tmat/(24*3600)>183.5)] = 0


aWCT[np.where(points1<.9)] = np.nan

## Apply COI
coi1_matrix = (np.ones([1, len(freqs1)]) * coi1[:, None]).T
period_matrix = (np.ones([1, n]) * period1[:, None])
aWCT[np.where(period_matrix>coi1_matrix)] = np.nan

from copy import copy
aWCT1 = copy(aWCT)
aWCT2 = copy(aWCT)
aWCT1[:, int(len(t)/2)-10:int(len(t)/2)+10] = np.nan
# aWCT1 = np.nan
aWCT2[:, :int(len(t)/2)-10] = np.nan
aWCT2[:, int(len(t)/2)+10:] = np.nan

Tdelay = (aWCT/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))
Tdelay1 = (aWCT1/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))
Tdelay2 = (aWCT2/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))

tforcing_matrix = period1[:,None]*np.ones((1,n))

########################################################################################################################
## Compute Tadj and Tdelay
####################################################################################################
# average Tadj over window size
Tadj_matrix = np.zeros((len(freqs1), len(Tadj)))
window_arr = np.zeros(len(freqs1))
for i in range(len(freqs1)):
        M = int(np.ceil(2*np.pi*period1[i]/dt))
        y = np.linspace(-np.pi,np.pi, M)
        gaus = np.exp(-.5*y**2)/(np.sqrt(2*np.pi))

        Qmean = np.zeros(len(Tadj))
        for j in  range(0, len(Tadj)):
            Qr = Q[np.maximum(j-int(np.ceil(M/2)), 0):np.minimum(j+int(np.floor(M/2)), len(Tadj)-1)]
            if np.maximum(j-int(np.ceil(M/2)), 0)==0:
                gaus_t = gaus[-len(Qr):]
            elif np.minimum(j+int(np.floor(M/2)), len(Tadj)-1) ==len(Tadj)-1:
                gaus_t = gaus[:len(Qr)]
            else:
                gaus_t = gaus

            Qmean[j] = np.mean(gaus_t*Qr)/np.mean(gaus_t)


        # window = int(2*np.ceil(period1[i]/dt))
        # window_arr[i] = window
        # scipy.signal.morlet()
        # # Tadj_matrix[i,:] = np.asarray([np.mean(Tadj[np.maximum(j-window, 0):np.minimum(j+window, len(Tadj)-1)]) for j in range(0, len(Tadj))])
        Tadj_matrix[i,:] = A0**2*Kh0/Qmean**2
        # # Tadj_matrix[i,:] = np.asarray([A0/np.mean((Q[np.maximum(j-window, 0):np.minimum(j+window, len(Tadj)-1)]))*np.mean((Ls[np.maximum(j-window, 0):np.minimum(j+window, len(Tadj)-1)]))/3.4 for j in range(0, len(Tadj))])
        # # Tadj_matrix[i,:] = np.mean(A0**2*Kh0/(Q0)**2)

aWCT_sgn = np.ones(aWCT.shape)
aWCT_sgn[np.where(np.isnan(aWCT))] = np.nan

########################################################################################################################
## Theory lines
########################################################################################################################
Qvec = np.asarray([0.01, 0.5, 0.9])*Q0
Tvec_theory, Tdelay_theory, _, _ = delay_harmonic(Kh0, A[0], ssea0, Q0, Qvec, k, 20)

########################################################################################################################
## Plots
########################################################################################################################
plt.figure(1,figsize=(2,2))
plt.subplot(2,2,1)
plt.plot(t/(24*3600), s1)
plt.xlabel('t (d)')
plt.ylabel('Q')


plt.subplot(2,2,2)
plt.plot(t/(24*3600), s2)
plt.xlabel('t (d)')
plt.ylabel('s at x1')

plt.subplot(2,2,3)
plt.plot(t/(24*3600), Ls)
plt.xlabel('t (d)')
plt.ylabel('Ls')

plt.figure(2,figsize=(1,2))
plt.plot(x, s[:,0])
plt.plot(x, s[:,100])
plt.plot(x, s[:,400])

########################################################################################################################
## FIG 3 - CWT
########################################################################################################################
plt.figure(3,figsize=(2,2))
plt.subplot(2,1,1)
plt.pcolormesh(t/(3600*24), period1/(3600*24), power1_sc, shading='gouraud', cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-3))
plt.colorbar()
plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgn, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.contour(t/(3600*24), period1/(3600*24), sig95_1, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

plt.subplot(2,1,2)
# plt.pcolormesh(t/(3600*24), period2/(3600*24), power2_sc, shading='gouraud', cmap='jet', vmax=1)
plt.pcolormesh(t/(3600*24), period2/(3600*24), power2_sc, shading='gouraud', cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-3))
plt.colorbar()
plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgn, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.contour(t/(3600*24), period1/(3600*24), sig95_1, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data2['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

########################################################################################################################
## FIG 30 - CWT
########################################################################################################################
plt.figure(30,figsize=(2,2))
plt.subplot(2,1,1)
plt.pcolormesh(t/(3600*24), period1/(3600*24), sig95_1, shading='gouraud', cmap='jet', vmax=2)
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), sig95_1, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

plt.subplot(2,1,2)
plt.pcolormesh(t/(3600*24), period2/(3600*24), sig95_2, shading='gouraud', cmap='jet', vmax=2)
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), sig95_2, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data2['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

########################################################################################################################
## FIG 6
########################################################################################################################
plt.figure(6,figsize=(1,2))
plt.pcolormesh(t/(3600*24), period1/(3600*24), np.log10(cor_sig), shading='gouraud', cmap='jet',vmin=-2, vmax=1)
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), cor_sig, [1,1.5,2], colors='k', linewidths=[2,1,2])
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

########################################################################################################################
## FIG 7
########################################################################################################################
plt.figure(7,figsize=(1,2))
plt.pcolormesh(t/(3600*24), period1/(3600*24), np.log10(cross_power), shading='gouraud', cmap='jet')
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), cross_sig, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)


# W12, cross_coi, freq, signif = wavelet.xwt(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True)
# cross_power = np.abs(W12)**2
# cross_sig = np.ones([1, n]) * signif[:, None]
# cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
# cross_period = 1/freq

########################################################################################################################
## FIG 4
########################################################################################################################
plt.figure(4, figsize=(2,2))
plt.subplot(2,1,1)
index = 550
plt.plot(period1/(24*3600),np.log10(power1[:,550]))
plt.plot(period1/(24*3600),np.log10(signif1))
# plt.plot([2,2], [np.min(np.log10(power1[:,550])), np.max(np.log10(power1[:,550]))], 'k--')
# plt.plot([5,5], [np.min(np.log10(power1[:,550])), np.max(np.log10(power1[:,550]))], 'k--')
# plt.plot([10,10], [np.min(np.log10(power1[:,550])), np.max(np.log10(power1[:,550]))], 'k--')
# plt.plot([20,20], [np.min(np.log10(power1[:,550])), np.max(np.log10(power1[:,550]))], 'k--')
# plt.plot([30,30], [np.min(np.log10(power1[:,550])), np.max(np.log10(power1[:,550]))], 'k--')

plt.subplot(2,1,2)
plt.plot(period1/(24*3600),np.log10(power1_sc[:,550]))
# plt.plot(period1/(24*3600),np.log10(signif2))
# plt.plot([2,2], [np.min(np.log10(power1_sc[:,550])), np.max(np.log10(power1_sc[:,550]))], 'k--')
# plt.plot([5,5], [np.min(np.log10(power1_sc[:,550])), np.max(np.log10(power1_sc[:,550]))], 'k--')
# plt.plot([10,10], [np.min(np.log10(power1_sc[:,550])), np.max(np.log10(power1_sc[:,550]))], 'k--')
# plt.plot([20,20], [np.min(np.log10(power1_sc[:,550])), np.max(np.log10(power1_sc[:,550]))], 'k--')
# plt.plot([30,30], [np.min(np.log10(power1_sc[:,550])), np.max(np.log10(power1_sc[:,550]))], 'k--')

# plt.subplot(2,1,1)
# plt.plot(period1/(24*3600),np.log10(np.abs(WCT[:,200])))
# plt.plot(period1/(24*3600),np.log10(sig))
# plt.plot([2,2], [np.min(np.log10(np.abs(WCT[:,200]))), np.max(np.log10(np.abs(WCT[:,200])))], 'k--')
# plt.plot([5,5], [np.min(np.log10(np.abs(WCT[:,200]))), np.max(np.log10(np.abs(WCT[:,200])))], 'k--')
# plt.plot([10,10], [np.min(np.log10(np.abs(WCT[:,200]))), np.max(np.log10(np.abs(WCT[:,200])))], 'k--')
# plt.plot([20,20], [np.min(np.log10(np.abs(WCT[:,200]))), np.max(np.log10(np.abs(WCT[:,200])))], 'k--')
# plt.plot([30,30], [np.min(np.log10(np.abs(WCT[:,200]))), np.max(np.log10(np.abs(WCT[:,200])))], 'k--')


########################################################################################################################
## FIG 5
########################################################################################################################
plt.figure(5, figsize=(2,2))
plt.subplot(2,2,1)
plt.plot(t/(24*3600), s1)
plt.xlabel('t (d)')
plt.ylabel('Q $(m^3/s)$')
plt.ylim(0,1500)
ax = plt.gca()
ax.ticklabel_format(axis='y',useOffset=False, style='plain')
plt.yticks([0,400,900,1400])
# plt.xlim(150,215)

plt.twinx()
plt.plot(t/(24*3600), Ls/1000, color=colours[1])
ax1 = plt.gca()
ax1.set_ylabel('Ls (km)', color=colours[1])
ax1.tick_params(axis='y', labelcolor=colours[1])
plt.ylim(0, 4)

plt.subplot(2,2,2)
# plt.pcolormesh(t/(3600*24), cor_period/(3600*24), np.log10((aWCT/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))/(3600*24)), cmap='jet', vmin=-0.5,vmax=.5, shading='gouraud')
# plt.pcolormesh(t/(3600*24), period1/(3600*24), np.log10(cross_power), shading='gouraud', cmap='jet')
# plt.colorbar()
plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgn, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.contour(t/(3600*24), period1/(3600*24), cross_sig, [1], colors='w', linewidths=.7)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]), np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]), 'k', alpha=0.6)
plt.yscale('log')
plt.ylim(1,None)
plt.xlabel('t (d)')
plt.ylabel('wavelet scale (d)')


plt.subplot(2,2,4)
plt.pcolormesh(t/(3600*24), cor_period/(3600*24),Tadj_matrix/(3600*24), cmap='jet', shading='gouraud')
fig = plt.gcf()
ax = plt.gca()
# divider = make_axes_locatable(ax)
# cax = divider.new_vertical(size = '5%', pad = 0.5)
# fig.add_axes(cax)
cbar = plt.colorbar(orientation = 'horizontal', fraction=.1, aspect=30, pad=.3)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(label='$T_{adj,0}$ (d)',size=6)

plt.pcolormesh(t/(3600*24), cor_period/(3600*24), aWCT_sgn, cmap='binary', vmin=0, vmax=1, shading='gouraud')
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]), np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]), 'k', alpha=0.6)

plt.yscale('log')
plt.ylim(1,None)
plt.xlabel('t (d)')
plt.ylabel('wavelet scale (d)')



plt.subplot(2,2,3)
for qqq, Q1 in enumerate(Qvec):
    plt.plot(Tvec_theory/Tadj_eq,  Tdelay_theory[:,qqq]/Tadj_eq, label='$|\hat{Q}_1|/Q_0=$'+str(Q1/Q0))

Tplt = 10**np.linspace(-1,3,100)
plt.fill_between(Tplt, Tplt, 15*np.ones(100), color='grey')
# plt.plot(((tforcing_matrix/Tadj[:,None].T)).flatten(),((Tdelay/Tadj[:,None].T)).flatten(), 'r.')
plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((Tdelay1/Tadj_matrix)).flatten(), '.', label=r'$t\neq\frac{1}{2}T$', markersize=0.5)
plt.plot(((tforcing_matrix/Tadj_matrix)).flatten(),((Tdelay2/Tadj_matrix)).flatten(), 'o', label=r'$t=\frac{1}{2}T$', color='k')
plt.plot(10**np.linspace(-1,2,100),10**np.linspace(-1,2,100), 'k-')
plt.plot(10**np.linspace(-1,2,100),.25*10**np.linspace(-1,2,100), '--', color=colours[5])
plt.text(6, 1.9, '90 deg', fontsize=6, color=colours[5], va='bottom', rotation=70)



plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.xscale('log')
plt.ylim(0, 10)
plt.xlim(1e-1,1e3)
plt.legend(fontsize=5.5)


st.show()
