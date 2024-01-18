import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
import pycwt as wavelet
import scipy.interpolate
import scipy.optimize
import scipy.linalg
import netCDF4 as nc
import nifty as ny
colours = st.configure()


def timeStep(A, Q, Kh, ssea, dx, dt, s_prev):
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
    # s[:, n+1] = snew
    return snew

def funF(s, sold, Kh, Ls, Q, A, ssea, x, dx, dt, s_psu):
    M = np.zeros((jmax, jmax))
    P = np.zeros((jmax, jmax))
    c = np.zeros(jmax)
    rhs = np.zeros(jmax)

    M[range(1,jmax-1), range(2,jmax)] = -Q/dx
    M[range(1,jmax-1), range(1,jmax-1)] = Q/dx + A[1:-1]/dt
    M[0,0] = 1
    M[-1,-1] = 1

    P[range(1,jmax-1), range(2,jmax)] = -0.5*(A[2:]+A[1:-1])/dx**2
    P[range(1,jmax-1), range(1,jmax-1)] = 0.5*(A[2:]+A[1:-1])/dx**2 + 0.5*(A[:-2]+A[1:-1])/dx**2
    P[range(1,jmax-1), range(0,jmax-2)] = -0.5*(A[:-2]+A[1:-1])/dx**2

    rhs[1:-1] = A[1:-1]*sold[1:-1]/dt
    rhs[0] = ssea

    xind1 = np.max(np.where(x<=Ls))
    a = (Ls-x[xind1+1])/(x[xind1]-x[xind1+1])
    c[xind1] = a
    c[xind1+1] = (1-a)

    # Jacobian
    J = np.zeros((jmax+1, jmax+1))
    J[slice(0,jmax),slice(0,jmax)] = M+Kh*P
    J[-1,:-1] = c.T
    J[:-1,-1] = np.dot(P, s)

    # Function
    F = np.zeros(jmax+1)
    F[:jmax] = np.dot(M, s)+Kh*np.dot(P, s) - rhs
    F[-1] = np.dot(c, s)-s_psu
    return F, J

def integ(u, x, L):
    dx = (x[1]-x[0])
    xind_max = np.max(np.where(x<L))
    if xind_max==1:
        return u[0]*L
    else:
        int = u[0]*0.5*dx
        int += np.sum(u[1:xind_max]*dx)
        int += u[xind_max]*(0.5*dx+L-x[xind_max])
        return int

pt_Q = r"Examples/Dijkstra_2024_JGR/ModaomenCase/ElevationModelValidationData.nc"
fh = nc.Dataset(pt_Q, mode='r')
Q = np.asarray(0.3185*(fh.variables['West_river_discharge'][:] + fh.variables['North_river_discharge'][:]))
t = np.asarray(fh.variables['River_date'][:])

pt_Ls = r"Examples/Dijkstra_2024_JGR/ModaomenCase/SaltIntrusionLength0p5PSU.nc"
fh = nc.Dataset(pt_Ls, mode='r')
dat = fh.variables['bottom_0p5PSU_Ls_meters'][:]
ts = dat[0,:]
Ls = dat[1,:]
Ls = ny.savitzky_golay(Ls, 25,1)
Ls = ny.savitzky_golay(Ls, 13,1)
Ls_obs = scipy.interpolate.interp1d(ts, Ls, bounds_error=False, fill_value=-1)
Ls = Ls_obs(t)

t = t[720:9192]*3600*24
Ls = Ls[720:9192]
Ls = Ls+6000            # Move mouth 6 km downstream
Q = Q[720:9192]
Q = ny.savitzky_golay(Q, 25, 1) # smooth to prevent jumps
Q = ny.savitzky_golay(Q, 13, 1)
# Q = ny.savitzky_golay(Q, 25, 1)
# H = H[371:6195]
# B = B[371:6195]
# Av = Av[371:6195]


# Ls[1:] = Ls[0]
# Q[1:] = Q[0]

# plt.figure(1, figsize=(2,2))
# plt.subplot(2,1,1)
# plt.plot(t, Q)
# plt.subplot(2,1,2)
# plt.plot(t, Ls)
#
# st.show()

########################################################################################################################
## Settings & Init
########################################################################################################################
L = 1e5
jmax = 400
loc = 1e4       # location where we analyse the signal
A = 13300*np.ones(jmax)
tmax = len(t)

x = np.linspace(0, L, jmax)
s = np.zeros((jmax, tmax))
x_ind = np.argmin(np.abs(x-loc))

Kh = np.zeros((jmax, tmax))
ssea = 30*np.ones(tmax)         # TODO need data for this
dx = x[1]-x[0]
dt = t[1]-t[0]

#Initial Kh and s
Kh[:, 0] = -Q[0]*Ls[0]/(A*np.log(0.5/ssea[0]))
s[:,0] = ssea[0]*np.exp(-Q[0]*x/(A*Kh[:,0]))

Lsmodel = np.zeros(len(t))
########################################################################################################################
## Time integration
########################################################################################################################
for n in range(len(t[:-1])):
    Kh[:, n+1] = Kh[:,n]
    s[:,n+1] = s[:, n]
    dif = [np.inf]
    s_kh = np.zeros(jmax+1)
    s_kh[:jmax] = s[:, n]
    s_kh[-1] = Kh[0,n]

    counter = 0
    psu_Ls = 0.5
    maxchange = 0.05
    while dif[-1]>1e-11:
        F, J = funF(s[:,n+1], s[:, n], Kh[0,n+1], Ls[n+1], Q[n+1], A, ssea[n+1], x, dx, dt, psu_Ls)
        ds_kh = np.linalg.solve(J, -F)

        frac = np.abs(ds_kh[-1])/np.abs(Kh[0,n+1])
        if frac<maxchange:
            s[:,n+1] += ds_kh[:jmax]
            Kh[:,n+1] += ds_kh[-1]
        else:
            # print('in catch')
            s[:,n+1] += maxchange/frac*ds_kh[:jmax]
            Kh[:,n+1] += maxchange/frac*ds_kh[-1]

        # dif.append(np.max(np.abs(ds_kh/(s_kh+1e-3))))
        dif.append(np.max(np.abs(F)))
        counter+=1

        if Kh[0,n+1]>2000 or counter>700:
            print('WARNING: diverging/max number of iterations')
            Kh[:,n+1] = Kh[:,n]
            break

    print(n, counter, Kh[0,n+1], dif[-1])
    Lsmodel[n+1] = x[np.min(np.where(s[:,n+1]<0.5))]

    # check
    snew = timeStep(A, Q, Kh, ssea, dx, dt, s[:,n])
    print(np.max(np.abs(s[:,n+1]-snew)))

np.savetxt(r'Examples/Dijkstra_2024_JGR/ModaomenCase/Kh_fitted.txt', Kh[0,:])
plt.figure(1, figsize=(2,2))
plt.subplot(2,1,1)
plt.plot(Ls)
plt.plot(Lsmodel)
plt.subplot(2,1,2)
plt.plot(Kh[0,:])
plt.plot(Q[:])
st.show()



########################################################################################################################
## Adjustment time
########################################################################################################################
## Equilibrium adjustment time
Tadj_eq = (A**2).reshape((jmax,1))*Kh/(Q**2).reshape((1,tmax))
Tadj_eq = Tadj_eq[0,:]        # TODO temporary definition

## Actual adjustment time
isoline = 1
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
# alpha1 = 0.0
# alpha2 = 0.0           # Lag-1 autocorrelation for white noise
alpha1 = 0.7        # YMD manual (inspired on Torrence & Compo)
alpha2 = .95

## CWT
std1 = s1.std()
std2 = s2.std()
W1, scales1, freqs1, coi1, _, _ = wavelet.cwt(s1/std1, dt, dj, s0, J, mother)
coi1 = coi1*.3
signif1, fft_theor1 = wavelet.significance(1.0, dt, scales1, 0, alpha1, significance_level=slevel, wavelet=mother)
W2, scales2, freqs2, coi2, _, _ = wavelet.cwt(s2/std2, dt, dj, s0, J, mother)
coi2 = coi2*.3
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

power1_sc[np.where(sig95_1<1)] = np.nan
power2_sc[np.where(sig95_2<1)] = np.nan

## XWT
# W12, cross_coi, freq, signif = wavelet.xwt(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True)
# cross_power = np.abs(W12)**2
# cross_sig = np.ones([1, n]) * signif[:, None]
# cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
# cross_period = 1/freq

# Calculate the wavelet coherence (WTC). The WTC finds regions in time frequency space where the two time seris co-vary, but do not necessarily have high power.
WCT, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1/12, s0=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True, cache=True)

cor_period = 1 / freq
try:
    cor_sig = np.ones([1, n]) * sig[:, None]
    cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
except:
    cor_sig = np.abs(WCT)
    print('WARNING: WCT significance could not be computed')

# Calculates the phase between both time series. The phase arrows in the
# cross wavelet power spectrum rotate clockwise with 'north' origin.
# The relative phase relationship convention is the same as adopted
# by Torrence and Webster (1999), where in phase signals point
# upwards (N), anti-phase signals point downwards (S). If X leads Y,
# arrows point to the right (E) and if X lags Y, arrow points to the
# left (W).
angle = 0.5 * np.pi - aWCT
u, v = np.cos(angle), np.sin(angle)

########################################################################################################################
## Select significant points to measure delay
########################################################################################################################
rel_power1 = power1/(np.ones([1, n]) * signif1[:, None])
points1 = np.zeros(power1.shape)
points1[1:-1,:] = 0.25*(np.sign(rel_power1[1:-1]-rel_power1[:-2])+1) + 0.25*(np.sign(rel_power1[1:-1]-rel_power1[2:])+1)


aWCT[np.where(points1<1)] = np.nan
coi1_matrix = (np.ones([1, len(freqs1)]) * coi1[:, None]).T
period_matrix = (np.ones([1, n]) * period1[:, None])
aWCT[np.where(period_matrix>coi1_matrix)] = np.nan

########################################################################################################################
## Plots
########################################################################################################################
plt.figure(1,figsize=(2,2))
plt.subplot(2,1,1)
plt.plot(t/(24*3600), Q)
plt.plot(t[200:400]/(24*3600), Q[0]*np.ones(200), 'r')
plt.plot(t[700:900]/(24*3600), Q[0]*np.ones(200), 'r')

plt.subplot(2,1,2)
plt.plot(t, s[10,:])

plt.figure(2,figsize=(1,2))
plt.plot(x, s[:,0])
plt.plot(x, s[:,100])
plt.plot(x, s[:,400])

########################################################################################################################
## FIG 3 - CWT
########################################################################################################################
plt.figure(3,figsize=(2,2))
plt.subplot(2,1,1)
plt.pcolormesh(t/(3600*24), period1/(3600*24), power1_sc, shading='gouraud', cmap='jet', vmax=1)
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), sig95_1, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

plt.subplot(2,1,2)
# plt.pcolormesh(t/(3600*24), period2/(3600*24), power2_sc, shading='gouraud', cmap='jet', vmax=1)
plt.pcolormesh(t/(3600*24), period2/(3600*24), power2_sc, shading='gouraud', cmap='jet', vmax=1)
plt.colorbar()
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
## FIG 50 - XWT
########################################################################################################################
plt.figure(50,figsize=(2,2))
plt.subplot(2,1,1)
plt.pcolormesh(t/(3600*24), period1/(3600*24), np.log10(cor_sig), shading='gouraud', cmap='jet',vmin=-2, vmax=1)
plt.colorbar()
plt.contour(t/(3600*24), period1/(3600*24), cor_sig, [1], colors='k', linewidths=2)
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
         np.concatenate([coi1/(3600*24), [1e-9/(3600*24)], period1[-1:]/(3600*24), period1[-1:]/(3600*24), [1e-9/(3600*24)]]),
         'k', alpha=0.6)
plt.title('{} Wavelet Power Spectrum ({})'.format(data1['nick'], mother.name))
plt.yscale('log')
plt.ylim(1,None)

# plt.subplot(2,1,2)
# plt.pcolormesh(t/(3600*24), period2/(3600*24), sig95_2, shading='gouraud', cmap='jet', vmax=2)
# plt.colorbar()
# plt.contour(t/(3600*24), period1/(3600*24), sig95_2, [1], colors='k', linewidths=2)
# plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]),
#          np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]),
#          'k', alpha=0.6)
# plt.title('{} Wavelet Power Spectrum ({})'.format(data2['nick'], mother.name))
# plt.yscale('log')
# plt.ylim(1,None)

########################################################################################################################
## FIG 4
########################################################################################################################
plt.figure(4, figsize=(2,2))
plt.subplot(2,1,1)
plt.plot(period1/(24*3600),np.log10(power1[:,200]))
plt.plot(period1/(24*3600),np.log10(signif1))
plt.plot([2,2], [np.min(np.log10(power1[:,200])), np.max(np.log10(power1[:,200]))], 'k--')
plt.plot([5,5], [np.min(np.log10(power1[:,200])), np.max(np.log10(power1[:,200]))], 'k--')
plt.plot([10,10], [np.min(np.log10(power1[:,200])), np.max(np.log10(power1[:,200]))], 'k--')
plt.plot([20,20], [np.min(np.log10(power1[:,200])), np.max(np.log10(power1[:,200]))], 'k--')
plt.plot([30,30], [np.min(np.log10(power1[:,200])), np.max(np.log10(power1[:,200]))], 'k--')

plt.subplot(2,1,2)
plt.plot(period1/(24*3600),np.log10(power2[:,200]))
plt.plot(period1/(24*3600),np.log10(signif2))
plt.plot([2,2], [np.min(np.log10(power2[:,200])), np.max(np.log10(power2[:,200]))], 'k--')
plt.plot([5,5], [np.min(np.log10(power2[:,200])), np.max(np.log10(power2[:,200]))], 'k--')
plt.plot([10,10], [np.min(np.log10(power2[:,200])), np.max(np.log10(power2[:,200]))], 'k--')
plt.plot([20,20], [np.min(np.log10(power2[:,200])), np.max(np.log10(power2[:,200]))], 'k--')
plt.plot([30,30], [np.min(np.log10(power2[:,200])), np.max(np.log10(power2[:,200]))], 'k--')

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
plt.figure(5, figsize=(2,3))
plt.subplot(2,2,1)
# plt.pcolormesh(t/(3600*24), cor_period/(3600*24), ((aWCT/(2*np.pi)+1/2)*360), cmap='jet', vmin=0,vmax=90)
plt.pcolormesh(t/(3600*24), cor_period/(3600*24), np.log10((aWCT/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))/(3600*24)), cmap='jet', vmin=-0.5,vmax=.5)
# plt.pcolormesh(t/(3600*24), period1/(3600*24), sig95_1, cmap='jet')
plt.colorbar()
#
plt.fill(np.concatenate([t/(3600*24), (t[-1:]+dt)/(3600*24), (t[-1:]+dt)/(3600*24), (t[:1]-dt)/(3600*24), (t[:1]-dt)/(3600*24)]), np.concatenate([coi2/(3600*24), [1e-9/(3600*24)], period2[-1:]/(3600*24), period2[-1:]/(3600*24), [1e-9/(3600*24)]]), 'k', alpha=0.6)
plt.yscale('log')
plt.ylim(1,None)
plt.plot(t[200:400]/(24*3600), 1.1*np.ones(200), 'r', linewidth=4)
plt.plot(t[700:900]/(24*3600), 1.1*np.ones(200), 'r', linewidth=4)

plt.subplot(2,2,2)
# plt.plot(t/(3600*24), Tadj_eq/(3600*24))
plt.plot(t/(3600*24), Tadj/(3600*24))
# plt.twinx()
# plt.plot(t/(3600*24), Ls)
plt.xlabel('t (days)')
plt.ylabel('$T_{adj}$ (days)')
plt.ylim(0, None)

plt.subplot(2,2,3)
Tdelay = (aWCT/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))
tforcing_matrix = period1[:,None]*np.ones((1,n))
plt.plot(((tforcing_matrix/Tadj[:,None].T)[:,::10]).flatten(),((Tdelay/Tadj[:,None].T)[:,::10]).flatten(), 'k.')
# plt.plot((tforcing_matrix/Tadj[:,None].T).flatten(),(Tdelay).flatten(), 'k.')

plt.plot(np.asarray([2*24*3600]*len(Tadj))/Tadj[:],Tdelay[19,:]/Tadj[:], '.', label='2')
plt.plot(np.asarray([5*24*3600]*len(Tadj))/Tadj[:],Tdelay[35,:]/Tadj[:], '.', label='5')
plt.plot(np.asarray([10*24*3600]*len(Tadj))/Tadj[:],Tdelay[47,:]/Tadj[:], '.', label='10')
plt.plot(np.asarray([20*24*3600]*len(Tadj))/Tadj[:],Tdelay[59,:]/Tadj[:], '.', label='20')
plt.plot(np.asarray([30*24*3600]*len(Tadj))/Tadj[:],Tdelay[66,:]/Tadj[:], '.', label='30')
plt.plot(np.asarray([40*24*3600]*len(Tadj))/Tadj[:],Tdelay[71,:]/Tadj[:], '.', label='40')


# plt.plot(np.asarray([2*24*3600]*200)/Tadj[200:400],Tdelay[19,200:400]/Tadj[200:400], '.', label='2a')
# plt.plot(np.asarray([5*24*3600]*200)/Tadj[200:400],Tdelay[35,200:400]/Tadj[200:400], '.', label='5a')
# plt.plot(np.asarray([10*24*3600]*200)/Tadj[200:400],Tdelay[47,200:400]/Tadj[200:400], '.', label='10a')
# plt.plot(np.asarray([20*24*3600]*200)/Tadj[200:400],Tdelay[59,200:400]/Tadj[200:400], '.', label='20a')
# plt.plot(np.asarray([30*24*3600]*200)/Tadj[200:400],Tdelay[66,200:400]/Tadj[200:400], '.', label='30a')
# plt.plot(np.asarray([40*24*3600]*200)/Tadj[200:400],Tdelay[71,200:400]/Tadj[200:400], '.', label='40a')

# plt.plot(np.asarray([2*24*3600]*200)/Tadj[700:900],Tdelay[19,700:900]/Tadj[700:900], '.', label='2b')
# plt.plot(np.asarray([5*24*3600]*200)/Tadj[700:900],Tdelay[35,700:900]/Tadj[700:900], '.', label='5b')
# plt.plot(np.asarray([10*24*3600]*200)/Tadj[700:900],Tdelay[47,700:900]/Tadj[700:900], '.', label='10b')
# plt.plot(np.asarray([20*24*3600]*200)/Tadj[700:900],Tdelay[59,700:900]/Tadj[700:900], '.', label='20b')
# plt.plot(np.asarray([30*24*3600]*200)/Tadj[700:900],Tdelay[66,700:900]/Tadj[700:900], '.', label='30b')
# plt.plot(np.asarray([40*24*3600]*200)/Tadj[700:900],Tdelay[71,700:900]/Tadj[700:900], '.', label='40b')
plt.xlabel('$T_{forcing}/T_{adj}$')
plt.ylabel('$T_{delay}/T_{adj}$')
plt.xscale('log')
plt.ylim(0, None)
plt.legend(fontsize=5)

plt.subplot(2,2,4)
plt.plot((tforcing_matrix/Tadj_eq[:,None].T).flatten(),(Tdelay/Tadj_eq[:,None].T).flatten(), 'k.')
plt.plot(np.asarray([2*24*3600]*200)/Tadj_eq[200:400],Tdelay[19,200:400]/Tadj_eq[200:400], '.', label='2a')
plt.plot(np.asarray([5*24*3600]*200)/Tadj_eq[200:400],Tdelay[35,200:400]/Tadj_eq[200:400], '.', label='5a')
plt.plot(np.asarray([10*24*3600]*200)/Tadj_eq[200:400],Tdelay[47,200:400]/Tadj_eq[200:400], '.', label='10a')
plt.plot(np.asarray([20*24*3600]*200)/Tadj_eq[200:400],Tdelay[59,200:400]/Tadj_eq[200:400], '.', label='20a')
plt.plot(np.asarray([30*24*3600]*200)/Tadj_eq[200:400],Tdelay[66,200:400]/Tadj_eq[200:400], '.', label='30a')
plt.plot(np.asarray([40*24*3600]*200)/Tadj_eq[200:400],Tdelay[71,200:400]/Tadj_eq[200:400], '.', label='40a')

plt.plot(np.asarray([2*24*3600]*200)/Tadj_eq[700:900],Tdelay[19,700:900]/Tadj_eq[700:900], '.', label='2b')
plt.plot(np.asarray([5*24*3600]*200)/Tadj_eq[700:900],Tdelay[35,700:900]/Tadj_eq[700:900], '.', label='5b')
plt.plot(np.asarray([10*24*3600]*200)/Tadj_eq[700:900],Tdelay[47,700:900]/Tadj_eq[700:900], '.', label='10b')
plt.plot(np.asarray([20*24*3600]*200)/Tadj_eq[700:900],Tdelay[59,700:900]/Tadj_eq[700:900], '.', label='20b')
plt.plot(np.asarray([30*24*3600]*200)/Tadj_eq[700:900],Tdelay[66,700:900]/Tadj_eq[700:900], '.', label='30b')
plt.xlabel('$T_{forcing}/T_{adj,eq}$')
plt.ylabel('$T_{delay}/T_{adj,eq}$')
plt.xscale('log')
plt.ylim(0, None)
# plt.plot(np.asarray([2*24*3600]*200)/(24*3600),Tdelay[19,200:400]/(24*3600), '.', label='2a')
# plt.plot(np.asarray([5*24*3600]*200)/(24*3600),Tdelay[35,200:400]/(24*3600), '.', label='5a')
# plt.plot(np.asarray([10*24*3600]*200)/(24*3600),Tdelay[47,200:400]/(24*3600), '.', label='10a')
# plt.plot(np.asarray([20*24*3600]*200)/(24*3600),Tdelay[59,200:400]/(24*3600), '.', label='20a')
# plt.plot(np.asarray([30*24*3600]*200)/(24*3600),Tdelay[66,200:400]/(24*3600), '.', label='30a')
# plt.plot(np.asarray([40*24*3600]*200)/(24*3600),Tdelay[71,200:400]/(24*3600), '.', label='40a')
# 
# plt.plot(np.asarray([2*24*3600]*200)/(24*3600),Tdelay[19,700:900]/(24*3600), '.', label='2b')
# plt.plot(np.asarray([5*24*3600]*200)/(24*3600),Tdelay[35,700:900]/(24*3600), '.', label='5b')
# plt.plot(np.asarray([10*24*3600]*200)/(24*3600),Tdelay[47,700:900]/(24*3600), '.', label='10b')
# plt.plot(np.asarray([20*24*3600]*200)/(24*3600),Tdelay[59,700:900]/(24*3600), '.', label='20b')
# plt.plot(np.asarray([30*24*3600]*200)/(24*3600),Tdelay[66,700:900]/(24*3600), '.', label='30b')
# plt.plot(np.asarray([40*24*3600]*200)/(24*3600),Tdelay[71,700:900]/(24*3600), '.', label='40b')


st.show()



