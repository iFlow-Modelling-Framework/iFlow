import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st

colours = st.configure()
Ls_contour = 1

"""
Linear dispersion model
Ast = Qs_x +AKh s_xx = 0
with s(0,t)=ssea

Solved assuming Q=Q^0+Q^1(t), or Kh = Kh^0 + Kh^1(t) or ssea = ssea^0 + ssea^1(t)
All other parameters constant.
Forcings are assumed periodic -> use FT.
"""

########################################################################################################################
# Parameters
########################################################################################################################
Kh = 100
A = 1e4
ssea = 30
Q0 = 100
krange = [0.4, 1.1, 2.7,3.4]
isohaline = [20, 10, 2, 1]

########################################################################################################################
# Perturbation behaviour for omega
########################################################################################################################
Tadj = ((Q0/(A*Kh))**2*Kh)**(-1)#*(1+0.5*k)
T = 10**np.linspace(-1,3,100)*Tadj
om_fun = 2*np.pi/T


phi = np.zeros((len(T), len(krange)))
s1_an = np.zeros((len(krange)))
phi_an = np.zeros((len(krange)))
magn = np.zeros((len(T), len(krange)))
Tdelay_ssea = np.zeros((len(T), len(krange)))
for i,k in enumerate(krange):
    x = k*A*Kh/Q0

    s0 = ssea*np.exp(-Q0/(A*Kh)*x)

    ## adaptation to Q and Kh
    gamma = -1j*ssea*Q0/(A**2*Kh*om_fun)
    r2 = -Q0/(2*A*Kh)-np.sqrt((Q0**2-4*1j*om_fun*A**2*Kh)/(4*Kh**2*A**2))
    exp_fun = np.exp(-Q0/(A*Kh)*x)-np.exp(r2*x)
    s1 = gamma*exp_fun
    phi[:,i] = np.angle(s1)*180/np.pi

    # limits for omega -> 0
    s1_an[i] = -ssea/(A*Kh)*x*np.exp(-Q0*x/(A*Kh))
    phi_an[i] = A*np.pi/Q0**2*(Q0*x+2*A*Kh)  *180/np.pi        # analytical phase*T

    magn[:,i] = s1/s0*Q0

    ## adaptation to ssea
    s1 = np.exp(r2*x)
    phi_ssea = np.angle(s1)*180/np.pi

    set = 0
    for j in range(len(T)):
        if phi_ssea[j]<0:
            set = 1
        if set==1 and phi_ssea[j]>=0:
            phi_ssea[:j]+=360
            set = 0
        # phi[:np.max(np.where(phi<0))+1] = phi[:np.max(np.where(phi<0))+1]+360
    Tdelay_ssea[:,i] = (phi_ssea/360*T)

########################################################################################################################
## Wavelets
########################################################################################################################
# Tdelay_wavelet = np.zeros(len(T))
# angle_wavelet =  np.zeros(len(T))
# for qq in range(len(T)):
#     omega = 2*np.pi/T[qq]
#     t = np.linspace(0, 12*T[qq], 288, endpoint=False)
#     dt = t[1]-t[0]
#     Qsig = Q0 + np.cos(omega*t)
#     ssig = s0 + np.real(s1[qq]*np.exp(-1j*omega*t))
#
#     ## Settings
#     data1 = dict(name='Q', nick='Q', var=Qsig)
#     data2 = dict(name='s', nick='s', var=ssig)
#     mother = wavelet.Morlet(6)          # Morlet mother wavelet with m=6
#     s1_wv = data1['var']
#     s2_wv = data2['var']
#
#     slevel = 0.95                       # Significance level
#     dj = 1/12                           # Twelve sub-octaves per octaves
#     s0_wv = -1  # 2 * dt                   # Starting scale, here 6 months
#     J = -1  # 7 / dj                    # Seven powers of two with dj sub-octaves
#     n = len(t)
#     alpha1 = 0.0
#     alpha2 = 0.0           # Lag-1 autocorrelation for white noise
#
#     ## CWT
#     std1 = s1_wv.std()
#     std2 = s2_wv.std()
#     W1, scales1, freqs1, coi1, _, _ = wavelet.cwt(s1_wv/std1, dt, dj, s0_wv, J, mother)
#     signif1, fft_theor1 = wavelet.significance(1.0, dt, scales1, 0, alpha1, significance_level=slevel, wavelet=mother)
#     W2, scales2, freqs2, coi2, _, _ = wavelet.cwt(s2_wv/std2, dt, dj, s0_wv, J, mother)
#     signif2, fft_theor2 = wavelet.significance(1.0, dt, scales2, 0, alpha2, significance_level=slevel, wavelet=mother)
#
#     power1 = (np.abs(W1)) ** 2             # Normalized wavelet power spectrum
#     power2 = (np.abs(W2)) ** 2             # Normalized wavelet power spectrum
#     power1_sc = (np.abs(W1)/scales1.reshape((len(freqs1),1)))
#     power2_sc = (np.abs(W2)/scales2.reshape((len(freqs2),1)))
#     period1 = 1/freqs1
#     period2 = 1/freqs2
#     sig95_1 = np.ones([1, n]) * signif1[:, None]
#     sig95_1 = power1 / sig95_1             # Where ratio > 1, power is significant
#     sig95_2 = np.ones([1, n]) * signif2[:, None]
#     sig95_2 = power2 / sig95_2             # Where ratio > 1, power is significant
#
#     WCT, aWCT, corr_coi, freq, sig = wavelet.wct(s1_wv, s2_wv, dt, dj=1/12, s0_wv=-1, J=-1, significance_level=0.8646, wavelet='morlet', normalize=True, cache=True)
#     cor_period = 1 / freq
#     angle = 0.5 * np.pi - aWCT
#     u, v = np.cos(angle), np.sin(angle)
#
#     ind1 = np.argmin(abs(cor_period-T[qq]))
#     Tdelay_wavelet[qq] = (aWCT[ind1, int(len(t)/2)]/(2*np.pi)+1/2)*cor_period[ind1]
#     angle_wavelet[qq] = (aWCT[ind1, int(len(t)/2)]/(2*np.pi)+1/2)
########################################################################################################################
# Plot
########################################################################################################################
print('T_adj = '+str(Tadj/(3600*24)))

plt.figure(1, figsize=(1,1))
for i,k in enumerate(krange):
    plt.plot(T/Tadj,  k*np.ones(len(T)), '--', color=colours[3])
    plt.text(20, k*1.02, r'$k=$'+str(k), fontsize=6,color=colours[3])      # asymptote T-> infty

    plt.plot(T/Tadj,  np.abs(magn[:,i]))

plt.ylabel('magnitude')
plt.xlabel('$T_{forcing}/T_{adj}$')
plt.xscale('log')
plt.ylim(0, np.max(krange)*1.1)

### FIG 2 ###
plt.figure(2, figsize=(1,2))
plt.subplot(1,2,1)
plt.fill_between(T/Tadj, T/Tadj, 3*np.ones(len(T)), color='grey')

plt.plot(T/Tadj, T/Tadj, 'k--')
plt.text(0.15, 1.5, '$T_{delay}>T$', fontsize=6)


plt.plot(T/Tadj, 0.25*T/Tadj, '--', color=colours[5])
plt.text((T/Tadj)[int(len(T)/2.3)]*0.7, 0.25*(T/Tadj)[int(len(T)/2.3)], '90 deg', fontsize=6, color=colours[5], rotation=80)

for i,k in enumerate(krange):
    plt.plot(T/Tadj,  (phi[:,i]/360*T+.5*T)/Tadj, color=colours[i])
# plt.plot(T/Tadj,  Tdelay_wavelet/Tadj, color=colours[1])


plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.ylim(0, 3)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

plt.subplot(1,2,2)
#plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
for i,k in enumerate(krange):
    plt.plot(T/Tadj,  (phi[:,i]/360*T+.5*T)/T, color=colours[i], label='x/L='+str(k) + ' ('+str(isohaline[i])+' psu)')
# plt.plot(T/Tadj,  angle_wavelet, color=colours[1])
# plt.plot(T/Tadj, (1+0.5*k)*np.ones(len(T)), '--' , color=colours[2])      # asymptote T-> infty
#
# plt.plot(T/Tadj, T/Tadj, 'k--')
# plt.text(0.15, 0.5*(1+0.5*k), '$T_{delay}>T$', fontsize=6)
# plt.fill_between(T/Tadj, T/Tadj, phi_an/360/Tadj*1.1*np.ones(len(T)), color='grey')
#
plt.plot(T/Tadj, 0.25*np.ones(len(T)), '--', color=colours[5])
plt.text(100, 0.25*0.98, '90 deg', fontsize=6, color=colours[5], va='top')

plt.legend(fontsize=6)
plt.ylabel('$T_{delay}/T_{forcing}$')
plt.ylim(0, .5)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

### FIG 4 ###
plt.figure(4, figsize=(1,2))
plt.subplot(1,2,1)
# plt.fill_between(T/Tadj, T/Tadj, 3.5*np.ones(len(T)), color='grey')

plt.plot(T/Tadj, T/Tadj, 'k--')
plt.text(0.15, 2, '$T_{delay}>T$', fontsize=6)

for i,k in enumerate(krange):
    plt.plot(T/Tadj,  Tdelay_ssea[:,i]/Tadj, color=colours[i])
# plt.plot(T/Tadj,  Tdelay_wavelet/Tadj, color=colours[1])

plt.ylabel('$T_{delay}/T_{adj,0}$')
plt.ylim(0, 3.5)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')

plt.subplot(1,2,2)
#plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
for i,k in enumerate(krange):
    plt.plot(T/Tadj,  Tdelay_ssea[:,i]/T, color=colours[i], label='x/L='+str(k) + ' ('+str(isohaline[i])+' psu)')

plt.plot(T/Tadj, 0.25*np.ones(len(T)), '--', color=colours[5])
plt.text(100, 0.25*0.98, '90 deg', fontsize=6, color=colours[5], va='top')
plt.plot(T/Tadj, np.ones(len(T)), '--', color=colours[5])
plt.text(100, 0.98, '360 deg', fontsize=6, color=colours[5], va='top')

plt.legend(fontsize=6)
plt.ylabel('$T_{delay}/T_{forcing}$')
plt.ylim(0, None)
plt.xlabel('$T_{forcing}/T_{adj,0}$')
plt.xscale('log')



#
#
# ########################################################################################################################
# # Perturbation behaviour for x
# ########################################################################################################################
# om_fun = 2*np.pi/(np.asarray([1000, 100, 10, 1])*3600*24).reshape((1,4))#10**np.linspace(-8,-4,100)
# x = (np.linspace(0.001, 4, 100)*A*Kh/Q0).reshape((100,1))
# gamma = -1j*ssea*Q0/(A**2*Kh*om_fun)
# r2 = -Q0/(2*A*Kh)-np.sqrt((Q0**2-4*1j*om_fun*A**2*Kh)/(4*Kh**2*A**2))
# exp_fun = np.exp(-Q0/(A*Kh)*x)-np.exp(r2*x)
# s1 = gamma*exp_fun
# s1_an = -ssea/(A*Kh)*x*np.exp(-Q0*x/(A*Kh))
# s0 = ssea*np.exp(-Q0*x/(A*Kh))
#
# phi = np.angle(s1)*180/np.pi
# phi_an = A*np.pi/Q0**2*(Q0*x+2*A*Kh)  *180/np.pi        # analytical phase*T
#
# T = 2*np.pi/om_fun
#
# plt.figure(3, figsize=(2,2))
# plt.subplot(2,2,1)
# #plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
# plt.plot(x/1000,  np.abs(s1))
# plt.plot(x/1000,  np.abs(s1_an), 'k--')
# plt.ylabel('magnitude ($psu.s/m^3$)')
# plt.xlabel('x (km)')
# plt.legend(['1000 d', '100 d', '10 d', '1 d'])
# # plt.xscale('log')
# plt.xlim(0, None)
#
# plt.subplot(2,2,2)
# #plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
# plt.plot(x/1000,  (phi/360*T/Tadj+.5*T/Tadj))
# plt.plot(x/1000, phi_an/360/Tadj*np.ones(len(x)), 'k--')
# plt.plot(x/1000, np.zeros(len(x)), 'k--')
# plt.ylabel('$T_{delay}/T_{adj}$')
# plt.xlabel('x (km)')
# # plt.xscale('log')
# plt.xlim(0, None)
#
# plt.subplot(2,2,3)
# #plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
# plt.plot(x/1000,  phi)
# plt.plot(x/1000,  -90*np.ones(len(x)), 'k--')
# plt.plot(x/1000,  -180*np.ones(len(x)), 'k--')
# plt.ylabel('phase difference (deg)')
# plt.xlabel('x (km)')
# # plt.yscale('log')
# plt.xlim(0, None)
#
# plt.subplot(2,2,4)
# #plt.plot(T/Tadj,  np.abs(phi/360)*T/Tadj-.5*T/Tadj)
# plt.plot(x/1000,  np.abs(s1)/s0*Q0)
# # plt.plot(x/1000,  np.abs(s1_an), 'k--')
# plt.ylabel(r'$\frac{|s_1|}{s^0} \, \frac{Q^0}{|F[Q^1]|}$')
# plt.xlabel('x (km)')
# plt.legend(['1000 d', '100 d', '10 d', '1 d'])
# # plt.xscale('log')
# plt.xlim(0, None)
st.show()

