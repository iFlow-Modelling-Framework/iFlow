import numpy as np
import pycwt as wavelet
import scipy.interpolate
import scipy.optimize
import scipy.linalg


def dispersion_peak(x, x_k, t, Q, Kh, ssea, A, var='Q', more_output=False):
    ########################################################################################################################
    ## Init
    ############ ############################################################################################################
    jmax = len(x)
    tmax = len(t)
    s = np.zeros((jmax, tmax))
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    ## S at t=0
    s_prev = ssea[0]*np.exp(-Q[0]/(Kh[0]*A)*x)
    s[:, 0] = s_prev

    ## index corresponding to k for background Q
    x_ind = np.argmin(np.abs(x-x_k))

    ########################################################################################################################
    ## Time integration
    ########################################################################################################################
    for n in range(len(t[:-1])):
        M = np.zeros((3, jmax))
        rhs = np.zeros(jmax)
        M[0,2:] = -0.5*(A*Kh[n+1]+A*Kh[n+1])/dx**2 - Q[n+1]/dx
        M[1,1:-1] = 0.5*(A*Kh[n+1]+A*Kh[n+1])/dx**2 + 0.5*(A*Kh[n+1]+A*Kh[n+1])/dx**2 + Q[n+1]/dx + A/dt
        M[2,:-2] = -0.5*(A*Kh[n+1]+A*Kh[n+1])/dx**2
        rhs[1:-1] = A*s_prev[1:-1]/dt

        M[1,0] = 1
        rhs[0] = ssea[n+1]
        M[1,-1] = 1
        rhs[-1] = 0

        snew = scipy.linalg.solve_banded((1,1),M,rhs)
        s[:, n+1] = snew
        s_prev = snew

    ########################################################################################################################
    ## Wavelets
    ########################################################################################################################
    ## Settings
    if var =='Q':
        varval = Q
    elif var=='Kh':
        varval = Kh
    elif var=='ssea':
        varval = ssea
    data1 = dict(name=var, nick=var, var=varval)
    data2 = dict(name='s', nick='s', var=s[x_ind,:])
    mother = wavelet.Morlet(6)          # Morlet mother wavelet with m=6
    s1 = data1['var']
    s2 = data2['var']

    slevel = 0.7                       # Significance level
    dj = 1/12                           # Twelve sub-octaves per octaves
    s0 = -1  # 2 * dt                   # Starting scale, here 6 months
    J = -1  # 7 / dj                    # Seven powers of two with dj sub-octaves
    n = tmax
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

    aWCT[np.where(points1<.9)] = np.nan

    ## Apply COI
    coi1_matrix = (np.ones([1, len(freqs1)]) * coi1[:, None]).T
    period_matrix = (np.ones([1, n]) * period1[:, None])
    aWCT[np.where(period_matrix>coi1_matrix)] = np.nan

    from copy import copy
    aWCT1 = copy(aWCT)
    aWCT2 = copy(aWCT)
    aWCT1[:, int(len(t)/2):] = np.nan
    aWCT2[:, :int(len(t)/2)] = np.nan

    if var=='Q':
        Tdelay = (aWCT/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))
    else:
        Tdelay = (aWCT/(2*np.pi))*cor_period.reshape((len(cor_period),1))
    # Tdelay1 = (aWCT1/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))
    # Tdelay2 = (aWCT2/(2*np.pi)+1/2)*cor_period.reshape((len(cor_period),1))

    tforcing_matrix = period1[:,None]*np.ones((1,n))
    aWCT_sgn = np.ones(aWCT.shape)
    aWCT_sgn[np.where(np.isnan(aWCT))] = np.nan

    ########################################################################################################################
    ## Compute Tadj and Tdelay
    ####################################################################################################
    # average Tadj over window size
    Tadj_matrix = np.zeros((len(freqs1), len(t)))
    for i in range(len(freqs1)):
        M = int(np.ceil(2*np.pi*period1[i]/dt))
        y = np.linspace(-np.pi,np.pi, M)
        gaus = np.exp(-.5*y**2)/(np.sqrt(2*np.pi))

        Qmean = np.zeros(len(t))
        Khmean = np.zeros(len(t))
        for j in  range(0, len(t)):
            Qr = Q[np.maximum(j-int(np.ceil(M/2)), 0):np.minimum(j+int(np.floor(M/2)), len(t)-1)]
            Khr = Kh[np.maximum(j-int(np.ceil(M/2)), 0):np.minimum(j+int(np.floor(M/2)), len(t)-1)]
            if np.maximum(j-int(np.ceil(M/2)), 0)==0:
                gaus_t = gaus[-len(Qr):]
            elif np.minimum(j+int(np.floor(M/2)), len(t)-1) ==len(t)-1:
                gaus_t = gaus[:len(Qr)]
            else:
                gaus_t = gaus

            Qmean[j] = np.mean(gaus_t*Qr)/np.mean(gaus_t)
            Khmean[j] = np.mean(gaus_t*Khr)/np.mean(gaus_t)
            Tadj_matrix[i,:] = A**2*Khmean/Qmean**2

    # aWCT_sgn = np.ones(aWCT.shape)
    # aWCT_sgn[np.where(np.isnan(aWCT))] = np.nan
    if more_output:
        return s2, tforcing_matrix, Tadj_matrix, Tdelay, cor_period, aWCT_sgn, coi2, period2
    else:
        return s2, tforcing_matrix, Tadj_matrix, Tdelay
