import numpy as np
import nifty as ny
import step as st
import matplotlib.pyplot as plt

def shearstress(tau_order, data, submodule=None, friction='Roughness'):
    # return shearstressCheb(tau_order, data, submodule=submodule, friction=friction)
    return shearstressGS(tau_order, data, submodule=submodule, friction=friction)
    # return shearstress_truncated(tau_order, data, submodule=submodule, friction=friction)

# def comparestresses(tau_order, data, submodule=None, friction='Roughness'):
#     taub_abs = shearstressGS(tau_order, data, submodule, friction=friction)
#     taub = shearstressCheb(tau_order, data, submodule, friction=friction)
#     taub_exact = shearstress_exact(tau_order, data, submodule, friction=friction)
#
#     st.configure()
#     plt.figure(1,figsize=(1,3))
#     for n in range(0, 3):
#         plt.subplot(1, 3, n+1)
#         plt.plot(np.real((taub[:-1, 0, n]-taub_exact[:-1, 0, n])/(taub_exact[:-1, 0, n]+10**-10)), label='Cheb')
#         plt.plot(np.real((taub_abs[:-1, 0, n]-taub_exact[:-1, 0, n])/(taub_exact[:-1, 0, n]+10**-10)), label='GS')
#         plt.legend()
#     plt.figure(2,figsize=(1,3))
#     for n in range(0, 3):
#         plt.subplot(1, 3, n+1)
#         plt.plot(np.imag((taub[:-1, 0, n]-taub_exact[:-1, 0, n])/(taub_exact[:-1, 0, n]+10**-10)), label='Cheb')
#         plt.plot(np.imag((taub_abs[:-1, 0, n]-taub_exact[:-1, 0, n])/(taub_exact[:-1, 0, n]+10**-10)), label='GS')
#         plt.legend()
#
#     st.show()

def shearstressGS(tau_order, data, submodule=None, friction='Roughness'):     # Shear stress following the formulation of Schramkowski
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    rho0 = data.v('RHO0')
    sf = data.v(friction, range(0, jmax+1), 0, 0)

    if submodule is None:
        submodule = (None, )*(tau_order+1)

    ## 1. bed shear stress
    # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
    ulist = []
    for i in range(0, tau_order+1):
        if submodule[i] is None:
            u = data.v('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1))
        else:
            u = data.v('u'+str(i), submodule[i], range(0, jmax+1), [kmax], range(0, fmax+1))
        if u is None:
            u = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        ulist.append(u)

    taub_abs = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
    if tau_order == 0:
        uabs0 = ny.absoluteU(ulist[0][:, 0, 1]+10**-6, 0)
        uabs2 = ny.absoluteU(ulist[0][:, 0, 1]+10**-6, 2)+np.conj(ny.absoluteU(ulist[0][:, 0, 1]+10**-6, -2))
        taub_abs[:, 0, 0] = rho0*sf*uabs0
        taub_abs[:, 0, 2] = rho0*sf*uabs2
    elif tau_order ==1:
        signu = np.zeros((jmax+1, 1, np.maximum(fmax+1, 4)), dtype=complex)
        signu[:, 0, 1] = ny.signU(ulist[0][:, 0, 1]+10**-6, 1) + np.conj(ny.signU(ulist[0][:, 0, 1]+10**-6, -1))
        signu[:, 0, 3] = ny.signU(ulist[0][:, 0, 1]+10**-6, 3) + np.conj(ny.signU(ulist[0][:, 0, 1]+10**-6, -3))
        if fmax+1 < 4:
            ulist[1] = np.concatenate((ulist[1], np.zeros((jmax+1, 1, 4-fmax-1))), 2)
        taub_abs = rho0*sf.reshape((jmax+1, 1, 1))*ny.complexAmplitudeProduct(ulist[1], signu, 2)
    elif tau_order ==2:
        uabs = np.abs(ulist[1][:, 0, 0])        # only for subtidal flows
        taub_abs[:, 0, 0] = rho0*sf*uabs

    return taub_abs[:, :, :fmax+1]

def shearstress_exact(tau_order, data, submodule=None, friction='Roughness'):     # Shear stress derived using time series (ordering is difficult)
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    rho0 = data.v('RHO0')
    sf = data.v(friction, range(0, jmax+1), 0, 0)

    if submodule is None:
        submodule = (None, )*(tau_order+1)

    ## 1. bed shear stress
    # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
    ulist = []
    for i in range(0, tau_order+1):
        if submodule[i] is None:
            u = data.v('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1))
        else:
            u = data.v('u'+str(i), submodule[i], range(0, jmax+1), [kmax], range(0, fmax+1))
        if u is None:
            u = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        ulist.append(u)
    u = sum(ulist)
    utim = ny.invfft(np.concatenate((u, np.zeros((jmax+1, 1, 500))), 2), 2)
    utim = np.abs(utim)
    uabs = ny.fft(utim, 2)[:, :, :fmax+1]
    taub_abs = rho0*sf.reshape((jmax+1, 1, 1))*uabs
    return taub_abs

def shearstressCheb(tau_order, data, submodule=None, friction='Roughness'):       # Shear stress using Chebyshev polynomials
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    if submodule is None:
        submodule = (None, )*(tau_order+1)

    ## 1. bed shear stress
    # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
    # Method 1: using Av

    taub1 = []
    for i in range(0, tau_order+1):
        taub1.append(0)
        for j in range(0, i+1):
            # friction
            if j == 0:
                Av = data.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            elif submodule[i] is None or submodule[i]== 'mixing':
                Av = data.v('Av'+str(i), range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            else:
                Av = np.zeros((jmax+1, kmax+1, fmax+1))
            # velocity shear
            q = i-j
            if submodule[q] is None:
                uz = data.d('u'+str(q), range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            else:
                uz = data.d('u'+str(q), submodule[q], range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            if uz is None:
                uz = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

            #  extend and multiply
            Av = np.concatenate((Av, np.zeros((jmax+1, kmax+1, fmax+1))), 2)
            uz = np.concatenate((uz, np.zeros((jmax+1, 1, fmax+1))),2)
            taub1[i] += ny.complexAmplitudeProduct(Av[:, [kmax], :], uz, 2)

    # Method 2: using sf (equivalent to using Av)
    taub = []
    for i in range(0, tau_order+1):
        taub.append(0)
        for j in range(0, i+1):
            # friction
            if j == 0:
                sf = data.v(friction, range(0, jmax+1), [0], range(0, fmax+1))
            elif submodule[i] is None or submodule[i]== 'mixing':
                sf = data.v(friction+str(i), range(0, jmax+1), [0], range(0, fmax+1))
            else:
                sf = np.zeros((jmax+1, 1, fmax+1))
            # velocity
            q = i-j
            if submodule[q] is None:
                u = data.v('u'+str(q), range(0, jmax+1), [kmax], range(0, fmax+1))
            else:
                u = data.v('u'+str(q), submodule[q], range(0, jmax+1), [kmax], range(0, fmax+1))
            if u is None:
                u = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
            # extend vectors and multiply
            sf = np.concatenate((sf, np.zeros((jmax+1, 1, fmax+1))), 2)
            u = np.concatenate((u, np.zeros((jmax+1, 1, fmax+1))),2)
            taub[i] += ny.complexAmplitudeProduct(sf, u, 2)

    taub = taub1
    for i, t in enumerate(taub):
        dif = np.max(np.abs(t - taub1[i]))
        if dif>1.e-15:
            print 'test stress ' + str(np.max(np.abs(t - taub1[i])))
            print 'order ' + str(tau_order)


    # amplitude
    tau_amp = (np.sum(np.abs(sum(taub)), axis=-1)+10**-3).reshape((jmax+1, 1, 1))
    taub = [t/tau_amp for t in taub]

    # absolute value
    c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs
    taub_abs = np.zeros(taub[0].shape, dtype=complex)
    if tau_order==0:
        taub_abs[:, :, 0] = c[0]
    taub_abs += c[2]*umultiply(2, tau_order, taub)
    taub_abs += c[4]*umultiply(4, tau_order, taub)
    taub_abs += c[6]*umultiply(6, tau_order, taub)
    taub_abs += c[8]*umultiply(8, tau_order, taub)

    rho0 = data.v('RHO0')
    taub_abs = taub_abs*tau_amp*rho0

    return taub_abs[:, :, :fmax+1]

def shearstress_truncated(tau_order, data, submodule=None, friction='Roughness'):     # Shear stress derived using time series (truncated, only for standard forcing conditions)
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    rho0 = data.v('RHO0')
    sf = data.v(friction, range(0, jmax+1), 0, 0)

    if submodule is None:
        submodule = (None, )*(tau_order+1)

    ## 1. bed shear stress
    # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
    ulist = []
    for i in range(0, 2):
        if submodule[i] is None:
            u = data.v('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1))
        else:
            u = data.v('u'+str(i), submodule[i], range(0, jmax+1), [kmax], range(0, fmax+1))
        if u is None:
            u = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        ulist.append(u)
    u = sum(ulist)
    utim = ny.invfft2(u, 2, 90)
    utim = np.abs(utim)
    uabs = ny.fft(utim, 2)[:, :, :fmax+1]
    taub_abs = rho0*sf.reshape((jmax+1, 1, 1))*uabs
    if tau_order == 0:
        taub_abs[:, :, 1] =0
    elif tau_order == 1:
        taub_abs[:, :, 0] =0
        taub_abs[:, :, 2] =0
    else:
        taub_abs[:, :, 1:] =0
    return taub_abs

def umultiply(pow, N, u):
        """ Compute the sum of all possible combinations yielding the power 'pow' of signal 'u' with a total order 'N'
        i.e. (u^pow)^<N>
        """
        v = 0
        if pow>2:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(umultiply(2, i, u), umultiply(pow-2, N-i, u), 2)
        else:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(u[i], u[N-i], 2)
        return v
