import numpy as np
import nifty as ny

def shearstress(Av, tau_order, data, submodule=None):
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    if submodule is None:
        submodule = (None, )*(tau_order+1)

    ## 1. bed shear stress
    # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
    taub = []
    Av = np.concatenate((Av, np.zeros((jmax+1, kmax+1, fmax+1))), 2)
    for i in range(0, tau_order+1):
        if submodule[i] is None:
            uz = data.d('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
        else:
            uz = data.d('u'+str(i), submodule[i], range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
        if uz is None:
            uz = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        uz = np.concatenate((uz, np.zeros((jmax+1, 1, fmax+1))),2)
        taub.append(ny.complexAmplitudeProduct(Av[:, [kmax], :], uz, 2))

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
