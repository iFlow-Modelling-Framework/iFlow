import numpy as np
import nifty as ny
from .shearstress import shearstress

def erosion(ws, tau_order, data, method='Chernetsky', submodule=None, friction='Roughness'):
    jmax = data.v('grid', 'maxIndex', 'x')
    kmax = data.v('grid', 'maxIndex', 'z')
    fmax = data.v('grid', 'maxIndex', 'f')
    rho0 = data.v('RHO0')

    taub_abs = shearstress(tau_order, data, submodule=submodule, friction=friction)
    ## 2. erosion
    finf = data.v('finf')
    if method == 'Partheniades':
        hatE = finf*taub_abs
    else:
        rhos = data.v('RHOS')
        gred = data.v('G')*(rhos-rho0)/rho0
        ds = data.v('DS')
        
        hatE = finf*rhos/(gred*ds*rho0)*ny.complexAmplitudeProduct(ws[:,[kmax],:], taub_abs, 2)
    
    return hatE[:, :, :fmax+1]

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
