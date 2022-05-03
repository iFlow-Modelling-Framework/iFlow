"""
Date:
Authors:
"""
import copy
import logging
import numpy as np
from src.util.diagnostics import KnownError
from nifty import toList
import numbers
import os
from itertools import product
import nifty as ny
import matplotlib.pyplot as plt
import step as st
from nifty.harmonicDecomposition import absoluteU
from scipy import integrate

class Plot1ch:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input

        return

    def run(self):

        fmax = self.input.v('grid', 'maxIndex', 'f')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        jmax = self.input.v('grid', 'maxIndex', 'x')

        x = self.input.v('grid', 'axis', 'x')             
        z = self.input.v('grid', 'axis', 'z', 0)            
        x_km = ny.dimensionalAxis(self.input.slice('grid'), 'x', x=x, z=0, f=0) 

        L = self.input.v('grid', 'high', 'x')            
        B = self.input.v('B', x=x, z=[0], f=[0])[:,0,0] 
   
        zeta0 = self.input.v('zeta0', x=x, z=0, f=range(fmax+1))


        c = self.input.v('c0', x=x, z=z, f=range(0, fmax+1)) #+ self.input.v('c1', x=x, z=z, f=range(0, fmax+1)) + self.input.v('c2', x=x, z=z, f=range(0, fmax+1))

        c0 = np.real(c[:,-1,0])
 
        ws = 2.e-3
        finf = 1e-4
        sf = 0.002
        rho0 = 1000.
        rhos = 2650.
        Gprime = self.input.v('G') * (rhos - rho0) / rho0
        ds = self.input.v('DS')
        # omega = self.input.v('OMEGA')

        # complex amplitudes of M4 deposition at the bottom
        cbM0 = self.input.v('c0', range(jmax+1), -1, 0)
        cbM4 = self.input.v('c0', range(jmax+1), -1, 2)
        # D = ws * (self.input.v('c0', range(jmax+1), -1, 0) + self.input.v('c0', range(jmax+1), -1, 2))

        # complex amplitudes of M4 erosion by M2 tide
        u0b = self.input.v('u0', range(jmax+1), -1, 1)
        ubM0 = absoluteU(u0b, 0)
        ubM4 = absoluteU(u0b, 2)

        # potential erosion copied from SedimentCapacity module
        Ehat = ws * rhos * finf * sf / (Gprime * ds)
        
        # DE = D - E
        t = np.linspace(0, 2*np.pi, 100)
        Sbed = np.zeros(jmax+1, dtype='float64')

        # for i in range(jmax+1):
        #     # time-series of deposition at location i
        #     D = ws * (cbM0[i] + np.real(cbM4[i] * np.exp(2j*t)))

        #     # time-series of erosion at location i
        #     # E = np.abs(np.real(ws * rhos * finf * sf / (Gprime * ds) * (cM4[i] * np.exp(2j*t))))
        #     E = Ehat * (ubM0[i]*0 + np.real(ubM4[i]* np.exp(2j*t)))   

        #     Sbed_tmp = integrate.cumtrapz(D - E, x=t, initial=0)

        #     Sbed[i] = np.mean(Sbed_tmp - np.min(Sbed_tmp))
        #     Sbed[i] = np.trapz(Sbed_tmp, x=t) / (2*np.pi)

        # plt.figure()
        # plt.plot(self.input.v('Smod')[:,0,0])

        # plt.figure()
        # # plt.plot(integrate.cumtrapz(np.real((D[7] - E[7]) * np.exp(1j*omega*t)), x=t, initial=0))
        # # plt.plot(np.abs((D[5]) * np.exp(2j*t)))
        # # plt.plot(np.real(( E[5]) * np.exp(2j*t)) + np.abs(np.min(np.abs(D[5] - E[5]))))
        # i=20
        # D = ws * (cbM0[i] + np.real(cbM4[i] * np.exp(2j*t)))
        # # E = np.abs(np.real(
        # #      ws * rhos * finf * sf / (Gprime * ds) * (u0b[i] * np.exp(1j*t))
        # # ))
        # E = Ehat * (ubM0[i] + np.real(ubM4[i]* np.exp(2j*t)))
        # plt.plot(D)
        # plt.plot(E)
        # DEint = integrate.cumtrapz(((D - E)), x=t, initial=0)
        # plt.plot(DEint - np.min(DEint))
        # print(np.mean(D-E))

        # plt.plot()



        jmax = self.input.v('grid', 'maxIndex', 'x')

        # Plot bottom concentration, erodibility, transport with the capacity
        T = self.input.v('T')
        F = self.input.v('F')
        f = self.input.v('f0')[:,0]
        # f = self.input.v('c0', range(jmax+1), -1, 0) / self.input.v('hatc0','a')
        transport = (B * T * f + B * F * np.gradient(f, x_km, edge_order=2))

        # 0: erodibility
        # 1: bottom concentration
        # 2: transport capacity
        c00 = np.real(self.input.v('hatc0', 'a', range(jmax+1), -1, 0)) 
        # alpha = ny.integrate(c00, 'z', kmax, 0, self.input.slice('grid'))[:, 0]

        fig, axs = plt.subplots(2, 1, figsize=(9, 9))
        axs[0].plot(x_km, f)
        axs[0].set_ylabel('f')
        axs[0].set_ylim([-0.1,1.1])
        # axs[1].plot(x_km, c0)
        # axs[1].plot(x_km, c00, '--')
        # axs[1].set_ylabel('$c_b$')
        axs[1].plot(x_km, transport)
        # axs[1].plot(x_km, B*T, ls='--')
        axs[1].set_ylabel('transport')
        # axs[1].set_ylim([-11, 30])

        # for mod in ['TM2', 'TM4']:
        #     plt.figure(figsize=(8,8))
        #     plt.plot(x_km, self.input.v('T', 'tide', mod))
        #     plt.ylabel(mod)

        # plt.figure(figsize=(8,8))
        # plt.plot(x_km, f)
        # plt.ylabel('f')
        # plt.ylim([0,1])


        # plt.figure()
        # plt.plot(x_km, B)



       # self.input.v('hatc0', 'a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # s = self.input.v('s0', range(jmax+1), 0, 0)
        # sx = np.gradient(s, x_km, edge_order=2)
        # plt.figure()
        # plt.plot(x_km, s)
        # plt.ylabel('salinity')
        # plt.ylim([0,30])

        """kmax: z=-H"""
        # plt.figure()
        # plt.plot(x_km, np.abs(u0[:,0,1]))
        # plt.ylabel('$u^0_1$')
        
        mod = 'nostress'

        """Hydro"""
        # fig, axs = plt.subplots(4, 1, figsize=(9, 9))
        # zeta1 = self.input.v('zeta1', mod, range(jmax+1), kmax, range(fmax+1))
        # u1 = self.input.v('u1', mod, range(jmax+1), kmax, range(fmax+1))   

        # # axs[0].plot(x_km, f)
        # axs[0].plot(x_km, zeta1[:, 0])
        # axs[0].set_ylabel('$\zeta^1_0$ ' + mod)

        # axs[1].plot(x_km, u1[:, 0])
        # axs[1].set_ylabel('$u^1_0$ ' + mod)


        # axs[2].plot(x_km, np.abs(zeta1[:, 2]))
        # axs[2].set_ylabel('$\zeta^1_2$ ' + mod)

        # axs[3].plot(x_km, np.abs(u1[:, 2]))
        # axs[3].set_ylabel('$u^1_2$ ' + mod)

        # fig, axs = plt.subplots(4, 1, figsize=(9, 9))
        # zeta0 = self.input.v('zeta0', 'tide', range(jmax+1), 0, 1)
        # u0 = self.input.v('u0', 'tide', range(jmax+1), 0, 1)   

        # # axs[0].plot(x_km, f)
        # axs[0].plot(x_km, np.abs(zeta0))
        # axs[0].set_ylabel('$|\zeta^0|$ ' )

        # axs[1].plot(x_km, u0)
        # axs[1].set_ylabel('$|u^0|$ ' )


        # axs[2].plot(x_km, np.angle(zeta0))
        # axs[2].set_ylabel('arg $\zeta^0$ ')

        # axs[3].plot(x_km, np.angle(u0))
        # axs[3].set_ylabel('arg $u^0$ ')
        

        """SSC"""


        # mod = 'TM0'

        # for mod in self.input.getKeysOf('T'):
            
        #     T = self.input.v('T', mod)
        #     if np.max(np.abs(T) )> 1e-7:
        #         plt.figure()
        #         plt.plot(x_km, B * T)
        #         plt.ylabel('T ' + mod)
        # T = self.input.v('T', mod)
        # plt.figure()
        # plt.plot(x_km, B * T)
        # plt.ylabel('T ' + mod)

        # plt.figure()
        # c0x = self.input.d('hatc0', 'a', range(jmax+1), 0, 0, dim='x') 
        # c0  = self.input.v('hatc0', 'a', range(jmax+1), 0, 0) 
        # tes = np.gradient(c0, x_km, edge_order=2)
        # # plt.plot(x_km, c0x)
        # plt.plot(x_km, c0x/tes)
        # plt.ylabel('T ' + mod)

        plt.show()
        return

# c0 the same
# diff in the dirivatives!!