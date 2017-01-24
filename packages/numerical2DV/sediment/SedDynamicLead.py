"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction


class SedDynamicLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        self.logger.info('Running module SedDynamic')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        # H = self.input.v('H', range(0, jmax+1))

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])  #TODO: better input for PrSchm
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = Av/PrSchm

        # ws = np.zeros((jmax+1, kmax+1, fmax+1))
        # ws[:,:,0] = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), 0)
        ws = self.input.v('ws', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        ws[:,:,1:] = 0                                                                 #TODO: better input for Ws

        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax+1, kmax+1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)

        # erosion
        E = self.erosion_Chernetsky(ws, Kv)
        Fbed[:, :, :, 0] = -E

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
        c = c.reshape((jmax+1, kmax+1, ftot, ftot))

        d = {}
        d['hatc0'] = c
        return d

    def erosion_Chernetsky(self, ws, Kv):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        ## 1. bed shear stress
        uz = self.input.d('u0', range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
        taub = ny.complexAmplitudeProduct(Kv[:, [kmax], :], uz, 2)  #NB bed shear stress / rho_0

        # amplitude
        tau_amp = (np.sum(np.abs(taub), axis=-1)+10**-3).reshape((jmax+1, 1, 1))
        taub = taub/tau_amp

        # absolute value
        c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs
        taub_abs = np.zeros(taub.shape, dtype=complex)
        taub_abs[:, :, 0] = c[0]
        u2 = ny.complexAmplitudeProduct(taub, taub, 2)
        taub_abs += c[2]*u2
        u4 = ny.complexAmplitudeProduct(u2, u2, 2)
        taub_abs += c[4]*u4
        u6 = ny.complexAmplitudeProduct(u2, u4, 2)
        taub_abs += c[6]*u6
        del u2, u6
        u8 = ny.complexAmplitudeProduct(u4, u4, 2)
        taub_abs += c[8]*u8

        taub_abs = taub_abs*tau_amp

        ########################################################################################################################
        # Leading order abs - Ronald using absoluteU
        ########################################################################################################################
        # taub = taub*tau_amp
        # taub_abs = np.zeros((taub.shape), dtype=complex)
        # taub[:,:,1] += 10**-6
        # taub_abs[:, :, 0] = ny.absoluteU(taub[:, :, 1], 0)
        # for i in range(1, fmax+1):
        #     taub_abs[:, :, i] = ny.absoluteU(taub[:, :, 1], i)+np.conj(ny.absoluteU(taub[:, :, 1], -i))
        ########################################################################################################################
        # Plotting
        ########################################################################################################################
        # import matplotlib.pyplot
        # p = plt.plot(taub_abs[:, 0, 0])
        # # # plt.plot(taub_abs2[:, 0, 0], '--', color = p[0].get_color())
        # p = plt.plot(abs(taub_abs[:, 0, 2]))
        # # # plt.plot(abs(taub_abs2[:, 0, 2]), '--', color = p[0].get_color())
        # plt.show()

        ## 2. erosion
        rhos = self.input.v('RHOS')
        rho0 = self.input.v('RHO0')
        gred = self.input.v('G')*(rhos-rho0)/rho0
        ds = self.input.v('DS')
        finf = self.input.v('finf')

        hatE = finf*rhos/(gred*ds)*ny.complexAmplitudeProduct(ws[:,[kmax],:], taub_abs, 2)
        return hatE




