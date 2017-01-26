"""
SedDynamic

Date: 24-01-2017
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction


class SedDynamicSecond:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SedDynamic')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        self.submodulesToRun = self.input.v('submodules')
        # H = self.input.v('H', range(0, jmax+1))

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])  # assume it is constant in time; else division with AV fails
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = Av/PrSchm

        # ws = np.zeros((jmax+1, kmax+1, fmax+1))
        ws = self.input.v('ws', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax+1, kmax+1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)

        # erosion
        if self.input.v('u1', 'river') is not None:
            E = self.erosion_Chernetsky(ws, Kv, 2)
            Fbed[:, :, fmax:, 0] = -E

            ################################################################################################################
            # Solve equation
            ################################################################################################################
            c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
            c = c.reshape((jmax+1, kmax+1, ftot))
        else:
            c = np.zeros((jmax+1, kmax+1, ftot))

        d = {}
        d['hatc2'] = {}
        d['hatc2']['a'] = {}
        d['hatc2']['a']['erosion'] = {}
        d['hatc2']['a']['erosion']['river_river'] = ny.eliminateNegativeFourier(c, 2)
        return d

    def erosion_Chernetsky(self, ws, Kv, tau_order):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        ## 1. bed shear stress
        # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
        taub = []
        Kv = np.concatenate((Kv, np.zeros((jmax+1, kmax+1, fmax+1))), 2)
        ws = np.concatenate((ws, np.zeros((jmax+1, kmax+1, fmax+1))), 2)
        for i in range(0, tau_order+1):
            if i == 1:
                uz = self.input.d('u1', 'river', range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            elif i ==2:
                uz = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
            else:
                uz = self.input.d('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            uz = np.concatenate((uz, np.zeros((jmax+1, 1, fmax+1))),2)
            taub.append(ny.complexAmplitudeProduct(Kv[:, [kmax], :], uz, 2))

        # amplitude
        tau_amp = (np.sum(np.abs(sum(taub)), axis=-1)+10**-3).reshape((jmax+1, 1, 1))   # NB not entirely the same as in first order
        taub = [t/tau_amp for t in taub]

        # absolute value
        c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs
        taub_abs = np.zeros(taub[0].shape, dtype=complex)
        if tau_order==0:
            taub_abs[:, :, 0] = c[0]
        taub_abs += c[2]*self.umultiply(2, tau_order, taub)
        taub_abs += c[4]*self.umultiply(4, tau_order, taub)
        taub_abs += c[6]*self.umultiply(6, tau_order, taub)
        taub_abs += c[8]*self.umultiply(8, tau_order, taub)

        taub_abs = taub_abs*tau_amp

        ########################################################################################################################
        # Leading order abs - Ronald using absoluteU
        ########################################################################################################################
        # taub = [t*tau_amp for t in taub]
        # taub_abs = np.zeros((taub[0].shape), dtype=complex)
        # taub[0][:,:,1] += 10**-6
        # if tau_order == 0:
        #     taub_abs[:, :, 0] = ny.absoluteU(taub[0][:, :, 1], 0)
        #     for i in range(1, fmax+1):
        #         taub_abs[:, :, i] = ny.absoluteU(taub[0][:, :, 1], i)+np.conj(ny.absoluteU(taub[0][:, :, 1], -i))
        #
        # else:
        #     sguM2 = ny.signU(taub[0][:, :, 1], 1)
        #     sguM6 = ny.signU(taub[0][:, :, 1], 3)
        #     # Calculate the M2 contribution of u1 * u0 / |u0| at the bottom, which can be separated into a part that is due
        #     # to the M0-part of u1 and a part that is due to the M4-part of u1
        #     uM0 = 2. * taub[1][:, :, 0] * sguM2
        #     uM4 = taub[1][:, :, 2]*np.conj(sguM2) + np.conj(taub[1][:, :, 2]) * sguM6
        #     taub_abs[:, :, 1] = uM0 + uM4
        ########################################################################################################################
        # import matplotlib.pyplot
        # p = plt.plot(abs(taub_abs[:, 0, 1]))
        # # # plt.plot(taub_abs2[:, 0, 0], '--', color = p[0].get_color())
        # plt.show()

        ## 2. erosion
        rhos = self.input.v('RHOS')
        rho0 = self.input.v('RHO0')
        gred = self.input.v('G')*(rhos-rho0)/rho0
        ds = self.input.v('DS')
        finf = 1 #self.input.v('finf')

        hatE = finf*rhos/(gred*ds)*ny.complexAmplitudeProduct(ws[:,[kmax],:], taub_abs, 2)
        return hatE[:, :, :fmax+1]

    def umultiply(self, pow, N, u):
        """ Compute the sum of all possible combinations yielding the power 'pow' of signal 'u' with a total order 'N'
        i.e. (u^pow)^<N>
        """
        v = 0
        if pow>2:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(self.umultiply(2, i, u), self.umultiply(pow-2, N-i, u), 2)
        else:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(u[i], u[N-i], 2)
        return v


