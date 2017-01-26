"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction


class SedDynamicFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SedDynamic first order')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        OMEGA = self.input.v('OMEGA')
        self.submodulesToRun = self.input.v('submodules')

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])       # assume it is constant in time; else division with AV fails
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = Av/PrSchm

        ws = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        if 'sedadv' in self.submodulesToRun and 'sedadv_ax' not in self.submodulesToRun:
            self.submodulesToRun.append('sedadv_ax')
        nRHS = len(self.submodulesToRun)

        F = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)

        c0 = self.input.v('hatc0')
        cx0 = self.input.d('hatc0', dim='x')
        cz0 = self.input.d('hatc0', dim='z')
        # 1. Erosion
        if 'erosion' in self.submodulesToRun:
            # erosion due to first-order bed shear stress
            E = self.erosion_Chernetsky(ws, Kv, 1)
            Fbed[:, :, fmax:, self.submodulesToRun.index('erosion')] = -E

        # 2. Advection
        if 'sedadv' in self.submodulesToRun:
            u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            w0 = self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

            eta = ny.complexAmplitudeProduct(u0, cx0, 2)+ny.complexAmplitudeProduct(w0, cz0, 2)
            F[:, :, fmax:, self.submodulesToRun.index('sedadv')] = -eta
            F[:, :, fmax:, self.submodulesToRun.index('sedadv_ax')] = -ny.complexAmplitudeProduct(u0, c0, 2)

        # 3. First-order fall velocity
        if 'settling' in self.submodulesToRun:
            # surface and internal terms
            ws1 = self.input.v('ws1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            ksi = ny.complexAmplitudeProduct(ws1, c0, 2)
            ksiz = ny.derivative(ksi, 'z', self.input.slice('grid'))
            zeta0 = self.input.v('zeta0', range(0, jmax+1), 0, range(0, fmax+1))

            F[:, :, fmax:, self.submodulesToRun.index('settling')] = ksiz
            Fsurf[:, 0, fmax:, self.submodulesToRun.index('settling')] = -ny.complexAmplitudeProduct(ksiz[:,0,:], zeta0, 1)

            # adjustment to erosion
            E = self.erosion_Chernetsky(ws1, Kv, 0)
            Fbed[:, :, fmax:, self.submodulesToRun.index('settling')] = -E

        # 4. First-order eddy diffusivity
        if 'mixing' in self.submodulesToRun:
            # surface, bed and internal terms
            Kv1 = self.input.v('Av1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))/PrSchm
            psi = ny.complexAmplitudeProduct(Kv1, cz0, 2)
            psiz = ny.derivative(psi, 'z', self.input.slice('grid'))

            F[:, :, fmax:, self.submodulesToRun.index('mixing')] = psiz
            Fsurf[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, 0, :]
            Fbed[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, -1, :]

            # adjustment to erosion
            E = self.erosion_Chernetsky(ws, Kv1, 0)
            Fbed[:, :, fmax:, self.submodulesToRun.index('mixing')] = -E

        # 5. No-flux surface correction
        if 'noflux' in self.submodulesToRun:
            zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
            D = np.zeros((jmax+1, 1, fmax+1, fmax+1), dtype=complex)
            D[:, :, range(0, fmax+1), range(0, fmax+1)] = np.arange(0, fmax+1)*1j*OMEGA
            Dc0 = ny.arraydot(D, c0[:, [0], :])

            chi = ny.complexAmplitudeProduct(Dc0, zeta0, 2)
            Fsurf[:, :, fmax:, self.submodulesToRun.index('noflux')] = -chi

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        cmatrix = self.input.v('cMatrix')
        if cmatrix is not None:
            c, cMatrix = cFunction(None, cmatrix, F, Fsurf, Fbed, self.input, hasMatrix = True)
        else:
            c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
        c = c.reshape((jmax+1, kmax+1, ftot, nRHS))
        c = ny.eliminateNegativeFourier(c, 2)

        ################################################################################################################
        # Prepare output
        ################################################################################################################
        d = {}
        d['hatc1'] = {}
        d['hatc1']['a'] = {}
        d['hatc1']['ax'] = {}
        for i, submod in enumerate(self.submodulesToRun):
            if submod == 'sedadv_ax':
                d['hatc1']['ax']['sedadv'] = c[:, :, :, i]
            else:
                d['hatc1']['a'][submod] = c[:, :, :, i]
        if 'sedadv' not in self.submodulesToRun:
            d['hatc1']['ax'] = 0

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
            uz = self.input.d('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            uz = np.concatenate((uz, np.zeros((jmax+1, 1, fmax+1))),2)
            taub.append(ny.complexAmplitudeProduct(Kv[:, [kmax], :], uz, 2))

        # amplitude
        tau_amp = (np.sum(np.abs(sum(taub)), axis=-1)+10**-3).reshape((jmax+1, 1, 1))
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
