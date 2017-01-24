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
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        self.logger.info('Running module SedDynamic first order')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        OMEGA = self.input.v('OMEGA')

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = Av/PrSchm

        ws = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        if 'sedadv' in self.submodulesToRun and 'sedadv_ax' not in self.submodulesToRun:
            self.submodulesToRun.append('sedadv_ax')
        if 'erosion' in self.submodulesToRun and 'erosion_a1' not in self.submodulesToRun:
            self.submodulesToRun.append('erosion_a1')
        nRHS = len(self.submodulesToRun)

        F = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)

        c0 = self.input.v('hatc0')
        cx0 = ny.derivative(c0.reshape((jmax+1, kmax+1, 1, ftot)), 'x', self.input.slice('grid')).reshape((jmax+1, kmax+1, ftot))
        cz0 = ny.derivative(c0.reshape((jmax+1, kmax+1, 1, ftot)), 'z', self.input.slice('grid')).reshape((jmax+1, kmax+1, ftot))

        # 1. Erosion
        if 'erosion' in self.submodulesToRun:
            # erosion due to first-order bed shear stress
            E = self.erosion_Chernetsky(ws, Kv, 1)
            Fbed[:, :, :, self.submodulesToRun.index('erosion')] = -E

            # erosion due to first-order erodability
            E = self.erosion_Chernetsky(ws, Kv, 0)
            Fbed[:, :, :, self.submodulesToRun.index('erosion_a1')] = -E

        # 2. Advection
        if 'sedadv' in self.submodulesToRun:
            u0 = np.concatenate((np.zeros((jmax+1, kmax+1, fmax), dtype=complex), self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))), 2)
            w0 = np.concatenate((np.zeros((jmax+1, kmax+1, fmax), dtype=complex), self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))), 2)

            eta = ny.complexAmplitudeProduct(u0, cx0, 2, includeNegative=True)+ny.complexAmplitudeProduct(w0, cz0, 2, includeNegative=True)
            F[:, :, :, self.submodulesToRun.index('sedadv')] = -eta
            F[:, :, :, self.submodulesToRun.index('sedadv_ax')] = -ny.complexAmplitudeProduct(u0, c0, 2, includeNegative=True)

        # 3. First-order fall velocity
        if 'settling' in self.submodulesToRun:
            # surface and internal terms
            ws1 = np.concatenate((np.zeros((jmax+1, 1, fmax), dtype=complex), self.input.v('ws1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))), 2)
            ksi = ny.complexAmplitudeProduct(ws1, c0, 2, includeNegative=True)
            ksiz = ny.derivative(ksi, 'z', self.input.slice('grid'))
            zeta0 = np.concatenate((np.zeros((jmax+1, fmax), dtype=complex), self.input.v('zeta0', range(0, jmax+1), 0, range(0, fmax+1))), 1)

            F[:, :, :, self.submodulesToRun.index('settling')] = ksiz
            Fsurf[:, 0, :, self.submodulesToRun.index('settling')] = -ny.complexAmplitudeProduct(ksiz[:,0,:], zeta0, 1, includeNegative=True)

            # adjustment to erosion
            E = self.erosion_Chernetsky(ws1, Kv, 0)
            Fbed[:, :, :, self.submodulesToRun.index('settling')] = -E

        # 4. First-order eddy diffusivity
        if 'mixing' in self.submodulesToRun:
            # surface, bed and internal terms
            Kv1 = np.concatenate((np.zeros((jmax+1, 1, fmax), dtype=complex), self.input.v('Av1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))/PrSchm), 2)
            psi = ny.complexAmplitudeProduct(Kv1, cz0, 2, includeNegative=True)
            psiz = ny.derivative(psi, 'z', self.input.slice('grid'))

            F[:, :, :, self.submodulesToRun.index('mixing')] = psiz
            Fsurf[:, 0, :, self.submodulesToRun.index('mixing')] = -psi[:, 0, :]
            Fbed[:, 0, :, self.submodulesToRun.index('mixing')] = -psi[:, -1, :]

            # adjustment to erosion
            E = self.erosion_Chernetsky(ws, Kv1, 0)
            Fbed[:, :, :, self.submodulesToRun.index('mixing')] = -E

        # 5. No-flux surface correction
        if 'noflux' in self.submodulesToRun:
            zeta0 = np.concatenate((np.zeros((jmax+1, 1, fmax), dtype=complex), self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))), 2)
            D = np.zeros(c0[:, [0], Ellipsis].shape, dtype=complex)
            D[:, :, range(0, ftot), range(0, ftot)] = np.arange(-fmax, fmax+1)*1j*OMEGA
            Dc0 = ny.arraydot(D, c0[:, [0], Ellipsis])

            chi = ny.complexAmplitudeProduct(Dc0, zeta0, 2, includeNegative=True)
            Fsurf[:, :, :, self.submodulesToRun.index('noflux')] = -chi

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        cmatrix = self.input.v('cMatrix')
        if cmatrix:
            c, cMatrix = cFunction(None, cmatrix, F, Fsurf, Fbed, self.input, hasMatrix = False)
        else:
            c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
        c = c.reshape((jmax+1, kmax+1, ftot, nRHS))

        ################################################################################################################
        # Prepare output
        ################################################################################################################
        d = {}
        d['hatc1_a'] = {}
        d['hatc1_ax'] = {}
        d['hatc1_a1'] = {}
        for i, submod in enumerate(self.submodulesToRun):
            if submod == 'erosion_a1':
                d['hatc1_a1']['erosion'] = c[:, :, :, i]
            elif submod == 'sedadv_ax':
                d['hatc1_ax']['sedadv'] = c[:, :, :, i]
            else:
                d['hatc1_a'][submod] = c[:, :, :, i]
        if 'erosion' not in self.submodulesToRun:
            d['hatc1_a1'] = 0
        if 'sedadv' not in self.submodulesToRun:
            d['hatc1_ax'] = 0

        # self.input.merge(d)
        # plt.plot(abs(self.input.v('hatc1_a', x=0.75, z=np.linspace(0,1, 50), f=1)))
        # plt.plot(abs(self.input.v('hatc a', 'c12', x=0.75, z=np.linspace(0,1, 50), f=1)))
        # plt.show()

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
        finf = self.input.v('finf')

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
