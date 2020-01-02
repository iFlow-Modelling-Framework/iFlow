"""
SedimentCapacity - numerical method
Compute the sediment capacity at leading, first and second order using a numerical method.

From version: 2.5
Notes: merged on 9-1-19 from separate leading-, first- and second-order modules
Date: 09-01-19
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from .cFunction import cFunction
from .erosion import erosion


class SedimentCapacity:
    # Variables
    logger = logging.getLogger(__name__)
    timer = ny.Timer()

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.timer.tic()
        self.logger.info('Running module Sediment Capacity')

        ## Init
        d = {}
        
        self.submodulesToRun = ny.toList(self.input.v('submodules'))
        self.erosion_method = self.input.v('erosion_formulation')
        self.frictionpar = self.input.v('friction')  # friction parameter used for the erosion, by default the total roughness
        if self.frictionpar == None:
            self.frictionpar = 'Roughness'

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.Kv = self.input.v('Kv', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
        self.ws = self.input.v('ws0', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))

        ## Run orders
        d.update(self.leading_order(d))
        d.update(self.first_order(d))
        d.update(self.second_order_river(d))

        self.timer.toc()
        self.timer.disp('time sediment capacity')
        self.timer.reset()
        # d['hatc0']['a']['erosion'][:, :, 0] += 1e-4
        return d

    def leading_order(self, d):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2 * fmax + 1
        
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax + 1, kmax + 1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax + 1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax + 1, 1, ftot, 1], dtype=complex)

        # erosion
        E = erosion(self.ws, 0, self.input, self.erosion_method, friction=self.frictionpar)
        Fbed[:, :, fmax:, 0] = -E

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        # Coupled system
        c, cMatrix = cFunction(self.ws, self.Kv, F, Fsurf, Fbed, self.input, hasMatrix=False)
        c = c.reshape((jmax + 1, kmax + 1, ftot))
        hatc0 = ny.eliminateNegativeFourier(c, 2)

        # Uncoupled system
        # hatc0 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # hatc0[:, :, 0] = cFunctionUncoupled(self.ws, self.Kv, F, Fsurf, Fbed, self.input, 0).reshape((jmax+1, kmax+1))
        # hatc0[:, :, 2] = cFunctionUncoupled(self.ws, self.Kv, F, Fsurf, Fbed, self.input, 2).reshape((jmax+1, kmax+1))

        # correction at the bed (optional)
        # cbed00 = E[:, -1, 0]/self.ws[:, -1, 0]
        # frac = cbed00/hatc0[:, -1, 0]
        # hatc0 = hatc0*frac.reshape((jmax+1, 1, 1))

        ## analytical solutions
        # ahatc0 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # ahatc02 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        # self.ws = self.ws[:, 0, 0]
        # self.Kv = self.Kv[:, 0, 0]
        # OMEGA = self.input.v('OMEGA')
        # H = self.input.v('H', range(0, jmax+1))
        #
        # # M0
        # ahatc0[:, :, 0] = (E[:, 0, 0]/self.ws).reshape((jmax+1, 1))*np.exp(-(self.ws/self.Kv).reshape((jmax+1, 1))*(z+H.reshape((jmax+1, 1))))
        #
        #
        # # M4
        #
        # r1 = -self.ws/(2.*self.Kv)+np.sqrt(self.ws**2+8*1j*OMEGA*self.Kv)/(2*self.Kv)
        # r2 = -self.ws/(2.*self.Kv)-np.sqrt(self.ws**2+8*1j*OMEGA*self.Kv)/(2*self.Kv)
        # k2 = -E[:, 0, 2]*(self.Kv*(-r1*(self.ws+self.Kv*r2)/(self.ws+self.Kv*r1)*np.exp(-r1*H) + r2*np.exp(-r2*H)))**-1.
        # k1 = -k2*(self.ws+self.Kv*r2)/(self.ws+self.Kv*r1)
        # ahatc0[:, :, 2] = k1.reshape((jmax+1, 1))*np.exp(r1.reshape((jmax+1, 1))*z) + k2.reshape((jmax+1, 1))*np.exp(r2.reshape((jmax+1, 1))*z)

        # import step as st
        # import matplotlib.pyplot as plt
        #
        # st.configure()
        # plt.figure(1, figsize=(1,3))
        # plt.subplot(1,3,1)
        # plt.plot(np.real(ahatc0[0, :, 0]), z[0, :], label='analytical')
        # plt.plot(np.real(hatc0[0, :, 0]), z[0, :], label='numerical')
        # plt.xlim(0, np.max(np.maximum(abs(ahatc0[0, :, 0]), abs(hatc0[0, :, 0])))*1.05)
        #
        # plt.subplot(1,3,2)
        # plt.plot(np.abs(ahatc0[0, :, 2]), z[0, :], label='analytical')
        # # plt.plot(np.abs(ahatc02[0, :, 2]), z[0, :], label='analytical2')
        # plt.plot(np.abs(hatc0[0, :, 2]), z[0, :], label='numerical')
        # # plt.xlim(np.min(np.minimum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05, np.max(np.maximum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05)
        # plt.legend()
        #
        # plt.subplot(1,3,3)
        # plt.plot(np.imag(ahatc0[0, :, 2]), z[0, :], label='analytical')
        # # plt.plot(np.imag(ahatc02[0, :, 2]), z[0, :], label='analytical2')
        # plt.plot(np.imag(hatc0[0, :, 2]), z[0, :], label='numerical')
        # # plt.xlim(np.min(np.minimum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05, np.max(np.maximum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05)
        # plt.legend()
        # st.show()

        d['hatc0'] = {}
        d['hatc0']['a'] = {}
        d['hatc0']['a']['erosion'] = hatc0
        d['cMatrix'] = cMatrix
        return d

    def first_order(self, d):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2 * fmax + 1
        OMEGA = self.input.v('OMEGA')
        
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        if 'sedadv' in self.submodulesToRun and 'sedadv_ax' not in self.submodulesToRun:
            self.submodulesToRun.append('sedadv_ax')
        # determine number of submodules
        nRHS = len(self.submodulesToRun)
        if 'erosion' in self.submodulesToRun:
            keysu1 = self.input.getKeysOf('u1')
            try:
                keysu1.remove('mixing')  # flow due to 'mixing' should not be included in the erosion term
            except:
                pass
            self.submodulesToRun.remove('erosion')  # move to the end of the list
            self.submodulesToRun.append('erosion')
            nRHS = nRHS - 1 + len(self.input.getKeysOf('u1'))

        F = np.zeros([jmax + 1, kmax + 1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax + 1, 1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([jmax + 1, 1, ftot, nRHS], dtype=complex)

        self.input.merge(d)
        c0 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        cx0 = self.input.d('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        cz0 = self.input.d('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
        # 1. Erosion
        if 'erosion' in self.submodulesToRun:
            # erosion due to first-order bed shear stress
            # E = erosion(self.ws, Av, 1, self.input, self.erosion_method)                        # 24-04-2017 Obsolete
            # Fbed[:, :, fmax:, self.submodulesToRun.index('erosion')] = -E     # 24-04-2017 Obsolete
            for submod in keysu1:
                E = erosion(self.ws, 1, self.input, self.erosion_method, submodule=(None, submod), friction=self.frictionpar)
                Fbed[:, :, fmax:, len(self.submodulesToRun) - 1 + keysu1.index(submod)] = -E

        # 2. Advection
        if 'sedadv' in self.submodulesToRun:
            u0 = self.input.v('u0', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
            w0 = self.input.v('w0', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))

            eta = ny.complexAmplitudeProduct(u0, cx0, 2) + ny.complexAmplitudeProduct(w0, cz0, 2)
            F[:, :, fmax:, self.submodulesToRun.index('sedadv')] = -eta
            F[:, :, fmax:, self.submodulesToRun.index('sedadv_ax')] = -ny.complexAmplitudeProduct(u0, c0, 2)

        # 3. First-order fall velocity
        if 'fallvel' in self.submodulesToRun:
            # surface and internal terms
            ws1 = self.input.v('ws1', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
            ksi = ny.complexAmplitudeProduct(ws1, c0, 2)
            ksiz = ny.derivative(ksi, 'z', self.input.slice('grid'))

            F[:, :, fmax:, self.submodulesToRun.index('fallvel')] = ksiz
            Fsurf[:, 0, fmax:, self.submodulesToRun.index('fallvel')] = -ksi[:, 0, :]

            # adjustment to erosion; only if erosion depends on the settling velocity
            if self.erosion_method == 'Chernetsky':
                E = erosion(ws1, 0, self.input, self.erosion_method, friction=self.frictionpar)
                Fbed[:, :, fmax:, self.submodulesToRun.index('fallvel')] = -E

        # 4. First-order eddy diffusivity
        if 'mixing' in self.submodulesToRun:
            # surface, bed and internal terms
            Kv1 = self.input.v('Kv1', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
            psi = ny.complexAmplitudeProduct(Kv1, cz0, 2)
            psiz = ny.derivative(psi, 'z', self.input.slice('grid'))

            F[:, :, fmax:, self.submodulesToRun.index('mixing')] = psiz
            Fsurf[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, 0, :]
            Fbed[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, -1, :]

            # adjustment to erosion
            E = erosion(self.ws, 1, self.input, self.erosion_method, submodule=(None, 'mixing'), friction=self.frictionpar)
            Fbed[:, :, fmax:, self.submodulesToRun.index('mixing')] = -E

        # 5. No-flux surface correction
        if 'noflux' in self.submodulesToRun:
            zeta0 = self.input.v('zeta0', range(0, jmax + 1), [0], range(0, fmax + 1))
            D = np.zeros((jmax + 1, 1, fmax + 1, fmax + 1), dtype=complex)
            D[:, :, range(0, fmax + 1), range(0, fmax + 1)] = np.arange(0, fmax + 1) * 1j * OMEGA
            Dc0 = ny.arraydot(D, c0[:, [0], :])

            chi = ny.complexAmplitudeProduct(Dc0, zeta0, 2)
            Fsurf[:, :, fmax:, self.submodulesToRun.index('noflux')] = -chi

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        cmatrix = self.input.v('cMatrix')
        if cmatrix is not None:
            c, cMatrix = cFunction(None, cmatrix, F, Fsurf, Fbed, self.input, hasMatrix=True)
        else:
            c, cMatrix = cFunction(self.ws, self.Kv, F, Fsurf, Fbed, self.input, hasMatrix=False)
        c = c.reshape((jmax + 1, kmax + 1, ftot, nRHS))
        c = ny.eliminateNegativeFourier(c, 2)

        ################################################################################################################
        # Prepare output
        ################################################################################################################
        d['hatc1'] = {}
        d['hatc1']['a'] = {}
        d['hatc1']['ax'] = {}
        for i, submod in enumerate(self.submodulesToRun):
            if submod == 'sedadv_ax':
                d['hatc1']['ax']['sedadv'] = c[:, :, :, i]
            elif submod == 'erosion':
                d['hatc1']['a']['erosion'] = {}
                for j, subsubmod in enumerate(keysu1):
                    d['hatc1']['a']['erosion'][subsubmod] = c[:, :, :, len(self.submodulesToRun) - 1 + j]
            else:
                d['hatc1']['a'][submod] = c[:, :, :, i]
        if 'sedadv' not in self.submodulesToRun:
            d['hatc1']['ax'] = 0

        return d
    
    def second_order_river(self, d):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2 * fmax + 1
        
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax + 1, kmax + 1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax + 1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax + 1, 1, ftot, 1], dtype=complex)

        # erosion
        if self.input.v('u1', 'river') is not None:
            E = erosion(self.ws, 2, self.input, self.erosion_method, submodule=(None, 'river', None), friction=self.frictionpar)
            Fbed[:, :, fmax:, 0] = -E

            ################################################################################################################
            # Solve equation
            ################################################################################################################
            cmatrix = self.input.v('cMatrix')
            if cmatrix is not None:
                c, cMatrix = cFunction(None, cmatrix, F, Fsurf, Fbed, self.input, hasMatrix=True)
            else:
                c, cMatrix = cFunction(self.ws, self.Kv, F, Fsurf, Fbed, self.input, hasMatrix=False)
            c = c.reshape((jmax + 1, kmax + 1, ftot))
        else:
            c = np.zeros((jmax + 1, kmax + 1, ftot))

        d['hatc2'] = {}
        d['hatc2']['a'] = {}
        d['hatc2']['a']['erosion'] = {}
        d['hatc2']['a']['erosion']['river_river'] = ny.eliminateNegativeFourier(c, 2)
        return d






