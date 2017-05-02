"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
from erosion import erosion


class SedDynamicFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SedDynamic - first order')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        OMEGA = self.input.v('OMEGA')
        self.submodulesToRun = ny.toList(self.input.v('submodules'))
        method = self.input.v('erosion_formulation')

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        # PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])  # assume it is constant in time; else division with AV fails
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # Kv = Av/PrSchm
        Kv = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        ws = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        if 'sedadv' in self.submodulesToRun and 'sedadv_ax' not in self.submodulesToRun:
            self.submodulesToRun.append('sedadv_ax')
        # determine number of submodules
        nRHS = len(self.submodulesToRun)
        if 'erosion' in self.submodulesToRun:
            keysu1 = self.input.getKeysOf('u1')
            self.submodulesToRun.remove('erosion')
            self.submodulesToRun.append('erosion')
            nRHS = nRHS - 1 + len(self.input.getKeysOf('u1'))

        F = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)

        c0 = self.input.v('hatc0')
        cx0 = self.input.d('hatc0', dim='x')
        cz0 = self.input.d('hatc0', dim='z')
        # 1. Erosion
        if 'erosion' in self.submodulesToRun:
            # erosion due to first-order bed shear stress
            # E = erosion(ws, Av, 1, self.input, method)                        # 24-04-2017 Obsolete
            # Fbed[:, :, fmax:, self.submodulesToRun.index('erosion')] = -E     # 24-04-2017 Obsolete
            for submod in keysu1:
                E = erosion(ws, Av, 1, self.input, method, submodule=(None, submod))
                Fbed[:, :, fmax:, len(self.submodulesToRun)-1 + keysu1.index(submod)] = -E

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
            E = erosion(ws1, Av, 0, self.input, method)
            Fbed[:, :, fmax:, self.submodulesToRun.index('settling')] = -E

        # 4. First-order eddy diffusivity
        if 'mixing' in self.submodulesToRun:
            # surface, bed and internal terms
            Av1 = self.input.v('Av1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            Kv1 = self.input.v('Kv1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            psi = ny.complexAmplitudeProduct(Kv1, cz0, 2)
            psiz = ny.derivative(psi, 'z', self.input.slice('grid'))

            F[:, :, fmax:, self.submodulesToRun.index('mixing')] = psiz
            Fsurf[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, 0, :]
            Fbed[:, 0, fmax:, self.submodulesToRun.index('mixing')] = -psi[:, -1, :]

            # adjustment to erosion
            E = erosion(ws, Av1, 0, self.input, method)
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
            elif submod == 'erosion':
                d['hatc1']['a']['erosion'] = {}
                for j, subsubmod in enumerate(keysu1):
                    d['hatc1']['a']['erosion'][subsubmod] = c[:, :, :, len(self.submodulesToRun)-1+j]
            else:
                d['hatc1']['a'][submod] = c[:, :, :, i]
        if 'sedadv' not in self.submodulesToRun:
            d['hatc1']['ax'] = 0

        return d

    
    
    