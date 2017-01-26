"""
SalinitySecond

Date: 15-01-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from svarFunction import svarFunction
from sclosureFunction import sclosureFunction


class SalinitySecond:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SalinitySecond')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        SIGMASAL = self.input.v('SIGMASAL')
        OMEGA = self.input.v('OMEGA')
        submodulesToRun = self.input.v('submodules')

        sx0 = self.input.d('s0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        sx1 = self.input.d('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x') + self.input.d('s1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        s1var = self.input.v('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        zeta1 = self.input.v('zeta1', range(0, jmax+1), [0], range(0, fmax+1))
        H = self.input.v('H', range(0, jmax+1))
        B = self.input.v('B', range(0, jmax+1))
        AKh = self.input.v('Kh', range(0, jmax+1))*B*H

        ################################################################################################################
        # Third-order salinity variation as function of second-order salinity closure
        ################################################################################################################
        # build, save and solve the velocity matrices in every water column
        #   LHS
        Kv = self.input.v('salinityMatrix')
        hasMatrix = True
        if Kv is None:
            Kv = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))/SIGMASAL
            hasMatrix = False

        #   RHS
        #   Allow for three submodules: advection, diffusion and nostress
        nRHS = len(self.input.getKeysOf('u1'))+len(self.input.getKeysOf('u2'))+len(self.input.getKeysOf('s2var'))+3
        f_index = -1
        f_names = []

        F = np.zeros([jmax+1, kmax+1, fmax+1, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, fmax+1, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, fmax+1, nRHS], dtype=complex)
        if 'advection' in submodulesToRun:
            # advection by u0*sx2
            w0 = self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            for mod in self.input.getKeysOf('s2var'):
                sx2var = self.input.d('s2var', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
                sz2var = self.input.d('s2var', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')

                f_index += 1
                f_names.append(['advection', 'u0-s2_'+mod])
                F[:, :, :, f_index] = -ny.complexAmplitudeProduct(u0, sx2var, 2) - ny.complexAmplitudeProduct(w0, sz2var, 2)
                del sx2var, sz2var
            del w0

            # advection by u1*sx1
            sz1var = self.input.d('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            for mod in self.input.getKeysOf('u1'):
                u1 = self.input.v('u1', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                w1 = self.input.v('w1', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

                f_index += 1
                f_names.append(['advection', 'u1_'+mod+'-s1'])
                F[:, :, :, f_index] = -ny.complexAmplitudeProduct(u1, sx1, 2) - ny.complexAmplitudeProduct(w1, sz1var, 2)
                del u1, w1
            del sz1var

            # advection by u2*sx0
            submods = self.input.getKeysOf('u2')
            for mod in submods:
                u2 = self.input.v('u2', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

                f_index += 1
                f_names.append(['advection', 'u2_'+mod+'-s0'])
                F[:, :, :, f_index] = -ny.complexAmplitudeProduct(u2, sx0, 2)
                del u2

        if 'diffusion' in submodulesToRun:
            f_index += 1
            f_names.append(['diffusion', 's1'])
            F[:, :, :, f_index] = (ny.derivative(AKh.reshape((jmax+1, 1, 1))*sx1, 'x', self.input.slice('grid')))/((B*H).reshape((jmax+1, 1, 1)))

        if 'nostress' in submodulesToRun:
            D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
            # no-stress by s1-zeta1
            Kvsz1z = D*s1var[:, [0], :] + ny.complexAmplitudeProduct(u0[:, [0], :], sx0[:, [0], :], 2)

            f_index += 1
            f_names.append(['nostress', 's1-zeta1'])
            Fsurf[:, 0, :, f_index] = -ny.complexAmplitudeProduct(Kvsz1z, zeta1, 2).reshape((jmax+1, fmax+1))

            # no-stress by s2-zeta0
            u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            s2var = self.input.v('s2var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            Kvsz2z = D*s2var[:, [0], :] + ny.complexAmplitudeProduct(u0[:, [0], :], sx1[:, [0], :], 2) \
                                        + ny.complexAmplitudeProduct(u1[:, [0], :], sx0[:, [0], :], 2) \
                                        - (ny.derivative(AKh.reshape((jmax+1, 1, 1))*sx0[:, [0], :], 'x', self.input.slice('grid')))/((B*H).reshape((jmax+1, 1, 1)))

            f_index += 1
            f_names.append(['nostress', 's2-zeta0'])
            Fsurf[:, 0, :, f_index] = -ny.complexAmplitudeProduct(Kvsz2z, zeta0, 2).reshape((jmax+1, fmax+1))

            del D, Kvsz1z, Kvsz2z

        sCoef, sForced, szCoef, szForced, salinityMatrix = svarFunction(Kv, F, Fsurf, Fbed, self.input, hasMatrix=hasMatrix)
        del Kv

        ################################################################################################################
        # Second-order salinity closure
        ################################################################################################################
        ## LHS terms
        #   First-order river discharge
        Q = self.input.v('Q1', range(0, jmax+1))

        #   Diffusion coefficient
        us = ny.complexAmplitudeProduct(u0, sCoef, 2)[:, :, 0, 0]       # subtidal part of u*s
        us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)
        AK = np.real(AKh) - np.real(B*us)
        del us

        ## RHS terms
        nRHS_clo = nRHS+len(self.input.getKeysOf('u1'))*len(self.input.getKeysOf('s2var'))+len(self.input.getKeysOf('u2'))+4
        f_index_clo = -1
        f_names_clo = []

        F = np.zeros([jmax+1, nRHS_clo])
        Fopen = np.zeros([1, nRHS_clo])
        Fclosed = np.zeros([1, nRHS_clo])
        if 'advection' in submodulesToRun:
            # advection by u0*s3
            us = ny.complexAmplitudeProduct(u0, sForced, 2)[:, :, [0], :]
            us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1, nRHS)
            for i in range(0, nRHS):
                f_index_clo += 1
                f_names_clo.append(['advection', 'u0-s3_'+f_names[i][0]])
                F[:, f_index_clo] = -ny.derivative(np.real(B*us[:, i]), 'x', self.input.slice('grid'))

            # advection by u1*s2
            submods_u = self.input.getKeysOf('u1')
            submods_s = self.input.getKeysOf('s2var')
            for mod_u in submods_u:
                u1 = self.input.v('u1', mod_u, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                for mod_s in submods_s:
                    s2var = self.input.v('s2var', mod_s, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                    us = ny.complexAmplitudeProduct(u1, s2var, 2)[:, :, 0]
                    us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)

                    f_index_clo += 1
                    f_names_clo.append(['advection', 'u1_'+mod_u+'-s2'+mod_s])
                    F[:, f_index_clo] = -ny.derivative(np.real(B*us), 'x', self.input.slice('grid'))
                    del s2var
                del u1

            # advection by u2*s1
            submods = self.input.getKeysOf('u2')
            for mod in submods:
                u2 = self.input.v('u2', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                us = ny.complexAmplitudeProduct(u2, s1var, 2)[:, :, 0]
                us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)

                f_index_clo += 1
                f_names_clo.append(['advection', 'u2_'+mod+'-s1'])
                F[:, f_index_clo] = -ny.derivative(np.real(B*us), 'x', self.input.slice('grid'))
                del u2

            # surface term
            s2var = self.input.v('s2var', range(0, jmax+1), [0], range(0, fmax+1))
            u1 = self.input.v('u1', range(0, jmax+1), [0], range(0, fmax+1))
            us = ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], s1var[:, [0], :], 2), zeta1, 2)[:, 0, 0]
            us += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], s2var[:, [0], :], 2), zeta0, 2)[:, 0, 0]
            us += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u1[:, [0], :], s1var[:, [0], :], 2), zeta0, 2)[:, 0, 0]

            f_index_clo += 1
            f_names_clo.append(['advection', 'surface'])
            F[:, f_index_clo] = -ny.derivative(np.real(B*us), 'x', self.input.slice('grid'))
            del us

        if 'diffusion' in submodulesToRun:
            # Bed terms
            Hx = self.input.d('H', range(0, jmax+1), dim='x')
            s2var = self.input.v('s2var', range(0, jmax+1), kmax, 0)
            sx2var = self.input.d('s2var', range(0, jmax+1), kmax, 0, dim='x')

            f_index_clo += 1
            f_names_clo.append(['diffusion', 'bedslope'])
            F[:, f_index_clo] = - ny.derivative(np.real(AKh*s2var*Hx), 'x', self.input.slice('grid'))/H \
                                - np.real(AKh/H*sx2var*Hx)

            # Surface term
            f_index_clo += 1
            f_names_clo.append(['diffusion', 'surface'])
            F[:, f_index_clo] = np.real(zeta1[:, 0, 0])*ny.derivative(np.real(AKh*sx0[:, 0, 0]), 'x', self.input.slice('grid'))/H
            F[:, f_index_clo] += np.real(ny.complexAmplitudeProduct(zeta0, ny.derivative((AKh).reshape((jmax+1, 1, 1))*sx1[:, [0], :], 'x', self.input.slice('grid')), 2)[:, 0, 0]/H)
            del s2var, sx2var, Hx

        if 'sea' in submodulesToRun:
            f_index_clo += 1
            f_names_clo.append(['sea', ''])
            Fopen[0, f_index_clo] = -np.real(ny.complexAmplitudeProduct(s1var[0, 0, :], zeta0[0, 0, :], 0)[0]/H[0])

        ## Solve equation
        S2 = np.zeros((jmax+1, 1, fmax+1, nRHS_clo))
        Sx2 = np.zeros((jmax+1, 1, fmax+1, nRHS_clo))
        S2[:, 0, 0, :], Sx2[:, 0, 0, :] = sclosureFunction((Q, AK), F, Fopen, Fclosed, self.input)

        ################################################################################################################
        # Third-order salinity variation
        ################################################################################################################
        s3 = np.zeros((jmax+1, kmax+1, fmax+1, nRHS+1), dtype=complex)
        sz3 = np.zeros((jmax+1, kmax+1, fmax+1, nRHS+1), dtype=complex)

        s3[:, :, :, :-1] = sForced
        sz3[:, :, :, :-1] = szForced
        f_names.append(['advection', 'u0-S2'])
        s3[:, :, :, -1] = np.sum(ny.complexAmplitudeProduct(sCoef, Sx2, 2), 3)
        sz3[:, :, :, -1] = np.sum(ny.complexAmplitudeProduct(szCoef, Sx2, 2), 3)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['salinityMatrix'] = salinityMatrix

        d['s2'] = {}
        d['s3var'] = {}
        for submod in submodulesToRun:
            if submod in zip(*f_names_clo)[0]:
                d['s2'][submod] = {}
            if submod in zip(*f_names)[0]:
                d['s3var'][submod] = {}

        for i, mod in enumerate(f_names_clo):
            nf = ny.functionTemplates.NumericalFunctionWrapper(S2[:, :, :, i], self.input.slice('grid'))
            nf.addDerivative(Sx2[:, :, :, i], 'x')
            if len(mod) == 1:
                d['s2'][mod[0]] = nf.function
            if len(mod) == 2:
                d['s2'][mod[0]][mod[1]] = nf.function

        for i, mod in enumerate(f_names):
            nf2 = ny.functionTemplates.NumericalFunctionWrapper(s3[:, :, :, i], self.input.slice('grid'))
            nf2.addDerivative(sz3[:, :, :, i], 'z')
            if len(mod) == 1:
                d['s3var'][mod[0]] = nf2.function
            if len(mod) == 2:
                d['s3var'][mod[0]][mod[1]] = nf2.function

        return d



