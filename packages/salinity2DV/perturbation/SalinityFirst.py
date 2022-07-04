"""
SalinityFirst

Original date: 14-01-16
Update: 15-02-2022
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from .svarFunction import svarFunction
from .sclosureFunction import sclosureFunction


class SalinityFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SalinityFirst')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        SIGMASAL = self.input.v('SIGMASAL')
        OMEGA = self.input.v('OMEGA')
        submodulesToRun = self.input.v('submodules')

        sx0 = self.input.d('s0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        s1var = self.input.v('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        H = self.input.v('H', range(0, jmax+1))
        B = self.input.v('B', range(0, jmax+1))
        AKh = self.input.v('Kh', range(0, jmax+1))*B*H

        ################################################################################################################
        # Second-order salinity variation as function of first-order salinity closure
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
        nRHS = len(self.input.getKeysOf('u1'))+3
        f_index = -1
        f_names = []

        F = np.zeros([jmax+1, kmax+1, fmax+1, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, fmax+1, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, fmax+1, nRHS], dtype=complex)
        if 'advection' in submodulesToRun:
            # advection by u0*sx1
            sx1var = self.input.d('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
            sz1var = self.input.d('s1var', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            w0 = self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

            f_index += 1
            f_names.append(['advection', 'u0-s1'])
            F[:, :, :, f_index] = -ny.complexAmplitudeProduct(u0, sx1var, 2) - ny.complexAmplitudeProduct(w0, sz1var, 2)
            del sx1var, sz1var, w0

            # advection by u1*sx0
            submods = self.input.getKeysOf('u1')
            for mod in submods:
                u1 = self.input.v('u1', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                f_index += 1
                f_names.append(['advection', 'u1_'+mod+'-s0'])
                F[:, :, :, f_index] = -ny.complexAmplitudeProduct(u1, sx0, 2)
                del u1

        if 'diffusion' in submodulesToRun:
            f_index += 1
            f_names.append(['diffusion', 's0'])
            F[:, :, 0, f_index] = (ny.derivative(AKh*sx0[:, 0, 0], 'x', self.input.slice('grid'))/(B*H)).reshape((jmax+1, 1))*np.ones((1, kmax+1))

        if 'nostress' in submodulesToRun:
            D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
            Kvsz1z = D*s1var[:, [0], :] + ny.complexAmplitudeProduct(u0[:, [0], :], sx0[:, [0], :], 2)

            f_index += 1
            f_names.append(['nostress', 's1-zeta0'])
            Fsurf[:, 0, :, f_index] = -ny.complexAmplitudeProduct(Kvsz1z, zeta0, 2).reshape((jmax+1, fmax+1))
            del D, Kvsz1z

        sCoef, sForced, szCoef, szForced, salinityMatrix = svarFunction(Kv, F, Fsurf, Fbed, self.input, hasMatrix=hasMatrix)
        del Kv

        ################################################################################################################
        # First-order salinity closure
        ################################################################################################################
        ## LHS terms
        #   First-order river discharge
        Q = -self.input.v('Q1', range(0, jmax+1))

        #   Diffusion coefficient
        us = ny.complexAmplitudeProduct(u0, sCoef, 2)[:, :, 0, 0]       # subtidal part of u*s
        us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)
        AK = np.real(AKh) - np.real(B*us)
        del us

        ## RHS terms
        nRHS_clo = nRHS+len(self.input.getKeysOf('u1'))+3
        f_index_clo = -1
        f_names_clo = []

        F = np.zeros([jmax+1, nRHS_clo])
        Fopen = np.zeros([1, nRHS_clo])
        Fclosed = np.zeros([1, nRHS_clo])
        if 'advection' in submodulesToRun:
            # advection by u0*s2
            us = ny.complexAmplitudeProduct(u0, sForced, 2)[:, :, [0], :]
            us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1, nRHS)
            for i in range(0, nRHS):
                f_index_clo += 1
                f_names_clo.append(['advection', 'u0-s2_'+f_names[i][0]])
                F[:, f_index_clo] = -ny.derivative(np.real(B*us[:, i]), 'x', self.input.slice('grid'))

            # advection by u1*s1
            submods = self.input.getKeysOf('u1')
            for mod in submods:
                u1 = self.input.v('u1', mod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                us = ny.complexAmplitudeProduct(u1, s1var, 2)[:, :, 0]
                us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)

                f_index_clo += 1
                f_names_clo.append(['advection', 'u1_'+mod+'-s1'])
                F[:, f_index_clo] = -ny.derivative(np.real(B*us), 'x', self.input.slice('grid'))
                del u1

            # surface term
            us = ny.complexAmplitudeProduct(u0[:, [0], :], s1var[:, [0], :], 2)
            us = ny.complexAmplitudeProduct(us[:, [0], :], zeta0, 2)[:, 0, 0]

            f_index_clo += 1
            f_names_clo.append(['advection', 'surface'])
            F[:, f_index_clo] = -ny.derivative(np.real(B*us), 'x', self.input.slice('grid'))
            del us

        if 'diffusion' in submodulesToRun:
            # Bed terms
            Hx = self.input.d('H', range(0, jmax+1), dim='x')
            sx1var = self.input.d('s1var', range(0, jmax+1), kmax, 0, dim='x')

            f_index_clo += 1
            f_names_clo.append(['diffusion', 'bedslope'])
            F[:, f_index_clo] = - ny.derivative(np.real(AKh*s1var[:, -1, 0]*Hx), 'x', self.input.slice('grid'))/H \
                                - np.real(AKh/H*sx1var*Hx)

            # Surface term
            f_index_clo += 1
            f_names_clo.append(['diffusion', 'surface'])
            F[:, f_index_clo] = np.real(zeta0[:, 0, 0])*ny.derivative(np.real(AKh*sx0[:, 0, 0]), 'x', self.input.slice('grid'))/H
            del sx1var, Hx

        ## Solve equation
        S1 = np.zeros((jmax+1, 1, fmax+1, nRHS_clo))
        Sx1 = np.zeros((jmax+1, 1, fmax+1, nRHS_clo))
        S1[:, 0, 0, :], Sx1[:, 0, 0, :] = sclosureFunction(Q, AK, F, Fopen, Fclosed, self.input)

        ################################################################################################################
        # Second-order salinity variation
        ################################################################################################################
        s2 = np.zeros((jmax+1, kmax+1, fmax+1, nRHS+1), dtype=complex)
        sz2 = np.zeros((jmax+1, kmax+1, fmax+1, nRHS+1), dtype=complex)

        s2[:, :, :, :-1] = sForced
        sz2[:, :, :, :-1] = szForced
        f_names.append(['advection', 'u0-S1'])
        s2[:, :, :, -1] = np.sum(ny.complexAmplitudeProduct(sCoef, Sx1, 2), 3)
        sz2[:, :, :, -1] = np.sum(ny.complexAmplitudeProduct(szCoef, Sx1, 2), 3)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['salinityMatrix'] = salinityMatrix

        d['s1'] = {}
        d['s2var'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['x']['s1'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['z']['s2var'] = {}
        for submod in submodulesToRun:
            if submod in zip(*f_names_clo)[0]:
                d['s1'][submod] = {}
                d['__derivative']['x']['s1'][submod] = {}
            if submod in zip(*f_names)[0]:
                d['s2var'][submod] = {}
                d['__derivative']['z']['s2var'][submod] = {}

        #    if submod in zip(*f_names)[0]:
        for i, mod in enumerate(f_names_clo):
            if len(mod) == 1:
                d['s1'][mod[0]] = S1[:, :, :, i]
                d['__derivative']['x']['s1'][mod[0]] = Sx1[:, :, :, i]
            if len(mod) == 2:
                d['s1'][mod[0]][mod[1]] = S1[:, :, :, i]
                d['__derivative']['x']['s1'][mod[0]][mod[1]] = Sx1[:, :, :, i]

        for i, mod in enumerate(f_names):
            if len(mod) == 1:
                d['s2var'][mod[0]] = s2[:, :, :, i]
                d['__derivative']['z']['s2var'][mod[0]] = sz2[:, :, :, i]
            if len(mod) == 2:
                d['s2var'][mod[0]][mod[1]] = s2[:, :, :, i]
                d['__derivative']['z']['s2var'][mod[0]][mod[1]] = sz2[:, :, :, i]

        return d



