"""
SalinityLead

Date: 11-Jan-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from svarFunction import svarFunction
from sclosureFunction import sclosureFunction


class SalinityLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        self.logger.info('Running module SalinityLead')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        SIGMASAL = self.input.v('SIGMASAL')

        ################################################################################################################
        # First-order salinity variation as function of leading-order salinity closure
        ################################################################################################################
        # build, save and solve the velocity matrices in every water column
        Kv = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))/SIGMASAL
        F = np.zeros([jmax+1, kmax+1, fmax+1, 0])
        Fsurf = np.zeros([jmax+1, 1, fmax+1, 0])
        Fbed = np.zeros([jmax+1, 1, fmax+1, 0])

        sCoef, _, szCoef, _, salinityMatrix = svarFunction(Kv, F, Fsurf, Fbed, self.input)

        ################################################################################################################
        # Leading-order salinity closure
        ################################################################################################################
        ## LHS terms
        #   First-order river discharge
        Q = self.input.v('Q1', range(0, jmax+1))
        #   First-order river discharge or zero if not available
        # u1riv = self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax+1))
        # B = self.input.v('B', range(0, jmax+1))
        # if u1riv is None:
        #     Q = np.zeros((jmax+1))
        #     self.logger.warning('No first-order river discharge found in module SalinityLead')
        # else:
        #     Q = ny.integrate(u1riv, 'z', kmax, 0, self.input.slice('grid'))
        #     Q = Q*B
        # del u1riv

        #   Diffusion coefficient
        #   part 1) diffusive transport
        H = self.input.v('H', range(0, jmax+1))
        B = self.input.v('B', range(0, jmax+1))
        Kh = self.input.v('Kh', range(0, jmax+1))
        AK = np.real(H*B*Kh)

        #   part 2) advective transport
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        us = ny.complexAmplitudeProduct(u0, sCoef, 2)[:, :, 0, 0]       # subtidal part of u*s
        us = ny.integrate(us, 'z', kmax, 0, self.input.slice('grid')).reshape(jmax+1)
        AK += -np.real(B*us)

        del us, u0, H, B

        ## RHS terms
        F = np.zeros([jmax+1, 1])   # dimensions (jmax+1, NumberOfForcings)

        #   open BC: effect of the sea
        Fopen = np.zeros([1, 1])
        Fopen[0, 0] = self.input.v('ssea')

        # closed BC: always zero. This is by assumption that the salinity vanishes for x=L
        Fclosed = np.zeros([1, 1])

        ## Solve equation
        S0 = np.zeros((jmax+1, 1, fmax+1, 1))
        Sx0 = np.zeros((jmax+1, 1, fmax+1, 1))
        S0[:, 0, 0, :], Sx0[:, 0, 0, :] = sclosureFunction((Q, AK), F, Fopen, Fclosed, self.input)

        ################################################################################################################
        # First-order salinity variation
        ################################################################################################################
        s1 = ny.complexAmplitudeProduct(sCoef, Sx0, 2)
        sz1 = ny.complexAmplitudeProduct(szCoef, Sx0, 2)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['salinityMatrix'] = salinityMatrix
        nf = ny.functionTemplates.NumericalFunctionWrapper(S0[:, :, :, 0], self.input.slice('grid'))
        nf.addDerivative(Sx0[:, :, :, 0], 'x')
        d['s0'] = nf.function

        nf2 = ny.functionTemplates.NumericalFunctionWrapper(s1[:, :, :, 0], self.input.slice('grid'))
        nf2.addDerivative(sz1[:, :, :, 0], 'z')
        d['s1var'] = nf2.function
        return d