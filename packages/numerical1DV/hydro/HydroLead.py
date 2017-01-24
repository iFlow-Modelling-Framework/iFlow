"""



Date: 24-04-15
Authors: Y.M. Dijkstra
"""

import logging
import numpy as np
import nifty as ny
import nifty.functionTemplates
from scipy.linalg import solve
import uFunctionMomentumConservative
import uFunction


class HydroLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        """

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        #for qq in range(0,6):
        self.logger.info('Running module HydroLead')

        # Init
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1

        ################################################################################################################
        # velocity as function of water level
        ################################################################################################################
        # build, save and solve the velocity matrices in every water column
        Av = self.input.v('Av', range(0, kmax+1), range(0, fmax+1))
        F = np.zeros([kmax+1, ftot, 0])
        Fsurf = np.zeros([1, ftot, 0])
        Fbed = np.zeros([1, ftot, 0])

        uCoef, _, uzCoef, _, velocityMatrix = uFunctionMomentumConservative.uFunction(Av, F, Fsurf, Fbed, self.input)
        #uCoef, _, uzCoef, _, velocityMatrix = uFunction.uFunction(Av, F, Fsurf, Fbed, self.input)

        ################################################################################################################
        # velocity
        ################################################################################################################
        uCoefDA = ny.integrate(uCoef.reshape(kmax+1, 1, 2*fmax+1, 2*fmax+1), 0, kmax, 0, self.input.slice('grid'))/self.input.v('H')
        u = np.zeros((kmax+1, ftot, len(self.submodulesToRun)), dtype=complex)
        uz = np.zeros((kmax+1, ftot, len(self.submodulesToRun)), dtype=complex)
        U0 = np.zeros(ftot, dtype=complex)
        if 'tide' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('tide')
            U0[fmax:] = ny.amp_phase_input(self.input.v('U0'), self.input.v('phase0'), (fmax+1,))
            uCoefDA[0,0,fmax,fmax]=np.real(uCoefDA[0,0,fmax,fmax])
            factors = solve(uCoefDA.reshape(ftot, ftot), U0)
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors)
            uz[:, :, submodindex] = np.dot(uzCoef, factors)

        if 'river' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('river')
            U0 = -np.asarray([0]*fmax+ny.toList(self.input.v('q0')/self.input.v('H'))+[0]*fmax)
            factors = solve(uCoefDA.reshape(ftot, ftot), U0.reshape(ftot))
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors)
            uz[:, :, submodindex] = np.dot(uzCoef, factors)

        u = ny.eliminateNegativeFourier(u, 1)
        uz = ny.eliminateNegativeFourier(uz, 1)
        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        # import matplotlib.pyplot as plt
        #
        # plt.plot(np.abs(u[:, 1, 0]))
        # plt.show()

        d = {}
        d['velocityMatrix'] = velocityMatrix
        d['u0'] = {}
        for i, submod in enumerate(self.submodulesToRun):
            nfu = nifty.functionTemplates.NumericalFunctionWrapper(u[:, :, i], self.input.slice('grid'))
            nfu.addDerivative(uz[:, :, i], 'z')
            d['u0'][submod] = nfu.function

        return d



