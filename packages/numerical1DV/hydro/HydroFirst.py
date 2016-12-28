"""


Date: 29-07-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from scipy.linalg import solve
import uFunctionMomentumConservative
import uFunction


class HydroFirst:
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
        self.logger.info('Running module HydroFirst')

        # Init
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        BETA = self.input.v('BETA')
        ftot = 2*fmax+1

        ################################################################################################################
        # velocity as function of water level
        ################################################################################################################
        ## LHS terms
        #   try to get velocityMatrix from the input. If it is not found, proceed to calculate the matrix again
        A = self.input.v('velocityMatrix')  # known that this is numeric data, ask without arguments to retrieve full ndarray
        velocityMatrix = True
        if A is None:
            A = self.input.v('Av', range(0, kmax+1), range(0, fmax+1))
            velocityMatrix = False
        else:
            A = A

        ## RHS terms
        #   Determine number/names of right hand side
        submodulesVelocityForcing = [i for i in ['baroc', 'mixing'] if i in self.submodulesToRun]
        submodulesVelocityConversion = [i for i, mod in enumerate(self.submodulesToRun) if mod in ['baroc', 'mixing']]
        nRHS = len(submodulesVelocityForcing)
        F = np.zeros([kmax+1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([1, ftot, nRHS], dtype=complex)
        uFirst = np.zeros([kmax+1, ftot, len(self.submodulesToRun)], dtype=complex)
        uzFirst = np.zeros([kmax+1, ftot, len(self.submodulesToRun)], dtype=complex)

        if 'baroc' in submodulesVelocityForcing:
            sx = np.zeros((kmax+1, fmax+1))
            sx[:, 0] = self.input.v('sx', range(0, kmax+1))
            Jsx = -ny.integrate(sx, 'z', 0, range(0,kmax+1), self.input.slice('grid'))  # integral from z to 0 has its boundaries inverted and has a minus sign to compensate
            Jsx = np.concatenate((np.zeros([kmax+1, fmax]), Jsx), 1)
            F[:, :, submodulesVelocityForcing.index('baroc')] = -G*BETA*Jsx
        if 'mixing' in submodulesVelocityForcing:
            u0z = self.input.d('u0', range(0, kmax+1), range(0, fmax+1), dim='z')
            Av1 = self.input.v('Av1', range(0, kmax+1), range(0, fmax+1))
            ksi = ny.complexAmplitudeProduct(u0z, Av1, 1)
            ksiz = ny.derivative(ksi, 'z', self.input.slice('grid'))
            ksi = np.concatenate((np.zeros([kmax+1, fmax]), ksi), 1)
            ksiz = np.concatenate((np.zeros([kmax+1, fmax]), ksiz), 1)
            F[:, :, submodulesVelocityForcing.index('mixing')] = ksiz
            Fsurf[:, :, submodulesVelocityForcing.index('mixing')] = -ksi[[0], Ellipsis]
            if self.input.v('BottomBC') in ['PartialSlip', 'QuadraticSlip']:
                Fbed[:, :, submodulesVelocityForcing.index('mixing')] = -ksi[[kmax], Ellipsis]

        ## Solve equation
        uCoef, uFirst[:, :, submodulesVelocityConversion], uzCoef, uzFirst[:, :, submodulesVelocityConversion], _ = uFunctionMomentumConservative.uFunction(A, F, Fsurf, Fbed, self.input, hasMatrix=velocityMatrix)
        #uCoef, uFirst[:, :, submodulesVelocityConversion], uzCoef, uzFirst[:, :, submodulesVelocityConversion], _ = uFunction.uFunction(A, F, Fsurf, Fbed, self.input, hasMatrix=velocityMatrix)

        ################################################################################################################
        # velocity
        ################################################################################################################
        uCoefDA = ny.integrate(uCoef.reshape(kmax+1, 1, 2*fmax+1, 2*fmax+1), 0, kmax, 0, self.input.slice('grid'))/self.input.v('H')
        uFirstDA = ny.integrate(uFirst.reshape((kmax+1, 1, 2*fmax+1, len(self.submodulesToRun))), 0, kmax, 0, self.input.slice('grid'))/self.input.v('H')
        u = np.zeros((kmax+1, ftot, len(self.submodulesToRun)), dtype=complex)
        uz = np.zeros((kmax+1, ftot, len(self.submodulesToRun)), dtype=complex)
        U0 = np.zeros(ftot, dtype=complex)
        if 'tide' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('tide')
            U0[fmax:] = ny.amp_phase_input(self.input.v('U1'), self.input.v('phase1'), (fmax+1,))
            factors = solve(uCoefDA.reshape(ftot, ftot), U0)
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors)
            uz[:, :, submodindex] = np.dot(uzCoef, factors)

        if 'river' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('river')
            U0 = -np.asarray([0]*fmax+ny.toList(self.input.v('q1')/self.input.v('H'))+[0]*fmax)
            factors = solve(uCoefDA.reshape(ftot, ftot), U0.reshape(ftot))
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors)
            uz[:, :, submodindex] = np.dot(uzCoef, factors)

        if 'baroc' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('baroc')
            factors = solve(uCoefDA.reshape(ftot, ftot), -uFirstDA[0, 0, :, submodindex])
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors) + uFirst[:, :, submodindex]
            uz[:, :, submodindex] = np.dot(uzCoef, factors) + uzFirst[:, :, submodindex]

        if 'mixing' in self.submodulesToRun:
            submodindex = self.submodulesToRun.index('mixing')
            factors = solve(uCoefDA.reshape(ftot, ftot), -uFirstDA[0, 0, :, submodindex])
            factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1)]))    # add inverted complex conjugate to eliminate complex part of residual

            u[:, :, submodindex] = np.dot(uCoef, factors) + uFirst[:, :, submodindex]
            uz[:, :, submodindex] = np.dot(uzCoef, factors) + uzFirst[:, :, submodindex]


        u = ny.eliminateNegativeFourier(u, 1)
        uz = ny.eliminateNegativeFourier(uz, 1)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['u1'] = {}
        for i, submod in enumerate(self.submodulesToRun):
            nfu = ny.functionTemplates.NumericalFunctionWrapper(u[:, :, i], self.input.slice('grid'))
            nfu.addDerivative(uz[:, :, i], 'z')
            d['u1'][submod] = nfu.function

        return d

