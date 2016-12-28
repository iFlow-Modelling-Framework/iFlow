"""


Date: 29-07-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from scipy.linalg import solve
import uFunctionMomentumConservative


class HydroHigher:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        maxOrder = int(self.input.v('maxOrder'))
        self.currentOrder = iteration+2
        if self.currentOrder <= maxOrder:
            stop = False
        else:
            stop = True
        return stop

    def run_init(self):
        self.stopping_criterion(0)
        return self.run()

    def run(self):
        """

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        self.logger.info('Running module HydroHigher - order '+str(self.currentOrder))

        # Init
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
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
        ustr = 'u'+str(self.currentOrder-1)
        noRHS = len(self.input.getKeysOf(ustr))

        F = np.zeros([kmax+1, ftot, noRHS], dtype=complex)
        Fsurf = np.zeros([1, ftot, noRHS], dtype=complex)
        Fbed = np.zeros([1, ftot, noRHS], dtype=complex)
        fnames = []

        for i, submod in enumerate(self.input.getKeysOf(ustr)):
            fnames.append('mixing-'+submod.split('-')[-1])
            u0z = self.input.d(ustr, submod, range(0, kmax+1), range(0, fmax+1), dim='z')
            Av1 = self.input.v('Av1', range(0, kmax+1), range(0, fmax+1))
            ksi = ny.complexAmplitudeProduct(u0z, Av1, 1)
            ksiz = ny.derivative(ksi, 'z', self.input.slice('grid'))
            ksi = np.concatenate((np.zeros([kmax+1, fmax]), ksi), 1)
            ksiz = np.concatenate((np.zeros([kmax+1, fmax]), ksiz), 1)
            F[:, :, i] = ksiz
            Fsurf[:, :, i] = -ksi[[0], Ellipsis]
            if self.input.v('BottomBC') in ['PartialSlip', 'QuadraticSlip']:
                Fbed[:, :, i] = -ksi[[kmax], Ellipsis]

        ## Solve equation
        uCoef, uFirst, uzCoef, uzFirst, _ = uFunctionMomentumConservative.uFunction(A, F, Fsurf, Fbed, self.input, hasMatrix=velocityMatrix)

        ################################################################################################################
        # velocity
        ################################################################################################################
        uCoefDA = ny.integrate(uCoef.reshape(kmax+1, 1, 2*fmax+1, 2*fmax+1), 0, kmax, 0, self.input.slice('grid'))/self.input.v('H')
        uFirstDA = ny.integrate(uFirst.reshape((kmax+1, 1, 2*fmax+1, noRHS)), 0, kmax, 0, self.input.slice('grid'))/self.input.v('H')
        u = np.zeros((kmax+1, ftot, noRHS), dtype=complex)
        uz = np.zeros((kmax+1, ftot, noRHS), dtype=complex)

        factors = solve(uCoefDA.reshape(ftot, ftot), -uFirstDA[0, 0, :, :])
        factors = 0.5*(factors + np.conj(factors[np.arange(ftot-1,-1, -1), :]))    # add inverted complex conjugate to eliminate complex part of residual

        u[:, :, :] = np.dot(uCoef, factors) + uFirst
        uz[:, :, :] = np.dot(uzCoef, factors) + uzFirst


        u = ny.eliminateNegativeFourier(u, 1)
        uz = ny.eliminateNegativeFourier(uz, 1)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['u'+str(self.currentOrder)] = {}
        for i in range(0, noRHS):
            nfu = ny.functionTemplates.NumericalFunctionWrapper(u[:, :, i], self.input.slice('grid'))
            nfu.addDerivative(uz[:, :, i], 'z')
            d['u'+str(self.currentOrder)][fnames[i]] = nfu.function

        return d

