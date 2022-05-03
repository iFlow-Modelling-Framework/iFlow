"""

Date: 13-02-2020
Authors: J. Wang
"""
import logging
import numpy as np
from packages.semi_analytical2DV.hydro.HydroFirst import HydroFirst
from scipy import integrate
from nifty.functionTemplates.NumericalFunctionWrapper import NumericalFunctionWrapper
import nifty as ny
from packages.semi_analytical2DV.hydro.zetaFunctionUncoupled import zetaFunctionUncoupled
from src.util.diagnostics import KnownError


class HydroFirst_2(HydroFirst):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the first order water level and velocities

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        # self.logger.info('Running module HydroFirst_2')

        # Initiate variables
        self.submodule = self.input.v('submodules')
        
        self.OMEGA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.BETA = self.input.v('BETA')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        self.dx = self.x[1:]-self.x[:-1]
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=self.x/self.L).reshape((len(self.x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]
        
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=0).reshape(len(self.x), 1)
        self.Av0x = self.input.d('Av', x=self.x/self.L, z=0, f=0, dim='x').reshape(len(self.x), 1)
        
        self.sf = self.input.v('Roughness', x=self.x/self.L, z=0, f=0).reshape(len(self.x), 1)
        self.sfx = self.input.d('Roughness', x=self.x/self.L, z=0, f=0, dim='x').reshape(len(self.x), 1)
        
        self.r = np.sqrt(2. * 1j * self.OMEGA / self.Av0).reshape(len(self.x), 1)
        self.H = self.input.v('H', x=self.x/self.L).reshape(len(self.x), 1) + self.input.v('R', x=self.x/self.L).reshape(len(self.x), 1)        # YMD added reference level 15-08-17
        self.alpha = (self.sf / (self.r * self.Av0 * np.sinh(self.r * self.H) + self.sf * np.cosh(self.r * self.H))).reshape(len(self.x), 1)
        
        self.B = self.input.v('B', x=self.x/self.L).reshape(len(self.x), 1)
        self.Bx = self.input.d('B', x=self.x/self.L, dim='x').reshape(len(self.x), 1).reshape(len(self.x), 1)
        self.u0 = self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1)
        self.u0x = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='x')
        self.u0z = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z')
        self.u0zz = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='zz')

        self.M = ((self.alpha * np.sinh(self.r * self.H) / self.r) - self.H) * (self.G / (2 * 1j * self.OMEGA)) * self.B
        self.bca = ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (3,))[2]

        self.__bc = np.zeros(4)
        self.__F = []

        # compute and save results
        d = dict()
        d['zeta1'] = {}
        d['u1'] = {}

        for mod in self.submodule:
            # Compute results
            zeta, u = getattr(self, mod)()

            # save in dictionary
            nfz = NumericalFunctionWrapper(zeta[0], self.input.slice('grid'))
            nfz.addDerivative(zeta[1], dim='x')
            nfu = NumericalFunctionWrapper(u[0], self.input.slice('grid'))

            d['zeta1'][mod] = nfz.function
            d['u1'][mod] = nfu.function
 

        self.input.merge(d)
        return d

   

    def tide_2(self):
        """Calculates the first order contribution due to the external tide. This contribution only has an M4-component

        Returns:
            zeta - M4 water level due to the external tide
            u    - M4 horizontal velocity due to the external tide
        """
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # Calculate M4 contribution
        jmax = self.input.v('grid', 'maxIndex', 'x')
        F = np.zeros((jmax + 1, 1), dtype=complex)  # Forcing term shape (x, number of right-hand sides)
        Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
        Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
        # Fopen[0, 0] = self.bca
        Fclosed[0, 0] = self.bca

        Z, Zx, _ = zetaFunctionUncoupled(2, self.M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False, reverseBC = True)
        zeta[0, :, 0, 2] = Z[:, 0]
        zeta[1, :, 0, 2] = Zx[:, 0]

        # Calculate the velocity
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.OMEGA)) * zeta[1, :, 0, 2].reshape(len(self.x), 1) *
                         (1 - self.alpha * np.cosh(self.r * self.zarr)))
        return zeta, u
