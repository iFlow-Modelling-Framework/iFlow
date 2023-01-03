"""

Date: 13-02-2020
Authors: J. Wang
"""
import logging
import numpy as np
from packages.hydrodynamics2DV.perturbation.HydroFirst import HydroFirst
import nifty as ny
from packages.hydrodynamics2DV.perturbation.util.zetaFunctionUncoupled import zetaFunctionUncoupled
from src.util.diagnostics import KnownError


class HydroFirst_reverse(HydroFirst):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        HydroFirst.__init__(self,input)
        return

    def run(self):
        """Run function to initiate the calculation of the first order water level and velocities

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        # logger.info('Running module HydroFirst_2')

        # Initiate variables
        submodule = self.input.v('submodules')
        
        # compute and save results
        d = dict()
        d['zeta1'] = {}
        d['u1'] = {}

        # Compute results
        zeta, u = self.tide_2()

        d = dict()
        d['zeta1_reverse'] = {}
        d['u1_reverse'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}

        d['zeta1_reverse'] = zeta[0]
        d['__derivative']['x']['zeta1_reverse'] = zeta[1]

        d['u1_reverse'] = u[0]
        return d

    def tide_2(self):
        """Calculates the first order contribution due to the external tide. This contribution only has an M4-component

        Returns:
            zeta - M4 water level due to the external tide
            u    - M4 horizontal velocity due to the external tide
        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        x = self.input.v('grid', 'axis', 'x')
        z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=x).reshape((len(x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]

        OMEGA = self.input.v('OMEGA')
        G = self.input.v('G')
        H = self.input.v('H', x=x).reshape(len(x), 1) + self.input.v('R', x=x).reshape(len(x), 1)        # YMD added reference level 15-08-17
        B = self.input.v('B', x=x).reshape(len(x), 1)
        Av0 = self.input.v('Av', x=x, z=0, f=0).reshape(len(x), 1)
        sf = self.input.v('Roughness', x=x, z=0, f=0).reshape(len(x), 1)

        r = np.sqrt(2. * 1j * OMEGA / Av0).reshape(len(x), 1)
        alpha = (sf / (r * Av0 * np.sinh(r * H) + sf * np.cosh(r * H))).reshape(len(x), 1)
        M = ((alpha * np.sinh(r * H) / r) - H) * (G / (2 * 1j * OMEGA)) * B
        bca = ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (3,))[2]

        # Initiate variables zeta and u
        zeta = np.zeros((3, len(x), 1, 3), dtype=complex)
        u = np.zeros((1, len(x), len(z), 3), dtype=complex)

        # Calculate M4 contribution

        F = np.zeros((jmax + 1, 1), dtype=complex)  # Forcing term shape (x, number of right-hand sides)
        Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
        Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
        # Fopen[0, 0] = bca
        Fclosed[0, 0] = bca

        Z, Zx, _ = zetaFunctionUncoupled(2, M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False, reverseBC = True)
        zeta[0, :, 0, 2] = Z[:, 0]
        zeta[1, :, 0, 2] = Zx[:, 0]

        # Calculate the velocity
        u[0, :, :, 2] = (-(G / (2. * 1j * OMEGA)) * zeta[1, :, 0, 2].reshape(len(x), 1) *
                         (1 - alpha * np.cosh(r * zarr)))
        return zeta, u
