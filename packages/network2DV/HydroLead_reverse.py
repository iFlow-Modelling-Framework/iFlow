"""
Date: 03-07-2019
Authors: J. Wang, Y.M. Dijkstra
"""
import logging
import numpy as np
from packages.hydrodynamics2DV.perturbation.HydroLead import HydroLead
import nifty as ny
from packages.hydrodynamics2DV.perturbation.util.zetaFunctionUncoupled import zetaFunctionUncoupled
from src.util.diagnostics import KnownError


class HydroLead_reverse(HydroLead):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        HydroLead.__init__(self,input)
        return

    def run(self):
        """Run function to initiate the calculation of the leading order water level and velocities

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        # self.logger.info('Running module HydroLead_2')
        

        # Initiate variables
        self.OMEGA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=self.x/self.L).reshape((len(self.x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]
        self.bca = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (2,))[1]

        # Prepare output
        zeta = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
        zetax = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
        zetaxx = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        # Run computations
        zeta[:, 0, 1], zetax[:, 0, 1], zetaxx[:, 0, 1] = self.waterlevel()
        u, w = self.velocity(zeta[:, 0, 1], zetax[:, 0, 1], zetaxx[:, 0, 1])

        # Save water level results
        d = dict()
        d['zeta0_reverse'] = {}
        d['u0_reverse'] = {}
        d['w0_reverse'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['xx'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['zz'] = {}
        d['__derivative']['zzx'] = {}
        d['__derivative']['x']['zeta0_reverse'] = {}
        d['__derivative']['xx']['zeta0_reverse'] = {}
        d['__derivative']['x']['u0_reverse'] = {}
        d['__derivative']['z']['u0_reverse'] = {}
        d['__derivative']['zz']['u0_reverse'] = {}
        d['__derivative']['zzx']['u0_reverse'] = {}
        d['__derivative']['z']['w0_reverse'] = {}

        d['zeta0_reverse'] = zeta
        d['__derivative']['x']['zeta0_reverse'] = zetax
        d['__derivative']['xx']['zeta0_reverse'] = zetaxx

        # Save velocity results
        d['u0_reverse'] = u[0]
        d['__derivative']['x']['u0_reverse'] = u[1]
        d['__derivative']['z']['u0_reverse'] = u[2]
        d['__derivative']['zz']['u0_reverse'] = u[3]
        d['__derivative']['zzx']['u0_reverse'] = u[4]

        d['w0_reverse'] = w[0]
        d['__derivative']['z']['w0_reverse'] = w[1]
        return d

    def waterlevel(self):
        """Solves the boundary value problem for the water level

        Returns:
            zeta - water level and its first and second derivative w.r.t. x
        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        r, rx, rxx = self.rf(self.x)
        a, ax, axx = self.af(self.x, r, rx, rxx)
        H = self.input.v('H', x=self.x / self.L) + self.input.v('R', x=self.x / self.L)

        M = ((a * np.sinh(r * H) / r) - H) * self.input.v('B', x=self.x / self.L) * (self.G / (1j * self.OMEGA))
        F = np.zeros((jmax+1, 1), dtype=complex)    # Forcing term shape (x, number of right-hand sides)
        Fopen = np.zeros((1, 1), dtype=complex)     # Forcing term shape (1, number of right-hand sides)
        Fclosed = np.zeros((1, 1), dtype=complex)   # Forcing term shape (1, number of right-hand sides)
        Fclosed[0,0] = self.bca

        Z, Zx, _ = zetaFunctionUncoupled(1, M, F, Fopen, Fclosed, self.input, hasMatrix = False, reverseBC = True)
        zeta = Z[:, 0]
        zeta_x = Zx[:, 0]
        zeta_xx = np.gradient(Zx[:, 0], self.x[1], edge_order=2)

        return zeta, zeta_x, zeta_xx

  