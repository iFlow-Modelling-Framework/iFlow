"""



Date: 27-10-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from src.util.diagnostics.KnownError import KnownError


class ReferenceLevel:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10**-3         # tollerance for convergence of 1 mm

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        if hasattr(self, 'difference'):
            print(self.difference)
            if self.difference < self.TOLLERANCE:
                del self.difference
                return True
        return False

    def run_init(self):
        # river discharge is leading-order discharge or first-order discharge if Q0=0
        self.Q = self.input.v('Q0')
        if self.Q==None or self.Q == 0.:
            self.Q = self.input.v('Q1')
            if self.Q==None:
                self.Q = 0

        # try initial guess if Av is unknown
        if self.input.v('Av') is None:
            self.logger.info('Running module ReferenceLevel - init')
            d={}
            if self.Q == 0.:
                d['R'] = 0
                self.difference = 0
            else:
                R = initialRef('x', self.input)
                d['R'] = R.value
                d['__derivative'] = {}
                d['__derivative']['x'] = {}
                d['__derivative']['x']['R'] = R.derivative
                if hasattr(self, 'difference'):
                    del self.difference

        # If Av is known, do a full calculation
        else:
            d = self.run()
        return d

    def run(self):
        if self.input.v('Av') is None:
            raise KnownError('Reference level runs, but cannot find Av.\n'
                             'Check if this variable is provided by the module that promisses this and if it is not deleted ')

        self.logger.info('Running module ReferenceLevel')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')

        self.G = self.input.v('G')
        self.bottomBC = self.input.v('BottomBC')

        self.z = np.linspace(0, 1, np.minimum(kmax, 100))
        self.xaxis = self.input.v('grid', 'axis', 'x', x=np.linspace(0, 1, jmax+1))
        self.Av = np.real(self.input.v('Av', x=self.xaxis, z=self.z, f=0))
        self.H = self.input.v('H', x=self.xaxis)

        self.sf = np.real(self.input.v('Roughness', x=self.xaxis, z=0, f=0))
        if self.bottomBC == 'QuasiQuadraticSlip' and (self.input.v('Q0') is None or (self.input.v('Q0') == 0)):   #NB. since the river discharge is first order, add a multiplier for sf
            self.sf = 2*self.sf
        B = self.input.v('B', x=self.xaxis)

        # Construct reference level by stepping from x=0 towards x=L
        R = np.zeros((jmax+1))
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        dx = x[1:]-x[0:-1]
        R[0] = 0.
        for j in range(0, jmax):
            c = self.uSolver(j+1, R[j])*B[j+1]
            b = self.uSolver2(j+1, R[j])*B[j+1]*dx[j]
            a = self.uSolver3(j+1, R[j])*B[j+1]*(dx[j]**2)
            Rx_all = np.roots([a, b, c, self.Q])
            for Rx in Rx_all:
                if np.imag(Rx)==0:
                    Rx = np.real(Rx)
                    break
            R[j+1] = R[j]+Rx*dx[j]

        # if self.input.v('zeta1', range(0, jmax+1), 0, 0) is not None:
        #     R = R+self.input.v('zeta1', range(0, jmax+1), 0, 0)-self.input.v('zeta1','river', range(0, jmax+1), 0, 0)

        # Compute convergence
        R = np.maximum(R, -self.H+0.01)
        self.difference = np.linalg.norm(R - self.input.v('R', range(0, jmax+1)), np.inf)

        d = {}
        d['R'] = R

        return d

    def uSolver(self, j, R):
        z = self.z*(-self.H[j]-R)+R

        dz = (z[0]-z[1])
        f = self.quickInt(((R-z)/self.Av[j, :])[::-1], dz, cumulative=True)[::-1]

        if self.bottomBC in ['PartialSlip', 'QuasiQuadraticSlip']:
            f += ((R+self.H[j])/self.sf[j])
        f = -self.G*self.quickInt(f, dz)
        return f

    def uSolver2(self, j, R):
        z = self.z*(-self.H[j]-R)+R

        dz = (z[0]-z[1])
        f = self.quickInt((1./self.Av[j, :])[::-1], dz, cumulative=True)[::-1]
        f += (R-z)/self.Av[j, :]

        if self.bottomBC in ['PartialSlip', 'QuasiQuadraticSlip']:
            f +=  2./self.sf[j]
        f = -self.G*self.quickInt(f, dz)
        return f

    def uSolver3(self, j, R):
        z = self.z*(-self.H[j]-R)+R

        dz = (z[0]-z[1])
        f = self.quickInt(1./self.Av[j,:], dz)

        if self.bottomBC in ['PartialSlip', 'QuasiQuadraticSlip']:
            f +=  1./self.sf[j]
        f = -self.G*f
        return f

    def quickInt(self, u, dz, cumulative=False):
        """Quick integral over a 1D function assuming a uniform grid with grid points on the boundary"""
        u[1:] = .5*u[0:-1]+.5*u[1:]
        u[0] = 0
        if cumulative:
            return np.cumsum(u)*dz
        else:
            return np.sum(u)*dz

class initialRef():
    def __init__(self, dimNames, data):
        self.dimNames = dimNames
        self.H = data
        return

    def value(self, x, **kwargs):
        return np.maximum(self.H.n('H', x=x)+10., 0)

    def derivative(self, x, **kwargs):
        der = -self.H.d('H', x=x, dim='x')
        der[np.where(self.H.n('H', x=x)+10. < 0)] = 0
        return der
