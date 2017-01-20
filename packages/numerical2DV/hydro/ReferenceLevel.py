"""



Date: 27-10-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from nifty.functionTemplates import FunctionBase
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
            print self.difference
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
                d['R'] = R.function
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

        # Construct reference level by stepping from x=0 towards x=L
        R = np.zeros(jmax+1)
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        dx = x[1:]-x[0:-1]
        for j in range(0, jmax):
            f = self.uSolver(j, R[j])
            B = self.input.v('B', j)
            intfac = B*f
            Rx = -self.Q/intfac

            R[j+1] = R[j]+Rx*dx[j]
            self.input.data['grid']['low']['z'] = R # add R to grid to be used in next iteration

        # Compute convergence
        self.difference = np.linalg.norm(R - self.input.v('R', range(0, jmax+1)), np.inf)

        d = {}
        d['R'] = R
        return d

    def uSolver(self, j, R):
        kmax = self.input.v('grid', 'maxIndex', 'z')
        G = self.input.v('G')
        Av = np.real(self.input.v('Av', j, range(0,kmax+1), 0))
        bottomBC = self.input.v('BottomBC')
        H = self.input.v('H', j)
        z = self.input.v('grid', 'axis', 'z')[0,:]
        z = z*(-H-R)+R

        rescale = (R+H)/(self.input.v('grid', 'low', 'z', 0) - self.input.v('grid', 'high', 'z', 0))
        f = ny.integrate(((R-z)/Av).reshape((1,len(z))), 'z', kmax, range(0, kmax+1),self.input.slice('grid'), INTMETHOD='INTERPOLSIMPSON')
        f = f*rescale
        if bottomBC == 'PartialSlip':
            sf = np.real(self.input.v('Roughness', j, 0, 0))
            f += ((R+H)/sf)
        f = -G*rescale*ny.integrate(f,'z',kmax,0,self.input.slice('grid'), INTMETHOD='INTERPOLSIMPSON')[0,0]

        return f

class initialRef(FunctionBase):
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.H = data
        return

    def value(self, x, **kwargs):
        return np.maximum(self.H.n('H', x=x)+10., 0)

    def derivative(self, x, **kwargs):
        der = -self.H.d('H', x=x, dim='x')
        der[np.where(self.H.n('H', x=x)+10. < 0)] = 0
        return der
