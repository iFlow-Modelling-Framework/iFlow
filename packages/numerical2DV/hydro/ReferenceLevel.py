"""



Date: 24-04-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from uFunctionMomentumConservative import uFunctionMomentumConservative
from zetaFunctionMassConservative import zetaFunctionMassConservative
from nifty.functionTemplates import FunctionBase


class ReferenceLevel:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        if hasattr(self, 'difference'):
            print self.difference
            if self.difference < 10**-3:
                del self.difference
                return True
        return False

    def run_init(self):
        if self.input.v('Av') is None:
            self.logger.info('Running module ReferenceLevel - init')
            d={}
            R = initialRef('x', self.input)
            d['R'] = R.function
            # self.difference = 0
        else:
            d = self.run()
        return d

    def run(self):
        """

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        if self.input.v('Av') is None:
            self.logger.info('Running module ReferenceLevel - init')
            d={}
            d['R'] = np.loadtxt('output/Rtrue.txt')
            self.difference = 0.
            return d
        self.logger.info('Running module ReferenceLevel')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        ftot = 2*fmax+1

        R = np.zeros(jmax+1)
        Q = -self.input.v('Q1')
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        dx = x[1:]-x[0:-1]
        for j in range(0, jmax):
            f = self.uSolver(j, R[j])
            B = self.input.v('B', j)
            intfac = B*f
            Rx = Q/intfac

            R[j+1] = R[j]+Rx*dx[j]

            self.input.data['grid']['low']['z'] = R

        self.difference = np.linalg.norm(R - self.input.v('R', range(0, jmax+1)), np.inf)
        d = {}
        d['R'] = R
        #self.plot(R)
        #np.savetxt('output/Rtest.txt', abs(zeta[:,0,0,0]), fmt='%.6e')
        return d

    def plot(self, R):
        import matplotlib.pyplot as plt
        import nifty as ny
        import step as st
        jmax = self.input.v('grid', 'maxIndex', 'x')
        st.configure()
        plt.figure(1, figsize=(1,2))
        plt.subplot(1,2,1)
        z = ny.dimensionalAxis(self.input.slice('grid'), 'z')
        plt.plot(self.input.v('R', range(0, jmax+1)))
        plt.plot(z[:,-1,0])
        plt.plot(R)
        plt.subplot(1,2,2)
        plt.plot(np.real(self.input.v('Av', range(0, jmax+1), 0, 0)))

        st.show()


    def uSolver(self, j, R):
        G = self.input.v('G')
        Av = self.input.v('Av', j, 0, 0)
        sf = self.input.v('Roughness', j, 0, 0)
        H = self.input.v('H', j)
        z = self.input.v('grid', 'axis', 'z')[0,:]
        z = z*(-H-R)+R

        f = 0.5*G/Av*(z**2-H**2)+G*R*(-H-z)/Av+G/sf*(-H-R)
        f = np.mean(f)*(H+R)
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.hold(True)
        # plt.plot(np.real(f),z,'-')
        # plt.show()
        return f

class initialRef(FunctionBase):
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.H = data
        return

    def value(self, x, **kwargs):
        return np.maximum(self.H.n('H', x=x)+3, 0)

    def derivative(self, x, **kwargs):
        der = -self.H.d('H', x=x, dim='x')
        der[np.where(self.H.n('H', x=x)+3 < 0)] = 0
        return der
