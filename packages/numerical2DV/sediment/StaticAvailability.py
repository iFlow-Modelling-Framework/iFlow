"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
import step as st
import matplotlib.pyplot as plt


class StaticAvailability:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        self.logger.info('Running module StaticAvailability')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1

        ## Make transport terms
        zeta0 = self.input.v('zeta0', range(0, jmax+1), 0, range(0, fmax+1))
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c0 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c0x = self.input.d('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        c1_a = self.input.v('hatc1_a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_ax = self.input.v('hatc1_ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        #
        # c0 = c0*0
        # c1_a = c1_a*0
        # c1_ax = c1_ax*0
        # c0[:,:,0] = self.input.v('hatc a', 'c00')
        # c0[:,:,2] = self.input.v('hatc a', 'c04')
        # c1_a[:,:,1] = self.input.v('hatc a', 'c12')
        # c1_ax[:,:,1] = self.input.v('hatc ax', 'c12')
        # c0x = ny.derivative(c0, 'x', self.input.slice('grid'))
        #

        Kh = self.input.v('Kh', range(0, jmax+1), [0])

        for n in range(0, fmax+1):
            T = ny.complexAmplitudeProduct(u0, c1_a, 2)[:,:,0] + ny.complexAmplitudeProduct(u1, c0, 2)[:,:,0] - Kh*c0x[:,:,0]
        T = ny.integrate(T, 'z', kmax, 0, self.input.slice('grid')) [:,0]
        T += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, 0, :], c0[:, 0, :], 1), zeta0, 1)[:, 0]

        F = ny.complexAmplitudeProduct(u0, c1_ax, 2)[:,:,0] - Kh*c0[:,:,0]
        F = ny.integrate(F, 'z', kmax, 0, self.input.slice('grid')) [:,0]

        ## Solve Ta+Fax = 0
        integral = -ny.integrate(T/(F+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
        B = self.input.v('B', range(0, jmax+1))
        astar = self.input.v('astar')
        A = astar * ny.integrate(B, 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))
        # A = A**-1.      # mistake Ronald?

        a = A*np.exp(integral)
        ax = integral*T/F*a

        d = {}
        d['a'] = a


        # c0
        if self.input.getKeysOf('hatc0'):
            d['c0'] = {}
            for key in self.input.getKeysOf('hatc0'):
                d['c0'][key] = self.input.v('hatc0', key)*a.reshape((jmax+1, 1, 1))
        else:
            d['c0'] = self.input.v('hatc0')*a.reshape((jmax+1, 1, 1))

        # c1
        keys = list(set(ny.toList(self.input.getKeysOf('hatc1_a'))+ny.toList(self.input.getKeysOf('hatc1_ax'))))
        if keys:
            d['c1'] = {}
            for key in keys:
                d['c1'][key] = 0
                try:
                    d['c1'][key] += self.input.v('hatc1_a', key)*a.reshape((jmax+1, 1, 1))
                except:
                    pass
                try:
                    d['c1'][key] += self.input.v('hatc1_ax', key)*ax.reshape((jmax+1, 1, 1))
                except:
                    pass
        else:
            d['c1'] = self.input.v('hatc1_a')*a.reshape((jmax+1, 1, 1)) + self.input.v('hatc1_ax')*ax.reshape((jmax+1, 1, 1))

        # d['T']
        #
        # d['F']
        return d

