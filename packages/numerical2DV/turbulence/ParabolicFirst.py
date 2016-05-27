"""
Parabolic depth dependent eddy viscosity with x, z and f dependency

Date: 16-Jul-15
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
from profiles.ParabolicXZF import ParabolicXZF
import numpy as np


class ParabolicFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        # self.logger.info('Running parabolic turbulence model - first order')
        #
        # fmax = self.input.v('grid', 'maxIndex', 'f')
        # jmax = self.input.v('grid', 'maxIndex', 'x')
        # z = self.input.v('grid', 'axis', 'z')
        # z = z.reshape((1, z.shape[-1], 1))
        # # Av1 = ny.amp_phase_input(self.input.v('Av1amp'), self.input.v('Av1phase'), (1, 1, fmax + 1))
        # # zs_dimless = 0.
        # #
        # # data = self.input.slice('grid')
        # # data.merge(self.input.slice('z0*'))
        # # data.addData('coef', Av1)
        # # data.addData('zs*', zs_dimless)
        # #
        # # Av = ParabolicXZF(['x', 'z', 'f'], data, self.input.v('m'))
        # #
        # # d = {}
        # # d['Av1'] = Av.function
        #
        # Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, 1, fmax + 1))
        # zeta = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        # Av1 = ny.complexAmplitudeProduct(Av0*np.ones((jmax+1, 1, 1)),zeta, 2) *(1.-z)/self.input.v('H', range(0, jmax+1), [0], range(0, fmax+1))
        # d = {}
        # d['Av1'] = Av1
        # d['Av2'] = 0
        # d['Av3'] = 0
        return d
