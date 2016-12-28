"""
Parabolic depth dependent eddy viscosity with z and f dependency

Date: 16-Jul-15
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
from profiles.ParabolicZF import ParabolicZF


class Parabolic:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Running parabolic turbulence model')

        fmax = self.input.v('grid', 'maxIndex', 'f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))
        H = self.input.v('H')
        Av = ParabolicZF(['z', 'f'], Av0, self.input.v('z0*'), H)

        z0 = self.input.v('z0*')*self.input.v('H')

        d = {}
        d['Av'] = Av.function
        d['Roughness'] = z0
        d['BottomBC'] = 'NoSlip'
        return d
