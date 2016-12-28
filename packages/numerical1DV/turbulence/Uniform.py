"""
Date: 02-06-15
Authors: R.L. Brouwer, Y.M. Dijkstra
"""
import logging
import nifty as ny


class Uniform:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Running uniform turbulence model')

        d = {}
        fmax = self.input.v('grid','maxIndex','f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))

        d['Roughness'] = self.input.v('sf')
        d['Av'] = Av0
        d['BottomBC'] = 'PartialSlip'
        return d
