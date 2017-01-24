"""
Date: 02-06-15
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
import numpy as np


class UniformQuadratic:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10.**-5

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        stop = False
        if hasattr(self, 'ubed'):
            kmax = self.input.v('grid','maxIndex','z')
            fmax = self.input.v('grid','maxIndex','f')
            ubed = self.input.v('u0', kmax, range(0, fmax+1))

            maxchange = max(np.abs((self.ubed-ubed)/(ubed+10**-6)))
            self.logger.info('uniformQuadratic - relative change in last iteration %s' % str(maxchange))

            if maxchange < self.TOLLERANCE:
                stop = True
        return stop

    def run_init(self):
        self.logger.info('Running uniformQuadratic turbulence model - initialise')

        d = dict()
        fmax = self.input.v('grid','maxIndex','f')
        d['Roughness'] = self.input.v('cf')
        d['Av'] = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))

        d['BottomBC'] = 'QuadraticSlip'
        return d

    def run(self):
        self.logger.info('Running uniformQuadratic turbulence model - iteration %s' % str(self.iteration))
        kmax = self.input.v('grid','maxIndex','z')
        fmax = self.input.v('grid','maxIndex','f')
        self.ubed = self.input.v('u0', kmax, range(0, fmax+1))   # needed for stopping criterion

        return {}
