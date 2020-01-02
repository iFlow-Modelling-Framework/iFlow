"""
Leading-order k-epsilon fitted turbulence model
Implements TurbulenceKepFitted_core. See parent class for explanation

This module sets the order to 0 and runs its parent's methods

Date: 02-11-2016 (original date: 20-11-2015)
Authors: Y.M. Dijkstra
"""
from .TurbulenceKepFitted_core import TurbulenceKepFitted_core


class KEFittedLead(TurbulenceKepFitted_core):
    # Variables
    order = 0

    # Methods
    def __init__(self, input):
        TurbulenceKepFitted_core.__init__(self, input)
        return

    def run_init(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - Leading order - init')

        Av, roughness, BottomBC, R = self.main(self.order, init=True)

        # load to dictionary
        d = {}
        d['Roughness'] = roughness
        d['Av'] = Av
        d['BottomBC'] = BottomBC
        if self.input.v('referenceLevel')=='True':
            d['R'] = R
        return d

    def run(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - Leading order')

        Av, roughness, _, R= self.main(self.order)

        # load to dictionary
        d = {}
        d['Roughness'] = roughness
        d['Av'] = Av
        if self.input.v('referenceLevel')=='True':
            d['R'] = R
        return d

