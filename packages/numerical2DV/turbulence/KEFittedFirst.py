"""
First-order k-epsilon fitted turbulence model
Implements TurbulenceKepFitted_core. See parent class for explanation

This module sets the order to 1 and runs its parent's methods

Date: 02-11-2016 (original date: 20-11-2015)
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted_core import TurbulenceKepFitted_core


class KEFittedFirst(TurbulenceKepFitted_core):
    # Variables
    order = 1

    # Methods
    def __init__(self, input):
        TurbulenceKepFitted_core.__init__(self, input)
        return

    def run_init(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - First order - init')

        Av, roughness, BottomBC, _ = self.main(self.order, init=True)

        # load to dictionary
        d = {}
        d['Roughness1'] = roughness
        d['Av1'] = Av
        return d

    def run(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - First order')

        Av, roughness, _, _ = self.main(self.order)

        # load to dictionary
        d = {}
        d['Roughness1'] = roughness
        d['Av1'] = Av
        return d

