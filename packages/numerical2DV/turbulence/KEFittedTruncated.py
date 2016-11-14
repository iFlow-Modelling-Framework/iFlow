"""
k-epsilon fitted turbulence model using truncation of the velocity and water level
Implements TurbulenceKepFitted_core. See parent class for explanation

Runs its parent's methods.

Date: 02-11-2016 (original date: 20-11-2015)
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted_core import TurbulenceKepFitted_core


class KEFittedTruncated(TurbulenceKepFitted_core):
    # Variables
    order = None        # order = None is used to indicate truncation

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFitted_core.__init__(self, input, submodulesToRun)
        return

    def run_init(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - init')
        self.truncationorder = self.input.v('truncationOrder')
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
        self.logger.info('Running $k-\epsilon$ fitted turbulence model')

        Av, roughness, _, R = self.main(self.order)

        # load to dictionary
        d = {}
        d['Roughness'] = roughness
        d['Av'] = Av
        if self.input.v('referenceLevel')=='True':
            d['R'] = R
        return d

