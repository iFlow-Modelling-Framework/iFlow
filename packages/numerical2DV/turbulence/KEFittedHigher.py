"""
Higher-order (n>=2) k-epsilon fitted turbulence model
Implements TurbulenceKepFitted_core. See parent class for explanation

Takes the current order from the data (e.g. provided by 'HigherOrderIterator') and runs its parent's methods

Date: 02-11-2016 (original date: 20-11-2015)
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted_core import TurbulenceKepFitted_core


class KEFittedHigher(TurbulenceKepFitted_core):
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFitted_core.__init__(self, input, submodulesToRun)
        return

    def run_init(self):
        self.order = self.input.v('order')
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - order '+str(self.order)+' - init')

        Av, roughness, BottomBC,_ = self.main(self.order, init=True)

        # load to dictionary
        d = {}
        d['Roughness'+str(self.order)] = roughness
        d['Av'+str(self.order)] = Av
        return d

    def run(self):
        self.logger.info('Running $k-\epsilon$ fitted turbulence model - order '+str(self.order))

        Av, roughness, _, _ = self.main(self.order)

        # load to dictionary
        d = {}
        d['Roughness'+str(self.order)] = roughness
        d['Av'+str(self.order)] = Av
        return d

