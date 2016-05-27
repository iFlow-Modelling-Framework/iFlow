"""
TurbulenceKepFittedFirst
First-order model of the k-epsilon fitted turbulence model.

Subclass of the TurbulenceKepFitted class, which implements all the content.

Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted import TurbulenceKepFitted


class TurbulenceKepFittedFirst(TurbulenceKepFitted):
    # Variables
    order = 1

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFitted.__init__(self, input, submodulesToRun)
        return

    def stopping_criterion(self, iteration):
        return TurbulenceKepFitted.stopping_criterion(self, iteration)

    def run_init(self):
        self.logger.info('Running k-epsilon fitted turbulence model  - first order, init')
        Av, roughness, _ = self.main(init=True)

        d = {}
        d['Roughness1'] = roughness
        d['Av1'] = Av
        return d

    def run(self):
        self.logger.info('Running k-epsilon fitted turbulence model - first order')

        Av, roughness, _ = self.main()

        # load to dictionary
        d = {}
        d['Roughness1'] = roughness
        d['Av1'] = Av
        return d