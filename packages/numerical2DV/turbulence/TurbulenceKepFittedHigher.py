"""
TurbulenceKepFittedHigher
Higher-order model of the k-epsilon fitted turbulence model.

Subclass of the TurbulenceKepFitted class, which implements all the content.
Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted import TurbulenceKepFitted


class TurbulenceKepFittedHigher(TurbulenceKepFitted):
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFitted.__init__(self, input, submodulesToRun)
        return

    def stopping_criterion(self, iteration):
        return TurbulenceKepFitted.stopping_criterion(self, iteration)

    def run_init(self):
        self.order = self.input.v('order')
        self.logger.info('Running k-epsilon fitted turbulence model - order '+str(self.order)+', init')

        Av, roughness, _ = self.main(init=True)

        d = {}
        d['Roughness'+str(self.order)] = roughness
        d['Av'+str(self.order)] = Av
        return d

    def run(self):
        self.order = self.input.v('order')
        self.logger.info('Running k-epsilon fitted turbulence model - order '+str(self.order))

        Av, roughness, _ = self.main()

        # load to dictionary
        d = {}
        d['Roughness'+str(self.order)] = roughness
        d['Av'+str(self.order)] = Av
        return d
