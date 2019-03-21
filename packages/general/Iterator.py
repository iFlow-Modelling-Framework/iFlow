"""
Iterator

Date: 19-06-2017
Authors: Y.M. Dijkstra
"""
import nifty as ny
import logging


class Iterator:
    # Variables
    logger = logging.getLogger(__name__)
    MAXITERS = 500

    # Methods
    def __init__(self, input):
        self.input = input

        # also instantiate dependent modules
        modules = ny.toList(self.input.v('modules'))
        for module in modules:
            module.instantiateModule()
        return

    def stopping_criterion(self, iteration):
        """Stop when all dependent module say stop.
        Stop immediately if none of the dependent modules has a stopping criterion"""
        self.iteration = iteration
        stop = [True]
        if iteration > self.MAXITERS:
            self.logger.warning('Iterator module stops, but has not converged.')
            return True

        modules = ny.toList(self.input.v('modules'))
        for module in modules:
            if hasattr(module, 'stopping_criterion'):
                module.addInputData(self.input.data)
                stop.append(module.stopping_criterion(iteration))

        if all(stop):
            return True
        else:
            return False

    def run_init(self):
        d = {}
        modules = ny.toList(self.input.v('modules'))
        for module in modules:
            module.addInputData(self.input.data)
            d1 = module.run(init=True)
            d.update(d1.data)

        return d

    def run(self):
        d = {}
        modules = ny.toList(self.input.v('modules'))
        for module in modules:
            module.addInputData(self.input.data)
            d1 = module.run()
            d.update(d1.data)

        if self.iteration >= self.MAXITERS:     # output variable indicating if iterotor has converged or not.
            d['converged'] = False
        else:
            d['converged'] = True

        return d