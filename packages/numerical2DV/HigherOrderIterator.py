"""
HigherOrderIterator

Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
import logging


class HigherOrderIterator:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.currentOrder = iteration+2
        if self.currentOrder <= self.maxOrder:
            stop = False
        else:
            stop = True
        return stop

    def run_init(self):
        self.maxOrder = int(self.input.v('maxOrder'))
        self.stopping_criterion(0)
        return self.run()

    def run(self):
        self.logger.info('Starting computation at order '+str(self.currentOrder))

        d = {}
        d['order'] = self.currentOrder
        d['maxOrder'] = self.maxOrder
        return d