"""
Nutrient-discharge relation

Date: 17-09-2017
Authors: Y.M. Dijkstra
"""
import logging


class NutrientsDischarge:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        c0 = self.input.v('QN_c')
        cexp = self.input.v('QN_exp')
        c1 = self.input.v('QPhos_c')
        c1exp = self.input.v('QPhos_exp')
        Q = self.input.v('Q1')

        QN = c0*Q**cexp                 # In mol N/s
        if c1exp is not None:
            QPhos = c1*Q**c1exp       # In mol P/s
        else:
            QPhos = 0

        d = {}
        d['QN'] = QN
        d['QPhos'] = QPhos
        return d

