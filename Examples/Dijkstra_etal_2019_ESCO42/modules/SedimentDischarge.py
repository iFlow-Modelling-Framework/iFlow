"""
Sediment-discharge relation

Date: 17-09-2017
Authors: Y.M. Dijkstra
"""
import logging


class SedimentDischarge:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        c0 = self.input.v('QC_c')
        cexp = self.input.v('QC_exp')
        Q = self.input.v('Q1')

        Qsed = c0*Q**cexp

        d = {}
        d['Qsed'] = Qsed
        d['xsedsource'] = self.input.v('xsource')
        d['Qsedsource'] = 0.7*Qsed/self.input.v('B', jmax)

        return d

