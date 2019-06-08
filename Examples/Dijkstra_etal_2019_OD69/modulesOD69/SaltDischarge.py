"""
Salinity parameter x_c-discharge relation

Date: 14-07-2018
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny


class SaltDischarge:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        Q = self.input.v('Q1')
        Qsource = ny.toList(self.input.v('Qsource'))
        Q = Q + sum(Qsource)

        d = {}
        d['xc'] = 41000.*Q**-0.24
        return d

