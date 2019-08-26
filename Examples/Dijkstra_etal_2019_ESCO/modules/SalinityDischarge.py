"""
Salinity-discharge relation

Date: 17-09-2017
Authors: Y.M. Dijkstra
"""
import logging


class SalinityDischarge:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        Q = self.input.v('Q1')

        d = {}
        d['xc'] = 100000.*Q**(-1./7.)

        return d