"""
Converter

Date: 07-Dec-16
Authors: Y.M. Dijkstra
"""
import numpy as np
from copy import copy

class Converter:
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        d = {}
        # d['hatc1_a'] = {}
        # d['hatc1_ax'] = {}
        # d['hatc1_a1'] = {}

        hatc0 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        hatc1_a = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        hatc1_ax = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        hatc1_a1 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        hatc0[:, :, 0] = self.input.v('hatc a', 'c00', range(0, jmax+1), range(0, kmax+1))
        hatc0[:, :, 2] = self.input.v('hatc a', 'c04', range(0, jmax+1), range(0, kmax+1))

        d['hatc0'] = hatc0

        # for key in self.input.getKeysOf('hatc a', 'c12'):
        #     hatc1_a[:, :, 1] = self.input.v('hatc a', 'c12', key, range(0, jmax+1), range(0, kmax+1))
        #     d['hatc1_a'][key] = copy(hatc1_a)
        #
        # for key in self.input.getKeysOf('hatc ax', 'c12'):
        #     hatc1_ax[:, :, 1] = self.input.v('hatc ax', 'c12', key, range(0, jmax+1), range(0, kmax+1))
        #     d['hatc1_ax'][key] = copy(hatc1_ax)
        #
        # d['hatc1_a1']['def'] = hatc1_a1



        return d