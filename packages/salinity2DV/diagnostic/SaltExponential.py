"""
SaltExponential

Date: 02-03-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from nifty import toList
import logging


class SaltExponential():
    # Variables
    __logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.__input = input
        self.dimNames = ['x', 'f']
        return

    def run(self):
        self.__logger.info('Running module SaltExponential')
        self.L = self.__input.v('L')
        self.ssea = self.__input.v('ssea')
        self.Ls = self.__input.v('Ls')

        d = {}
        d['s0'] = self.value
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['x']['s0'] = self.derivative

        return d

    def value(self, x, f, **kwargs):
        x = x*self.L
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1, len(f))

        s = self.ssea*np.exp(-x/self.Ls)
        s = s.reshape(len(s), 1)*f
        return s

    def derivative(self, x, f, **kwargs):
        x = np.asarray(x*self.L)
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1, len(f))

        sx = -self.ssea/self.Ls*np.exp(-x/self.Ls)
        sx = sx.reshape(len(sx), 1)*f
        return sx
