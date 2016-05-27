"""
SaltExponential

Date: 02-03-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from nifty.functionTemplates import FunctionBase
from nifty import toList
import logging


class SaltExponential(FunctionBase):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.__input = input
        FunctionBase.__init__(self, ['x', 'f'])
        return

    def run(self):
        self.logger.info('Running module SaltExponential')
        self.L = self.__input.v('L')
        self.ssea = self.__input.v('ssea')
        self.Ls = self.__input.v('Ls')

        d = {}
        d['s0'] = self.function
        return d

    def value(self, x, f, **kwargs):
        x = x*self.L
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1, len(f))

        s = self.ssea*np.exp(-x/self.Ls)
        s = s.reshape(len(s), 1)*f
        return s

    def derivative(self, x, f, **kwargs):
        if kwargs['dim'] == 'x':
            x = np.asarray(x*self.L)
            f = [1-bool(j) for j in toList(f)]
            f = np.asarray(f).reshape(1, len(f))

            sx = -self.ssea/self.Ls*np.exp(-x/self.Ls)
            sx = sx.reshape(len(sx), 1)*f
        else:
            sx = None
            FunctionBase.derivative(self)
        return sx
