"""
SaltHyperbolicTangent

Date: 02-03-2016
Authors: Y.M. Dijkstra
"""
import numpy as np
from nifty.functionTemplates import FunctionBase
from nifty import toList
import logging

class SaltHyperbolicTangent(FunctionBase):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.__input = input
        FunctionBase.__init__(self, ['x', 'f'])
        return

    def run(self):
        self.logger.info('Running module SaltHyperbolicTangent')
        self.L = self.__input.v('L')
        self.ssea = self.__input.v('ssea')
        self.xc = self.__input.v('xc')
        self.xl = self.__input.v('xl')

        d = {}
        d['s0'] = self.function
        return d

    def value(self, x, f, **kwargs):
        x = x*self.L
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1,len(f))

        s = self.ssea/2.*(1-np.tanh((x-self.xc)/self.xl))
        s = s.reshape(len(s), 1)*f
        return s

    def derivative(self, x, f, **kwargs):
        if kwargs['dim'] == 'x':
            x = np.asarray(x*self.L)
            f = [1-bool(j) for j in toList(f)]
            f = np.asarray(f).reshape(1,len(f))

            sx = self.ssea/2.*(np.tanh((x-self.xc)/self.xl)**2-1)/self.xl
            sx = sx.reshape(len(sx),1)*f
        else:
            sx = None
            FunctionBase.derivative(self)
        return sx
