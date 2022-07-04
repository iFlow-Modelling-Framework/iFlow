"""
SaltHyperbolicTangent

Original date: 02-03-2016
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
from nifty import toList
import logging


class SaltHyperbolicTangent():
    # Variables
    __logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.__input = input
        self.dimNames = ['x', 'f']
        return

    def run(self):
        self.__logger.info('Running module SaltHyperbolicTangent')
        self.L = self.__input.v('L')
        self.ssea = self.__input.v('ssea')
        self.xc = self.__input.v('xc')
        self.xl = self.__input.v('xl')

        d = {}
        d['s0'] = self.value
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['x']['s0'] = self.derivative
        return d

    def value(self, x, f, **kwargs):
        x = x*self.L
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1,len(f))

        s = self.ssea/2.*(1-np.tanh((x-self.xc)/self.xl))
        s = s.reshape(len(s), 1)*f
        return s

    def derivative(self, x, f, **kwargs):
        x = np.asarray(x*self.L)
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1,len(f))

        sx = self.ssea/2.*(np.tanh((x-self.xc)/self.xl)**2-1)/self.xl
        sx = sx.reshape(len(sx),1)*f
        return sx
