"""
SaltHyperbolicTangent

Date: 16-05-2019
Authors: R.J.A. de Goede
"""
import numpy as np
from nifty import toList
import logging


class RoughnessHyperbolicTangent():
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.__input = input
        return

    def run(self):
        self.logger.info('Running module RoughnessHyperbolicTangent')
        self.L = self.__input.v('L')
        self.z0_est = self.__input.v('z0_est')
        self.z0_riv = self.__input.v('z0_riv')
        self.xc = self.__input.v('xc')
        self.xl = self.__input.v('xl')
        self.dimNames = ['x', 'f']

        d = {}
        d['z0*'] = self.value
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['x']['z0*'] = self.derivative
        return d

    def value(self, x, f, **kwargs):
        x = x*self.L
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1,len(f))

        r = self.z0_est + self.z0_riv*(1+np.tanh((x-self.xc)/self.xl))
        r = r.reshape(len(r), 1)*f
        return r

    def derivative(self, x, f, **kwargs):
        x = np.asarray(x*self.L)
        f = [1-bool(j) for j in toList(f)]
        f = np.asarray(f).reshape(1,len(f))

        rx = self.z0_riv*(np.tanh((x-self.xc)/self.xl)**2-1)/self.xl
        rx = rx.reshape(len(rx),1)*f
        return rx
