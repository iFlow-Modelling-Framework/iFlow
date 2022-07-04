"""
Eddy viscosity with uniform profile (with or without time variations)
Used in combination with partial slip boundary condition and parameter sf0

Date: 02-06-15
Authors: Y.M. Dijkstra, R.L. Brouwer (modified by I.Jalon-Rojas to include the tanh option)
"""
import logging
import nifty as ny
from packages.turbulence2DV.util.UniformXF import UniformXF
import numpy as np


class Uniform_VaryingSfAv:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running turbulence model Uniform')

        # Determine complex amplitudes of the eddy viscosity and roughness parameter
        fmax = self.input.v('grid', 'maxIndex', 'f')
        jmax = self.input.v('grid', 'maxIndex', 'x')

        sf0a = self.input.v('sf0a')
        sf0b = self.input.v('sf0b')
        x = self.input.v('grid', 'axis', 'x')
        option = self.input.v('option')

        # Determine the change of roughness
        aux1 = np.nonzero(x >= self.input.v('Lend_sf0a') / self.input.v('L'))
        jch = aux1[0][0]
        aux2 = np.nonzero(x >= self.input.v('Lini_sf0b') / self.input.v('L'))
        jch1 = aux2[0][0]

        sf0 = np.zeros((jmax + 1, fmax + 1))

        if option == 'linear':
            sf0[0:jch, 0] = sf0a * np.ones(np.shape(sf0[0:jch, 0]))
            sf0avector= sf0a * np.ones(np.shape(sf0[jch:jch1, 0]))
            coefvector= ((sf0b-sf0a)/(x[jch1]-x[jch])) * np.ones(np.shape(sf0[jch:jch1, 0]))
            sf0[jch:jch1, 0] = sf0avector + coefvector * (x[jch:jch1] - x[jch])
            sf0[jch1:jmax + 1, 0] = sf0[jch1 - 1, 0] * np.ones(np.shape(sf0[jch1:jmax + 1, 0]))
        elif option == 'exponential':
            sf0[0:jch, 0] = sf0a * np.ones(np.shape(sf0[0:jch, 0]))
            sf0avector = sf0a * np.ones(np.shape(sf0[jch:jch1, 0]))
            coefvector = ((sf0b - sf0a) / (x[jch1] - x[jch])) * np.ones(np.shape(sf0[jch:jch1, 0]))
            sf0[jch:jch1, 0] = sf0avector + coefvector * (x[jch:jch1] - x[jch])

            b = (sf0a/sf0b) ** (1/(x[jch]-x[jch1]))
            a = sf0a/b**x[jch]
            sf0[jch:jch1, 0] = a*b**x[jch:jch1]
            sf0[jch1:jmax + 1, 0] = sf0[jch1 - 1, 0] * np.ones(np.shape(sf0[jch1:jmax + 1, 0]))
        elif option== 'tanh':
            xcenter = (x[jch1]-x[jch])/2
            xaux = 3*(x-xcenter-x[jch])/xcenter
            sf0[:, 0]=(sf0a+sf0b)/2+(sf0b-sf0a)/2*np.tanh(xaux)

        depth = self.input.v('grid', 'low', 'z', x=0) - self.input.v('grid', 'high', 'z', x=0)
        Av0 = 0.49 * sf0 * depth

        # prepare the smaller datacontainers used in the functions UniformXF. These functions will be called when calling
        #   Av and sf0. These functions require the grid and complex amplitudes above
        data = self.input.slice('grid')
        data.addData('coef', Av0)

        dataRough = self.input.slice('grid')
        dataRough.addData('coef', sf0)

        sf = UniformXF( dataRough , self.input.v('n'))
        Av = UniformXF(data, self.input.v('m'))

        # prepare output
        d = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['xx'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['zz'] = {}

        d['Roughness'] = sf.value
        d['__derivative']['x']['Roughness'] = sf.derivative
        d['__derivative']['xx']['Roughness'] = sf.secondDerivative

        d['skin_friction'] = sf.value(0, 0)

        d['Av'] = Av.value
        d['__derivative']['x']['Roughness'] = Av.derivative
        d['__derivative']['xx']['Roughness'] = Av.secondDerivative

        d['BottomBC'] = 'PartialSlip'
        return d


