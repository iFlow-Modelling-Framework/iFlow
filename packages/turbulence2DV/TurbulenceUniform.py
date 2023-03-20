"""
Eddy viscosity with uniform profile (with or without time variations)
Used in combination with partial slip boundary condition and parameter sf0

Date: 02-06-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import logging
import nifty as ny
from .util.UniformXF import UniformXF


class TurbulenceUniform:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running turbulence model Uniform')

        # Determine complex amplitudes of the eddy viscosity and roughness parameter
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))
        sf0 = self.input.v('sf0', range(0, jmax+1), 0, range(0, fmax+1))

        # prepare the smaller datacontainers used in the functions UniformXF. These functions will be called when calling
        #   Av and sf0. These functions require the grid and complex amplitudes above
        data = self.input.slice('grid')
        data.addData('coef', Av0)

        dataRough = self.input.slice('grid')
        dataRough.addData('coef', sf0)

        sf = UniformXF(dataRough, self.input.v('n'))
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

        d['Av'] = Av.value
        d['__derivative']['x']['Av'] = Av.derivative
        d['__derivative']['xx']['Av'] = Av.secondDerivative

        d['BottomBC'] = 'PartialSlip'
        return d


