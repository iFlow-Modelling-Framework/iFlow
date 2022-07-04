"""
Parabolic depth dependent eddy viscosity with x, z and f dependency (with or without time variations)
Used in combination with no-slip boundary condition. Uses roughness parameters z0* (=z0/H) and zs* (=zs/H)

Original date: 16-Jul-15
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
from .util.ParabolicXZF import ParabolicXZF
from .util.UniformXF import UniformXF
import numpy as np


class TurbulenceParabolic:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running parabolic turbulence model')

        # Determine complex amplitudes of the eddy viscosity
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, 1, fmax+1))

        # Set roughness parameters
        zs_dimless = np.zeros((jmax+1, 1, fmax+1))
        zb_dimless = np.zeros((jmax+1, fmax+1))
        zs_dimless[:, 0, 0] = np.real(10. ** -6 / (Av0[:,0,0] * (1 + self.input.v('z0*', range(0, jmax+1)))))
        zb_dimless[:, 0] = self.input.v('z0*')

        # prepare the smaller datacontainers used in the function sParabolicXZF and UniformXF. These functions will be called when calling
        #   Av and z0. These functions require the grid and complex amplitudes above
        data = self.input.slice('grid')
        data.merge(self.input.slice('z0*'))
        data.addData('coef', Av0)
        data.addData('zs*', zs_dimless)

        dataRough = self.input.slice('grid')
        dataRough.addData('coef', zb_dimless*self.input.v('H', x=0))

        z0 = UniformXF   (dataRough, self.input.v('n')+1   ) # add +1 to n to convert z0* to z0 (i.e. add extra affect of depth)
        Av = ParabolicXZF(data     , self.input.v('m')     )

        # prepare output
        d = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['xx'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['zz'] = {}
        
        d['Av'] = Av.value
        d['__derivative']['z']['Av'] = Av.derivative
        d['__derivative']['zz']['Av'] = Av.secondDerivative
        
        d['Roughness'] = z0.value
        d['__derivative']['x']['Roughness'] = z0.derivative
        d['__derivative']['xx']['Roughness'] = z0.secondDerivative

        d['BottomBC'] = 'NoSlip'
        return d
