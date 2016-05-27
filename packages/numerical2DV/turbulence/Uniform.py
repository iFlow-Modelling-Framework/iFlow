"""
Eddy viscosity with uniform profile (with or without time variations)
Used in combination with partial slip boundary condition and parameter sf0

Date: 02-06-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import logging
import nifty as ny
from profiles.UniformXF import UniformXF


class Uniform:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Running turbulence model Uniform')

        # Determine complex amplitudes of the eddy viscosity and roughness parameter
        fmax = self.input.v('grid', 'maxIndex', 'f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))
        sf0 = ny.amp_phase_input(self.input.v('sf0'), 0, (1, fmax+1))

        # prepare the smaller datacontainers used in the functions UniformXF. These functions will be called when calling
        #   Av and sf0. These functions require the grid and complex amplitudes above
        data = self.input.slice('grid')
        data.addData('coef', Av0)

        dataRough = self.input.slice('grid')
        dataRough.addData('coef', sf0)

        sf = UniformXF(['x', 'f'], dataRough , self.input.v('n'))
        Av = UniformXF(['x', 'f'], data, self.input.v('m'))

        # prepare output
        d = dict()
        d['Roughness'] = sf.function
        d['Av'] = Av.function
        d['BottomBC'] = 'PartialSlip'
        return d


