"""
Numerical method for Flow 2DV
Uses the vertical eddy viscosity and Prandtl-Schmidt number (sigma_rho) to compute the vertical eddy diffusivity Kv

Date: 12-04-2017
Authors: Y.M. Dijkstra
"""


class DiffusivityUndamped:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        d = {}

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = Av/self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])

        d['Kv'] = Kv
        return d


