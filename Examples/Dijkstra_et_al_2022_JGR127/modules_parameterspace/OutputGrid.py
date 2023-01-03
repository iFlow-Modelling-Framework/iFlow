"""

Date: 28-08-2019
Authors: Y.M. Dijkstra
"""
import numpy as np


class OutputGrid:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Prepare a regular grid for a numerical 2DV computation

        Returns:
            Dictionary containing:
                grid:   dimensions (list) - list [x,z,f]
                        gridType (str) - 'Regular'
                        axis: x,z,f (ndarray - grid axes between 0 and 1
                        high: x,z,f (any) - dimension-full limit belonging to point 1 on the dimensionless axis
                        low: x,z,f (any) - dimension-full limit belonging to point 0 on the dimensionless axis
                        maxIndex: x,z,f (int) - maximum index number
                outgrid:    idem
        """
        grid = self.input.slice('outputgrid')
        jmaxout = self.input.v('outputgrid', 'maxIndex', 'x')
        xout_eq = np.linspace(0, 1, jmaxout+1)

        xout = self.input.v('adaptive_grid', 'axis', 'x', x=xout_eq)

        d = grid._data
        d['outputgrid']['axis']['x'] = xout

        return d


