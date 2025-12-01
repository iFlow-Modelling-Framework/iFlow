"""
Numerical method for Flow MTS_2DV
Prepare a regular temporal grid for a MTS_2DV

Date: 26-02-2019
Author: D.D.Bouwman

Based on "RegularGrid.py" by Y.M.Dijkstra
"""
from nifty import makeRegularGrid


class GridTime:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Prepare a regular grid for the time dependent Multiple-Timescale computations

        Returns:
            Dictionary containing:
                grid:   dimensions (list) - list [x,z,f,t]
                        gridType (str) - 'Regular'
                        axis: x,z,f,t (ndarray - grid axes between 0 and 1
                        high: x,z,f,t (any) - dimension-full limit belonging to point 1 on the dimensionless axis
                        low: x,z,f,t (any) - dimension-full limit belonging to point 0 on the dimensionless axis
                        maxIndex: x,z,f,t (int) - maximum index number
                outgrid:    idem
        """
        d = {}

        dimensions = ['x', 'z', 'f', 't']
        enclosures = [(0, self.input.v('L')),
                      (self.input.v('R'), self.input.n('H')),
                      None,
                      (self.input.v('startDay'), self.input.v('endDay'))]
        contraction = [[], ['x'], [], []]     # Enclosures of each dimension depend on these parameters
        copy = [1, 1, 0, 1]                  # Copy lower-dimensional arrays over these dimensions. 1: yes, copy. 0: no, only keep in the zero-index

        # grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xgrid', 'zgrid', 'fgrid', 'tgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['MTS_grid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)

        # output grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xoutputgrid', 'zoutputgrid', 'foutputgrid', 'toutputgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['MTS_outputgrid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)
        d['__outputGrid'] = {}
        d['__outputGrid']['MTS_grid'] = 'MTS_outputgrid'

        return d


