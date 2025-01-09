"""
Numerical method for Flow 2DV
Prepare a regular grid for a 2DV model

Date: 21-07-15
Authors: Y.M. Dijkstra
"""
from nifty import makeRegularGrid


class RegularGrid:
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
        d = {}

        dimensions = ['x', 'z', 'f']
        enclosures = [(0, self.input.v('L')),
                      (self.input.v('R'), self.input.n('H')),
                      None]
        contraction = [[], ['x'], []]     # Enclosures of each dimension depend on these parameters
        copy = [1, 1, 0]                  # Copy lower-dimensional arrays over these dimensions. 1: yes, copy. 0: no, only keep in the zero-index

        # grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xgrid', 'zgrid', 'fgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['grid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)

        # output grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xoutputgrid', 'zoutputgrid', 'foutputgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['outputgrid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)
        d['__outputGrid'] = {}
        d['__outputGrid']['grid'] = 'outputgrid'
        return d


