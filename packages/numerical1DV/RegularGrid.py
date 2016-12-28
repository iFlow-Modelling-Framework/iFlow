"""
Numerical method for Flow 1DV
Prepare a regular grid for a 1DV model

Date: 21-07-15
Authors: Y.M. Dijkstra
"""
from nifty import makeRegularGrid


class RegularGrid:
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        """Prepare a regular grid for a numerical 2DV computation

        Returns:
            Dictionary containing:
                grid:   dimensions (list) - list [z,f]
                        gridType (str) - 'Regular'
                        axis: z,f (ndarray - grid axes between 0 and 1
                        high: z,f (any) - dimension-full limit belonging to point 1 on the dimensionless axis
                        low: z,f (any) - dimension-full limit belonging to point 0 on the dimensionless axis
                        maxIndex: z,f (int) - maximum index number
                outgrid:    idem
        """
        d = {}

        dimensions = ['z', 'f']
        enclosures = [(0, -self.input.v('H0')),
                      None]

        # grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['zgrid', 'fgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['grid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures)

        # output grid
        axisTypes = []
        axisSize = []
        variableNames = ['zoutputgrid', 'foutputgrid']
        for var in variableNames:
            axisTypes.append(self.input.v(var)[0])
            axisSize.append(self.input.v(var)[1])
            axisOther.append(self.input.v(var)[2:])
        d['outputgrid'] = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures)

        # depth
        d['H'] = self.input.v('H0')
        return d


