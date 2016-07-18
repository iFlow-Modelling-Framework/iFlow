"""
TurbulenceKepFitted_1

Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFittedUnscaled import TurbulenceKepFittedUnscaled
import numpy as np
from nifty.functionTemplates import FunctionBase
import nifty as ny


class TurbulenceKepFittedUnscaledReference(TurbulenceKepFittedUnscaled):
    # Variables
    RELAX = 0.5
    TOLLERANCE = 2.*1e-2*RELAX       # relative change allowed for converged result
    order = None                     # no ordering

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFittedUnscaled.__init__(self, input, submodulesToRun)
        return

    def run_init(self):

        # reference level
        d = {}
        R = initialRef('x', self.input)
        d['R'] = R.function
        self.input.merge(d)

        # grid for initial eddy viscosity
        grid = {}
        oldgrid = self.input.slice('grid')

        # Grid for initial run
        dimensions = ['x', 'z', 'f']
        enclosures = [(0, self.input.v('L')),
                      (self.input.v('R'), self.input.v('-H')),
                      None]
        contraction = [[], ['x'], []]     # enclosures of each dimension depend on these parameters

        # grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xgrid', 'zgrid', 'fgrid']
        gridsize = {}
        gridsize['xgrid'] = ['equidistant', 100]
        gridsize['zgrid'] = ['equidistant', 100]
        gridsize['fgrid'] = ['integer', 2]
        for var in variableNames:
            axisTypes.append(gridsize[var][0])
            axisSize.append(gridsize[var][1])
            axisOther.append(gridsize[var][2:])
        grid['grid'] = ny.makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction)
        self.input.merge(grid)

        dresult = TurbulenceKepFittedUnscaled.run_init(self)
        d.update(dresult)

        # restore old grid
        self.input.merge(oldgrid)

        return d

    def run(self):
        d = {}
        jmax = self.input.v('grid', 'maxIndex', 'x')
        d['R'] = 0.1*(self.input.v('zeta1', 'river', range(0, jmax+1), 0, 0) + self.input.v('R', range(0, jmax+1))) + 0.9*self.input.v('R', range(0, jmax+1))
        d['R'] = np.maximum(d['R'], 1+self.input.n('H', range(0, jmax+1)))
        self.input.addData('R', d['R'])
        import matplotlib.pyplot as plt
        plt.plot(d['R'])
        plt.plot(self.input.n('H', range(0, jmax+1)))
        plt.show()


        dnew = TurbulenceKepFittedUnscaled.run(self)
        d.update(dnew)
        return d

class initialRef(FunctionBase):
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.H = data
        return

    def value(self, x, **kwargs):
        return np.maximum(self.H.n('H', x=x)+2, 0)

    def derivative(self, x, **kwargs):
        der = -self.H.d('H', x=x, dim='x')
        der[np.where(self.H.n('H', x=x)+2 < 0)] = 0
        return der