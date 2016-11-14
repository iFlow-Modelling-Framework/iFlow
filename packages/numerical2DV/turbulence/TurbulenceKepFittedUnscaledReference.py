"""
TurbulenceKepFitted_1

Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFittedUnscaled import TurbulenceKepFittedUnscaled
import nifty as ny
from ..hydro.ReferenceLevel import ReferenceLevel


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
        self.RL = ReferenceLevel(self.input, [])
        d.update(self.RL.run_init())
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

        dresult = TurbulenceKepFittedUnscaled.run_init(self, init=True)
        d.update(dresult)

        # restore old grid
        self.input.merge(oldgrid)

        return d

    def run(self):
        d = {}
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        # run reference level
        d.update(self.RL.run())

        # trick riverine water level for computation
        zetariv0 = self.input.v('zeta0', 'river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zetariv1 = self.input.v('zeta1', 'river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        if zetariv0 is not None:
            self.input.merge({'zeta0':{'river':0*zetariv0}})     # temporarily delete zetariv to prevent oscillation of numerical iteration
        if zetariv1 is not None:
            self.input.merge({'zeta1':{'river':0*zetariv1}})     # temporarily delete zetariv to prevent oscillation of numerical iteration

        dnew = TurbulenceKepFittedUnscaled.run(self)
        d.update(dnew)

        if zetariv0 is not None:
            self.input.merge({'zeta0':{'river':zetariv0}})       # place zetariv back
        if zetariv1 is not None:
            self.input.merge({'zeta1':{'river':zetariv1}})       # place zetariv back

        # import matplotlib.pyplot as plt
        # import step as st
        # st.configure()
        # plt.figure(1,figsize=(2,2))
        # plt.subplot(2,2,1)
        # plt.plot(d['R'], 'b')
        # plt.plot(self.input.v('R', range(0, jmax+1)), 'b--')
        # # plt.plot(d['R']+self.input.v('zeta1', range(0, jmax+1),0,0), 'b--')
        # plt.plot(self.input.n('H', range(0, jmax+1)))
        # plt.subplot(2,2,2)
        # try:
        #     plt.plot(self.input.v('Av', range(0, jmax+1),0,0), 'b--')
        #     self.input.addData('Av', d['Av'])
        #     plt.plot(self.input.v('Av', range(0, jmax+1),0,0), 'b')
        # except:
        #     pass
        # plt.subplot(2,2,3)
        # try:
        #     plt.plot(self.input.v('zeta1', 'river', range(0, jmax+1),0,0))
        #     plt.plot(self.input.v('zeta1', range(0, jmax+1),0,0))
        # except:
        #     pass
        # st.show()

        return d
