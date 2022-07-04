"""
Plot model SPM modeled distribution including flocculation processes (Fig. 5a), the corresponding distribution of the \
settling velocity (Fig. 5c), and transport capacity mechanisms (Fig. 6a).

Date: 17-Aug-20
Authors: D.M.L. Horemans
"""
import numpy as np
from step import Step
import step as st
import nifty as ny

class Plot_base_model:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):

        kmax = self.input.v('grid', 'maxIndex', 'z')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        smoothing = True

        st.configure()
        step = Step.Step(self.input)

        ############### Plot modeled SPM distribution ################
        if smoothing:
            c0_tmp = self.input.v('c0', range(0, jmax+1), range(0, kmax+1))
            c1_tmp = self.input.v('c1', range(0, jmax+1), range(0, kmax+1))
            c2_tmp = self.input.v('c2', range(0, jmax+1), range(0, kmax+1))

            ord = 1  # order of smoothing
            xstart = 0  # start smoothing from longitudinal x-axis value.
            for zi in range(0, kmax + 1):
                c0_tmp[xstart:-1, zi, 0] = ny.savitzky_golay(c0_tmp[xstart:-1, zi, 0], window_size=7, order=ord)#.reshape(c0_tmp.shape[0]-1, 1)
                c1_tmp[xstart:-1, zi, 0] = ny.savitzky_golay(c1_tmp[xstart:-1, zi, 0], window_size=7, order=ord)#.reshape(c1_tmp.shape[0]-1, 1)
                c2_tmp[xstart:-1, zi, 0] = ny.savitzky_golay(c2_tmp[xstart:-1, zi, 0], window_size=7, order=ord)#.reshape(c2_tmp.shape[0]-1, 1)

            self.input.addData('csubt', (c0_tmp + c1_tmp + c2_tmp) * 1000)

            self.input.addData('c0', c0_tmp)
            self.input.addData('c1', c1_tmp)
            self.input.addData('c2', c2_tmp)

        else:
            self.input.addData('csubt', (self.input.v('c0') + self.input.v('c1') + self.input.v('c2')) * 1000)

        step.contourplot('x', 'z', 'csubt', f=0, sublevel=False, operation=np.abs, plotno=1)
        st.show()

        ############### Plot modeled settling velocity distribution ################
        self.input.addData('ws0ms', self.input.v('ws0') * 1000)
        step.contourplot('x', 'z', 'ws0ms', f=0, sublevel=False, operation=np.abs, plotno=2)
        st.show()

        ############### Plot transport capacity mechanisms ###############
        step.transportplot_mechanisms(sublevel='sublevel', concentration=True, plotno=3, display=7, legend='out',
                                      capacity=True)
        st.show()
        d = {}
        return d
