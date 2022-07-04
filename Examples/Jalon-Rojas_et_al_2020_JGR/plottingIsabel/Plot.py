"""
PlotHydro

Date: 27-05-16
Authors: R.L. Brouwer, Y.M. Dijkstra
"""
import matplotlib.pyplot as plt
import numpy as np
from step import Step
import step as st
import logging


class Plot:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Module Plot is running')

        st.configure()

        # load the STeP module
        step = Step.Step(self.input)

        ## LINEPLOT ##
        # lineplot(var1, var2, options)
        # Parameters:
        #   var1: (string) name of the variable on the horizontal axis
        #   var2: (string) name of the variable on the vertical axis
        #       either var1 or var2 should be a grid axis (x, z, f)
        #   options include:
        #       x, z, f: (float, list or 1D array) values of the other axes; either one value or a range
        #       operation: (function) any operation on the data. Simple examples are abs or np.real. This can also be a function
        #           reference doing more complex operations
        #       sublevel: (bool) use decomposition of this variable (e.g. for u, zeta, c) in its submodules
        #       subplots: (string) If this option is provided, subplots will be generated varying the variable provided.
        #            This can be 'x', 'z', 'f', or 'sublevel'
        #       plotno: number of this figure
        #   See some examples below

        # step.lineplot('x', 'hatc1', 'a', z=0, f=range(0,3), sublevel=True, subplots='sublevel', operation=np.abs, plotno=1)
        # step.lineplot('x', 'hatc1', 'a', 'erosion', z=0, f=range(0,3), sublevel=True, subplots='sublevel', operation=np.abs, plotno=2)
        # step.lineplot('x', 'zeta1', z=0, f=range(0,3), sublevel=True, subplots='sublevel', operation=np.abs, plotno=2)
        # step.lineplot('x', 'B', z=0, plotno=3)
        # step.lineplot('x', 'zeta0', z=0, f=1, sublevel=True, operation=np.angle, plotno=4)
        # step.lineplot('x', 'Av', z=0, f=0, sublevel=False, operation=np.abs, plotno=5)
        # step.lineplot('x', 'zeta1', z=0, f=range(0, 3), sublevel=False, subplots='f', operation=np.abs, plotno=6)

        ## CONTOURPLOT ##
        # contourplot(var1, var2, var3, options)
        # Parameters:
        #   var1: (string) name of the variable on the horizontal axis, should be grid axis
        #   var2: (string) name of the variable on the vertical axis, should be grid axis
        #   var3: (string) name of the variable for the contour levels
        #   options: see lineplot
        #
        #   See some examples below
        # step.contourplot('x', 'z', 'u0', f=1, subplots='f', operation=np.abs, plotno=10)
        # step.contourplot('x', 'z', 'hatc', f=0, sublevel=True, subplots='sublevel', operation=np.real, plotno=11)
        step.contourplot('x', 'z', 'c0', f=0, sublevel=True, subplots='sublevel', operation=np.abs, plotno=12)

        ## TRANSPORTPLOT ##
        # transportplot_mechanisms(options)
        # Plots the sediment transport mechanisms
        # Parameters:
        #   sublevel: (string) can be 'sublevel' or 'subsublevel' indicating to take the first level decomposition or second (deeper) decomposition level
        #   concentration: (bool) plot subtidal leading-order concentration in the background
        #   display: (int) number of decompositon components to plot. Will select the most important contributions by their RMS value
        #   plotno: (int) number of this figure
        # step.transportplot_mechanisms(sublevel='sublevel', concentration=True, plotno=20, legend='out', display=7)

        # plt.figure(100, figsize=(1,2))
        # Tk = self.input.getKeysOf('T')
        # jmax = self.input.v('grid', 'maxIndex', 'x')
        # x = self.input.v('grid', 'axis', 'x')
        # for key in Tk:
        #     T = self.input.v('T', key, range(0, jmax+1))
        #     plt.plot(x, T, label=key)
        # plt.legend()

        st.show()
        return