"""
PlotSensitivity

Date: 26-Oct-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import step as st
import matplotlib.pyplot as plt
import nifty as ny


class PlotSensitivity:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input

        return

    def run(self):
        files = self.input.getKeysOf('experimentdata') # get list of file names

        # initiate variables
        data_dummy = self.input.v('experimentdata', files[0])
        jmax = data_dummy.v('grid', 'maxIndex', 'x')        # get grid dimensions from one random file
        kmax = data_dummy.v('grid', 'maxIndex', 'z')
        fmax = data_dummy.v('grid', 'maxIndex', 'f')
        zeta1 = np.zeros((jmax+1, 1, fmax+1, len(files)), dtype=complex)    # set empty variable
        Q1 = np.zeros(len(files))
        x = ny.dimensionalAxis(data_dummy.slice('grid'), 'x')[:,0,0]

        # load data from each file
        for i, file in enumerate(files):
            data = self.input.v('experimentdata', file)
            zeta1[:,0,:, i] = data.v('zeta1', range(0, jmax+1), 0, range(0, fmax+1))
            Q1[i] = data.v('Q1')

        # plot
        st.configure()
        plt.figure(1, figsize=(1,2))
        for i, file in enumerate(files):
            plt.plot(x/1000., np.abs(zeta1[:,0,0,i]))
        plt.xlabel('x (km)')
        plt.ylabel('$\zeta^1$ (m)')
        plt.title('First-order subtidal water level amplitude for different Q')
        plt.legend(Q1)

        plt.figure(2, figsize=(1,2))
        for i, file in enumerate(files):
            plt.plot(Q1[i], np.abs(zeta1[-1,0,0,i]), 'ko')
        plt.xlabel('Q ($m^3/s$)')
        plt.ylabel('$\zeta^1(L)$ (m)')
        plt.title('First-order subtidal water level amplitude at the upstream boundary for different Q')

        st.show()

        return {}