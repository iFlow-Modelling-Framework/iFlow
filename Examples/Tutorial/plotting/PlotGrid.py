"""
Test

Date: 19-Oct-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import step as st
import matplotlib.pyplot as plt
import nifty as ny


class PlotGrid:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):

        st.configure()

        z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:,:,0]
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,:,0]

        print 'maximum depth: ' + str(self.input.v('H', x=0))
        print 'minimum depth: ' + str(self.input.v('H', x=1))
        print 'maximum width: ' + str(self.input.v('B', x=0))
        print 'minimum width: ' + str(self.input.v('B', x=1))

        plt.figure(1, figsize=(1,2))
        for i in range(0, z.shape[1]):
            plt.plot(x[:,0]/1000., z[:,i], 'k-')

        for i in range(0, x.shape[0]):
            plt.plot(x[i,:]/1000., z[i,:], 'k-')
        plt.xlabel('x (km)')
        plt.ylabel('z (m)')
        plt.xlim(0, np.max(x[:,0])/1000.)
        plt.ylim(np.min(z), np.max(z))

        st.show()

        # twoxu1 = 2*u1
        d = {}

        return d