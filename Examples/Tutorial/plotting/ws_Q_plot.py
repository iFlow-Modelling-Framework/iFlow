"""
ETMPlot

Date: 10-Jun-16
Authors: Y.M. Dijkstra
"""
from scipy.signal import argrelextrema
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import step as st
import nifty as ny


class ws_Q_plot:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        st.configure()
        data = self.input.getKeysOf('experimentdata')
        var1 = 'ws0'
        var2 = 'Q1'

        v2 = []
        v1 = []
        for dat in data:
            dc = self.input.v('experimentdata', dat)
            v1.append(dc.v(var1))
            v2.append(dc.v(var2))
        v1 = list(set(v1))
        v2 = list(set(v2))
        v1.sort()
        v2.sort()
        x = dc.v('grid', 'axis', 'x')
        xdim = ny.dimensionalAxis(dc.slice('grid'), 'x')[:,0,0]

        ETMloc = np.nan*np.ones((len(v1), len(v2)))
        maxc = np.nan*np.ones((len(v1), len(v2)))
        for q, dat in enumerate(data):
            print(q)

            dc = self.input.v('experimentdata', dat)
            i = v1.index(dc.v(var1))
            j = v2.index(dc.v(var2))

            c = dc.v('c', x=x, z=1, f=0)
            if c is None:
                c = dc.v('c0', x=x, z=1, f=0) + dc.v('c2', x=x, z=1, f=0)
            maxc[i, j] = np.max(c)

            try:
                # loc = ny.toList(self.findETM(xdim, T, Tx))
                loc = ny.toList(self.findETM(xdim, c))
                ETMloc[i,j] = np.asarray(loc)
            except:
                print('no etm')


        plt.figure(1, figsize=(1,1))
        log1 = 'True'
        log2 = 'False'
        if log1=='True':
            v1 = np.log10(np.asarray(v1))
        if log2=='True':
            v2 = np.log10(np.asarray(v2))

        x_up_round = np.ceil(max(xdim)/10000.)*10.
        ETMloc[np.where(ETMloc>=x_up_round*1000.)] = x_up_round*1000.
        levels = np.linspace(0., x_up_round, int(x_up_round/2+1))
        plt.contourf(np.asarray(v1), np.asarray(v2), ETMloc.T/1000., levels=levels)
        plt.title('ETM location (km)')
        if log1 == 'True':
            xmin = np.min(np.asarray(v1))
            xmax = np.max(np.asarray(v1))
            xtick = np.arange(np.ceil(xmin),np.floor(xmax)+1)
            plt.xticks(xtick, [10.**i for i in xtick])
            plt.xlabel('$\log_{10} w_s$ (m/s)')

        plt.ylabel('Q $(m^3/s)$')

        plt.colorbar()

        # ## Fig 2
        # plt.figure(2, figsize=(1,1))
        # plt.contourf(np.asarray(v1), np.asarray(v2), maxc.T/1000., 40)
        # plt.title('Maximum near-bed\nconcentration (g/l)')
        # if log1 == 'True':
        #     xmin = np.min(np.asarray(v1))
        #     xmax = np.max(np.asarray(v1))
        #     xtick = np.arange(np.ceil(xmin),np.floor(xmax)+1)
        #     plt.xticks(xtick, [10.**i for i in xtick])
        #     plt.xlabel('$\log_{10} w_s$ (m/s)')
        #
        # plt.ylabel('Q $(m^3/s)$')

        # plt.colorbar()

        st.show()
        d = {}
        return d

    def findETM(self, x, c):
        roots = np.argmax(c)
        return x[roots]