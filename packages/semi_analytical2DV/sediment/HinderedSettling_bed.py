"""
HinderedSettling_DA
Depth-averaged time-averaged hindered settling

Date: 24-08-2017
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
import logging


class HinderedSettling_bed:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 1.e-3  # DEFAULT VALUE USED IN THE EMS PAPER  #1.e-3  #
    RELAX = 0.8         # percentage of old ws
    local = 0.5

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration

        stop = False
        if hasattr(self, 'difference'):
            self.logger.info('\t'+str(self.difference))
            if self.difference < self.TOLLERANCE*(1-self.RELAX):
                stop = True
        return stop

    def run_init(self):
        self.logger.info('Running module HinderedSettling - init')
        self.difference = np.inf
        d = {}

        if self.input.v('ws0') is None or self.input.v('grid', 'maxIndex', 'x') is None:
            d['ws0'] = self.input.v('ws00')
        else:
            jmax = self.input.v('grid', 'maxIndex', 'x')
            d['ws0'] = self.input.v('ws0', range(0, jmax+1))
        return d

    def run(self):
        self.logger.info('Running module HinderedSettling')
        d = {}

        ################################################################################################################
        # Load data
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        cgel = self.input.v('cgel')
        mhs = self.input.v('mhs')
        ws0 = self.input.v('ws00')
        wsmin = self.input.v('wsmin')

        c = np.real(self.input.v('c0', range(0, jmax+1), range(0, kmax+1), 0)) + np.real(self.input.v('c2', range(0, jmax+1), range(0, kmax+1), 0))      # only subtidal
        phi = np.max(c, axis=1)/cgel

        ################################################################################################################
        # Computations
        ################################################################################################################
        # Richardson & Zaki 1954 formulation
        phi = np.maximum(np.minimum(phi, 1), 0)
        ws = np.maximum(ws0*(1.-phi)**mhs, wsmin)

        ws_old = self.input.v('ws0', range(0, jmax+1), 0, 0)
        ws_new = (1-self.RELAX)*ws + self.RELAX*ws_old          # Relaxation over the iterands
        dws = ws_new-ws_old

        ################################################################################################################
        # Smoothing of increments in space
        ################################################################################################################
        # smooth by averaging with neighbours twice
        dws[1:-1] = self.local*dws[1:-1]+(1-self.local)*(0.5*dws[2:]+0.5*dws[:-2])
        dws[1:-1] = self.local*dws[1:-1]+(1-self.local)*(0.5*dws[2:]+0.5*dws[:-2])

        # final result and computation of change in this iteration
        ws = ws_old + dws
        ws = ny.savitzky_golay(ws, 7, 1)
        nf = ny.functionTemplates.NumericalFunctionWrapper(ws, self.input.slice('grid'))
        nf.addDerivative(ny.savitzky_golay(ny.derivative(ws, 'x', self.input), 7, 1), 'x')
        nf.addDerivative(ny.savitzky_golay(ny.secondDerivative(ws, 'x', self.input), 7, 1), 'xx')

        d['ws0'] =  nf.function
        self.difference = np.max(np.abs((ws_old-ws)/(ws_old)))
        return d
