"""
HinderedSettling

Date: 11-Nov-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny


class HinderedSettling:
    # Variables
    TOLLERANCE = 10**-3.

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        stop = False
        if hasattr(self, 'difference'):
            print self.difference
            if self.difference < self.TOLLERANCE:
                stop = True
        return stop

    def run_init(self):
        d = {}
        if self.input.v('c0') is not None and self.input.v('c1') is not None:
            d = self.run()
        else:
            jmax = self.input.v('grid', 'maxIndex', 'x')
            kmax = self.input.v('grid', 'maxIndex', 'z')
            fmax = self.input.v('grid', 'maxIndex', 'f')

            ws = np.zeros((1, 1, fmax+1))
            ws[:, :, 0] = self.input.v('ws0')

            d['ws'] = ws
        return d

    def run(self):
        d = {}
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c = self.input.v('c0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1)) + self.input.v('c1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        cgel = self.input.v('cgel')
        ws0 = self.input.v('ws0')
        # rhos = self.input.v('RHOS')
        phi = c/cgel
        phi = np.concatenate((phi, (np.zeros((jmax+1, kmax+1, fmax+1)))), 2)
        I = np.zeros(phi.shape)
        I[:, :, 0] = 1.
        # phip = c/rhos
        mhs = self.input.v('mhs')

        ws = ws0*self.umultiply(mhs, 0, [(I-phi)])        #*(1-phip)/(1+2.5*phi)
        ws = ws[:, :, :fmax+1]
        d['ws'] = ws

        ## difference
        ws_old = self.input.v('ws', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        self.difference = np.linalg.norm(np.sum(np.abs((ws_old-ws)/(ws_old+0.01*ws0)), axis=-1), np.inf)

        return d

    def umultiply(self, pow, N, u):
        """ Compute the sum of all possible combinations yielding the power 'pow' of signal 'u' with a total order 'N'
        i.e. (u^pow)^<N>
        """
        v = 0
        if pow>2:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(self.umultiply(2, i, u), self.umultiply(pow-2, N-i, u), 2)
        else:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(u[i], u[N-i], 2)
        return v