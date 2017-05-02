"""
HinderedSettling

Date: 11-Nov-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
import logging


class HinderedSettling:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 0.10#0.05 #10**-2.
    RELAX = 0.5     # percentage of old ws


    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        stop = False
        if hasattr(self, 'difference'):
            self.logger.info('\t'+str(self.difference))
            if self.difference < self.TOLLERANCE*self.RELAX:
                stop = True
        return stop

    def run_init(self):
        self.logger.info('Running module HinderedSettling - init')
        d = {}
        self.difference = np.inf
        if self.input.v('ws0') is not None:
            # d['ws0'] = self.input.v('ws0')    # TODO check
            d['ws0'] = self.input.v('ws00')
        else:
            d['ws0'] = self.input.v('ws00')
        return d

    def run(self):
        self.logger.info('Running module HinderedSettling')
        d = {}
        # Load data
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c = self.input.v('c0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        #c += self.input.v('c1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # c[:, :, 1:] = 0     # TODO
        cgel = self.input.v('cgel')
        mhs = self.input.v('mhs')
        ws0 = self.input.v('ws00')
        wsmin = self.input.v('wsmin')

        # convert to time series
        c = np.concatenate((c, np.zeros((jmax+1, kmax+1, 100))), 2)
        c = ny.invfft(c, 2)

        # Richardson & Zaki 1954 formulation
        phi = c/cgel
        phi = np.maximum(np.minimum(phi, 1), 0)
        ws = np.maximum(ws0*(1.-phi)**mhs, wsmin)
        ws = ny.fft(ws, 2)
        ws = ws[:, :, :fmax+1]
        ws[:, :, 1:] = 0

        # if np.any(ws[:, :, 0] < ws0/100.):
        #     self.logger.warning('Subtidal settling velocity computed by hindered settling drops below its minimum 1% of ws0.\n'
        #                         'The minimum is automatically corrected to 1% of ws0.')
        #     ws[:, :, 0] = np.maximum(ws[:, :, 0], ws0/100.)       # set minimum fall velocity at 1% (arbitrarily chosen).
        #     self.difference = 0
        # else:
        #     ## difference

        ws_old = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        d['ws0'] = (1-self.RELAX)*ws + self.RELAX*ws_old
        # d['ws0'][1:-1] = 0.5*d['ws0'][1:-1]+0.5*(0.5*d['ws0'][2:]+0.5*d['ws0'][:-2])
        self.difference = np.linalg.norm(np.sum(np.abs((ws_old-d['ws0'])/((ws_old+0.1*self.input.v('ws00'))[:, :, [0]])), axis=-1), np.inf)
        # from plot_paper.boundary_solver import boundary_solver
        # from erosion import erosion
        # import scipy.optimize
        # import step as st
        # import matplotlib.pyplot as plt
        #
        # bs = boundary_solver()
        # phi_vec = np.linspace(0, 1, 100)
        # ws_vec = self.input.v('ws00')*(1.-phi_vec)**self.input.v('mhs')
        # Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # sol=[]
        # for phi in phi_vec:
        #     sol.append(scipy.optimize.fsolve(bs.boundary_switch, 0.3, (phi, 'RZ', 5, False, 0)))
        # Etil = erosion(ws, Av, 0, self.input, method='Partheniades')/(self.input.v('cgel')*self.input.v('ws00'))
        # f = self.input.v('f', range(0, jmax+1))
        #
        # st.configure()
        # plt.figure(1, figsize=(1,2))
        # plt.subplot(1,2,1)
        # plt.plot(sol, ws_vec)
        # plt.plot(Etil[:, -1, 0]*f, ws_old[:, -1, 0], 'g-o')
        # plt.plot(Etil[:, -1, 0]*f, ws[:, -1, 0], 'r.')
        # plt.xlabel(r'$\tilde{E}$')
        # plt.ylabel(r'$w_s$')
        # plt.subplot(1,2,2)
        # plt.plot(abs(ws_old[:, -1, 0]))
        # st.show()





        # d['converged'] = self.iteration>1  or self.difference < self.TOLLERANCE #
        # M = raw_input('M:')
        # d['finf'] = M
        # if self.iteration == 10:
        #     print self.input.v('f')
        #
        # if self.iteration>20:
        #     import step as st
        #     import matplotlib.pyplot as plt
        #     st.configure()
        #     plt.figure(1, figsize=(1,1))
        #     plt.plot(np.real(d['ws0'][:, -1, 0])/self.input.v('ws00'))
        #     plt.plot(self.input.v('f'))
        #     plt.plot(np.real(self.input.v('c0', range(0, jmax+1), kmax, 0))/np.max(np.real(self.input.v('c0', range(0, jmax+1), kmax, 0))))
        #     st.show()

        return d
