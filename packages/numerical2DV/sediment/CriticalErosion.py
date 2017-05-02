"""
CriticalErosion

Date: 07-03-2017
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunctionSingle import cFunctionSingle
from copy import copy
from copy import deepcopy

class CriticalErosion:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module CriticalErosion')

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])  # assume it is constant in time; else division with AV fails
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        # ws = np.zeros((jmax+1, kmax+1, fmax+1))
        cgel = self.input.v('cgel')
        mhs = self.input.v('mhs')
        ws0 = self.input.v('ws00')

        taub = self.taub(Av, 0)
        taub += self.taub(Av, 1)

        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([1, kmax+1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([1, 1, ftot, 1], dtype=complex)
        Ecrit = np.nan*np.zeros(jmax+1)

        jlist = [int(np.floor(i)) for i in np.linspace(0, jmax, 6)]

        # x = self.input.v('grid', 'axis', 'x', jlist, 0, 0)
        x = self.input.v('grid', 'axis', 'x', np.asarray(jlist))
        newgrid = deepcopy(self.input.slice('grid'))
        newgrid.merge({'grid':{'axis':{'x':x}, 'maxIndex':{'x':len(x)-1}}})

        for j in jlist:
            found = False

            Etil_ran = [0, 0.15]
            while found == False:
                # forcing
                Etil = 0.5*(Etil_ran[0]+Etil_ran[1])
                E = Etil*cgel*ws0/taub[j, 0, 0]
                Fbed[:, :, fmax:, 0] = -E*taub[[j], Ellipsis]

                ws_old = ws0
                ws = self.input.v('ws00', [0], range(0, kmax+1), range(0, fmax+1))
                difference = np.inf
                while difference > 10**-2:
                    ################################################################################################################
                    # Solve equation
                    ################################################################################################################
                    c, cMatrix = cFunctionSingle(ws, Kv[[j], Ellipsis], F, Fsurf, Fbed, self.input, hasMatrix = False)
                    c = c.reshape((1, kmax+1, ftot))
                    c = ny.eliminateNegativeFourier(c, 2)

                    ################################################################################################################
                    # Fall velocity
                    ################################################################################################################
                    # convert to time series
                    c = np.concatenate((c, np.zeros((1, kmax+1, 100))), 2)
                    c = ny.invfft(c, 2)

                    # Richardson & Zaki 1954 formulation
                    phi = c/cgel
                    ws = np.maximum(ws0*(1.-phi)**mhs, 0)
                    ws = ny.fft(ws, 2)
                    ws = ws[:, :, :fmax+1]
                    if np.any(ws[:, :, 0] < 10**-6):
                        difference = 0
                    else:
                        ## difference
                        difference = np.linalg.norm(np.sum(np.abs((ws-ws_old)/(ws_old+0.01*ws0)), axis=-1), np.inf)
                        ws_old = copy(ws)

                # set new Etil
                if np.any(ws[:, :, 0] < 10**-6):
                    Etil_ran[1] = Etil
                else:
                    Etil_ran[0] = Etil

                if Etil_ran[1]-Etil_ran[0] < 2*10**-4:
                    found = True

            # process
            self.logger.info('\tProgress: ' + str(j) +' of ' + str(jmax))
            Ecrit[j] = Etil
        Ecrit = Ecrit[jlist]
        # import step as st
        # import matplotlib.pyplot as plt
        # st.configure()
        # plt.figure(1, figsize=(1,1))
        # plt.plot(range(0, jmax+1), Ecrit, '.')
        #
        #
        # st.show()
        # load to dict
        d = {}
        nf = ny.functionTemplates.NumericalFunctionWrapper(Ecrit, newgrid)
        d['Ecrit'] = nf.function

        return d

    def taub(self, Av, tau_order):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        ## 1. bed shear stress
        # the bed shear stress is extended over fmax+1 frequency components to prevent inaccuracies in truncation
        taub = []
        Av = np.concatenate((Av, np.zeros((jmax+1, kmax+1, fmax+1))), 2)
        for i in range(0, tau_order+1):
            uz = self.input.d('u'+str(i), range(0, jmax+1), [kmax], range(0, fmax+1), dim='z')
            uz = np.concatenate((uz, np.zeros((jmax+1, 1, fmax+1))),2)
            taub.append(ny.complexAmplitudeProduct(Av[:, [kmax], :], uz, 2))

        # amplitude
        tau_amp = (np.sum(np.abs(sum(taub)), axis=-1)+10**-3).reshape((jmax+1, 1, 1))
        taub = [t/tau_amp for t in taub]

        # absolute value
        c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs
        taub_abs = np.zeros(taub[0].shape, dtype=complex)
        if tau_order==0:
            taub_abs[:, :, 0] = c[0]
        taub_abs += c[2]*self.umultiply(2, tau_order, taub)
        taub_abs += c[4]*self.umultiply(4, tau_order, taub)
        taub_abs += c[6]*self.umultiply(6, tau_order, taub)
        taub_abs += c[8]*self.umultiply(8, tau_order, taub)

        taub_abs = taub_abs*tau_amp

        return taub_abs[:, :, :fmax+1]

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