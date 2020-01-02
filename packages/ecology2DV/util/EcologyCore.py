"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
TODO: dynamically adjust time step and end time to equilibrium and growth rates
"""
import logging
import numpy as np
import nifty as ny
from .pFunction_subtidal import pFunction
from .cSolverTime import cSolverTime
from src.util.mergeDicts import mergeDicts
from copy import copy


class EcologyCore:
    # Variables
    logger = logging.getLogger(__name__)
    time = [ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer()]
    DT = 24*3600.*10#0.05

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def main(self, components, ws, Kv, Kh, Csea, QC, S, Gamma, Gamma_factor, source):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        taumax = self.input.v('taumax') or 1

        if False:#hasattr(self, 'X'):       #DEBUG
            X = self.X
            spinup = False
        else:
            X = np.zeros((jmax+1, len(components)))
            spinup = True

        # time integration and initial condition
        Chat_prim = np.zeros((jmax+1, kmax+1, len(components)))
        Chat_int = np.zeros((jmax+1, len(components)))
        Ts = np.zeros((jmax+1, len(components)))
        Fs = np.zeros((jmax+1, len(components)))
        Hs = np.zeros((jmax+1, len(components)))
        Chat = np.zeros((jmax+1, kmax+1, len(components)))

        ################################################################################################################
        # Compute transport rates & potential concentrations
        ################################################################################################################
        dc = self.input.slice('grid')
        for i, comp_name in enumerate(components):
            T, F, Hs[:, i], Chat[:, :, i] = self.component(ws[i], Kv[i], Kh[i])
            dc.merge({comp_name: {'T': T}})
            dc.merge({comp_name: {'F': F}})
            Ts[:, i] = dc.v(comp_name, 'T', range(0, jmax+1))
            Fs[:, i] = dc.v(comp_name, 'F', range(0, jmax+1))

            # integrals
            Chat_prim[:, :, i] = -ny.primitive(np.real(Chat[:, :, i]), 'z', 0, kmax, self.input.slice('grid'))   # primitive counted from surface
            Chat_int[:, i] = np.sum(Chat_prim[:, :, i], axis=1)

        ## process growth function so that no duplicate computations are done
        Gamma_flat = [item for sublist in Gamma for item in sublist]
        Gamma_list = list(set(Gamma_flat))
        Gamma_index = [[Gamma_list.index(i) for i in j] for j in Gamma]

        ################################################################################################################
        # Initial conditions
        ################################################################################################################
        if spinup:
            for i, comp_name in enumerate(components):
                if i==0:
                    P0 = Csea[i]*np.exp(-10*np.linspace(0, 1, jmax+1))
                    H = self.input.v('grid', 'low', 'z', range(0, jmax+1)) - self.input.v('grid', 'high', 'z', range(0, jmax+1))
                    X[:, i] = P0/np.real(ny.integrate(Chat[:, :, i], 'z', kmax, 0, self.input.slice('grid'))[:, 0])*H
                else:
                    # initial condition (equilibrium without growth)
                    integrand = ny.integrate(Ts[:, i]/(Fs[:, i]), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
                    P = QC[i]/(Fs[:, i])*np.exp(integrand.reshape((jmax+1, 1))*np.ones((1, jmax+1))-integrand.reshape((1, jmax+1)))
                    Pint = ny.integrate(P.reshape((jmax+1, 1, 1, jmax+1)), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
                    Pint = Pint[range(0, jmax+1), 0, 0, range(0, jmax+1)]
                    C0 = np.exp(-integrand)*Csea[i] + Pint
                    X[:, i] = C0/Chat_int[:, i]*H

            self.Xprev = np.ones(X.shape)*np.inf

        ################################################################################################################
        # Time integration
        ################################################################################################################
        init_growth = True
        ctd = True

        # set time step
        if not spinup and self.input.v('dtau'):
            dtau = self.input.v('dtau')*24*3600.
            tau = np.linspace(0, dtau, np.ceil(dtau/self.DT)+1)
            i_tau_now = 0
            dt = tau[1]-tau[0]
        else:
            dt = self.DT
            spinup = True

        # run time stepping
        dind = 0                    # index for the iteration number
        self.dif_prev = np.inf      # difference in previous step (only in spin-up)
        while ctd:
            dind +=1
            ## Growth functions
            Gamma_eval = [fun(Chat, Chat_prim, Chat_int, X[:, :], init_growth) for fun in Gamma_list]
            init_growth = False
            Gamma_sum = [self.sumdict(copy(j)) for j in Gamma_eval]

            Gamma = [[Gamma_sum[i] for i in j] for j in Gamma_index]
            for i, comp_name in enumerate(components):
                G = sum([a*b for a,b in zip(Gamma_factor[i], Gamma[i])])
                G += source[:, i]
                X[:, i] = cSolverTime(Ts[:, i], Fs[:, i], np.zeros(jmax+1), G, Hs[:, i], Hs[:, i]*X[:, i], X[0, i], 'flux', QC[i], dt, X[:, i], self.input)

            if spinup:
                dif = self.DT/dt*np.linalg.norm((X - self.Xprev)/(X+0.001*np.max(X)), np.inf)
                print(dind, dif)

                ## DEBUG
                # if np.max(X[:, 0]) > 50/1000. and dind > 100:
                #     print 'exit simulation; non-realistic result'
                #     ctd = False
                #     X = np.nan*X
                ## END
                # print dif
                if dif < 10**-5 or dind>2000:#10**-8:
                    ctd = False
                elif self.dif_prev < dif and dind > 100:                                     # adjust time step if diffence increases after 200th iteration (only in spin-up)
                    dt = dt/2.
                    dind = 0
                    print('timestep: ' + str(dt))
                else:
                    self.Xprev = copy(X)
                    self.dif_prev = dif
            else:
                i_tau_now += 1
                if i_tau_now == len(tau)-1:
                    ctd = False


        ################################################################################################################
        ## Return
        ################################################################################################################
        self.X = copy(X)

        Gamma = [[Gamma_eval[i] for i in j] for j in Gamma_index]
        # Gamma2 = [0]*len(Gamma)
        # for i in range(0, len(Gamma)):
        #     for j in range(1, len(Gamma[i])):
        #         mergeDicts(Gamma[i][0], Gamma[i][j])
        #     Gamma2[i] = Gamma[i][0]
        d = {}


        d['Ceco'] = {}
        d['Teco'] = {}
        d['Feco'] = {}
        d['Geco'] = {}
        for i, comp_name in enumerate(components):
            d['Ceco'][comp_name] = Chat[:, :, i]*X[:, i].reshape(jmax+1, 1)
            d['Geco'][comp_name] = {}
            for j in range(0, len(Gamma[i])):
                d['Geco'][comp_name].update(Gamma[i][j])
            #      d['Geco'][comp_name] = mergeDicts(d['Geco'][comp_name], Gamma[i][j])
            d['Teco'][comp_name] = dc.data[comp_name]['T']
            d['Feco'][comp_name] = dc.data[comp_name]['F']

        return d

    def update(self, dlist_old, dlist_new):
        return [mergeDicts(dlist_old[j], dlist_new[j]) for j in range(0, len(dlist_old))]

    def sumdict(self, d):
        for k in d.keys():
            if isinstance(d[k], dict):
                d[k] = self.sumdict(d[k])

        return sum(d.values())

    def component(self, ws, Kv, Kh):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        depth = (self.input.v('grid', 'low', 'z', range(0, jmax+1)) - self.input.v('grid', 'high', 'z', range(0, jmax+1))).reshape((jmax+1, 1))
        B = self.input.v('B', range(0, jmax+1))

        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        w0 = self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))

        ################################################################################################################
        # Leading order
        ################################################################################################################
        Chat = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        if ws[0,0] != 0:
            k = depth*ws[:, [-1]]/Kv[:, [-1]]*(1-np.exp(-ws[:, [-1]]/Kv[:, [-1]]*depth))**-1.
        else:
            k = 1
        Chat[:, :, 0] = k*np.exp(-ws[:, [-1]]/Kv[:, [-1]]*(z+depth))        # k is such that depth-av Chat = 1.

        Chatx = ny.derivative(Chat, 'x', self.input.slice('grid'))
        Chatz = -(ws[:, [-1]]/Kv[:, [-1]]).reshape((jmax+1, 1, 1))*Chat

        ################################################################################################################
        # First order
        ################################################################################################################
        F = np.zeros((jmax+1, kmax+1, 2*fmax+1, 2), dtype=complex)
        Fsurf = np.zeros((jmax+1, 1, 2*fmax+1, 2), dtype=complex)
        Fbed = np.zeros((jmax+1, 1, 2*fmax+1, 2), dtype=complex)

        ## forcing terms
        # advection
        F[:, :, fmax:, 0] = -ny.complexAmplitudeProduct(u0, Chatx, 2) - ny.complexAmplitudeProduct(w0, Chatz, 2)
        F[:, :, fmax:, 1] = -ny.complexAmplitudeProduct(u0, Chat, 2)

        ## solve
        Chat1, _ = pFunction(1, ws, Kv, F[:, :, fmax+1], Fsurf[:, :, fmax+1], Fbed[:, :, fmax+1], self.input)
        Chat1 = ny.eliminateNegativeFourier(Chat1, 2)

        ################################################################################################################
        # Closure
        ################################################################################################################
        # transport
        T = {}
        T['adv'] = {}
        T['adv']['tide'] = np.real(ny.integrate(ny.complexAmplitudeProduct(u0, Chat1[:, :, :, 0], 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]*B)
        for key in self.input.getKeysOf('u1'):
            utemp = self.input.v('u1', key, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            try:
                T['adv'][key] += np.real(ny.integrate(ny.complexAmplitudeProduct(utemp, Chat, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]*B)
            except:
                T['adv'][key] = np.real(ny.integrate(ny.complexAmplitudeProduct(utemp, Chat, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]*B)
        T['dif'] = - np.real(ny.integrate(ny.complexAmplitudeProduct(Kh, Chatx, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]*B)
        T['noflux'] = np.real(ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], Chat[:, [0], :], 2), zeta0, 2)[:, 0, 0]*B)

        F = {}
        F['adv'] = {}
        F['adv']['tide'] = np.real(ny.integrate(ny.complexAmplitudeProduct(u0, Chat1[:, :, :, -1], 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]*B)
        F['dif'] = - np.real(Kh[:, 0]*depth[:, 0]*B)

        H1 = np.real(depth[:, 0]*B)

        return T, F, H1, Chat[:, :, 0]




