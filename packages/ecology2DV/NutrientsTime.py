"""
NPP model with time dependent behaviour in temperature
2 types of phytoplankton
2 types of nutrients

Date: 01-05-2018
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from .util.EcologyCore import EcologyCore


class NutrientsTime:
    # Variables
    logger = logging.getLogger(__name__)


    # Methods
    def __init__(self, input):
        self.input = input
        self.EC = EcologyCore(self.input)
        return

    def run(self):
        self.logger.info('Running module NutrientsTime')
        components = ['phytoplankton', 'nitrogen', 'phosphorous']

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.B = self.input.v('B', range(0, jmax+1))
        self.depth = self.input.v('grid', 'low', 'z', range(0, jmax+1)) - self.input.v('grid', 'high', 'z', range(0, jmax+1))
        N_to_P = self.input.v('N_P_rat')

        Csea =  [self.input.v('Psea'),  self.input.v('Nsea'), self.input.v('Phossea')]
        QC =    [-self.input.v('QP'),  -self.input.v('QN'), -self.input.v('QPhos')]
        S = [np.zeros(jmax+1)]*3

        recycle = self.input.v('recycle')

        ws = [self.input.v('wp0', range(0, jmax+1), range(0, kmax+1), 0), np.zeros((jmax+1, kmax+1)), np.zeros((jmax+1, kmax+1))]
        Kv = [self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), 0)]*3
        Kh = [self.input.v('Kh', range(0, jmax+1), range(0, kmax+1), 0)]*3

        Gamma = [[self.PgrowthNE, self.Pmortality],                     # Phytoplankton
                 [self.PgrowthNE, self.Pmortality],                     # Nitrogen
                 [self.PgrowthNE, self.Pmortality]]                     # Phosphorous

        Gamma_factor = [[1.,        1. ],
                        [-1., -recycle],
                        [-1./N_to_P, -1./N_to_P*recycle]]

        ## add sources (now only allow for point sources)
        source = np.zeros((jmax+1, len(components)))
        source_words = ['Psource', 'Nsource', 'Phossource']
        source_indc = [0, 1, 2]     #index of source in component array
        for i, s in enumerate(source_words):
            if self.input.v(s):
                if self.input.v(s)[0] == 'point':
                    x = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]
                    xloc = np.argmin(np.abs(x-self.input.v(s)[1]))

                    if xloc == 0:
                        dx = x[1]-x[0]
                    elif xloc == jmax:
                        dx = x[jmax]-x[jmax-1]
                    else:
                        dx = 0.5*(x[xloc+1]-x[xloc-1])

                    source[xloc, source_indc[i]] = self.input.v(s)[2]/dx

                elif self.input.v(s)[0] == 'line':
                    x = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]
                    xmin = np.where(x>=self.input.v(s)[1])[0][0]
                    xmax = np.where(x<=self.input.v(s)[2])[0][-1]                       #TODO: improve case where line source stops halfway a cell
                    source[xmin:xmax+1, source_indc[i]] = self.input.v(s)[3]

        d = self.EC.main(components, ws, Kv, Kh, Csea, QC, S, Gamma, Gamma_factor, source)
        d['mu0'], d['tau_night'], d['mumax'], d['FN'], d['FP'], d['FE'] = self.growthrate(d['Ceco'])
        return d

########################################################################################################################
# functions for mu and m
########################################################################################################################
    def PgrowthNE(self, Ch, Chprim, Chint, X, init = False):
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')

        kp = self.input.v('kp')
        E0 = self.input.v('E0')
        HI = self.input.v('HI')

        T = 29*25.655*3600      # 25.655 hrs is the difference frequency of the M2 tide and a day. After 29 cycles, exactly 60 tides and 31 days have passed
        t = np.linspace(0, T, 1000)

        # Growth rate (Eppley, 1972)
        Temp = self.input.v('Temp', range(0, jmax+1), [0])
        mu0 = self.input.v('mu00', range(0, jmax+1), [0])
        mu_Eppley = .851*1.066**Temp
        mumax = mu0*np.log(2)*mu_Eppley     # need log(2) as Eppley's number is in doublings/day (not in 1/day)

        # Limitations:
        N = Ch[:, 0, 1]*X[:, 1]
        Phos = Ch[:, 0, 2]*X[:, 2]
        Ph = Ch[:, :, 0]
        Phprim = Chprim[:, :, 0]

        if init:
            self.G_Pgrowth = {}
            z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]

            # Light - background
            kbg = self.input.v('kbg')
            self.kbg = kbg*z.reshape((jmax+1, kmax+1, 1))

            # Light - sediment
            kc = self.input.v('kc')
            c00 = np.real(self.input.v('c0', range(0, jmax+1), range(0, kmax+1), [0]))
            c00int = ny.integrate(c00, 'z', 0, range(0, kmax+1), self.input.slice('grid'))
            c04 = np.real(self.input.v('c0', range(0, jmax+1), range(0, kmax+1), [2]))
            c04int = ny.integrate(c04, 'z', 0, range(0, kmax+1), self.input.slice('grid'))
            OMEGA = self.input.v('OMEGA')
            cint = c00int + c04int*np.cos(2*OMEGA*t.reshape((1,1,len(t))))

            self.kc = (kc*cint)

            # Light - daily cycle
            omega_E = self.input.v('omega_E')       # in 1/hr
            self.E = np.maximum(E0*np.sin(t/3600.*omega_E), 0).reshape((1,1,len(t)))

        if init or kp > 0:
            # Light - self-shading
            self.kself = -kp*np.cumsum(Phprim*X[:, [0]], axis=1).reshape((jmax+1, kmax+1, 1))

        # Light - total
        k = self.kbg+self.kc+self.kself
        E = self.E*np.exp(k)
        FE = E/np.sqrt(HI**2+E**2)

        # N
        HN = self.input.v('HN')
        FN = (N/(HN+N)).reshape((jmax+1, 1, 1))

        # Phos
        HP = self.input.v('HP')
        FP = (Phos/(HP+Phos)).reshape((jmax+1, 1, 1))

        mu = mumax*np.mean(np.minimum(np.minimum(FN, FP), FE), axis=2)
        # mu = mumax*np.minimum(np.minimum(np.mean(FN, axis=2), np.mean(FP, axis=2)), np.mean(FE, axis=2))

        muint = ny.integrate(mu*Ph, 'z', kmax, 0, self.input.slice('grid'))[:, 0]
        return {'growth':self.B*muint*X[:, 0]}

    def Pmortality(self, Ch, Chprim, Chint, X, init = False):
        if init:
            self.G_Pmort = {}
            Phint = Chint[:, 0]
            m = self.input.v('mp')
            self.m_int = - np.real(m*Phint)
        return {'mortality': self.B*self.m_int*X[:, 0]}

########################################################################################################################
# Other functions
########################################################################################################################
    def dict_multiply(self, d, factor):
        d2 = {}
        for key in d.keys():
            if isinstance(d[key], dict):
                d2[key] = self.dict_multiply(d[key], factor)
            else:
                d2[key] = factor*d[key]
        return d2

    def growthrate(self, Ceco, init = False):
        """
        Function largely copied from growth function above, returns the growth rate mu for output purposes
        (IN FUTURE FIND BETTER WAY TO STRUCTURE THIS)
        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')

        kp = self.input.v('kp')
        E0 = self.input.v('E0')
        HI = self.input.v('HI')
        mu = {}

        T = 29*25.655*3600      # 25.655 hrs is the difference frequency of the M2 tide and a day. After 29 cycles, exactly 60 tides and 31 days have passed
        t = np.linspace(0, T, 1000)

        # Growth rate (Eppley, 1972)
        Temp = self.input.v('Temp', range(0, jmax+1), [0], [0])
        mu0 = self.input.v('mu00', range(0, jmax+1), [0], [0])
        mu_Eppley = .851*1.066**Temp
        mumax = self.input.v('mu00')*np.log(2)*mu_Eppley     # need log(2) as Eppley's number is in doublings/day (not in 1/day)
        mu['mumax'] = mumax

        # Limitations:
        N = Ceco['nitrogen'][:, 0]              # Nitrogen. Assume well-mixed
        Phos = Ceco['phosphorous'][:, 0]              # Phosphorous. Assume well-mixed
        Pprim = -ny.primitive(np.real(Ceco['phytoplankton']), 'z', 0, kmax, self.input.slice('grid'))   # primitive counted from surface

        z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]

        # Light - background
        kbg = self.input.v('kbg')
        kbg = kbg*z.reshape((jmax+1, kmax+1, 1, 1))

        # Light - sediment
        kc = self.input.v('kc')
        c00 = np.real(self.input.v('c0', range(0, jmax+1), range(0, kmax+1), [0], [0]))
        c00int = ny.integrate(c00, 'z', 0, range(0, kmax+1), self.input.slice('grid'))
        c04 = np.real(self.input.v('c0', range(0, jmax+1), range(0, kmax+1), [2], [0]))
        c04int = ny.integrate(c04, 'z', 0, range(0, kmax+1), self.input.slice('grid'))
        OMEGA = self.input.v('OMEGA')
        cint = c00int + c04int*np.cos(2*OMEGA*t.reshape((1,1,1,len(t))))

        kc = (kc*cint)

        # Light - daily cycle
        omega_E = self.input.v('omega_E')       # in 1/hr
        dE = np.maximum(np.sin(t/3600*omega_E), 0).reshape((1,1,1,len(t)))

        # Light - self-shading
        kself = -kp*np.cumsum(Pprim, axis=1).reshape((jmax+1, kmax+1,1,1))

        # Light - total and decomposition
        #   night
        dayinds = np.where(dE[0,0,0,:]>0)[0]
        tau_night = np.float(len(dayinds))/np.float(len(t))

        #   day
        k = kbg+kc+kself
        alpha_day = np.exp(k[:, :, :, dayinds])
        E_day = E0*dE[:, :, :, dayinds]*alpha_day
        dE_day = dE[:, :, :, dayinds]
        FE_day = E_day/np.sqrt(HI**2+E_day**2)
        FE = {}
        FE['daily'] = (1.-dE_day)/(1-E_day/E0)*FE_day
        FE['background'] = dE_day*(1.-alpha_day)/(1-E_day/E0)*(1-np.exp(kbg))/(3.+1e-4-np.exp(kbg)-np.exp(kc[:, :,:, dayinds])-np.exp(kself))*FE_day
        FE['sediment'] = dE_day*(1.-alpha_day)/(1-E_day/E0)*(1-np.exp(kc[:,:, :, dayinds]))/(3.+1e-4-np.exp(kbg)-np.exp(kc[:,:, :, dayinds])-np.exp(kself))*FE_day
        FE['self-shading'] = dE_day*(1.-alpha_day)/(1-E_day/E0)*(1-np.exp(kself))/(3.+1e-4-np.exp(kbg)-np.exp(kc[:,:, :, dayinds])-np.exp(kself))*FE_day
        # mu['FE']['daily']       = (1.-dE_day)/(4.-dE_day-np.exp(kbg)-np.exp(kc[:, :,:, dayinds])-np.exp(kself))*FE_day
        # mu['FE']['background']  = (1.-np.exp(kbg))/(4.-dE_day-np.exp(kbg)-np.exp(kc[:, :,:, dayinds])-np.exp(kself))*FE_day
        # mu['FE']['sediment']    = (1.-np.exp(kc[:, :,:, dayinds]))/(4.-dE_day-np.exp(kbg)-np.exp(kc[:, :,:, dayinds])-np.exp(kself))*FE_day
        # mu['FE']['self-shading']= (1.-np.exp(kself))/(4.-dE_day-np.exp(kbg)-np.exp(kc[:, :,:, dayinds])-np.exp(kself))*FE_day

        # N
        HN = self.input.v('HN')
        FN = (N/(HN+N)).reshape((jmax+1, 1, 1, 1))

        # Phos
        HP = self.input.v('HP')
        FP = (Phos/(HP+Phos)).reshape((jmax+1, 1, 1, 1))

        mu = mumax*tau_night*np.mean(np.minimum(np.minimum(FN, FP), FE_day), axis=-1)
        return mu, tau_night, mumax, FN, FP, FE





