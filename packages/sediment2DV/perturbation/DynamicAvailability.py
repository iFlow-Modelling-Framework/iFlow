"""
Dynamic erodibility computation integrating numerically over the long time scale.

Options on input
    sedbc: type of horizontal boundary condition; only 'csea' for seaward concentration is implemented.
    initial: initial condition type 'erodibility', 'stock' for specifying initial f or S. 'equilibrium' for starting in
                equilibrium using Q1 in hydrodynamics modules
    finit: initial f, only used if 'initial_condition_sediment' = 'erodibility'
    Sinit: initial S, only used if 'initial_condition_sediment' = 'stock'

Optional input parameters
    Qsed: sediment inflow from upstream
    sedsource: other sediment sources

From version: 3.x
Date: 04-10-2023
Authors: Y.M. Dijkstra

REGISTRY ENTRY
module		DynamicAvailability
packagePath sediment/
input       grid hatc0 hatc1 hatc2 u0 zeta0 u1 Kh B sedbc @sedbc t toutput Q1 initial if{finit,@{initial}=='erodibility'} if{Sinit,@{initial}=='stock'}        #optional: Qsed sedsource
output		c0 c1 c2 a f F T t
"""

import logging
import numpy as np
import scipy.linalg
import nifty as ny
from packages.sediment2DV.perturbation.EquilibriumAvailability import EquilibriumAvailability
from src.DataContainer import DataContainer


class DynamicAvailability(EquilibriumAvailability):
    # Variables
    logger = logging.getLogger(__name__)
    timer = ny.Timer()

    # Methods
    def __init__(self, input):
        EquilibriumAvailability.__init__(self, input)
        self.input = input
        return

    def run(self):
        """         """
        self.logger.info('Running module DynamicAvailability')

        ################################################################################################################
        # Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]
        self.dx = (self.x[1:]-self.x[:-1])
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]

        # variables for the erodibility-stock relation
        c00 = np.real(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 0))
        c04 = np.abs(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 2))
        c04_int = np.trapz(c04, x=-self.zarr)
        hatc2 = np.abs(self.input.v('hatc2', range(0, jmax+1), range(0, kmax+1), 0))
        alpha1 = np.trapz(c00 + hatc2, x=-self.zarr, axis=1)
        if alpha1[-1] == 0:
            alpha1[-1] = alpha1[-2]
        alpha2 = c04_int/alpha1 + 1e-3


        ################################################################################################################
        # Compute transport, source and BC
        ################################################################################################################
        ## Transport
        d = self.compute_transport()
        dc = DataContainer(d)
        dc.merge(self.input.slice('grid'))
        T = dc.v('T', range(0, jmax+1), 0, 0, 0)
        F = dc.v('F', range(0, jmax+1), 0, 0, 0)

        ## Source
        G = self.compute_source()                                                                                       #NB does not change over long time scale

        ## Seaward boundary condition
        if self.input.v('sedbc') == 'csea':
            csea = self.input.v('csea')
            fsea = csea / alpha1[0] * (self.input.v('grid', 'low', 'z', 0) - self.input.v('grid', 'high', 'z', 0))        #NB does not change over long time scale
        else:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('incorrect seaward boundary type (sedbc) for sediment module')

        ################################################################################################################
        # Initialise X = (f, S)
        ################################################################################################################
        if self.input.v('initial_condition_sediment') == 'erodibility':
            finit = self.input.v('finit', range(0, jmax+1))
            Sinit = self.init_stock(finit, alpha1, alpha2)

        elif self.input.v('initial_condition_sediment') == 'stock':
            Sinit = self.input.v('Sinit', range(0, jmax+1))

        elif self.input.v('initial_condition_sediment') == 'equilibrium':
            _, finit, _ = self.availability(F, T, G, alpha1, alpha2)
            Sinit = self.init_stock(finit, alpha1, alpha2)

        else:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('incorrect initial value for sediment module. Use erodibility, stock or equilibrium')


        ################################################################################################################
        # Time integrator
        ################################################################################################################
        S, Transport = self.timestepping2(T, F, alpha1, alpha2, Sinit, fsea, G)

        ################################################################################################################
        # Prepare output
        ################################################################################################################
        f = self.erodibility_stock_relation(alpha2, S/alpha1)
        d['f'] = f
        d['a'] = S
        d['sediment_transport'] = Transport

        fx = np.gradient(f, self.x, axis=0, edge_order=2)
        hatc0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        hatc1 = self.input.v('hatc1', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        hatc1x = self.input.v('hatc1', 'ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        hatc2 = self.input.v('hatc2', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        d['c0'] = hatc0*f[:, None, None]
        d['c1'] = hatc1*f[:, None, None] + hatc1x*fx[:, None, None]
        d['c2'] = hatc2*f[:, None, None]

        return d

    def init_stock(self, f, alpha1, alpha2):
        """Compute stock given f, alpha1 and alpha2.
        For f=1, S equals Smax, which is S corresponding to f=0.999.
        """

        # Establish initial minimum stock Smax, such that f=0.999
        Smax = np.array([0.999])  # initial guess
        F = self.erodibility_stock_relation(alpha2[[0]], Smax) - np.array([0.999])
        while max(abs(F)) > 10 ** -4 :
            dfdS = self.erodibility_stock_relation_der(alpha2[[0]], Smax)
            Smax = Smax - F / dfdS
            F = self.erodibility_stock_relation(alpha2[[0]], Smax) - np.array([0.999])

        # Newton-Raphson iteration towards actual Shat
        #   init
        Shat = f
        F = self.erodibility_stock_relation(alpha2, Shat) - f

        i = 0
        while max(abs(F)) > self.TOL and i < 50:
            i += 1
            dfdS = self.erodibility_stock_relation_der(alpha2, Shat)
            Shat = Shat - F / (dfdS + 10 ** -12)
            F = self.erodibility_stock_relation(alpha2, Shat) - f
            if i == 50:
                f = self.erodibility_stock_relation(alpha2, Shat)

        Shat = np.minimum(Shat, Smax)
        return Shat*alpha1

    def timestepping(self, T, F, alpha1, alpha2, Sold, fsea, G):
        """        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        dt = self.input.v('dt')
        A = np.zeros((4, jmax+1))
        rhs = np.zeros((jmax+1))

        B = self.input.v('B', range(0, jmax+1))
        Bx = self.input.d('B', range(0, jmax+1), dim='x')

        hatSold = Sold/alpha1

        h = self.erodibility_stock_relation(alpha2, hatSold)
        hder = self.erodibility_stock_relation_der(alpha2, hatSold)
        beta = h - hder*hatSold

        Tx = np.gradient(T, self.x, edge_order=2)
        Fx = np.gradient(F, self.x, edge_order=2)
        dif = B*F

        adv = B*T+Bx*F+B*Fx
        BTx = B*Tx+Bx*T

        # interior
        A[0, 2:] = + np.minimum(adv[1:-1], 0)*hder[2:]/(alpha1[2:]*self.dx[1:]) \
                   + dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[2:]/alpha1[2:]/self.dx[1:]
        A[1, 1:-1] = B[1:-1]/dt + BTx[1:-1]*hder[1:-1]/alpha1[1:-1] + np.maximum(adv[1:-1], 0)*hder[1:-1]/(alpha1[1:-1]*self.dx[:-1]) - np.minimum(adv[1:-1], 0)*hder[1:-1]/(alpha1[1:-1]*self.dx[1:]) \
                     - dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[1:-1]/alpha1[1:-1]*(1./self.dx[1:]+1./self.dx[:-1])
        A[2, :-2] = -np.maximum(adv[1:-1], 0)*hder[:-2]/(alpha1[:-2]*self.dx[:-1]) \
                    + dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[:-2]/alpha1[:-2]/self.dx[:-1]

        rhs[1:-1] = B[1:-1]/dt*Sold[1:-1] - BTx[1:-1]*beta[1:-1] - np.maximum(adv[1:-1], 0)*(beta[1:-1]-beta[:-2])/self.dx[:-1] - np.minimum(adv[1:-1], 0)*(beta[2:]-beta[1:-1])/self.dx[1:] \
                    -dif[1:-1]*((beta[2:]-beta[1:-1])/self.dx[1:] - (beta[1:-1]-beta[:-2])/self.dx[:-1])/(0.5*(self.dx[1:]+self.dx[:-1]))
        rhs[1:-1] += -B[1:-1]*G[1:-1]

        # Quick fix for ensuring positivity (Patankar, 1980); could be neater for greater accuracy
        A[1, 1:-1] += -np.minimum(rhs[1:-1], 0)/(Sold[1:-1]+1.e-6)
        rhs[1:-1] = np.maximum(rhs[1:-1], 0)

        # Boundaries
        #   x=0
        if hder[0] == 0:
            A[1, 0] = 1
            rhs[0] = Sold[0]
        else:
            A[1, 0] = hder[0]/alpha1[0]
            rhs[0] = fsea - h[0] + hder[0]*hatSold[0]

        #   x=L
        if hder[-1] == 0:
            A[1, -1] = 1        # TODO - needs correction; VIOLATION OF MASS BALANCE. This line will only apply in case Q1 = 0
            rhs[-1] = Sold[-1]
            self.logger.warning('f=1 at upstream boundary. The code is not correct for this case and mass balance may be violated. Please investigate.')
        else:
            A[1, -1] = B[-1]*T[-1]*hder[-1]/alpha1[-1] + 3./2.*B[-1]*F[-1]*hder[-1]/alpha1[-1]/self.dx[-1]
            A[2, -2] = -2.*B[-1]*F[-1]*hder[-2]/alpha1[-2]/self.dx[-1]
            A[3, -3] = 0.5*B[-1]*F[-1]*hder[-3]/alpha1[-3]/self.dx[-1]
            rhs[-1] = -B[-1]*T[-1]*beta[-1] - B[-1]*F[-1]*(3./2.*beta[-1] - 2.*beta[-2] + 0.5*beta[-3])/self.dx[-1]
            rhs[-1] += -B[-1]*G[-1]

            #   alternative first order
            # A[1, -1] = B[-1]*T[-1]*hder[-1]/alpha1[-1] + B[-1]*F[-1]*hder[-1]/alpha1[-1]/self.dx[-1]
            # A[2, -2] = -1.*B[-1]*F[-1]*hder[-2]/alpha1[-2]/self.dx[-1]
            # rhs[-1] = -B[-1]*T[-1]*beta[-1] - B[-1]*F[-1]*(1*beta[-1] - 1.*beta[-2])/self.dx[-1]
            # rhs[-1] += B[-1]*G[-1]


        S = scipy.linalg.solve_banded((2, 1), A, rhs, overwrite_ab=False, overwrite_b=False)
        return S, np.ones(jmax+1)*np.nan

    def timestepping2(self, T, F, alpha1, alpha2, Sold, fsea, G, Sintermediate = None):
        """        """
        if Sintermediate is None:
            Sintermediate = Sold
        jmax = self.input.v('grid', 'maxIndex', 'x')
        dt = self.input.v('dt')
        A = np.zeros((3, jmax+1))
        rhs = np.zeros((jmax+1))
        Transport = np.zeros((jmax+1))

        B = self.input.v('B', range(0, jmax+1))

        hatSold = Sintermediate/alpha1

        h = self.erodibility_stock_relation(alpha2, hatSold)
        hder = self.erodibility_stock_relation_der(alpha2, hatSold)/alpha1
        beta = h - hder*Sintermediate
        dif = B*F
        adv = B*T
        source = B*G

        # interior
        a =  np.minimum(0.5*(adv[1:-1]+adv[2:]), 0)/(0.5*(adv[1:-1]+adv[2:]))*adv[2:]     + 0.5*(dif[1:-1]+dif[2:])/self.dx[1:]
        b = -np.minimum(0.5*(adv[1:-1]+adv[:-2]), 0)/(0.5*(adv[1:-1]+adv[:-2]))*adv[1:-1] - 0.5*(dif[1:-1]+dif[2:])/self.dx[1:] + \
             np.maximum(0.5*(adv[1:-1]+adv[2:]), 0)/(0.5*(adv[1:-1]+adv[2:]))*adv[1:-1]   - 0.5*(dif[1:-1]+dif[:-2])/self.dx[:-1]
        c = -np.maximum(0.5*(adv[1:-1]+adv[:-2]), 0)/(0.5*(adv[1:-1]+adv[:-2]))*adv[:-2]  + 0.5*(dif[1:-1]+dif[:-2])/self.dx[:-1]

        A[0, 2:] = a*hder[2:]
        A[1, 1:-1] = b*hder[1:-1] + 0.5*(self.dx[1:]+self.dx[:-1])/dt*B[1:-1]
        A[2, :-2] = c*hder[:-2]

        rhs[1:-1] = a*(-beta[2:]) + b*(-beta[1:-1]) + c*(-beta[:-2]) + 0.5*(self.dx[1:]+self.dx[:-1])/dt*B[1:-1]*Sold[1:-1]
        rhs[1:-1] += - 0.5*(source[1:-1]+source[2:]) + 0.5*(source[1:-1]+source[:-2])

        # Quick fix for ensuring positivity (Patankar, 1980); could be neater for greater accuracy
        # A[1, 1:-1] += -np.minimum(rhs[1:-1], 0)/(Sold[1:-1]+1.e-6)
        # rhs[1:-1] = np.maximum(rhs[1:-1], 0)

        # Boundaries
        #   x=0
        if hder[0] == 0:
            A[1, 0] = 1
            rhs[0] = Sold[0]
        else:
            A[1, 0] = hder[0]
            rhs[0] = fsea - beta[0]

        #   x=L
        b = - np.minimum(0.5*(adv[-1]+adv[-2]), 0)/(0.5*(adv[-1]+adv[-2]))*adv[-1] - 0.5*(dif[-1]+dif[-2])/self.dx[-1]
        c = - np.maximum(0.5*(adv[-1]+adv[-2]), 0)/(0.5*(adv[-1]+adv[-2]))*adv[-2] + 0.5*(dif[-1]+dif[-2])/self.dx[-1]
        A[1, -1] = b*hder[-1] + 0.5*self.dx[-1]/dt*B[-1]
        A[2, -2] = c*hder[-2]
        rhs[-1] = b*(-beta[-1]) + c*(-beta[-2]) + 0.5*self.dx[-1]/dt*B[-1]*Sold[-1]
        rhs[-1] += - source[-1] + 0.5*(source[-1]+source[-2])

        S = scipy.linalg.solve_banded((1, 1), A, rhs, overwrite_ab=False, overwrite_b=False)

        #   Transport
        # f = self.erodibility_stock_relation(alpha2, S/alpha1)
        f = beta + hder*S   # this is the value of f following the numerical scheme. Note that overshoots/undershoots may occur here.
        Transport[:-1] = (np.minimum(0.5*(adv[1:]+adv[:-1]), 0)/(0.5*(adv[1:]+adv[:-1]))*adv[1:])*f[1:] \
                       + (np.maximum(0.5*(adv[1:]+adv[:-1]), 0)/(0.5*(adv[1:]+adv[:-1]))*adv[:-1])*f[:-1] \
                       + (0.5*(dif[1:]+dif[:-1])/self.dx)*(f[1:] - f[:-1])

        #DEBUG
        # B = self.input.v('B', range(jmax+1))
        # St = B[1:]*(S[1:]-Sold[1:])*self.dx/dt
        # St[-1] = St[-1]/2
        # Fx = Transport[1:]-Transport[:-1]
        # res = St+Fx
        # print(np.max(abs(res)))

        # CFL = np.abs(T*dt/self.dx[0])
        # Pe = np.abs(self.dx[0]*T/F)
        # print(np.max(CFL), np.max(Pe))
        return S, Transport



