"""
Dynamic erodibility computation integrating numerically over the long time scale.
This routine only allows variation of the first-order river discharge over the long time scale. NB. variations of the
tide and sediment boundary conditions or sources are not included (also not Qsed ~ Q). The loop over time is enclosed,
so it cannot be used together with time variations in other modules.

Options on input
    sedbc: type of horizontal boundary condition; only 'csea' for seaward concentration is implemented.
    initial: initial condition type 'erodibility', 'stock' for specifying initial f or S. 'equilibrium' for starting in
                equilibrium using Q1 in hydrodynamics modules
    finit: initial f, only used if 'initial' = 'erodibility'
    Sinit: initial S, only used if 'initial' = 'stock'

Optional input parameters
    Qsed: sediment inflow from upstream
    sedsource: other sediment sources

From version: 2.5
Date: 23-01-2019
Authors: Y.M. Dijkstra, R.L. Brouwer

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
from src.util.diagnostics import KnownError
import sys
from nifty import toList
from copy import copy
from .EquilibriumAvailability import EquilibriumAvailability
from src.DataContainer import DataContainer

## TODO: first iteration Told is not properly set. This is a problem in a repeated 2-step experiment with theta!=1.


class DynamicAvailability(EquilibriumAvailability):
    # Variables
    logger = logging.getLogger(__name__)
    theta = 1.              # theta=0 is Forward Euler, theta=1 is Backward Euler and theta=0.5 is Crank-Nicolson
    TOL = 1.e-10
    MAXITER = 100
    timer = ny.Timer()

    # Methods
    def __init__(self, input):
        EquilibriumAvailability.__init__(self, input)
        self.input = input
        return

    def run(self):
        """         """
        self.logger.info('Running module DynamicAvailability_upwind')

        ################################################################################################################
        # Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]
        self.dx = (self.x[1:]-self.x[:-1])
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]

        self.B = self.input.v('B', range(0, jmax+1))
        self.Bx = self.input.d('B', range(0, jmax+1), dim='x')

        self.Kh = self.input.v('Kh')
        self.u0tide_bed = self.input.v('u0', 'tide', range(0, jmax+1), kmax, 1)
        c00 = np.real(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 0))

        c04 = np.abs(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 2))
        c04_int = np.trapz(c04, x=-self.zarr)
        hatc2 = np.abs(self.input.v('hatc2', range(0, jmax+1), range(0, kmax+1), 0))
        alpha1 = np.trapz(c00 + hatc2, x=-self.zarr, axis=1)
        if alpha1[-1] == 0:
            alpha1[-1] = alpha1[-2]
        alpha2 = c04_int/alpha1 + 1e-3

        ## load time series Q
        t = self.interpretValues(self.input.v('t'))
        toutput = self.interpretValues(self.input.v('toutput'))
        toutput[0] = t[0]                       # correct output time; first time level is always equal to initial computation time
        Qarray = self.interpretValues(self.input.v('Q1'))
        if len(Qarray)!=len(t):
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('Length of Q does not correspond to length of time array.')


       ################################################################################################################
        # Compute transport, source and BC
        ################################################################################################################
        ## Transport
        d = self.compute_transport()
        dc = DataContainer(d)

        #       change size of those components that depend on the river discharge and put init value in first element
        T_r = copy(dc.v('T', 'river', range(0, jmax+1)))
        T_rr = copy(dc.v('T', 'river_river', range(0, jmax+1)))
        T_dr = copy(dc.v('T', 'diffusion_river', range(0, jmax+1)))
        F_dr = dc.v('F', 'diffusion_river', range(0, jmax+1))

        d['T']['river'] = np.zeros((jmax+1, 1, 1, len(toutput)))
        d['T']['river'][:, 0, 0, 0] = T_r
        d['T']['river_river'] = np.zeros((jmax+1, 1, 1, len(toutput)))
        d['T']['river_river'][:, 0, 0, 0] = T_rr
        d['T']['diffusion_river'] = np.zeros((jmax+1, 1, 1, len(toutput)))
        d['T']['diffusion_river'][:, 0, 0, 0] = T_dr
        d['F']['diffusion_river'] = np.zeros((jmax+1, 1, 1, len(toutput)))
        d['F']['diffusion_river'][:, 0, 0, 0] = F_dr

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

        ## compute TQ, uQ, hatc2Q: quantities relative to the river discharge
        u1river = np.real(self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax + 1), 0))
        Q_init = -np.trapz(u1river[-1, :], x=-self.zarr[-1, :]) * self.B[-1]    # initial discharge
        self.TQ = T_r/Q_init            # river transport per discharge unit
        self.uQ = u1river/Q_init        # river velocity per discharge unit

        ################################################################################################################
        # Initialise X = (f, S)
        ################################################################################################################
        if self.input.v('initial') == 'erodibility':
            finit = self.input.v('finit', range(0, jmax+1))
            Sinit = self.init_stock(finit, alpha1, alpha2)

        elif self.input.v('initial') == 'stock':
            Sinit = self.input.v('Sinit', range(0, jmax+1))
            finit = self.erodibility_stock_relation(alpha2, Sinit/alpha1)

        elif self.input.v('initial') == 'equilibrium':
            _, finit, _ = self.availability(F, T, G, alpha1, alpha2)
            Sinit = self.init_stock(finit, alpha1, alpha2)
            
        else:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('incorrect initial value for sediment module. Use erodibility, stock or equilibrium')

        X = np.concatenate((finit, Sinit))
        f = np.zeros((jmax+1, 1, 1, len(toutput)))
        S = np.zeros((jmax+1, 1, 1, len(toutput)))
        f[:, 0, 0, 0] = finit
        S[:, 0, 0, 0] = Sinit

        ################################################################################################################
        # Time integrator
        ################################################################################################################
        T_base = dc.v('T', range(0, jmax+1), 0, 0, 0) - dc.v('T', 'river', range(0, jmax+1), 0, 0, 0) - dc.v('T', 'river_river', range(0, jmax+1), 0, 0, 0) - dc.v('T', 'diffusion_river', range(0, jmax+1), 0, 0, 0)
        F_base = dc.v('F', range(0, jmax+1), 0, 0, 0) - dc.v('F', 'diffusion_river', range(0, jmax+1), 0, 0, 0)
        
        #   loop
        self.timer.tic()
        qq = 1      # counter for saving
        for i, Q in enumerate(Qarray[1:]):
            # quantities at old time step
            Told = copy(T)
            Fold = copy(F)
            alpha1old = copy(alpha1)
            alpha2old = copy(alpha2)

            # Update transport terms and hatc2 & load new transport terms
            T_riv, T_rivriv, T_difriv, F_difriv = self.update_transport(Q)
            ur = self.uQ[:, -1]*Q
            hatc2 = self.update_hatc2(ur)
            T = T_base + T_riv + T_rivriv + T_difriv
            F = F_base + F_difriv
            
            # Make one time step and iterate over non-linearity
            self.dt = t[i+1]-t[i]

            alpha1 = np.trapz(c00 + hatc2, x=-self.zarr, axis=1)
            if alpha1[-1] == 0:
                alpha1[-1] = alpha1[-2]
            alpha2 = c04_int/alpha1 + 1e-3
            X = self.timestepping(T, F, alpha1, alpha2, Told, Fold, alpha1old, alpha2old, X, fsea, G)

            # save output on output timestep
            if t[i+1]>=toutput[qq]:
                toutput[qq] = t[i+1]        # correct output time to real time if time step and output time do not correspond
                d['T']['river'][:, 0, 0, qq] = T_riv
                d['T']['river_river'][:, 0, 0, qq] = T_rivriv
                d['T']['diffusion_river'][:, 0, 0, qq] = T_difriv
                d['F']['diffusion_river'][:, 0, 0, qq] = F_difriv
                f[:, 0, 0, qq] = X[:jmax+1]
                S[:, 0, 0, qq] = X[jmax+1:]
                qq += 1
                qq = np.minimum(qq, len(toutput)-1)

            # display progress
            if i % np.floor(len(Qarray[1:]) / 100.) == 0:
                percent = float(i) / len(Qarray[1:])
                hashes = '#' * int(round(percent * 10))
                spaces = ' ' * (10 - len(hashes))
                sys.stdout.write("\rProgress: [{0}]{1}%".format(hashes + spaces, int(round(percent * 100))))
                sys.stdout.flush()
        sys.stdout.write('\n')
        self.timer.toc()
        self.timer.disp('time integration time')

        ################################################################################################################
        # Prepare output
        ################################################################################################################
        d['f'] = f
        d['a'] = S

        fx = np.gradient(f, self.x, axis=0, edge_order=2)
        hatc0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), [0])
        hatc1 = self.input.v('hatc1', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), [0])
        hatc1x = self.input.v('hatc1', 'ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), [0])
        hatc2 = self.input.v('hatc2', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), [0])
        d['c0'] = hatc0*f
        d['c1'] = hatc1*f + hatc1x*fx
        d['c2'] = hatc2*f

        d['t'] = toutput

        return d

    def update_transport(self, Q):
        """Update transport terms T and F for time dependent river discharge forcing Q.

        Parameters:
            Q - river discharge Q

        Returns:
            T - advection function
            F - diffusion function
            A - variable appearing the the Exner equation, i.e. A = (BF)_x / B
            C20 - second-order sediment concentration
        """
        # T - river
        T_riv = self.TQ * Q

        # T - river_river
        ur = self.uQ*Q
        hatc2 = self.update_hatc2(ur[:, -1])
        T_rivriv = np.real(np.trapz(ur * hatc2, x=-self.zarr, axis=1))

        # T - diffusion river
        hatc2x = np.gradient(hatc2, self.x, axis=0, edge_order=2)
        T_difriv = -self.Kh*np.real(np.trapz(hatc2x, x=-self.zarr, axis=1))

        # F - diffusion river
        F_difriv = -self.Kh*np.real(np.trapz(hatc2, x=-self.zarr, axis=1))
        return T_riv, T_rivriv, T_difriv, F_difriv

    def update_hatc2(self, u_riv_bed):
        """Update shear stress for hatc2 based on varying river discharge"""
        jmax = self.input.v('grid', 'maxIndex', 'x')
        RHOS = self.input.v('RHOS')
        RHO0 = self.input.v('RHO0')
        DS = self.input.v('DS')
        GPRIME = self.input.v('G') * (RHOS - RHO0) / RHO0    #
        HR = (self.input.v('H', range(0, jmax+1)).reshape(jmax+1, 1) +
                  self.input.v('R', range(0, jmax+1)).reshape(jmax+1, 1))
        WS = self.input.v('ws0', range(0, jmax+1), [0], 0)
        Kv0 = self.input.v('Kv', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        if self.input.v('friction') is not None:
            sf = self.input.v(self.input.v('friction'), range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        else:
            sf = self.input.v('Roughness', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        finf = self.input.v('finf')

        # shear stress
        T = np.linspace(0, 2*np.pi, 100)
        utid = np.zeros((jmax+1, len(T)), dtype='complex')
        ucomb = np.zeros((jmax+1, len(T)), dtype='complex')
        for i, t in enumerate(T):
            utid[:, i] = 0.5 * (self.u0tide_bed * np.exp(1j*t) + np.conj(self.u0tide_bed) * np.exp(-1j*t))
            ucomb[:, i] = u_riv_bed + 0.5 * (self.u0tide_bed * np.exp(1j*t) + np.conj(self.u0tide_bed) * np.exp(-1j*t))
        uabs_tid = np.mean(np.abs(utid), axis=1)
        uabs_tot = np.mean(np.abs(ucomb), axis=1)
        uabs_eps = (uabs_tot - uabs_tid).reshape(jmax+1, 1)

        # erosion
        if self.input.v('erosion_formulation') == 'Partheniades':
            Ehat = finf*sf*(uabs_eps)*RHO0
        else:
            Ehat = (WS * finf*RHOS * sf / (GPRIME * DS))*(uabs_eps)

        hatc2 = np.real(Ehat/WS * np.exp(-WS * (HR + self.zarr) / Kv0))
        return hatc2

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

    def timestepping(self, T, F, alpha1, alpha2, Told, Fold, alpha1old, alpha2old, Xold, fsea, G):
        """        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        A = np.zeros((4, jmax+1))
        rhs = np.zeros((jmax+1))

        Sold = Xold[jmax + 1:]
        hatSold = Sold/alpha1old
        fold = Xold[:jmax+1]
        h = self.erodibility_stock_relation(alpha2old, hatSold)
        hder = self.erodibility_stock_relation_der(alpha2old, hatSold)
        hda2 = self.erodibility_stock_relation_da2(alpha2old, hatSold)
        beta = h - hder*hatSold + hda2*(alpha2-alpha2old)

        Tx = np.gradient(T, self.x, edge_order=2)
        Fx = np.gradient(F, self.x, edge_order=2)
        dif = self.B*F

        adv = self.B*T+self.Bx*F+self.B*Fx
        BTx = self.B*Tx+self.Bx*T

        Txold = np.gradient(Told, self.x, edge_order=2)
        Fxold = np.gradient(Fold, self.x, edge_order=2)
        dif_old = self.B*Fold
        adv_old = self.B*Told+self.Bx*Fold+self.B*Fxold
        BTx_old = self.B*Txold+self.Bx*Told

        # interior
        A[0, 2:] = + self.theta*np.minimum(adv[1:-1], 0)*hder[2:]/(alpha1[2:]*self.dx[1:]) + self.theta*dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[2:]/alpha1[2:]/self.dx[1:]
        A[1, 1:-1] = self.B[1:-1]/self.dt + self.theta*BTx[1:-1]*hder[1:-1]/alpha1[1:-1] + \
                     self.theta*np.maximum(adv[1:-1], 0)*hder[1:-1]/(alpha1[1:-1]*self.dx[:-1]) - self.theta*np.minimum(adv[1:-1], 0)*hder[1:-1]/(alpha1[1:-1]*self.dx[1:]) \
                     - self.theta*dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[1:-1]/alpha1[1:-1]*(1./self.dx[1:]+1./self.dx[:-1])
        A[2, :-2] = -self.theta*np.maximum(adv[1:-1], 0)*hder[:-2]/(alpha1[:-2]*self.dx[:-1]) + self.theta*dif[1:-1]/(0.5*(self.dx[1:]+self.dx[:-1]))*hder[:-2]/alpha1[:-2]/self.dx[:-1]

        rhs[1:-1] = self.B[1:-1]/self.dt*Sold[1:-1] - self.theta*BTx[1:-1]*beta[1:-1] - self.theta*np.maximum(adv[1:-1], 0)*(beta[1:-1]-beta[:-2])/self.dx[:-1] - self.theta*np.minimum(adv[1:-1], 0)*(beta[2:]-beta[1:-1])/self.dx[1:] \
                    -self.theta*dif[1:-1]*((beta[2:]-beta[1:-1])/self.dx[1:] - (beta[1:-1]-beta[:-2])/self.dx[:-1])/(0.5*(self.dx[1:]+self.dx[:-1])) \
                    + (1-self.theta) * (-BTx_old[1:-1]*fold[1:-1] - np.maximum(adv_old[1:-1], 0)*(fold[1:-1]-fold[:-2])/self.dx[:-1] - np.minimum(adv_old[1:-1], 0)*(fold[2:]-fold[1:-1])/self.dx[1:] \
                                      -dif_old[1:-1]*((fold[2:]-fold[1:-1])/self.dx[1:] - (fold[1:-1]-fold[:-2])/self.dx[:-1])/(0.5*(self.dx[1:]+self.dx[:-1])))
        rhs[1:-1] += -self.B[1:-1]*G[1:-1]

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
            A[1, -1] = self.B[-1]*T[-1]*hder[-1]/alpha1[-1] + 3./2.*self.B[-1]*F[-1]*hder[-1]/alpha1[-1]/self.dx[-1]
            A[2, -2] = -2.*self.B[-1]*F[-1]*hder[-2]/alpha1[-2]/self.dx[-1]
            A[3, -3] = 0.5*self.B[-1]*F[-1]*hder[-3]/alpha1[-3]/self.dx[-1]
            rhs[-1] = -self.B[-1]*T[-1]*beta[-1] - self.B[-1]*F[-1]*(3./2.*beta[-1] - 2.*beta[-2] + 0.5*beta[-3])/self.dx[-1]
            rhs[-1] += -self.B[-1]*G[-1]

            #   alternative first order
            # A[1, -1] = self.B[-1]*T[-1]*hder[-1]/alpha1[-1] + self.B[-1]*F[-1]*hder[-1]/alpha1[-1]/self.dx[-1]
            # A[2, -2] = -1.*self.B[-1]*F[-1]*hder[-2]/alpha1[-2]/self.dx[-1]
            # rhs[-1] = -self.B[-1]*T[-1]*beta[-1] - self.B[-1]*F[-1]*(1*beta[-1] - 1.*beta[-2])/self.dx[-1]
            # rhs[-1] += self.B[-1]*G[-1]


        try:
            S = scipy.linalg.solve_banded((2, 1), A, rhs, overwrite_ab=False, overwrite_b=False)
        except:
            print(Xold[:jmax+1])
            raise KnownError('Time integration failed.')


        f = self.erodibility_stock_relation(alpha2, S/alpha1)
        X = np.concatenate((f, S), axis=0)

        return X

    def interpretValues(self, values):
        """inpterpret values on input as space-separated list or as pure python input

        Parameters
            values - values to evaluate

        Returns:
            values - evaluated values
        """
        values = toList(values)

        # case 1: pure python: check for (, [, range, np.arange
        #   merge list to a string
        valString = ' '.join([str(f) for f in values])
        #   try to interpret as python string
        if any([i in valString for i in ['(', '[', ',', '/', '*', '+', '-']]):
            try:
                valuespy = None
                exec('valuespy ='+valString)
                return valuespy
            except Exception as e:
                raise KnownError('Failed to interpret input as python command in input', e)

        # case 2: else interpret as space-separated list
        else:
            return values

    # def stability_availability(self, A, T, F, D, aeq):
    #     """Calculates the eigenvalues and eigenvectors of the Sturm-Liousville problem related to the dynamic
    #     availability concept
    #
    #     Parameters
    #         A - variable (BF)_x/B
    #         T - advection function
    #         F - diffusion function
    #         D - see Eq. (5) above
    #         aeq  - availability in morphodynamic equilibrium
    #
    #     Returns:
    #          eigval - eigenvalues
    #          eigvec - eigenvectors
    #     """
    #     M = np.zeros((len(self.x), 1))
    #     # initialize Jacobian matrix with zeros
    #     jac = sps.dia_matrix((len(self.x), len(self.x)))
    #     # define the values for the diagonals for the interior points for the jacobian matrix to the generalized
    #     # eigenvalue problem
    #     R = (1 - self.FCAP * aeq[1:-1] * self.FSEA)**2
    #     M[1:-1] = self.gamma * (self.H[0] * R / (self.input.v('ell') + D[1:-1] * R * self.H[0]))
    #     jval_c = -2 * M * F / self.dx[0]**2
    #     jval_l = M * (F / self.dx[0] - (A - T) / 2.) / (self.dx[0])
    #     jval_r = M * (F / self.dx[0] + (A - T) / 2.) / (self.dx[0])
    #     # Modify jacobian matrix to use it for the standard eigenvalue problem.
    #     jval_c[-2] += 4 * jval_r[-2] / 3 # Sturm-Liousville modification
    #     jval_l[-2] += -jval_r[-2] / 3 # Sturm-Liousville modification
    #     jac += sps.diags([jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-1, 0, 1])
    #     jac = jac[1:-1, 1:-1]
    #     #Determine eigenvalues and eigenvectors
    #     jacd = jac.todense()
    #     eigval, eigvec = linalg.eig(jacd)
    #     return eigval, eigvec

    # def stability_stock(self, A, T, F, alpha1):
    #     """Calculates the eigenvalues and eigenvectors of the Sturm-Liouville problem related to the dynamic
    #     erodibility concept
    #
    #     Parameters
    #         A - variable (BF)_x/B
    #         T - advection function
    #         F - diffusion function
    #         aeq  - availability in morphodynamic equilibrium
    #         alpha1 - factor indicating the maximum amount of sediment in the water column in a tidally averaged sense
    #
    #     Returns:
    #          eigval - eigenvalues
    #          eigvec - eigenvectors
    #     """
    #     M = np.zeros((len(self.x), 1))
    #     # initialize Jacobian matrix with zeros
    #     jac = sps.dia_matrix((len(self.x), len(self.x)))
    #     # define the values for the diagonals for the interior points for the jacobian matrix to the generalized
    #     # eigenvalue problem
    #     M[1:-1] = self.Mhat / alpha1[1:-1]
    #     jval_c = -2 * M * F / self.dx[0]**2
    #     jval_l = M * (F / self.dx[0] - (A - T) / 2.) / (self.dx[0])
    #     jval_r = M * (F / self.dx[0] + (A - T) / 2.) / (self.dx[0])
    #     # Modify jacobian matrix to use it for the standard eigenvalue problem.
    #     jval_c[-2] += 4 * jval_r[-2] / 3 # Sturm-Liousville modification
    #     jval_l[-2] += -jval_r[-2] / 3 # Sturm-Liousville modification
    #     jac += sps.diags([jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-1, 0, 1])
    #     jac = jac[1:-1, 1:-1]
    #     #Determine eigenvalues and eigenvectors
    #     jacd = jac.todense()
    #     eigval, eigvec = linalg.eig(jacd)
    #     return eigval, eigvec


