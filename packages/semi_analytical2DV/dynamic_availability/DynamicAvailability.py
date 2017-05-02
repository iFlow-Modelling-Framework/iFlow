"""
DynamicAvailability

Date: 10/08/16
Authors: R.L. Brouwer
"""
import logging
import numpy as np
from scipy import integrate
import scipy.sparse as sps
import scipy.linalg as linalg
import nifty as ny
from src.util.diagnostics import KnownError
import sys
from nifty import toList
from erodibility_stock_relation import erodibility_stock_relation, erodibility_stock_relation_der
import numbers
from copy import copy


class DynamicAvailability_new:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the sediment concentration based on dynamic availability.
        We have to find an expression for f = g*a_eq, where a_eq is the availability when the system is in morphodynamic
        equilibrium and g is a function that describes the dependence on Q (or t). We solve for a and g from the
        following equation:

            ell * a_t + D(x) * (a_eq * g)_t = -gamma * a_eq * [F * g_xx + (A - T) * g_x],                       (1)

        where,

            ell     = l / H_0                                                                                   (2)
            gamma   = 1 / (mu_s * rho_s * (1-p) * H_0),                                                         (3)
            A       = (B * F)_x / B,                                                                            (4)
            D(x)    = gamma * H * (c00 + c20)_depth_averaged.                                                   (5)

         Returns:
             Dictionary with results. At least contains the variables listed as output in the registry
         """
        self.logger.info('Running module DynamicAvailability_new')

        # Initiate variables
        self.RHOS = self.input.v('RHOS')
        self.DS = self.input.v('DS')
        self.WS = self.input.v('ws')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')
        self.FINF = self.input.v('finf')
        self.CSEA = self.input.v('csea')
        self.FCAP = self.input.v('fcap')
        self.P = self.input.v('p')
        self.TOL = self.input.v('tol')
        self.ASTAR = self.input.v('astar')
        self.Kh = self.input.v('Kh')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.L
        self.dx = (self.x[1:]-self.x[:-1]).reshape(len(self.x)-1, 1)
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        self.H = (self.input.v('H', range(0, jmax+1)).reshape(jmax+1, 1) +
                  self.input.v('R', range(0, jmax+1)).reshape(jmax+1, 1))
        self.Hx = (self.input.d('H', range(0, jmax+1), dim='x').reshape(jmax+1, 1) +
                   self.input.d('R', range(0, jmax+1), dim='x').reshape(jmax+1, 1))
        self.B = self.input.v('B', range(0, jmax+1)).reshape(jmax+1, 1)
        self.Bx = self.input.d('B', range(0, jmax+1), dim='x').reshape(jmax+1, 1)
        self.sf = self.input.v('Roughness', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        self.Av0 = self.input.v('Av', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        self.Fc = (self.input.v('F', range(0, jmax+1)) -
                   self.input.v('F', 'diffusion_river', range(0, jmax+1))).reshape(jmax+1, 1)
        self.Tc = (self.input.v('T') - (self.input.v('T', 'river', range(0, jmax+1)) +
                                        self.input.v('T', 'river_river', range(0, jmax+1)) +
                                        self.input.v('T', 'diffusion_river', range(0, jmax+1)))).reshape(len(self.x), 1)
        self.u1river = np.real(self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax + 1), 0))
        self.Q_fromhydro = -np.trapz(self.u1river[-1, :], x=-self.zarr[-1, :]) * self.B[-1]
        self.TQ = self.input.v('T', 'river', range(0, jmax+1)).reshape(jmax+1, 1) / self.Q_fromhydro
        self.c00 = np.real(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 0))
        self.c04 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 2)
        # self.ALPHA2 = (abs(np.trapz(self.c04, x=-self.zarr, axis=1)) / (np.trapz(self.c00, x=-self.zarr, axis=1)) + 1.e-15).reshape(len(self.x), 1)
        # self.ALPHA2[-1] = 3 * (self.ALPHA2[-2]-self.ALPHA2[-3]) + self.ALPHA2[-4] #c04 and c00 are zero at x=L, but alpha2 remains finite based on backward euler and central differences
        # self.FSEA = self.CSEA * self.H[0] / np.trapz(self.c00[0, :], dx=-self.zarr[0, 1])
        # # self.FSEA = self.CSEA / (np.mean(self.c00[0, :]) * self.FINF)
        # # Calculate constant factor gamma
        # self.gamma = 1. / (self.RHOS * (1. - self.P) * self.H[0])

        #load time serie Q
        self.dt = self.interpretValues(self.input.v('dt'))
        if self.input.v('t') is not None:
            self.t = self.interpretValues(self.input.v('t'))
        self.Q = self.interpretValues(self.input.v('Q'))

        self.logger.info('Running time-integrator')

        vars = self.implicit()

        d = {}
        for key, value in vars.iteritems():
            d[key] = value
        return d

    def init_stock(self, aeq, alpha1):
        """Initiates the solution vector X = (g, S), when describing the amount of sediment in the system as a function
        of the stock S

        Parameters:
            aeq - availability at morphodynamic availability
            alpha1 - depth-integrated, tide-averaged concentration f_inf * int_-H^R <c> dz

        Returns:
            X - solution vector (g, S), with g and stock S
        """
        # initiate global variables alpha_2 and f_sea
        if isinstance(self.input.v('alpha2'), numbers.Real):
            self.ALPHA2 = self.input.v('alpha2')
        else:
            self.ALPHA2 = (abs(np.trapz(self.c04, x=-self.zarr, axis=1)) /
                           (np.trapz(self.c00, x=-self.zarr, axis=1) + 1.e-15)).reshape(len(self.x), 1)
            self.ALPHA2[-1] = 3 * (self.ALPHA2[-2]-self.ALPHA2[-3]) + self.ALPHA2[-4] #c04 and c00 are zero at x=L, but alpha2 remains finite based on backward euler and central differences
        self.FSEA = self.CSEA / (np.mean(self.c00[0, :]) * self.FINF)
        # Calculate variable alpha_1
        # ALPHA1 = self.FINF * np.trapz(self.c00 + c20, x=-self.zarr, axis=1).reshape(len(self.x), 1)
        # Initialize vector g
        g = self.interpretValues(self.input.v('ginit')).reshape(len(self.x), 1)
        # define initial erodibility f
        f = aeq * g
        if self.input.v('concept')[1] == 'approximation':
            # Initiate stock S with approximation S = alpha1 * Shat = alpha1 * f / beta * (1 - f)
            Shat = f / (1. - f)
        elif self.input.v('concept')[1] == 'exact':
            # Initiate stock S with exact expression (see notes Yoeri)
            Shat = f
            F = erodibility_stock_relation(self.ALPHA2, Shat) - f
            # Newton-Raphson iteration towards actual Shat
            while max(abs(F)) > self.TOL:
                dfdS = erodibility_stock_relation_der(self.ALPHA2, Shat)
                Shat = Shat - F / dfdS
                F = erodibility_stock_relation(self.ALPHA2, Shat) - f
        # Define solution vector X = (g, S)
        X = np.append(g, Shat).reshape(2 * len(self.x), 1)
        return X

    def init_availability(self, aeq):
        """Initiates the solution vector X = (g, a), when describing the amount of sediment in the system as a function
        of the availability a

        Parameters:
            aeq - availability at morphodynamic availability

        Returns:
            X - solution vector (g, a), with g and availability
        """
        # Initiate global variables gamma and f_sea
        self.gamma = 1. / (self.RHOS * (1. - self.P) * self.H[0])
        self.FSEA = self.FSEA = self.CSEA / np.mean(self.c00[0, :])
        # Initialize vector g
        g = self.interpretValues(self.input.v('ginit'))
        # Define availability as: a = f / (1 - f_sea * f) or equivalently f = a / (1 + f_sea * a)
        a = aeq * self.FSEA / (1. - self.FCAP * aeq * self.FSEA)
        # Define solution vector X = (g, a)
        X = np.append(g, a).reshape(2*len(self.x), 1)
        return X

    def implicit(self):
        """Calculates the time-evolution of the sediment distribution in an estuary using an implicit solution method

        Returns:
             Xt - solution vector X = (g, a) or X = (g, S) as a function of a time-dependent forcing variable, e.g. Q(t)
             Ft - diffusion function F as a function of a time-dependent forcing variable, e.g. Q(t)
             Tt - advection function T as a function of a time-dependent forcing variable, e.g. Q(t)
             etc.
        """
        # Calculate transport functions F and T
        T_old, F_old, A_old, C20_old = self.transport_terms(self.Q[0])
        Tt = []
        Tt.append(T_old)
        Ft = []
        Ft.append(F_old)
        # Calculate equilibrium availability a_eq
        aeq_old = self.availability(F_old, T_old, self.Q[0])
        aeq_old = aeq_old / aeq_old[0]  # scale the availability such that aeq=0 at x=0

        ##### Initialise solution vector X depending on which concept is used ######
        Xt = []
        if self.input.v('concept') == 'availability':
            # Solution matrix X = (g, a)
            X = self.init_availability(aeq_old)
            X_old = copy(X)
            Xt.append(X)
            # Define factor D
            D_old = self.gamma * self.H * np.mean(self.c00 + C20_old, axis=1).reshape(len(self.x), 1)
        elif self.input.v('concept')[0] == 'stock':
            ALPHA1_old = self.FINF * np.trapz(self.c00 + C20_old, x=-self.zarr, axis=1).reshape(len(self.x), 1)
            # Solution matrix X = (g, S)
            X = self.init_stock(aeq_old, ALPHA1_old)
            X_old = copy(X)
            Xt.append(X_old * np.append(np.ones(len(self.x)), ALPHA1_old).reshape(2*len(self.x), 1))

        ft = []
        ft.append(aeq_old * X_old[:len(self.x)])
        aeqt = []
        aeqt.append(aeq_old)
        # Calculate river transport term
        Fr = self.interpretValues(self.input.v('rivertrans'))
        if len(Fr) == 1.:
            Fr = Fr * np.ones(len(self.Q))
        fr_old = -aeq_old * Fr[0] * integrate.cumtrapz(1. / (self.B * F_old * aeq_old), x=self.x, axis=0, initial=0)
        # # Calculate flux over open boundary
        # flux = []
        # flux_old = self.B * F_old * aeq_old * np.gradient(X_old[:len(self.x)], self.dx[0], axis=0, edge_order=2).reshape(len(self.x), 1)
        # flux.append(flux_old)
        # # Total sediment volume in the bottom layer
        # Msed = []
        # M_old = self.RHOS * (1 - self.P) * np.trapz(self.B * self.input.v('ell') * self.H[0] * X_old[len(self.x):], x=self.x.reshape(len(self.x), 1), axis=0)
        # Msed.append(M_old)
        # Calculate tidally-averaged actual and equilibrium sediment concentration
        c0bar = []
        c0bar.append(self.FINF * (self.c00 + C20_old) * aeq_old * X_old[:len(self.x)])
        c0bareq = []
        c0bareq.append(self.FINF * (self.c00 + C20_old) * aeq_old * self.FSEA)
        eigvals = []
        eigvecs = []
        # c0hat = []
        # c0hat.append(self.c00[:, 0] + C20_old[:, 0])
        for i, q in enumerate(self.Q[1:]):
            T, F, A, C20 = self.transport_terms(q)
            aeq = self.availability(F, T, q)
            aeq = aeq / aeq[0]  # scale the availability such that aeq=0 at x=0
            fr = -aeq * Fr[i+1] * integrate.cumtrapz(1. / (self.B * F * aeq), x=self.x, axis=0, initial=0)

            if self.input.v('concept') == 'availability':
                D = self.gamma * self.H * np.mean(self.c00 + C20, axis=1).reshape(len(self.x), 1)
                Xvec, jac = self.jacvec_availability(A, A_old, T, T_old, F, F_old, aeq, aeq_old, D, D_old, X, X_old, fr, fr_old)
            elif self.input.v('concept')[0] == 'stock':
                ALPHA1 = self.FINF * np.trapz(self.c00 + C20, x=-self.zarr, axis=1).reshape(len(self.x), 1)
                Xvec, jac = self.jacvec_stock(A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old, ALPHA1_old, ALPHA1, fr, fr_old)

            while max(abs(Xvec)) > self.TOL:
                dX = linalg.solve(jac, Xvec)
                X = X - dX.reshape(len(X), 1)
                if self.input.v('concept') == 'availability':
                    Xvec, jac = self.jacvec_availability(A, A_old, T, T_old, F, F_old, aeq, aeq_old, D, D_old, X, X_old, fr, fr_old)
                elif self.input.v('concept')[0] == 'stock':
                    Xvec, jac = self.jacvec_stock(A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old, ALPHA1_old, ALPHA1, fr, fr_old)

            # Determine eigenvalues of the system for a certain discharge
            if self.input.v('concept') == 'availability' and self.FCAP == 0:
                eigval, eigvec = self.stability_availability(A, T, F, D, aeq)
                eigvals.append(eigval)
                eigvecs.append(eigvec)
            elif self.input.v('concept')[0] == 'stock':
                eigval, eigvec = self.stability_stock(A, T, F, ALPHA1)
                eigvals.append(eigval)
                eigvecs.append(eigvec)
            else:
                eigvals.append(0.)
                eigvals.append(0.)

            aeq_old = aeq
            X_old = copy(X)
            T_old = copy(T)
            F_old = copy(F)
            A_old = copy(A)
            if self.input.v('concept') == 'availability':
                D_old = copy(D)
                Xt.append(X)
            elif self.input.v('concept')[0] == 'stock':
                ALPHA1_old = copy(ALPHA1)
                Xt.append(X * np.append(np.ones(len(self.x)), ALPHA1).reshape(2*len(self.x), 1))
            fnew = aeq * X[:len(self.x)]
            ft.append(fnew)
            c0barnew = self.FINF * (self.c00 + C20) * fnew.reshape(len(self.x), 1)
            c0bareqnew = self.FINF * (self.c00 + C20) * (aeq * self.FSEA).reshape(len(self.x), 1)
            # flux.append(self.B * F * aeq * np.gradient(X[:len(self.x)], self.dx[0], axis=0, edge_order=2).reshape(len(self.x), 1))
            # Msed.append(self.RHOS * (1 - self.P) * np.trapz(self.B * self.input.v('ell') * self.H[0] * X[len(self.x):], x=self.x.reshape(len(self.x), 1), axis=0))
            c0bar.append(c0barnew)
            c0bareq.append(c0bareqnew)
            Tt.append(T)
            Ft.append(F)
            aeqt.append(aeq)
            # c0hat.append(self.c00[:, 0] + C20[:, 0])

            # display progress
            if i % np.floor(len(self.Q[1:]) / 10.) == 0:
                percent = float(i) / len(self.Q[1:])
                hashes = '#' * int(round(percent * 10))
                spaces = ' ' * (10 - len(hashes))
                sys.stdout.write("\rProgress: [{0}]{1}%".format(hashes + spaces, int(round(percent * 100))))
                sys.stdout.flush()
        St = np.array(Xt)[:, len(self.x):, :]
        gt = np.array(Xt)[:, :len(self.x), :]
        return {'St': St, 'gt': gt, 'aeqt': np.array(aeqt), 'Tt': np.array(Tt), 'Ft': np.array(Ft),
                'ft': np.array(ft), 'c0bar': {'t': np.array(c0bar), 'eq': np.array(c0bareq)},
                'eigs': {'eigvals': np.array(eigvals),'eigvecs': np.array(eigvecs)}, 'alpha2': self.ALPHA2}

    def transport_terms(self, q):
        """Calculates the transport terms T, F and A and the second-order sediment concentration c20 based on the time-
        dependent forcing variable, e.g. Q(t)

        Parameters:
            q - river discharge Q

        Returns:
            T - advection function
            F - diffusion function
            A - variable appearing the the Exner equation, i.e. A = (BF)_x / B
            C20 - second-order sediment concentration
        """
        Triver_river, C20 = self.river_river_interaction(q)
        C20x, __ = np.gradient(C20, self.x[1], edge_order=2)
        Tdiff_river = np.real(-np.trapz(self.Kh * C20x, x=-self.zarr, axis=1)).reshape(len(self.x), 1)
        T = self.Tc + self.TQ * q + Triver_river + Tdiff_river
        # Calculate and append diffusive coefficient F
        Fdiff_river = np.real(-np.trapz(self.Kh * C20, x=-self.zarr, axis=1))
        F = self.Fc + Fdiff_river.reshape(len(self.x), 1)
        Fx = np.gradient(F, self.dx[0], axis=0, edge_order=2)
        A = Fx + self.Bx * F / self.B
        return T, F, A, C20

    def river_river_interaction(self, q):
        """Calculates the transport and second-order sediment concentration due to the river-river interaction

        Parameters:
            q - river discharge

        Returns:
            Tr - transport due to river-river interaction
            c20 - second-order sediment concentration due to river-river interaction
        """
        u0b = self.input.v('u0', 'tide', range(0, len(self.x)), len(self.z) - 1, 1)
        ur = self.u1river * q / self.Q_fromhydro
        u1b = ur[:, -1]
        time = np.linspace(0, 2 * np.pi, 100)
        utid = np.zeros((len(self.x), len(time))).astype('complex')
        ucomb = np.zeros((len(self.x), len(time))).astype('complex')
        for i, t in enumerate(time):
            utid[:, i] = 0.5 * (u0b * np.exp(1j * t) + np.conj(u0b) * np.exp(-1j * t))  # YMD
            ucomb[:, i] = u1b + 0.5 * (u0b * np.exp(1j * t) + np.conj(u0b) * np.exp(-1j * t))
        uabs_tid = np.mean(np.abs(utid), axis=1)
        uabs_tot = np.mean(np.abs(ucomb), axis=1)
        uabs_eps = uabs_tot.reshape(len(self.x), 1) - uabs_tid.reshape(len(self.x), 1)
        c20 = np.real((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_eps *
               np.exp(-self.WS * (self.H + self.zarr) / self.Av0))

        Tr = np.real(np.trapz(ur * c20, x=-self.zarr, axis=1)).reshape(len(self.x), 1)
        return Tr, c20

    def availability(self, F, T, Q):
        """Calculates the availability of sediment needed to derive the sediment concentration

        Parameters:
            F - diffusive coefficient in the availability equation that goes with a_x
            T - coefficient (advective, diffusive and stokes) in the availability equation that goes with a

        Returns:
            a - availability of sediment throughout the estuary
        """
        # Exponent in the availability function.
        if Q > 0:
            exponent = np.exp(-integrate.cumtrapz(T / F, dx=self.dx, axis=0, initial=0))
        else:
            exponent = np.append(np.exp(-np.append(0, integrate.cumtrapz(T / F, dx=self.dx, axis=0)[:-1])), 0).reshape(len(self.x), 1)
        A = (self.ASTAR * np.trapz(self.B, dx=self.dx, axis=0) /
             np.trapz(self.B * exponent, dx=self.dx, axis=0))
        a = A * exponent
        return a

    def jacvec_availability(self, A, A_old, T, T_old, F, F_old, aeq, aeq_old, D, D_old, X, X_old, fr, fr_old):
        """Calculates the vector containing the Exner equation and the algebraic expression between a and f, and the
        Jacobian matrix needed to solve for g and a

        Parameters:
            A, A_old - variable (BF)_x/B on the old and current time step
            T, T_old - transport function T on the old and current time step
            F, F_old -  diffusion furnction F on the old and current time step
            aeq, aeq_old - availability in morphodynamic equilibrium on the old and current time step
            D, D_old - factor D, Eq. (5) above, on the old and current time step
            X, X_old - solution vector X = (g, a) on the old and current time step
            fr, fr_old - erodibility f due to river transport term on the old and current time step

        Returns:
             Xvec - vector containing the Exner equation and the algebraic expression between a and f
             jac - corresponding Jacobian matrix
        """
        # Initiate jacobian matrix, Xvec and local variable N
        jac = sps.csc_matrix((len(X), len(X)))
        Xvec, N = np.zeros((len(X), 1)), np.zeros((len(self.x), 1))
        # define length of xgrid
        lx = len(self.x)
        ivec = range(1, lx-1)  #interior points for g
        jvec = range(lx+1, 2*lx-1)  #interior points for a
        # Define local variable N^(n+1) = gamma * a_eq * [F*(g - 2g + g) / dx**2 + (A - T) * (g - g) / 2 * dx]
        N[ivec] = self.gamma * aeq[1:-1] * (F[1:-1] * (X[2:lx] - 2 * X[ivec] + X[:lx-2]) / self.dx[0]**2 +
                                            (A[1:-1] - T[1:-1]) * (X[2:lx] - X[:lx-2]) / (2 * self.dx[0]))
        # Fill interior points of Xvec related to N in Eq. (1)
        Xvec[ivec] += self.dt * N[ivec] / 2.
        # Fill interior points of Xvec related to D * (a_eq * g)_t in Eq. (1)
        Xvec[ivec] += D[1:-1] * aeq[1:-1] * X[ivec]
        # # Fill interior points of Xvec related to river flux
        # Xvec[ivec] -= self.dt * self.gamma * self.input.v('rivertrans') * (X[2:lx] - X[:lx-2]) / (2 * self.B[ivec] * self.dx[0])
        # Fill interior points of the jacobian matrix related to g for Eq. (1)
        jval_c_g = D[1:-1, 0] * aeq[1:-1, 0] - self.dt * self.gamma * aeq[1:-1, 0] * F[1:-1, 0] / self.dx[0]**2
        jac += sps.csc_matrix((jval_c_g, (ivec, ivec)), shape=jac.shape)
        jval_l_g = (self.dt * self.gamma * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) /
                    (2 * self.dx[0]))
        # jval_l_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) +
        #                                    self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_l_g, (ivec, range(lx-2))), shape=jac.shape)
        jval_r_g = self.dt * self.gamma * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) / (2 * self.dx[0])
        # jval_r_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) -
        #                         self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_r_g, (ivec, range(2, lx))), shape=jac.shape)

        # Boundary condition at sea
        Xvec[0] = X[0] - self.FSEA
        jac += sps.csc_matrix(([1.], ([0.], [0.])), shape=jac.shape)
        # Boundary condition at weir
        Xvec[lx-1] = (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3])
        jac += sps.csc_matrix(([3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Xvec[lx-1] = self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) + 2. * self.dx[0] * self.input.v('rivertrans')
        # jac += sps.csc_matrix((self.B[-1] * F[-1] * aeq[-1] * [3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Xvec[lx-1] = (self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) -
        #               2. * self.dx[0] * self.input.v('rivertrans') * (X[lx-1] - 1))
        # jval_g = self.B[-1] * F[-1] * aeq[-1] * [(3. - 2. * self.dx[0] * self.input.v('rivertrans'))[0], -4., 1.]
        # jac += sps.csc_matrix((jval_g, ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Fill interior points of Xvec and diagonal of jacobian related to ell*a in Eq. (1)
        ell = self.input.v('ell')
        Xvec[ivec] += ell * X[jvec]
        jac += sps.csc_matrix((ell * np.ones(lx-2), (ivec, jvec)), shape=jac.shape)
        # Fill Xvec and jacobian for the algebraic relation between g and a, Eq. (2)
        Xvec[lx:] = aeq * X[:lx] + self.FCAP * aeq * X[:lx] * X[lx:] - X[lx:]
        # Xvec[lx:] += fr + self.FCAP * fr * X[lx:]
        jac += sps.csc_matrix((self.FCAP * aeq[:, 0] * X[:lx, 0] - 1., (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        jac += sps.csc_matrix((aeq[:, 0] + self.FCAP * aeq[:, 0] * X[lx:, 0], (range(lx, 2*lx), range(lx))), shape=jac.shape)
        # jac += sps.csc_matrix((self.FCAP * fr[:, 0], (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        # Convert jacobian matrix to dense matrix
        jac = jac.todense()
        # Inhomogeneous part of the PDE related to a and g at interior points
        Xvec[ivec] -= ell * X_old[jvec]
        N[ivec] = self.gamma * aeq_old[1:-1] * (F_old[1:-1] * (X_old[2:lx] - 2 * X_old[ivec] + X_old[:lx-2]) / self.dx[0]**2 +
                                                (A_old[1:-1] - T_old[1:-1]) * (X_old[2:lx] - X_old[:lx-2]) / (2 * self.dx[0]))
        Xvec[ivec] += self.dt * N[ivec] / 2. - D_old[1:-1] * aeq_old[1:-1] * X_old[ivec]
        # Extra term in the inhomogeneous part due to the river transport
        # Xvec[ivec] += D[1:-1] * fr[1:-1] - D_old[1:-1] * fr_old[1:-1]
        return Xvec, jac

    def jacvec_stock(self, A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old, alpha1_old, alpha1, fr, fr_old):
        """Calculates the vector containing the Exner equation and the algebraic expression between S and f, and the
        Jacobian matrix needed to solve for g and S

        Parameters:
            A, A_old - variable (BF)_x/B on the old and current time step
            T, T_old - transport function T on the old and current time step
            F, F_old -  diffusion furnction F on the old and current time step
            aeq, aeq_old - availability in morphodynamic equilibrium on the old and current time step
            X, X_old - solution vector X = (g, a) on the old and current time step
            alpha1, alpha1_old - alpha1 on the old and current time step
            fr, fr_old - erodibility f due to river transport term on the old and current time step

        Returns:
             Xvec - vector containing the Exner equation and the algebraic expression between S and f
             jac - corresponding Jacobian matrix
        """
        # Initiate jacobian matrix, Xvec and local variable N
        jac = sps.csc_matrix((len(X), len(X)))
        Xvec, N = np.zeros((len(X), 1)), np.zeros((len(self.x), 1))
        # define length of xgrid
        lx = len(self.x)
        ivec = range(1, lx-1)  #interior points for g
        jvec = range(lx+1, 2*lx-1)  #interior points for S
        # Define local variable N^(n+1) = f_inf * a_eq * [F*(g - 2g + g) / dx**2 + (A - T) * (g - g) / 2 * dx]
        N[ivec] = self.FINF * aeq[1:-1] * (F[1:-1] * (X[2:lx] - 2 * X[ivec] + X[:lx-2]) / self.dx[0]**2 +
                                            (A[1:-1] - T[1:-1]) * (X[2:lx] - X[:lx-2]) / (2 * self.dx[0]))
        # Fill interior points of Xvec related to N in Eq. (1)
        Xvec[ivec] += self.dt * N[ivec] / (2. * alpha1[ivec])

        # # Fill interior points of Xvec related to river flux
        # Xvec[ivec] -= self.dt * self.gamma * self.input.v('rivertrans') * (X[2:lx] - X[:lx-2]) / (2 * self.B[ivec] * self.dx[0])

        # Fill interior points of the jacobian matrix related to g for Eq. (1)
        jval_c_g = - self.dt * self.FINF * aeq[1:-1, 0] * F[1:-1, 0] / (self.dx[0]**2 * alpha1[1:-1, 0])
        jac += sps.csc_matrix((jval_c_g, (ivec, ivec)), shape=jac.shape)
        jval_l_g = (self.dt * self.FINF * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) /
                    (2 * alpha1[1:-1, 0] * self.dx[0]))
        # jval_l_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) +
        #                                    self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_l_g, (ivec, range(lx-2))), shape=jac.shape)
        jval_r_g = self.dt * self.FINF * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) / \
                   (2 * alpha1[1:-1, 0] * self.dx[0])
        # jval_r_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) -
        #                         self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_r_g, (ivec, range(2, lx))), shape=jac.shape)

        # Boundary condition at sea
        Xvec[0] = X[0] - self.FSEA
        jac += sps.csc_matrix(([1.], ([0.], [0.])), shape=jac.shape)
        # Boundary condition at weir
        Xvec[lx-1] = (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3])
        jac += sps.csc_matrix(([3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)

        # Xvec[lx-1] = self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) + 2. * self.dx[0] * self.input.v('rivertrans')
        # jac += sps.csc_matrix((self.B[-1] * F[-1] * aeq[-1] * [3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Xvec[lx-1] = (self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) -
        #               2. * self.dx[0] * self.input.v('rivertrans') * (X[lx-1] - 1))
        # jval_g = self.B[-1] * F[-1] * aeq[-1] * [(3. - 2. * self.dx[0] * self.input.v('rivertrans'))[0], -4., 1.]
        # jac += sps.csc_matrix((jval_g, ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)

        # Fill interior points of Xvec and diagonal of jacobian related to S in Eq. (1)
        Xvec[ivec] += X[jvec]
        jac += sps.csc_matrix((np.ones(lx-2), (ivec, jvec)), shape=jac.shape)

        if self.input.v('concept')[1] == 'approximation':
            # Fill Xvec and jacobian for the algebraic relation between g and stock S when using the approximation S = alpha1 * f / (1 + beta * f)
            Xvec[lx:] = aeq * X[:lx] * (1. + X[lx:]) - X[lx:]
            # Xvec[lx:] += fr + self.FCAP * fr * X[lx:]
            jac += sps.csc_matrix(((aeq * X[:lx] - 1.)[:, 0], (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
            jac += sps.csc_matrix((aeq[:, 0] * (1. + X[lx:])[:, 0], (range(lx, 2*lx), range(lx))), shape=jac.shape)
            # jac += sps.csc_matrix((self.FCAP * fr[:, 0], (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        elif self.input.v('concept')[1] == 'exact':
            # Fill Xvec and jacobian for the algebraic relation between g and stock S when using the exact relation between S and f
            Xvec[lx:] = aeq * X[:lx] - erodibility_stock_relation(self.ALPHA2, X[lx:])
            jac += sps.csc_matrix((-erodibility_stock_relation_der(self.ALPHA2, X[lx:])[:, 0], (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
            jac += sps.csc_matrix((aeq[:, 0], (range(lx, 2*lx), range(lx))), shape=jac.shape)
            # jac += sps.csc_matrix((self.FCAP * fr[:, 0], (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        # Convert jacobian matrix to dense matrix
        jac = jac.todense()
        # Inhomogeneous part of the PDE related to g and S at interior points
        Xvec[ivec] -= alpha1_old[ivec] * X_old[jvec] / alpha1[ivec]
        N[ivec] = self.FINF * aeq_old[1:-1] * (F_old[1:-1] * (X_old[2:lx] - 2 * X_old[ivec] + X_old[:lx-2]) / self.dx[0]**2 +
                                                (A_old[1:-1] - T_old[1:-1]) * (X_old[2:lx] - X_old[:lx-2]) / (2 * self.dx[0]))
        Xvec[ivec] += self.dt * N[ivec] / (2. * alpha1[ivec])
        # Extra term in the inhomogeneous part due to the river transport
        # Xvec[ivec] += D[1:-1] * fr[1:-1] - D_old[1:-1] * fr_old[1:-1]
        return Xvec, jac

    def stability_availability(self, A, T, F, D, aeq):
        """Calculates the eigenvalues and eigenvectors of the Sturm-Liousville problem related to the dynamic 
        availability concept
        
        Parameters
            A - variable (BF)_x/B
            T - advection function
            F - diffusion function
            D - see Eq. (5) above
            aeq  - availability in morphodynamic equilibrium

        Returns:
             eigval - eigenvalues
             eigvec - eigenvectors
        """
        M = np.zeros((len(self.x), 1))
        # initialize Jacobian matrix with zeros
        jac = sps.dia_matrix((len(self.x), len(self.x)))
        # define the values for the diagonals for the interior points for the jacobian matrix to the generalized
        # eigenvalue problem
        R = (1 - self.FCAP * aeq[1:-1] * self.FSEA)**2
        M[1:-1] = self.gamma * (self.H[0] * R / (self.input.v('ell') + D[1:-1] * R * self.H[0]))
        jval_c = -2 * M * F / self.dx[0]**2
        jval_l = M * (F / self.dx[0] - (A - T) / 2.) / (self.dx[0])
        jval_r = M * (F / self.dx[0] + (A - T) / 2.) / (self.dx[0])
        # Modify jacobian matrix to use it for the standard eigenvalue problem.
        jval_c[-2] += 4 * jval_r[-2] / 3 # Sturm-Liousville modification
        jval_l[-2] += -jval_r[-2] / 3 # Sturm-Liousville modification
        jac += sps.diags([jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-1, 0, 1])
        jac = jac[1:-1, 1:-1]
        #Determine eigenvalues and eigenvectors
        jacd = jac.todense()
        eigval, eigvec = linalg.eig(jacd)
        return eigval, eigvec

    def stability_stock(self, A, T, F, alpha1):
        """Calculates the eigenvalues and eigenvectors of the Sturm-Liousville problem related to the dynamic
        erodibility concept

        Parameters
            A - variable (BF)_x/B
            T - advection function
            F - diffusion function
            aeq  - availability in morphodynamic equilibrium
            alpha1 - factor indicating the maximum amount of sediment in the water column in a tidally averaged sense

        Returns:
             eigval - eigenvalues
             eigvec - eigenvectors
        """
        M = np.zeros((len(self.x), 1))
        # initialize Jacobian matrix with zeros
        jac = sps.dia_matrix((len(self.x), len(self.x)))
        # define the values for the diagonals for the interior points for the jacobian matrix to the generalized
        # eigenvalue problem
        M[1:-1] = self.FINF / alpha1[1:-1]
        jval_c = -2 * M * F / self.dx[0]**2
        jval_l = M * (F / self.dx[0] - (A - T) / 2.) / (self.dx[0])
        jval_r = M * (F / self.dx[0] + (A - T) / 2.) / (self.dx[0])
        # Modify jacobian matrix to use it for the standard eigenvalue problem.
        jval_c[-2] += 4 * jval_r[-2] / 3 # Sturm-Liousville modification
        jval_l[-2] += -jval_r[-2] / 3 # Sturm-Liousville modification
        jac += sps.diags([jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-1, 0, 1])
        jac = jac[1:-1, 1:-1]
        #Determine eigenvalues and eigenvectors
        jacd = jac.todense()
        eigval, eigvec = linalg.eig(jacd)
        return eigval, eigvec

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
                try: errorString = ': '+ e.msg
                except: errorString = ''
                raise KnownError('Failed to interpret input as python command %s in input: %s' %(errorString, valString), e)

        # case 2: else interpret as space-separated list
        else:
            return values