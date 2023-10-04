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
import numbers
from copy import copy
from .EquilibriumAvailability import EquilibriumAvailability
from src.DataContainer import DataContainer


class DynamicAvailability(EquilibriumAvailability):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        EquilibriumAvailability.__init__(self, input)
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the sediment concentration based on dynamic erodibility. Hereby,
        we solve the following three equations:

        S_t = - [Flux_x + Flux * (B_x/B)]                    (1)

        Flux = T*f + F*f_x                                   (2)

        f = f(Stilde)                                        (3)

        with:

        S      = sediment stock, which is the total amount of sediment in the water column and the erodible bottom
        Flux   = sediment transport
        f      = relative sediment erodibility
        B      = estuary width
        T      = transport function
        F      = diffusion function
        Stilde = S / Chat, with Chat is the subtidal carrying capacity

         Returns:
             Dictionary with results. At least contains the variables listed as output in the registry
         """
        self.logger.info('Running module DynamicAvailability')

        ## Initiate variables
        # general variables
        self.RHOS = self.input.v('RHOS')
        self.DS = self.input.v('DS')
        self.WS = self.input.v('ws0')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')
        self.Mhat = self.input.v('Mhat')
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

        # velocity
        self.u1river = np.real(self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax + 1), 0))
        self.Q_fromhydro = -np.trapz(self.u1river[-1, :], x=-self.zarr[-1, :]) * self.B[-1]

        # c hat
        self.c00 = np.real(self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 0))
        self.c04 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), 2)

        ## Compute transport (in superclass)
        d = self.compute_transport()
        dc = DataContainer(d)
        dc.merge(self.input.slice('grid'))
        self.Fc = (dc.v('F', range(0, jmax+1)) -
                   dc.v('F', 'diffusion_river', range(0, jmax+1))).reshape(jmax+1, 1)
        self.Tc = (dc.v('T') - (dc.v('T', 'river', range(0, jmax+1)) +
                                dc.v('T', 'river_river', range(0, jmax+1)) +
                                dc.v('T', 'diffusion_river', range(0, jmax+1)))).reshape(len(self.x), 1)
        self.TQ = dc.v('T', 'river', range(0, jmax+1)).reshape(jmax+1, 1) / self.Q_fromhydro

        ## User input
        #TODO: make a consistent and effective script that handles user input of the discharge time series or any other time-dependent variable
        #load time serie Q
        self.dt = self.interpretValues(self.input.v('dt'))
        if self.input.v('t') is not None:
            self.t = self.interpretValues(self.input.v('t'))
        self.Q = self.interpretValues(self.input.v('Q'))

        ## Run
        self.logger.info('Running time-integrator')
        vars = self.implicit()

        ## Collect output
        d = {}
        for key in vars.keys():
            d[key] = vars[key]
        return d

    def init_stock(self, T, F, Chat):
        """Initiates the solution vector X = (S, Flux, f), when describing the amount of sediment in the system as a function
        of the stock S

        Parameters:
            T - transport function
            F - diffusion function
            Chat - depth-integrated, tide-averaged concentration Mhat * int_-H^R <c> dz

        Returns:
            X - solution vector (S, Flux, f), with stock S, Flux and erodibility f
        """
        # initiate global variables alpha_2 and f_sea
        if isinstance(self.input.v('alpha2'), numbers.Real):
            self.ALPHA2 = self.input.v('alpha2')
        else:
            self.ALPHA2 = (abs(self.Mhat * np.trapz(self.c04, x=-self.zarr, axis=1)) /
                           (Chat[:, 0] + 1.e-15)).reshape(len(self.x), 1)
            self.ALPHA2[-1] = 3 * (self.ALPHA2[-2]-self.ALPHA2[-3]) + self.ALPHA2[-4] #c04 and c00 are zero at x=L, but alpha2 remains finite based on backward euler and central differences
        self.FSEA = self.CSEA / (np.mean(self.c00[0, :]) * self.Mhat)
        # Initialize erodibility f
        f = self.interpretValues(self.input.v('finit')).reshape(len(self.x), 1)
        if self.input.v('concept')[1] == 'approximation':
            # Initiate stock S with approximation S = alpha1 * Shat = alpha1 * f / beta * (1 - f)
            Shat = f / (1. - f)
        elif self.input.v('concept')[1] == 'exact':
            # Initiate stock S with exact expression (see notes Yoeri)
            Shat = f
            test_fun = self.erodibility_stock_relation(self.ALPHA2, Shat) - f
            # Newton-Raphson iteration towards actual Shat
            while max(abs(test_fun)) > self.TOL:
                dfdS = self.erodibility_stock_relation_der(self.ALPHA2, Shat)
                Shat = Shat - test_fun / dfdS
                test_fun = self.erodibility_stock_relation(self.ALPHA2, Shat) - f
        # Initialize flux F
        Flux = T * f + F * np.gradient(f, self.dx[0,0], axis=0, edge_order=2)
        # Define solution vector X = (S, Flux, f)
        X = np.concatenate((Shat, Flux, f)).reshape(3 * len(self.x), 1)
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
        T, F, __, C20 = self.transport_terms(self.Q[0])
        Tt = []
        Tt.append(T)
        Ft = []
        Ft.append(F)

        ##### Initialise solution vector X ######
        Xt = []
        if self.input.v('concept')[0] == 'stock':
            Chat_old = self.Mhat * np.trapz(self.c00 + C20, x=-self.zarr, axis=1).reshape(len(self.x), 1)
            # Solution matrix X = (S, Flux, f)
            X = self.init_stock(T, F, Chat_old)
            X_old = copy(X)
            Xt.append(X_old * np.concatenate((Chat_old[:, 0], np.ones(len(self.x)), np.ones(len(self.x)))).reshape(3*len(self.x), 1))
        else:
            raise KnownError('Availability concept is not implemented in the upwind scheme. Use module DynamicAvailability.')

        # Calculate tidally-averaged sediment concentration
        #TODO: In this module, only the tidally-averaged concentration is returned. If we want full output of all concentration contribution, chat needs to be time-dependent!!
        c0bar = []
        c0bar.append(self.Mhat * (self.c00 + C20) * X_old[2*len(self.x):])
        for i, q in enumerate(self.Q[1:]):
            T, F, __, C20 = self.transport_terms(q)

            if self.input.v('concept')[0] == 'stock':
                Chat = self.Mhat * np.trapz(self.c00 + C20, x=-self.zarr, axis=1).reshape(len(self.x), 1)
                Xvec, jac = self.jacvec_stock(T, F, X, X_old, Chat_old, Chat)
            else:
                raise KnownError('Availability concept is not implemented in the upwind scheme. Use module DynamicAvailability.')

            while max(abs(Xvec)) > self.TOL:
                dX = linalg.solve(jac, Xvec)
                X = X - dX.reshape(len(X), 1)
                if self.input.v('concept')[0] == 'stock':
                    Xvec, jac = self.jacvec_stock(T, F, X, X_old, Chat_old, Chat)
                else:
                    raise KnownError('Availability concept is not implemented in the upwind scheme. Use module DynamicAvailability.')

            X_old = copy(X)
            if self.input.v('concept')[0] == 'stock':
                Chat_old = copy(Chat)
                Xt.append(X * np.concatenate((Chat_old[:, 0], np.ones(len(self.x)), np.ones(len(self.x)))).reshape(3*len(self.x), 1))
            else:
                raise KnownError('Availability concept is not implemented in the upwind scheme. Use module DynamicAvailability.')
            c0barnew = self.Mhat * (self.c00 + C20) * X[2*len(self.x):].reshape(len(self.x), 1)
            c0bar.append(c0barnew)
            Tt.append(T)
            Ft.append(F)

            # display progress
            if i % np.floor(len(self.Q[1:]) / 10.) == 0:
                percent = float(i) / len(self.Q[1:])
                hashes = '#' * int(round(percent * 10))
                spaces = ' ' * (10 - len(hashes))
                sys.stdout.write("\rProgress: [{0}]{1}%".format(hashes + spaces, int(round(percent * 100))))
                sys.stdout.flush()
        #TODO: incorperate a 4th dimension in the output grid: x, z, f, t. Instead of this manual solution.
        #Prepare variables for saving
        if self.input.v('save_x') is not None:
            dx = self.interpretValues(self.input.v('save_x'))[0]
        else:
            dx = 1
        if self.input.v('save_t') is not None:
            dt = self.interpretValues(self.input.v('save_t'))[0]
        else:
            dt = 1
        St = np.array(Xt)[::dt, :len(self.x):dx, :]
        flux = np.array(Xt)[::dt, len(self.x):2*len(self.x):dx, :]
        ft = np.array(Xt)[::dt, 2*len(self.x)::dx, :]
        Tt = np.array(Tt)[::dt, ::dx]
        Ft = np.array(Ft)[::dt, ::dx]
        Ft = np.array(Ft)[::dt, ::dx]
        c0bar = np.array(c0bar)[::dt, ::dx]
        return {'St': St, 'Flux': flux, 'Tt': Tt, 'Ft': Ft, 'ft': ft, 'c0bar': c0bar, 'alpha2': self.ALPHA2}

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
        Fx = np.gradient(F, self.dx[0,0], axis=0, edge_order=2)
        A = Fx + self.Bx * F / self.B
        return T, F, A, C20

    def river_river_interaction(self, q):#TODO: make separate sript because this is also used in de SedDynamic module
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

    def jacvec_stock(self, T, F, X, X_old, Chat_old, Chat):
        """Calculates the vector containing the Stock and Flux equation and the algebraic expression between S and f, and the
        Jacobian matrix needed to solve for S, Flux and f:
        
        Stilde_t + Stilde * Chat_t / Chat= - (1/Chat) * [Flux_x + Flux * (B_x/B)]       (1)
        
        Flux = T*f + F*f_x                                                              (2)
        
        f = f(Stilde)                                                                   (3)
        

        Parameters:
            T - transport function T on the current time step
            F -  diffusion furnction F on the current time step
            X, X_old - solution vector X = (Stilde, Flux, f) on the old and current time step
            Chat, Chat_old - Chat on the current and old time step

        Returns:
             Xvec - vector containing the Eqs. (1)-(3)
             jac - corresponding Jacobian matrix
        """
        # Initiate jacobian matrix, Xvec and local variable N
        theta = self.input.v('theta') # theta=0 is Forward Euler, theta=1 is Backward Euler and theta=0.5 is Crank-Nicolson
        jac = sps.csc_matrix((len(X), len(X)))
        Xvec, N, N_old = np.zeros((len(X), 1)), np.zeros((len(self.x), 1)), np.zeros((len(self.x), 1))
        # define length of xgrid
        lx = len(self.x)

        # Fill Xvec and jac related to the equation of Stilde: Stilde_t + (Stilde * <Chat>_t) / <Chat> + (1 / <Chat>) * Flux_x + (Flux / B * <Chat>) * B_x = 0
        # First, Stilde only:
        Xvec[:lx-1] += X[:lx-1] * (2. - Chat_old[:lx-1] / Chat[:lx-1]) - X_old[:lx-1]
        jac += sps.csc_matrix(((2. - Chat_old[:lx-1, 0] / Chat[:lx-1, 0]), (range(lx-1), range(lx-1))), shape=jac.shape)
        # Second, (1 / <Chat>) * Flux_x + (Flux / B * <Chat>) * B_x
        # Define local variable N^(n+1) = [(-3Flux_i + 4Flux_i+1 - Flux_i+2) / (2 * dx) + (Flux_i * (B_i)_x) / B_i]
        N[:-2] = (-3 * X[lx:2*lx-2] + 4 * X[lx+1:2*lx-1] - X[lx+2:2*lx]) / (2 * self.dx[0]) + X[lx:2*lx-2] * self.Bx[:-2] / self.B[:-2]
        N[-2] = (-X[2*lx-3] + X[2*lx-1]) / (2 * self.dx[0]) + X[2*lx-2] * self.Bx[-2] / self.B[-2]
        N_old[:-2] = (-3 * X_old[lx:2*lx-2] + 4 * X_old[lx+1:2*lx-1] - X_old[lx+2:2*lx]) / (2 * self.dx[0]) + X_old[lx:2*lx-2] * self.Bx[:-2] / self.B[:-2]
        N_old[-2] = (-X_old[2*lx-3] + X_old[2*lx-1]) / (2 * self.dx[0]) + X_old[2*lx-2] * self.Bx[-2] / self.B[-2]
        # Fill Xvec
        Xvec[:lx-1] += self.Mhat * self.dt * N[:-1] * theta / Chat[:-1] + self.dt * self.Mhat * N_old[:-1] * (1. - theta) / Chat_old[:-1]
        # Fill jac
        jac_F_i = self.Mhat * theta * self.dt * (-3. / (2. * self.dx[0]) + (self.Bx[:-2, 0] / self.B[:-2, 0])) / Chat[:-2, 0]
        jac_F_i1 = 2. * self.Mhat * theta * self.dt / (self.dx[0] * Chat[:-2, 0])
        jac_F_i2 = -self.Mhat * theta * self.dt / (2 * self.dx[0] * Chat[:-2, 0])
        jac += sps.csc_matrix((jac_F_i, (range(lx-2), range(lx, 2*lx-2))), shape=jac.shape)
        jac += sps.csc_matrix((jac_F_i1, (range(lx-2), range(lx+1, 2*lx-1))), shape=jac.shape)
        jac += sps.csc_matrix((jac_F_i2, (range(lx-2), range(lx+2, 2*lx))), shape=jac.shape)
        jac += sps.csc_matrix(([-self.Mhat * theta * self.dt / (2 * self.dx[0, 0] * Chat[-2, 0])], ([lx-2], [2*lx-3])), shape=jac.shape)
        jac += sps.csc_matrix(([self.Mhat * theta * self.dt * self.Bx[-2, 0] / ( self.B[-2, 0] * Chat[-2, 0])], ([lx-2], [2*lx-2])), shape=jac.shape)
        jac += sps.csc_matrix(([self.Mhat * theta * self.dt / (2 * self.dx[0, 0] * Chat[-2, 0])], ([lx-2], [2*lx-1])), shape=jac.shape)
        # Boundary condition at the weir Flux = Flux_river
        Friver = self.interpretValues(self.input.v('Friver'))
        Xvec[lx-1] = X[2*lx-1] + Friver
        jac += sps.csc_matrix(([1.], ([lx-1], [2*lx-1])), shape=jac.shape)

        # Fill Xvec and jac related to the equation of Flux: Flux - Tf - Ff_x = 0
        # First, Flux only:
        Xvec[lx+1:2*lx] += X[lx+1:2*lx]
        jac += sps.csc_matrix((np.ones(lx-1), (range(lx+1, 2*lx), range(lx+1, 2*lx))), shape=jac.shape)
        # Second, -Tf -Ff_x only: -T_i * f_i - F_i * (f_i-2 - 4f_i-1 + 3f_i) / (2 * dx)
        # Fill Xvec backward euler from index=2:end
        Xvec[lx+2:2*lx] += -T[2:] * X[2*lx+2:] - F[2:] * (X[2*lx:3*lx-2] - 4 * X[2*lx+1:3*lx-1] + 3 * X[2*lx+2:]) / (2 * self.dx[0])
        # Fill Xvec central differences for index=1
        Xvec[lx+1] += -T[1] * X[2*lx+1] - F[1] * (-X[2*lx] + X[2*lx+2]) / (2 * self.dx[0])
        # Fill jac
        jac_f_i = -T[2:, 0] - 3 * F[2:, 0] / (2 * self.dx[0])
        jac_f_i1 = 2 * F[1:-1, 0] / self.dx[0]
        jac_f_i2 = -F[:-2, 0] / (2 * self.dx[0])
        jac += sps.csc_matrix((jac_f_i, (range(lx+2, 2*lx), range(2*lx+2, 3*lx))), shape=jac.shape)
        jac += sps.csc_matrix((jac_f_i1, (range(lx+2, 2*lx), range(2*lx+1, 3*lx-1))), shape=jac.shape)
        jac += sps.csc_matrix((jac_f_i2, (range(lx+2, 2*lx), range(2*lx, 3*lx-2))), shape=jac.shape)
        jac += sps.csc_matrix(([F[1, 0] / (2 * self.dx[0, 0])], ([lx+1], [2*lx])), shape=jac.shape)
        jac += sps.csc_matrix(([-T[1, 0]], ([lx+1], [2*lx+1])), shape=jac.shape)
        jac += sps.csc_matrix(([-F[1, 0] / (2 * self.dx[0, 0])], ([lx+1], [2*lx+2])), shape=jac.shape)
        # Boundary condition at sea: f = fsea
        Xvec[lx] = X[2*lx] - self.FSEA
        jac += sps.csc_matrix(([1.], ([lx], [2*lx])), shape=jac.shape)

        # Fill Xvec and jacobian for the algebraic relation: f - f(Stilde) = 0, when using the exact relation between S and f
        if self.input.v('concept')[1] == 'exact':
            Xvec[2*lx:] += X[2*lx:] - self.erodibility_stock_relation(self.ALPHA2, X[:lx])
            jac += sps.csc_matrix((-self.erodibility_stock_relation_der(self.ALPHA2, X[:lx])[:, 0], (range(2*lx, 3*lx), range(lx))), shape=jac.shape)
            jac += sps.csc_matrix((np.ones(lx), (range(2*lx, 3*lx), range(2*lx, 3*lx))), shape=jac.shape)
        else:
            raise KnownError('Approximation stock or availability concept is not implemented in the upwind scheme. Use module DynamicAvailability.')

        # Convert jacobian matrix to dense matrix
        jac = jac.todense()
        return Xvec, jac

    def stability_availability(self, A, T, F, D, aeq):#TODO: discuss with George if this is still useful
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

    def stability_stock(self, A, T, F, alpha1):#TODO: same as stability_availability...
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
        M[1:-1] = self.Mhat / alpha1[1:-1]
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

    def interpretValues(self, values):#TODO: make a separate script! To nifty???
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
                if valString is None or valString=='':
                    valuespy = None
                else:
                    valuespy = eval(valString)
                return valuespy
            except Exception as e:
                try: errorString = ': '+ str(e)
                except: errorString = ''
                raise KnownError('Failed to interpret input as python command %s in input: %s' %(errorString, valString), e)

        # case 2: else interpret as space-separated list
        else:
            return values

    def erodibility_stock_relation(self,alpha2, Shat):
        #Define masks where the bottom is empty, partly covered or covered
        empty = np.where(Shat < 1 - alpha2)[0]
        part_cover = np.where((Shat >= 1 - alpha2) & (Shat <= 1 + alpha2))[0]
        cover = np.where(Shat > 1 + alpha2)[0]
        #Build f
        f = np.full(Shat.shape, np.nan)
        f[empty] = Shat[empty]
        f[cover] = 1.
        xi = np.full(f.shape, np.nan)
        xi[part_cover] = np.arcsin((Shat[part_cover] - 1.) / alpha2[part_cover])
        f[part_cover] = (Shat[part_cover] * (0.5 - xi[part_cover] / np.pi) + 0.5 +
                        xi[part_cover] / np.pi - alpha2[part_cover] * np.cos(xi[part_cover]) / np.pi)
        return np.real(f)

    def erodibility_stock_relation_der(self,alpha2, Shat):
        #Define masks where the bottom is empty, partly covered or covered
        empty = np.where(Shat <= 1 - alpha2)[0]
        part_cover = np.where((Shat > 1 - alpha2) & (Shat < 1 + alpha2))[0]
        cover = np.where(Shat >= 1 + alpha2)[0]
        #Build df/dS
        dfdS = np.full(Shat.shape, np.nan)
        dfdS[empty] = 1.
        dfdS[cover] = 0.
        xi = np.full(dfdS.shape, np.nan)
        xi[part_cover] = np.arcsin((Shat[part_cover] - 1.) / alpha2[part_cover])
        dfdS[part_cover] = 0.5 - xi[part_cover] / np.pi
        return np.real(dfdS)