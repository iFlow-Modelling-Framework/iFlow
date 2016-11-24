"""
DynamicAvailability

Date: 10/08/16
Authors: R.L. Brouwer
"""
import logging
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.sparse as sps
import scipy.linalg as linalg
import nifty as ny
from src.util.diagnostics import KnownError
import sys
from math import sqrt


class DynamicAvailability:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the sediment concentration based on dynamic availability. We have
        to find an expression for f = g*a, where a is the availability when the system is in morphodynamic equilibrium
        and g is a function that describes the dependence on Q (or t). We solve g from the following equation:

            g_t = gamma * (1-beta*a*g)[F*g_xx + (A - T)*g_x] - g*a_t/a,                         (1)

        where,

            gamma = 1 / (mu_s*rho_s*(1-p)H_0),                                                  (2)
            A = F_x + B_x*F/B,                                                                  (3)
            a_t/a = -int_0^x[(T/F)_t dx].                                                       (4)

         Returns:
             Dictionary with results. At least contains the variables listed as output in the registry
         """
        self.logger.info('Module DynamicAvailability is running')

        # Initiate variables
        self.RHOS = self.input.v('RHOS')
        self.DS = self.input.v('DS')
        self.WS = self.input.v('ws')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')
        self.MUS = self.input.v('mus')
        self.FCAP = self.input.v('fcap')
        self.CSEA = self.input.v('csea')
        self.P = self.input.v('p')
        self.TOL = self.input.v('tol')
        self.ASTAR = self.input.v('astar')
        self.Kh = self.input.v('Kh')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.L
        self.dx = (self.x[1:]-self.x[:-1]).reshape(len(self.x)-1, 1)
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        self.H = (self.input.v('H', x=self.x/self.L).reshape(len(self.x), 1) +
                  self.input.v('R', x=self.x/self.L).reshape(len(self.x), 1))
        self.Hx = (self.input.d('H', x=self.x/self.L, dim='x').reshape(len(self.x), 1) +
                   self.input.d('R', x=self.x / self.L, dim='x').reshape(len(self.x), 1))
        self.B = self.input.v('B', x=self.x/self.L).reshape(len(self.x), 1)
        self.Bx = self.input.d('B', x=self.x/self.L, dim='x').reshape(len(self.x), 1)
        self.sf = self.input.v('Roughness', x=self.x/self.L, f=0).reshape(len(self.x), 1)
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=0).reshape(len(self.x), 1).reshape(len(self.x), 1)
        self.Fc = self.input.v('F', x=self.x/self.L).reshape(len(self.x), 1) - self.input.v('F', 'Fdiff', 'c20', x=self.x/self.L).reshape(len(self.x), 1)
        self.Tc = (self.input.v('T') - (self.input.v('T', 'TM0', 'tide_river', x=self.x/self.L) +
                                        self.input.v('T', 'TM0', 'river_river', x=self.x/self.L) +
                                        self.input.v('T', 'TM2', 'TM2M0', 'tide_river', x=self.x / self.L))).reshape(len(self.x), 1)
        self.TQ = (self.input.v('T', 'TM0', 'tide_river', x=self.x/self.L) +
                   self.input.v('T', 'TM2', 'TM2M0', 'tide_river', x=self.x/self.L)).reshape(len(self.x), 1) / self.input.v('Q1')
        self.c00 = np.real(self.input.v('hatc', 'a', 'c00', range(0, len(self.x)), range(0, len(self.z))))
        self.FSEA = self.CSEA * self.H[0] / np.trapz(self.c00[0, :], dx=-self.zarr[0, 1])
        # Calculate constant factor gamma
        self.gamma = 1. / (self.MUS * self.RHOS * (1. - self.P) * self.H[0])

        #load time serie Q
        self.step = 5
        self.dt = 24 * 3600
        self.t = np.arange(0, 366 * self.dt, self.dt/self.step)
        ### constant profile ###
        # self.Q = np.ones(1000) * 20 #self.input.v('Q1')
        # self.Qt = np.zeros(1000)
        ### sinusoidal profile ###
        # self.Q = 40 + 10 * np.sin(2 * np.pi * self.t / (365 * self.dt))
        # self.Qt = 10 * np.cos(2 * np.pi * self.t / (365 * self.dt)) * 2 * np.pi / (365 * self.dt)
        ### tangent hyperbolic profile ###
        self.Q = -17.5*np.tanh((self.t - self.dt*200)/2e6) + 42.5
        # self.Q = np.append(self.Q, 17.5*np.tanh((self.t - self.dt*200)/2e6) + 42.5)
        # self.t = np.arange(0, 2 * 366 * self.dt, self.dt/self.step)
        self.t = np.arange(0, 366 * self.dt, self.dt/self.step)
        # self.Qt = np.gradient(self.Q, self.dt, edge_order=2)
        ### fitted data Scheldt ###
        # data = np.loadtxt('/Users/RLBrouwer/Box Sync/WL Antwerpen/WL2016R13_103_4_WP1Sed_texfiles/python/WP14/input/tday_Qmeasured_Qfit_dQfitdt.dat')
        # data = np.loadtxt('/Users/RLBrouwer/Box Sync/WL Antwerpen/WL2016R13_103_4_WP1Sed_texfiles/python/WP14/input/Qfit_91days.dat')
        # Q = data[5000:7000, 0]
        # # Qt = data[5000:7000, 1]
        # t = np.arange(0, len(Q) * self.dt, self.dt)
        # t2 = np.arange(0, len(Q) * self.dt, self.dt / 10)[:-9]
        # Qinterp = interp1d(t, Q, kind='cubic')
        # self.Q = Qinterp(t2)
        # Qtinterp = interp1d(t, Qt, kind='cubic')
        # self.Qt = Qtinterp(t2)
        ### linear Q ###
        # self.Q = np.arange(5, 121)
        # self.Qt = np.gradient(self.Q, self.dt, edge_order=2)

        # Allocate space to save results
        # d = {}
        # d['f'] = {}
        # d['flux'] ={}
        # # d['Tt'] = {}
        # # d['Ft'] = {}
        # d['c0bar'] = {}
        # # d['Msed'] = {}
        # d['eigs'] = {}

        print 'Time-integrator is running'

        # # Initialize vector g
        # g = self.FSEA * np.ones((len(self.x),1)) #(1. - self.x / self.L)
        # # g = (self.FSEA * (1. - self.x / self.L)**2).reshape(len(self.x), 1)
        # # G = []
        # # G.append(g)
        # g_old = g
        # # Initialize availability a
        # Tt = []
        # Tr_old, C20_old = self.river_river_interaction(self.Q[0])
        # T_old = self.Tc + self.TQ * self.Q[0] + Tr_old
        # Tt.append(T_old)
        # # # Calculate and append diffusive coefficient F
        # Ft = []
        # fdiff_c20_old = np.real(-np.trapz(self.Kh * C20_old, x=-self.zarr, axis=1))
        # F_old = self.Fc + fdiff_c20_old.reshape(len(self.x), 1)
        # Ft.append(F_old)
        # # a_old = self.availability(F[0], T[0])
        # a_old = self.availability(F_old, T_old)
        # a_old = a_old / a_old[0]
        # Fx_old = np.gradient(F_old, self.dx[0], axis=0, edge_order=2)
        # A_old = Fx_old + self.Bx * F_old / self.B
        # Initialize variables to be saved
        # f = []
        # f_old = g_old * a_old
        # f.append(f_old)
        # flux = []
        # flux_old = self.B * F_old * a_old * np.gradient(g, self.dx[0], axis=0, edge_order=2)
        # flux.append(flux_old)
        # c0bar = []
        # c0bareq = []
        # M = []
        # Meq = []
        # eigvals = []
        # eigvecs = []
        # a_init = self.FSEA * a_old

        # vars = self.rk4()
        vars = self.implicit()

        # for i, q in enumerate(self.Q[1:]):
        #     # print 'Q =', q, 'step: ', i+1, '/', len(self.Q[1:])
        #     Tr, C20 = self.river_river_interaction(q)
        #     T = self.Tc + self.TQ * q + Tr
        #     # Calculate and append diffusive coefficient F
        #     fdiff_c20 = np.real(-np.trapz(self.Kh * C20, x=-self.zarr, axis=1))
        #     F = self.Fc + fdiff_c20.reshape(len(self.x), 1)
        #     Fx = np.gradient(F, self.dx[0], axis=0, edge_order=2)
        #     A = Fx + self.Bx * F / self.B
        #     a = self.availability(F, T)
        #     # scale the availability such that a = 1 at the sea boundary
        #     a = a / a[0]
        #     # test if min[1 - cap * a * g] is negative at t=0. If so raise an error
        #     cap = 1 - self.FCAP * a * g
        #     ind, = np.where(cap[:, 0] < 0.)
        #     if not len(ind) == 0:
        #         # print 'capped'
        #         g[ind] = (1. - 10 * np.finfo(float).eps) / (self.FCAP * a[ind])
        #     # print min(1 - self.FCAP * a * g)
        #
        #     if i == 0 and min(1 - self.FCAP * a * g) < 0:
        #         maxfcap = 1 / max(a * g)
        #         raise KnownError('Given the initial and boundary conditions, fcap should be smaller than %.2g' % maxfcap)
        #
        #     # Determine eigenvalues of the system for a certain discharge
        #     if self.FCAP == 0:
        #         eigval, eigvec = self.stability(A, T, F, a)
        #         eigvals.append(eigval)
        #         eigvecs.append(eigvec)
        #     else:
        #         eigvals.append(0.)
        #         eigvals.append(0.)
        #
        #     gvec, jac = self.jacvec(A, A_old, T, T_old, F, F_old, g, g_old, a, a_old)
        #     while max(abs(gvec)) > acc:
        #         dg = linalg.solve(jac, gvec)
        #         g = g - dg.reshape(len(self.x), 1)
        #         gvec, jac = self.jacvec(A, A_old, T, T_old, F, F_old, g, g_old, a, a_old)
        #     cap = 1 - self.FCAP * a * g
        #     ind, = np.where(cap[:, 0] < 0.)
        #     if not len(ind) == 0:
        #         # print 'capped again'
        #         g[ind] = (1. - 10 * np.finfo(float).eps) / (self.FCAP * a[ind])
        #     # print min(1 - self.FCAP * a * g)
        #     g_old = g
        #     a_old = a
        #     T_old = T
        #     F_old = F
        #     A_old = A
        #     fnew = a * g
        #     c0barnew = (self.c00 + C20) * fnew.reshape(len(self.x), 1)
        #     c0bareqnew = (self.c00 + C20) * (a * self.FSEA).reshape(len(self.x), 1)
        #     # m = np.trapz(self.B * np.trapz(c0barnew, x=-self.zarr, axis=1).reshape(len(self.x), 1), dx=self.dx, axis=0)
        #     # meq = np.trapz(self.B * np.trapz(c0bareqnew, x=-self.zarr, axis=1).reshape(len(self.z), 1), dx=self.dx[0], axis=0)
        #     flux.append(self.B * F * a * np.gradient(g, self.dx[0], axis=0, edge_order=2))
        #     f.append(fnew)
        #     c0bar.append(c0barnew)
        #     c0bareq.append(c0bareqnew)
        #     # M.append(m)
        #     # Meq.append(meq)
        #     Tt.append(T)
        #     Ft.append(F)
        #     # G.append(g)
        #
        #     if (i%np.floor(len(self.Q[1:])/10.)==0):
        #         percent = float(i) / len(self.Q[1:])
        #         hashes = '#' * int(round(percent * 10))
        #         spaces = ' ' * (10 - len(hashes))
        #         sys.stdout.write("\rProgress: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        #         sys.stdout.flush()

        d = {}
        for key, value in vars.iteritems():
            d[key] = value
        return d

    def implicit(self):
        # Initialize vector g
        g = self.FSEA * np.ones((len(self.x), 1))  # (1. - self.x / self.L)
        # g = (self.FSEA * (1. - self.x / self.L)**2).reshape(len(self.x), 1)
        # G = []
        # G.append(g)
        g_old = g
        # Initialize availability a
        T_old, F_old, A_old, C20_old = self.transport_terms(self.Q[0])
        Tt = []
        Tt.append(T_old)
        # # Calculate and append diffusive coefficient F
        Ft = []
        Ft.append(F_old)
        aeq_old = self.availability(F_old, T_old)
        aeq_old = aeq_old / aeq_old[0]
        a_old = aeq_old * self.FSEA
        a = a_old
        Xt = []
        X = np.append(g, a).reshape(2*len(self.x), 1)
        X_old = np.append(g_old, a_old).reshape(2*len(self.x), 1)
        # Initialize variables to be saved
        # f = []
        # f_old = g_old * a_old
        # f.append(f_old)
        flux = []
        flux_old = self.B * F_old * aeq_old * np.gradient(g, self.dx[0], axis=0, edge_order=2)
        flux.append(flux_old)
        c0bar = []
        c0bareq = []
        # M = []
        # Meq = []
        eigvals = []
        eigvecs = []
        for i, q in enumerate(self.Q[1:]):
            T, F, A, C20 = self.transport_terms(q)
            aeq = self.availability(F, T)
            # scale the availability such that a = 1 at the sea boundary
            aeq = aeq / aeq[0]
            # test if min[1 - cap * a * g] is negative at t=0. If so raise an error
            # cap = 1 - self.FCAP * a * g
            # ind, = np.where(cap[:, 0] < 0.)
            # if not len(ind) == 0:
            #     # print 'capped'
            #     g[ind] = (1. - 10 * np.finfo(float).eps) / (self.FCAP * a[ind])
            # print min(1 - self.FCAP * a * g)

            # if i == 0 and min(1 - self.FCAP * a * g) < 0:
            #     maxfcap = 1 / max(a * g)
            #     raise KnownError(
            #         'Given the initial and boundary conditions, fcap should be smaller than %.2g' % maxfcap)

            # Determine eigenvalues of the system for a certain discharge
            if self.FCAP == 0:
                eigval, eigvec = self.stability(A, T, F, aeq)
                eigvals.append(eigval)
                eigvecs.append(eigvec)
            else:
                eigvals.append(0.)
                eigvals.append(0.)

            Xvec, jac = self.jacvec2(A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old)
            while max(abs(Xvec)) > self.TOL:
                dX = linalg.solve(jac, Xvec)
                X = X - dX.reshape(len(X), 1)
                Xvec, jac = self.jacvec2(A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old)
            # cap = 1 - self.FCAP * aeq * g
            # ind, = np.where(cap[:, 0] < 0.)
            # if not len(ind) == 0:
            #     # print 'capped again'
            #     g[ind] = (1. - 10 * np.finfo(float).eps) / (self.FCAP * a[ind])
            # print min(1 - self.FCAP * a * g)
            # g_old = g
            aeq_old = aeq
            X_old = X
            T_old = T
            F_old = F
            A_old = A
            fnew = aeq * X[:len(self.x)]
            c0barnew = (self.c00 + C20) * fnew.reshape(len(self.x), 1)
            c0bareqnew = (self.c00 + C20) * (aeq * self.FSEA).reshape(len(self.x), 1)
            # m = np.trapz(self.B * np.trapz(c0barnew, x=-self.zarr, axis=1).reshape(len(self.x), 1), dx=self.dx, axis=0)
            # meq = np.trapz(self.B * np.trapz(c0bareqnew, x=-self.zarr, axis=1).reshape(len(self.z), 1), dx=self.dx[0], axis=0)
            flux.append(self.B * F * aeq * np.gradient(X[:len(self.x)], self.dx[0], axis=0, edge_order=2))
            Xt.append(X)
            c0bar.append(c0barnew)
            c0bareq.append(c0bareqnew)
            # M.append(m)
            # Meq.append(meq)
            Tt.append(T)
            Ft.append(F)
            # G.append(g)

            if (i % np.floor(len(self.Q[1:]) / 10.) == 0):
                percent = float(i) / len(self.Q[1:])
                hashes = '#' * int(round(percent * 10))
                spaces = ' ' * (10 - len(hashes))
                sys.stdout.write("\rProgress: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
                sys.stdout.flush()
        return {'Xt': np.array(X)[::self.step], 'flux': np.array(flux)[::self.step], 'Tt': np.array(Tt)[::self.step],
                'c0bar': {'t': np.array(c0bar)[::self.step], 'eq': np.array(c0bareq)[::self.step]},
                'Ft': np.array(Ft)[::self.step], 'eigs': {'eigvals': np.array(eigvals)[::self.step],
                                                          'eigvecs': np.array(eigvecs)[::self.step]}}
        # return {'f': np.array(f)[::self.step], 'flux': np.array(flux)[::self.step], 'Tt': np.array(Tt)[::self.step],
        #         'c0bar': {'t': np.array(c0bar)[::self.step], 'eq': np.array(c0bareq)[::self.step]},
        #         'Ft': np.array(Ft)[::self.step], 'eigs': {'eigvals': np.array(eigvals)[::self.step],
        #                                                   'eigvecs': np.array(eigvecs)[::self.step]}}

    def river_river_interaction(self, q):
        u0 = self.input.v('u0', 'tide', range(0, len(self.x)), len(self.z) - 1, 1)
        ur = self.input.v('u1', 'river', range(0, len(self.x)), range(0, len(self.z)), 0) * q / self.input.v('Q1')
        u1b = ur[:, -1]
        time = np.linspace(0, 2 * np.pi, 100)
        utid = np.zeros((len(self.x), len(time))).astype('complex')
        ucomb = np.zeros((len(self.x), len(time))).astype('complex')
        for i, t in enumerate(time):
            utid[:, i] = 0.5 * (u0 * np.exp(1j * t) + np.conj(u0) * np.exp(-1j * t))  # YMD
            ucomb[:, i] = u1b + 0.5 * (u0 * np.exp(1j * t) + np.conj(u0) * np.exp(-1j * t))
        uabs_tid = np.mean(np.abs(utid), axis=1)
        uabs_tot = np.mean(np.abs(ucomb), axis=1)
        uabs_eps = uabs_tot.reshape(len(self.x), 1) - uabs_tid.reshape(len(self.x), 1)
        c20 = np.real((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_eps *
               np.exp(-self.WS * (self.H + self.zarr) / self.Av0))

        Tr = np.real(np.trapz(ur * c20, x=-self.zarr, axis=1)).reshape(len(self.x), 1)
        return Tr, c20

    def availability(self, F, T):
        """Calculates the availability of sediment needed to derive the sediment concentration

        Parameters:
            F - diffusive coefficient in the availability equation that goes with a_x
            T - coefficient (advective, diffusive and stokes) in the availability equation that goes with a

        Returns:
            a - availability of sediment throughout the estuary
        """
        # This exponent is set to zero (hard-coded) at the landward boundary because here the availability is zero too!
        # exponent = np.append(np.exp(-np.append(0, integrate.cumtrapz(T / F, dx=self.dx, axis=0)[:-1])), 0)

        # CORRECTION 13/9/16 R.L. BROUWER: BECAUSE WE INCLUDED THE RIVER-RIVER-RIVER INTERACTION, T AND F, AND THUS a,
        # ARE NOT 0 AT THE WEIR ANYMORE!!!
        exponent = np.exp(-integrate.cumtrapz(T / F, dx=self.dx, axis=0, initial=0))
        A = (self.ASTAR * np.trapz(self.B, dx=self.dx, axis=0) /
             np.trapz(self.B * exponent, dx=self.dx, axis=0))
        a = A * exponent
        return a

    def jacvec2(self, A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old):
        # Initiate jacobian matrix, Xvec and local variable N
        jac = sps.csc_matrix((len(X), len(X)))
        Xvec, N = np.zeros((len(X), 1)), np.zeros((len(self.x), 1))
        # define length of xgrid
        lx = len(self.x)
        ivec = range(1, lx-1)
        # Define local variable N^(n+1)
        N[ivec] = self.gamma * aeq[1:-1] * (F[1:-1] * (X[2:lx] - 2 * X[ivec] + X[:lx-2]) / self.dx[0]**2 +
                                              (A[1:-1] - T[1:-1]) * (X[2:lx] - X[:lx-2]) / (2 * self.dx[0]))
        # Fill interior points of Xvec related g for the differential equation (1)
        Xvec[ivec] += self.dt * N[ivec] / 2.
        # Fill interior points of the jacobian matrix related to g for Eq. (1)
        jval_c_g = -self.dt * self.gamma * aeq[1:-1, 0] * F[1:-1, 0] / self.dx[0]**2
        jac += sps.csc_matrix((jval_c_g, (ivec, ivec)), shape=jac.shape)
        jval_l_g = self.dt * self.gamma * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_l_g, (ivec, range(lx-2))), shape=jac.shape)
        jval_r_g = self.dt * self.gamma * aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_r_g, (ivec, range(2, lx))), shape=jac.shape)
        # Boundary condition at sea
        Xvec[0] = X[0] - self.FSEA
        jac += sps.csc_matrix(([1.], ([0.], [0.])), shape=jac.shape)
        # Boundary condition at weir
        Xvec[lx-1] = (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3])
        jac += sps.csc_matrix(([3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Fill interior points of Xvec and diagonal of jacobian related to a for Eq. (1)
        Xvec[lx+1:-1] += X[lx+1:-1]
        jac += sps.csc_matrix((np.ones(lx-2), (ivec, range(lx+1, len(X)-1))), shape=jac.shape)
        # Fill Xvec and jacobian for the algebraic relation between g and a, Eq. (2)
        Xvec[lx:] = aeq * X[:lx] + self.FCAP * aeq * X[:lx] * X[lx:] - X[lx:]
        jac += sps.csc_matrix((self.FCAP * aeq[:, 0] * X[:lx, 0] - 1., (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        jac += sps.csc_matrix((aeq[:, 0] + self.FCAP * aeq[:, 0] * X[lx:, 0], (range(lx, 2*lx), range(lx))), shape=jac.shape)
        # Convert jacobian matrix to dense matrix
        jac = jac.todense()
        # Inhomogeneous part of the PDE related to a and g at interior points
        Xvec[lx:] += -X_old[lx:]
        N[ivec] = self.gamma * aeq_old[1:-1] * (F_old[1:-1] * (X_old[2:lx] - 2 * X_old[ivec] + X_old[:lx-2]) / self.dx[0]**2 +
                                                (A_old[1:-1] - T_old[1:-1]) * (X_old[2:lx] - X_old[:lx-2]) / (2 * self.dx[0]))
        Xvec[ivec] += self.dt * N[ivec] / 2.
        return Xvec, jac

    def jacvec(self, A, A_old, T, T_old, F, F_old, g, g_old, a, a_old):
        gvec, M, N = np.zeros((len(self.x),1)), np.zeros((len(self.x),1)), np.zeros((len(self.x),1))
        # fill g^n+1 and build Jacobian matrix
        gvec[1:-1] = g[1:-1]
        jac = sps.identity(len(self.x))
        # add values to the vector G and define the values for the diagonals for the interior points for the jacobian matrix
        M[1:-1] = self.gamma * (1 - self.FCAP * a[1:-1] * g[1:-1])
        N[1:-1] = F[1:-1] * (g[2:] - 2 * g[1:-1] + g[:-2]) / self.dx[0]**2 + (A[1:-1] - T[1:-1]) * (g[2:] - g[:-2]) / (2 * self.dx[0])
        gvec += self.dt * (M * N) / 2.
        jval_c = (self.dt / 2) * (-self.gamma * self.FCAP * a * N - 2 * M * F / self.dx[0]**2)
        jval_l = self.dt * M * (F / self.dx[0] - (A - T) / 2.) / (2 * self.dx[0])
        jval_r = self.dt * M * (F / self.dx[0] + (A - T) / 2.) / (2 * self.dx[0])
        # Boundary condition at sea (jacobian matrix at x=0 is already defined by the identity matrix)
        gvec[0] = g[0] - self.FSEA
        # Boundary condition at weir
        # gvec[-1] = g[-4] - 3 * g[-3] + 3 * g[-2] - g[-1]
        gvec[-1] = (3 * g[-1] - 4 * g[-2] + g[-3]) / (2 * self.dx[-1])
        # jval_c[-1] = -2 # the final result should be -1 at J[-1, -1]. since we initialized the main diagonal with ones, we have to subtract 2!
        jval_c[-1] = -1 + 3 / (2 * self.dx[-1])
        # jval_l[-1] = 3
        jval_l[-1] = -2 / self.dx[-1]
        jval_2l, jval_3l = np.zeros(len(self.x) - 2), np.zeros(len(self.x) - 3)
        # jval_2l[-1] = -3
        jval_2l[-1] = 1 / (2 * self.dx[-1])
        # jval_3l[-1] = 1
        # add diagonals to jacobian matrix
        # jac += sps.diags([jval_3l, jval_2l, jval_l[1:], jval_c, jval_r[:-1]], [-3, -2, -1, 0, 1])
        jac += sps.diags([jval_2l, jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-2, -1, 0, 1])
        jac = jac.todense()
        # Inhomogeneous part of the PDE at interior points
        gvec[1:-1] += -a_old[1:-1] * g_old[1:-1] / a[1:-1]
        M[1:-1] = self.gamma * (1 - self.FCAP * a_old[1:-1] * g_old[1:-1])
        N[1:-1] = F_old[1:-1] * (g_old[2:] - 2 * g_old[1:-1] + g_old[:-2]) / self.dx[0]**2 + (A_old[1:-1] - T_old[1:-1]) * (g_old[2:] - g_old[:-2]) / (2 * self.dx[0])
        gvec += self.dt * a_old * (M * N) / (2. * a)
        return gvec, jac

    def stability(self, A, T, F, a):
        M = np.zeros((len(self.x), 1))
        # initialize Jacobian matrix with zeros
        jac = sps.dia_matrix((len(self.x), len(self.x)))
        # define the values for the diagonals for the interior points for the jacobian matrix to the generalized
        # eigenvalue problem
        M[1:-1] = self.gamma * (1 - self.FCAP * a[1:-1] * self.FSEA)
        jval_c = -2 * M * F / self.dx[0]**2
        jval_l = M * (F / self.dx[0] - (A - T) / 2.) / (self.dx[0])
        jval_r = M * (F / self.dx[0] + (A - T) / 2.) / (self.dx[0])
        # Boundary condition at weir. Actually, this is not necessary since we used the standard eigenvalue problem
        # jval_c[-1] += 3
        # jval_l[-1] += -4
        # jval_2l = np.zeros(len(self.x) - 2)
        # jval_2l[-1] += 1
        # Modify jacobian matrix to use it for the standard eigenvalue problem.
        jval_c[-2] += 4 * jval_r[-2] / 3 # Sturm-Liousville modification
        jval_l[-2] += -jval_r[-2] / 3 # Sturm-Liousville modification
        jac += sps.diags([jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-1, 0, 1])
        jac = jac[1:-1, 1:-1]
        #Determine eigenvalues and eigenvectors
        jacd = jac.todense()
        eigval, eigvec = linalg.eig(jacd)
        return eigval, eigvec

    def rk4(self):
        T0, F0, A0, C20 = self.Transport_terms(self.Q[0])
        a0 = self.availability(F0, T0)
        a0 = a0 / a0[0]
        a_init = self.FSEA * a0
        a = []
        a.append(a_init)
        c20 = []
        Tt = []
        Ft = []
        for i, q in enumerate(self.Q[1:]):
            qb = self.Q[i]
            qe = q
            qi = (qb + qe) / 2.
            h = self.t[i+1] - self.t[i]
            T0, F0, A0, C20 = self.transport_terms(qb)
            if i==0:
                Tt.append(T0)
                Ft.append(F0)
                c20.append(C20)
            trans1 = self.transport_rk4(T0, F0, A0, a[i])
            k1 = h * trans1
            k1[0] = 0.
            Ti, Fi, Ai, C20i = self.transport_terms(qi)
            trans2 = self.transport(Ti, Fi, Ai, a[i] + 0.5 * k1)
            k2 = h * trans2
            k2[0] = 0.
            trans3 = self.transport(Ti, Fi, Ai, a[i] + 0.5 * k2)
            k3 = h * trans3
            k3[0] = 0.
            Te, Fe, Ae, C20e = self.transport_terms(qe)
            trans4 = self.transport(Te, Fe, Ae, a[i] + k3)
            k4 = h * trans4
            k4[0] = 0.
            a.append(a[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.)
            f3 = (1. / self.FCAP) * (1. - np.exp(-self.FCAP * a[i+1][-3]))
            f2 = (1. / self.FCAP) * (1. - np.exp(-self.FCAP * a[i+1][-2]))
            f1 = Fi[-1] * (4 * f2 - f3) / (2 * self.dx[0] * Ti[-1] + 3 * Fi[-1])
            a[i+1][-1] = -(1. / self.FCAP) * np.log(1. - self.FCAP * f1)
            c20.append(C20e)
            Tt.append(Te)
            Ft.append(Fe)
            if (i%np.floor(len(self.Q[1:])/10.)==0):
                percent = float(i) / len(self.Q[1:])
                hashes = '#' * int(round(percent * 10))
                spaces = ' ' * (10 - len(hashes))
                sys.stdout.write("\rProgress: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
                sys.stdout.flush()
        a = np.array(a)
        f = (1. / self.FCAP) * (1. - np.exp(-self.FCAP * a))
        return {'a': a[::self.step], 'f': f[::self.step], 'c20': np.array(c20)[::self.step],
                'Tt': np.array(Tt)[::self.step], 'Ft': np.array(Ft)[::self.step]}

    def transport_terms(self, q):
        Tr, C20 = self.river_river_interaction(q)
        T = self.Tc + self.TQ * q + Tr
        # Calculate and append diffusive coefficient F
        fdiff_c20 = np.real(-np.trapz(self.Kh * C20, x=-self.zarr, axis=1))
        F = self.Fc + fdiff_c20.reshape(len(self.x), 1)
        Fx = np.gradient(F, self.dx[0], axis=0, edge_order=2)
        A = Fx + self.Bx * F / self.B
        return T, F, A, C20

    def transport_rk4(self, T, F, A, a):
        Tx = np.gradient(T, self.dx[0], axis=0, edge_order=2)
        # Calculate f
        f = (1. / self.FCAP) * (1. - np.exp(-self.FCAP * a))
        fx = np.gradient(f, self.dx[0], axis=0, edge_order=2)
        fxx = np.gradient(fx, self.dx[0], axis=0, edge_order=2)
        # Calculate transport at interior points for rk4 method only
        transport = -self.gamma * (F * fxx + (A + T) * fx + (Tx + self.Bx * T / self.B) * f)
        return transport

# def jacvec2(self, A, T, F, g, g_old, a, a_old, intTF):
#     gvec, M, N, R = np.zeros((len(self.x), 1)), np.zeros((len(self.x), 1)), np.zeros((len(self.x), 1)), np.zeros(
#         (len(self.x), 1))
#     # fill g^n+1 and build Jacobian matrix
#     gvec[1:-1] = g[1:-1]
#     jac = sps.identity(len(self.x))
#     # add values to the vector G and define the values for the diagonals for the interior points for the jacobian matrix
#     M[1:-1] = self.gamma * (1 - self.FCAP * a[1:-1] * g[1:-1])
#     N[1:-1] = F[1, 1:-1] * (g[2:] - 2 * g[1:-1] + g[:-2]) / self.dx[0] ** 2 + (A[1, 1:-1] - T[1, 1:-1]) * (
#     g[2:] - g[:-2]) / (2 * self.dx[0])
#     # R[1:-1] = -(Qt * self.intTQF[1:-1] + intTrt[1:-1]) * g[1:-1]
#     R[1:-1] = -intTF[1, 1:-1] * g[1:-1]
#     gvec += self.dt * (M * N + R) / 2.
#     # jval_c = (self.dt / 2) * (-gamma * self.FCAP * a * N - 2 * M * F / self.dx[0]**2 - Qt * self.intTQF - intTrt)
#     jval_c = (self.dt / 2) * (-self.gamma * self.FCAP * a * N - 2 * M * F[1] / self.dx[0] ** 2 - intTF[1])
#     jval_l = self.dt * M * (F[1] / self.dx[0] - (A[1] - T[1]) / 2.) / (2 * self.dx[0])
#     jval_r = self.dt * M * (F[1] / self.dx[0] + (A[1] - T[1]) / 2.) / (2 * self.dx[0])
#     # Boundary condition at sea (jacobian matrix at x=0 is already defined by the identity matrix)
#     gvec[0] = g[0] - self.FSEA
#     # Boundary condition at weir
#     # gvec[-1] = g[-4] - 3 * g[-3] + 3 * g[-2] - g[-1]
#     gvec[-1] = (3 * g[-1] - 4 * g[-2] + g[-3]) / (2 * self.dx[-1])
#     # jval_c[-1] = -2 # the final result should be -1 at J[-1, -1]. since we initialized the main diagonal with ones, we have to subtract 2!
#     jval_c[-1] = -1 + 3 / (2 * self.dx[-1])
#     # jval_l[-1] = 3
#     jval_l[-1] = -2 / self.dx[-1]
#     jval_2l, jval_3l = np.zeros(len(self.x) - 2), np.zeros(len(self.x) - 3)
#     # jval_2l[-1] = -3
#     jval_2l[-1] = 1 / (2 * self.dx[-1])
#     # jval_3l[-1] = 1
#     # add diagonals to jacobian matrix
#     # jac += sps.diags([jval_3l, jval_2l, jval_l[1:], jval_c, jval_r[:-1]], [-3, -2, -1, 0, 1])
#     jac += sps.diags([jval_2l, jval_l[1:, 0], jval_c[:, 0], jval_r[:-1, 0]], [-2, -1, 0, 1])
#     jac = jac.todense()
#     # Inhomogeneous part of the PDE at interior points
#     gvec[1:-1] += -g_old[1:-1]
#     M[1:-1] = self.gamma * (1 - self.FCAP * a_old[1:-1] * g_old[1:-1])
#     N[1:-1] = F[0, 1:-1] * (g_old[2:] - 2 * g_old[1:-1] + g_old[:-2]) / self.dx[0] ** 2 + (A[0, 1:-1] - T[0, 1:-1]) * (
#     g_old[2:] - g_old[:-2]) / (2 * self.dx[0])
#     # R[1:-1] = -(Qt * self.intTQF[1:-1] + intTrt[1:-1]) * gold[1:-1]
#     R[1:-1] = -intTF[0, 1:-1] * g_old[1:-1]
#     gvec += self.dt * (M * N + R) / 2.
#     return gvec, jac

