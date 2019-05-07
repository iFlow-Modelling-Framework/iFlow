"""
Equilibrium availability or erodibility computation using semi-analytical method

Options on input
    sedbc: type of horizontal boundary condition: 'csea' for seaward concentration or 'astar' for total amount of sediment in the domain

Optional input parameters
    Qsed: sediment inflow from upstream
    sedsource: other sediment sources

From version: 2.4 - update 2.6, 2.7
Date: 18 April 2019
Authors: R.L. Brouwer, Y.M. Dijkstra
Additions by Y.M. Dijkstra: - erodability (with time integrator); v2.6
                            - sources of sediment; v2.7
# TODO: allow for csea=0 in timeinteg (S not set error)
"""
import logging
import numpy as np
import nifty as ny
from src.DataContainer import DataContainer
import scipy.linalg
from scipy import integrate


class EquilibriumAvailability:
    # Variables
    logger = logging.getLogger(__name__)
    TOL = 10 ** -4  # Tolerance for determining the stock for the first iteration; not important for accuracy of the final answer
    TOL2 = 10 ** -13  # Tolerance for the convergence of the time-integrator for erodibility
    MAXITER = 1000  # maximum number of time steps in the time-integrator for erodibility
    dt = 3600 * 24 * 10  # time step in sec in the time-integrator
    # timers = [ny.Timer(), ny.Timer(),ny.Timer(), ny.Timer(),ny.Timer(), ny.Timer()]

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        # self.timers[0].tic()
        self.logger.info('Running module StaticAvailability')

        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x')
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=self.x/L).reshape((len(self.x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]

        c00 = np.real(self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), 0))
        c04 = np.abs(self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), 2))
        # c20 = np.real(self.input.v('hatc2', 'a', range(0, jmax+1), range(0, kmax+1), 0))       # NB. do not include hatc2 in the definition of alpha1 here
        alpha1 = ny.integrate(c00, 'z', kmax, 0, self.input.slice('grid'))[:, 0]
        alpha1[-1] += alpha1[-2]                                                                 # correct alpha1 at last point to prevent zero value
        alpha2 = ny.integrate(c04, 'z', kmax, 0, self.input.slice('grid'))[:, 0]/(alpha1+1e-10) + 1.e-3
        # self.timers[0].toc()

        ################################################################################################################
        ## Compute T and F
        ################################################################################################################
        # self.timers[1].tic()
        d = self.compute_transport()
        G = self.compute_source()
        # self.timers[1].toc()

        ################################################################################################################
        ## 4. Calculate availability
        ################################################################################################################
        # self.timers[2].tic()
        # Add all mechanisms to datacontainer
        dctrans = DataContainer(d)

        # Calculate availability
        a, f0, f0x = self.availability(dctrans.v('F', range(0, jmax+1)), dctrans.v('T', range(0, jmax+1)), G, alpha1, alpha2)
        f0 = f0.reshape(jmax+1, 1)
        f0x = f0x.reshape(jmax+1, 1)

        d['a'] = a
        nfu = ny.functionTemplates.NumericalFunctionWrapper(f0[:, 0], self.input.slice('grid'))
        nfu.addDerivative(f0x[:, 0], 'x')
        d['f'] = nfu.function
        # self.timers[2].toc()

        ################################################################################################################
        # 5. Calculate concentrations, i.e. a*hatc(a) + ax*hatc(ax)
        ################################################################################################################
        # self.timers[3].tic()
        d['c0'] = {}
        d['c1'] = {}
        d['c2'] = {}

        # Calculate c0=f*hatc0
        for submod in self.input.getKeysOf('hatc0', 'a'):
            c0_comp = self.input.v('hatc0', 'a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['c0'][submod] = {}
            tmp = f0[:, None] * c0_comp
            d['c0'][submod] = tmp

        # Calculate c1 = f*hatc1_f + fx*hatc1_fx
        for submod in self.input.getKeysOf('hatc1', 'a'):
            if submod == 'erosion':
                for subsubmod in self.input.getKeysOf('hatc1', 'a', 'erosion'):
                    c1_comp = self.input.v('hatc1', 'a', 'erosion', subsubmod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                    d['c1'] = self.dictExpand(d['c1'], 'erosion', subsubmod)
                    tmp = f0[:, None] * c1_comp
                    d['c1']['erosion'][subsubmod] = tmp

            elif submod == 'sedadv':
                c1_comp_a = self.input.v('hatc1', 'a', 'sedadv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                c1_comp_ax = self.input.v('hatc1', 'ax', 'sedadv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                d['c1'][submod] = {}
                tmp = f0[:, None] * c1_comp_a + f0x[:, None] * c1_comp_ax
                d['c1'][submod] = tmp

            else:
                c1_comp = self.input.v('hatc1', 'a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                d['c1'][submod] = {}
                tmp = f0[:, None] * c1_comp
                d['c1'][submod] = tmp

        # Calculate c2 = f*hatc2
        for subsubmod in self.input.getKeysOf('hatc2', 'a', 'erosion'):
            c2_comp = self.input.v('hatc2', 'a', 'erosion', subsubmod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['c2'] = self.dictExpand(d['c2'], 'erosion', subsubmod)
            tmp = f0[:, None] * c2_comp
            d['c2']['erosion'][subsubmod] = tmp

        # self.timers[3].toc()
        # self.timers[0].disp('time availability - init')
        # self.timers[1].disp('time availability - T, F')
        # self.timers[2].disp('time availability - a, f')
        # self.timers[3].disp('time availability - to dict')
        # self.timers[4].disp('time availability - cap')
        # self.timers[5].disp('time availability - trap')
        # [self.timers[i].reset() for i in range(0, len(self.timers))]

        return d

########################################################################################################################
## Functions related to the transport terms
########################################################################################################################
    def compute_transport(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c2 = self.input.v('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), 0)
        c0x = self.input.d('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        c2x = self.input.d('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), 0, dim='x')

        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        Kh = self.input.v('Kh', range(0, jmax+1), 0, 0)

        d = {}
        d['T'] = {}
        d['F'] = {}

        ## Transport T #################################################################################################
        # Transport terms that are a function of the first order velocity, i.e. u1*c0 terms.
        for submod in self.input.getKeysOf('u1'):
            u1_comp = self.input.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['T'] = self.dictExpand(d['T'], submod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
            # calculate residual Transport terms
            for n in (0, 2):
                tmp = u1_comp[:, :, n]
                if n==0:
                    if submod == 'stokes':
                        tmp = np.real(np.trapz(tmp * c0[:, :, 0], x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM0', ['return', 'drift'])
                            d['T'][submod]['TM0']['return'] += tmp
                    else:
                        tmp = np.real(np.trapz(tmp * c0[:, :, 0], x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp
                elif n==2:
                    if submod == 'stokes':
                        tmp = np.real(np.trapz((tmp * np.conj(c0[:, :, 2]) + np.conj(tmp) * c0[:, :, 2]) / 4., x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM4', ['return', 'drift'])
                            d['T'][submod]['TM4']['return'] += tmp
                    else:
                        tmp = np.real(np.trapz((tmp * np.conj(c0[:, :, 2]) + np.conj(tmp) * c0[:, :, 2]) / 4., x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp

        # Transport terms that are a function of the first order concentration, i.e. u0*c1 terms.
        for submod in self.input.getKeysOf('hatc1', 'a'):
            if submod == 'erosion':
                for subsubmod in self.input.getKeysOf('hatc1', 'a', submod):
                    d['T'] = self.dictExpand(d['T'], subsubmod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                    tmp = self.input.v('hatc1', 'a', submod, subsubmod, range(0, jmax+1), range(0, kmax+1), 1)
                    tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-self.zarr, axis=1))
                    if subsubmod == 'stokes':
                        if any(tmp) > 10**-14:
                            d['T'][subsubmod] = self.dictExpand(d['T'][subsubmod], 'TM2', ['return', 'drift'])
                            d['T'][subsubmod]['TM2']['return'] += tmp
                    else:
                        if any(tmp) > 10**-14:
                            d['T'][subsubmod]['TM2'] += tmp
            else:
                d['T'] = self.dictExpand(d['T'], submod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                tmp = self.input.v('hatc1', 'a', submod, range(0, jmax+1), range(0, kmax+1), 1)
                tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-self.zarr, axis=1))
                if any(tmp) > 10**-14:
                    d['T'][submod]['TM2'] += tmp

        # Transport terms that are related to diffusion, i.e. K_h*c0 or K_h*c2
        d['T'] = self.dictExpand(d['T'], 'diffusion_tide', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        d['T']['diffusion_tide']['TM0'] = np.real(-Kh*np.trapz(c0x[:, :, 0], x=-self.zarr, axis=1))

        d['T'] = self.dictExpand(d['T'], 'diffusion_river', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        tmp = np.real(-Kh*np.trapz(c2x, x=-self.zarr, axis=1))
        if any(tmp) > 10**-14:
            d['T']['diffusion_river']['TM0'] = tmp

        # Transport terms that are related to Stokes drift, i.e. u0*c0*zeta0
        if 'stokes' in self.input.getKeysOf('hatc1', 'a', 'erosion'):
            for n in (0, 2):
                u0s = u0[:, 0, 1]
                tmp = c0[:, 0, n]
                if n==0:
                    tmp = np.real(np.conj(u0s) * tmp * zeta0[:, 0, 1] + u0s * tmp * np.conj(zeta0[:, 0, 1])) / 4
                elif n==2:
                    tmp = np.real(u0s * np.conj(tmp) * zeta0[:, 0, 1] + np.conj(u0s) * tmp * np.conj(zeta0[:, 0, 1])) / 8
                if any(tmp) > 10**-14:
                    d['T']['stokes']['TM' + str(2 * n)]['drift'] = tmp


        # Transport term that is related to the river-river interaction u1river*c2river
        d['T'] = self.dictExpand(d['T'], 'river_river', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        if self.input.v('u1', 'river') is not None:
            u1_comp = self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax+1), 0)
            d['T']['river_river']['TM0'] = np.real(np.trapz(u1_comp * c2, x=-self.zarr, axis=1))

        ## Diffusion F #################################################################################################
        # Diffusive part, i.e. Kh*c00 and Kh*c20
        d['F'] = self.dictExpand(d['F'], 'diffusion_tide', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
        d['F']['diffusion_tide']['FM0'] = np.real(-Kh*np.trapz(c0[:, :, 0], x=-self.zarr, axis=1))

        d['F'] = self.dictExpand(d['F'], 'diffusion_river', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
        d['F']['diffusion_river']['FM0'] = np.real(-Kh*np.trapz(c2, x=-self.zarr, axis=1))

        # Part of F that is related to sediment advection, i.e. u0*c1sedadv
        for submod in self.input.getKeysOf('hatc1', 'ax'):
            d['F'] = self.dictExpand(d['F'], submod, ['FM' + str(2 * n) for n in range(0, fmax + 1)])
            tmp = self.input.v('hatc1', 'ax', submod, range(0, jmax+1), range(0, kmax+1), 1)
            tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-self.zarr, axis=1))
            if any(tmp) > 10**-14:
                d['F']['sedadv']['FM2'] += tmp
        return d

    def compute_source(self):
        ################################################################################################################
        ## Sourceterm G
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        B = self.input.v('B', range(0, jmax + 1), 0, 0)
        x = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]
        G = np.zeros(jmax+1)

        #   Upstream source
        if self.input.v('Qsed') is not None:
            G[:] = self.input.v('Qsed') / B

        #   Other sources
        if self.input.v('sedsource') is not None:
            Qsedsource = ny.toList(self.input.v('sedsource'))
            for i, s in enumerate(Qsedsource):
                if s[0] == 'point':
                    xs = s[1]
                    source = s[2]
                    G[np.where(x < xs)] += source / B[np.where(x < xs)]
                elif s[0] == 'line':
                    xmin = s[1]
                    xmax = s[2]

                    xi = np.zeros(jmax+1)
                    xi[1:] = 0.5 * x[1:] + 0.5 * x[:-1]
                    source = s[3]
                    G += source / B * np.minimum(np.maximum(xmax - xi, 0), xmax - xmin)
        return G

########################################################################################################################
## Functions related to the bed-evolution equation (erodibility or availability)
########################################################################################################################
    def availability(self, F, T, G, alpha1, alpha2):
        """Calculates the solution to the bed-evolution equation: the erodibility and availability of sediment

        Parameters:
            F - diffusive coefficient in the availability equation that goes with a_x
            T - coefficient (advective, diffusive and stokes) in the availability equation that goes with a

        Returns:
            a - availability of sediment
            f - erodibility of sediment
        """
        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        B = self.input.v('B', range(0, jmax + 1), 0, 0)
        x = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]
        dx = x[1:] - x[:-1]

        ################################################################################################################
        ## Solution to bed-evolution equation
        ################################################################################################################
        ## analytical solution
        if np.all(G == 0):
            P = np.zeros(jmax+1)
        else:
            P = integrate.cumtrapz(G / (F - 10 ** -6) * np.exp(integrate.cumtrapz(T / F, dx=dx, axis=0, initial=0)), dx=dx, axis=0, initial=0)

        exponent = np.exp(-integrate.cumtrapz(T / F, dx=dx, axis=0, initial=0))

        # Boundary conditions (analytical solution)
            # BC 1: total amount of sediment in the system
        if self.input.v('sedbc') == 'astar':
            astar = self.input.v('astar')
            k = (astar * np.trapz(B, dx=dx, axis=0) / np.trapz(B * exponent, dx=dx, axis=0))

            # BC 2: concentration at the seaward boundary
        elif self.input.v('sedbc') == 'csea':
            csea = self.input.v('csea')
            c000 = alpha1[0]
            k = csea / c000 * (self.input.v('grid', 'low', 'z', 0) - self.input.v('grid', 'high', 'z', 0))

            # BC 3: incorrect boundary description
        else:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('incorrect seaward boundary type (sedbc) for sediment module')

        # final solution (analytical)
        f0uncap = (k - P) * exponent

        ## Check if f<1 everywhere,
        # if not compute numerical solution instead with a time-stepping routine that maximises f at 1.
        if all(f0uncap < 1):
            f0 = f0uncap
        else:
            f0, Smod = self.availability_numerical(np.real(T), np.real(F), np.real(f0uncap), k, alpha1, alpha2, G)

        ## compute derivative
        if np.all(G == 0) and all(f0uncap < 1):
            f0x = -T / F * f0uncap  # only use analytical derivative in case with no fluvial source of sediment and f<1; else not reliable.
        else:
            f0x = ny.derivative(f0, 'x', self.input.slice('grid'))

        return f0uncap, f0, f0x

    def availability_numerical(self, T, F, f0uncap, fsea, alpha1, alpha2, G):
        """Calculates the time-evolution of the sediment distribution in an estuary using an implicit solution method
        """
        ## init
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]
        jmax = self.input.v('grid', 'maxIndex', 'x')
        dt = self.dt
        dx = x[1:] - x[:-1]
        B = self.input.v('B', range(0, jmax + 1))

        # check seaward boundary condition; correct the boundary condition if fsea>1
        if fsea > 1:
            self.logger.warning('Sediment concentration cannot satisfy the prescribed boundary condition at x=0.\n'
                                'Setting the seaward concentration to its maximum.')
            fsea = 0.999

        # remove any nans or infs from f0uncap
        f0uncap[np.where(f0uncap==np.inf)] = 0
        f0uncap[np.where(np.isnan(f0uncap))] = 0

        aeq = f0uncap / f0uncap[0]  # scale the availability such that aeq=1 at x=0

        ## Initialise solution vector X = (f/aeq, S)
        X = self.init_erodibility_stock(f0uncap, alpha1, alpha2)
        X[np.where(np.isnan(X))] = 0.

        ## Time stepping
        i = 0  # counter for number of time steps
        f0 = aeq * X[:jmax + 1]  # compute f at beginning of time step 0
        Smod = X[jmax + 1:]  # S at beginning of time step 0
        difference = np.inf

        while difference > self.TOL2 * dt / (3600 * 24) and i < self.MAXITER:
            i += 1
            X = self.timestepping_stock(X, B, F, aeq, fsea, dx, dt, alpha1, alpha2, G, f0uncap[0])  # time stepping routine
            difference = max(abs(aeq * X[:jmax + 1] - f0) / abs(f0 + 10 ** -2 * np.max(f0)))  # check convergence

            f0 = aeq * X[:jmax + 1]  # compute f at beginning of time step i+1
            Smod = X[jmax + 1:]  # S at beginning of time step i+1

        ## Report result
        self.logger.info('\t Erosion limited conditions; time iterator took %s iterations, last change: %s' % (str(i), str(difference)))
        self.X = X

        return f0, Smod

    def init_erodibility_stock(self, f0uncap, alpha1, alpha2):
        # Initialize vector g
        jmax = self.input.v('grid', 'maxIndex', 'x')
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]

        # Establish initial minimum stock Smax, such that f=0.999
        Smax = np.array([0.999])  # initial guess
        F = self.erodibility_stock_relation(alpha2[[0]], Smax) - np.array([0.999])
        while max(abs(F)) > self.TOL:
            dfdS = self.erodibility_stock_relation_der(alpha2[[0]], Smax)
            Smax = Smax - F / dfdS
            F = self.erodibility_stock_relation(alpha2[[0]], Smax) - np.array([0.999])

        # establish starting X=(g, S)
        if not hasattr(self, 'X'):
            Lb = -x / np.log(1. / (f0uncap + 1.e-200))
            Lb = Lb[np.where(Lb > 0)]
            if len(Lb) > 0:
                Lb = np.min(Lb) * 0.9
                g = f0uncap[0] * np.exp(-x / Lb)
            else:
                g = f0uncap[0] * np.ones(x.shape)

            # define initial erodibility f
            f = f0uncap / f0uncap[0] * g

            # Initiate stock S with exact expression (see notes Yoeri)
            Shat = f
            F = self.erodibility_stock_relation(alpha2, Shat) - f

            # Newton-Raphson iteration towards actual Shat
            i = 0
            while max(abs(F)) > self.TOL and i < 50:
                i += 1
                dfdS = self.erodibility_stock_relation_der(alpha2, Shat)
                Shat = Shat - F / (dfdS + 10 ** -12)
                F = self.erodibility_stock_relation(alpha2, Shat) - f
                if i == 50:
                    g = self.erodibility_stock_relation(alpha2, Shat) * f0uncap[0] / f0uncap

            Shat = np.minimum(Shat, Smax)

            # Define solution vector X = (g, S)
            X = np.append(g, Shat * alpha1).reshape(2 * len(x))
        else:
            X = self.X
            X[jmax + 1:] = np.minimum(X[jmax + 1:], Smax * alpha1)

        return X

    def timestepping_stock(self, X, B, F, atil, fsea, dx, dt, alpha1, alpha2, G, a0):
        jmax = len(B) - 1
        # AX=rhs, with X=(g, S)
        A = np.zeros((3, jmax + 1))
        rhs = np.zeros(jmax + 1)

        ################################################################################################################
        # Exner equation
        ################################################################################################################
        htil = self.erodibility_stock_relation(alpha2, X[jmax + 1:] / alpha1) / (atil+1e-20)
        htil_der = self.erodibility_stock_relation_der(alpha2, X[jmax + 1:] / alpha1) / (alpha1 * (atil+1e-20))

        # transport
        dxav = 0.5 * (dx[:-1] + dx[1:])
        a = 0.5 * (B[:-2] + B[1:-1]) / B[1:-1] * 0.5 * (F[:-2] + F[1:-1]) * 0.5 * (atil[:-2] + atil[1:-1]) / (dx[:-1] * dxav)
        c = 0.5 * (B[2:] + B[1:-1]) / B[1:-1] * 0.5 * (F[2:] + F[1:-1]) * 0.5 * (atil[2:] + atil[1:-1]) / (dx[1:] * dxav)
        b = -(a + c) * htil_der[1:-1] + 1. / dt

        d = -c * (htil[2:jmax + 1] - htil[1:jmax]) + a * (htil[1:jmax] - htil[:jmax - 1])
        d += c * (htil_der[2:jmax + 1] * X[jmax + 3:] - htil_der[1:jmax] * X[jmax + 2:-1]) - a * (htil_der[1:jmax] * X[jmax + 2:-1] - htil_der[:jmax - 1] * X[jmax + 1:-2])
        d += X[jmax + 2:-1] / dt

        a = a * htil_der[:-2]
        c = c * htil_der[2:]

        # # upstream source term                    ## 10-07-2018 Incorrect
        # b +=  G[1:-1]/(a0*dx[:-1])*htil_der[1:-1]
        # c += -G[1:-1]/(a0*dx[:-1])*htil_der[2:]
        # d += -G[1:-1]/(a0*dx[:-1])*(htil_der[2:]*X[jmax+3:]-htil_der[1:-1]*X[jmax+2:-1])
        # d +=  G[1:-1]/(a0*dx[:-1])*(htil[2:]-htil[1:-1])

        # Source term
        # div = B[1:-1]*a0*(dx[:-1]+dx[1:])
        # a += B[:-2]*G[:-2]*htil_der[:-2]/div
        # c += -B[2:]*G[2:]*htil_der[2:]/div
        # d += -B[2:]*G[2:]*(a0-htil[2:]+htil_der[2:]*X[jmax+3:])/div + B[:-2]*G[:-2]*(a0-htil[:-2]+htil_der[:-2]*X[jmax+1:-2])/div
        div = B[1:-1] * a0 * (dx[1:])
        b += B[1:-1] * G[1:-1] * htil_der[1:-1] / div
        c += -B[2:] * G[2:] * htil_der[2:] / div
        d += -B[2:] * G[2:] * (a0 - htil[2:] + htil_der[2:] * X[jmax + 3:]) / div + B[1:-1] * G[1:-1] * (a0 - htil[1:-1] + htil_der[1:-1] * X[jmax + 2:-1]) / div

        A[0, range(2, jmax + 1)] = c
        A[1, range(1, jmax)] = b
        A[2, range(0, jmax - 1)] = a
        rhs[range(1, jmax)] = d

        # Boundaries
        #   x=0
        if htil_der[0] == 0:
            A[1, 0] = 1
            rhs[0] = X[jmax + 1]
        else:
            A[1, 0] = htil_der[0] * atil[0]
            rhs[0] = fsea - htil[0] * atil[0] + htil_der[0] * atil[0] * X[jmax + 1]

        #   x=L
        a = 0.5 * (B[jmax] + B[jmax - 1]) / B[jmax] * 0.5 * (F[jmax] + F[jmax - 1]) * 0.5 * (atil[jmax] + atil[jmax - 1]) / (dx[-1] * 0.5 * dx[-1])
        b = -a * htil_der[-1] + 1. / dt

        d = a * (htil[jmax] - htil[jmax - 1])
        d += - a * (htil_der[jmax] * X[-1] - htil_der[jmax - 1] * X[-2])
        d += X[-1] / dt

        a = a * htil_der[-2]

        # upstream source
        # a += G[-1]*htil_der[-2]/(dx[-1]*a0)
        # b += G[-1]*htil_der[-1]/(dx[-1]*a0)
        # d += 1./(.5*dx[-1])*(G[-1]-.5*G[-1]/a0*(htil[-2]+htil[-1]))
        # d += 1./(.5*dx[-1])*(.5*G[-1]/a0*(htil_der[-2]*X[-2]+htil_der[-1]*X[-1]))
        fac = 0.5 * (B[-1] + B[-2]) * 0.5 * (G[-1] + G[-2]) / (0.5 * B[-1] * a0 * dx[-1])
        a += 0.5 * htil_der[-2] * fac
        b += 0.5 * htil_der[-1] * fac
        d += (a0 - 0.5 * (htil[-1] + htil[-2]) + 0.5 * (htil_der[-1] * X[-1] + htil_der[-2] * X[-2])) * fac

        A[1, -1] = b
        A[2, -2] = a
        rhs[-1] = d

        # solve system of equations
        S = scipy.linalg.solve_banded((1, 1), A, rhs)

        X[:jmax + 1] = htil + htil_der * (S - X[jmax + 1:])
        X[jmax + 1:] = S
        return X

    def erodibility_stock_relation(self, alpha2, Shat):
        # warnings.filterwarnings("ignore")       # suppress runtime warning on arcsin for Shat>1
        xi = np.arcsin(np.maximum(np.minimum((Shat - 1.) / alpha2, 1), -1))
        f = Shat * (0.5 - xi / np.pi) + 0.5 + xi / np.pi - alpha2 * np.cos(xi) / np.pi
        f[np.where(Shat < 1 - alpha2)[0]] = Shat[np.where(Shat < 1 - alpha2)[0]]
        f[np.where(Shat > 1 + alpha2)[0]] = 1.
        # warnings.filterwarnings("default")
        return np.real(f)

    def erodibility_stock_relation_der(self, alpha2, Shat):
        # warnings.filterwarnings("ignore")       # suppress runtime warning on arcsin for Shat>1
        xi = np.arcsin(np.maximum(np.minimum((Shat - 1.) / alpha2, 1), -1))
        dfdS = 0.5 - xi / np.pi
        dfdS[np.where(Shat <= 1 - alpha2)[0]] = 1.
        dfdS[np.where(Shat >= 1 + alpha2)[0]] = 0.
        # warnings.filterwarnings("default")
        return np.real(dfdS)

    def erodibility_stock_relation_da2(self, alpha2, Shat):
        # warnings.filterwarnings("ignore")       # suppress runtime warning on arcsin for Shat>1
        xi = np.maximum((alpha2**2.-(Shat-1)**2.)/alpha2**2., 0)
        dfda = -np.sqrt(xi)/np.pi
        dfda[np.where(Shat <= 1 - alpha2)[0]] = 0.
        dfda[np.where(Shat >= 1 + alpha2)[0]] = 0.
        # warnings.filterwarnings("default")
        return dfda

    def dictExpand(self, d, subindex, subsubindices):
        """Adds a maximum of two sublayers to a dictionary

        Parameters:
            d             - dictionary to expand
            subindex      - first layer expansion (only one subindex possible); string
            subsubindices - second layer expansion (multiple subsubindices possible; list of strings

        Returns:
            d - expanded dictionary
        """

        if not subindex in d:
            d[subindex] = {}
        elif not isinstance(d[subindex], dict):
            d[subindex] = {}
        for ssi in ny.toList(subsubindices):
            if not ssi in d[subindex]:
                d[subindex][ssi] = 0.
        return d
