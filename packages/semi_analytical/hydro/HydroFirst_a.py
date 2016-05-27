"""

Date: 01-06-15
Authors: R.L. Brouwer
"""
import matplotlib.pyplot as plt
import numpy as np
import scikits.bvp_solver
from scipy import integrate
from nifty.functionTemplates.NumericalFunctionWrapper import NumericalFunctionWrapper


class HydroFirst:
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        self.submodule = submodulesToRun
        self.input = input
        return

    def run(self):
        print 'HydroFirst runs ...'
        # Initiate variables
        self.SIGMA = self.input.v('OMEGA')
        self.NGODIN = self.input.v('NGODIN')
        self.G = self.input.v('G')
        self.BETA = self.input.v('BETA')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        self.dx = self.x[1:]-self.x[:-1]
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = (self.z.reshape(1, len(self.z)) * self.input.v('-H', x=self.x/self.L).reshape(len(self.x), 1)).T
        self.dz = -(self.zarr[1, :] - self.zarr[0, :])
        self.zud = np.flipud(self.zarr)
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=0)
        self.H = self.input.v('H', x=self.x/self.L)
        self.B = self.input.v('B', x=self.x/self.L)
        self.Bx = self.input.d('B', x=self.x/self.L, dim='x')
        self.__bc = np.zeros(4)
        self.__F = []

        # Allocate space for results
        d = dict()
        d['zeta1'] = {}
        d['u1'] = {}
        for mod in self.submodule:
            zeta, u = getattr(self, mod)()
            nfz = NumericalFunctionWrapper(zeta[0], self.input.slice('grid'))
            nfz.addDerivative(zeta[1], dim='x')
            nfz.addSecondDerivative(zeta[1], dim='x')
            nfu = NumericalFunctionWrapper(u[0], self.input.slice('grid'))
            d['zeta1'][mod] = nfz.function
            d['u1'][mod] = nfu.function

        self.input.merge(d)
        self.plot3()
        return d

    def rf(self, x):
        """Calculate the root of the characteristic equation.

        Parameters:
            x - x-coordinate or coordinates

        Returns:
            root of the characteristic equation

        """
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.dd('Av', x=x/self.L, z=0, f=0, dim='x')])

        r = np.zeros(3, dtype=complex)
        r[0] = np.sqrt(2 * 1j * self.SIGMA / Av[0])
        r[1] = -np.sqrt(2 * 1j * self.SIGMA) * Av[1] / (2 * Av[0]**(3./2.))
        r[2] = np.sqrt(2 * 1j * self.SIGMA) * (3 * Av[1]**2 - 2 * Av[0] * Av[2]) / (4 * Av[0]**(5./2.))
        return r

    def af(self, x):
        """Calculate the coefficient alpha.

        Parameters:
            x - x-coordinate or coordinates

        Returns:
            coefficient alpha

        """
        H = np.array([self.input.v('H', x=x/self.L),
                      self.input.d('H', x=x/self.L, dim='x'),
                      self.input.dd('H', x=x/self.L, dim='x')])
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.dd('Av', x=x/self.L, z=0, f=0, dim='x')])
        sf = self.NGODIN * np.array([self.input.v('Roughness', x=x/self.L),
                                     self.input.d('Roughness', x=x/self.L, dim='x'),
                                     self.input.dd('Roughness', x=x/self.L, dim='x')])
        r = self.rf(x)
        # Define trigonometric values for ease of reference
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        cothrh = coshrh / sinhrh
        cschrh = 1. / sinhrh
        # Define parameters and their (second) derivative wrt x
        E = r[1] * H[0] + r[0] * H[1]
        Ex = r[2] * H[0] + 2. * r[1] * H[1] + r[0] * H[2]
        F = r[1] + r[0] * E * cothrh
        Fx = r[2] + r[0] * Ex * cothrh + E * (r[1] * cothrh - r[0] * E**2. * cschrh**2.)
        K = r[0] * Av[1] + Av[0] * F + sf[1] * cothrh + sf[0] * E
        Kx = (r[0] * Av[2] + r[1] * Av[1] + Av[1] * F + Av[0] * Fx + sf[2] * cothrh -
              sf[1] * E * cschrh**2. + sf[1] * E + sf[0] * Ex)
        G = r[0] * Av[0] * sinhrh + sf[0] * coshrh
        Gx = sinhrh * K
        Gxx = E * K * coshrh + Kx * sinhrh
        # Calculate coefficient alpha
        a = np.zeros(3, dtype=complex)
        a[0] = sf[0] / G    # a
        a[1] = sf[1] / G - sf[0] * Gx / G**2.    # a_x
        a[2] = (sf[2] - 2. * (sf[1] * Gx + sf[0] * Gxx) / G + 2. * sf[0] * Gx**2. / G**2.) / G    # a_xx
        return a

    def variables(self, x):
        vars = ['Roughness', 'r', 'a']
        v = {key: [] for key in vars}
        for var in v:
            for xi in x:
                if var is 'r':
                    v[var].append(self.rf(xi))
                elif var is 'a':
                    v[var].append(self.af(xi))
                elif var is 'Roughness':
                    v[var].append([self.NGODIN * self.input.v(var, x=xi/self.L),
                                   self.NGODIN * self.input.d(var, x=xi/self.L, dim='x'),
                                   self.NGODIN * self.input.dd(var, x=xi/self.L, dim='x')])
            v[var] = np.array(v[var])
        return v

    def bcs(self, Za, Zb):
        """Defines the boundary conditions at x = 0 and x = 1"""
        bca = np.zeros(2)
        bcb = np.zeros(2)
        bca[0] = Za[0] + self.__bc[0]  # Real valued boundary condition at x = 0
        bca[1] = Za[1] + self.__bc[1]  # Imaginary valued boundary condition at x = 0
        bcb[0] = Zb[2] + self.__bc[2]  # Real valued boundary condition at x = 1
        bcb[1] = Zb[3] + self.__bc[3]  # Imaginary valued boundary condition at x = 1
        return (bca), (bcb)

    def system_ode(self, x, Z):
        """Defines the system of ODEs to be passed on to bvp_solver"""
        # Import system variables
        H = np.array([self.input.v('H', x=x/self.L),
                      self.input.d('H', x=x/self.L, dim='x'),
                      self.input.dd('H', x=x/self.L, dim='x')])
        B = np.array([self.input.v('B', x=x/self.L),
                      self.input.d('B', x=x/self.L, dim='x'),
                      self.input.dd('B', x=x/self.L, dim='x')])
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.dd('Av', x=x/self.L, z=0, f=0, dim='x')])

        r = self.rf(x)
        a = self.af(x)
        # Define local parameters
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        # Factors in ODE
        T1 = ((a[0] * sinhrh / r[0]) - H[0])
        T2 = ((-B[1] * T1/B[0] + H[1] * (1. - a[0] * coshrh) - a[1] * sinhrh / r[0] +
               a[0] * r[1] * (sinhrh - r[0] * H[0] * coshrh) / r[0]**2.))
        T3 = 4 * self.SIGMA**2. / self.G
        if self.__F is 'tide':
            T4 = 0
        elif self.__F is 'stokes':
            T4 = -2.*Av[0] * r[0]**2. * (self.input.v('gamma_x', x=x/self.L, z=0, f=2) +
                                         B[1] * self.input.v('gamma', x=x/self.L, z=0, f=2) / B[0]) / self.G
        elif self.__F is 'nostress':
            FnsB = (Av[0]*r[0]**2. / self.G) * (B[1]/B[0]) * (2. * self.input.v('xi', x=x/self.L, z=0, f=2) * (1. - a[0]) / r[0]**2.)
            Fnsdx = (Av[0]*r[0]**2. / self.G) * 2. * ((self.input.v('xi_x', x=x/self.L, z=0, f=2) * r[0] * (1. - a[0]) -
                                                       self.input.v('xi', x=x/self.L, z=0, f=2) * (r[0]*a[1] + 2.*r[1]*(1. - a[0]))) /
                                                      r[0]**3.)
            T4 = FnsB + Fnsdx
        elif self.__F is 'adv':
            T4 = self.input.v('Fadv', x=x/self.L)

        # Definition of ODE
        dzdx = np.zeros(4)
        dzdx[0] = Z[2]
        dzdx[1] = Z[3]
        dzdx[2] = (np.real(T2 / T1) * Z[2] + np.real(T3 / T1) * Z[0] -
                   np.imag(T2 / T1) * Z[3] - np.imag(T3 / T1) * Z[1] + np.real(T4 / T1))
        dzdx[3] = (np.imag(T2 / T1) * Z[2] + np.imag(T3 / T1) * Z[0] +
                   np.real(T2 / T1) * Z[3] + np.real(T3 / T1) * Z[1] + np.imag(T4 / T1))
        return dzdx

    def tide(self):
        """Calculates the first order contribution due to the external tide. This contribution only has an M4-component
        """
        ## Define and initiate variables ##
        self.__F = 'tide'
        # Extract parameters alpha and r
        v = self.variables(self.x)

        ## M4 contribution ##
        # Define boundary conditions
        self.__bc = np.array([-self.input.v('A1', 2)*np.cos(self.input.v('phase1', 2)*np.pi/180),
                              self.input.v('A1', 2)*np.sin(self.input.v('phase1', 2)*np.pi/180), 0., 0.])
        # Define problem to solve the ode for the water level
        problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                       num_parameters=0,
                                                       num_left_boundary_conditions=2,
                                                       boundary_points=(self.x[0], self.x[-1]),
                                                       function=self.system_ode,
                                                       boundary_conditions=self.bcs)
        guess = np.array([self.input.v('A1', 2), 0.0, 0.0, 0.0])
        solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.input.v('tolerance'))
        # Extract solution
        Z1, Z2 = solution(self.x, eval_derivative=True)
        # Surface elevation and its derivatives wrt x
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
        zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
        zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        # Calculate the velocity
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2] * (1. - v['a'][:, 0] * np.cosh(v['r'][:, 0] * self.zarr))).T
        return zeta, u

    def stokes(self):
        """Calculates the first order contribution due to the Stokes return flow.
        """
        ## Calculate and save forcing terms for the Stokes contribution
        gammaM0 = 0.5 * np.real(self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                                np.conj(self.input.v('u0', 'tide', range(0, len(self.x)), 0, 1)))
        gammaM4 = 0.25 * (self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                          self.input.v('u0', 'tide', range(0, len(self.x)), 0, 1))
        gammaM4x = 0.25 * (self.input.d('zeta0', 'tide', range(0, len(self.x)), 0, 1, dim='x') *
                           self.input.v('u0', 'tide', range(0, len(self.x)), 0, 1) +
                           self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                           self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='x'))
        gamma = np.zeros((len(gammaM0), 1, 3), dtype=complex)
        gamma[:, 0, 0] = gammaM0
        gamma[:, 0, 2] = gammaM4
        gamma_x = np.zeros((len(gammaM4x), 1, 3), dtype=complex)
        gamma_x[:, 0, 2] = gammaM4x
        self.input.addData('gamma', gamma)
        self.input.addData('gamma_x', gamma_x)
        ## Define and initiate variables ##
        self.__F = 'stokes'
        # Extract parameters alpha, r, H, B, Av, and roughness as a function of the x-coordinate
        v = self.variables(self.x)
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        ### M0 contribution ###
        # Calculate M0 water level
        # zeta[1, :, 0, 0] = (self.input.v('gamma', x=self.x/self.L, z=0, f=0) /
        #                     (G*(self.H**3 / (3*self.Av0) +
        #                         self.H**2 / v['Roughness'][:, 0])))
        zeta[1, :, 0, 0] = (gammaM0 / (self.G * (self.H**3. / (3. * self.Av0) + self.H**2. / v['Roughness'][:, 0])))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], dx=self.dx, axis=0)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = ((0.5 * self.zarr**2. / self.Av0 - 0.5 * self.H**2. / self.Av0 -
                          self.H / v['Roughness'][:, 0]) * self.G * zeta[1, :, 0, 0]).T

        ## M4 contribution ##
        # Define boundary conditions
        self.__bc = np.array([0., 0.,
                              np.real(4. * 1j * self.SIGMA * self.input.v('gamma', x=1, z=0, f=2) /
                                      (self.G * (v['a'][-1, 0] * np.sinh(v['r'][-1, 0] * self.input.v('H', x=1)) /
                                       v['r'][-1, 0] - self.input.v('H', x=1)))),
                              np.imag(4. * 1j * self.SIGMA * self.input.v('gamma', x=1, z=0, f=2) /
                                      (self.G * (v['a'][-1, 0] * np.sinh(v['r'][-1, 0] * self.input.v('H', x=1)) /
                                       v['r'][-1, 0] - self.input.v('H', x=1))))])
        # Define problem to solve the ode for the water level
        problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                       num_parameters=0,
                                                       num_left_boundary_conditions=2,
                                                       boundary_points=(self.x[0], self.x[-1]),
                                                       function=self.system_ode,
                                                       boundary_conditions=self.bcs)
        guess = np.array([0.01, 0., 0., 0.])
        solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.input.v('tolerance'))
        # Extract solution
        Z1, Z2 = solution(self.x, eval_derivative=True)
        # Surface elevation and its derivatives wrt x
        zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
        zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
        zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        # Calculate the velocity
        u[0, :, :, 2] = -((self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2] * (1 - v['a'][:, 0] * np.cosh(v['r'][:, 0] * self.zarr))).T

        # plot results
        # self.plot2(zeta, u)
        return zeta, u

    def nostress(self):
        """Calculates the first order contribution due to the no-stress boundary condition.
        """
        ## Calculate and save forcing terms for the no-stress contribution
        xiM0 = 0.5 * np.real(self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                             np.conj(self.input.dd('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z')))
        xiM4 = 0.25 * (self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                       self.input.dd('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z'))
        xiM4x = 0.25 * (self.input.d('zeta0', 'tide', range(0, len(self.x)), 0, 1, dim='x') *
                        self.input.dd('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z') +
                        self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                        self.input.v('u0_zz_x', 'tide', range(0, len(self.x)), 0, 1))
        xi = np.zeros((len(xiM0), 1, 3), dtype=complex)
        xi[:, 0, 0] = xiM0
        xi[:, 0, 2] = xiM4
        xi_x = np.zeros((len(xiM4x), 1, 3), dtype=complex)
        xi_x[:, 0, 2] = xiM4x
        self.input.addData('xi', xi)
        self.input.addData('xi_x', xi_x)
        ## Define and initiate variables ##
        self.__F = 'nostress'
        # Extract parameters alpha, r, H, B, Av, and roughness as a function of the x-coordinate
        v = self.variables(self.x)
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        ### M0 contribution ###
        # Calculate M0 water level
        # f = np.trapz(xiM0 * (self.zud + self.H + self.Av0 / v['Roughness'][:, 0]), dx=self.dz, axis=0)
        # f = np.trapz(xiM0 * (self.zud + self.H + self.Av0 / v['Roughness'][:, 0]), x=-self.zarr, axis=0)
        # zeta[1, :, 0, 0] = -f / (self.G * (self.H**3. / (3. * self.Av0) + self.H**2. / v['Roughness'][:, 0]))
        zeta[1, :, 0, 0] = -(self.Av0 * xiM0 * (self.H / 2. + self.Av0 / v['Roughness'][:, 0])) / (self.G * self.H * (self.H / 3 + self.Av0 / v['Roughness'][:, 0]))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], dx=self.dx, axis=0)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = ((0.5 * self.zarr**2. / self.Av0 - 0.5 * self.H**2. / self.Av0 -
                          self.H / v['Roughness'][:, 0]) * self.G * zeta[1, :, 0, 0] -
                         (self.zarr + self.H + self.Av0 / v['Roughness'][:, 0]) * xiM0).T

        ## M4 contribution ##
        # Define boundary conditions
        self.__bc = np.array([0., 0., 0., 0.])
        # Define problem to solve the ode for the water level
        problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                       num_parameters=0,
                                                       num_left_boundary_conditions=2,
                                                       boundary_points=(self.x[0], self.x[-1]),
                                                       function=self.system_ode,
                                                       boundary_conditions=self.bcs)
        guess = np.array([0., 0., 0., 0.])
        solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.input.v('tolerance'))
        # Extract solution
        Z1, Z2 = solution(self.x, eval_derivative=True)
        # Surface elevation and its derivatives wrt x
        zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
        zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
        zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2] * (1. - v['a'][:, 0] * np.cosh(v['r'][:, 0] * self.zarr)) -
                         2. * v['a'][:, 0] * self.input.v('xi', x=self.x/self.L, z=0, f=2) *
                         (self.Av0 * np.cosh(v['r'][:, 0] * (self.zarr + self.H)) +
                          v['Roughness'][:, 0] * np.sinh(v['r'][:, 0] * (self.zarr + self.H)) / v['r'][:, 0]) /
                         v['Roughness'][:, 0]).T

        # plot results
        # self.plot2(zeta, u)
        return zeta, u

    def adv(self):
        """Calculates the first order contribution due to the advection of momentum.
        """
        ## Calculate and save forcing terms for the advection of momentum contribution
        etaM0 = 0.5 * np.real(self.input.v('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                              np.conj(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x')) +
                              self.input.v('w0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                              np.conj(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z'))).T
        etaM4 = 0.25 * (self.input.v('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                        self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x') +
                        self.input.v('w0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                        self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z')).T
        eta = np.zeros((etaM0.shape[0], etaM0.shape[1], 3), dtype=complex)
        eta[:, :, 0] = etaM0[:]
        eta[:, :, 2] = etaM4[:]
        self.input.addData('eta', eta)
        ## Define and initiate variables ##
        self.__F = 'adv'
        # Extract parameters alpha, r, H, B, Av, and roughness as a function of the x-coordinate
        v = self.variables(self.x)
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        ## M0 contribution ##
        # Particular solution Up of the advection term
        Up = np.zeros(self.zarr.shape, dtype=complex)
        f = integrate.cumtrapz(etaM0 * self.zud, dx=self.dz, axis=0)
        g = integrate.cumtrapz(etaM0, dx=self.dz, axis=0) * self.zud[1:, :]
        Up[1:, :] = g - f
        # Vertical integral of the advection term
        # h = np.trapz(etaM0, dx=self.dz, axis=0)
        h = np.trapz(etaM0, x=-self.zarr, axis=0)
        # Vertical integral of the particular solution
        # k = np.trapz(Up, dx=self.dz, axis=0)
        k = np.trapz(Up, x=-self.zarr, axis=0)
        # Calculate M0 water level
        zeta[1, :, 0, 0] = ((k - (0.5 * self.H**2. + self.Av0 * self.H / v['Roughness'][:, 0]) * h) /
                            (self.G * (self.H**3. / 3. + self.Av0 * self.H**2. / v['Roughness'][:, 0])))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], dx=self.dx, axis=0)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = ((0.5 * self.zarr**2. / self.Av0 - 0.5 * self.H**2. / self.Av0 -
                          self.H / v['Roughness'][:, 0]) * self.G * zeta[1, :, 0, 0] + np.flipud(Up / self.Av0) -
                         (self.zarr / self.Av0 + self.H / self.Av0 + 1. / v['Roughness'][:, 0]) * h).T

        ## M4 contribution ##
        # Define boundary conditions
        self.__bc = np.array([0., 0., 0., 0.])
        # Calculate variables
        # G1 variable from manual, separated in an (a) and a (b) part first
        G1a = integrate.cumtrapz(etaM4 * np.exp(-v['r'][:, 0] * self.zud), dx=self.dz, axis=0)
        G1a = np.append(G1a, np.zeros((1, len(self.x))), axis=0)
        G1b = integrate.cumtrapz(etaM4 * np.exp(v['r'][:, 0] * self.zud), dx=self.dz, axis=0)
        G1b = np.append(G1b, np.zeros((1, len(self.x))), axis=0)
        G1 = np.exp(v['r'][:, 0] * self.zud) * G1a - np.exp(-v['r'][:, 0] * self.zud) * G1b
        # G2 variable from manual
        # G2 = (np.trapz(etaM4 * np.exp(-v['r'][:, 0] * self.zud), dx=self.dz, axis=0) +
        #       np.trapz(etaM4 * np.exp(v['r'][:, 0] * self.zud), dx=self.dz, axis=0))
        G2 = (np.trapz(etaM4 * np.exp(-v['r'][:, 0] * self.zud), x=-self.zarr, axis=0) +
              np.trapz(etaM4 * np.exp(v['r'][:, 0] * self.zud), x=-self.zarr, axis=0))

        # calculating the advective forcing term for the water level and save it to the data container
        # Fadv1 = (G2 * (1 - v['a'][:, 0]) / (self.Av0 * v['r'][:, 0]**2) -
        #          np.trapz(G1, dx=self.dz, axis=0) / (self.Av0 * v['r'][:, 0]))
        Fadv1 = (G2 * (1. - v['a'][:, 0]) / (self.Av0 * v['r'][:, 0]**2.) -
                 np.trapz(G1, x=-self.zarr, axis=0) / (self.Av0 * v['r'][:, 0]))
        Fadv2 = np.zeros(len(self.x), dtype=complex)
        Fadv2[0] = (-3. * Fadv1[0] / 2. + 2. * Fadv1[1] - Fadv1[2] / 2.) / self.dx[0]
        Fadv2[-1] = (3. * Fadv1[-1] / 2. - 2. * Fadv1[-2] + Fadv1[-3] / 2.) / self.dx[-1]
        Fadv2[1:-1] = (Fadv1[2:] - Fadv1[:-2]) / (2. * self.dx[1:])
        Fadv = (self.Av0 * v['r'][:, 0]**2. / self.G) * (Fadv2 + self.Bx * Fadv1 / self.B)
        self.input.addData('Fadv', Fadv)
        # Define problem to solve the ode for the water level
        problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                       num_parameters=0,
                                                       num_left_boundary_conditions=2,
                                                       boundary_points=(self.x[0], self.x[-1]),
                                                       function=self.system_ode,
                                                       boundary_conditions=self.bcs)
        guess = np.array([0., 0., 0., 0.])
        solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.input.v('tolerance'))
        # Extract solution
        Z1, Z2 = solution(self.x, eval_derivative=True)
        # Surface elevation and its derivatives wrt x
        zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
        zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
        zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2] * (1. - v['a'][:, 0] * np.cosh(v['r'][:, 0] * self.zarr)) +
                         G1 /
                         (self.Av0 * v['r'][:, 0]) - (v['a'][:, 0] * G2 / (self.Av0 * v['r'][:, 0] *
                                                                           v['Roughness'][:, 0])) *
                         (self.Av0 * v['r'][:, 0] * np.cosh(v['r'][:, 0] * (self.zarr + self.H)) +
                          v['Roughness'][:, 0] * np.sinh(v['r'][:, 0] * (self.zarr + self.H)))).T

        # plot results
        # self.plot2(zeta, u)
        return zeta, u

    def baroc(self):
        """Calculates the first order contribution due to the salinity field.
        """
        ## Define and initiate variables ##
        self.__F = 'baroc'
        # Extract parameters alpha, r, H, B, Av, and roughness as a function of the x-coordinate
        v = self.variables(self.x)
        # Initiate variables zeta and u
        zeta = np.zeros((2, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        ### M0 contribution ###
        # Calculate M0 water level
        zeta[1, :, 0, 0] = -(self.BETA * self.input.d('s0', x=self.x/self.L, z=0, f=0, dim='x') * (self.H**4. / (8. * self.Av0) +
                            self.H**3. / (2. * v['Roughness'][:, 0])) / (self.H**3. / (3. * self.Av0) +
                            self.H**2. / v['Roughness'][:, 0]))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], dx=self.dx, axis=0)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = ((0.5 * self.zarr**2. / self.Av0 - 0.5 * self.H**2 / self.Av0 -
                          self.H / v['Roughness'][:, 0]) * self.G * zeta[1, :, 0, 0] -
                         self.G * self.BETA * self.input.d('s0', x=self.x/self.L, z=0, f=0, dim='x') *
                         (self.zarr**3. / (6. * self.Av0) + self.H**3. / (6. * self.Av0) + self.H**2. /
                          (2. * v['Roughness'][:, 0]))).T
        # self.plot2(zeta, u)
        return zeta, u

    def river(self):
        """Calculates the first order contribution due to the river flow.
        """
        ## Define and initiate variables ##
        self.__F = 'river'
        # Extract parameters alpha, r, H, B, Av, and roughness as a function of the x-coordinate
        v = self.variables(self.x)
        # Initiate variables zeta and u
        zeta = np.zeros((2, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        ### M0 contribution ###
        # Calculate M0 water level
        zeta[1, :, 0, 0] = (self.input.v('Q1') / (self.G * self.B * (self.H**3 / (3*self.Av0) +
                                                                     self.H**2 / v['Roughness'][:, 0])))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], dx=self.dx, axis=0)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = ((0.5 * self.zarr**2. / self.Av0 - 0.5 * self.H**2. / self.Av0 -
                          self.H / v['Roughness'][:, 0]) * self.G * zeta[1, :, 0, 0]).T

        # self.plot2(zeta, u)
        return zeta, u

    def plot(self):
        f, ax = plt.subplots(2,1)
        p1 = ax[0].pcolormesh(np.real(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x').T))
        p2 = ax[1].pcolormesh(np.imag(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x').T))
        plt.colorbar(p1,ax=ax[0])
        plt.colorbar(p2,ax=ax[1])
        f.suptitle(r'iflow $u_x$')

        f,ax = plt.subplots(2,1)
        p1 = ax[0].pcolormesh(np.real(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z').T))
        p2 = ax[1].pcolormesh(np.imag(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z').T))
        plt.colorbar(p1,ax=ax[0])
        plt.colorbar(p2,ax=ax[1])
        f.suptitle(r'iflow $u_z$')

        f,ax = plt.subplots(2,1)
        p1 = ax[0].pcolormesh(np.real(self.input.dd('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z').T))
        p2 = ax[1].pcolormesh(np.imag(self.input.dd('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z').T))
        plt.colorbar(p1, ax=ax[0])
        plt.colorbar(p2, ax=ax[1])
        f.suptitle(r'iflow $u_zz$')

    def plot2(self, zeta, u):
        f, ax = plt.subplots(1, 2)
        ax[0].plot(np.real(zeta[0, :, 0, 0]), 'b')
        ax[0].plot(np.abs(zeta[0, :, 0, 2]), 'r')
        ax[0].set_title(r'surface elevation')

        ax[1].plot(self.x/1e3, self.input.v('Roughness', x=self.x/self.L)*np.real(u[0, :, -1, 0].T), 'b')
        # ax[1].plot(np.abs(u[0, :, -1, 2].T), 'r')
        ax[1].set_title('hor. velocity')

        f.suptitle(self.__F)
        plt.show()

    def plot3(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        subList = ['tide', 'river', 'nostress', 'adv', 'baroc', 'stokes']
        # ZETA1 = np.abs(self.input.v('zeta1', 'tide', x=self.x/self.L, z=0, f=2) +
        #          self.input.v('zeta1', 'nostress', x=self.x/self.L, z=0, f=2) +
        #          self.input.v('zeta1', 'stokes', x=self.x/self.L, z=0, f=2) +
        #          self.input.v('zeta1', 'adv', x=self.x/self.L, z=0, f=2))
        dpi = 75
        f, ax = plt.subplots(3, 2, num=1, figsize=(900/dpi, 900/dpi), dpi=dpi, facecolor='w', edgecolor='k')
        plt.hold(True)
        for i, mod in enumerate(subList):
            pos = np.unravel_index(i, (3, 2))
            for n in range(0, 3):
                try:
                    if self.input.v('zeta0', mod) is not None:
                        wl = self.input.v('zeta0', mod, range(0, jmax+1), 0, n)+self.input.v('zeta1', mod, range(0, jmax+1), 0, n)
                    else:
                        wl = self.input.v('zeta1', mod, range(0, jmax+1), 0, n)
                    ax[pos].plot(self.x/1e3, abs(wl), label='M'+str(2*n))
                    ax[pos].set_title(mod)
                    ax[pos].set_xlabel(r'$x$ [km]')
                    ax[pos].set_ylabel(r'elevation amplitude [m]')
                except:
                    pass
        ax[0,0].legend(loc=0, fontsize=8, frameon=False)

        f.set_tight_layout(True)
        plt.show()


