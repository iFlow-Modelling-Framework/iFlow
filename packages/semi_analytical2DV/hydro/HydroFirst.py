"""

Date: 01-06-15
Authors: R.L. Brouwer
"""
import numpy as np
from scipy import integrate
from nifty.functionTemplates.NumericalFunctionWrapper import NumericalFunctionWrapper
import logging
import nifty as ny
from zetaFunctionUncoupled import zetaFunctionUncoupled
from src.util.diagnostics import KnownError


class HydroFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the first order water level and velocities

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        self.logger.info('Running module HydroFirst')

        # Initiate variables
        self.submodule = self.input.v('submodules')
        self.TOL = self.input.v('TOLERANCEBVP')
        self.SIGMA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.BETA = self.input.v('BETA')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        self.dx = self.x[1:]-self.x[:-1]
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = (self.z.reshape(1, len(self.z)) * self.input.n('H', x=self.x/self.L).reshape(len(self.x), 1))
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=0).reshape(len(self.x), 1)
        self.Av0x = self.input.d('Av', x=self.x/self.L, z=0, f=0, dim='x').reshape(len(self.x), 1)
        self.sf = self.input.v('Roughness', x=self.x/self.L, f=0).reshape(len(self.x), 1)
        self.sfx = self.input.d('Roughness', x=self.x/self.L, f=0, dim='x').reshape(len(self.x), 1)
        self.r = np.sqrt(2. * 1j * self.SIGMA / self.Av0).reshape(len(self.x), 1)
        self.H = self.input.v('H', x=self.x/self.L).reshape(len(self.x), 1)
        self.Hx = self.input.d('H', x=self.x/self.L, dim='x').reshape(len(self.x), 1)
        self.alpha = (self.sf / (self.r * self.Av0 * np.sinh(self.r * self.H) + self.sf * np.cosh(self.r * self.H))).reshape(len(self.x), 1)
        self.B = self.input.v('B', x=self.x/self.L).reshape(len(self.x), 1)
        self.Bx = self.input.d('B', x=self.x/self.L, dim='x').reshape(len(self.x), 1).reshape(len(self.x), 1)
        self.u0 = self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1)
        self.u0x = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='x')
        self.u0z = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z')
        self.u0zz = self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='zz')
        self.M = ((self.alpha * np.sinh(self.r * self.H) / self.r) - self.H) * (self.G / (2 * 1j * self.SIGMA)) * self.B
        self.bca = ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (3,))[2]
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
            nfu = NumericalFunctionWrapper(u[0], self.input.slice('grid'))
            d['zeta1'][mod] = nfz.function
            d['u1'][mod] = nfu.function

        self.input.merge(d)
        return d

    def rf(self, x):
        """Calculate the root r = \sqrt(2i\sigma / Av) of the characteristic equation and its derivatives wrt x.

        Parameters:
            x - x-coordinate

        Returns:
            r - root of the characteristic equation of the first order horizontal velocity
        """
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='xx')])

        r = np.zeros((3, np.array(x).size), dtype=complex)
        r[0] = np.sqrt(2. * 1j * self.SIGMA / Av[0])
        r[1] = -np.sqrt(2. * 1j * self.SIGMA) * Av[1] / (2. * Av[0]**(3./2.))
        r[2] = np.sqrt(2. * 1j * self.SIGMA) * (3. * Av[1]**2. - 2. * Av[0] * Av[2]) / (4. * Av[0]**(5./2.))
        return r

    def af(self, x, r):
        """Calculate the coefficient alpha that appears in the solution for the leading order horizontal velocity.

        Parameters:
            x - x-coordinate

        Returns:
            a - coefficient alpha
        """
        H = np.array([self.input.v('H', x=x/self.L),
                      self.input.d('H', x=x/self.L, dim='x'),
                      self.input.d('H', x=x/self.L, dim='xx')])
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='xx')])
        sf = np.array([self.input.v('Roughness', x=x/self.L, f=0),
                       self.input.d('Roughness', x=x/self.L, f=0, dim='x'),
                       self.input.d('Roughness', x=x/self.L, f=0, dim='xx')])
        sf = sf[:, 0]
        # Define trigonometric values for ease of reference
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        cothrh = coshrh / sinhrh
        cschrh = 1 / sinhrh
        # Define parameters and their (second) derivative wrt x
        E = r[1] * H[0] + r[0] * H[1]
        Ex = r[2] * H[0] + 2 * r[1] * H[1] + r[0] * H[2]
        F = r[1] + r[0] * E * cothrh
        Fx = r[2] + r[0] * Ex * cothrh + E * (r[1] * cothrh - r[0] * E**2 * cschrh**2)
        K = r[0] * Av[1] + Av[0] * F + sf[1] * cothrh + sf[0] * E
        Kx = (r[0] * Av[2] + r[1] * Av[1] + Av[1] * F + Av[0] * Fx + sf[2] * cothrh -
              sf[1] * E * cschrh**2 + sf[1] * E + sf[0] * Ex)
        G = r[0] * Av[0] * sinhrh + sf[0] * coshrh
        Gx = sinhrh * K
        Gxx = E * K * coshrh + Kx * sinhrh
        # Calculate coefficient alpha
        a = np.zeros((3, np.array(x).size), dtype=complex)
        a[0] = sf[0] / G    # a
        a[1] = sf[1] / G - sf[0] * Gx / G**2    # a_x
        a[2] = (sf[2] - 2 * (sf[1] * Gx + sf[0] * Gxx) / G + 2 * sf[0] * Gx**2 / G**2) / G    # a_xx
        return a

    def bcs(self, Za, Zb):
        """Defines the boundary conditions for the barotropic forcing at x = 0 and x = 1

        Parameters:
            Za - water level at the seaward side zeta(0)
            Zb - water level gradient at the landward side dzeta(1)/dx

        Returns:
            bca = boundary condition at the seaward side
            bcb = boundary condition at the landward side
        """
        bca = np.zeros(2)
        bcb = np.zeros(2)
        bca[0] = Za[0] + self.__bc[0]  # Real valued boundary condition at x = 0
        bca[1] = Za[1] + self.__bc[1]  # Imaginary valued boundary condition at x = 0
        bcb[0] = Zb[2] + self.__bc[2]  # Real valued boundary condition at x = 1
        bcb[1] = Zb[3] + self.__bc[3]  # Imaginary valued boundary condition at x = 1
        return (bca), (bcb)

    def system_ode(self, x, Z):
        """Callback function which evaluates the ODE.

        Parameters:
            x - x-coordinate (independent variable)
            Z - dependent variable appearing in the ODE. In this case zeta and zeta_x. See manual for more information.

        Returns:
            dzdx - numpy array that contains the value of the first derivative of each dependent variable, i.e.
                   zeta_x and zeta_xx. See manual for more information
        """
        # Import system variables
        H = np.array([self.input.v('H', x=x/self.L),
                      self.input.d('H', x=x/self.L, dim='x'),
                      self.input.d('H', x=x/self.L, dim='xx')])
        B = np.array([self.input.v('B', x=x/self.L),
                      self.input.d('B', x=x/self.L, dim='x'),
                      self.input.d('B', x=x/self.L, dim='xx')])
        r = self.rf(x)
        a = self.af(x, r)
        # Define local parameters
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        # Factors in ODE
        T1 = ((a[0] * sinhrh / r[0]) - H[0])
        T2 = (B[1] * T1 / B[0] + H[1] * (a[0] * coshrh - 1.) + a[1] * sinhrh / r[0] +
              a[0] * r[1] * (r[0] * H[0] * coshrh - sinhrh) / r[0] ** 2)
        T3 = 4 * self.SIGMA**2 / self.G
        if self.__F is 'tide':
            T4 = 0
        elif self.__F is 'stokes':
            T4 = -4. * 1j * self.SIGMA * (self.input.v('gamma_x', x=x/self.L, z=0, f=2) +
                                          B[1] * self.input.v('gamma', x=x/self.L, z=0, f=2) / B[0]) / self.G
        elif self.__F is 'nostress':
            FnsB = (B[1]/B[0]) * (self.input.v('xi', x=x/self.L, z=0, f=2) * (1 - a[0]) / r[0]**2.)
            Fnsdx = ((self.input.v('xi_x', x=x/self.L, z=0, f=2) * r[0] * (1 - a[0]) -
                      self.input.v('xi', x=x/self.L, z=0, f=2) * (r[0] * a[1] + 2. * r[1] * (1 - a[0]))) / r[0]**3.)
            T4 = 4. * 1j * self.SIGMA * (FnsB + Fnsdx) / self.G
        elif self.__F is 'adv':
            T4 = self.input.v('Fadv', x=x/self.L)

        # Definition of ODE
        dzdx = np.zeros(4)
        dzdx[0] = Z[2]
        dzdx[1] = Z[3]
        dzdx[2] = (-np.real(T2 / T1) * Z[2] + np.real(T3 / T1) * Z[0] +
                    np.imag(T2 / T1) * Z[3] - np.imag(T3 / T1) * Z[1] + np.real(T4 / T1))
        dzdx[3] = (-np.imag(T2 / T1) * Z[2] + np.imag(T3 / T1) * Z[0] +
                   -np.real(T2 / T1) * Z[3] + np.real(T3 / T1) * Z[1] + np.imag(T4 / T1))
        return dzdx

    def system_ode_der(self, x, Z):
        """Callback function which evaluates the partial derivatives of the dependent variables in the ODE.

        Parameters:
            x - x-coordinate (independent variable)
            Z - dependent variable appearing in the ODE. In this case zeta and zeta_x. See manual for more information.

        Returns:
            dydz - numpy array that contains the value of the first derivative of each dependent variable. See manual
                   for more information
        """
        # Import system variables
        H = np.array([self.input.v('H', x=x / self.L),
                      self.input.d('H', x=x / self.L, dim='x'),
                      self.input.d('H', x=x / self.L, dim='xx')])
        B = np.array([self.input.v('B', x=x / self.L),
                      self.input.d('B', x=x / self.L, dim='x'),
                      self.input.d('B', x=x / self.L, dim='xx')])
        r = self.rf(x)
        a = self.af(x, r)
        # Define local parameters
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        # Factors in ODE
        T1 = ((a[0] * sinhrh / r[0]) - H[0])
        T2 = (B[1] * T1 / B[0] + H[1] * (a[0] * coshrh - 1.) + a[1] * sinhrh / r[0] +
              a[0] * r[1] * (r[0] * H[0] * coshrh - sinhrh) / r[0] ** 2)
        T3 = 4 * self.SIGMA ** 2 / self.G
        if self.__F is 'tide':
            T4 = 0
        elif self.__F is 'stokes':
            T4 = -4. * 1j * self.SIGMA * (self.input.v('gamma_x', x=x / self.L, z=0, f=2) +
                                          B[1] * self.input.v('gamma', x=x / self.L, z=0, f=2) / B[0]) / self.G
        elif self.__F is 'nostress':
            FnsB = (B[1] / B[0]) * (self.input.v('xi', x=x / self.L, z=0, f=2) * (1 - a[0]) / r[0] ** 2.)
            Fnsdx = ((self.input.v('xi_x', x=x / self.L, z=0, f=2) * r[0] * (1 - a[0]) -
                      self.input.v('xi', x=x / self.L, z=0, f=2) * (r[0] * a[1] + 2. * r[1] * (1 - a[0]))) / r[0] ** 3.)
            T4 = 4. * 1j * self.SIGMA * (FnsB + Fnsdx) / self.G
        elif self.__F is 'adv':
            T4 = self.input.v('Fadv', x=x / self.L)

        # Definition of ODE
        dydz = np.zeros((4, 4))
        dydz[0, 2] = 1.
        dydz[1, 3] = 1.
        dydz[2, :] = [np.real(T3 / T1), -np.imag(T3 / T1), -np.real(T2 / T1),  np.imag(T2 / T1)]
        dydz[3, :] = [np.imag(T3 / T1),  np.real(T3 / T1), -np.imag(T2 / T1), -np.real(T2 / T1)]
        return dydz

    def tide(self):
        """Calculates the first order contribution due to the external tide. This contribution only has an M4-component

        Returns:
            zeta - M4 water level due to the external tide
            u    - M4 horizontal velocity due to the external tide
        """
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # Calculate M4 contribution
        if self.input.v('solver') == (None or 'numerical'):
            jmax = self.input.v('grid', 'maxIndex', 'x')
            F = np.zeros((jmax + 1, 1), dtype=complex)  # Forcing term shape (x, number of right-hand sides)
            Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fopen[0, 0] = self.bca
            Z, Zx, _ = zetaFunctionUncoupled(2, self.M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False)
            zeta[0, :, 0, 2] = Z[:, 0]
            zeta[1, :, 0, 2] = Zx[:, 0]
        elif self.input.v('solver') == 'bvp':
            try:
                import scikits.bvp_solver
            except ImportError:
                raise KnownError('The scikits.bvp_solver module is not known')
            # Define and initiate variables #
            self.__F = 'tide'
            # Define boundary conditions
            self.__bc = np.array([-np.real(self.bca), -np.imag(self.bca), 0, 0])
            # Define problem to solve the ode for the water level
            problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                           num_parameters=0,
                                                           num_left_boundary_conditions=2,
                                                           boundary_points=(self.x[0], self.x[-1]),
                                                           function=self.system_ode,
                                                           boundary_conditions=self.bcs,
                                                           function_derivative=self.system_ode_der)
            guess = np.array([np.real(self.bca), np.imag(self.bca), 0., 0.])
            solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.TOL)
            # Extract solution
            Z1, Z2 = solution(self.x, eval_derivative=True)
            # Surface elevation and its derivatives wrt x
            zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
            zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
            zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
            zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        # Calculate the velocity
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2].reshape(len(self.x), 1) *
                         (1 - self.alpha * np.cosh(self.r * self.zarr)))
        return zeta, u

    def stokes(self):
        """Calculates the first order contribution due to the Stokes return flow.

        Returns:
            zeta - residual and M4 water level due to the Stokes return flow
            u    - M4 horizontal velocity due to the Stokes return flow
        """
        # Calculate and save forcing terms for the Stokes contribution
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
        # Define and initiate variables #
        self.__F = 'stokes'
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # M0 contribution #
        # Calculate M0 water level
        zeta[1, :, 0, 0] = (gammaM0.reshape(len(self.x), 1) /
                            (self.G * self.H**2. * (self.H / (3. * self.Av0) + 1. / self.sf))).reshape(len(self.x))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], x=self.x)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = (((self.zarr**2. - self.H**2.) / (2 * self.Av0) - self.H / self.sf) * self.G * zeta[1, :, 0, 0].reshape(len(self.x), 1))

        # M4 contribution #
        if self.input.v('solver') == (None or 'numerical'):
            F = -2. * (self.B * gammaM4x.reshape(len(self.x), 1) + self.Bx * gammaM4.reshape(len(self.x), 1))
            Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fclosed[0] = (-2. * gammaM4[-1] / (self.alpha[-1, 0] * np.sinh(self.r[-1, 0] * self.H[-1]) / self.r[-1, 0] - self.H[-1]))
            Z, Zx, _ = zetaFunctionUncoupled(2, self.M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False)
            zeta[0, :, 0, 2] = Z[:, 0]
            zeta[1, :, 0, 2] = Zx[:, 0]
        elif self.input.v('solver') == 'bvp':
            try:
                import scikits.bvp_solver
            except ImportError:
                raise KnownError('The scikits.bvp_solver module is not known')
            # Define boundary conditions
            self.__bc = np.array([0., 0.,
                                  np.real(4. * 1j * self.SIGMA * gammaM4[-1] /
                                          (self.G * (self.alpha[-1, 0] * np.sinh(self.r[-1, 0] * self.H[-1]) /
                                           self.r[-1, 0] - self.H[-1]))),
                                  np.imag(4. * 1j * self.SIGMA * gammaM4[-1] /
                                          (self.G * (self.alpha[-1, 0] * np.sinh(self.r[-1, 0] * self.H[-1]) /
                                           self.r[-1, 0] - self.H[-1])))])
            # Define problem to solve the ode for the water level
            problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                           num_parameters=0,
                                                           num_left_boundary_conditions=2,
                                                           boundary_points=(self.x[0], self.x[-1]),
                                                           function=self.system_ode,
                                                           boundary_conditions=self.bcs,
                                                           function_derivative=self.system_ode_der)
            guess = np.array([0., 0., 0., 0.])
            solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.TOL)
            # Extract solution
            Z1, Z2 = solution(self.x, eval_derivative=True)
            # Surface elevation and its derivatives wrt x
            zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
            zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
            zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        # Calculate the velocity
        u[0, :, :, 2] = -((self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2].reshape(len(self.x), 1) *
                          (1 - self.alpha * np.cosh(self.r * self.zarr)))
        return zeta, u

    def nostress(self):
        """Calculates the first order contribution due to the no-stress boundary condition.

        Returns:
            zeta - residual and M4 water level due to the no-stress boundary condition.
            u    - residual and M4 horizontal velocity due to the no-stress boundary condition.
        """
        # Calculate and save forcing terms for the no-stress contribution #
        xiM0 = 0.5 * np.real(self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                             np.conj(self.input.dd('u0', 'tide', range(0, len(self.x)), 0, 1, dim='z')))
        xiM4 = 0.25 * (self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                       self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='zz'))
        xiM4x = 0.25 * (self.input.d('zeta0', 'tide', range(0, len(self.x)), 0, 1, dim='x') *
                        self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='zz') +
                        self.input.v('zeta0', 'tide', range(0, len(self.x)), 0, 1) *
                        self.input.d('u0', 'tide', range(0, len(self.x)), 0, 1, dim='zzx'))
        xi = np.zeros((len(xiM0), 1, 3), dtype=complex)
        xi[:, 0, 0] = xiM0
        xi[:, 0, 2] = xiM4
        xi_x = np.zeros((len(xiM4x), 1, 3), dtype=complex)
        xi_x[:, 0, 2] = xiM4x
        self.input.addData('xi', xi)
        self.input.addData('xi_x', xi_x)
        # Define and initiate variables #
        self.__F = 'nostress'
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # M0 contribution #
        # Calculate M0 water level
        zeta[1, :, 0, 0] = (-(self.Av0 * xiM0.reshape(len(self.x), 1) * (self.H / 2. + self.Av0 / self.sf)) /
                            (self.G * self.H * (self.H / 3. + self.Av0 / self.sf))).reshape(len(self.x))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], x=self.x)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = (((self.zarr ** 2 - self.H ** 2) / (2 * self.Av0) - self.H / self.sf) * self.G *
                         zeta[1, :, 0, 0].reshape(len(self.x), 1) -
                         (self.zarr + self.H + self.Av0 / self.sf) * xiM0.reshape(len(self.x), 1))

        # M4 contribution #
        if self.input.v('solver') == (None or 'numerical'):
            FnsB = (self.Bx / self.B) * (xiM4.reshape(len(self.x), 1) * (1 - self.alpha) / self.r ** 2.)
            Fnsdx = ((xiM4x.reshape(len(self.x), 1) * self.r * (1 - self.alpha) -
                      xiM4.reshape(len(self.x), 1) * (self.r * self.af(self.x, self.rf(self.x))[1].reshape(len(self.x), 1) +
                                                      2. * self.rf(self.x)[1].reshape(len(self.x), 1) * (1 - self.alpha))) / self.r ** 3.)
            F = 2. * self.B * (FnsB + Fnsdx)
            Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Z, Zx, _ = zetaFunctionUncoupled(2, self.M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False)
            zeta[0, :, 0, 2] = Z[:, 0]
            zeta[1, :, 0, 2] = Zx[:, 0]
        elif self.input.v('solver') == 'bvp':
            try:
                import scikits.bvp_solver
            except ImportError:
                raise KnownError('The scikits.bvp_solver module is not known')

            # Define boundary conditions
            self.__bc = np.array([0., 0., 0., 0.])
            # Define problem to solve the ode for the water level
            problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                           num_parameters=0,
                                                           num_left_boundary_conditions=2,
                                                           boundary_points=(self.x[0], self.x[-1]),
                                                           function=self.system_ode,
                                                           boundary_conditions=self.bcs,
                                                           function_derivative=self.system_ode_der)
            guess = np.array([0., 0., 0., 0.])
            solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.TOL)
            # Extract solution
            Z1, Z2 = solution(self.x, eval_derivative=True)
            # Surface elevation and its derivatives wrt x
            zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
            zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
            zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        u[0, :, :, 2] = (-(self.G / (2 * 1j * self.SIGMA)) * zeta[1, :, 0, 2].reshape(len(self.x), 1) *
                         (1 - self.alpha * np.cosh(self.r * self.zarr)) - 2 * self.alpha *
                         xiM4.reshape(len(self.x), 1) * (self.Av0 * np.cosh(self.r * (self.zarr + self.H)) + self.sf *
                                                         np.sinh(self.r * (self.zarr + self.H)) / self.r) / self.sf)
        return zeta, u

    def adv(self):
        """Calculates the first order contribution due to the advection of momentum.
        """
        # Calculate and save forcing terms for the advection of momentum contribution
        etaM0 = 0.5 * np.real(self.input.v('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                              np.conj(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x')) +
                              self.input.v('w0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                              np.conj(self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z')))
        etaM4 = 0.25 * (self.input.v('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                        self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='x') +
                        self.input.v('w0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1) *
                        self.input.d('u0', 'tide', range(0, len(self.x)), range(0, len(self.z)), 1, dim='z'))
        eta = np.zeros((etaM0.shape[0], etaM0.shape[1], 3), dtype=complex)
        eta[:, :, 0] = etaM0[:]
        eta[:, :, 2] = etaM4[:]
        self.input.addData('eta', eta)
        # Define and initiate variables #
        self.__F = 'adv'
        # Initiate variables zeta and u
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # M0 contribution #
        # Particular solution Up of the advection term
        f = np.fliplr(integrate.cumtrapz(np.fliplr(etaM0 * self.zarr), x=-self.zarr, axis=1, initial=0))
        g = np.fliplr(integrate.cumtrapz(np.fliplr(etaM0), x=-self.zarr, axis=1, initial=0)) * self.zarr
        Up = g - f
        # Vertical integral of the advection term
        h = np.trapz(etaM0, x=-self.zarr, axis=1)
        # Vertical integral of the particular solution
        k = np.trapz(Up, x=-self.zarr, axis=1)
        # Calculate M0 water level
        zeta[1, :, 0, 0] = ((k.reshape(len(self.x), 1) / self.H - (self.H / 2. + self.Av0 / self.sf) * h.reshape(len(self.x), 1)) /
                            (self.G * self.H * (self.H / 3. + self.Av0 / self.sf))).reshape(len(self.x))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], x=self.x)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = (((self.zarr**2. - self.H**2.) / (2 * self.Av0) - self.H / self.sf) * self.G * zeta[1, :, 0, 0].reshape(len(self.x), 1) +
                         (Up / self.Av0) - (self.zarr / self.Av0 + self.H / self.Av0 + 1. / self.sf) * h.reshape(len(self.x), 1))

        # M4 contribution #
        # Calculate variables
        # G1 variable from manual, separated in an (a) and a (b) part first
        G1a = np.fliplr(integrate.cumtrapz(np.fliplr(etaM4 * np.exp(-self.r * self.zarr)),
                                           x=-self.zarr, axis=1, initial=0))
        # G1a = np.append(G1a, np.zeros((len(self.x), 1)), axis=1)
        G1b = np.fliplr(integrate.cumtrapz(np.fliplr(etaM4 * np.exp(self.r * self.zarr)),
                                           x=-self.zarr, axis=1, initial=0))
        # G1b = np.append(G1b, np.zeros((len(self.x), 1)), axis=1)
        G1 = (np.exp(self.r * self.zarr) * G1a -
              np.exp(-self.r * self.zarr) * G1b)
        # G2 variable from manual
        G2 = (np.trapz(etaM4 * np.exp(-self.r * self.zarr), x=-self.zarr, axis=1) +
              np.trapz(etaM4 * np.exp(self.r * self.zarr), x=-self.zarr, axis=1))

        # calculating the advective forcing term for the water level and save it to the data container
        Fadv1 = (G2.reshape(len(self.x), 1) * (1. - self.alpha) / (self.Av0 * self.r**2.) -
                 np.trapz(G1, x=-self.zarr, axis=1).reshape(len(self.x), 1) / (self.Av0 * self.r))
        Fadv2 = np.zeros((len(self.x), 1), dtype=complex)
        Fadv2[0, 0] = (-3. * Fadv1[0, 0] / 2. + 2. * Fadv1[1, 0] - Fadv1[2, 0] / 2.) / self.dx[0]
        Fadv2[-1, 0] = (3. * Fadv1[-1, 0] / 2. - 2. * Fadv1[-2, 0] + Fadv1[-3, 0] / 2.) / self.dx[-1]
        Fadv2[1:-1, 0] = (Fadv1[2:, 0] - Fadv1[:-2, 0]) / (2. * self.dx[1:])
        Fadv = ((self.Av0 * self.r**2. / self.G) * (Fadv2 + self.Bx * Fadv1 / self.B)).reshape(len(self.x))
        self.input.addData('Fadv', Fadv)
        if self.input.v('solver') == (None or 'numerical'):
            F = -self.B * Fadv.reshape(len(self.x), 1) * (self.G / (2 * 1j * self.SIGMA))
            Fopen = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Fclosed = np.zeros((1, 1), dtype=complex)  # Forcing term shape (1, number of right-hand sides)
            Z, Zx, _ = zetaFunctionUncoupled(2, self.M[:, 0], F, Fopen, Fclosed, self.input, hasMatrix=False)
            zeta[0, :, 0, 2] = Z[:, 0]
            zeta[1, :, 0, 2] = Zx[:, 0]
        elif self.input.v('solver') == 'bvp':
            try:
                import scikits.bvp_solver
            except ImportError:
                raise KnownError('The scikits.bvp_solver module is not known')

            # Define boundary conditions
            self.__bc = np.array([0., 0., 0., 0.])
            # Define problem to solve the ode for the water level
            problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                           num_parameters=0,
                                                           num_left_boundary_conditions=2,
                                                           boundary_points=(self.x[0], self.x[-1]),
                                                           function=self.system_ode,
                                                           boundary_conditions=self.bcs,
                                                           function_derivative=self.system_ode_der)
            guess = np.array([0., 0., 0., 0.])
            solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.TOL)
            # Extract solution
            Z1, Z2 = solution(self.x, eval_derivative=True)
            # Surface elevation and its derivatives wrt x
            zeta[0, :, 0, 2] = Z1[0] + 1j * Z1[1]
            zeta[1, :, 0, 2] = Z1[2] + 1j * Z1[3]
            zeta[2, :, 0, 2] = Z2[2] + 1j * Z2[3]
        u[0, :, :, 2] = (-(self.G / (2. * 1j * self.SIGMA)) * zeta[1, :, 0, 2].reshape(len(self.x), 1) * (1. - self.alpha * np.cosh(self.r * self.zarr)) +
                         G1 /
                         (self.Av0 * self.r) - (self.alpha * G2.reshape(len(self.x), 1) / (self.Av0 * self.r *
                                                                           self.sf)) *
                         (self.Av0 * self.r * np.cosh(self.r * (self.zarr + self.H)) +
                          self.sf * np.sinh(self.r * (self.zarr + self.H))))
        return zeta, u

    def baroc(self):
        """Calculates the first order contribution due to the salinity field.
        """
        # Define and initiate variables #
        self.__F = 'baroc'
        # Initiate variables zeta and u
        zeta = np.zeros((2, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        sx = self.input.d('s0', x=self.x / self.L, z=0, f=0, dim='x').reshape(len(self.x), 1)
        # M0 contribution #
        # Calculate M0 water level
        zeta[1, :, 0, 0] = -(self.H * self.BETA * sx * (self.H / (8. * self.Av0) + 1. / (2. * self.sf)) /
                             (self.H / (3. * self.Av0) + 1. / self.sf)).reshape(len(self.x))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], x=self.x)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = (((self.zarr ** 2. - self.H ** 2.) / (2 * self.Av0) - self.H / self.sf) * self.G * zeta[1, :, 0, 0].reshape(len(self.x), 1) -
                         ((self.zarr ** 3. + self.H ** 3.) / (6. * self.Av0) + self.H ** 2. / (2. * self.sf)) *
                         self.G * self.BETA * sx)
        return zeta, u

    def river(self):
        """Calculates the first order contribution due to the river flow.
        """
        # Define and initiate variables #
        self.__F = 'river'
        # Initiate variables zeta and u
        zeta = np.zeros((2, len(self.x), 1, 3), dtype=complex)
        u = np.zeros((1, len(self.x), len(self.z), 3), dtype=complex)

        # M0 contribution #
        # Calculate M0 water level
        zeta[1, :, 0, 0] = (self.input.v('Q1') / (self.G * self.B * self.H**2. * (self.H / (3. * self.Av0) + 1. / self.sf))).reshape(len(self.x))
        zeta[0, 1:, 0, 0] = integrate.cumtrapz(zeta[1, :, 0, 0], x=self.x)
        # Calculate M0 flow velocity
        u[0, :, :, 0] = (((self.zarr**2. - self.H**2.) / (2 * self.Av0) - self.H / self.sf) * self.G * zeta[1, :, 0, 0].reshape(len(self.x), 1))
        return zeta, u