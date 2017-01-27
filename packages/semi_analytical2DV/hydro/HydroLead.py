"""
Date: 01-06-15
Authors: R.L. Brouwer
"""

import logging
import numpy as np
from nifty.functionTemplates.NumericalFunctionWrapper import NumericalFunctionWrapper
import nifty as ny
from zetaFunctionUncoupled import zetaFunctionUncoupled
from src.util.diagnostics import KnownError


class HydroLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the leading order water level and velocities

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        self.logger.info('Running module HydroLead')

        # Initiate variables
        self.TOL = self.input.v('TOLERANCEBVP')
        self.SIGMA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = (self.z.reshape(1, len(self.z)) * self.input.n('H', x=self.x/self.L).reshape(len(self.x), 1))
        self.bca = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (2,))[1]
       # Allocate space for results
        d = dict()
        d['zeta0'] = {}
        d['u0'] = {}
        d['w0'] = {}
        # Save water level results
        zeta = self.waterlevel()
        nf = NumericalFunctionWrapper(zeta[0], self.input.slice('grid'))
        nf.addDerivative(zeta[1], 'x')
        nf.addDerivative(zeta[2], 'xx')
        d['zeta0']['tide'] = nf.function
        # Save velocity results
        u, w = self.velocity(zeta)
        nfu = NumericalFunctionWrapper(u[0], self.input.slice('grid'))
        nfu.addDerivative(u[1], 'x')
        nfu.addDerivative(u[2], 'z')
        nfu.addDerivative(u[3], 'zz')
        nfu.addDerivative(u[4], 'zzx')
        d['u0']['tide'] = nfu.function

        nfw = NumericalFunctionWrapper(w[0], self.input.slice('grid'))
        nfw.addDerivative(w[1], 'z')
        d['w0']['tide'] = nfw.function
        return d

    def rf(self, x):
        """Calculate the root r = \sqrt(i\sigma / Av) of the characteristic equation and its derivatives wrt x.

        Parameters:
            x - x-coordinate

        Returns:
            r - root of the characteristic equation of the leading order horizontal velocity
        """
        Av = np.array([self.input.v('Av', x=x/self.L, z=0, f=0),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='x'),
                       self.input.d('Av', x=x/self.L, z=0, f=0, dim='xx')])

        r = np.zeros((3, np.array(x).size), dtype=complex)
        r[0] = np.sqrt(1j * self.SIGMA / Av[0])
        r[1] = -np.sqrt(1j * self.SIGMA) * Av[1] / (2. * Av[0]**(3./2.))
        r[2] = np.sqrt(1j * self.SIGMA) * (3. * Av[1]**2 - 2. * Av[0] * Av[2]) / (4. * Av[0]**(5./2.))
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
        Ex = r[2] * H[0] + 2. * r[1] * H[1] + r[0] * H[2]
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
        a[2] = (sf[2] - 2. * (sf[1] * Gx + sf[0] * Gxx) / G + 2. * sf[0] * Gx**2 / G**2) / G    # a_xx
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
        bca[0] = Za[0] - np.real(self.bca)  # Real valued boundary condition at x = 0
        bca[1] = Za[1] - np.imag(self.bca)  # Imaginary valued boundary condition at x = 0
        bcb[0] = Zb[2]  # Real valued boundary condition at x = 1
        bcb[1] = Zb[3]  # Imaginary valued boundary condition at x = 1
        return bca, bcb

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
                      self.input.d('H', x=x/self.L, dim='x')])
        B = np.array([self.input.v('B', x=x/self.L),
                      self.input.d('B', x=x/self.L, dim='x')])
        r = self.rf(x)
        a = self.af(x, r)
        # Define local parameters
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        # Factors in ODE
        T1 = ((a[0] * sinhrh / r[0]) - H[0])
        T2 = (B[1] * T1 / B[0] + H[1] * (a[0] * coshrh - 1.) + a[1] * sinhrh / r[0] +
              a[0] * r[1] * (r[0] * H[0] * coshrh - sinhrh) / r[0] ** 2)
        T3 = self.SIGMA**2 / self.G

        # Definition of ODE
        dzdx = np.zeros(4)
        dzdx[0] = Z[2]
        dzdx[1] = Z[3]
        dzdx[2] = (-np.real(T2 / T1) * Z[2] + np.real(T3 / T1) * Z[0] +
                    np.imag(T2 / T1) * Z[3] - np.imag(T3 / T1) * Z[1])
        dzdx[3] = (-np.imag(T2 / T1) * Z[2] + np.imag(T3 / T1) * Z[0] +
                   -np.real(T2 / T1) * Z[3] + np.real(T3 / T1) * Z[1])
        return dzdx

    def system_ode_der(self, x, Z):
        """Callback function which evaluates the partial derivatives of the dependent variables in the ODE.

        Parameters:
            x - x-coordinate (independent variable)
            Z - dependent variable appearing in the ODE. In this case zeta and zeta_x. See manual for more information.

        Returns:
            dydz - numpy array that contains the values of the partial derivative of each dependent variable. See manual
                   for more information
        """
        # Import system variables
        H = np.array([self.input.v('H', x=x / self.L),
                      self.input.d('H', x=x / self.L, dim='x')])
        B = np.array([self.input.v('B', x=x / self.L),
                      self.input.d('B', x=x / self.L, dim='x')])
        r = self.rf(x)
        a = self.af(x, r)
        # Define local parameters
        sinhrh = np.sinh(r[0] * H[0])
        coshrh = np.cosh(r[0] * H[0])
        # Factors in ODE
        T1 = ((a[0] * sinhrh / r[0]) - H[0])
        T2 = (B[1] * T1 / B[0] + H[1] * (a[0] * coshrh - 1.) + a[1] * sinhrh / r[0] +
              a[0] * r[1] * (r[0] * H[0] * coshrh - sinhrh) / r[0] ** 2)
        T3 = self.SIGMA ** 2 / self.G

        # Definition of partial derivatives of ODE
        dydz = np.zeros((4, 4))
        dydz[0, 2] = 1.
        dydz[1, 3] = 1.
        dydz[2, :] = [np.real(T3 / T1), -np.imag(T3 / T1), -np.real(T2 / T1),  np.imag(T2 / T1)]
        dydz[3, :] = [np.imag(T3 / T1),  np.real(T3 / T1), -np.imag(T2 / T1), -np.real(T2 / T1)]
        return dydz

    def waterlevel(self):
        """Solves the boundary value problem for the water level

        Returns:
            zeta - water level and its first and second derivative w.r.t. x
        """
        zeta = np.zeros((3, len(self.x), 1, 3), dtype=complex)
        if self.input.v('solver') == (None or 'numerical'):
            jmax = self.input.v('grid', 'maxIndex', 'x')
            r = self.rf(self.x)
            a = self.af(self.x, r)
            H = self.input.v('H', x=self.x / self.L)
            M = ((a[0] * np.sinh(r[0] * H) / r[0]) - H) * self.input.v('B', x=self.x / self.L) * (self.G / (1j * self.SIGMA))
            F = np.zeros((jmax+1, 1), dtype=complex)    # Forcing term shape (x, number of right-hand sides)
            Fopen = np.zeros((1, 1), dtype=complex)     # Forcing term shape (1, number of right-hand sides)
            Fclosed = np.zeros((1, 1), dtype=complex)   # Forcing term shape (1, number of right-hand sides)
            Fopen[0,0] = self.bca
            Z, Zx, _ = zetaFunctionUncoupled(1, M, F, Fopen, Fclosed, self.input, hasMatrix = False)
            zeta[0, :, 0, 1] = Z[:, 0]
            zeta[1, :, 0, 1] = Zx[:, 0]
            zeta[2, :, 0, 1] = np.gradient(Zx[:, 0], self.x[1], edge_order=2)
        elif self.input.v('solver') == 'bvp':
            try:
                import scikits.bvp_solver
            except ImportError:
                raise KnownError('The scikits.bvp_solver module is not known')
            # Define problem
            problem = scikits.bvp_solver.ProblemDefinition(num_ODE=4,
                                                           num_parameters=0,
                                                           num_left_boundary_conditions=2,
                                                           boundary_points=(self.x[0], self.x[-1]),
                                                           function=self.system_ode,
                                                           boundary_conditions=self.bcs,
                                                           function_derivative=self.system_ode_der)
            # Define solution guess
            guess = np.array([np.real(self.bca), np.imag(self.bca), 0., 0.])
            # Solve boundary value problem
            solution = scikits.bvp_solver.solve(problem, solution_guess=guess, tolerance=self.TOL)
            # Extract solution
            Z1, Z2 = solution(self.x, eval_derivative=True)
            # Assign water level, including derivatives, to variable zeta
            zeta[0, :, 0, 1] = Z1[0] + 1j * Z1[1]
            zeta[1, :, 0, 1] = Z1[2] + 1j * Z1[3]
            zeta[2, :, 0, 1] = Z2[2] + 1j * Z2[3]
        return zeta

    def velocity(self, zeta):
        """Calculates the horizontal and vertical flow velocities based on the water level zeta

        Parameters:
            zeta - water level and its first and second derivative w.r.t. x

        Returns:
            u - horizontal flow velocity and several derivatives w.r.t. x and z
            w - vertical flow velocity and its derivative w.r.t. z
        """
        # Initiate variables
        u = np.zeros((5, len(self.x), len(self.z), 3), dtype=complex)
        w = np.zeros((2, len(self.x), len(self.z), 3), dtype=complex)
        # Extract parameters alpha and r and B
        rf = self.rf(self.x)
        r = rf[0, :].reshape(len(self.x), 1)
        rx = rf[1, :].reshape(len(self.x), 1)
        af = self.af(self.x, rf)
        a = af[0, :].reshape(len(self.x), 1)
        ax = af[1, :].reshape(len(self.x), 1)
        B = np.array([self.input.v('B', x=self.x/self.L), self.input.d('B', x=self.x/self.L, dim='x')])
        b = B[0].reshape(len(self.x), 1)
        bx = B[1].reshape(len(self.x), 1)
        # rename derivatives zeta
        zeta0 = zeta[0, :, 0, 1].reshape(len(self.x), 1)
        zetax = zeta[1, :, 0, 1].reshape(len(self.x), 1)
        zetaxx = zeta[2, :, 0, 1].reshape(len(self.x), 1)
        # Calculate velocities and derivatives
        c = self.G / (1j * self.SIGMA)
        sinhrz = np.sinh(r * self.zarr)
        coshrz = np.cosh(r * self.zarr)
        var1 = c * zetax
        var2 = (a * coshrz - 1.)
        var3 = a * rx * self.zarr * sinhrz
        # u
        u[0, :, :, 1] = var1 * var2
        # u_x
        u[1, :, :, 1] = c * zetaxx * var2 + var1 * (ax * coshrz + var3)
        # u_z
        u[2, :, :, 1] = var1 * (a * r * sinhrz)
        # u_zz
        u[3, :, :, 1] = var1 * (a * r**2 * coshrz)
        # u_zz_x
        u[4, :, :, 1] = c * (zetaxx * a * r**2 * coshrz + zetax * (ax * r**2 * coshrz + 2. * a * r * rx * coshrz +
                                                                   r**2 * var3))
        # w
        w[0, :, :, 1] = c * ((zetaxx + (bx / b) * zetax) * (self.zarr - (a / r) * sinhrz) - (1 / r) * zetax *
                         (sinhrz * ax + a * rx * (self.zarr * coshrz - (sinhrz / r))) - self.SIGMA**2 * zeta0 / self.G)
        # w_z
        w[1, :, :, 1] = -c * (var2 * (zetaxx + (bx / b) * zetax) + zetax * (ax * coshrz + var3))
        return u, w

