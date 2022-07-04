"""
Original date: 01-06-15
Update: 15-02-2022
Original authors: R.L. Brouwer
Update authors: Y.M. Dijkstra
"""

import logging
import numpy as np
import nifty as ny
from .util.zetaFunctionUncoupled import zetaFunctionUncoupled


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
        self.OMEGA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=self.x/self.L).reshape((len(self.x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]
        self.bca = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (2,))[1]

        # Prepare output
        d = dict()
        d['zeta0'] = {}
        d['u0'] = {}
        d['w0'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['xx'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['zz'] = {}
        d['__derivative']['zzx'] = {}
        d['__derivative']['x']['zeta0'] = {}
        d['__derivative']['xx']['zeta0'] = {}
        d['__derivative']['x']['u0'] = {}
        d['__derivative']['z']['u0'] = {}
        d['__derivative']['zz']['u0'] = {}
        d['__derivative']['zzx']['u0'] = {}
        d['__derivative']['z']['w0'] = {}

        zeta = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
        zetax = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
        zetaxx = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        # Run computations
        zeta[:, 0, 1], zetax[:, 0, 1], zetaxx[:, 0, 1] = self.waterlevel()
        u, w = self.velocity(zeta[:, 0, 1], zetax[:, 0, 1], zetaxx[:, 0, 1])

        # Save water level results
        d['zeta0']['tide'] = zeta
        d['__derivative']['x']['zeta0']['tide'] = zetax
        d['__derivative']['xx']['zeta0']['tide'] = zetaxx

        # Save velocity results
        d['u0']['tide'] = u[0]
        d['__derivative']['x']['u0']['tide'] = u[1]
        d['__derivative']['z']['u0']['tide'] = u[2]
        d['__derivative']['zz']['u0']['tide'] = u[3]
        d['__derivative']['zzx']['u0']['tide'] = u[4]

        d['w0']['tide'] = w[0]
        d['__derivative']['z']['w0']['tide'] = w[1]
        return d

    def rf(self, x):
        """Calculate the root r = \sqrt(i\sigma / Av) of the characteristic equation and its derivatives wrt x.

        Parameters:
            x - x-coordinate

        Returns:
            r - root of the characteristic equation of the leading order horizontal velocity
        """
        Av = self.input.v('Av', x=x/self.L, z=0, f=0)
        Avx = self.input.d('Av', x=x/self.L, z=0, f=0, dim='x')
        Avxx = self.input.d('Av', x=x/self.L, z=0, f=0, dim='xx')

        r = np.sqrt(1j * self.OMEGA / Av)
        rx = -np.sqrt(1j * self.OMEGA) * Avx / (2. * Av**(3./2.))
        rxx = np.sqrt(1j * self.OMEGA) * (3. * Avx**2 - 2. * Av * Avxx) / (4. * Av**(5./2.))
        return r, rx, rxx

    def af(self, x, r, rx, rxx):
        """Calculate the coefficient alpha that appears in the solution for the leading order horizontal velocity.

        Parameters:
            x - x-coordinatemm

        Returns:
            a - coefficient alpha
        """
        H   = self.input.v('H', x=x/self.L) + self.input.v('R', x=x/self.L)
        Hx  = self.input.d('H', x=x/self.L, dim='x') + self.input.d('R', x=x/self.L, dim='x'),
        Hxx = self.input.d('H', x=x/self.L, dim='xx') + self.input.d('R', x=x/self.L, dim='xx')   # YMD 15-08-17 Reference level
        Av   = self.input.v('Av', x=x/self.L, z=0, f=0)
        Avx  = self.input.d('Av', x=x/self.L, z=0, f=0, dim='x')
        Avxx = self.input.d('Av', x=x/self.L, z=0, f=0, dim='xx')
        sf   = self.input.v('Roughness', x=x/self.L, z=0, f=0)
        sfx  = self.input.d('Roughness', x=x/self.L, z=0, f=0, dim='x')
        sfxx = self.input.d('Roughness', x=x/self.L, z=0, f=0, dim='xx')

        # sf = sf[:, 0]             # BUG (?) 23-02-2018
        # Define trigonometric values for ease of reference
        sinhrh = np.sinh(r * H)
        coshrh = np.cosh(r * H)
        cothrh = coshrh / sinhrh
        cschrh = 1 / sinhrh

        # Define parameters and their (second) derivative wrt x
        E = rx * H + r * Hx
        Ex = rxx * H + 2. * rx * Hx + r * Hxx
        F = rx + r * E * cothrh
        Fx = rxx + r * Ex * cothrh + E * (rx * cothrh - r * E**2 * cschrh**2)

        K = r * Avx + Av * F + sfx * cothrh + sf * E
        Kx = (r * Avxx + rx * Avx + Avx * F + Av * Fx + sfxx * cothrh -
              sfx * E * cschrh**2 + sfx * E + sf * Ex)

        G = r * Av * sinhrh + sf * coshrh
        Gx = sinhrh * K
        Gxx = E * K * coshrh + Kx * sinhrh

        # Calculate coefficient alpha
        a = sf / G    # a
        ax = sfx / G - sf * Gx / G**2    # a_x
        axx = (sfxx-(2.*sfx*Gx + sf*Gxx)/G + 2.*sf*Gx**2/G**2) / G    # YMD bug corrected 27-2-2018
        return a, ax, axx

    def waterlevel(self):
        """Solves the boundary value problem for the water level

        Returns:
            zeta - water level and its first and second derivative w.r.t. x
        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        r, rx, rxx = self.rf(self.x)
        a, ax, axx = self.af(self.x, r, rx, rxx)
        H = self.input.v('H', x=self.x / self.L) + self.input.v('R', x=self.x / self.L)

        M = ((a * np.sinh(r * H) / r) - H) * self.input.v('B', x=self.x / self.L) * (self.G / (1j * self.OMEGA))
        F = np.zeros((jmax+1, 1), dtype=complex)    # Forcing term shape (x, number of right-hand sides)
        Fopen = np.zeros((1, 1), dtype=complex)     # Forcing term shape (1, number of right-hand sides)
        Fclosed = np.zeros((1, 1), dtype=complex)   # Forcing term shape (1, number of right-hand sides)
        Fopen[0,0] = self.bca

        Z, Zx, _ = zetaFunctionUncoupled(1, M, F, Fopen, Fclosed, self.input, hasMatrix = False)
        zeta = Z[:, 0]
        zeta_x = Zx[:, 0]
        zeta_xx = np.gradient(Zx[:, 0], self.x[1], edge_order=2)

        return zeta, zeta_x, zeta_xx

    def velocity(self, zeta0, zetax, zetaxx):
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
        r, rx, rxx = self.rf(self.x)
        a, ax, axx = self.af(self.x, r, rx, rxx)

        r = r.reshape(len(self.x), 1)
        rx = rx.reshape(len(self.x), 1)

        a = a.reshape(len(self.x), 1)
        ax = ax.reshape(len(self.x), 1)

        B = self.input.v('B', x=self.x/self.L).reshape(len(self.x), 1)
        Bx = self.input.d('B', x=self.x/self.L, dim='x').reshape(len(self.x), 1)

        # reshape (derivatives of) zeta
        zeta0 = zeta0.reshape(len(self.x), 1)
        zetax = zetax.reshape(len(self.x), 1)
        zetaxx = zetaxx.reshape(len(self.x), 1)

        # Calculate velocities and derivatives
        c = self.G / (1j * self.OMEGA)
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
        w[0, :, :, 1] = c * ((zetaxx + (Bx / B) * zetax) * (self.zarr - (a / r) * sinhrz) - (1 / r) * zetax *
                         (sinhrz * ax + a * rx * (self.zarr * coshrz - (sinhrz / r))) - self.OMEGA**2 * zeta0 / self.G)
        # w_z
        w[1, :, :, 1] = -c * (var2 * (zetaxx + (Bx / B) * zetax) + zetax * (ax * coshrz + var3))
        return u, w

