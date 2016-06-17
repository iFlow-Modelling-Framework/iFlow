"""
Date: 24-08-15
Authors: R.L. Brouwer
"""

import numpy as np
from scipy import integrate
from nifty.harmonicDecomposition import absoluteU, signU
from src.DataContainer import DataContainer
import logging


class SedDynamic:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the sediment transport

         Returns:
             Dictionary with results. At least contains the variables listed as output in the registry
         """
        self.logger.info('Running module Sediment')

        # Initiate variables
        self.SIGMA = self.input.v('OMEGA')
        self.RHOS = self.input.v('RHOS')
        self.DS = self.input.v('DS')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')    #
        self.ASTAR = self.input.v('astar')
        self.WS = self.input.v('ws')
        self.KH = self.input.v('Kh')
        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        self.dx = self.x[1:]-self.x[:-1]
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = self.z.reshape(1, len(self.z)) * (self.input.v('-H', x=self.x/self.L).reshape(len(self.x), 1) +
                                                      self.input.v('R', x=self.x / self.L).reshape(len(self.x), 1))
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=0).reshape(len(self.x), 1).reshape(len(self.x), 1)
        self.Av0x = self.input.d('Av', x=self.x/self.L, z=0, f=0, dim='x').reshape(len(self.x), 1)
        self.H = (self.input.v('H', x=self.x/self.L).reshape(len(self.x), 1) +
                  self.input.v('R', x=self.x/self.L).reshape(len(self.x), 1))
        self.Hx = (self.input.d('H', x=self.x/self.L, dim='x').reshape(len(self.x), 1) +
                   self.input.d('R', x=self.x / self.L, dim='x').reshape(len(self.x), 1))
        self.B = self.input.v('B', x=self.x/self.L)
        self.Bx = self.input.d('B', x=self.x/self.L, dim='x').reshape(len(self.x), 1)
        self.sf = self.input.v('Roughness', x=self.x/self.L, f=0).reshape(len(self.x), 1)
        self.sfx = self.input.d('Roughness', x=self.x/self.L, f=0, dim='x').reshape(len(self.x), 1)
        self.submodules0 = self.input.data['u0'].keys()
        self.submodules1 = self.input.data['u1'].keys()
        # Allocate space to save results
        d = dict()
        d['c'] = {'M0': {}, 'M2': {}, 'M4': {}}
        d['hatc a'] = {'c00': {}, 'c04': {}, 'c12': {'M0': {}, 'M4': {}, 'M2': {}}, 'c20': {}}
        d['hatc ax'] = {'c12': {'M0': {}, 'M4': {}}}
        d['a'] = {}
        d['T'] = {'TM0': {}, 'TM2': {'TM2M0': {}, 'TM2M4': {}, 'TM2M2': {}}, 'TM4': {}, 'Tdiff': {}, 'Tstokes': {}}
        d['F'] = {'Fdiff': {}, 'Fadv': {'FadvM0': {}, 'FadvM4': {}}}
        # assign values to the concentration amplitudes and calculate the transport function T and F
        for mod0 in self.submodules0:
            # leading order and first order horizontal velocity
            u0 = self.input.v('u0', mod0, range(0, len(self.x)), range(0, len(self.z)), 1)
            w0 = self.input.v('w0', mod0, range(0, len(self.x)), range(0, len(self.z)), 1)
            # if u0 is exactly 0 at certain indices, then add a very small number, because otherwise dividing by zero
            # will occur when calculating absoluteU(), uabs_M0_x and signU(). This will suppress a RunTimeWarning
            zero_index = np.where(u0 == 0)
            if zero_index[0].any:
                u0[zero_index[0], zero_index[1]] = 1.e-50
            # leading order horizontal velocity at the surface
            u0s = u0[:, 0].reshape(len(self.x), 1)
            # leading order water level
            zeta0 = self.input.v('zeta0', mod0, range(0, len(self.x)), 0, 1).reshape(len(self.x), 1)
            # extract leading order concentration amplitudes due M0 and M4 tide
            c00, c00x, c00z, c04, c04x, c04z, c20 = self.concentration_amplitudes_lead(u0, mod0)
            # extract first order concentration amplitude due to the surface boundary condition
            dummy_u1 = np.resize(u0, (len(self.x), len(self.z), 3))
            __, __, c12M2, c12M0adv_a, c12M4adv_a, c12M0adv_ax, c12M4adv_ax = self.concentration_amplitude_first(u0,
                                                                               dummy_u1, w0, zeta0,
                                                                               c00, c00x, c00z, c04, c04x, c04z)
            # save results
            d['hatc a']['c00'][mod0] = c00
            d['hatc a']['c04'][mod0] = c04
            d['hatc a']['c20'][mod0] = c20
            d['hatc a']['c12']['M2'][mod0] = c12M2
            d['hatc a']['c12']['M0']['sed adv'] = c12M0adv_a
            d['hatc ax']['c12']['M0'] = c12M0adv_ax
            d['hatc a']['c12']['M4']['sed adv'] = c12M4adv_a
            d['hatc ax']['c12']['M4'] = c12M4adv_ax
            # Calcuate the river-river interaction when river actually is a leading order contributor
            uriver = self.input.v('u1', 'river', range(0, len(self.x)), range(0, len(self.z)), 0)
            d['T']['TM0']['river_river'] = np.real(np.trapz(uriver * c20, x=-self.zarr, axis=1))
            d['T']['Tdiff'][mod0] = np.real(-np.trapz(self.KH * c00x, x=-self.zarr, axis=1))
            d['T']['Tstokes'][mod0] = np.real(2. * (np.conj(u0s) * c00[:, 0].reshape(len(self.x), 1) * zeta0 +
                                            u0s * c00[:, 0].reshape(len(self.x), 1) * np.conj(zeta0)) +
                                      u0s * np.conj(c04[:, 0].reshape(len(self.x), 1)) * zeta0 +
                                      np.conj(u0s) * c04[:, 0].reshape(len(self.x), 1) * np.conj(zeta0)).reshape(len(self.x)) / 8.
            d['T']['TM2']['TM2M2'][mod0] = np.real(np.trapz((u0 * np.conj(c12M2) + np.conj(u0) * c12M2) / 4., x=-self.zarr, axis=1))
            d['T']['TM2']['TM2M0']['sed adv'] = np.real(np.trapz((u0 * np.conj(c12M0adv_a) + np.conj(u0) * c12M0adv_a) / 4., x=-self.zarr, axis=1))
            d['T']['TM2']['TM2M4']['sed adv'] = np.real(np.trapz((u0 * np.conj(c12M4adv_a) + np.conj(u0) * c12M4adv_a) / 4., x=-self.zarr, axis=1))
            d['F']['Fdiff'][mod0] = np.real(-np.trapz(self.KH * c00, x=-self.zarr, axis=1))
            d['F']['Fadv']['FadvM0'][mod0] = np.real(np.trapz((u0 * np.conj(c12M0adv_ax) + np.conj(u0) * c12M0adv_ax) / 4., x=-self.zarr, axis=1))
            d['F']['Fadv']['FadvM4'][mod0] = np.real(np.trapz((u0 * np.conj(c12M4adv_ax) + np.conj(u0) * c12M4adv_ax) / 4., x=-self.zarr, axis=1))
            for mod1 in self.submodules1:
                # first order horizontal velocity
                u1 = self.input.v('u1', mod1, range(0, len(self.x)), range(0, len(self.z)), range(0, 3))
                # extract first order concentration amplitudes due to M0 and M4 tide
                c12M0, c12M4, __, __, __, __, __ = self.concentration_amplitude_first(u0, u1, w0, zeta0, c00, c00x,
                                                                                      c00z, c04, c04x, c04z)
                # save results
                d['hatc a']['c12']['M0'][mod1] = c12M0
                d['hatc a']['c12']['M4'][mod1] = c12M4
                d['T']['TM0'][mod0 + '_' + mod1] = np.real(np.trapz(u1[:, :, 0] * c00, x=-self.zarr, axis=1))
                d['T']['TM2']['TM2M0'][mod0 + '_' + mod1] = np.real(np.trapz((u0 * np.conj(c12M0) + np.conj(u0) * c12M0) / 4.,
                                                                 x=-self.zarr, axis=1))
                d['T']['TM2']['TM2M4'][mod0 + '_' + mod1] = np.real(np.trapz((u0 * np.conj(c12M4) + np.conj(u0) * c12M4) / 4.,
                                                                 x=-self.zarr, axis=1))
                d['T']['TM4'][mod0 + '_' + mod1] = np.real(np.trapz((u1[:, :, 2] * np.conj(c04) +
                                                            np.conj(u1[:, :, 2]) * c04) / 4., x=-self.zarr, axis=1))
        # place results in datacontainer
        dctrans = DataContainer(d)
        # calculate availability
        d['a'] = self.availability(dctrans.v('F'), dctrans.v('T')).reshape(len(self.x), 1)
        ax = np.gradient(d['a'][:, 0], self.x[1], edge_order=2).reshape(len(self.x), 1)
        # calculate ETM location a * hatc
        d['c']['M0'] = d['a'] * dctrans.v('hatc a', 'c00')
        d['c']['M2'] = d['a'] * dctrans.v('hatc a', 'c12') + ax * dctrans.v('hatc ax')
        d['c']['M4'] = d['a'] * dctrans.v('hatc a', 'c04')
        return d

    def concentration_amplitudes_lead(self, u0, component):
        """Calculates the amplitudes of the concentration for each tidal component at leading order

        Parameters:
            u0(x, z)  - leading order complex M2 velocity amplitude
            component - component of the leading order M2 velocity amplitude

        Returns:
            c00  - amplitude of the leading order M0 contribution of the sediment concentration
            c00x - x-derivative of the amplitude of the leading order M0 contribution of the sediment concentration
            c00z - z-derivative of the amplitude of the leading order M0 contribution of the sediment concentration
            c04  - amplitude of the leading order M4 contribution of the sediment concentration
            c04x - x-derivative of the amplitude of the leading order M4 contribution of the sediment concentration
            c04z - z-derivative of the amplitude of the leading order M4 contribution of the sediment concentration
        """
        # Extract velocity at the bottom
        u0b = u0[:, -1].reshape(len(self.x), 1)
        # M0 contribution
        uabs_M0 = absoluteU(u0b, 0).reshape(len(self.x), 1)
        uabs_M0_x = np.gradient(uabs_M0[:, 0], self.x[1], edge_order=2).reshape(len(self.x), 1)
        c00 = ((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_M0 *
               np.exp(-self.WS * (self.H + self.zarr) / self.Av0))
        c00x = ((self.RHOS / (self.GPRIME * self.DS)) * np.exp(-self.WS * (self.H + self.zarr) / self.Av0) *
                (self.sfx * uabs_M0 + self.sf * uabs_M0_x + self.sf * uabs_M0 * self.WS *
                 (self.Av0x * (self.H + self.zarr) - self.Av0 * self.Hx) / self.Av0**2))
        c00x[-1, :] = (0.5 * c00[-3, :] - 2 * c00[-2, :]) / (self.x[-1]-self.x[-2])
        c00z = -self.WS * c00 / self.Av0

        # Make time series of total velocity signal to extract residual velocities at the bottom due to order epsilon terms
        u1b = self.input.v('u1', range(0, len(self.x)), len(self.z)-1, range(0, 3))
        T = np.linspace(0, 2*np.pi, 100)
        u = np.zeros((len(self.x), len(T))).astype('complex')
        M4flag = 0.
        for i, t in enumerate(T):
            u[:, i] = u1b[:, 0] + 0.5 * (u0[:, -1] * np.exp(1j*t) + np.conj(u0[:, -1]) * np.exp(-1j*t) +
                                         M4flag * (u1b[:, 2] * np.exp(2*1j*t) + np.conj(u1b[:, 2]) * np.exp(-2*1j*t)))
        uabs_tot = np.mean(np.abs(u), axis=1)
        uabs_eps = uabs_tot.reshape(len(self.x), 1) - uabs_M0
        c20 = ((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_eps *
               np.exp(-self.WS * (self.H + self.zarr) / self.Av0))


        # M4 contribution
        uabs_M4 = absoluteU(u0b, 2)
        lambda_M4 = np.sqrt(self.WS**2 + 8 * 1j * self.SIGMA * self.Av0)
        r1_M4 = (lambda_M4 - self.WS) / (2 * self.Av0)
        r2_M4 = -(lambda_M4 + self.WS) / (2 * self.Av0)
        A2 = ((4 * self.WS * self.RHOS * self.sf / (self.GPRIME * self.DS)) * uabs_M4 * (lambda_M4 + self.WS) /
              ((lambda_M4 + self.WS)**2 * np.exp(-r2_M4 * self.H) - (lambda_M4 - self.WS)**2 * np.exp(-r1_M4 * self.H)))
        A1 = A2 * (lambda_M4 - self.WS) / (lambda_M4 + self.WS)
        c04 = (A1 * np.exp(r1_M4 * self.zarr) + A2 * np.exp(r2_M4 * self.zarr))
        c04x, __ = np.gradient(c04, self.x[1], edge_order=2)
        c04z = (A1 * r1_M4 * np.exp(r1_M4 * self.zarr) + A2 * r2_M4 * np.exp(r2_M4 * self.zarr))
        return c00, c00x, c00z, c04, c04x, c04z, c20

    def concentration_amplitude_first(self, u0, u1, w0, zeta0, c00, c00x, c00z, c04, c04x, c04z):
        """Calculates the amplitudes of the concentration for each tidal component at first order

        Parameters:
            u0(x, z)   - leading order complex horizontal M2 velocity amplitude
            u1(x, z)   - first order complex horizontal velocity amplitude
            w0(x, z)   - leading order complex vertical M2 velocity amplitude
            zeta0(x)   - leading order complex water level amplitude (at the surface z = 0) for x=0 to x=L
            c00(x, z)  - leading order amplitude of the M0 contribution of the sediment concentration
            c00x(x, z) - x-derivative of the amplitude of the leading order M0 contribution of the sediment concentration
            c00z(x, z) - z-derivative of the amplitude of the leading order M0 contribution of the sediment concentration
            c04(x, z)  - amplitude of the leading order M4 contribution of the sediment concentration
            c04x(x, z) - x-derivative of the amplitude of the leading order M4 contribution of the sediment concentration
            c04z(x, z) - z-derivative of the amplitude of the leading order M4 contribution of the sediment concentration

        Returns:
            c12M0     - amplitude of the first order M2 contribution of the sediment concentration due to the M0-part of
                        the bottom boundary condition z=-H
            c12M4     - amplitude of the first order M2 contribution of the sediment concentration due to the M4-part of
                        the bottom boundary condition z=-H
            c12M2     - amplitude of the first order M2 contribution of the sediment concentration due to the surface
                        boundary condition at z=0
            c12adv    - amplitude of the first order M2 contribution of the sediment concentration due to sediment
                        advection. This term can be divided in a part that goes with a(x) and da(x)/dx. Furthermore,
                        each of those parts have a M0 and a M4 part.
        """
        # CALCULATE THE PART OF C12 DUE TO THE BOTTOM BOUNDARY CONDITION
        # Extract leading and first order velocity at the bottom
        u0b = u0[:, -1].reshape(len(self.x), 1)
        u1b = u1[:, -1, :]
        # Extract M2 and M6 contribution of the sign of the leading order M2 velocity amplitude
        sguM2 = signU(u0b, 1).reshape(len(self.x), 1)
        sguM6 = signU(u0b, 3).reshape(len(self.x), 1)
        # Calculate the M2 contribution of u1 * u0 / |u0| at the bottom, which can be separated into a part that is due
        # to the M0-part of u1 and a part that is due to the M4-part of u1
        uM0 = 2. * u1b[:, 0].reshape(len(self.x), 1) * sguM2
        uM4 = u1b[:, 2].reshape(len(self.x), 1) * np.conj(sguM2) + np.conj(u1b[:, 2].reshape(len(self.x), 1)) * sguM6
        # Define variables
        lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Av0)
        r1_M2 = (lambda_M2 - self.WS) / (2 * self.Av0)
        r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Av0)
        p = (self.WS * self.RHOS * self.sf / (self.GPRIME * self.DS * self.Av0))
        B1M0 = (p * uM0 * (self.WS - lambda_M2) / (r2_M2 * (self.WS + lambda_M2) * np.exp(-r2_M2 * self.H) -
                                                   r1_M2 * (self.WS - lambda_M2) * np.exp(-r1_M2 * self.H)))
        B2M0 = -B1M0 * (self.WS + lambda_M2) / (self.WS - lambda_M2)
        B1M4 = (p * uM4 * (self.WS - lambda_M2) / (r2_M2 * (self.WS + lambda_M2) * np.exp(-r2_M2 * self.H) -
                                                   r1_M2 * (self.WS - lambda_M2) * np.exp(-r1_M2 * self.H)))
        B2M4 = -B1M4 * (self.WS + lambda_M2) / (self.WS - lambda_M2)
        # Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the M0- and M4-
        # part of u1
        c12M0 = (B1M0 * np.exp(r1_M2 * self.zarr) + B2M0 * np.exp(r2_M2 * self.zarr))
        c12M4 = (B1M4 * np.exp(r1_M2 * self.zarr) + B2M4 * np.exp(r2_M2 * self.zarr))

        # CALCULATE THE PART OF C12 DUE TO THE SURFACE BOUNDARY CONDITION
        # Extract the leading order M4 concentration at the surface
        c04s = c04[:, 0].reshape(len(self.x), 1)
        # Define variables
        var1 = self.WS + self.Av0 * r1_M2
        var2 = self.WS + self.Av0 * r2_M2
        # B1M2 = -1j * self.SIGMA * np.conj(zeta0) * c04s / (self.WS * (1. - r1_M2 * np.exp((r2_M2 - r1_M2) * self.H) / r2_M2) +
        #                                                    self.Av0 * r1_M2 * (1. - np.exp((r2_M2 - r1_M2) * self.H)))
        B1M2 = -1j * self.SIGMA * np.conj(zeta0) * c04s / (var1 - (r1_M2 / r2_M2) * var2 * np.exp((r2_M2 - r1_M2) * self.H))
        # B2M2 = 1j * self.SIGMA * np.conj(zeta0) * c04s / (self.WS * (r2_M2 * np.exp((r1_M2 - r2_M2) * self.H) / r1_M2 - 1.) +
        #                                                   self.Av0 * r2_M2 * (np.exp((r1_M2 - r2_M2) * self.H) - 1.))
        B2M2 = -1j * self.SIGMA * np.conj(zeta0) * c04s / (var2 - (r2_M2 / r1_M2) * var1 * np.exp((r1_M2 - r2_M2) * self.H))
        # Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the surface
        # boundary condition
        c12M2 = (B1M2 * np.exp(r1_M2 * self.zarr) + B2M2 * np.exp(r2_M2 * self.zarr))

        # CAlCULATE THE PART OF C12 DUE TO THE ADVECTION OF SEDIMENT
        # Extract the forcing term that is a function of a(x) and a_x(x)
        chi_a_M0 = u0 * c00x + w0 * c00z
        chi_a_M4 = 0.5 * (np.conj(u0) * c04x + np.conj(w0) * c04z)
        chi_ax_M0 = u0 * c00
        chi_ax_M4 = 0.5 * np.conj(u0) * c04
        chi = [chi_a_M0, chi_a_M4, chi_ax_M0, chi_ax_M4]
        c12adv = []
        for f in chi:
            int_r = np.trapz((var2 * np.exp(-r2_M2 * self.zarr) - var1 * np.exp(-r1_M2 * self.zarr)) * f /
                             (self.Av0 * (r2_M2 - r1_M2)), x=-self.zarr, axis=1).reshape(len(self.x), 1)
            A = - r2_M2 * np.exp(-r2_M2 * self.H) * int_r / (r2_M2 * var1 * np.exp(-r2_M2 * self.H) - r1_M2 * var2 * np.exp(-r1_M2 *self.H))
            B = - A * r1_M2 * np.exp((r2_M2 - r1_M2) * self.H) / r2_M2
            C = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r1_M2 * self.zarr) / (self.Av0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
            D = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r2_M2 * self.zarr) / (self.Av0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
            c12adv.append((A - C) * np.exp(r1_M2 * self.zarr) + (B + D) * np.exp(r2_M2 * self.zarr))
        return c12M0, c12M4, c12M2, c12adv[0], c12adv[1], c12adv[2], c12adv[3]

    def availability(self, F, T):
        """Calculates the availability of sediment needed to derive the sediment concentration

        Parameters:
            F - diffusive coefficient in the availability equation that goes with a_x
            T - coefficient (advective, diffusive and stokes) in the availability equation that goes with a

        Returns:
            a - availability of sediment throughout the estuary
        """
        # Exponent in the availability function. This exponent is set to zero (hard-coded) at the landward boundary
        # because here the availability is zero too!
        exponent = np.append(np.exp(-np.append(0, integrate.cumtrapz(T / F, dx=self.dx, axis=0)[:-1])), 0)
        A = (self.ASTAR * np.trapz(self.B * exponent, dx=self.dx, axis=0) /
             np.trapz(self.B, dx=self.dx, axis=0))
        a = A * exponent
        return a