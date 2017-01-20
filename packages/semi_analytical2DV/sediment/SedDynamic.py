"""
Date: 24-08-15
Authors: R.L. Brouwer
"""

import numpy as np
from scipy import integrate
from nifty.harmonicDecomposition import absoluteU, signU
from src.DataContainer import DataContainer
import nifty as ny
import logging


class SedDynamic_new:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
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
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        self.Av0 = self.input.v('Av', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        self.Av0x = self.input.d('Av', range(0, jmax+1), 0, 0, dim='x').reshape(jmax+1, 1)
        self.H = (self.input.v('H', range(0, jmax+1)).reshape(jmax+1, 1) +
                  self.input.v('R', range(0, jmax+1)).reshape(jmax+1, 1))
        self.Hx = (self.input.d('H', range(0, jmax+1), dim='x').reshape(jmax+1, 1) +
                   self.input.d('R', range(0, jmax+1), dim='x').reshape(jmax+1, 1))
        self.B = self.input.v('B', range(0, jmax+1))
        self.Bx = self.input.d('B', range(0, jmax+1), dim='x').reshape(jmax+1, 1)
        self.sf = self.input.v('Roughness', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        self.sfx = self.input.d('Roughness', range(0, jmax+1), 0, 0, dim='x').reshape(jmax+1, 1)
        self.submodules_hydro = self.input.data['u1'].keys()
        self.submodules_sed = self.input.v('submodules')
        # Extract leading order surface elevation and horizontal and vertical velocities
        self.zeta0 = self.input.v('zeta0', 'tide', range(0, jmax+1), 0, 1).reshape(jmax+1, 1)
        self.u0 = self.input.v('u0', 'tide', range(0, jmax+1), range(0, kmax+1), 1)
        self.w0 = self.input.v('w0', 'tide', range(0, jmax+1), range(0, kmax+1), 1)
        # Initiate dictionary to save results
        d = {}

        ################################################################################################################
        ## Calculate leading, first and second order concentration amplitudes hatc0, hatc1 and hatc2
        ################################################################################################################
        # Allocate space
        d['hatc0'] = {}
        d['hatc1'] = {'a': {}, 'ax': {}}
        d['hatc2'] = {}
        # Calculate leading order concentration amplitudes
        d['hatc0'] = self.erosion_lead()
        # Calculate first order concentration amplitudes
        for sedmod in self.submodules_sed:
            hatc1 = getattr(self, sedmod)()
            for k in hatc1.keys():
                d['hatc1'][k].update(hatc1[k])
        # Calculate second order concentration amplitudes
        d['hatc2'] = self.erosion_second()

        ################################################################################################################
        ## Calculate Transport function T and diffusion function F
        ################################################################################################################
        # Allocate space
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
                        tmp = np.real(np.trapz(tmp * self.c00, x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM0', ['return', 'drift'])
                            d['T'][submod]['TM0']['return'] += tmp
                    else:
                        tmp = np.real(np.trapz(tmp * self.c00, x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp
                elif n==2:
                    if submod == 'stokes':
                        tmp = np.real(np.trapz((tmp * np.conj(self.c04) + np.conj(tmp) * self.c04) / 4., x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM4', ['return', 'drift'])
                            d['T'][submod]['TM4']['return'] += tmp
                    else:
                        tmp = np.real(np.trapz((tmp * np.conj(self.c04) + np.conj(tmp) * self.c04) / 4., x=-self.zarr, axis=1))
                        if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp

        # Transport terms that are a function of the first order concentration, i.e. u0*c1 terms.
        for submod in d['hatc1']['a'].keys():
            if submod == 'erosion':
                for subsubmod in d['hatc1']['a'][submod].keys():
                    c1_comp = d['hatc1']['a'][submod][subsubmod]
                    d['T'] = self.dictExpand(d['T'], subsubmod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                    tmp = c1_comp[:, :, 1]
                    tmp = np.real(np.trapz((self.u0 * np.conj(tmp) + np.conj(self.u0) * tmp) / 4., x=-self.zarr, axis=1))
                    if subsubmod == 'stokes':
                        if any(tmp) > 10**-14:
                            d['T'][subsubmod] = self.dictExpand(d['T'][subsubmod], 'TM2', ['return', 'drift'])
                            d['T'][subsubmod]['TM2']['return'] += tmp
                    else:
                        if any(tmp) > 10**-14:
                            d['T'][subsubmod]['TM2'] += tmp
            else:
                c1_comp = d['hatc1']['a'][submod]
                d['T'] = self.dictExpand(d['T'], submod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                tmp = c1_comp[:, :, 1]
                tmp = np.real(np.trapz((self.u0 * np.conj(tmp) + np.conj(self.u0) * tmp) / 4., x=-self.zarr, axis=1))
                if any(tmp) > 10**-14:
                    d['T'][submod]['TM2'] += tmp

        # Transport terms that are related to diffusion, i.e. K_h*c0 or K_h*c2
        d['T'] = self.dictExpand(d['T'], 'diffusion_tide', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        d['T']['diffusion_tide']['TM0'] = np.real(-np.trapz(self.KH * self.c00x, x=-self.zarr, axis=1))
        d['T'] = self.dictExpand(d['T'], 'diffusion_river', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        tmp = d['hatc2']['a']['erosion']['river_river'][:, :, 0]
        tmp, __ = np.gradient(tmp, self.x[1], edge_order=2)
        tmp = np.real(-np.trapz(self.KH * tmp, x=-self.zarr, axis=1))
        if any(tmp) > 10**-14:
            d['T']['diffusion_river']['TM0'] = tmp

        # Transport terms that are related to Stokes drift, i.e. u0*c0*zeta0
        for n in (0, 2):
            u0s = self.u0[:, 0]
            tmp = d['hatc0']['a']['erosion'][:, 0, n]
            if n==0:
                tmp = np.real(np.conj(u0s) * tmp * self.zeta0[:, 0] + u0s * tmp * np.conj(self.zeta0[:, 0])) / 4
            elif n==2:
                tmp = np.real(u0s * np.conj(tmp) * self.zeta0[:, 0] + np.conj(u0s) * tmp * np.conj(self.zeta0[:, 0])) / 8
            if any(tmp) > 10**-14:
                d['T']['stokes']['TM' + str(2 * n)]['drift'] = tmp


        # Transport term that is related to the river-river interaction u1river*c2river
        d['T'] = self.dictExpand(d['T'], 'river_river', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
        u1_comp = self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax+1), 0)
        tmp = d['hatc2']['a']['erosion']['river_river'][:, :, 0]
        d['T']['river_river']['TM0'] = np.real(np.trapz(u1_comp * tmp, x=-self.zarr, axis=1))

        ## Diffusion F #################################################################################################
        # Diffusive part, i.e. Kh*c00 and Kh*c20
        d['F'] = self.dictExpand(d['F'], 'diffusion_tide', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
        d['F']['diffusion_tide']['FM0'] = np.real(-np.trapz(self.KH * self.c00, x=-self.zarr, axis=1))
        d['F'] = self.dictExpand(d['F'], 'diffusion_river', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
        tmp = d['hatc2']['a']['erosion']['river_river'][:, :, 0]
        tmp = np.real(-np.trapz(self.KH * tmp, x=-self.zarr, axis=1))
        d['F']['diffusion_river']['FM0'] = tmp

        # Part of F that is related to sediment advection, i.e. u0*c1sedadv
        for submod in d['hatc1']['ax'].keys():
            c1_comp = d['hatc1']['ax'][submod]
            d['F'] = self.dictExpand(d['F'], submod, ['FM' + str(2 * n) for n in range(0, fmax + 1)])
            tmp = c1_comp[:, :, 1]
            tmp = np.real(np.trapz((self.u0 * np.conj(tmp) + np.conj(self.u0) * tmp) / 4., x=-self.zarr, axis=1))
            if any(tmp) > 10**-14:
                d['F']['sedadv']['FM2'] += tmp

        ################################################################################################################
        # Calculate availability
        ################################################################################################################
        # Add all mechanisms to datacontainer
        dctrans = DataContainer(d)
        # Calculate availability
        d['a'] = {}
        d['a'] = self.availability(dctrans.v('F'), dctrans.v('T')).reshape(len(self.x), 1)
        ax = np.gradient(d['a'][:, 0], self.x[1], edge_order=2).reshape(len(self.x), 1)

        ################################################################################################################
        # Calculate concentrations, i.e. a*hatc(a) + ax*hatc(ax)
        ################################################################################################################
        d['c0'] = {}
        d['c1'] = {}
        d['c2'] = {}
        # Calculate a*c0(a)
        for submod in d['hatc0']['a'].keys():
            c0_comp = d['hatc0']['a'][submod]
            d['c0'][submod] = {}
            tmp = d['a'][:, None] * c0_comp
            d['c0'][submod] = tmp

        # Calculate a*c1(a) + ax*c1(ax)
        for submod in d['hatc1']['a'].keys():
            if submod == 'erosion':
                for subsubmod in d['hatc1']['a'][submod].keys():
                    c1_comp = d['hatc1']['a'][submod][subsubmod]
                    d['c1'] = self.dictExpand(d['c1'], submod, subsubmod)
                    tmp = d['a'][:, None] * c1_comp
                    d['c1'][submod][subsubmod] = tmp
            elif submod == 'sedadv':
                c1_comp_a = d['hatc1']['a'][submod]
                c1_comp_ax = d['hatc1']['ax'][submod]
                d['c1'][submod] = {}
                tmp = d['a'][:, None] * c1_comp_a + ax[:, None] * c1_comp_ax
                d['c1'][submod] = tmp
            else:
                c1_comp = d['hatc1']['a'][submod]
                d['c1'][submod] = {}
                tmp = d['a'][:, None] * c1_comp
                d['c1'][submod] = tmp

        # Calculate a*c2(a)
        for submod in d['hatc2']['a']['erosion'].keys():
            c2_comp = d['hatc2']['a']['erosion'][submod]
            d['c2'] = self.dictExpand(d['c2'], 'erosion', submod)
            tmp = d['a'][:, None] * c2_comp
            d['c2']['erosion'][submod] = tmp
        return d

    def erosion_lead(self):
        """Calculates the amplitudes of the concentration due to erosion for each tidal component at leading order,
        which has an M0 (residual) and an M4 tidal component.

        Returns:
            hatc0 - concentration amplitude of the M0 and M4 tidal component

        Additionally makes the following variables available to the module:
            c00  - concentration amplitude of the leading order M0 tidal component
            c00x - x-derivative of the concentration amplitude of the leading order M0 tidal component
            c00z - z-derivative of the concentration amplitude of the leading order M0 tidal component
            c04  - concentration amplitude of the leading order M4 tidal component
            c04x - x-derivative of the concentration amplitude of the leading order M4 tidal component
            c04z - z-derivative of the concentration amplitude of the leading order M4 tidal component
        """
        # Extract velocity at the bottom
        u0b = self.u0[:, -1].reshape(len(self.x), 1)
        # M0 contribution
        uabs_M0 = absoluteU(u0b, 0).reshape(len(self.x), 1)
        uabs_M0_x = np.gradient(uabs_M0[:, 0], self.x[1], edge_order=2).reshape(len(self.x), 1)
        self.c00 = ((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_M0 *
                np.exp(-self.WS * (self.H + self.zarr) / self.Av0))
        self.c00x = ((self.RHOS / (self.GPRIME * self.DS)) * np.exp(-self.WS * (self.H + self.zarr) / self.Av0) *
                (self.sfx * uabs_M0 + self.sf * uabs_M0_x + self.sf * uabs_M0 * self.WS *
                 (self.Av0x * (self.H + self.zarr) - self.Av0 * self.Hx) / self.Av0**2))
        self.c00x[-1, :] = (0.5 * self.c00[-3, :] - 2 * self.c00[-2, :]) / (self.x[-1]-self.x[-2])
        self.c00z = -self.WS * self.c00 / self.Av0

        # M4 contribution
        uabs_M4 = absoluteU(u0b, 2)
        lambda_M4 = np.sqrt(self.WS**2 + 8 * 1j * self.SIGMA * self.Av0)
        r1_M4 = (lambda_M4 - self.WS) / (2 * self.Av0)
        r2_M4 = -(lambda_M4 + self.WS) / (2 * self.Av0)
        p = (self.WS * self.RHOS * self.sf / (self.GPRIME * self.DS * self.Av0))
        A2 = ((4 * self.WS * self.RHOS * self.sf / (self.GPRIME * self.DS)) * uabs_M4 * (lambda_M4 + self.WS) /
              ((lambda_M4 + self.WS)**2 * np.exp(-r2_M4 * self.H) - (lambda_M4 - self.WS)**2 * np.exp(-r1_M4 * self.H)))
        A1 = A2 * (lambda_M4 - self.WS) / (lambda_M4 + self.WS)
        self.c04 = (A1 * np.exp(r1_M4 * self.zarr) + A2 * np.exp(r2_M4 * self.zarr))
        self.c04x, __ = np.gradient(self.c04, self.x[1], edge_order=2)
        self.c04z = (A1 * r1_M4 * np.exp(r1_M4 * self.zarr) + A2 * r2_M4 * np.exp(r2_M4 * self.zarr))

        hatc0 = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc0[:, :, 0] = self.c00
        hatc0[:, :, 2] = self.c04
        return {'a': {'erosion': hatc0}}

    def erosion(self):
        """Calculates the amplitudes of the concentration due to erosion at first order, which has an M2 tidal component

        Returns:
            hatc1 - concentration amplitude due to erosion of the M2 tidal component
        """
        # Extract leading and first order velocity at the bottom
        u0b = self.u0[:, -1].reshape(len(self.x), 1)
        hatc1 = {'a': {'erosion': {}}}
        for mod1 in self.submodules_hydro:
            # first order horizontal velocity
            u1 = self.input.v('u1', mod1, range(0, len(self.x)), range(0, len(self.z)), range(0, 3))
            u1b = u1[:, -1, :]
            # Extract M2 and M6 contribution of the sign of the leading order M2 velocity amplitude
            sguM2 = signU(u0b, 1).reshape(len(self.x), 1)
            sguM6 = signU(u0b, 3).reshape(len(self.x), 1)
            # Calculate the M2 contribution of u1 * u0 / |u0| at the bottom, which can be separated into a part that is
            # due to the M0-part of u1 and a part that is due to the M4-part of u1
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
            # Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the M0-
            # and M4-part of u1
            hatc12_erosion = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
            hatc12_erosion[:, :, 1] = (B1M0 * np.exp(r1_M2 * self.zarr) + B2M0 * np.exp(r2_M2 * self.zarr))
            hatc12_erosion[:, :, 1] += (B1M4 * np.exp(r1_M2 * self.zarr) + B2M4 * np.exp(r2_M2 * self.zarr))
            hatc1['a']['erosion'].update({mod1: hatc12_erosion})
        return hatc1

    def erosion_second(self):
        """Calculates the amplitude of the concentration due to the river-river interaction, which is
        a second order contribution with an M0 tidal component.

        Returns:
            hatc2 - concentration amplitude due to river-river interaction of the M0 tidal component
        """
        # Make time series of total velocity signal to extract residual velocities at the bottom due to order epsilon terms
        hatc2 = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        if self.input.v('u1', 'river'):
            u1b = self.input.v('u1', 'river', range(0, len(self.x)), len(self.z)-1, 0)
            T = np.linspace(0, 2*np.pi, 100)
            utid = np.zeros((len(self.x), len(T))).astype('complex')
            ucomb = np.zeros((len(self.x), len(T))).astype('complex')
            for i, t in enumerate(T):
                utid[:, i] = 0.5 * (self.u0[:, -1] * np.exp(1j*t) + np.conj(self.u0[:, -1]) * np.exp(-1j*t))
                ucomb[:, i] = u1b + 0.5 * (self.u0[:, -1] * np.exp(1j*t) + np.conj(self.u0[:, -1]) * np.exp(-1j*t))
            uabs_tid = np.mean(np.abs(utid), axis=1)
            uabs_tot = np.mean(np.abs(ucomb), axis=1)
            uabs_eps = uabs_tot.reshape(len(self.x), 1) - uabs_tid.reshape(len(self.x), 1)
            hatc2[:, :, 0] = ((self.RHOS / (self.GPRIME * self.DS)) * self.sf * uabs_eps *
                              np.exp(-self.WS * (self.H + self.zarr) / self.Av0))
        return {'a': {'erosion': {'river_river': hatc2}}}

    def noflux(self):
        """Calculates the amplitude of the concentration due to the no-flux boundary condition at the surface, which is
        a first order contribution with an M2 tidal component.

        Returns:
            hatc1 - concentration amplitude due to the no-flux boundary condition
        """
        # Extract the leading order M4 concentration at the surface
        c04s = self.c04[:, 0].reshape(len(self.x), 1)
        # Define variables
        lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Av0)
        r1_M2 = (lambda_M2 - self.WS) / (2 * self.Av0)
        r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Av0)
        var1 = self.WS + self.Av0 * r1_M2
        var2 = self.WS + self.Av0 * r2_M2
        B1M2 = -1j * self.SIGMA * np.conj(self.zeta0) * c04s / (var1 - (r1_M2 / r2_M2) * var2 * np.exp((r2_M2 - r1_M2) * self.H))
        B2M2 = -1j * self.SIGMA * np.conj(self.zeta0) * c04s / (var2 - (r2_M2 / r1_M2) * var1 * np.exp((r1_M2 - r2_M2) * self.H))
        # Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the no-flux
        # surface boundary condition
        hatc12_noflux = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_noflux[:, :, 1] = (B1M2 * np.exp(r1_M2 * self.zarr) + B2M2 * np.exp(r2_M2 * self.zarr))
        hatc1 = {'a': {'noflux': hatc12_noflux}}
        return hatc1

    def sedadv(self):
        """Calculates the amplitude of the concentration due to sediment advection, which is a first order contribution
        with an M2 tidal component.

        Returns:
            hatc2 - concentration amplitude due to river-river interaction of the M0 tidal component
        """
        # CAlCULATE THE PART OF C12 DUE TO THE ADVECTION OF SEDIMENT
        # Define variables
        lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Av0)
        r1_M2 = (lambda_M2 - self.WS) / (2 * self.Av0)
        r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Av0)
        var1 = self.WS + self.Av0 * r1_M2
        var2 = self.WS + self.Av0 * r2_M2
        # Extract the forcing term that is a function of a(x) and a_x(x)
        chi_a_M0 = self.u0 * self.c00x + self.w0 * self.c00z
        chi_a_M4 = 0.5 * (np.conj(self.u0) * self.c04x + np.conj(self.w0) * self.c04z)
        chi_ax_M0 = self.u0 * self.c00
        chi_ax_M4 = 0.5 * np.conj(self.u0) * self.c04
        chi = [chi_a_M0, chi_a_M4, chi_ax_M0, chi_ax_M4]
        hatc12_sedadv = []
        for f in chi:
            int_r = np.trapz((var2 * np.exp(-r2_M2 * self.zarr) - var1 * np.exp(-r1_M2 * self.zarr)) * f /
                             (self.Av0 * (r2_M2 - r1_M2)), x=-self.zarr, axis=1).reshape(len(self.x), 1)
            A = - r2_M2 * np.exp(-r2_M2 * self.H) * int_r / (r2_M2 * var1 * np.exp(-r2_M2 * self.H) - r1_M2 * var2 * np.exp(-r1_M2 *self.H))
            B = - A * r1_M2 * np.exp((r2_M2 - r1_M2) * self.H) / r2_M2
            C = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r1_M2 * self.zarr) / (self.Av0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
            D = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r2_M2 * self.zarr) / (self.Av0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
            c12 = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
            c12[:, :, 1] = (A - C) * np.exp(r1_M2 * self.zarr) + (B + D) * np.exp(r2_M2 * self.zarr)
            hatc12_sedadv.append(c12)
        hatc1 = {'a': {'sedadv': hatc12_sedadv[0]+hatc12_sedadv[1]}, 'ax': {'sedadv': hatc12_sedadv[2]+hatc12_sedadv[3]}}
        return hatc1

    def mixing(self):
        return

    def availability(self, F, T):
        """Calculates the availability of sediment needed to derive the sediment concentration

        Parameters:
            F - diffusive coefficient in the availability equation that goes with a_x
            T - coefficient (advective, diffusive and stokes) in the availability equation that goes with a

        Returns:
            a - availability of sediment throughout the estuary
        """
        # Exponent in the availability function.
        if self.input.v('Q1') > 0:
            exponent = np.exp(-integrate.cumtrapz(T / F, dx=self.dx, axis=0, initial=0))
        else:
            exponent = np.append(np.exp(-np.append(0, integrate.cumtrapz(T / F, dx=self.dx, axis=0)[:-1])), 0)
        A = (self.ASTAR * np.trapz(self.B, dx=self.dx, axis=0) /
             np.trapz(self.B * exponent, dx=self.dx, axis=0))
        a = A * exponent
        return a

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
                d[subindex][ssi] = 0
        return d
