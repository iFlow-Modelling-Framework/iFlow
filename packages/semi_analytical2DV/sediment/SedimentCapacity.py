"""
Sediment dynamic computation using semi-analytical method

Options on input
    erosion_formulation: what erosion parameter to use; 'Partheniades' or 'Chernetsky'
    sedbc: type of horizontal boundary condition: 'csea' for seaward concentration or 'astar' for total amount of sediment in the domain
    friction: what roughness value to use; 'roughness' for same roughness as hydrodynamics, 'skin_friction' for a skin friction coefficient which is different to the hydrodynamic roughness

Optional input parameters
    Qsed: sediment inflow from upstream
    sedsource: other sediment sources

From version: 2.4 - update 2.6, 2.7
Date: December 2018
Authors: R.L. Brouwer, Y.M. Dijkstra
Additions by Y.M. Dijkstra: - erodability (with time integrator); v2.6
                            - skin friction option instead of form drag; v2.6
"""
import numpy as np
from scipy import integrate
from nifty.harmonicDecomposition import absoluteU, signU
from src.DataContainer import DataContainer
import nifty as ny
import logging
import scipy.linalg


class SedimentCapacity:
    # Variables
    # timer = ny.Timer()

    # Methods
    def __init__(self, input):
        self.logger = logging.getLogger(__name__)
        self.input = input
        return

    def run(self):
        """Run function to initiate the calculation of the sediment transport

         Returns:
             Dictionary with results. At least contains the variables listed as output in the registry
         """
        # self.timer.tic()
        self.logger.info('Running module Sediment Capacity')

        ################################################################################################################
        ## 1. Initiate variables
        ################################################################################################################
        self.SIGMA = self.input.v('OMEGA')
        self.RHOS = self.input.v('RHOS')
        self.RHO0 = self.input.v('RHO0')
        self.DS = self.input.v('DS')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')    #

        self.L = self.input.v('L')
        self.x = self.input.v('grid', 'axis', 'x') * self.input.v('L')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.z = self.input.v('grid', 'axis', 'z', 0, range(0, kmax+1))
        self.zarr = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]-self.input.v('R', x=self.x/self.L).reshape((len(self.x), 1))      #YMD 22-8-17 includes reference level; note that we take a reference frame z=[-H-R, 0]

        self.WS = self.input.v('ws0', range(0, jmax+1), [0], 0)
        self.WSx = self.input.d('ws0', range(0, jmax+1), [0], 0, dim='x')
        self.Kv0 = self.input.v('Kv', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
        self.Kv0x = self.input.d('Kv', range(0, jmax+1), 0, 0, dim='x').reshape(jmax+1, 1)

        self.HR = (self.input.v('H', range(0, jmax+1)).reshape(jmax+1, 1) +
                  self.input.v('R', range(0, jmax+1)).reshape(jmax+1, 1))
        self.Hx = self.input.d('H', range(0, jmax+1), dim='x').reshape((jmax+1, 1))

        if self.input.v('friction') is not None:
            self.sf = self.input.v(self.input.v('friction'), range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
            self.sfx = self.input.d(self.input.v('friction'), range(0, jmax+1), 0, 0, dim='x').reshape(jmax+1, 1)
        else:
            self.sf = self.input.v('Roughness', range(0, jmax+1), 0, 0).reshape(jmax+1, 1)
            self.sfx = self.input.d('Roughness', range(0, jmax+1), 0, 0, dim='x').reshape(jmax+1, 1)
        self.submodules_hydro = self.input.data['u1'].keys()
        self.submodules_sed = self.input.v('submodules')

        # Extract leading order surface elevation and horizontal and vertical velocities
        self.zeta0 = self.input.v('zeta0', 'tide', range(0, jmax+1), 0, 1).reshape(jmax+1, 1)
        self.u0 = self.input.v('u0', 'tide', range(0, jmax+1), range(0, kmax+1), 1)
        self.w0 = self.input.v('w0', 'tide', range(0, jmax+1), range(0, kmax+1), 1)
        self.finf = self.input.v('finf')
        self.finfx = self.input.d('finf', dim='x')

        # Initiate dictionary to save results
        d = {}

        ################################################################################################################
        ## 2. Calculate leading, first and second order concentration amplitudes hatc0, hatc1 and hatc2
        ################################################################################################################
        # Initiate variables
        d['hatc0'] = {}
        d['hatc1'] = {'a': {},
                      'ax': {}
                      }
        d['hatc2'] = {}

        # Calculate leading-order concentration amplitudes
        d['hatc0'] = self.erosion_lead()

        # Calculate first order concentration amplitudes
        for sedmod in self.submodules_sed:
            hatc1 = getattr(self, sedmod)()
            for k in hatc1.keys():
                d['hatc1'][k].update(hatc1[k])

        # Calculate second order concentration amplitudes
        d['hatc2'] = self.erosion_second()
        # self.timer.toc()
        # self.timer.disp('time sediment capacity')
        # self.timer.reset()
        return d

########################################################################################################################
## Functions related to the sediment concentration capacity
########################################################################################################################
    def erosion_lead(self):
        """Calculates the amplitudes of the concentration due to erosion for each tidal component at leading order,
        which has an M0 (residual) and an M4 tidal component.

        Returns:
            hatc0 - concentration amplitude of the M0 and M4 tidal component

        Additionally makes the following variables available to the module:
            c00  - concentration amplitude of the leading order M0 tidal component
            c00x - x-derivative of the concentration amplitude of the leading order M0 tidal component
            c00z - z-derivative of the concentration amplitude of the leading order M0 tidal component
            c00z - zz-derivative of the concentration amplitude of the leading order M0 tidal component
            c04  - concentration amplitude of the leading order M4 tidal component
            c04x - x-derivative of the concentration amplitude of the leading order M4 tidal component
            c04z - z-derivative of the concentration amplitude of the leading order M4 tidal component
            c04zz - zz-derivative of the concentration amplitude of the leading order M4 tidal component
        """
        # Extract velocity at the bottom
        u0b = self.u0[:, -1].reshape(len(self.x), 1)
        
        ## M0 contribution
        # erosion
            # bed shear stress
        uabs_M0 = absoluteU(u0b, 0).reshape(len(self.x), 1)
        uabs_M0_x = np.gradient(uabs_M0[:, 0], self.x[1], edge_order=2).reshape(len(self.x), 1)

            # near-bed concentration, from erosion law
        if self.input.v('erosion_formulation') == 'Partheniades':
            Ehat = self.finf*self.sf*uabs_M0*self.RHO0
            cb = Ehat/self.WS
            cbx = self.RHO0*(self.finfx*self.sf*uabs_M0/self.WS + self.finf*self.sf*uabs_M0_x/self.WS + self.finf*uabs_M0*self.sfx/self.WS - self.finf*self.sf*uabs_M0*self.WSx/self.WS**2)
        else:
            cb = self.RHOS *self.finf/(self.GPRIME * self.DS)*self.sf*uabs_M0
            cbx = self.RHOS *self.finfx/(self.GPRIME * self.DS)*self.sf*uabs_M0 + self.RHOS *self.finf/(self.GPRIME * self.DS)*self.sf*uabs_M0_x + self.RHOS *self.finf/(self.GPRIME * self.DS)*self.sfx*uabs_M0

        self.c00 = cb * np.exp(-self.WS * (self.HR + self.zarr) / self.Kv0)

        # Derivative (analytical)
        self.c00x = cbx*np.exp(-self.WS * (self.HR + self.zarr) / self.Kv0) - self.c00*(self.WSx/self.Kv0 * (self.HR + self.zarr) - self.WS*self.Kv0x/self.Kv0**2 * (self.HR + self.zarr) + self.WS/self.Kv0*self.Hx)
        self.c00z = -self.WS * self.c00 / self.Kv0
        self.c00zz = - self.WS * self.c00z / self.Kv0

        # M4 contribution
        uabs_M4 = absoluteU(u0b, 2)
        if self.input.v('erosion_formulation') == 'Partheniades':
            Ehat = 2*self.finf*self.sf*uabs_M4*self.RHO0
        else:
            Ehat = 2*self.WS *self.RHOS *self.finf/(self.GPRIME * self.DS)*self.sf*uabs_M4

        lambda_M4 = np.sqrt(self.WS**2 + 8 * 1j * self.SIGMA * self.Kv0)
        r1_M4 = (lambda_M4 - self.WS) / (2 * self.Kv0)
        r2_M4 = -(lambda_M4 + self.WS) / (2 * self.Kv0)

        coef = (lambda_M4 - self.WS) / (lambda_M4 + self.WS)
        A2 = -Ehat/(r1_M4*self.Kv0*coef*np.exp(-r1_M4*self.HR)+r2_M4*self.Kv0*np.exp(-r2_M4*self.HR))
        A1 = A2*coef

        self.c04 = A1 * np.exp(r1_M4 * (self.zarr)) + A2 * np.exp(r2_M4 * (self.zarr))

        # Derivative (analytical)
        self.c04x, __ = np.gradient(self.c04, self.x[1], edge_order=2)
        self.c04z = (A1 * r1_M4 * np.exp(r1_M4 * (self.zarr+self.HR)) + A2 * r2_M4 * np.exp(r2_M4 * (self.zarr+self.HR)))
        self.c04zz = (A1 * r1_M4 ** 2 * np.exp(r1_M4 * (self.zarr+self.HR)) + A2 * r2_M4 ** 2 * np.exp(r2_M4 * (self.zarr+self.HR)))

        ## Prepare results
        hatc0 = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc0[:, :, 0] = self.c00
        hatc0[:, :, 2] = self.c04

        hatc0x = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc0x[:, :, 0] = self.c00x
        hatc0x[:, :, 2] = self.c04x

        nfu = ny.functionTemplates.NumericalFunctionWrapper(hatc0, self.input.slice('grid'))
        nfu.addDerivative(hatc0x, 'x')
        return {'a': {'erosion': nfu.function}}

    def erosion(self):
        """Calculates the amplitudes of the concentration due to erosion at first order, which has an M2 tidal component

        Returns:
            hatc1 - concentration amplitude due to erosion of the M2 tidal component
        """
        ## Extract leading and first order velocity at the bottom
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

            # erosion
            if self.input.v('erosion_formulation') == 'Partheniades':
                Ehat = self.finf*self.sf*(uM0+uM4)*self.RHO0
            else:
                Ehat = (self.WS * self.finf*self.RHOS * self.sf / (self.GPRIME * self.DS))*(uM0+uM4)

            # Define axiliary variables
            lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Kv0)
            r1_M2 = (lambda_M2 - self.WS) / (2 * self.Kv0)
            r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Kv0)

            B1 = (Ehat/self.Kv0 * (self.WS - lambda_M2) / (r2_M2 * (self.WS + lambda_M2) * np.exp(-r2_M2 * self.HR) -
                                                       r1_M2 * (self.WS - lambda_M2) * np.exp(-r1_M2 * self.HR)))
            B2 = -(Ehat/self.Kv0  * (self.WS + lambda_M2) / (r2_M2 * (self.WS + lambda_M2) * np.exp(-r2_M2 * self.HR) -
                                                         r1_M2 * (self.WS - lambda_M2) * np.exp(-r1_M2 * self.HR)))

            ## Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the M0-
            ## and M4-part of u1
            hatc12_erosion = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
            hatc12_erosion[:, :, 1] = (B1 * np.exp(r1_M2 * self.zarr) + B2 * np.exp(r2_M2 * self.zarr))

            ## Prepare result
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
            utid = np.zeros((len(self.x), len(T)), dtype='complex')
            ucomb = np.zeros((len(self.x), len(T)), dtype='complex')
            for i, t in enumerate(T):
                utid[:, i] = 0.5 * (self.u0[:, -1] * np.exp(1j*t) + np.conj(self.u0[:, -1]) * np.exp(-1j*t))
                ucomb[:, i] = u1b + 0.5 * (self.u0[:, -1] * np.exp(1j*t) + np.conj(self.u0[:, -1]) * np.exp(-1j*t))
            uabs_tid = np.mean(np.abs(utid), axis=1)
            uabs_tot = np.mean(np.abs(ucomb), axis=1)
            uabs_eps = uabs_tot.reshape(len(self.x), 1) - uabs_tid.reshape(len(self.x), 1)

            # erosion
            if self.input.v('erosion_formulation') == 'Partheniades':
                Ehat = self.finf*self.sf*(uabs_eps)*self.RHO0
            else:
                Ehat = (self.WS * self.finf*self.RHOS * self.sf / (self.GPRIME * self.DS))*(uabs_eps)

            hatc2[:, :, 0] = Ehat/self.WS * np.exp(-self.WS * (self.HR + self.zarr) / self.Kv0)
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
        lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Kv0)
        r1_M2 = (lambda_M2 - self.WS) / (2 * self.Kv0)
        r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Kv0)
        var1 = self.WS + self.Kv0 * r1_M2
        var2 = self.WS + self.Kv0 * r2_M2
        B1M2 = -1j * self.SIGMA * np.conj(self.zeta0) * c04s / (var1 - (r1_M2 / r2_M2) * var2 * np.exp((r2_M2 - r1_M2) * self.HR))
        B2M2 = -1j * self.SIGMA * np.conj(self.zeta0) * c04s / (var2 - (r2_M2 / r1_M2) * var1 * np.exp((r1_M2 - r2_M2) * self.HR))
        # Calculate the amplitude of the first order M2 contribution of the sediment concentration due to the no-flux
        # surface boundary condition
        hatc12_noflux = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_noflux[:, :, 1] = (B1M2 * np.exp(r1_M2 * self.zarr) + B2M2 * np.exp(r2_M2 * self.zarr))
        hatc1 = {'a': {'noflux': hatc12_noflux}}
        return hatc1

    # def sedadv(self):
    #     """Calculates the amplitude of the concentration due to sediment advection, which is a first order contribution
    #     with an M2 tidal component.
    #
    #     Returns:
    #         hatc1 - concentration amplitude due to river-river interaction of the M0 tidal component
    #     """
    #     # CAlCULATE THE PART OF C12 DUE TO THE ADVECTION OF SEDIMENT
    #     # Define variables
    #     lambda_M2 = np.sqrt(self.WS**2 + 4 * 1j * self.SIGMA * self.Kv0)
    #     r1_M2 = (lambda_M2 - self.WS) / (2 * self.Kv0)
    #     r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Kv0)
    #     var1 = self.WS + self.Kv0 * r1_M2
    #     var2 = self.WS + self.Kv0 * r2_M2
    #     # Extract the forcing term that is a function of a(x) and a_x(x)
    #     chi_a_M0 = self.u0 * self.c00x + self.w0 * self.c00z
    #     chi_a_M4 = 0.5 * (np.conj(self.u0) * self.c04x + np.conj(self.w0) * self.c04z)
    #     chi_ax_M0 = self.u0 * self.c00
    #     chi_ax_M4 = 0.5 * np.conj(self.u0) * self.c04
    #     chi = [chi_a_M0, chi_a_M4, chi_ax_M0, chi_ax_M4]
    #     hatc12_sedadv = []
    #     for f in chi:
    #         int_r = np.trapz((var2 * np.exp(-r2_M2 * self.zarr) - var1 * np.exp(-r1_M2 * self.zarr)) * f /
    #                          (self.Kv0 * (r2_M2 - r1_M2)), x=-self.zarr, axis=1).reshape(len(self.x), 1)
    #         A = - r2_M2 * np.exp(-r2_M2 * self.HR) * int_r / (r2_M2 * var1 * np.exp(-r2_M2 * self.HR) - r1_M2 * var2 * np.exp(-r1_M2 *self.HR))
    #         B = - A * r1_M2 * np.exp((r2_M2 - r1_M2) * self.HR) / r2_M2
    #         C = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r1_M2 * self.zarr) / (self.Kv0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
    #         D = np.fliplr(integrate.cumtrapz(np.fliplr(f * np.exp(-r2_M2 * self.zarr) / (self.Kv0 * (r2_M2 - r1_M2))), x=-self.zarr, axis=1, initial=0))
    #         c12 = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
    #         c12[:, :, 1] = (A - C) * np.exp(r1_M2 * self.zarr) + (B + D) * np.exp(r2_M2 * self.zarr)
    #         hatc12_sedadv.append(c12)
    #     hatc1 = {'a': {'sedadv': hatc12_sedadv[0]+hatc12_sedadv[1]}, 'ax': {'sedadv': hatc12_sedadv[2]+hatc12_sedadv[3]}}
    #     return hatc1
    def sedadv(self):
        """Calculates the amplitude of the concentration due to sediment advection, which is a first order contribution
        with an M2 tidal component.

        Returns:
            hatc1 - concentration amplitude due to river-river interaction of the M0 tidal component
        """
        # CAlCULATE THE PART OF C12 DUE TO THE ADVECTION OF SEDIMENT
        # Define variables
        self.Av0 = self.input.v('Av', x=self.x/self.L, z=0, f=[0])
        self.H = self.HR
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
        """Calculates the amplitude of the concentration due to asymmetric mixing, which is a first order contribution
        with an M2 tidal component.

        Returns:
            hatc1 - concentration amplitude due to asymmetric mixing
        """
        # import variables
        Kv1 = self.input.v('Kv1', len(self.x), len(self.z), 0)

        # CALCULATE THE PART OF C12 DUE THE MIX TERM TO I (INHOMOGENEOUS PDE)
        D1M0 = self.c00zz * Kv1
        D1M4 = np.conj(self.c04zz) * Kv1 * 0.5
        # Define Variables
        lambda_M2 = np.sqrt(self.WS ** 2 + 4 * 1j * self.SIGMA * self.Kv0)
        r1_M2 = (lambda_M2 - self.WS) / (2 * self.Kv0)
        r2_M2 = -(lambda_M2 + self.WS) / (2 * self.Kv0)
        B2mix = 2 * (lambda_M2 - self.WS) * self.WS * np.exp(-r1_M2 * self.HR) / (1j * self.SIGMA * (
                    (lambda_M2 - self.WS) ** 2 * np.exp(-r1_M2 * self.HR) - (lambda_M2 + self.WS) ** 2 * np.exp(
                        -r2_M2 * self.HR)))
        B1mix = B2mix * (lambda_M2 - self.WS) / (lambda_M2 + self.WS) - 2 * self.WS / (
            1j * self.SIGMA * (lambda_M2 + self.WS))
        hatc12_mixI = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_mixIb = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_mixI[:, :, 1] = D1M0 * (B1mix * np.exp(r1_M2 * self.zarr) + B2mix * np.exp(r2_M2 * self.zarr)
                            + 1 / (1j * self.SIGMA))
        # c12M0mixIz = D1M0 * (B1mix * r1_M2 * np.exp(r1_M2 * self.zarr) + B2mix * r2_M2 * np.exp(r2_M2 * self.zarr))
        # c12M0mixIzz = D1M0 * (
        #     B1mix * r1_M2 ** 2 * np.exp(r1_M2 * self.zarr) + B2mix * r2_M2 ** 2 * np.exp(r2_M2 * self.zarr))

        hatc12_mixIb[:, :, 1] = D1M4 * (B1mix * np.exp(r1_M2 * self.zarr) + B2mix * np.exp(r2_M2 * self.zarr)
                            + 1 / (1j * self.SIGMA))
        # c12M4mixIz = D1M4 * (B1mix * r1_M2 * np.exp(r1_M2 * self.zarr) + B2mix * r2_M2 * np.exp(r2_M2 * self.zarr))
        # c12M4mixIzz = D1M4 * (
        #     B1mix * r1_M2 ** 2 * np.exp(r1_M2 * self.zarr) + B2mix * r2_M2 ** 2 * np.exp(r2_M2 * self.zarr))

        # CALCULATE THE PART OF C12 DUE TO THE MIX TERM II
        G1M0 = self.c00z * Kv1
        G1M4 = np.conj(self.c04z) * Kv1 * 0.5
        # Define variables
        A2mix = (lambda_M2-self.WS)*np.exp(-r1_M2*self.HR) / (
            -(self.WS + lambda_M2) ** 2 * np.exp(-r2_M2 * self.HR)+(lambda_M2 - self.WS) ** 2 * np.exp(-r1_M2 * self.HR) )
        A1mix = -(1 + (self.WS - lambda_M2) * A2mix) / (self.WS + lambda_M2)
        hatc12_mixII = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_mixIIb = np.zeros((len(self.x), len(self.z), 3), dtype=complex)
        hatc12_mixII[:, :, 1] = 2 * G1M0 * (A1mix * np.exp(r1_M2 * self.zarr) + A2mix * np.exp(r2_M2 * self.zarr))
        # c12M0mixIIz = 2 * G1M0 * (A1mix * r1_M2 * np.exp(r1_M2 * self.zarr) + A2mix * r2_M2 * np.exp(r2_M2 * self.zarr))
        # c12M0mixIIzz = 2 * G1M0 * (A1mix * r1_M2 ** 2 * np.exp(r1_M2 * self.zarr)
        #                            + A2mix * r2_M2 ** 2 * np.exp(r2_M2 * self.zarr))

        hatc12_mixIIb[:, :, 1] = 2 * G1M4 * (A1mix * np.exp(r1_M2 * self.zarr) + A2mix * np.exp(r2_M2 * self.zarr))
        # c12M4mixIIz = 2 * G1M4 * (A1mix * r1_M2 * np.exp(r1_M2 * self.zarr) + A2mix * r2_M2 * np.exp(r2_M2 * self.zarr))
        # c12M4mixIIzz = 2 * G1M4 * (A1mix * r1_M2 ** 2 * np.exp(r1_M2 * self.zarr)
        #                            + A2mix * r2_M2 ** 2 * np.exp(r2_M2 * self.zarr))
        hatc1 = {'a': {'mixing': hatc12_mixI + hatc12_mixII}}
        return hatc1

