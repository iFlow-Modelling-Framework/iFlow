"""
TurbulenceKepFitted
Implements a set of turbulence models obtained by fitting to solution of the k-epsilon model of Dijkstra et al (2016) 
for barotropic tidal flows. This class is not a full module, but provides to core for implementing modules.

Options:
    1. Profiles and roughness values through 'profile' and 'roughnessParameter':
    a.  profile='uniform' & roughnessParameter='sf0': uses relation Av = 0.49*sf0*(H+R+zeta); no velocity dependence
    b.  profile='uniform' & roughnessParameter='z0*: uses relation that scales with u and H+R+zeta

    2. Comination with reference level through 'referenceLevel'=True/False
        When True it combines the turbulence model with the computation of the module 'ReferenceLevel'. This option is
        recommended over using ReferenceLevel separately.

    3. Time dependence of Av and sf through 0<='lambda'<=1
        lambda = 0: no time dependence
        lambda = 1: full time dependence
        Values between 0 and 1 mean that a fraction lambda of the time dependence is taken into account

    4. Ignoring certain submodules through 'ignoreSubmodule' = ['submod1', 'submod2', ...]
        Sets submodule names to be ignored in the velocity and water level dependencies.

    5. Depth scaling of the roughness through 'n' (any scalar value)
        roughnessParameter = roughnessParameter [H/H(x=0)]^n

    6. Minimum subtidal eddy viscosity through 'Avmin' (any scalar value)
        Av = max(Av computed, Avmin)
        In case of a uniform profile, sf is automatically scaled to sf = max(sf computed, Av/(0.49*H))
        The time dependent eddy viscosity has no minimum value

Original date: 20-11-2015)
Update: 20-03-2023
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
import logging
from src.util.diagnostics.KnownError import KnownError
from .UniformXF import UniformXF
from .UniformX import UniformX
from packages.hydrodynamics2DV.perturbation.ReferenceLevel import ReferenceLevel


class TurbulenceKepFitted_core:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 1e-5       # relative change allowed for converged result
    RELAX = 0.5             # Relaxation factor. 1: no relaxation, 0: equal to previous iteration

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        """Stop iteration if differences is smaller than self.TOLLERANCE
        """
        stop = False        # default: do not stop
        if hasattr(self, 'difference'):
            if self.difference < self.TOLLERANCE*self.RELAX:
                stop = True
                del self.difference                 # delete any information that must be reset before rerunning this module
                if hasattr(self, 'u_prev_iter'):
                    del self.u_prev_iter
                    del self.uH_prev_iter
            else:
                self.logger.info('Turbulence model - iteration ' + str(iteration))  # print information on iteration number and progress of convergence
                self.logger.info(self.difference)
        return stop

    def main(self, order, init=False):
        """Main core of turbulence model. This refers to the several other functions that make the final functions
        Parameters:
            init (bool, default=False) - inital run? if True the input will be read, velocity will be estimated and convergence will not be checked
        """
        # Init - read input variables
        d = {}

        if init:
            self.reset = self.input.v('resetiFlow')
            if self.reset == 'True':
                self.reset = True
            else:
                self.reset = False
            self.roughnessParameter = self.input.v('roughnessParameter')
            self.n = self.input.v('n')
            self.timedependence = self.input.v('lambda')
            self.referenceLevel = self.input.v('referenceLevel') or 'False'
            self.ignoreSubmodule = ny.toList(self.input.v('ignoreSubmodule')) or []
            self.Avmin = self.input.v('Avmin')
            self.truncationorder = self.input.v('truncationOrder')

        ################################################################################################################
        # 0. Reference level (initial)
        ################################################################################################################

        if self.referenceLevel == 'True' and init and (order is None or order < 1):
            # make initial grid
            if self.input.v('grid') == None:
                grid = self._makegrid()
                self.input.merge(grid)

            # make reference level (only initial)
            self.RL = ReferenceLevel(self.input)
            R = self.RL.run_init()['R']
            jmax = self.input.v('grid', 'maxIndex', 'x')
            if not self.reset and np.linalg.norm(self.input.v('R', range(0, jmax+1)), np.inf)>0:      # instead take from DC R if it exists already. In all cases generate and init RL above.
                R = self.input.v('R', range(0, jmax+1))
            self.input.merge({'R': R})
            self.input._data['grid']['low']['z'] = R # add R to grid to be used in next iteration

        ################################################################################################################
        # 1. make the absolute velocity components
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        uabs, uabsH = self.uRelax(order, init)

        ################################################################################################################
        # 2. make the functions for the eddy viscosity and roughness parameter
        ################################################################################################################
        Av, roughness, BottomBC = self.profileSelection(uabs, uabsH, order)

        ################################################################################################################
        # 3. check convergence (not in inital run)
        ################################################################################################################
        if not init:
            if order == 0 or order is None:
                Avstr = 'Av'
            else:
                Avstr = 'Av'+str(order)
            Av_dc = self.input.slice('grid')
            Av_dc.addData('Av', Av)
            Avprev = self.input.v(Avstr, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            self.difference = np.max(np.abs(Avprev - Av_dc.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1)))/np.abs(Avprev+10**-4))
        ################################################################################################################
        # 4. Reference level (non-initial)
        ################################################################################################################
        if self.referenceLevel == 'True' and not init and (order is None or order < 1):
            self.input.merge({'Av': Av, 'Roughness': roughness})
            oldR = self.input.v('R', range(0, jmax+1))
            R = self.RL.run()['R']
            self.input._data['grid']['low']['z'] = R         # add R to grid to be used in next iteration

            self.difference = max(self.difference, np.linalg.norm(R-oldR, np.inf)/10.) # divide error in R by 10 to satisfy a weaker tollerance
        else:
            R = self.input.v('R')

        return Av, roughness, BottomBC, R

    def uRelax(self, order, init):
        """Compute the absolute velocity and absolute velocity times the depth at 'order',
        i.e. |u|^<order>, (|u|(H+R+zeta))^<order>.
        Then make a relaxation of these signals using the the previous iteration and relaxtion factor set as class var.

        Implements two methods:
            order == None: truncation method. Else: scaling method
            init (bool): initial iteration?
        """
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
        kmax = self.input.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
        fmax = self.input.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
        depth = self.input.v('grid', 'low', 'z', range(0, jmax+1), [0], [0]) - self.input.v('grid', 'high', 'z', range(0, jmax+1), [0], [0])

        # test if present u is grid-conform (may not be if the previous runs were on a different resolution)
        utest = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        no_u = False
        if isinstance(utest, bool):
            no_u = True


        ################################################################################################################
        # 1. make the absolute velocity
        ################################################################################################################
        c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs

        ## Truncation method
        if order == None:
            ##   1a. Gather velocity and zeta components
            zeta = 0
            u = 0
            comp = 0
            while not self.reset and self.input.has('zeta'+str(comp)) and self.input.has('u'+str(comp)) and comp <= self.truncationorder and not no_u:
                zeta += self.input.v('zeta'+str(comp), range(0, jmax + 1), [0], range(0, fmax + 1))
                u += self.input.v('u'+str(comp), range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
                for submod in self.ignoreSubmodule:     # remove submodules to be ignored
                    try:
                        zeta -= self.input.v('zeta'+str(comp), submod, range(0, jmax + 1), [0], range(0, fmax + 1))
                        u -= self.input.v('u'+str(comp), submod, range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
                    except:
                        pass
                comp += 1
            # if no data for u and zeta is in de DC, then take an initial estimate
            if comp == 0:
                u = np.zeros((jmax+1, kmax+1, fmax+1))
                u[:, :, 0] = 1.
                zeta = np.zeros((jmax+1, 1, fmax+1))

            usurf = u[:, [0], :]
            u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid'))

            ##   1b. Divide velocity by a maximum amplitude
            uamp = [(np.sum(np.abs(u), axis=-1)+10**-3).reshape((jmax+1, 1, 1)), (np.sum(np.abs(usurf), axis=-1)+10**-3).reshape((jmax+1, 1, 1))]
            u = u/uamp[0]
            usurf = usurf/uamp[1]

            ##  1c. Make absolute value
            uabs = [np.zeros(u.shape, dtype=complex), np.zeros(u.shape, dtype=complex)]  # uabs at depth-av, surface
            for n in [0, 1]:        # compute for DA (0) and surface (1)
                if n == 0:
                    ut = u
                else:
                    ut = usurf
                uabs[n][:, :, 0] = c[0]
                u2 = ny.complexAmplitudeProduct(ut, ut, 2)
                uabs[n] += c[2]*u2
                u4 = ny.complexAmplitudeProduct(u2, u2, 2)
                uabs[n] += c[4]*u4
                u6 = ny.complexAmplitudeProduct(u2, u4, 2)
                uabs[n] += c[6]*u6
                del u2, u6
                u8 = ny.complexAmplitudeProduct(u4, u4, 2)
                uabs[n] += c[8]*u8
                del u4, u8

                uabs[n] = uabs[n] * uamp[n].reshape((jmax+1, 1, 1))

            #   Absolute velocity * depth
            uabsH = uabs[0] + ny.complexAmplitudeProduct(uabs[1], zeta, 2)
            uabs = uabs[0]/depth + ny.complexAmplitudeProduct(uabs[1]-uabs[0]/depth, zeta, 2)

        ## Scaling method
        else:
            ##   1a. Gather velocity and zeta components
            zeta = []
            u = []
            usurf = []
            for comp in range(0, order+1):
                if not self.reset and self.input.has('zeta'+str(comp)) and self.input.has('u'+str(comp)) and not no_u:
                    zetatemp = self.input.v('zeta'+str(comp), range(0, jmax + 1), [0], range(0, fmax + 1))
                    utemp = self.input.v('u'+str(comp), range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
                    for submod in self.ignoreSubmodule:     # remove submodules to be ignored
                        try:
                            zetatemp -= self.input.v('zeta'+str(comp), submod, range(0, jmax + 1), [0], range(0, fmax + 1))
                            utemp -= self.input.v('u'+str(comp), submod, range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
                        except:
                            pass
                # if no u and zeta in DC, then ..
                elif comp == 0:     # .. add velocity 1 to subtidal leading order
                    zetatemp = np.zeros((jmax+1, 1, fmax+1))
                    utemp = np.zeros((jmax+1, kmax+1, fmax+1))
                    utemp[:, :, 0] = 1.
                else:               # .. add nothing at higher orders
                    zetatemp = np.zeros((jmax+1, 1, fmax+1))
                    utemp = np.zeros((jmax+1, kmax+1, fmax+1))

                zeta.append(zetatemp)
                usurf.append(utemp[:, [0], :])
                u.append(ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid')) / depth)

                ##   1b. Divide velocity by a maximum amplitude
                uamp = []
                uamp.append((np.sum(np.abs(sum(u)), axis=-1)+10**-3).reshape((jmax+1, 1, 1)))
                uamp.append((np.sum(np.abs(sum(usurf)), axis=-1)+10**-3).reshape((jmax+1, 1, 1)))
                u = [i/uamp[0] for i in u]
                usurf = [i/uamp[1] for i in usurf]

            ##  1c. Make absolute value
            uabs = [np.zeros(u[0].shape+(order+1,), dtype=complex), np.zeros(u[0].shape+(order+1,), dtype=complex)]
            for n in [0, 1]:        # compute for DA (0) and surface (1)
                if n == 0:
                    ut = u
                else:
                    ut = usurf
                uabs[n][:, :, 0, 0] = c[0]
                for q in range(0, order+1):
                    uabs[n][:, :, :, q] += c[2]*self.umultiply(2, q, ut)
                    uabs[n][:, :, :, q] += c[4]*self.umultiply(4, q, ut)
                    uabs[n][:, :, :, q] += c[6]*self.umultiply(6, q, ut)
                    uabs[n][:, :, :, q] += c[8]*self.umultiply(8, q, ut)

                uabs[n] = uabs[n] * uamp[n].reshape((jmax+1, 1, 1, 1))

            #   Absolute velocity * depth
            uabsH = uabs[0][:, :, :, order]*depth
            for q in range(0, order):
                uabsH += ny.complexAmplitudeProduct(uabs[1][:, :, :, q]-uabs[0][:, :, :, q], zeta[order-q-1], 2)

            #   Only keep uabs at current order
            uabs = uabs[0][:, :, :, order]      # only keep DA part

        ################################################################################################################
        # 2. Relaxtion
        ################################################################################################################
        ##   2a. Relaxation on uabs
        if hasattr(self, 'u_prev_iter') and self.u_prev_iter.shape == uabs.shape:
            u_prev_iter = self.u_prev_iter
        else:
            u_prev_iter = uabs      # take current uabs if no previous is available

        u_prev_iter2 = u_prev_iter - (uabs)
        u0 = np.max((uabs, np.min((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + (uabs)), axis=0)
        u0 = np.min((u0,   np.max((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + uabs), axis=0)
        self.u_prev_iter = u0  # save velocity at bed for next iteration

        ##    2b. Relaxation on uabs*depth
        if hasattr(self, 'uH_prev_iter') and self.uH_prev_iter.shape == uabsH.shape:
            u_prev_iter = self.uH_prev_iter
        else:
            u_prev_iter = uabsH     # take current uabsH if no previous is available

        u_prev_iter2 = u_prev_iter - (uabsH)
        uH0 = np.max((uabsH, np.min((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + (uabsH)), axis=0)
        uH0 = np.min((uH0,   np.max((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + uabsH), axis=0)
        self.uH_prev_iter = uH0  # save velocity at bed for next iteration

        return u0, uH0

    def umultiply(self, pow, N, u):
        """ Compute the sum of all possible combinations yielding the power 'pow' of signal 'u' with a total order 'N'
        i.e. (u^pow)^<N>
        """
        v = 0
        if pow>2:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(self.umultiply(2, i, u), self.umultiply(pow-2, N-i, u), 2)
        else:
            for i in range(0, N+1):
                v += ny.complexAmplitudeProduct(u[i], u[N-i], 2)
        return v

    def profileSelection(self, uabs, uabsH, order):
        """ Go through a menu with turbulence profiles and roughness parameters. Selects the correct one based on the input
         Then determines the coefficients and prepares the functions for the eddy viscosity and Roughness

        Parameters:
            uabs (array) - approximation of the absolute depth-averaged velocity (at the current order)
            uabsH (array) - approximation of the absolute depth-averaged velocity * depth (at the current order)
            order (int or None) - current order of the calculation

        Returns:
            prepared functions for Av and the roughness parameter of choice. Also returns the type of boundary condition
        """
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        if order is None or order < 1:
            Avmin = self.Avmin

        #   Make a new data container with the roughness parameter with the depth-scaling param n incorporated
        data = self.input.slice('grid')
        data.addData('coef', self.input.v(self.roughnessParameter))
        roughness = self.input.slice('grid')
        roughness.addData('Roughness', UniformX(data, self.n).value)

        ################################################################################################################
        # Select the correct profile and roughness parameter:
        #   1. Uniform
        #       1a. uniform + sf0 (linear)
        #       1b. uniform + z0*(non-linear)
        ################################################################################################################

        ################################################################################################################
        ## case 1a: uniform + sf0 (linear)
        ################################################################################################################
        if self.roughnessParameter == 'sf0':
            ## 1. Eddy viscosity
            Av0 = np.zeros((jmax + 1, fmax + 1), dtype=complex)
            # truncated model
            if order == None:
                depth = np.zeros((jmax+1, fmax+1), dtype=complex)
                depth[:, 0] = self.input.v('grid', 'low', 'z', range(0,jmax+1)) - self.input.v('grid', 'high', 'z', range(0,jmax+1))
                i = 0
                while self.input.v('zeta'+str(i)) is not None and i <= self.truncationorder:
                    depth += self.input.v('zeta'+str(i), range(0, jmax+1), 0, range(0, fmax+1))
                    for submod in self.ignoreSubmodule:
                        try:
                            depth -= self.input.v('zeta'+str(i), submod, range(0, jmax+1), 0, range(0, fmax+1))
                        except:
                            pass
                    i += 1

                Av0[:, :] = 0.49 * roughness.v('Roughness', range(0, jmax + 1), 0, [0]) * depth

            # Leading order
            elif order == 0:
                depth = self.input.v('grid', 'low', 'z', x=0) - self.input.v('grid', 'high', 'z', x=0)
                Av0[:, 0] = 0.49 * roughness.v('Roughness', range(0, jmax + 1)) * depth

            # Higher orders
            else:
                depth = self.input.v('zeta'+str(order-1), range(0, jmax+1), 0, range(0, fmax+1))
                for submod in self.ignoreSubmodule:
                    try:
                        depth -= self.input.v('zeta'+str(order-1), submod, range(0, jmax+1), 0, range(0, fmax+1))
                    except:
                        pass
                Av0[:, :] = 0.49 * roughness.v('Roughness', range(0, jmax + 1)).reshape((jmax+1, 1)) * self.input.v('zeta'+str(order-1), range(0, jmax+1), 0, range(0, fmax+1))

            #   background eddy viscosity
            if order is None or order < 1:       # i.e. None or 0
                Av0[:, 0] = np.maximum(Av0[:, 0], Avmin)

            #   adjust time dependence (NB no risk of negative eddy viscosity if zeta < H+R)
            Av0[:, 1:] = self.timedependence*Av0[:, 1:]

            #   put data in output variables
            data = self.input.slice('grid')
            data.addData('coef', Av0)
            if order == 0:
                Av = UniformXF(data, 1.).value
            else:
                Av = UniformXF(data, 0.).value

            ## 2. Roughness
            if order == None or order == 0:
                sf0 = np.zeros((jmax + 1, fmax + 1))
                sf0[:, 0] = roughness.v('Roughness', range(0, jmax+1))

                dataRough = self.input.slice('grid')
                dataRough.addData('coef', sf0)

                roughness = UniformXF(dataRough, 0.).value
            else:
                roughness = 0.

            ## 3. Boundary type
            BottomBC = 'PartialSlip'

            # No iteration required unless the reference level is computed
            if self.referenceLevel == 'False':
                self.difference = 0.

        ################################################################################################################
        ## case 1b: constant + z0* (non-linear)
        ################################################################################################################
        elif self.roughnessParameter in ['z0*']:
            # 1. prepare coefficients
            z0st = roughness.v('Roughness', range(0, jmax + 1))
            Cd_div_k2 = (((1. + z0st) * np.log(1. / z0st + 1) - 1) ** -2.).reshape(jmax + 1, 1)

            Av0 = 0.10 / 0.636 * Cd_div_k2 * uabsH[:, 0, :]
            sf0 = 0.22 / 0.636 * Cd_div_k2 * uabs[:, 0, :]

            #   background eddy viscosity
            if order is None or order < 1:       # i.e. None or 0
                Av0[:, 0] = np.maximum(Av0[:,0], self.Avmin)
                depth = self.input.v('grid', 'low', 'z', x=0) - self.input.v('grid', 'high', 'z', x=0)
                sf0[:, 0] = np.maximum(sf0[:,0], self.Avmin*2/depth)       # minimum sf = 2*Avmin/H (relation from case 1a)

            #   remove time dependence if required
            Av0[:, 1:] = self.timedependence*Av0[:, 1:]
            sf0[:, 1:] = self.timedependence*sf0[:, 1:]

            #   correct possible negative eddy viscosity
            Av0 = self.positivity_correction('Av', Av0, order, False)
            sf0 = self.positivity_correction('Roughness', sf0, order, False)

            sf0t = ny.invfft2(sf0, 1, 90)
            Av0t = ny.invfft2(Av0, 1, 90)
            ind = 0
            while (sf0t<0).any() and ind < 50:
                sf0 = self.positivity_correction('Roughness', sf0, order, False)
                sf0t = ny.invfft2(sf0, 1, 90)
                ind += 1
            if ind == 50:
                raise KnownError('sf not sufficiently corrected for positivity')
            ind = 0
            while (Av0t<0).any() and ind < 50:
                Av0 = self.positivity_correction('Av', Av0, order, False)
                Av0t = ny.invfft2(Av0, 1, 90)
                ind += 1
            if ind == 50:
                raise KnownError('Av not sufficiently corrected for positivity')

            # 2. prepare smaller DataContainers
            data = self.input.slice('grid')
            data.addData('coef', Av0)

            dataRough = self.input.slice('grid')
            dataRough.addData('coef', sf0)

            # 3. prepare functions
            Av = UniformXF(data, 0.).value
            roughness = UniformXF(dataRough, 0.).value
            BottomBC = 'PartialSlip'

        else:
            raise KnownError('Combination of turbulence profile and roughness parameter is not implemented')

        return Av, roughness, BottomBC

    def positivity_correction(self, quantity, value_currentorder, order, include_vertical):
        """Correct 'quantity' so that its total over all orders does not become negative.
        Do this by reducing the time-varying components at the current order

        Parameters:
            quantity (str) - name of the quantity without order marking (assumes that leading order is not marked with a number, e.g. Av, Av1, Av2 ..)
            value_currentorder (int) - value of quantity at current order of computation
            order (int or None) - current order of computation (None=truncation)
            include_vertical (bool) - is there a vertical dependency?

        Returns:
            corrected result of 'quantity' only at the current order

        """
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        if include_vertical:
            kmax = self.input.v('grid', 'maxIndex', 'z')
        else:
            kmax = 0
            value_currentorder = value_currentorder.reshape((value_currentorder.shape[0], 1, value_currentorder.shape[-1]))
        fmax = self.input.v('grid', 'maxIndex', 'f')

        # Add all eddy viscosity components
        if order is None:
            Av = value_currentorder         # use value supplied in truncation method
        else:
            Av = 0 + 0*1j
            for i in range(0, order):       # take all lower orders from DC, then add the value at current order
                if i==0:
                    ordstr = quantity
                else:
                    ordstr = quantity+str(order)
                Av += self.input.v(ordstr, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            Av += value_currentorder

        # make a time series with 100 time steps
        Avt = ny.invfft2(Av, len(Av.shape)-1, 90) - Av[:, :, [0]]

        # correct by reducing time varying components
        if not np.min(np.real(np.minimum(1, abs(Av[:, :, 0])/(-np.min(Avt, axis=-1)+10**-10))))==1.:
            value_currentorder[:, :, 1:] = value_currentorder[:, :, 1:]*np.real(np.minimum(1, .95*abs(Av[:, :, 0])/(-np.min(Avt, axis=-1)+10**-10)).reshape((jmax+1, kmax+1, 1)))
            # value_currentorder = self.positivity_correction(quantity, value_currentorder, include_vertical)

        if not include_vertical:
            value_currentorder = value_currentorder.reshape((value_currentorder.shape[0], value_currentorder.shape[-1]))
        return value_currentorder

    def _makegrid(self):
        """ Create a temporary grid if necessary
        """
        grid = {}
        dimensions = ['x', 'z', 'f']
        enclosures = [(0, self.input.v('L')),
                      (self.input.v('R'), self.input.n('H')),
                      None]
        contraction = [[], ['x'], []]     # enclosures of each dimension depend on these parameters
        copy = [1, 1, 0]

        # grid
        axisTypes = []
        axisSize = []
        axisOther = []
        variableNames = ['xgrid', 'zgrid', 'fgrid']
        gridsize = {}
        gridsize['xgrid'] = ['equidistant', 100]
        gridsize['zgrid'] = ['equidistant', 100]
        gridsize['fgrid'] = ['integer', 2]
        for var in variableNames:
            axisTypes.append(gridsize[var][0])
            axisSize.append(gridsize[var][1])
            axisOther.append(gridsize[var][2:])
        grid['grid'] = ny.makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)
        return grid