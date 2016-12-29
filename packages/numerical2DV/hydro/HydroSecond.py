"""


Date: 20-01-16
Authors: Y.M. Dijkstra
"""

import logging
import numpy as np
import nifty as ny
from zetaFunctionMassConservative import zetaFunctionMassConservative
import math
from uFunctionMomentumConservative import uFunctionMomentumConservative


class HydroSecond:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """
        """
        self.logger.info('Running module HydroSecond')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        BETA = self.input.v('BETA')
        OMEGA = self.input.v('OMEGA')
        ftot = 2*fmax+1
        self.submodulesToRun = self.input.v('submodules')
        try:
            maxContributions = int(self.input.v('maxContributions'))
        except:
            maxContributions = self.input.v('maxContributions')
        d = {}

        ################################################################################################################
        # Velocity in terms of water level gradient
        ################################################################################################################
        ## LHS terms
        #   try to get velocityMatrix from the input. If it is not found, proceed to calculate the matrix again
        A = self.input.v('velocityMatrix')  # known that this is numeric data, ask without arguments to retrieve full ndarray
        velocityMatrix = True
        if A is None:
            A = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            velocityMatrix = False

        ## RHS terms
        #   Determine number/names of right hand side
        nRHS, nRHSVelocity = self.__numberOfForcings()
        f_index = -1
        f_names = []

        # instantiate the forcing components in the equation
        F = np.zeros([jmax+1, kmax+1, ftot, nRHSVelocity], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, nRHSVelocity], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, nRHSVelocity], dtype=complex)
        JuFirst = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        uFirst = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
        uzFirst = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)

        # determine RHS terms per submodule - first for velocity
        # 1. Advection
        if 'adv' in self.submodulesToRun:
            for order1 in range(0, 2):
                order2 = 2-order1-1

                # labels and submodules
                u_str1 = 'u'+str(order1)
                u_keys1 = self.input.getKeysOf(u_str1)
                u_str2 = 'u'+str(order2)
                u_keys2 = self.input.getKeysOf(u_str2)
                w_str = 'w'+str(order1)

                # retrieve data and make advection forcing
                for submod1 in u_keys1:
                    for submod2 in u_keys2:
                        u = self.input.v(u_str1, submod1, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                        ux= self.input.d(u_str2, submod2, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
                        w = self.input.v(w_str , submod1, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                        uz= self.input.d(u_str2, submod2, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')

                        eta = ny.complexAmplitudeProduct(u, ux, 2) + ny.complexAmplitudeProduct(w, uz, 2)
                        eta = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), eta), 2)
                        f_index += 1
                        f_names.append(['adv', submod1+str(order1)+'-'+submod2+str(order2)])
                        F[:, :, :, f_index] = -eta


        # 2. No-Stress
        if 'nostress' in self.submodulesToRun:
            # labels and submodules
            hydro0_keys = list(set(self.input.getKeysOf('u0') + self.input.getKeysOf('zeta0')))
            hydro1_keys = list(set(self.input.getKeysOf('u1') + self.input.getKeysOf('zeta1')))

            # retrieve data and make forcing
            for submod1 in hydro0_keys:
                for submod2 in hydro1_keys:
                    D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
                    u0 = self.input.v('u0', submod1, range(0, jmax+1), [0], range(0, fmax+1))
                    u0x = self.input.d('u0', submod1, range(0, jmax+1), [0], range(0, fmax+1), dim='x')
                    zeta0 = self.input.v('zeta0', submod1, range(0, jmax+1), [0], range(0, fmax+1))
                    zeta0x = self.input.d('zeta0', submod1, range(0, jmax+1), [0], range(0, fmax+1), dim='x')

                    zeta1 = self.input.v('zeta1', submod2, range(0, jmax+1), [0], range(0, fmax+1))
                    zeta1x = self.input.d('zeta1', submod2, range(0, jmax+1), [0], range(0, fmax+1), dim='x')
                    u1 = self.input.v('u1', submod2, range(0, jmax+1), [0], range(0, fmax+1))

                    Avuz0z = D*u0 + G*zeta0x
                    Avuz1z = D*u1 + G*zeta1x
                    if submod2 in ['adv']:
                        adv = ny.complexAmplitudeProduct(u0, u0x, 2)
                        Avuz1z += adv

                    chi = ny.complexAmplitudeProduct(Avuz0z, zeta1, 2) + ny.complexAmplitudeProduct(Avuz1z, zeta0, 2)
                    chi = np.concatenate((np.zeros([jmax+1, 1, fmax]), chi), 2)

                    f_index += 1
                    f_names.append(['nostress', submod1+'0'+'-'+submod2+'1'])
                    Fsurf[:, :, :, f_index] = -chi

        # 3. Density-drift # TODO
        # if 'densitydrift' in self.submodulesToRun:
        #     beta_delta = 0
        #     for m in range(1, self.currentOrder):
        #         for k in range(0, self.currentOrder-m):
        #             zetapermutations = self.multiindex(m, self.currentOrder-m-k-1)
        #             # a. make the zeta^m product
        #             zetasum = 0
        #             for perm in range(0, zetapermutations.shape[0]):
        #                 zetaname = 'zeta'+str(zetapermutations[perm, 0])
        #                 zeta = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
        #                 for comp in range(1, zetapermutations.shape[1]):
        #                     zetaname = 'zeta'+str(zetapermutations[perm, comp])
        #                     zeta2 = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
        #                     zeta = ny.complexAmplitudeProduct(zeta, zeta2, 2)
        #                 zetasum = zetasum + zeta
        #
        #             # b. make the (s_x)^(m) term
        #             sname = 's'+str(k)
        #             sx = self.input.d(sname, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        #             sxmdz = self.surfaceDerivative(sx, m-1, self.NUMORDER_SURFACE)
        #
        #             # c. add all to beta_delta
        #             beta_delta += G*BETA*1./np.math.factorial(m)*ny.complexAmplitudeProduct(sxmdz, zetasum, 2)
        #
        #     beta_delta = np.concatenate((np.zeros([jmax+1, 1, fmax]), beta_delta), 2)
        #
        #     f_index += 1
        #     f_names.append(['densitydrift', ''])
        #     F[:, :, :, f_index] = -beta_delta*np.ones((jmax+1, kmax+1, fmax+1))

        # 4. Baroclinic
        #   Only use this when salinity on lower order is available
        if 'baroc' in self.submodulesToRun:
            sx = self.input.d('s1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
            if sx is not None:
                Jsx = -G*BETA*ny.integrate(sx, 'z', 0, range(0, kmax+1), self.input.slice('grid'))  # integral from z to 0 has its boundaries inverted and has a minus sign to compensate
                Jsx = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), Jsx), 2)
                f_index += 1
                f_names.append(['baroc',''])
                F[:, :, :, f_index] = -Jsx

        # 5. Stokes drift return flow
        # determine RHS terms per submodule - next for water level
        #   Need to place this separate to make sure that stokes term are at the end.
        #   This way they will not be taken into account for velocity equation
        if 'stokes' in self.submodulesToRun:
            # labels and submodules
            hydro0_keys = list(set(self.input.getKeysOf('u0') + self.input.getKeysOf('zeta0')))
            hydro1_keys = list(set(self.input.getKeysOf('u1') + self.input.getKeysOf('zeta1')))

            # retrieve data and make forcing
            for submod1 in hydro0_keys:
                for submod2 in hydro1_keys:
                    u0 = self.input.v('u0', submod1, range(0, jmax+1), [0], range(0, fmax+1))
                    zeta0 = self.input.v('zeta0', submod1, range(0, jmax+1), [0], range(0, fmax+1))
                    u1 = self.input.v('u1', submod2, range(0, jmax+1), [0], range(0, fmax+1))
                    zeta1 = self.input.v('zeta1', submod2, range(0, jmax+1), [0], range(0, fmax+1))

                    gamma = np.zeros([jmax+1, 1, fmax+1], dtype=complex)
                    if u0 is not None and zeta1 is not None:
                        gamma += ny.complexAmplitudeProduct(u0, zeta1, 2)
                    if u1 is not None and zeta0 is not None:
                        gamma += ny.complexAmplitudeProduct(u1, zeta0, 2)
                    gamma = np.concatenate((np.zeros([jmax+1, 1, fmax]), gamma), 2)

                    f_index += 1
                    f_names.append(['stokes', submod1+'0'+'-'+submod2+'1'])
                    JuFirst[:, :, :, f_index] = gamma

        ## Solve equation
        uCoef, uFirst[:, :, :, :nRHSVelocity], uzCoef, uzFirst[:, :, :, :nRHSVelocity], AMatrix = uFunctionMomentumConservative(A, F, Fsurf, Fbed, self.input, hasMatrix=velocityMatrix)
        if not velocityMatrix:
            d['velocityMatrix'] = AMatrix
        del AMatrix

        ################################################################################################################
        # water level
        ################################################################################################################
        ## LHS terms
        #   try to get zetaMatrix from the input. If it is not found, proceed to calculate the matrix again
        B = self.input.v('zetaMatrix')  # known that this is numeric data, ask without arguments to retrieve full ndarray
        zetaMatrix = True
        if B is None:
            zetaMatrix = False
            utemp = uCoef.reshape(uCoef.shape[:2]+(1,)+uCoef.shape[2:])     # reshape as the 'f' dimension is not grid conform; move it to a higher dimension
            JuCoef = ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid'))
            JuCoef = JuCoef.reshape(jmax+1, 1, ftot, ftot)        # reshape back to original grid
            B = -G*JuCoef*self.input.v('B', np.arange(0,jmax+1)).reshape(jmax+1, 1, 1, 1)

        ## RHS terms
        #   advection, no-stress, baroclinic
        utemp = uFirst.reshape(uFirst.shape[:2]+(1,)+uFirst.shape[2:])     # reshape as the 'f' dimension is not grid conform; move it to a higher dimension
        JTemp = ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid'))
        JuFirst += JTemp.reshape(jmax+1, 1, ftot, uFirst.shape[-1])        # reshape back to original grid
        BJuFirst = JuFirst*self.input.v('B', np.arange(0,jmax+1)).reshape(jmax+1, 1, 1, 1)

        #   no open BC forcing & all terms in closed BC
        Fopen = np.zeros([1, 1, ftot, nRHS], dtype=complex)
        Fclosed = np.zeros([1, 1, ftot, nRHS], dtype=complex)
        Fclosed += -JuFirst[jmax, 0, :, :]*self.input.v('B', jmax)

        ## Solve equation
        zetaCoef, zetaxCoef, BMatrix = zetaFunctionMassConservative(B, BJuFirst, Fopen, Fclosed, self.input, hasMatrix=zetaMatrix)
        if not zetaMatrix:
            d['zetaMatrix'] = BMatrix
        del BMatrix
        zetax = ny.eliminateNegativeFourier(zetaxCoef, 2)
        zeta = ny.eliminateNegativeFourier(zetaCoef, 2)

        ################################################################################################################
        # velocity
        ################################################################################################################
        u = np.empty((jmax+1, kmax+1, ftot, nRHS), dtype=uCoef.dtype)
        for j in range(0, jmax+1):
            u[j, :, :, :] = np.dot(uCoef[j, :, :, :], -G*zetaxCoef[j, 0, :, :])
        u += uFirst
        u = ny.eliminateNegativeFourier(u, 2)

        ################################################################################################################
        # Reduce number of components
        ################################################################################################################
        # Select components by their velocity magnitude.
        # This is measured as the 1-norm over z and f and the 2-norm over x
        if maxContributions != 'all':
            for submod in self.submodulesToRun:
                f_names_numbers = zip(f_names, range(0, len(f_names)))      # combine forcing names and its position in the 4th dimension of u and zeta
                f_submod = [f_names_numbers[i] for i in range(0, len(f_names)) if f_names[i][0]==submod]    # take only the forcing component of a particular submodule

                # determine norm and sort the list
                if f_submod:
                    unorm = [np.linalg.norm(np.linalg.norm(u[:,:,:,i], 1 , (1,2)), 2, 0) for i in zip(*f_submod)[1]]
                    sorted_fnames = [list(l) for l in zip(*sorted(zip(unorm, f_submod), key = lambda nrm: nrm[0]))]   # sorted list with entries (unorm, (forcing name, forcing position)) with smallest norm first
                else:
                    unorm = []
                    sorted_fnames = []

                # determine the contributions to be aggregated (redundant positions)
                redundant_positions = [sorted_fnames[1][i][1] for i in range(0, len(unorm)-maxContributions)]
                if len(redundant_positions)>=1:
                    # replace first redundant element by sum and label 'other'
                    first_red_pos = redundant_positions[0]
                    u[:, :, :, first_red_pos] = np.sum(u[:, :, :, redundant_positions], 3)
                    zeta[:, :, :, first_red_pos] = np.sum(zeta[:, :, :, redundant_positions], 3)
                    f_names[first_red_pos][1] = 'other'

                    # remove other redundant positions
                    u = np.delete(u, redundant_positions[1:], 3)
                    zeta = np.delete(zeta, redundant_positions[1:], 3)
                    [f_names.pop(i) for i in sorted(redundant_positions[1:], reverse=True)]

        ################################################################################################################
        # vertical velocity
        ################################################################################################################
        w = self.verticalVelocity(u)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d['zeta2'] = {}
        d['u2'] = {}
        d['w2'] = {}

        for submod in self.submodulesToRun:
            if submod in zip(*f_names)[0]:
                d['zeta2'][submod] = {}
                d['u2'][submod] = {}
                d['w2'][submod] = {}

        for i, submod in enumerate(f_names):
            nf = ny.functionTemplates.NumericalFunctionWrapper(zeta[:, :, :, i], self.input.slice('grid'))
            nf.addDerivative(zetax[:, :, :, i], 'x')
            if submod=='baroc':
                d['zeta2'][submod[0]] = nf.function
                d['u2'][submod[0]] = u[:, :, :, i]
                d['w2'][submod[0]] = w[:, :, :, i]
            else:
                d['zeta2'][submod[0]][submod[1]] = nf.function
                d['u2'][submod[0]][submod[1]] = u[:, :, :, i]
                d['w2'][submod[0]][submod[1]] = w[:, :, :, i]

        return d


    def verticalVelocity(self, u):
        B = self.input.v('B', x=self.input.v('grid', 'axis', 'x')).reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Bx = self.input.d('B', x=self.input.v('grid', 'axis', 'x'), dim='x').reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Hx = self.input.d('H', x=self.input.v('grid', 'axis', 'x'), dim='x').reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Bux = Bx/B*u+ny.derivative(u, 'x', self.input.slice('grid'))
        kmax = self.input.v('grid', 'maxIndex', 'z')

        w = -ny.integrate(Bux, 'z', kmax, np.arange(0, kmax+1), self.input.slice('grid'))-u[:, -1, None, Ellipsis]*Hx
        return w


    def __numberOfForcings(self):
        countVelocity = 0
        count = 0
        if 'adv' in self.submodulesToRun:
            for order1 in range(0, 2):
                order2 = 2-order1-1
                countVelocity += len(self.input.getKeysOf('u'+str(order1)))*len(self.input.getKeysOf('u'+str(order2)))
        if 'nostress' in self.submodulesToRun:
            hydro0_subs = list(set(self.input.getKeysOf('u0') + self.input.getKeysOf('zeta0')))
            hydro1_subs = list(set(self.input.getKeysOf('u1') + self.input.getKeysOf('zeta1')))
            countVelocity += len(hydro0_subs)*len(hydro1_subs)
        if 'baroc' in self.submodulesToRun:
            if self.input.v('s1'):  # only when s(n-1) is available
                countVelocity += max(len(self.input.getKeysOf('s1')),1)
        if 'densitydrift' in self.submodulesToRun:
            # for order1 in range(0, self.currentOrder-1):
            #     order2 = self.currentOrder-order1-2
            #     if self.input.v('s'+str(order1)): # only when s(order1) is available
            #         countVelocity += max(len(self.input.getKeysOf('s'+str(order1))),1)*len(self.input.getKeysOf('zeta'+str(order2)))
            countVelocity += 1
        if 'stokes' in self.submodulesToRun:
            hydro0_subs = list(set(self.input.getKeysOf('u0') + self.input.getKeysOf('zeta0')))
            hydro1_subs = list(set(self.input.getKeysOf('u1') + self.input.getKeysOf('zeta1')))
            count += len(hydro0_subs)*len(hydro1_subs)
        count += countVelocity
        return count, countVelocity

