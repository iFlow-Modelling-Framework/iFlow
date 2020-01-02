"""


Date: 29-07-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from .zetaFunctionMassConservative import zetaFunctionMassConservative
import scipy.misc
from .uFunctionMomentumConservative import uFunctionMomentumConservative


class HydroHigher:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.currentOrder = self.input.v('order')
        if self.currentOrder < 2:
            return
        # Init
        if self.currentOrder == 2:
            maxOrder = self.input.v('maxOrder')
            jmax = self.input.v('grid', 'maxIndex', 'x')
            fmax = self.input.v('grid', 'maxIndex', 'f')
            OMEGA = self.input.v('OMEGA')
            G = self.input.v('G')
            self.submodulesToRun = self.input.v('submodules')

            # initialise arrays with surface stress and velocity data
            self.surf_stress = np.nan*np.empty((jmax+1, 1, fmax+1, maxOrder, maxOrder), dtype=complex) # x, z, f, order, derivative
            self.surf_u_der = np.nan*np.empty((jmax+1, 1, fmax+1, maxOrder, maxOrder+2), dtype=complex) # x, z, f, order, derivative

            # update those parts of these arrays that are not updated later
            #   surface stress
            D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
            u0 = self.input.v('u0', range(0, jmax+1), [0], range(0, fmax+1))
            zeta0x = self.input.d('zeta0', range(0, jmax+1), [0], range(0, fmax+1), dim='x')
            self.surf_stress[:, :, :, 0, 0] = D*u0[:, [0], :] + G*zeta0x

            #   surface der of u
            self.surf_u_der[:, :, :, 0, 0] = u0
            self.surf_u_der[:, :, :, 0, 1] = self.input.d('u0', range(0, jmax+1), [0], range(0, fmax+1), dim='z')
            Av = self.input.v('Av', range(0, jmax+1), [0], range(0, fmax+1))
            Avz = self.input.d('Av', range(0, jmax+1), [0], range(0, fmax+1), dim='z')
            u0z = self.input.d('u0', range(0, jmax+1), [0], range(0, fmax+1), dim='z')
            self.surf_u_der[:, :, :, 0, 2] = -self.Avinv_multiply(Av, (ny.complexAmplitudeProduct(Avz, u0z, 2) - self.surf_stress[:, :, :, 0, 0]))

        self.logger.info('Running module HydroHigher - order '+str(self.currentOrder))
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        BETA = self.input.v('BETA')
        ftot = 2*fmax+1
        try:
            maxContributions = int(self.input.v('maxContributions'))
        except:
            maxContributions = self.input.v('maxContributions')
        d = {}

        # update surf_stress and surf_u_der
        self.updateSurfaceData()

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
            for order1 in range(0, self.currentOrder):
                order2 = self.currentOrder-order1-1

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
                        del eta


        # 2. No-Stress
        if 'nostress' in self.submodulesToRun:
            chi = 0
            for m in range(1, self.currentOrder+1):
                for k in range(0, self.currentOrder-m+1):
                    zetapermutations = self.multiindex(m, self.currentOrder-m-k)
                    # a. make the zeta^m product
                    zetasum = 0
                    for perm in range(0, zetapermutations.shape[0]):
                        zetaname = 'zeta'+str(zetapermutations[perm, 0])
                        zeta = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                        for comp in range(1, zetapermutations.shape[1]):
                            zetaname = 'zeta'+str(zetapermutations[perm, comp])
                            zeta2 = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                            zeta = ny.complexAmplitudeProduct(zeta, zeta2, 2)
                        zetasum = zetasum + zeta

                    # b. make the (Av*uz)^(m) term (use m-1 as surf_stress contains (Av*uz)_z^(m))
                    Avuz = self.surf_stress[:, :, :, k, m-1]

                    # c. add all to chi
                    chi += 1./np.math.factorial(m)*ny.complexAmplitudeProduct(Avuz, zetasum, 2)

            chi = np.concatenate((np.zeros([jmax+1, 1, fmax]), chi), 2)

            f_index += 1
            f_names.append(['nostress', ''])
            Fsurf[:, :, :, f_index] = -chi
            del chi

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
            s_str = 's'+str(self.currentOrder-1)
            sx = self.input.d(s_str, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
            if sx is not None:
                Jsx = -G*BETA*ny.integrate(sx, 'z', 0, range(0, kmax+1), self.input.slice('grid'))  # integral from z to 0 has its boundaries inverted and has a minus sign to compensate
                Jsx = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), Jsx), 2)
                f_index += 1
                f_names.append(['baroc',''])
                F[:, :, :, f_index] = -Jsx
                del Jsx, sx

        # 5. Mixing
        if 'mixing' in self.submodulesToRun:
            ksi = np.zeros([jmax+1, kmax+1, fmax+1], dtype=complex)
            for m in range(1, self.currentOrder+1):
                Avm = self.input.v('Av'+str(m), range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                if Avm is not None:
                    uz = self.input.d('u'+str(self.currentOrder-m), range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
                    ksi += ny.complexAmplitudeProduct(Avm, uz, 2)

            ksi_z = ny.derivative(ksi, 'z', self.input.slice('grid'))
            ksi = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), ksi), 2)
            ksi_z = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), ksi_z), 2)

            f_index += 1
            f_names.append(['mixing', 'general'])
            F[:, :, :, f_index] = ksi_z
            Fsurf[:, :, :, f_index] = -ksi[:, [0], :]
            if self.input.v('BottomBC') in ['PartialSlip']:
                Fbed[:, :, :, f_index] = -ksi[:, [-1], :]

                # 5.b higher order no-slip coefficient
                for m in range(1, self.currentOrder+1):
                    roughness = self.input.v('Roughness'+str(m), range(0, jmax+1), [0], range(0, fmax+1))
                    if roughness is not None:
                        u_str1 = 'u'+str(self.currentOrder-m)
                        u = self.input.v(u_str1, range(0, jmax+1), [kmax], range(0, fmax+1))
                        ksi = ny.complexAmplitudeProduct(u, roughness, 2)
                        ksi = np.concatenate((np.zeros([jmax+1, 1, fmax]), ksi), 2)
                        Fbed[:, :, :, f_index] = ksi

        # 5.b Mixing-no-stress interaction
            ksi = np.zeros([jmax+1, 1, fmax+1], dtype=complex)
            for m in range(1, self.currentOrder+1):
                for k in range(0, self.currentOrder-m+1):
                    for i in range(1, self.currentOrder-m-k+1):
                        zetapermutations = self.multiindex(m, self.currentOrder-m-k-i)
                        # a. make the zeta^m product
                        zetasum = 0
                        for perm in range(0, zetapermutations.shape[0]):
                            zetaname = 'zeta'+str(zetapermutations[perm, 0])
                            zeta = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                            for comp in range(1, zetapermutations.shape[1]):
                                zetaname = 'zeta'+str(zetapermutations[perm, comp])
                                zeta2 = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                                zeta = ny.complexAmplitudeProduct(zeta, zeta2, 2)
                            zetasum = zetasum + zeta

                        # b. make the (Av*uz)^(m) term
                        Avuz = 0
                        for j in range(0, m+1):
                            if j == 0:
                                Avder = self.input.v('Av'+str(i), range(0, jmax+1), [0], range(0, fmax+1))
                            else:
                                Avder = self.input.d('Av'+str(i), range(0, jmax+1), [0], range(0, fmax+1), dim='z'*j)
                            if Avder is not None:
                                Avuz += scipy.misc.comb(m, j)*ny.complexAmplitudeProduct(Avder, self.surf_u_der[:, :, :, k, m-j+1], 2)  # use m-j+1 as we need u_z^(m-j)

                        # c. add all to ksi
                        if not isinstance(Avuz, int):
                            ksi += 1./np.math.factorial(m)*ny.complexAmplitudeProduct(Avuz, zetasum, 2)

            ksi = np.concatenate((np.zeros([jmax+1, 1, fmax]), ksi), 2)

            f_index += 1
            f_names.append(['mixing', 'no-stress'])
            Fsurf[:, :, :, f_index] = -ksi

        # 6. Stokes drift return flow
        # determine RHS terms per submodule - next for water level
        #   Need to place this separate to make sure that stokes term are at the end.
        #   This way they will not be taken into account for velocity equation
        if 'stokes' in self.submodulesToRun:
            gamma = 0
            for m in range(1, self.currentOrder+1):
                for k in range(0, self.currentOrder-m+1):
                    zetapermutations = self.multiindex(m, self.currentOrder-m-k)
                    # a. make the zeta^m product
                    zetasum = 0
                    for perm in range(0, zetapermutations.shape[0]):
                        zetaname = 'zeta'+str(zetapermutations[perm, 0])
                        zeta = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                        for comp in range(1, zetapermutations.shape[1]):
                            zetaname = 'zeta'+str(zetapermutations[perm, comp])
                            zeta2 = self.input.v(zetaname, range(0, jmax+1), [0], range(0, fmax+1))
                            zeta = ny.complexAmplitudeProduct(zeta, zeta2, 2)
                        zetasum = zetasum + zeta

                    # b. make the (u)^(m-1) term
                    umz = self.surf_u_der[:, :, :, k, m-1]

                    # c. add all to chi
                    gamma += 1./np.math.factorial(m)*ny.complexAmplitudeProduct(umz, zetasum, 2)

            gamma = np.concatenate((np.zeros([jmax+1, 1, fmax]), gamma), 2)

            f_index += 1
            f_names.append(['stokes', ''])
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
        d['zeta'+str(self.currentOrder)] = {}
        d['u'+str(self.currentOrder)] = {}
        d['w'+str(self.currentOrder)] = {}

        for submod in self.submodulesToRun:
            if submod in zip(*f_names)[0]:
                d['zeta'+str(self.currentOrder)][submod] = {}
                d['u'+str(self.currentOrder)][submod] = {}
                d['w'+str(self.currentOrder)][submod] = {}

        for i, submod in enumerate(f_names):
            nf = ny.functionTemplates.NumericalFunctionWrapper(zeta[:, :, :, i], self.input.slice('grid'))
            nf.addDerivative(zetax[:, :, :, i], 'x')
            if submod=='baroc':
                d['zeta'+str(self.currentOrder)][submod[0]] = nf.function
                d['u'+str(self.currentOrder)][submod[0]] = u[:, :, :, i]
                d['w'+str(self.currentOrder)][submod[0]] = w[:, :, :, i]
            else:
                d['zeta'+str(self.currentOrder)][submod[0]][submod[1]] = nf.function
                d['u'+str(self.currentOrder)][submod[0]][submod[1]] = u[:, :, :, i]
                d['w'+str(self.currentOrder)][submod[0]][submod[1]] = w[:, :, :, i]

        d['surfder'] = self.surf_u_der
        d['surfstress'] = self.surf_stress
        return d


    def verticalVelocity(self, u):
        x = self.input.v('grid', 'axis', 'x')
        B = self.input.v('B', x=x).reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Bx = self.input.d('B', x=x, dim='x').reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Hx = self.input.d('H', x=x, dim='x').reshape([u.shape[0]]+[1]*(len(u.shape)-1))
        Bux = Bx/B*u+ny.derivative(u, 'x', self.input.slice('grid'))
        kmax = self.input.v('grid', 'maxIndex', 'z')

        w = -ny.integrate(Bux, 'z', kmax, np.arange(0, kmax+1), self.input.slice('grid'))-u[:, -1, None, Ellipsis]*Hx
        return w


    def __numberOfForcings(self):
        countVelocity = 0
        count = 0
        if 'adv' in self.submodulesToRun:
            for order1 in range(0, self.currentOrder):
                order2 = self.currentOrder-order1-1
                countVelocity += len(self.input.getKeysOf('u'+str(order1)))*len(self.input.getKeysOf('u'+str(order2)))
        if 'nostress' in self.submodulesToRun:
            countVelocity += 1
        if 'baroc' in self.submodulesToRun:
            if self.input.v('s'+str(self.currentOrder-1)):  # only when s(n-1) is available
                countVelocity += max(len(self.input.getKeysOf('s'+str(self.currentOrder-1))),1)
        #if 'densitydrift' in self.submodulesToRun:
            # countVelocity += 1 #TODO
        if 'mixing' in self.submodulesToRun:
            countVelocity += 2
        if 'stokes' in self.submodulesToRun:
            count += 1
        count += countVelocity
        return count, countVelocity

    def multiindex(self, numberOfComponents, sum):
        """Recursively builds a set of all permutations of 'numberOfComponents' non-negative integers that adds up to sum
         The result is an array with numberOfComponents columns, corresponding to the value of each component. Each row is a different permutation

         Parameters:
            numberOfComponents (int > 0)
            sum (int >=0)
        """
        if numberOfComponents==1:
            l_list = np.asarray(sum).reshape((1,1))

        else:
            l_list = []
            for i in range(0, sum+1):
                if sum-i >= 0 and numberOfComponents-1>0:
                    l_list_reduced = self.multiindex(numberOfComponents-1, sum-i)
                    first_element = np.ones((l_list_reduced.shape[0], 1), dtype=np.int8)*i
                    l_list_reduced = np.append(first_element, l_list_reduced, 1)
                    if l_list==[]:
                        l_list = l_list_reduced
                    else:
                        l_list = np.append(l_list, l_list_reduced, 0)
        return l_list


    def surfaceForcingDer(self, order, der):
        """Calculate forcing F in equation
        u_t^<order>(der)-( Av u_z^<order> )^(der+1) = F^<order>(der)    at surface z=0
        """
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        forcing = np.zeros((jmax+1, 1, fmax+1), dtype=complex)

        # forcing from barotropic pressure gradient
        if der == 0:
            forcing += self.input.v('G')*self.input.d('zeta'+str(order), range(0, jmax+1), [0], range(0, fmax+1), dim='x')

        #  forcing by advection; only include this in the reconstruction if the equation at 'order' contained advection
        if 'adv' in self.input.getKeysOf('u'+str(order)):
            for order1 in range(0, order):
                order2 = order-order1-1
                for i in range(0, der+1):
                    u = self.surf_u_der[:, :, :, order1, i]
                    ux = ny.derivative(self.surf_u_der[:, :, :, order2, der-i], 'x', self.input.slice('grid'))
                    forcing += scipy.misc.comb(der, i)*ny.complexAmplitudeProduct(u, ux, 2)

                    if i == 0:
                        wstr = 'w'+str(order1)
                        w = self.input.v(wstr, range(0, jmax+1), [0], range(0, fmax+1))
                    else:
                        Bu = self.input.v('B', range(0, jmax+1)).reshape((jmax+1, 1, 1))*self.surf_u_der[:, :, :, order1, i-1]
                        w = ny.derivative(Bu, 'x', self.input.slice('grid'))/self.input.v('B', range(0, jmax+1)).reshape((jmax+1, 1, 1))
                    uz = self.surf_u_der[:, :, :, order2, der-i+1]
                    forcing += scipy.misc.comb(der, i)*ny.complexAmplitudeProduct(w, uz, 2)

        # TODO
        #if 'baroc' in self.submodulesToRun:
        #if 'densitydrift' in self.submodulesToRun:

        return forcing


    def updateSurfaceData(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        OMEGA = self.input.v('OMEGA')

        # 1. update velocity and its derivative of previous order
        ustr = 'u'+str(self.currentOrder-1)
        self.surf_u_der[:, :, :, self.currentOrder-1, 0] = self.input.v(ustr, range(jmax+1), [0], range(fmax+1))
        self.surf_u_der[:, :, :, self.currentOrder-1, 1] = self.input.d(ustr, range(jmax+1), [0], range(fmax+1), dim='z')

        # 2. calculate other terms
        for order in range(0, self.currentOrder):
            der = self.currentOrder - order

            # 2a. update surface stress term
            forcing = self.surfaceForcingDer(order, der-1)
            D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
            self.surf_stress[:, :, :, order, der-1] = D*self.surf_u_der[:, :, :, order, der-1] + forcing

            # 2b. update surface velocity derivative
            sum_nu_u = np.zeros((jmax+1, 1, fmax+1), dtype=complex)
            for i in range(1, der+1):
                if i==1:
                    Avder = self.input.d('Av', range(0, jmax+1), [0], range(0, fmax+1), dim='z')     # first der of Av
                elif i==2:
                    Av = self.input.d('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='zz')
                    Avder = Av[:, [0], :]                                                           # 2nd der of Av
                else:
                    Av = ny.derivative(Av, 'z', self.input.slice('grid'))
                    Avder = Av[:, [0], :]                                                           # every step: take a higher der of Av

                sum_nu_u += scipy.misc.comb(der, i)*ny.complexAmplitudeProduct(Avder, self.surf_u_der[:, :, :, order, der+1-i], 2)

            Av = self.input.v('Av', range(0, jmax+1), [0], range(0, fmax+1))
            self.surf_u_der[:, :, :, order, der+1] = -self.Avinv_multiply(Av, sum_nu_u - self.surf_stress[:, :, :, order, der-1])
        return

    def Avinv_multiply(self, Av, b):
        """ Solve Av u = b
        """
        # make the full matrix for Av
        fmax = Av.shape[2]-1
        ftot = 2*fmax+1
        AvMatrix = np.zeros(Av.shape[:2] + (ftot, ftot,), dtype=complex)
        AvMatrix[:, :, range(0, ftot), range(0, ftot)] = Av[:, :, [0]]*np.ones((1, 1, ftot))
        for n in range(1, fmax+1):
            AvMatrix[:, :, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(Av[:, :, [n]])*np.ones((1, 1, ftot-n))
            AvMatrix[:, :, range(n, ftot), range(0, ftot-n)] = 0.5*Av[:, :, [n]]*np.ones((1, 1, ftot-n))

        # extend the rhs with negative components
        b = np.concatenate((np.zeros(Av.shape[:2] + (fmax,)), b), 2)

        ufull = np.linalg.solve(AvMatrix, b)
        u = np.zeros(Av.shape, dtype=complex)
        u[:, :, 0] = ufull[:, :, fmax]
        for n in range(1, fmax+1):
            u[:, :, n] = ufull[:, :, fmax+n] + np.conj(ufull[:, :, fmax-n])

        return u



