"""


Date: 29-07-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from zetaFunctionMassConservative import zetaFunctionMassConservative
from uFunctionMomentumConservative import uFunctionMomentumConservative


class HydroFirst:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """

        Returns:
            Dictionary with results. At least contains the variables listed as output in the registry
        """
        self.logger.info('Running module HydroFirst')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        BETA = self.input.v('BETA')
        OMEGA = self.input.v('OMEGA')
        ftot = 2*fmax+1
        submodulesToRun = self.input.v('submodules')

        # check if the river term should be compensated for by reference level. Only if river is on, non-zero and there is no leading-order contribution
        if 'river' in submodulesToRun and self.input.v('Q1')!=0 and not np.any(self.input.v('zeta0', 'river', range(0,jmax+1),0, 0)):
            RiverReferenceCompensation = 1
        else:
            RiverReferenceCompensation = 0

        ################################################################################################################
        # velocity as function of water level
        ################################################################################################################
        ## LHS terms
        #   try to get velocityMatrix from the input. If it is not found, proceed to calculate the matrix again
        A = self.input.v('velocityMatrix')  # known that this is numeric data, ask without arguments to retrieve full ndarray
        velocityMatrix = True
        if A is None:
            A = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            velocityMatrix = False
        else:
            A = A

        ## RHS terms
        #   Determine number/names of right hand side
        submodulesVelocityForcing = [i for i in ['adv', 'nostress', 'baroc', 'mixing', 'river'] if i in submodulesToRun]
        #submodulesVelocityConversion = [i for i, mod in enumerate(submodulesToRun) if mod in ['adv', 'nostress', 'baroc', 'mixing']]
        submodulesVelocityConversion = [submodulesToRun.index(i) for i in submodulesVelocityForcing]
        nRHS = len(submodulesVelocityForcing)
        F = np.zeros([jmax+1, kmax+1, ftot, nRHS], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, nRHS], dtype=complex)
        uFirst = np.zeros([jmax+1, kmax+1, ftot, len(submodulesToRun)], dtype=complex)
        uzFirst = np.zeros([jmax+1, kmax+1, ftot, len(submodulesToRun)], dtype=complex)

        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))

        if RiverReferenceCompensation:    # for reference level variation
            F[:, :, fmax, submodulesVelocityForcing.index('river')] = -G*self.input.d('R', range(0, jmax+1), dim='x').reshape((jmax+1, 1))*np.ones((1,kmax+1))

        if 'adv' in submodulesVelocityForcing:
            u0x = self.input.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
            u0z = self.input.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            w0 = self.input.v('w0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            eta = ny.complexAmplitudeProduct(u0, u0x, 2) + ny.complexAmplitudeProduct(w0, u0z, 2)
            eta = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), eta), 2)
            F[:, :, :, submodulesVelocityForcing.index('adv')] = -eta
        if 'nostress' in submodulesVelocityForcing:
            D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))*np.ones((jmax+1, 1, 1))
            zeta0x = self.input.d('zeta0', range(0, jmax+1), [0], range(0, fmax+1), dim='x')
            chi = D*u0[:, [0], :] + G*zeta0x
            chi = ny.complexAmplitudeProduct(chi, zeta0, 2)
            chi = np.concatenate((np.zeros([jmax+1, 1, fmax]), chi), 2)
            Fsurf[:, :, :, submodulesVelocityForcing.index('nostress')] = -chi
        if 'baroc' in submodulesVelocityForcing:
            sx = self.input.d('s0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
            Jsx = -ny.integrate(sx, 'z', 0, range(0,kmax+1), self.input.slice('grid'))  # integral from z to 0 has its boundaries inverted and has a minus sign to compensate
            Jsx = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), Jsx), 2)
            F[:, :, :, submodulesVelocityForcing.index('baroc')] = -G*BETA*Jsx
        if 'mixing' in submodulesVelocityForcing:
            u0z = self.input.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            Av1 = self.input.v('Av1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            ksi = ny.complexAmplitudeProduct(u0z, Av1, 2)
            ksiz = ny.derivative(ksi, 1, self.input.slice('grid'))

            ksi = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), ksi), 2)
            ksiz = np.concatenate((np.zeros([jmax+1, kmax+1, fmax]), ksiz), 2)
            F[:,:,:,submodulesVelocityForcing.index('mixing')] = ksiz
            Fsurf[:, :, :, submodulesVelocityForcing.index('mixing')] = -ksi[:,[0],Ellipsis]

            ## Removed 14-7-2017 YMD: Roughness1*u0 and Av1*u0z should be equal, so this term cancels
            # if self.input.v('BottomBC') in ['PartialSlip']:
            #     Fbed[:, :, :, submodulesVelocityForcing.index('mixing')] = -ksi[:,[kmax],Ellipsis]
            #     roughness1 = self.input.v('Roughness1', range(0, jmax+1), [0], range(0, fmax+1))
            #     if roughness1 is not None:
            #         ksi = ny.complexAmplitudeProduct(u0[:, [-1], :], roughness1, 2)
            #         ksi = np.concatenate((np.zeros([jmax+1, 1, fmax]), ksi), 2)
            #         Fbed[:, :, :, submodulesVelocityForcing.index('mixing')] = ksi

        ## Solve equation
        uCoef, uFirst[:, :, :, submodulesVelocityConversion], uzCoef, uzFirst[:, :, :, submodulesVelocityConversion], _ = uFunctionMomentumConservative(A, F, Fsurf, Fbed, self.input, hasMatrix=velocityMatrix)

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
        JuFirst = ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid'))
        JuFirst = JuFirst.reshape(jmax+1, 1, ftot, uFirst.shape[-1])        # reshape back to original grid

        #   stokes
        if 'stokes' in submodulesToRun:
            gamma = ny.complexAmplitudeProduct(u0[:, 0, None, Ellipsis], zeta0, 2)
            gamma = np.concatenate((np.zeros([jmax+1, 1, fmax]), gamma), 2)
            JuFirst[:, :, :, submodulesToRun.index('stokes')] = gamma
        BJuFirst = JuFirst*self.input.v('B', np.arange(0,jmax+1)).reshape(jmax+1, 1, 1, 1)

        #   open BC: tide
        Fopen = np.zeros([1, 1, ftot, len(submodulesToRun)], dtype=complex)
        if 'tide' in submodulesToRun:
            Fopen[0, 0, fmax:, submodulesToRun.index('tide')] = ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (fmax+1,))

        #   closed BC: river
        Fclosed = np.zeros([1, 1, ftot, len(submodulesToRun)], dtype=complex)
        if 'river' in submodulesToRun:
            Fclosed[0, 0, fmax, submodulesToRun.index('river')] = -self.input.v('Q1')

        #   closed BC: other terms
        Fclosed += -JuFirst[jmax, 0, :, :]*self.input.v('B', jmax)

        ## Solve equation
        zetaCoef, zetaxCoef, _ = zetaFunctionMassConservative(B, BJuFirst, Fopen, Fclosed, self.input, hasMatrix=zetaMatrix)
        zetax = ny.eliminateNegativeFourier(zetaxCoef, 2)
        zeta = ny.eliminateNegativeFourier(zetaCoef, 2)

        ################################################################################################################
        # velocity
        ################################################################################################################
        u = np.empty((jmax+1, kmax+1, ftot, len(submodulesToRun)), dtype=uCoef.dtype)
        uz = np.empty((jmax+1, kmax+1, ftot, len(submodulesToRun)), dtype=uCoef.dtype)
        for j in range(0, jmax+1):
            u[j, :, :, :] = np.dot(uCoef[j, :, :, :], -G*zetaxCoef[j, 0, :, :])
            uz[j, :, :, :] = np.dot(uzCoef[j, :, :, :], -G*zetaxCoef[j, 0, :, :])
        u += uFirst
        uz += uzFirst
        u = ny.eliminateNegativeFourier(u, 2)
        uz = ny.eliminateNegativeFourier(uz, 2)

        ################################################################################################################
        # vertical velocity
        ################################################################################################################
        w = self.verticalVelocity(u)

        ################################################################################################################
        # Make final dictionary to return
        ################################################################################################################
        d = {}
        d['zeta1'] = {}
        d['u1'] = {}
        d['w1'] = {}
        for i, submod in enumerate(submodulesToRun):
            nf = ny.functionTemplates.NumericalFunctionWrapper(zeta[:, :, :, i], self.input.slice('grid'))
            nf.addDerivative(zetax[:, :, :, i], 'x')
            d['zeta1'][submod] = nf.function

            nfu = ny.functionTemplates.NumericalFunctionWrapper(u[:, :, :, i], self.input.slice('grid'))
            nfu.addDerivative(uz[:, :, :, i], 'z')
            d['u1'][submod] = nfu.function
            d['w1'][submod] = w[:, :, :, i]
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

