"""
Original date: 24-04-15
Update: 15-02-2022
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from .util.uFunctionMomentumConservative import uFunctionMomentumConservative
from .util.zetaFunctionMassConservative import zetaFunctionMassConservative


class HydroLeadNumerical:
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
        self.logger.info('Running module HydroLead')

        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        G = self.input.v('G')
        ftot = 2*fmax+1
        submodulesToRun = self.input.v('submodules')

        # check if the river term should be compensated for by reference level. Only if river is on and non-zero
        if 'river' in submodulesToRun and self.input.v('Q0')!=0:
            RiverReferenceCompensation = 1
        else:
            RiverReferenceCompensation = 0

        ################################################################################################################
        # velocity as function of water level
        ################################################################################################################
        # build, save and solve the velocity matrices in every water column
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        F = np.zeros([jmax+1, kmax+1, ftot, RiverReferenceCompensation])
        Fsurf = np.zeros([jmax+1, 1, ftot, RiverReferenceCompensation])
        Fbed = np.zeros([jmax+1, 1, ftot, RiverReferenceCompensation])

        if RiverReferenceCompensation:    # for reference level variation
            F[:, :, fmax, 0] = -G*self.input.d('R', range(0, jmax+1), dim='x').reshape((jmax+1, 1))*np.ones((1,kmax+1))

        uCoef, uLead, uzCoef, uzLead, velocityMatrix = uFunctionMomentumConservative(Av, F, Fsurf, Fbed, self.input)

        ################################################################################################################
        # water level
        ################################################################################################################
        ## LHS terms
        utemp = uCoef.reshape(uCoef.shape[:2]+(1,)+uCoef.shape[2:])     # reshape as the 'f' dimension is not grid conform; move it to a higher dimension
        JuCoef = ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid'))
        JuCoef = JuCoef.reshape(jmax+1, 1, ftot, ftot)                  # reshape back to original grid
        BJuCoef = -G*JuCoef*self.input.v('B', np.arange(0, jmax+1)).reshape(jmax+1, 1, 1, 1)

        #   open BC: tide
        Fopen = np.zeros([1, 1, ftot, len(submodulesToRun)], dtype=complex)
        if 'tide' in submodulesToRun:
            Fopen[0, 0, fmax:, submodulesToRun.index('tide')] = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (fmax+1,))

        #   closed BC: river
        Fclosed = np.zeros([1, 1, ftot, len(submodulesToRun)], dtype=complex)
        if RiverReferenceCompensation:
            Fclosed[0, 0, fmax, submodulesToRun.index('river')] = -self.input.v('Q0')

        ## RHS terms
        IntForce = np.zeros([jmax+1, 1, ftot, len(submodulesToRun)], dtype=complex)
        if RiverReferenceCompensation:
            utemp = uLead.reshape(uLead.shape[:2]+(1,)+uLead.shape[2:])     # reshape as the 'f' dimension is not grid conform; move it to a higher dimension
            JuLead = ny.integrate(utemp, 'z', kmax, 0, self.input.slice('grid'))
            JuLead = JuLead.reshape(jmax+1, 1, ftot)*self.input.v('B', np.arange(0, jmax+1)).reshape(jmax+1, 1, 1)      # reshape back to original grid
            IntForce[:, :, :,submodulesToRun.index('river')] = JuLead[:,:,:]
            Fclosed[0, 0, :,submodulesToRun.index('river')] += -JuLead[jmax, 0, :]

        ## Solve equation
        zetaCoef, zetaxCoef, zetaMatrix = zetaFunctionMassConservative(BJuCoef, IntForce, Fopen, Fclosed, self.input)
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
        if RiverReferenceCompensation:
            u[:,:,:,submodulesToRun.index('river')] += uLead[:,:,:,0]
            uz[:,:,:,submodulesToRun.index('river')] += uzLead[:,:,:,0]
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
        d['velocityMatrix'] = velocityMatrix
        d['zetaMatrix'] = zetaMatrix
        d['zeta0'] = {}
        d['u0'] = {}
        d['w0'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['x']['zeta0'] = {}
        d['__derivative']['z']['u0'] = {}
        for i, submod in enumerate(submodulesToRun):
            d['zeta0'][submod] = zeta[:, :, :, i]
            d['__derivative']['x']['zeta0'][submod] = zetax[:, :, :, i]

            d['u0'][submod] = u[:, :, : ,i]
            d['__derivative']['z']['u0'][submod] = uz[:, :, :, i]
            d['w0'][submod] = w[:, :, :, i]

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



