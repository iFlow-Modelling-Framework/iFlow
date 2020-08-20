"""
Flocculation module

Date: 17-Aug-20
Authors: D.M.L. Horemans
"""
import numpy as np
import nifty as ny
import logging
from datetime import datetime
from modulesJPO50.savitzky_golay import savitzky_golay
from modulesJPO50.ws1ContributionNonlinearInC import ws1ContributionNonlinearInC
import math

class Flocculation:
    logger = logging.getLogger(__name__)
    # Parameters required for Picard iterative procedure
    TOLLERANCE = 0.005
    RELAX = 0.5  # percentage of old ws

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        stop = False
        if self.input.v('NaNProblem') == 'True':
            stop = True
            print 'NaN generated: proceeding to next parameter set in Sensitivity module'

        if hasattr(self, 'difference'):
            self.logger.info('\t'+str(self.difference))
            if self.difference < self.TOLLERANCE*(1-self.RELAX):
                stop = True
        return stop

    def run_init(self):
        self.logger.info('Running module Flocculation - init')
        d = {}
        self.timeInit = datetime.now()
        self.converging = 'True'
        self.difference = np.inf
        self.ws00Try = 0

        d['ws0'] = self.input.v('ws00')
        d['ws1'] = self.input.v('ws10')

        return d

    def shearRate(self, zz):
        H = self.input.v('H', x=self.input.v('grid', 'axis', 'x'))
        x = self.input.v('grid', 'axis', 'x')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        kappa = self.input.v('KAPPA') # Karman constant
        nu = 1.e-6 # Dynamic water viscocity
        z0 = self.input.v('Roughness').im_self.data.data['coef'][0, 0]

        # Calculate M0 from abs(u0). u0 has M2 --> abs(u0) results in M0. Add M0 of u1_river.
        u0 = self.input.v('u0', x=x, z=1-zz, f=range(0, fmax + 1))
        u1 = self.input.v('u1', 'river', x=x, z=1-zz, f=0)
        ub = ny.absoluteU(u0[:, 1]+u1, 0).reshape((jmax + 1))

        # Calculate shear rate following Chao Guo et al., 2017
        G = np.sqrt(np.power(ub, 3) * np.power(kappa / (np.log(zz*H / z0)), 3.) * (1 - zz) / (H * nu * kappa * zz))

        return G

    def run(self):
        self.logger.info('Running module flocculation')
        d = {}
        ################################################################################################################
        # Load data
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        a = 1

        ################################################################################################################
        # Computations
        ################################################################################################################

        # Define constants
        kA = self.input.v('kA')
        kB = self.input.v('kB')
        g = self.input.v('G') # gravitational accelaration
        mu = self.input.v('mu') # dynamic viscocity
        rho0 = self.input.v('RHO0')
        rhos = self.input.v('RHOS')
        fs = self.input.v('fs') #shape facor, if sperical --> pi/6
        G = self.input.v('Gs') # shear rate
        Dp = self.input.v('Dp') # primary particle size
        scaleShearRate = self.input.v('scaleShearRate')
        skip_first = self.input.v('skip_first')

        # Calculate shear rate G
        G = np.ones(jmax+1)*G
        if scaleShearRate == 'True':
            G = self.shearRate(0.5)
        G = G.repeat((kmax+1) * (fmax+1)).reshape(jmax+1, kmax+1, fmax+1)

        # Calculate impact salinity on kA
        xx = self.input.v('grid', 'axis', 'x') * self.input.v('L')

        fSal = 1
        kAmin = 3/2.*kA
        if self.input.v('kASal') == 'True':
            s00 = self.input.v('s0', range(0, len(xx)), 0, 0, dim='x').reshape(len(xx), 1)
            kAmin = 3/2.*0.2908872
            aPar = 4.08532
            cPar = 0.03401

            fSal = (1 + cPar / kAmin * (np.tanh(s00 - aPar) + 1)).reshape(jmax + 1)

        kA = 2/3.*kAmin*fSal
        kA = kA.repeat((kmax+1) * (fmax+1)).reshape(jmax+1, kmax+1, fmax+1)

        # Compute beta, tau and minimal settling veloccity wsmin
        beta = kA*g*(rhos-rho0)*np.power(G, -1./2.)/(kB*18*mu*rhos*fs)
        tau = (Dp ** 2 * g ** 2 * (-rho0 + rhos) ** 2) / (324. * G[:, :, 0] ** 1.5 * kB * mu ** 2)
        wsmin = (rhos-rho0)*g*math.pow(Dp,2.)/(18*mu) # primary particle size particle, equals beta * kappa

        # Obtain estimated concentration from previous iteration
        c0 = self.input.v('c0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1 = self.input.v('c1', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
        c2 = self.input.v('c2', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        # 1. Compute the leading order settling velocity ws0

        ws0 = a*beta*(c0
                      + int(self.input.v('includec2') == 'True')*c2)+wsmin

        if self.input.v('spatial') == 'True':
            ws0[:, :, 2] = 0

        # 2. Compute the first order settling velocity ws1

        if skip_first != 'True':
            # Define constants alpha, Av, delta and kappa
            Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

            # 2a. compute ws1 contribution which linearly scales to the suspended sediment concentration

            ws1 = beta * c1

            # 2b. compute ws1 contribution which does not linearly scales to the suspended sediment concentration and
            # is due to inertia effects, settling and vertical diffusion.

            # Compute derivative of c00 and c02 which is required by ws1ContributionNonlinearInC()
            dzc00 = ny.derivative(c0, 'z', self.input.slice('grid'))[:, :, 0]
            dzc02 = ny.derivative(c0, 'z', self.input.slice('grid'))[:, :, 2]

            # Compute ws1 contribution which does not linearly scale to the suspended sediment concentration using
            # ws1ContributionNonlinearInC()
            wsAdditional = ws1ContributionNonlinearInC(A0=c0[:, :, 0], dA0=dzc00[:, :], A2=c0[:, :, 2],
                                 dA2=dzc02[:, :], beta=beta[:, :, 0], Kv=Av[:, :, 0], tau=tau)

            # Add wsAdditional to ws1 which is due to the contribution which linearly scales to the suspended sediment
            # concentration
            ws1[:, :, 0] = ws1[:, :, 0] + wsAdditional[0]

            if self.input.v('spatial') != 'True':
                ws1[:, :, 2] = ws1[:, :, 2] + wsAdditional[1]
            else:
                 ws1[:, :, 1:] = 0
        else:
            ws1 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=int)
            c1 = self.input.v('c1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            print str('skipping first order ws')

        # Smooth settling velocity in longitudinal direction
        if self.input.v('smooth') == 'True':
            ord = 1 # order of smoothing
            xstart = 0 # start smoothing from longitudinal x-axis value.

            # Smooth ws0 over longitudinal x-axis for each depth zi and for f=0:2. Exclude upstream boundary point.
            for zi in range(0, kmax+1):
                for kj in range(0, 3):
                    ws0[xstart:-1, zi, kj] = savitzky_golay(ws0[xstart:-1, zi, kj], window_size=15, order=ord)
                    ws1[xstart:-1, zi, kj] = savitzky_golay(ws1[xstart:-1, zi, kj], window_size=15, order=ord)

        # At the upstream border, remove temporal fluctuations in ws1
        ws1[-1,:,1] = 0.
        ws1[-1, :, 2] = 0.

        ################################################################################################################
        # Iterative procedure
        ################################################################################################################

        # Check convergence criterion of Picard iterative procedure. If the convergence criterion is reached, stop
        # Picard iterative procedure.
        ws_old = self.input.v('ws0', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
        ws1_old = self.input.v('ws1', range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))

        self.difference = np.linalg.norm(
            np.sum(np.abs((ws_old[:, :, 0] - ws0[:, :, 0]) / (ws_old[:, :, 0] + 0.1 * self.input.v('ws00'))),
                   axis=-1), np.inf)
        print self.difference

        # Check for nan values, if nan are present, change 'NanProblem' to True which stops the Picard iterative
        # procedure. To avoid problems in subsequent modules, set ws0 and ws1 to initial values.
        if ~np.all(np.isfinite(ws0)) or ~np.all(np.isfinite(ws0)):
            self.difference = 0
            d['NaNProblem'] = 'True'
            ws0 = ws0 * 0 + self.input.v('ws00')
            ws1 = ws1 * 0
            print 'ws1 or ws0 is not finite, calling stopping_criterion'

        # Check whether timeLimit is reached, if true, stop the Picard iterative procedure or use a different initial
        # value for ws00 and ws10
        if (datetime.now() - self.timeInit).seconds > self.input.v('timeLimit'):
            if self.input.v('ws00Skip') == None or self.ws00Try == len(self.input.v('ws00Skip')):
                self.difference = 0  # this will stop the Picard method
                d['picardConverged'] = 'False'
                print 'Picard method did not converge'
            else:
                d['ws0'] = self.input.v('ws00Skip')[self.ws00Try]
                self.timeInit = datetime.now()
                print 'Picard method did not converge: trying next initial ws00 = ' + str(
                    self.input.v('ws00Skip')[self.ws00Try])
                self.ws00Try += 1

        # Save output of flocculation module
        d['ws1'] = 0.2*ws1 + 0.8*ws1_old
        d['ws0'] = 0.2*ws0 + 0.8*ws_old
        d['picardConverged'] = self.converging

        return d
