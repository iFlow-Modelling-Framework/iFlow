#!/usr/bin/python
"""
TurbulenceKEps

Date: 23-Oct-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import kepsmodel as ke
from nifty import polyApproximation
from nifty import complexAmplitudeProduct


class TurbulenceKEps:
    # Variables
    logger = logging.getLogger(__name__)
    TOLERANCE = 10**-4      # maximum abs difference allowed in eddy viscosity for convergence
    RELAXATION = 0.         # Relaxation factor; fraction of old solution to take to next iteration
    LLMAX = 2               # number of loops in k-epsilon model in every iteration step

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input

        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration

        stop = False
        if hasattr(self, 'vicww_difference'):
            self.logger.info('k-epsilon model, difference in last iteration %s' % str(self.vicww_difference))
            if self.vicww_difference < self.TOLERANCE:
                stop = True
                del self.vicww_difference
                self.plot(self.old_vicww)
        return stop

    def run_init(self):
        self.logger.info('Running k-epsilon model - initialising iteration')
        #d = self.debug()
        #return d

        Roughness = self.input.v('z0*')*self.input.v('H')

        fmax = self.input.v('grid', 'maxIndex', 'f')
        z = self.input.v('grid','axis','z')
        z = z.reshape((len(z), 1))
        Av0 = np.asarray([0.075]+[0]*fmax).reshape((1, fmax+1))
        Av = Av0*(self.input.v('z0*')-(-z))*(1+self.input.v('z0*')+(-z))

        d = {}
        d['Av'] = Av
        d['Roughness'] = Roughness


        #self.plot(Av)

        return d

    def run(self):
        self.logger.info('Running k-epsilon model')

        #d = self.debug()
        #return d
        ################################################################################################################
        # Load data
        ################################################################################################################
        # Load parameters from data container
        dt = self.input.v('dt')
        omega = self.input.v('OMEGA')
        kappa = self.input.v('KAPPA')
        alpha = self.input.v('ALPHA')
        cmu = self.input.v('CMU')

        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        tmax = (2*np.pi/omega)/dt                           # number of time steps

        # Load grid and forcing
        z = self.input.v('grid', 'axis', 'z')
        thick = abs(z[:-1]-z[1:])
        dp = self.input.v('H')
        z0 = self.input.v('Roughness')

        uz = self.input.d('u0', range(0, kmax+1), range(0, fmax+1), dim='z')
        ubed = self.input.v('u0', kmax-1, range(0, fmax+1))
        #uz, ubed = self.setuz()
        u_star = ubed*kappa/np.log(alpha+thick[-1]*dp/z0)

        rhoz = np.zeros(uz.shape)
        zeta = np.zeros((1, fmax+1))

        ################################################################################################################
        # Iteration number 1: init rtur1 if it does not exist
        ################################################################################################################
        if not hasattr(self, 'rtur1'):
            uscale = np.sum(abs(u_star))
            u_st_dimless = u_star/uscale
            u_st_dimless_sq = complexAmplitudeProduct(u_st_dimless, u_st_dimless, 0)
            u_st_dimless_four = complexAmplitudeProduct(u_st_dimless_sq, u_st_dimless_sq, 0)

            c = polyApproximation(self.__u_sq_uabs, 4)
            u_star_three = (c[0]+c[2]*u_st_dimless_sq+c[4]*u_st_dimless_four)

            self.rtur1 = np.zeros((1001, 2))
            self.rtur1[:kmax+1, 0] = np.sum( np.real( uscale**2*u_st_dimless_sq.reshape((1,fmax+1))/np.sqrt(cmu)*z.reshape(len(z), 1)), 1)   # make profile of TKE for t=0
            self.rtur1[:kmax+1, 1] = np.sum( np.real( uscale**3*u_star_three.reshape((1,fmax+1))/kappa*1./(1.-z.reshape(len(z),1)+z0) ), 1)   # make profile of TKE for t=0

            # initialise vicww for determining convergence
            self.old_vicww = np.zeros((kmax+1, fmax+1))

        ################################################################################################################
        # All iterations: Call model and post-process
        ################################################################################################################
        # Call k-epsilon model
        vicww, self.rtur1 = ke.kepsmodel(self.rtur1, uz, rhoz, u_star, zeta, thick, tmax, self.LLMAX, omega, dp, z0, self.RELAXATION)

        # remove time dependency if required by input variable 'timeDependent'
        if self.input.v('timeDependent') == 'False':
            vicww[:, 1:] = 0.

        # prepare data for convergence study:
        self.vicww_difference = np.max(np.abs(self.old_vicww-vicww[:kmax+1, :fmax+1]))
        self.old_vicww = vicww[:kmax+1, :fmax+1]

        self.plot(vicww)

        d = {}
        d['Av'] = vicww[:kmax+1, :fmax+1]

        return d

    def __u_sq_uabs(self, u):
        return u**2*np.abs(u)

    def setuz(self):
        import scipy.io
        mat = scipy.io.loadmat('mat.mat')
        return mat['uz'], mat['ubed']

    def debug(self):
        Roughness = self.input.v('z0*')*self.input.v('H')

        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        import scipy.io
        mat = scipy.io.loadmat('mat.mat')

        vicww = mat['nut'][:kmax+1, :fmax+1]
        self.old_vicww = vicww
        d = {}
        d['Av'] = vicww[:kmax+1, :fmax+1]
        d['Roughness'] = Roughness

        #self.plot(vicww)
        return d

    def plot(self, Av):
        import matplotlib.pyplot as plt
        import scipy.io
        import nifty as ny
        matd = scipy.io.loadmat('mat.mat')
        from src.DataContainer import DataContainer
        mat = DataContainer(matd)
        mat.merge(self.input.slice('grid'))

        kmax = self.input.v('grid', 'maxIndex', 'z')
        z = ny.dimensionalAxis(self.input.slice('grid'), 'z')
        zdimless = self.input.v('grid', 'axis', 'z')
        plt.subplot(2, 3, 1)
        plt.hold(True)
        plt.plot(np.abs(Av[:kmax+1, 0]),z,'b')
        plt.plot(np.abs(Av[:kmax+1, 2]),z,'r')
        # debug
        plt.plot(np.abs(mat.v('nut', z=zdimless, f=0)),z,'b--')
        plt.plot(np.abs(mat.v('nut', z=zdimless, f=2)),z,'r--')

        plt.title('$nu_t$')

        plt.subplot(2, 3, 2)
        plt.hold(True)
        try:
            # debug
            plt.plot(np.abs(mat.v('uz', z=zdimless, f=1)),z,'g--')
            plt.plot(np.abs(self.input.d('u0', range(0, kmax+1), 1, dim='z')),z,'g')
        except:
            pass
        plt.title('$u_{z,0}$')

        plt.subplot(2, 3, 3)
        plt.hold(True)
        try:
            plt.plot(self.rtur1[:kmax+1, 0],z)
            tke = mat.v('tke', z=zdimless, f=0)
            plt.plot(tke,z,'--g')
        except:
            pass
        plt.title('$k$')

        plt.subplot(2, 3, 4)
        plt.hold(True)
        try:
            plt.plot(self.rtur1[:kmax+1, 1],z)
            eps = mat.v('eps', z=zdimless, f=0)
            plt.plot(eps,z,'--g')
        except:
            pass
        plt.title('$\epsilon$')

        plt.subplot(2, 3, 5)
        plt.hold(True)
        try:
            plt.plot(np.abs(self.input.d('u0', range(0, kmax+1), 1, dim='z')*Av[:kmax+1, 0]),z,'g')
            plt.plot(np.abs(mat.v('uz', z=zdimless, f=1)*mat.v('nut', z=zdimless, f=0)),z,'--g')
        except:
            pass
        plt.title('$R$')

        plt.subplot(2, 3, 6)
        plt.hold(True)
        try:
            plt.plot(np.abs(self.input.v('u0', range(0, kmax+1), 1)),z,'g-')
            z_u = np.arange(0,1+1./100,1./100.)*self.input.n('H')
            z_cen = 0.5*(z_u[1:]+z_u[:-1])
            plt.plot(np.abs(matd['u'][:,1]), z_cen, '--g')
        except:
            pass
        plt.title('$u_0$')

        # plt.subplot(2, 3, 4)
        # plt.hold(True)
        # try:
        #     plt.plot(np.abs(self.input.d('Av', range(0,kmax+1), 0, dim='z')), z, 'b')
        #     plt.plot(np.abs(self.input.d('Av', range(0,kmax+1), 2, dim='z')), z, 'r')
        #     plt.title('$\nu_{t,z}$')
        # except:
        #     pass

        plt.show()
        return