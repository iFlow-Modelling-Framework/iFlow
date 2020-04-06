"""

Turbulence scheme based on the KEFitted models, with turbulence damping according to Munk & Anderson-like damping functions and a reduction of the bed roughness as in e.g. in Wang (2005)
This code relies on KEFittedTructated for its computation and then combines the result with damping functions.

Options on input
    - as in KEFittedTruncated
    - turbulence damping effect on the bed roughness may be turned off by setting Ribedmax = 0 on input

Iterative procedure:
    First allow KEFittedTructated to converge, then add damping functions and converge

From version: 2.6
Date: December 2018
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from packages.numerical2DV.turbulence.KEFittedTruncated import KEFittedTruncated
import copy
import logging
from packages.analytical2DV.turbulence.profiles import UniformXF


class KEFittedMAW:
    # Variables
    TOLLERANCE = 1.e-3      # relative convergence criterion for Av
    RELAX = 0.8             # relaxation value; as RELAX*old estimate + (1-RELAX)*new estimate
    invfft_length = 90
    LOCAL = 0.5
    filterlength = 15

    cmax = []
    ETMloc = []
    logger = logging.getLogger(__name__)
    timers = [ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer()]

    # Methods
    def __init__(self, input):
        self.input = input
        self.kem = KEFittedTruncated(self.input)
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration

        ## Check convergence
        if self.difference < self.TOLLERANCE*(1-self.RELAX):# or self.iteration>10:
            if self.betac == 0 and not self.input.v('BETAC') == 0:
                self.betac = self.input.v('BETAC')
                self.logger.info('\tMAW starting to include density differences, rel. difference may go up.')
                return False
            else:
                return True
        else:
            return False

    def run_init(self):
        self.logger.info('Running MAW turbulence model - init')
        self.difference = np.inf

        d = {}

        dkem = self.kem.run_init()          # initialise KE model
        self.kem.referenceLevel = False     # only let KE model compute the initial reference level
        self.betac = self.input.v('BETAC')  # set density dependency parameter for sediment
        self.referenceLevel = self.input.v('referenceLevel')    # include computation of reference level in computation (boolean)

        Av = self.input.v('Av')
        Kv = self.input.v('Kv')
        Ri = self.input.v('Ri')
        Roughness = self.input.v('Roughness')
        skin_friction = self.input.v('skin_friction')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        # Initialisation uses data that might be available, or starts clean
        if Av is None or Roughness is None:
            d.update(dkem)                      # use input from initial k-eps fitted run - contains Av and Roughness
            self.input.merge(d)

        if Ri is None or skin_friction is None:
            self.betac = 0                      # start without density effect, first allow KEFitted to converge
            self.Rida = 0.

            # initially set skin friction equal to roughness
            data = self.input.slice('grid')
            data.addData('coef', self.input.v('Roughness', range(0, jmax+1), 0, range(0, fmax+1)))
            ss = UniformXF(['x', 'f'], data, 0.).function
            d['skin_friction'] = ss


        if Kv is None:
            # make eddy diffusivity
            Av = self.input.v('Av', range(0, jmax+1), 0, range(0, fmax+1))
            data = self.input.slice('grid')
            data.addData('coef', Av/self.input.v('sigma_rho', range(0, jmax+1), 0, [0]))
            Kv = UniformXF(['x', 'f'], data, 0.).function
            d['Kv'] = Kv

        # all first-order turbulence quantities are set to zero
        d['Kv1'] = 0.
        d['Av1'] = 0.
        d['Roughness1'] = 0.
        d['skin_friction1'] = 0.
        d['BottomBC'] = 'PartialSlip'

        return d

    def run(self):
        ################################################################################################################
        ## 1. Init
        ################################################################################################################
        # self.timers[0].tic()

        ## prepare output message
        self.logger.info('Running MAW turbulence model')
        denstr = ''
        if self.betac ==0:
            denstr = '- not including density effects'
        self.logger.info('\tMAW rel. difference in Av in last iteration: %s %s' % (self.difference, denstr))

        d = {}

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        G = self.input.v('G')
        rho0 = self.input.v('RHO0')
        uzmin = self.input.v('uzmin')

        Avold = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kvold = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        sfold = self.input.v('Roughness', range(0, jmax+1), 0, 0)
        # self.timers[0].toc()

        ################################################################################################################
        ## 2. KEFitted run
        ################################################################################################################
        # self.timers[1].tic()
        d.update(self.kem.run())
        self.input.merge(d)

        # load data resulting from KEFitted model
        Avmid = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kvmid = Avmid/self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])
        sfmid = self.input.v('Roughness', range(0, jmax+1), 0, 0)
        # self.timers[1].toc()

        ################################################################################################################
        ## 3. Density effects
        ################################################################################################################
        if self.betac == 0:     # no density effect included, first let KEFitted spin up
            Av0 = Avmid[:, :, 0]
            Kv0 = Kvmid[:, :, 0]
            sf = sfmid

            Cd = 1.
            MA_av = 1.
            MA_kv = 1.
            Ri = 0.
        else:
            ## Load data
            # self.timers[2].tic()
            cz = self.input.d('c0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z') + self.input.d('c1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z') + self.input.d('c2', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            uz = self.input.d('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z') + self.input.d('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='z')
            zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
            H = self.input.v('grid', 'low', 'z', range(0, jmax+1)) - self.input.v('grid', 'high', 'z', range(0, jmax+1))
            # self.timers[2].tic()

            ## Convert to time series
            # self.timers[3].tic()
            cz = ny.invfft2(cz, 2, 90)
            uz = ny.invfft2(uz, 2, 90)
            zeta0 = ny.invfft2(zeta0, 2, 90)
            Avmid = ny.invfft2(Avmid, 2, 90)
            Kvmid = ny.invfft2(Kvmid, 2, 90)
            # self.timers[3].toc()
            # self.timers[4].tic()
            uzmin = np.ones(uz.shape)*uzmin

            ## Compute Richardson number
            Ri = -G*self.betac/rho0*cz/(uz**2+uzmin**2)

            Rida = 1./H.reshape((jmax+1, 1, 1))*ny.integrate(Ri.reshape((jmax+1, kmax+1, 1, Ri.shape[-1])), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, Ri.shape[-1]))  # depth-average
            Rida += zeta0*(Ri[:, [0], :]-Rida)/H.reshape((jmax+1, 1, 1))        # depth average continued
            Rida0 = np.maximum(Rida, 0.)                                        # only accept positive Ri
            Ribed = np.maximum(Ri[:, [-1], :], 0.)                              # only accept positive Ri
            Ribedmax = self.input.v('Ribedmax')
            Ribed = np.minimum(Ribed, Ribedmax)                                 # 5-3-2018 limit near-bed Ri

            ## relaxation on Rida
            if hasattr(self, 'Rida'):       # Relaxation of Ri_da using previously saved value
                dRida = self.Rida - Rida0
                Rida = np.max((Rida0, np.min((dRida * (1 - self.RELAX), dRida * (1. + self.RELAX)), axis=0) + Rida0), axis=0)
                Rida = np.min((Rida,   np.max((dRida * (1 - self.RELAX), dRida * (1. + self.RELAX)), axis=0) + Rida0), axis=0)
                self.Rida = Rida
            else:                           # No value saved if the init found an available Ri. Then start with full signal computed here (do not use saved value, as this has been truncated to frequency components)
                Rida = Rida0

            # self.timers[4].toc()

            ## Compute damping functions
            # self.timers[5].tic()
            # Av
            MA_av = (1+10*Rida)**(-0.5)
            dAv = ny.fft(Avmid*MA_av, 2)[:, :, :fmax+1] - Avold
            Av = Avold + (1-self.RELAX)*(self.LOCAL*dAv + .5*(1-self.LOCAL)*dAv[[0]+list(range(0, jmax)), :, :] + .5*(1-self.LOCAL)*dAv[list(range(1, jmax+1))+[jmax], :, :])
            # Av = Avold + dAv
            Av0 = Av[:, :, 0]

            # Kv
            MA_kv = (1+3.33*Rida)**(-1.5)
            dKv = ny.fft(Kvmid*MA_kv, 2)[:, :, :fmax+1] - Kvold
            Kv = Kvold + (1-self.RELAX)*(self.LOCAL*dKv + .5*(1-self.LOCAL)*dKv[[0]+list(range(0, jmax)), :, :] + .5*(1-self.LOCAL)*dKv[list(range(1, jmax+1))+[jmax], :, :])
            # Kv = Kvold + dKv
            Kv0 = Kv[:, :, 0]

            # Sf
            Rfmean = np.mean(Ribed[:, 0, :]*(Kvmid*MA_kv)[:, 0, :]/(Avmid*MA_av)[:, 0, :], axis=-1)
            Cd = (1+5.5*Rfmean)**-2.
            damp_sf = Cd
            sf = sfmid*damp_sf
            dsf = sf - sfold
            sf = sfold + (1-self.RELAX)*(self.LOCAL*dsf + .5*(1-self.LOCAL)*dsf[[0]+list(range(0, jmax))] + .5*(1-self.LOCAL)*dsf[list(range(1, jmax+1))+[jmax]])
            # self.timers[5].toc()

            # process for output
            MA_av = ny.fft(MA_av, 2)[:, :, :fmax+1]
            MA_kv = ny.fft(MA_kv, 2)[:, :, :fmax+1]
            Ri = ny.fft(Ri, 2)[:, :, :fmax+1]

        ################################################################################################################
        ## Reference level
        ################################################################################################################
        if self.referenceLevel == 'True':
            self.input.merge({'Av': Av0, 'Roughness': sf})
            R = self.kem.RL.run()['R']
            d['R'] = R
            d['grid']={}
            d['grid']['low']={}
            d['grid']['low']['z']=R

        ################################################################################################################
        ## Compute difference
        ################################################################################################################
        Av0s = ny.savitzky_golay(Av0[:, 0], self.filterlength, 1)
        difference = np.max(abs(Av0s-Avold[:, 0, 0])/abs(Av0s+10**-4))
        self.difference = copy.copy(difference)

        ###############################################################################################################
        # DEBUG plots
        ###############################################################################################################
        # import matplotlib.pyplot as plt
        # import step as st
        # x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0,0]
        #
        # if self.betac > 0 and np.mod(self.iteration, 3)==0:  # and self.difference>0.15:
        #     st.configure()
        #     plt.figure(1, figsize=(2,2))
        #     plt.subplot(1,2,1)
        #     plt.plot(x/1000., Avold[:, 0, 0], label='old')
        #     plt.plot(x/1000., Av0[:, 0], label='new')
        #     plt.ylim(0, np.maximum(np.max(Av0[:, 0]), np.max(Avold[:, 0, 0])))
        #     plt.legend()
        #
        #     plt.subplot(1,2,2)
        #     plt.plot(x/1000., Avold[:, 0, 0]-Av0[:, 0])
        #     plt.twinx()
        #     plt.plot(x/1000., self.input.v('f', range(0, jmax+1)), color='grey')
        #     plt.ylim(0, 1.)
        #
        #     st.save('plot_'+str(len(x))+'_'+str(self.iteration))
        #
        #     plt.figure(2, figsize=(1, 2))
        #     ws = self.input.v('ws0', range(0, jmax+1), 0, 0)
        #     plt.plot(x/1000., ws)
        #
        #     st.save('ws_'+str(len(x))+'_'+str(self.iteration))


        ################################################################################################################
        ## Prepare Output
        ################################################################################################################
        # self.timers[6].tic()

        x = ny.dimensionalAxis(self.input.slice('grid'),'x')[:, 0,0]
        nf = ny.functionTemplates.NumericalFunctionWrapper(ny.savitzky_golay(Av0[:, 0], self.filterlength, 1).reshape((jmax+1, 1)), self.input.slice('grid'))
        nf.addDerivative(ny.savitzky_golay(ny.derivative(Av0[:, 0], 'x', self.input), self.filterlength, 1).reshape((jmax+1, 1)), 'x')
        nf.addDerivative(ny.savitzky_golay(ny.secondDerivative(Av0[:, 0], 'x', self.input), self.filterlength, 1).reshape((jmax+1, 1)), 'xx')
        d['Av'] = nf.function

        nf2 = ny.functionTemplates.NumericalFunctionWrapper(ny.savitzky_golay(Kv0[:, 0], self.filterlength, 1).reshape((jmax+1, 1)), self.input.slice('grid'))
        nf2.addDerivative(ny.savitzky_golay(ny.derivative(Kv0[:, 0], 'x', self.input), self.filterlength, 1).reshape((jmax+1, 1)), 'x')
        nf2.addDerivative(ny.savitzky_golay(ny.secondDerivative(Kv0[:, 0], 'x', self.input), self.filterlength, 1).reshape((jmax+1, 1)), 'xx')
        d['Kv'] = nf2.function

        d['Roughness'] = sf
        d['skin_friction'] = sfmid
        d['dampingFunctions'] = {}
        d['dampingFunctions']['Roughness'] = Cd
        d['dampingFunctions']['Av'] = MA_av
        d['dampingFunctions']['Kv'] = MA_kv
        d['Ri'] = Ri
        # self.timers[6].toc()

        ## Timers
        # self.timers[0].disp('0 init MAW')
        # self.timers[1].disp('1 KEFitted')
        # self.timers[2].disp('2 load data')
        # self.timers[3].disp('3 invfft')
        # self.timers[4].disp('4 Ri')
        # self.timers[5].disp('5 Compute Av, Kv, sf')
        # self.timers[6].disp('6 Load in dict')

        return d


