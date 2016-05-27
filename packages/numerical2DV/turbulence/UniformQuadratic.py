"""
Eddy viscosity with uniform profile (with or without time variations)
Used in combination with a partial slip boundary condition, where the partial slip roughness parameter depends on the
velocity. This effectively makes a quadratic boundary condition. Input roughness parameters may be cf0 and z0

Date: 04-03-16
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
import numpy as np
from profiles.UniformXF import UniformXF
from src.util.diagnostics.KnownError import KnownError


class UniformQuadratic:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10.**-2

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        stop = False

        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        if hasattr(self, 'sf_prev_iter'):
            sf = self.input.v('Roughness', range(0, jmax+1), 0, range(0, fmax+1))

            # use an ofset in the denominator. This is to prevent endless iteration of velocities close to zero.
            # here take an ofset that is 5% of the maximum roughness
            offset = 0.05*np.max(np.abs(sf))
            maxchange = np.max(np.abs((self.sf_prev_iter-sf)/(sf+offset)))
            self.sf_prev_iter = sf

            self.logger.info('uniformQuadratic - relative change in last iteration %s' % str(maxchange))
            if maxchange < self.TOLLERANCE:
                stop = True
        else:
            self.sf_prev_iter = self.input.v('Roughness', range(0, jmax + 1), 0, range(0, fmax + 1))
        return stop

    def run_init(self):
        self.logger.info('Running uniformQuadratic turbulence model - initialise')

        # Determine complex amplitudes of the eddy viscosity
        fmax = self.input.v('grid', 'maxIndex', 'f')
        Av0 = ny.amp_phase_input(self.input.v('Av0amp'), self.input.v('Av0phase'), (1, fmax+1))

        # Determine complex amplitudes of the roughness, based on the input roughness parameter. The initial estimate
        #   for the velocity is a sub-tidal velocity of 1
        if self.input.v('roughnessParameter') == 'cf0':
            sf0 = ny.amp_phase_input(self.input.v('cf0'), 0, (1, fmax + 1))
        elif self.input.v('roughnessParameter') == 'z0':
            jmax = self.input.v('grid', 'maxIndex', 'x')
            self.dzbed = ny.dimensionalAxis(self.input.slice('grid'), 'z')
            self.dzbed = abs(self.dzbed[:,-1,0]-self.dzbed[:,-2,0])
            sf0 = np.zeros((jmax+1, fmax+1))
            sf0[:,0] = 0.4**2*np.log(1+0.5*self.dzbed/self.input.v('z0'))**(-2.)
        else:
            raise KnownError('roughness '+str(self.input.v('roughnessParameter'))+' not implemented.')

        # prepare the smaller datacontainers used in the functions UniformXF. These functions will be called when calling
        #   Av and sf0. These functions require the grid and complex amplitudes above
        data = self.input.slice('grid')
        data.addData('coef', Av0)

        dataRough = self.input.slice('grid')
        dataRough.addData('coef', sf0)

        sf = UniformXF(['x', 'f'], dataRough, self.input.v('n'))
        Av = UniformXF(['x', 'f'], data, self.input.v('m'))

        # Prepare output
        d = {}
        d['Roughness'] = sf.function
        d['Av'] = Av.function
        d['BottomBC'] = 'PartialSlip'
        return d

    def run(self):
        self.logger.info('Running uniformQuadratic turbulence model - iteration %s' % str(self.iteration))
        fmax = self.input.v('grid','maxIndex','f')

        # compute sf0 = cf0*|u|
        if self.input.v('roughnessParameter')=='cf0':
            sf0 = ny.amp_phase_input(self.input.v('cf0'), 0, (1, fmax + 1))
        elif self.input.v('roughnessParameter')=='z0':
            jmax = self.input.v('grid', 'maxIndex', 'x')
            sf0 = np.zeros((jmax+1, fmax+1))
            sf0[:,0] = 0.4**2*np.log(1+0.5*self.dzbed/self.input.v('z0'))**(-2.)
        ubedabs = self.uRelax()
        sf0 = ny.complexAmplitudeProduct(sf0, ubedabs, 1)

        # initiate function
        dataRough = self.input.slice('grid')
        dataRough.addData('coef', sf0)

        sf = UniformXF(['x', 'f'], dataRough, self.input.v('n'))

        # load final dictionary
        d = {}
        d['Roughness'] = sf.function
        return d


    def uRelax(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
        kmax = self.input.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
        fmax = self.input.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)

        # absolute velocity
        u0bed = self.input.v('u0', range(0, jmax + 1), kmax, range(0, fmax + 1))
        c = ny.polyApproximation(np.abs, 4)  # chebyshev coefficients for abs
        uamp = (np.sum(np.abs(u0bed), axis=-1).reshape((jmax+1, 1)) + 10 ** -6)
        uabs = np.zeros(u0bed.shape, dtype=complex)
        uabs[:, 0] = c[0]
        u_prod = u0bed / uamp
        for i in range(2, len(c), 2):
            u_prod = ny.complexAmplitudeProduct(u_prod, u_prod, 1)
            uabs = uabs + c[i] * u_prod
        uabs = uabs * uamp

        # relaxation:
        if hasattr(self, 'u_prev_iter'):
            u_prev_iter = self.u_prev_iter
        else:
            u_prev_iter = np.zeros(uabs.shape)
            u_prev_iter[:, 0] = 1.
        mar = 0.5  # take a max change of 50% wrt the difference between old and new velocity
        u_prev_iter2 = u_prev_iter - (uabs)
        u0 = np.max((uabs, np.min((u_prev_iter2 * (1 - mar), u_prev_iter2 * (1. + mar)), axis=0) + (uabs)),
                    axis=0)
        u0 = np.min((u0, np.max((u_prev_iter2 * (1 - mar), u_prev_iter2 * (1. + mar)), axis=0) + uabs), axis=0)
        self.u_prev_iter = u0       # save velocity at bed for next iteration

        # import matplotlib.pyplot as plt           # plotting code for testing convergence
        # plt.hold(True)
        # plt.subplot(121)
        # plt.plot(np.real(u0[:,0]))
        # plt.plot(np.real(u_prev_iter[:,0]), 'k--')
        # plt.plot(np.real(uabs[:,0]), 'r--')
        # plt.subplot(122)
        # plt.plot(np.imag(u0[:,1]))
        # plt.plot(np.imag(u_prev_iter[:,1]), 'k--')
        # plt.plot(np.imag(uabs[:,1]), 'r--')
        # plt.show()
        return u0

