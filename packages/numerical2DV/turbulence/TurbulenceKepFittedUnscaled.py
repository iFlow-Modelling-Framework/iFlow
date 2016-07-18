"""
TurbulenceKepFitted_1

Date: 25-Apr-16
Authors: Y.M. Dijkstra
"""
from TurbulenceKepFitted import TurbulenceKepFitted
import numpy as np
import nifty as ny


class TurbulenceKepFittedUnscaled(TurbulenceKepFitted):
    # Variables
    RELAX = 0.5
    TOLLERANCE = 2.*1e-2*RELAX       # relative change allowed for converged result
    order = None                     # no ordering

    # Methods
    def __init__(self, input, submodulesToRun):
        TurbulenceKepFitted.__init__(self, input, submodulesToRun)
        return

    def stopping_criterion(self, iteration):
        return TurbulenceKepFitted.stopping_criterion(self, iteration)

    def run_init(self):
        self.logger.info('Running k-epsilon fitted turbulence model - init')
        # if Av already exists, then do not restart the model, but continue from the previous result
        if self.input.v('Av') is not None:
            init = False

            # reset previous iterations by current iteration
            relfac = self.RELAX
            self.RELAX = 1.
            self.u_prev_iter, self.uH_prev_iter = self.uRelax(self.order)
            self.RELAX = relfac
        # else initialise
        else:
            init = True

        Av, roughness, BottomBC = self.main(init=init)

        d = {}
        d['Roughness'] = roughness
        d['Av'] = Av
        d['BottomBC'] = BottomBC

        return d

    def run(self):
        self.logger.info('Running k-epsilon fitted turbulence model')

        Av, roughness, _ = self.main()

        # load to dictionary
        d = {}
        d['Roughness'] = roughness
        d['Av'] = Av

        return d

    def uRelax(self, *args):
        # Init
        jmax = self.input.v('grid', 'maxIndex', 'x')  # maximum index of x grid (jmax+1 grid points incl. 0)
        kmax = self.input.v('grid', 'maxIndex', 'z')  # maximum index of z grid (kmax+1 grid points incl. 0)
        fmax = self.input.v('grid', 'maxIndex', 'f')  # maximum index of f grid (fmax+1 grid points incl. 0)
        depth = self.input.v('grid', 'low', 'z', range(0, jmax+1), [0], range(0, fmax+1)) - self.input.v('grid', 'high', 'z', range(0, jmax+1), [0], range(0, fmax+1))

        # 1. make the absolute velocity
        #   Gather velocity and zeta components
        zeta = 0
        u = 0
        comp = 0
        while comp is not None:
            if self.input.v('zeta'+str(comp)):
                zeta += self.input.v('zeta'+str(comp), range(0, jmax + 1), [0], range(0, fmax + 1))
                u += self.input.v('u'+str(comp), range(0, jmax + 1), range(0, kmax + 1), range(0, fmax + 1))
                comp += 1
            else:
                comp = None
        u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid')) / depth

        #   Divide velocity by a maximum amplitude
        uamp = (np.sum(np.abs(u), axis=-1)+10**-3).reshape((jmax+1, 1, 1))
        u = u/uamp

        #   Absolute velocity at all orders up to 'order'
        c = ny.polyApproximation(np.abs, 8)  # chebyshev coefficients for abs
        uabs = np.zeros(u.shape, dtype=complex)
        uabs[:, :, 0] = c[0]
        u2 = ny.complexAmplitudeProduct(u, u, 2)
        uabs += c[2]*u2
        u4 = ny.complexAmplitudeProduct(u2, u2, 2)
        uabs += c[4]*u4
        u6 = ny.complexAmplitudeProduct(u2, u4, 2)
        uabs += c[6]*u6
        del u2, u6
        u8 = ny.complexAmplitudeProduct(u4, u4, 2)
        uabs += c[8]*u8
        del u4, u8

        uabs = uabs * uamp.reshape((jmax+1, 1, 1))

        #   Absolute velocity * depth
        uabsH = uabs*depth
        uabsH += ny.complexAmplitudeProduct(uabs, zeta, 2)

        # 2. Relaxtion
        #   2a. Relaxation on uabs
        if hasattr(self, 'u_prev_iter'):
            u_prev_iter = self.u_prev_iter
        else:
            u_prev_iter = np.zeros(uabs.shape, dtype=complex)
            u_prev_iter[:, :, 0] = np.max(uabs[:,:,0]) # initially take the maximum velocity, so that the system has too much damping

        u_prev_iter2 = u_prev_iter - (uabs)
        u0 = np.max((uabs, np.min((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + (uabs)), axis=0)
        u0 = np.min((u0,   np.max((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + uabs), axis=0)
        self.u_prev_iter = u0  # save velocity at bed for next iteration

        #    2b. Relaxation on uabs*depth
        if hasattr(self, 'uH_prev_iter'):
            u_prev_iter = self.uH_prev_iter
        else:
            u_prev_iter = np.zeros(uabs.shape, dtype=complex)
            u_prev_iter[:, :, 0] = np.max(uabs[:,:,0])*depth[:,:,0] # initially take the maximum velocity, so that the system has too much damping

        u_prev_iter2 = u_prev_iter - (uabsH)
        uH0 = np.max((uabsH, np.min((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + (uabsH)), axis=0)
        uH0 = np.min((uH0,   np.max((u_prev_iter2 * (1 - self.RELAX), u_prev_iter2 * (1. + self.RELAX)), axis=0) + uabsH), axis=0)
        self.uH_prev_iter = uH0  # save velocity at bed for next iteration

        return u0, uH0
