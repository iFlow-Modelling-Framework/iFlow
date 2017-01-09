"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
import step as st
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from numpy.linalg import svd
from availabilitySolver import availabilitySolver
import matplotlib.pyplot as plt
import step as st


class StaticAvailabilityLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module StaticAvailability')

        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        OMEGA = self.input.v('OMEGA')

        beta = 1

        d = {}

        ################################################################################################################
        ## Leading order availability
        ################################################################################################################
        c0 = self.input.v('hatc0')
        c0_int = ny.integrate(c0.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot, ftot))

        D = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
        D[:, :, range(0, ftot), range(0, ftot)] = np.arange(-fmax, fmax+1)*1j*OMEGA
        b = beta*D + ny.arraydot(D, c0_int)

        G = []
        a0_til = availabilitySolver(b, G, self.input).reshape((jmax+1, 1, ftot))

        d['a0'] = a0_til

        res = ny.arraydot(D, a0_til) + ny.arraydot(D, ny.arraydot(c0_int, a0_til))

        ################################################################################################################
        ## First order availability
        ################################################################################################################
        # load chat from DC
        c0x = ny.derivative(c0.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'x', self.input.slice('grid')).reshape((jmax+1, kmax+1, ftot, ftot))
        c1_a0 = self.input.v('hatc1_a')
        c1_a0x = self.input.v('hatc1_ax')
        c1_a1 = self.input.v('hatc1_a1')

        # integrate over vertical
        c1_a0_int = ny.integrate(c1_a0.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot, ftot))
        c1_a0x_int = ny.integrate(c1_a0x.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot, ftot))
        c1_a1_int = ny.integrate(c1_a1.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot, ftot))

        B = self.input.v('B', range(0, jmax+1), [0], [0])
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u0_ext = np.concatenate((np.zeros((jmax+1, kmax+1, fmax)), u0), 2)
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        zeta0_ext = np.concatenate((np.zeros((jmax+1, 1, fmax)), zeta0), 2)
        Kh = self.input.v('Kh', range(0, jmax+1), [0], [0])
        a0x_til = ny.derivative(a0_til.reshape((jmax+1, 1, 1, ftot)), 'x', self.input.slice('grid')).reshape((jmax+1, 1, ftot))

        ## Solve for a1_til
        D = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
        D[:, :, range(0, ftot), range(0, ftot)] = np.arange(-fmax, fmax+1)*1j*OMEGA
        b = beta*D + ny.arraydot(D, c1_a1_int)

        # terms with f00/f10
        G1 = -ny.arraydot(D, (ny.arraydot(c1_a0_int, a0_til) + ny.complexAmplitudeProduct(zeta0_ext, ny.arraydot(c0[:, [0], :, :], a0_til), 2, includeNegative=True)))
        G1 +=-ny.arraydot(D, ny.arraydot(c1_a0x_int, a0x_til))
        ucf = ny.integrate(ny.complexAmplitudeProduct(u0_ext, ny.arraydot(c0, a0_til*np.ones((1, kmax+1, 1))), 2, includeNegative=True).reshape((jmax+1, kmax+1, 1, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot))
        G1 += -1./B*ny.derivative((B*ucf).reshape((jmax+1, 1, 1, ftot)), 'x', self.input.slice('grid')).reshape((jmax+1, 1, ftot))

        # terms with f00x/f10
        G2 = - ny.arraydot(D, ny.arraydot(c1_a0x_int, a0_til))
        G2 += - ucf

        # if a0c is not None:
        #     G1 += 0     # TODO add Kh
        #     G2 += 0     # TODO add Kh
        #     G3 = 0      # TODO add Kh

        a1_til = availabilitySolver(b, [G1, G2], self.input)

        ################################################################################################################
        ## First order closure
        ################################################################################################################
        if any(ucf[:, :, 0]) > 10**(-10):
            raise NotImplementedError('first order closure not yet implemented')
            # T = uc_int-Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid'))
            # F = -Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
            # T_til = ny.complexAmplitudeProduct(T, a0_til, 2) + ny.complexAmplitudeProduct(F, a0x_til, 2)
            # F_til = ny.complexAmplitudeProduct(F, a0_til, 2)
            # T_til = T_til[:, 0, 0]
            # F_til = F_til[:, 0, 0]
            #
            # integral = -ny.integrate(T_til/(F_til+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
            # astar = self.input.v('astar')
            # A = astar * ny.integrate(B, 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))
            #
            # a0c = A*np.exp(integral)
            # a0xc = integral*T_til/F_til*a0c

        ################################################################################################################
        ## Second order closure
        ################################################################################################################
        else:
            u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

            d['T'] = {}
            d['F'] = {}
            T0 = 0
            F0 = 0

            ## Transport T  ############################################################################################
            ## T.1. - u0*c1_a0
            # Total
            c1a_f0 = ny.eliminateNegativeFourier(ny.arraydot(c1_a0, a0_til), 2)
            T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1a_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            for submod in self.input.getKeysOf('hatc1_a'):
                c1_a0_comp = self.input.v('hatc1_a', submod)
                c1a_f0_comp_res = ny.eliminateNegativeFourier(c1_a0_comp[:, :, :, fmax], 2)   # C1a0 times residual part of f0
                d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)] + ['AM'+str(2*n) for n in range(1, fmax+1)])  # add submod index to dict if not already
                # transport with residual availability
                for n in range(0, fmax+1):
                    tmp = np.zeros(c1a_f0_comp_res.shape, dtype=complex)
                    tmp[:, :, n] = c1a_f0_comp_res[:, :, n]
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp


                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a0_til.shape, dtype=complex)
                    tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                    tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                    c1a_f0_comp_var = ny.eliminateNegativeFourier(ny.arraydot(c1_a0_comp, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, c1a_f0_comp_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['AM'+str(2*n)] += tmp

            ## T.2. - u1*c0
            # Total
            c0_f0 = ny.eliminateNegativeFourier(ny.arraydot(c0, a0_til), 2)
            T0 += ny.integrate(ny.complexAmplitudeProduct(u1, c0_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            c0_f0_res = ny.eliminateNegativeFourier(c0[:, :, :, fmax], 2)   # C0 times residual part of f0
            for submod in self.input.getKeysOf('u1'):
                u1_comp = self.input.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)] + ['AM'+str(2*n) for n in range(1, fmax+1)]) # add submod index to dict if not already
                # transport with residual availability
                for n in range(0, fmax+1):
                    tmp = np.zeros(u1_comp.shape, dtype=complex)
                    tmp[:, :, n] = u1_comp[:, :, n]
                    if submod == 'stokes' and n == 0:
                        tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0_f0_res, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                        if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM0', 'TM0stokes')
                            d['T'][submod]['TM0']['TM0stokes'] += tmp
                    else:
                        tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0_f0_res, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                        if any(tmp) > 10**-14:
                            d['T'][submod]['TM'+str(2*n)] += tmp

                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a0_til.shape, dtype=complex)
                    tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                    tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                    c0_f0_var = ny.eliminateNegativeFourier(ny.arraydot(c0, tmp), 2)    # C0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u1_comp, c0_f0_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['AM'+str(2*n)] += tmp

            ## T.3. - u0*C1ax*f0x
            a0_tilx = ny.derivative(a0_til.reshape((jmax+1, 1, 1, ftot)), 'x', self.input.slice('grid')).reshape((jmax+1, 1, ftot))
            # Total
            c1ax_f0 = ny.eliminateNegativeFourier(ny.arraydot(c1_a0x, a0_tilx), 2)
            T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1ax_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            for submod in self.input.getKeysOf('hatc1_ax'):
                c1_ax0_comp = self.input.v('hatc1_ax', submod)
                d['T'] = self.dictExpand(d['T'], submod, ['AM'+str(2*n) for n in range(1, fmax+1)])  # add submod index to dict if not already

                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a0_tilx.shape, dtype=complex)
                    tmp[:, :, fmax-n] = a0_tilx[:, :, fmax-n]
                    tmp[:, :, fmax+n] = a0_tilx[:, :, fmax+n]
                    c1ax_f0_comp_var = ny.eliminateNegativeFourier(ny.arraydot(c1_ax0_comp, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, c1ax_f0_comp_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['AM'+str(2*n)] += tmp

            ## T.4. - u0*C1a1*f1_(f0)
            # Total
            c1a1_f1 = ny.eliminateNegativeFourier(ny.arraydot(c1_a1, a1_til[:, :, :, 1]), 2)
            T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1a1_f1, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            for submod in self.input.getKeysOf('hatc1_a1'):
                c1_a1_comp = self.input.v('hatc1_a1', submod)
                d['T'] = self.dictExpand(d['T'], submod, ['AM'+str(2*n) for n in range(1, fmax+1)])  # add submod index to dict if not already

                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a1_til[:, :, :, 1].shape, dtype=complex)
                    tmp[:, :, fmax-n] = a1_til[:, :, fmax-n, 1]
                    tmp[:, :, fmax+n] = a1_til[:, :, fmax+n, 1]
                    c1a1_f1_comp_var = ny.eliminateNegativeFourier(ny.arraydot(c1_a1_comp, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, c1a1_f1_comp_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['AM'+str(2*n)] += tmp

            ## T.5. - u0*c0*zeta0
            # Total
            c0_f0 = ny.eliminateNegativeFourier(ny.arraydot(c0[:, [0], Ellipsis], a0_til), 2)
            T0 += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0_f0, 2), zeta0, 2)

            # Decomposition
            uzeta = ny.complexAmplitudeProduct(u0[:, [0], :], zeta0, 2)
            d['T'] = self.dictExpand(d['T'], 'drift', ['TM'+str(2*n) for n in range(0, fmax+1)] + ['AM'+str(2*n) for n in range(1, fmax+1)])
            # transport with residual availability
            c0_f0_res = ny.eliminateNegativeFourier(c0[:, 0, :, fmax], 1).reshape((jmax+1, 1, fmax+1))
            for n in range(0, fmax+1):
                tmp = np.zeros(c0_f0_res.shape, dtype=complex)
                tmp[:, :, n] = c0_f0_res[:, :, n]
                tmp = ny.complexAmplitudeProduct(uzeta, tmp, 2)[:, 0, 0]
                if any(tmp) > 10**-14:
                    d['T']['drift']['TM'+str(2*n)] = tmp

            # transport with time varying availability
            for n in range(1, fmax+1):
                tmp = np.zeros(a0_til[:, :, :].shape, dtype=complex)
                tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                c0_f0_var = ny.eliminateNegativeFourier(ny.arraydot(c0[:, [0], Ellipsis], tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                tmp = ny.complexAmplitudeProduct(uzeta, c0_f0_var, 2)[:, 0, 0]
                if any(tmp) > 10**-14:
                    d['T']['drift']['AM'+str(2*n)] = tmp

            ## T.6. - diffusive part
            # Total
            c0_f0 = ny.eliminateNegativeFourier(ny.arraydot(c0[:, :, Ellipsis], a0_til), 2)
            c0_f0_x = ny.derivative(c0_f0, 'x', self.input.slice('grid'))
            T0 += - Kh*ny.integrate(c0_f0_x, 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            d['T'] = self.dictExpand(d['T'], 'diff', ['TM0'] + ['AM'+str(2*n) for n in range(1, fmax+1)])
            # transport with residual availability
            c0_f0_res = ny.eliminateNegativeFourier(c0[:, :, :, fmax], 2)
            c0_f0_res_x = ny.derivative(c0_f0_res, 'x', self.input.slice('grid'))
            tmp = - (Kh*ny.integrate(c0_f0_res_x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
            if any(tmp) > 10**-14:
                d['T']['diff']['TM0'] = tmp

            # transport with time varying availability
            for n in range(1, fmax+1):
                tmp = np.zeros(a0_til[:, :, :].shape, dtype=complex)
                tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                c0_f0_var = ny.eliminateNegativeFourier(ny.arraydot(c0, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                c0_f0_var_x = ny.derivative(c0_f0_var, 'x', self.input.slice('grid'))
                tmp = - (Kh*ny.integrate(c0_f0_var_x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
                if any(tmp) > 10**-14:
                    d['T']['diff']['AM'+str(2*n)] = tmp

            ## Diffusion F  ############################################################################################
            ## F.1. - u0*C1ax*f0
            # Total
            c1ax_f0 = ny.eliminateNegativeFourier(ny.arraydot(c1_a0x, a0_til), 2)
            F0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1ax_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            for submod in self.input.getKeysOf('hatc1_ax'):
                c1_ax0_comp = self.input.v('hatc1_ax', submod)
                d['F'] = self.dictExpand(d['F'], submod, ['FM'+str(2*n) for n in range(0, fmax+1)] + ['AM'+str(2*n) for n in range(1, fmax+1)])  # add submod index to dict if not already
                # transport with residual availability
                for n in range(0, fmax+1):
                    c1_ax0_comp_res = ny.eliminateNegativeFourier(c1_ax0_comp[:, :, :, fmax], 2)
                    tmp = np.zeros(u0.shape, dtype=complex)
                    tmp[:, :, n] = u0[:, :, n]
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c1_ax0_comp_res, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['F'][submod]['FM'+str(2*n)] += tmp

                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a0_til.shape, dtype=complex)
                    tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                    tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                    c1ax_f0_comp_var = ny.eliminateNegativeFourier(ny.arraydot(c1_ax0_comp, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, c1ax_f0_comp_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['F'][submod]['AM'+str(2*n)] += tmp

            ## F.2. - u0*C1a1*f1_(f0x)
            # Total
            c1a1_f1 = ny.eliminateNegativeFourier(ny.arraydot(c1_a1, a1_til[:, :, :, 2]), 2)
            F0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1a1_f1, 2), 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            for submod in self.input.getKeysOf('hatc1_a1'):
                c1_a1_comp = self.input.v('hatc1_a1', submod)
                d['F'] = self.dictExpand(d['F'], submod, ['AM'+str(2*n) for n in range(1, fmax+1)])  # add submod index to dict if not already

                # transport with time varying availability
                for n in range(1, fmax+1):
                    tmp = np.zeros(a1_til[:, :, :, 2].shape, dtype=complex)
                    tmp[:, :, fmax-n] = a1_til[:, :, fmax-n, 2]
                    tmp[:, :, fmax+n] = a1_til[:, :, fmax+n, 2]
                    c1a1_f1_comp_var = ny.eliminateNegativeFourier(ny.arraydot(c1_a1_comp, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, c1a1_f1_comp_var, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['F'][submod]['AM'+str(2*n)] += tmp

            ## F.3. - diffusive part
            # Total
            c0_f0 = ny.eliminateNegativeFourier(ny.arraydot(c0[:, :, Ellipsis], a0_til), 2)
            F0 += - Kh*ny.integrate(c0_f0, 'z', kmax, 0, self.input.slice('grid'))

            # Decomposition
            d['F'] = self.dictExpand(d['F'], 'diff', ['TM0'] + ['AM'+str(2*n) for n in range(1, fmax+1)])
            # transport with residual availability
            c0_f0_res = ny.eliminateNegativeFourier(c0[:, :, :, fmax], 2)
            tmp = - (Kh*ny.integrate(c0_f0_res, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
            if any(tmp) > 10**-14:
                d['F']['diff']['TM0'] = tmp

            # transport with time varying availability
            for n in range(1, fmax+1):
                tmp = np.zeros(a0_til[:, :, :].shape, dtype=complex)
                tmp[:, :, fmax-n] = a0_til[:, :, fmax-n]
                tmp[:, :, fmax+n] = a0_til[:, :, fmax+n]
                c0_f0_var = ny.eliminateNegativeFourier(ny.arraydot(c0, tmp), 2)    # C1a0 times time-varying part of f0 of freq n
                tmp = - (Kh*ny.integrate(c0_f0_var, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
                if any(tmp) > 10**-14:
                    d['F']['diff']['AM'+str(2*n)] = tmp

            ## Solve    ################################################################################################
            ## Add all mechanisms & compute a0c
            from src.DataContainer import DataContainer
            dc = DataContainer(d)
            T_til = dc.v('T', range(0, jmax+1))
            F_til = dc.v('F', range(0, jmax+1))
            print np.max(abs((dc.v('T', range(0, jmax+1))-T0[:, 0, 0])/(T0[:, 0, 0]+10**-10)))
            print np.max(abs((dc.v('F', range(0, jmax+1))-F0[:, 0, 0])/(F0[:, 0, 0]+10**-10)))

            integral = -ny.integrate(T_til/(F_til+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
            astar = self.input.v('astar')
            A = astar * ny.integrate(B[:, 0, 0], 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B[:, 0, 0]*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))

            a0c = A*np.exp(integral)
            a0xc = -T_til/F_til*a0c

        ## Plot
        a0_til = ny.eliminateNegativeFourier(a0_til, 2)
        a1_til = ny.eliminateNegativeFourier(a1_til, 2)
        st.configure()

        ################################################################################################################
        # Fig 1: f0 and f1
        ################################################################################################################
        plt.figure(1, figsize=(1,3))
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        plt.subplot(1,3,1)

        plt.plot(x,np.abs(a0_til[:, 0, 0]*a0c), 'b-')
        plt.plot(x,np.abs(a0_til[:, 0, 1]*a0c), 'g-')
        plt.plot(x,np.abs(a0_til[:, 0, 2]*a0c), 'r-')

        # plt.subplot(1,3,2)
        plt.plot(x,np.abs(a1_til[:, 0, 0, 1]*a0c + a1_til[:, 0, 0, 2]*a0xc), 'b-')
        plt.plot(x,np.abs(a1_til[:, 0, 1, 1]*a0c + a1_til[:, 0, 1, 2]*a0xc), 'g-')
        plt.plot(x,np.abs(a1_til[:, 0, 2, 1]*a0c + a1_til[:, 0, 2, 2]*a0xc), 'r-')

        plt.subplot(1,3,3)
        plt.plot(x,np.abs(a1_til[:, 0, 0, 0]), 'b:')
        plt.plot(x,np.abs(a1_til[:, 0, 1, 0]), 'g:')
        plt.plot(x,np.abs(a1_til[:, 0, 2, 0]), 'r:')

        ################################################################################################################
        # Fig 4: u0 da
        ################################################################################################################
        plt.figure(4, figsize=(1, 2))
        u = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1)) + self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid'))/ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, [-1], :]
        for n in range(0, fmax+1):
            plt.plot(x, abs(u[:, 0, n]))

        ################################################################################################################
        # Fig 5: transport contributions
        ################################################################################################################
        plt.figure(5, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)

        tterms = {}
        for key in dc.getKeysOf('T'):
            tterms[key] = np.linalg.norm(dc.v('T', key, range(0, jmax+1)), 1)
        tterms = [i for i in sorted(tterms, key=tterms.get, reverse=True)]
        tterms = tterms[:5]
        for key in tterms:
            p = plt.plot(x/1000., dc.v('T', key, range(0, jmax+1)), label=key)
            try:
                plt.plot(x/1000., self.input.v('T', key, range(0, jmax+1)), 'o', label=key, color=p[0].get_color())
            except:
                pass
        plt.plot(x/1000., dc.v('T', range(0, jmax+1)), 'k', label='Total')
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.title('Transport terms')
        plt.xlabel('x (km)')
        plt.ylabel('T')

        ################################################################################################################
        # Fig 6: Diffusion contributions
        ################################################################################################################
        plt.figure(6, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for key in dc.getKeysOf('F'):
            p = plt.plot(x/1000., dc.v('F', key, range(0, jmax+1)), label=key)
            try:
                plt.plot(x/1000., self.input.v('F', key, range(0, jmax+1)), 'o', label=key, color=p[0].get_color())
            except:
                pass
        plt.plot(x/1000., dc.v('F', range(0, jmax+1)), 'k', label='Total')
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.title('Diffusive terms')
        plt.xlabel('x (km)')
        plt.ylabel('F')


        st.show()

        return d

    def dictExpand(self, d, subindex, subsubindices):
        if not subindex in d:
            d[subindex] = {}
        elif not isinstance(d[subindex], dict):
            d[subindex] = {}
        for ssi in ny.toList(subsubindices):
            if not ssi in d[subindex]:
                d[subindex][ssi] = 0
        return d


