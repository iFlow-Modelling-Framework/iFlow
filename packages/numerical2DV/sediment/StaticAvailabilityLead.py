"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
import step as st
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from numpy.linalg import svd
from availabilitySolver import availabilitySolver


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

        D = (np.arange(0, fmax+1)*1j*OMEGA).reshape((1, 1, fmax+1))
        d = {}

        ################################################################################################################
        ## Leading order availability
        ################################################################################################################
        c0 = self.input.v('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c0_int = ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))

        b1 = c0_int
        b1[:, :, 0] += beta
        G = []
        a0_til = availabilitySolver(b1, G, self.input).reshape((jmax+1, 1, fmax+1))

        d['a0'] = a0_til

        ################################################################################################################
        ## First order availability
        ################################################################################################################
        c0x = self.input.d('hatc0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        c1_a0 = self.input.v('hatc1_a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0x = self.input.v('hatc1_ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a1 = self.input.v('hatc1_a1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0_int = ny.integrate(c1_a0, 'z', kmax, 0, self.input.slice('grid'))
        c1_a0x_int = ny.integrate(c1_a0x, 'z', kmax, 0, self.input.slice('grid'))
        c1_a1_int = ny.integrate(c1_a1, 'z', kmax, 0, self.input.slice('grid'))

        B = self.input.v('B', range(0, jmax+1), [0], [0])
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        uc_int = ny.integrate(ny.complexAmplitudeProduct(u0, c0, 2), 'z', kmax, 0, self.input.slice('grid'))
        Kh = self.input.v('Kh', range(0, jmax+1), [0], [0])
        a0x_til = ny.derivative(a0_til, 'x', self.input.slice('grid'))

        ## Try to close a0c
        if any(uc_int[:, :, 0]) > 10**(-10):
            T = uc_int-Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid'))
            F = -Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
            T_til = ny.complexAmplitudeProduct(T, a0_til, 2) + ny.complexAmplitudeProduct(F, a0x_til, 2)
            F_til = ny.complexAmplitudeProduct(F, a0_til, 2)
            T_til = T_til[:, 0, 0]
            F_til = F_til[:, 0, 0]

            integral = -ny.integrate(T_til/(F_til+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
            astar = self.input.v('astar')
            A = astar * ny.integrate(B, 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))
            # A = A**-1.      # mistake Ronald?

            a0c = A*np.exp(integral)
            a0xc = integral*T_til/F_til*a0c
        else:
            a0c = None

        ## Solve for a1_til
        b1 = c1_a1_int
        b1[:, :, 0] += beta

        G1 = -1./B*ny.derivative(B*ny.complexAmplitudeProduct(uc_int, a0_til, 2), 'x', self.input.slice('grid'))
        G1 += -D*ny.complexAmplitudeProduct(c1_a0_int + ny.complexAmplitudeProduct(zeta0, c0[:,[0],:], 2), a0_til, 2)
        G1 += -D*ny.complexAmplitudeProduct(c1_a0x_int, a0x_til, 2)

        G2 = ny.complexAmplitudeProduct(uc_int, a0_til, 2)
        G2 += -D*ny.complexAmplitudeProduct(c1_a0x_int, a0_til, 2)

        if a0c is not None:
            G1 += 0     # TODO add Kh
            G2 += 0     # TODO add Kh
            G3 = 0      # TODO add Kh

        a1_til = availabilitySolver(b1, [G1, G2], self.input)

        ################################################################################################################
        ## Second order availability
        ################################################################################################################
        if a0c is None:
            u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

            d['T'] = {'TM0': {'TM0stokes': {}}, 'TM2': {}, 'TM4': {}, 'Tdiff': {}}
            # d['T'] = {'TM0': {'TM0stokes': {}}, 'Tdiff': {}}
            d['F'] = {'Fdiff': {}, 'Fadv': {'FadvM0': {}, 'FadvM4': {}}}
            T0 = 0
            F0 = 0

            ## Classical transport mechanisms
            # T0 component - part 1: u0*c1_a0
            T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1_a0, 2), 'z', kmax, 0, self.input.slice('grid'))
            for submod in self.input.getKeysOf('hatc1_a'):
                c1_a0_comp = self.input.v('hatc1_a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                for n in range(0, fmax+1):
                    tmp = np.zeros(c1_a0_comp.shape, dtype=complex)
                    tmp[:, :, n] = c1_a0_comp[:, :, n]
                    d['T']['TM'+str(2*n)][submod] = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]

            # T0 component - part 2: u1*c0
            T0 += ny.integrate(ny.complexAmplitudeProduct(u1, c0, 2), 'z', kmax, 0, self.input.slice('grid'))
            for submod in self.input.getKeysOf('u1'):
                u1_comp = self.input.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                for n in range(0, fmax+1):
                    tmp = np.zeros(u1_comp.shape, dtype=complex)
                    tmp[:, :, n] = u1_comp[:, :, n]
                    if submod == 'stokes' and n == 0:
                        d['T']['TM0']['TM0stokes'][submod] = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    else:
                        d['T']['TM'+str(2*n)][submod] = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]

            # T0 component - part 3: u0*c0*zeta0
            T0 += -ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0[:, [0], :], 2), zeta0, 2)
            d['T']['TM0']['TM0stokes']['drift'] = -ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0[:, [0], :], 2), zeta0, 2)[:, 0, 0]

            # T0 component - part 4: diffusive part
            T0 += - Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid'))
            d['T']['Tdiff'] = - (Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]

            # F0 component - part 1: u0*c1_a0x
            F0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1_a0x, 2), 'z', kmax, 0, self.input.slice('grid'))
            for n in range(0, fmax+1):
                tmp = np.zeros(c1_a0x.shape, dtype=complex)
                tmp[:, :, n] = c1_a0x[:, :, n]
                d['F']['Fadv']['FadvM'+str(2*n)] = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]

            # F0 component - part 2: diffusive part
            F0 += - Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
            d['F']['Fdiff'] = - (Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]

            ## New mechanisms

            # Transport due to T0*a04 + F0*ax04
            a0_til_M4 = np.zeros(a0_til.shape, dtype=complex)
            a0x_til_M4 = np.zeros(a0x_til.shape, dtype=complex)
            a0_til_M4[:, :, 2] = a0_til[:, :, 2]
            a0x_til_M4[:, :, 2] = a0x_til[:, :, 2]
            #d['T']['AM4'] = (ny.complexAmplitudeProduct(T0, a0_til_M4, 2) + ny.complexAmplitudeProduct(F0, a0x_til_M4, 2))[:, 0, 0]

            # Transport due to T1*a12
            T1 = ny.integrate(ny.complexAmplitudeProduct(u0, c1_a1, 2), 'z', kmax, 0, self.input.slice('grid'))
            #d['T']['AM2'] = ny.complexAmplitudeProduct(T1, a1_til[:, :, :, 1], 2)[:, 0, 0]

            # Diffusion component due to T1*a12
            #d['F']['AM2'] = ny.complexAmplitudeProduct(T1, a1_til[:, :, :, 2], 2)[:, 0, 0]

            ## Add all mechanisms & compute a0c
            from src.DataContainer import DataContainer
            dc = DataContainer(d)
            T_til = dc.v('T', range(0, jmax+1))
            F_til = dc.v('F', range(0, jmax+1))
            print dc.v('T', range(0, jmax+1))-T0[:, 0, 0]
            print dc.v('F', range(0, jmax+1))-F0[:, 0, 0]

            integral = -ny.integrate(T_til/(F_til+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))
            astar = self.input.v('astar')
            A = astar * ny.integrate(B[:, 0, 0], 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B[:, 0, 0]*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))

            a0c = A*np.exp(integral)
            a0xc = -T_til/F_til*a0c

        ## Plot
        import matplotlib.pyplot as plt
        import step as st
        st.configure()
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

        plt.figure(2, figsize=(1, 2))
        plt.plot(x, a0c)

        plt.figure(4, figsize=(1, 2))
        u = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1)) + self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid'))/ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, [-1], :]
        for n in range(0, fmax+1):
            plt.plot(x, abs(u[:, 0, n]))

        plt.figure(5, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)

        for key in dc.getKeysOf('T'):
            p = plt.plot(x/1000., dc.v('T', key, range(0, jmax+1)), label=key)
            try:
                plt.plot(x/1000., self.input.v('T', key, range(0, jmax+1)), '--', label=key, color=p[0].get_color())
            except:
                pass
        plt.plot(x/1000., dc.v('T', range(0, jmax+1)), 'k', label='Total')
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.title('Transport terms')
        plt.xlabel('x (km)')
        plt.ylabel('T')

        plt.figure(6, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for key in dc.getKeysOf('F'):
            p = plt.plot(x/1000., dc.v('F', key, range(0, jmax+1)), label=key)
            try:
                plt.plot(x/1000., self.input.v('F', key, range(0, jmax+1)), '--', label=key, color=p[0].get_color())
            except:
                pass
        plt.plot(x/1000., dc.v('F', range(0, jmax+1)), 'k', label='Total')
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.title('Diffusive terms')
        plt.xlabel('x (km)')
        plt.ylabel('F')


        st.show()

        return d