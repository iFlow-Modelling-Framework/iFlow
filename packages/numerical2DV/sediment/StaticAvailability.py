"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from scipy.linalg import solve_banded
from numpy.linalg import svd
import matplotlib.pyplot as plt
import step as st


class StaticAvailability:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10**-6

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module StaticAvailability')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1

        c0 = self.input.v('hatc0')
        c1_a0 = self.input.v('hatc1_a')
        c1_a0x = self.input.v('hatc1_ax')

        d = {}

        c0_int = ny.integrate(c0.reshape((jmax+1, kmax+1, 1, ftot, ftot)), 'z', kmax, 0, self.input.slice('grid')).reshape((jmax+1, 1, ftot, ftot))
        B = self.input.v('B', range(0, jmax+1), [0], [0])
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        Kh = self.input.v('Kh', range(0, jmax+1), [0], [0])

        ################################################################################################################
        ## Second order closure
        ################################################################################################################
        u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        d['T'] = {}
        d['F'] = {}
        T0 = 0
        F0 = 0
        a0_til = np.zeros((jmax+1, 1, ftot))
        a0_til[:, :, fmax] = 1

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
                        d['T'][submod] = self.dictExpand(d['T'][submod], 'TM0', 'return')
                        d['T'][submod]['TM0']['return'] += tmp
                else:
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0_f0_res, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(tmp) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp

        ## T.5. - u0*c0*zeta0
        # Total
        c0_f0 = ny.eliminateNegativeFourier(ny.arraydot(c0[:, [0], Ellipsis], a0_til), 2)
        T0 += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0_f0, 2), zeta0, 2)

        # Decomposition
        uzeta = ny.complexAmplitudeProduct(u0[:, [0], :], zeta0, 2)
        d['T'] = self.dictExpand(d['T'], 'stokes', ['TM'+str(2*n) for n in range(0, fmax+1)] + ['AM'+str(2*n) for n in range(1, fmax+1)])
        # transport with residual availability
        c0_f0_res = ny.eliminateNegativeFourier(c0[:, 0, :, fmax], 1).reshape((jmax+1, 1, fmax+1))
        for n in range(0, fmax+1):
            tmp = np.zeros(c0_f0_res.shape, dtype=complex)
            tmp[:, :, n] = c0_f0_res[:, :, n]
            tmp = ny.complexAmplitudeProduct(uzeta, tmp, 2)[:, 0, 0]
            if any(tmp) > 10**-14:
                d['T']['stokes']['TM'+str(2*n)] += tmp          # TODO still add subdict ['drift']

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

        ## Solve    ################################################################################################
        ## Add all mechanisms & compute a0c
        from src.DataContainer import DataContainer
        dc = DataContainer(d)
        T_til = dc.v('T', range(0, jmax+1))
        F_til = dc.v('F', range(0, jmax+1))
        # print np.max(abs((dc.v('T', range(0, jmax+1))-T0[:, 0, 0])/(T0[:, 0, 0]+10**-10)))
        # print np.max(abs((dc.v('F', range(0, jmax+1))-F0[:, 0, 0])/(F0[:, 0, 0]+10**-10)))

        integral = -ny.integrate(T_til/(F_til+10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))

        # Boundary condition 1
        if self.input.v('boundary')=='average':
            astar = self.input.v('astar')
            A = astar * ny.integrate(B[:, 0, 0], 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B[:, 0, 0]*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))

        # Boundary condition 2
        elif self.input.v('boundary')=='concentration':
            csea = self.input.v('csea')
            c000 = np.real(ny.arraydot(c0_int, a0_til)[0,0,fmax])
            A = csea/c000*(self.input.v('grid', 'low', 'z', range(0, jmax+1))-self.input.v('grid', 'high', 'z', range(0, jmax+1)))

        a0c = A*np.exp(integral)
        a0xc = -T_til/F_til*a0c

        ################################################################################################################
        # Store in dict
        ################################################################################################################

        ################################################################################################################
        ## Plot
        ################################################################################################################
        st.configure()

        c0_pot = ny.eliminateNegativeFourier(ny.arraydot(c0, a0_til), 2)        # potential maximum concentration if f00=1
        c0_pot = np.mean(c0_pot, axis=1).reshape((jmax+1, 1, fmax+1))
        c0 = ny.eliminateNegativeFourier(ny.arraydot(c0, a0_til*a0c.reshape((jmax+1, 1, 1))), 2)
        c0 = np.mean(c0, axis=1).reshape((jmax+1, 1, fmax+1))

        # # PLOT IN ITERATIONS
        # plt.figure(100, figsize=(2,2))
        # if hasattr(self, 'iteration'):
        #     plt.subplot(2,1,1)
        #     plt.plot(self.iteration, np.max(np.real(d['f0'][:,0,0])), 'ko')
        # else:
        #     self.iteration =0
        # #

        # PLOTS IN X
        # if self.iteration>60:
        plt.figure(102, figsize=(2, 2))
        x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        plt.subplot(2,1,1)
        for n in range(0, fmax+1):
            p = plt.plot(x/1000., abs(d['f0'][:,0,n]), label='$f_{new}$')
            plt.plot(x/1000., abs(f[:,0,n]), '--', color=p[0].get_color(), label='$f_{old}$')
            plt.legend()
            plt.title('f0')

        plt.subplot(2,1,2)
        for n in range(0, fmax+1):
            p = plt.plot(x/1000., abs(c0[:,0,n]), label='c0')
            plt.plot(x/1000., abs(c0_pot[:,0,n]), '--', color=p[0].get_color(), label='c0 pot.')
            plt.legend()
            plt.title('c0')
        st.show()


        # st.show()

        # ################################################################################################################
        # # Fig 1: f0 and f1
        # ################################################################################################################
        # plt.figure(1, figsize=(1,3))
        # x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        # x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:,0,0]
        # plt.subplot(1,3,1)
        #
        # plt.plot(x,np.abs(a0_til[:, 0, 0]*a0c), 'b-')
        # plt.plot(x,np.abs(a0_til[:, 0, 1]*a0c), 'g-')
        # plt.plot(x,np.abs(a0_til[:, 0, 2]*a0c), 'r-')
        #
        # # plt.subplot(1,3,2)
        # plt.plot(x,np.abs(a1_til[:, 0, 0, 1]*a0c + a1_til[:, 0, 0, 2]*a0xc), 'b-')
        # plt.plot(x,np.abs(a1_til[:, 0, 1, 1]*a0c + a1_til[:, 0, 1, 2]*a0xc), 'g-')
        # plt.plot(x,np.abs(a1_til[:, 0, 2, 1]*a0c + a1_til[:, 0, 2, 2]*a0xc), 'r-')
        #
        # plt.subplot(1,3,3)
        # plt.plot(x,np.abs(a1_til[:, 0, 0, 0]), 'b:')
        # plt.plot(x,np.abs(a1_til[:, 0, 1, 0]), 'g:')
        # plt.plot(x,np.abs(a1_til[:, 0, 2, 0]), 'r:')
        #
        # ################################################################################################################
        # # Fig 4: u0 da
        # ################################################################################################################
        # plt.figure(4, figsize=(1, 2))
        # u = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1)) + self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid'))/ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, [-1], :]
        # for n in range(0, fmax+1):
        #     plt.plot(x, abs(u[:, 0, n]))
        #
        # ################################################################################################################
        # # Fig 5: transport contributions
        # ################################################################################################################
        # plt.figure(5, figsize=(1, 2))
        # plt.subplot2grid((1,8), (0, 0),colspan=7)
        #
        # tterms = {}
        # for key in dc.getKeysOf('T'):
        #     tterms[key] = np.linalg.norm(dc.v('T', key, range(0, jmax+1)), 1)
        # tterms = [i for i in sorted(tterms, key=tterms.get, reverse=True)]
        # tterms = tterms[:5]
        # for key in tterms:
        #     p = plt.plot(x/1000., dc.v('T', key, range(0, jmax+1)), label=key)
        #     try:
        #         plt.plot(x/1000., self.input.v('T', key, range(0, jmax+1)), 'o', label=key, color=p[0].get_color())
        #     except:
        #         pass
        # plt.plot(x/1000., dc.v('T', range(0, jmax+1)), 'k', label='Total')
        # plt.legend(bbox_to_anchor=(1.15, 1.05))
        # plt.title('Transport terms')
        # plt.xlabel('x (km)')
        # plt.ylabel('T')
        #
        # ################################################################################################################
        # # Fig 6: Diffusion contributions
        # ################################################################################################################
        # plt.figure(6, figsize=(1, 2))
        # plt.subplot2grid((1,8), (0, 0),colspan=7)
        # for key in dc.getKeysOf('F'):
        #     p = plt.plot(x/1000., dc.v('F', key, range(0, jmax+1)), label=key)
        #     try:
        #         plt.plot(x/1000., self.input.v('F', key, range(0, jmax+1)), 'o', label=key, color=p[0].get_color())
        #     except:
        #         pass
        # plt.plot(x/1000., dc.v('F', range(0, jmax+1)), 'k', label='Total')
        # plt.legend(bbox_to_anchor=(1.15, 1.05))
        # plt.title('Diffusive terms')
        # plt.xlabel('x (km)')
        # plt.ylabel('F')
        #
        #


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

    def convertAvailability(self, a, gamma, f):
        # 1. Convert f to time series and compute a
        f_time = np.concatenate((f, np.zeros((f.shape[0], f.shape[1], 100-f.shape[2]), dtype=complex)), 2)
        f_time = ny.invfft(f_time, 2)

        # construct a
        a = a.reshape((f.shape[0], f.shape[1], f.shape[2]*2-1))
        acomplex = np.zeros(f.shape, dtype=complex)
        acomplex += a[:, :, :(a.shape[-1]+1)/2]
        acomplex[:, :, 1:] += 1j*a[:, :, (a.shape[-1]+1)/2:]
        a_time = 0
        t = np.linspace(0, 1, 100)
        for n in range(0, acomplex.shape[-1]):
            a_time += np.real(acomplex[:, :, n].reshape((f.shape[0], f.shape[1], 1))*(np.exp(n*2*np.pi*1j*t)).reshape((1, 1, len(t))))
        f_appr = 1.-np.exp(-gamma*a_time)

        res = f_appr - f_time
        # print np.sum(np.sqrt(res**2))
        return np.sum(np.sqrt(res**2))

    def convertAvailability2(self, gamma, f):
        # 1. Convert f to time series and compute a
        f_time = np.concatenate((f, np.zeros((f.shape[0], f.shape[1], 100-f.shape[2]), dtype=complex)), 2)
        f_time = ny.invfft(f_time, 2)
        f_time = np.minimum(f_time, np.ones(f_time.shape)*.99999)
        a_obj = -1./gamma*np.log(1-f_time)
        a = ny.fft(a_obj, 2)[:, :, :f.shape[-1]]
        return a