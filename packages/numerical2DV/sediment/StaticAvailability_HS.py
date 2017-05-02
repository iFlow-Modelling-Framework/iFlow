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
from copy import copy
import matplotlib.pyplot as plt
import step as st
import scipy.sparse as sps
from erodibility_stock_relation import erodibility_stock_relation_der, erodibility_stock_relation


class StaticAvailability_HS:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10**-6
    TOL = 10**-4
    RELAX = 0.5

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        stop = False
        if hasattr(self, 'difference'):
            if self.difference < 0.0005*self.dt/(3600*24):
                stop = True
        return stop

    def run_init(self):
        d = {}
        d['ws0'] = self.input.v('ws00')
        self.firstiteration = True

        return d

    def run(self):
        f0uncap, T_til, F_til, fsea, d = self.transport()
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        B = self.input.v('B', range(0, jmax+1))
        c0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0 = self.input.v('hatc1', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0x = self.input.v('hatc1', 'ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c2 = self.input.v('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        cint = ny.integrate(c0[:, :, 0], 'z', kmax, 0, self.input.slice('grid'))[:, 0]

        ################################################################################################################
        # time step
        ################################################################################################################
        if self.firstiteration == True:
            self.firstiteration = False

            self.x = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]
            self.gamma = 1.
            self.FSEA = fsea
            self.FCAP = 1
            self.FINF = 1
            self.dt = 0
            self.dx = self.x[1:] - self.x[:-1]
            # check seaward boundary condition
            if self.FSEA>1:
                self.logger.warning('Sediment concentration cannot satisfy the prescribed boundary condition at x=0.\n'
                                    'Setting the seaward concentration to its maximum.')
                self.FSEA = 0.999

            self.aeq = f0uncap / f0uncap[0]  # scale the availability such that aeq=1 at x=0

            ##### Initialise solution vector X  ######
            self.ALPHA2 = 0.65*np.ones(self.x.shape)

            # Solution matrix X = (g, S)
            X = self.init_stock(f0uncap)
            ALPHA1 = np.real(cint)
            self.S = X[jmax+1:]*ALPHA1

            self.f0 = self.aeq * X[:len(self.x)]
            self.difference = np.inf
            d['ws0'] = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        else:
            # time step
            A = ny.derivative(B*F_til, 'x', self.input.slice('grid'))/B
            ALPHA1 = np.real(cint) #np.ones(self.x.shape)
            Shat = self.S/ALPHA1
            f = erodibility_stock_relation(self.ALPHA2, Shat)
            g = f*f0uncap[0]/(f0uncap+10**-10)
            X = np.concatenate((g, Shat), 0)

            Xvec, jac = self.jacvec_stock(A, A, T_til, T_til, F_til, F_til, self.aeq, self.aeq, X, X, ALPHA1, ALPHA1)
            dX = np.linalg.solve(jac, Xvec)
            X = X - dX
            self.S = X[jmax+1:]*ALPHA1

            self.difference = max(abs(self.aeq * X[:len(self.x)]-self.f0)/abs(self.f0+10**-3*np.max(self.f0)))
            self.f0 = self.aeq * X[:len(self.x)]

        # compute derivative of f
        f0x = ny.derivative(self.f0, 'x', self.input.slice('grid'))

        ################################################################################################################
        # Store in dict
        ################################################################################################################
        d['a'] = f0uncap
        d['f'] = self.f0
        d['c0'] = c0*self.f0.reshape((jmax+1, 1, 1))
        d['c1'] = c1_a0*self.f0.reshape((jmax+1, 1, 1)) + c1_a0x*f0x.reshape((jmax+1, 1, 1))
        d['c2'] = c2*self.f0.reshape((jmax+1, 1, 1))

        # hindered settling
        wsold = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        d['ws0'] = self.hindered_settling(d['c0'])
        if np.max(abs(wsold-d['ws0'])/abs(wsold+0.1*self.input.v('ws00'))) < 0.01:
            self.dt = 24*3600*0.5
        else:
            self.dt = 0

        print self.difference, self.dt>0, np.max(abs(d['c0'][:, :, 0]))

        if np.max(abs(d['c0'][:, :, 0]))>10:
            from plot_paper.boundary_solver import boundary_solver
            from erosion import erosion
            import scipy.optimize
            import step as st
            import matplotlib.pyplot as plt

            bs = boundary_solver()
            phi_vec = np.linspace(0, 1, 100)
            ws_vec = self.input.v('ws00')*(1.-phi_vec)**self.input.v('mhs')
            Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            sol=[]
            for phi in phi_vec:
                sol.append(scipy.optimize.fsolve(bs.boundary_switch, 0.3, (phi, 'RZ', 5, False, 0)))
            Etil = erosion(d['ws0'], Av, 0, self.input, method='Partheniades')/(self.input.v('cgel')*self.input.v('ws00'))
            f = self.input.v('f', range(0, jmax+1))

            st.configure()
            plt.figure(1, figsize=(1,2))
            plt.subplot(1,2,1)
            plt.plot(sol, ws_vec)
            plt.plot(Etil[:, -1, 0]*self.f0, wsold[:, -1, 0], 'g.')
            plt.plot(Etil[:, -1, 0]*self.f0, d['ws0'][:, -1, 0], 'r.')
            plt.xlabel(r'$\tilde{E}$')
            plt.ylabel(r'$w_s$')
            plt.subplot(1,2,2)
            plt.plot(abs(wsold[:, -1, 0]))
            st.show()

        # if np.max(abs(d['c0'][:, :, 0]))> 10:
        #     st.configure()
        #     plt.figure(1, figsize=(2,2))
        #     plt.subplot(2,2,1)
        #     plt.plot(range(0, jmax+1), d['ws0'][:, 0, 0])
        #     plt.title('ws')
        #     plt.subplot(2,2,2)
        #     plt.plot(range(0, jmax+1), d['c0'][:, -1, 0])
        #     plt.title('c0')
        #     plt.subplot(2,2,3)
        #     plt.plot(range(0, jmax+1), d['f'])
        #     plt.plot(range(0, jmax+1), d['a'])
        #     plt.ylim(0, 1)
        #     plt.title('f0')
        #     plt.subplot(2,2,4)
        #     plt.plot(range(0, jmax+1), X[jmax+1:]*ALPHA1)
        #     plt.title('S')
        #     st.show()

        return d

    def hindered_settling(self, c):
        # Load data
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c[:, :, 1:] = 0     # TODO
        cgel = self.input.v('cgel')
        mhs = self.input.v('mhs')
        ws0 = self.input.v('ws00')
        wsmin = self.input.v('wsmin')

        # convert to time series
        c = np.concatenate((c, np.zeros((jmax+1, kmax+1, 100))), 2)
        c = ny.invfft(c, 2)

        # Richardson & Zaki 1954 formulation
        phi = c/cgel
        phi = np.maximum(np.minimum(phi, 1), 0)
        ws = np.maximum(ws0*(1.-phi)**mhs, wsmin)
        ws = ny.fft(ws, 2)
        ws = ws[:, :, :fmax+1]
        ws[:, :, 1:] = 0

        # ws_old = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        # self.difference = np.linalg.norm(np.sum(np.abs((ws_old-ws)/((ws_old+0.1*ws0)[:, :, [0]])), axis=-1), np.inf)
        # ws0 = (1-self.RELAX)*ws + self.RELAX*ws_old

        return ws

    def transport(self):
        self.logger.info('Running module StaticAvailability')

        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0 = self.input.v('hatc1', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0x = self.input.v('hatc1', 'ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        d = {}

        c0_int = ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
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

        ## Transport T  ############################################################################################
        ## T.1. - u0*c1_a0
        # Total
        c1a_f0 = c1_a0
        T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1a_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('hatc1', 'a'):
            if submod == 'erosion':
                for subsubmod in self.input.getKeysOf('hatc1', 'a', 'erosion'):
                    c1_a0_comp = self.input.v('hatc1', 'a', submod, subsubmod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                    c1a_f0_comp_res = c1_a0_comp
                    d['T'] = self.dictExpand(d['T'], subsubmod, ['TM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
                    # transport with residual availability
                    for n in range(0, fmax+1):
                        tmp = np.zeros(c1a_f0_comp_res.shape, dtype=complex)
                        tmp[:, :, n] = c1a_f0_comp_res[:, :, n]
                        tmp = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                        if any(abs(tmp)) > 10**-14:
                            d['T'][subsubmod]['TM'+str(2*n)] += tmp
            else:
                c1_a0_comp = self.input.v('hatc1', 'a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                c1a_f0_comp_res = c1_a0_comp
                d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
                # transport with residual availability
                for n in range(0, fmax+1):
                    tmp = np.zeros(c1a_f0_comp_res.shape, dtype=complex)
                    tmp[:, :, n] = c1a_f0_comp_res[:, :, n]
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp

        ## T.2. - u1*c0
        # Total
        T0 += ny.integrate(ny.complexAmplitudeProduct(u1, c0, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('u1'):
            u1_comp = self.input.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)]) # add submod index to dict if not already
            # transport with residual availability
            for n in range(0, fmax+1):
                tmp = np.zeros(u1_comp.shape, dtype=complex)
                tmp[:, :, n] = u1_comp[:, :, n]
                if submod == 'stokes':
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod] = self.dictExpand(d['T'][submod], 'TM'+str(2*n), ['return', 'drift'])
                        d['T'][submod]['TM0']['return'] += tmp
                else:
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp

        ## T.5. - u0*c0*zeta0
        # Total
        T0 += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0[:, [0], :], 2), zeta0, 2)

        # Decomposition
        uzeta = ny.complexAmplitudeProduct(u0[:, [0], :], zeta0, 2)
        d['T'] = self.dictExpand(d['T'], 'stokes', ['TM'+str(2*n) for n in range(0, fmax+1)])
        # transport with residual availability
        for n in range(0, fmax+1):
            tmp = np.zeros(c0[:, [0], :].shape, dtype=complex)
            tmp[:, :, n] = c0[:, [0], n]
            tmp = ny.complexAmplitudeProduct(uzeta, tmp, 2)[:, 0, 0]
            if any(abs(tmp)) > 10**-14:
                d['T']['stokes']['TM'+str(2*n)]['drift'] += tmp

        ## T.6. - u1riv*c2rivriv
        c2 = self.input.v('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u1riv = self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        if u1riv is not None:
            d['T'] = self.dictExpand(d['T'], 'river_river', 'TM0')  # add submod index to dict if not already
            tmp = ny.integrate(ny.complexAmplitudeProduct(u1riv, c2, 2), 'z', kmax, 0, self.input.slice('grid'))
            if any(abs(tmp[:, 0,0])) > 10**-14:
                d['T']['river_river']['TM0'] = tmp[:, 0,0]

            T0 += tmp

        ## T.7. - diffusive part
        # Total
        c0x = self.input.d('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        T0 += - Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid'))

        c2x = self.input.d('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        T0 += - Kh*ny.integrate(c2x, 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        d['T'] = self.dictExpand(d['T'], 'diffusion_tide', ['TM0'])
        d['T'] = self.dictExpand(d['T'], 'diffusion_river', ['TM0'])
        # transport with residual availability
        tmp = - (Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['T']['diffusion_tide']['TM0'] = tmp
        tmp = - (Kh*ny.integrate(c2x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['T']['diffusion_river']['TM0'] = tmp

        ## Diffusion F  ############################################################################################
        ## F.1. - u0*C1ax*f0
        # Total
        F0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1_a0x, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('hatc1', 'ax'):
            c1_ax0_comp = self.input.v('hatc1', 'ax', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['F'] = self.dictExpand(d['F'], submod, ['FM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
            # transport with residual availability
            for n in range(0, fmax+1):
                tmp = np.zeros(u0.shape, dtype=complex)
                tmp[:, :, n] = u0[:, :, n]
                tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c1_ax0_comp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                if any(abs(tmp)) > 10**-14:
                    d['F'][submod]['FM'+str(2*n)] += tmp

        ## F.3. - diffusive part
        # Total
        F0 += - Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
        F0 += - Kh*ny.integrate(c2, 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        d['F'] = self.dictExpand(d['F'], 'diffusion_tide', ['FM0'])
        d['F'] = self.dictExpand(d['F'], 'diffusion_river', ['FM0'])
        # transport with residual availability
        tmp = - (Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['F']['diffusion_tide']['FM0'] = tmp
        tmp = - (Kh*ny.integrate(c2, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['F']['diffusion_river']['FM0'] = tmp

        ## Solve    ################################################################################################
        ## Add all mechanisms & compute a0c
        from src.DataContainer import DataContainer
        dc = DataContainer(d)
        dc.merge(self.input.slice('grid'))
        T_til = np.real(dc.v('T', range(0, jmax+1)))
        F_til = np.real(dc.v('F', range(0, jmax+1)))

        integral = -ny.integrate(T_til/(F_til-10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))

        ################################################################################################################
        # Boundary condition 1
        ################################################################################################################
        if self.input.v('sedbc')=='astar':
            astar = self.input.v('astar')
            k = astar * ny.integrate(B[:, 0, 0], 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B[:, 0, 0]*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))

            f0uncap = k*np.exp(integral)

        ################################################################################################################
        # Boundary condition 2
        ################################################################################################################
        elif self.input.v('sedbc')=='csea':
            csea = self.input.v('csea')
            c000 = np.real(c0_int[0,0,0])
            k = csea/c000*(self.input.v('grid', 'low', 'z', 0)-self.input.v('grid', 'high', 'z', 0))

            ################################################################################################################
            # Determine equilibria (only if BC is csea)
            ################################################################################################################
            f0uncap = k*np.exp(integral)

        else: # incorrect boundary description
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('incorrect seaward boundary type (sedbc) for sediment module')

        return f0uncap, T_til, F_til, k, d

    def dictExpand(self, d, subindex, subsubindices):
        if not subindex in d:
            d[subindex] = {}
        elif not isinstance(d[subindex], dict):
            d[subindex] = {}
        for ssi in ny.toList(subsubindices):
            if not ssi in d[subindex]:
                d[subindex][ssi] = 0
        return d

    def init_stock(self, f0uncap):
        # Initialize vector g
        jmax = self.input.v('grid', 'maxIndex', 'x')
        f = self.input.v('f0', range(0, jmax+1))
        if f is None:
            Lb = -self.x/np.log(1./f0uncap)
            Lb = Lb[np.where(Lb>0)]
            if len(Lb) > 0:
                Lb = np.min(Lb)*0.9
                g = f0uncap[0]*np.exp(-self.x/Lb)
            else:
                g = f0uncap[0]*np.ones(self.x.shape)

            # define initial erodibility f
            f = f0uncap/f0uncap[0] * g
        else:
            g = f*f0uncap[0]/(f0uncap+10**-10)

        # Initiate stock S with exact expression (see notes Yoeri)
        Shat = f
        F = erodibility_stock_relation(self.ALPHA2, Shat) - f

        # Newton-Raphson iteration towards actual Shat
        i = 0
        while max(abs(F)) > self.TOL and i < 50:
            i += 1
            dfdS = erodibility_stock_relation_der(self.ALPHA2, Shat)
            Shat = Shat - F / (dfdS+10**-12)
            F = erodibility_stock_relation(self.ALPHA2, Shat) - f
            if i == 50:
                g = erodibility_stock_relation(self.ALPHA2, Shat)*f0uncap[0]/f0uncap
        # catch problems with infinite initial stock
        Smax = np.array([0.999])
        F = erodibility_stock_relation(self.ALPHA2[[0]], Smax) - np.array([0.999])
        while max(abs(F)) > self.TOL:
            dfdS = erodibility_stock_relation_der(self.ALPHA2[[0]], Smax)
            Smax = Smax - F / dfdS
            F = erodibility_stock_relation(self.ALPHA2[[0]], Smax) - np.array([0.999])
        Shat = np.minimum(Shat, Smax)

        # Define solution vector X = (g, S)
        X = np.append(g, Shat).reshape(2 * len(self.x))
        return X

    def jacvec_stock(self, A, A_old, T, T_old, F, F_old, aeq, aeq_old, X, X_old, alpha1_old, alpha1):
        """Calculates the vector containing the Exner equation and the algebraic expression between S and f, and the
        Jacobian matrix needed to solve for g and S

        Parameters:
            A, A_old - variable (BF)_x/B on the old and current time step
            T, T_old - transport function T on the old and current time step
            F, F_old -  diffusion furnction F on the old and current time step
            aeq, aeq_old - availability in morphodynamic equilibrium on the old and current time step
            X, X_old - solution vector X = (g, a) on the old and current time step
            alpha1, alpha1_old - alpha1 on the old and current time step
            fr, fr_old - erodibility f due to river transport term on the old and current time step

        Returns:
             Xvec - vector containing the Exner equation and the algebraic expression between S and f
             jac - corresponding Jacobian matrix
        """
        # Initiate jacobian matrix, Xvec and local variable N
        jac = sps.csc_matrix((len(X), len(X)))
        Xvec, N = np.zeros((len(X))), np.zeros((len(self.x)))

        # define length of xgrid
        lx = len(self.x)
        ivec = range(1, lx-1)  #interior points for g
        jvec = range(lx+1, 2*lx-1)  #interior points for S

        # Define local variable N^(n+1) = f_inf * a_eq * [F*(g - 2g + g) / dx**2 + (A - T) * (g - g) / 2 * dx]
        N[ivec] = self.FINF * aeq[1:-1] * (F[1:-1] * (X[2:lx] - 2 * X[ivec] + X[:lx-2]) / self.dx[0]**2 +
                                            (A[1:-1] - T[1:-1]) * (X[2:lx] - X[:lx-2]) / (2 * self.dx[0]))
        # N[ivec] = self.FINF * aeq[1:-1] * (F[1:-1] * (X[2:lx] - 2 * X[ivec] + X[:lx-2]) / self.dx[0]**2 +
        #                                     (A[1:-1] - T[1:-1]) * (X[1:lx-1] - X[:lx-2]) / (self.dx[0]))
        # Fill interior points of Xvec related to N in Eq. (1)
        Xvec[ivec] += self.dt * N[ivec] / (2. * alpha1[ivec])

        # # Fill interior points of Xvec related to river flux
        # Xvec[ivec] -= self.dt * self.gamma * self.input.v('rivertrans') * (X[2:lx] - X[:lx-2]) / (2 * self.B[ivec] * self.dx[0])

        # Fill interior points of the jacobian matrix related to g for Eq. (1)
        jval_c_g = - self.dt * self.FINF * aeq[1:-1] * F[1:-1] / (self.dx[0]**2 * alpha1[1:-1])
        jac += sps.csc_matrix((jval_c_g, (ivec, ivec)), shape=jac.shape)
        jval_l_g = (self.dt * self.FINF * aeq[1:-1] * (F[1:-1] / self.dx[0] - (A[1:-1] - T[1:-1]) / 2.) /
                    (2 * alpha1[1:-1] * self.dx[0]))
        # jval_l_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] - (A[1:-1, 0] - T[1:-1, 0]) / 2.) +
        #                                    self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_l_g, (ivec, range(lx-2))), shape=jac.shape)
        jval_r_g = self.dt * self.FINF * aeq[1:-1] * (F[1:-1] / self.dx[0] + (A[1:-1] - T[1:-1]) / 2.) / \
                   (2 * alpha1[1:-1] * self.dx[0])
        # jval_r_g = self.dt * self.gamma * (aeq[1:-1, 0] * (F[1:-1, 0] / self.dx[0] + (A[1:-1, 0] - T[1:-1, 0]) / 2.) -
        #                         self.input.v('rivertrans') / self.B[ivec, 0]) / (2 * self.dx[0])
        jac += sps.csc_matrix((jval_r_g, (ivec, range(2, lx))), shape=jac.shape)

        # Boundary condition at sea
        Xvec[0] = X[0] - self.FSEA
        jac += sps.csc_matrix(([1.], ([0.], [0.])), shape=jac.shape)
        # Boundary condition at weir
        Xvec[lx-1] = (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3])
        jac += sps.csc_matrix(([3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)

        # Xvec[lx-1] = self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) + 2. * self.dx[0] * self.input.v('rivertrans')
        # jac += sps.csc_matrix((self.B[-1] * F[-1] * aeq[-1] * [3., -4., 1.], ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)
        # Xvec[lx-1] = (self.B[-1] * F[-1] * aeq[-1] * (3. * X[lx-1] - 4. * X[lx-2] + X[lx-3]) -
        #               2. * self.dx[0] * self.input.v('rivertrans') * (X[lx-1] - 1))
        # jval_g = self.B[-1] * F[-1] * aeq[-1] * [(3. - 2. * self.dx[0] * self.input.v('rivertrans'))[0], -4., 1.]
        # jac += sps.csc_matrix((jval_g, ([lx-1, lx-1, lx-1], [lx-1, lx-2, lx-3])), shape=jac.shape)

        # Fill interior points of Xvec and diagonal of jacobian related to S in Eq. (1)
        Xvec[ivec] += X[jvec]
        jac += sps.csc_matrix((np.ones(lx-2), (ivec, jvec)), shape=jac.shape)

        # Fill Xvec and jacobian for the algebraic relation between g and stock S when using the exact relation between S and f
        Xvec[lx:] = aeq * X[:lx] - erodibility_stock_relation(self.ALPHA2, X[lx:])
        jac += sps.csc_matrix((-erodibility_stock_relation_der(self.ALPHA2, X[lx:]), (range(lx, 2*lx), range(lx, 2*lx))), shape=jac.shape)
        jac += sps.csc_matrix((aeq, (range(lx, 2*lx), range(lx))), shape=jac.shape)

        # Convert jacobian matrix to dense matrix
        jac = jac.todense()
        # Inhomogeneous part of the PDE related to g and S at interior points
        Xvec[ivec] -= alpha1_old[ivec] * X_old[jvec] / alpha1[ivec]
        N[ivec] = self.FINF * aeq_old[1:-1] * (F_old[1:-1] * (X_old[2:lx] - 2 * X_old[ivec] + X_old[:lx-2]) / self.dx[0]**2 +
                                                (A_old[1:-1] - T_old[1:-1]) * (X_old[2:lx] - X_old[:lx-2]) / (2 * self.dx[0]))
        # N[ivec] = self.FINF * aeq_old[1:-1] * (F_old[1:-1] * (X_old[2:lx] - 2 * X_old[ivec] + X_old[:lx-2]) / self.dx[0]**2 +
        #                                         (A_old[1:-1] - T_old[1:-1]) * (X_old[1:lx-1] - X_old[:lx-2]) / (self.dx[0]))
        Xvec[ivec] += self.dt * N[ivec] / (2. * alpha1[ivec])

        return Xvec, jac