"""
Fully non-linear truncation model for salinity

Date latest update: 03-01-2023
Authors: Y.M. Dijkstra
"""
import scipy.integrate
import scipy.linalg
import scipy.interpolate
import scipy.optimize
import logging
import numpy as np
from copy import copy,deepcopy

from .tools.toBandedTri import *
from .tools.discretiseOperatorHnonequi import discretiseOperator
from .tools.discretiseTensorHnonequi import discretiseTensor, TensorFluxSplitting
from .tools.merge_space_time import merge_space_time

from src.DataContainer import DataContainer
import nifty as ny


class SalinityTruncationNewton:
    # Variables
    TOLERANCE = 1e-5
    MAXITER = 60   # Maximum number of iterations per newton-raphson step
    RELAX = 1       # Relaxation parameter. 1=no relaxation

    NEWTON = 1.     # Newton iteration (1) or Picard (0)
    MOMADV = 1.     # Include momentum advection (1)
    W = 1.          # Include w-terms (1)
    SALADV = 1.     # Include salinity advection (1)
    counter = 0
    
    dxmax_frac = 1/10.#1./50.       # maximum grid spacing as a fraction of total length
    dxmin_frac = 1/30.#1./20.       # minimum grid spacing as a fraction of total length/jmax
    Avw = 0 #0.1                    # background mixing at the weir

    # The next settings can be overwritten from the inputfile
    SEABC = 'adaptive'          #'dirichlet'
    gridadaptation = 'semi'     # 'full'
    tide_type = 'discharge'     #'discharge' # 'amplitude'

    timers = [ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer(), ny.Timer()]
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        if self.convergedToGoal:
            return True
        elif self.input.v('continuationParam') is None or self.input.v('continuationParam') == 'None':
            return True
        else:
            return False

    def run_init(self):
        self.logger.info('Salt Truncation model - iteration 0')
        ################################################################################################################
        ## Init
        ################################################################################################################
        ## grid
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        self.M = int(self.input.v('M'))     # number of vertical elements
        self.F = self.input.v('grid', 'maxIndex', 'f')+1
        self.ftot = 2*self.F-1

        ## Switches
        #   Grid adaptation
        if self.input.v('gridadaptation_init') is not None:
            self.gridadaptation = self.input.v('gridadaptation_init')
        elif self.input.v('gridadaptation') is not None:
            self.gridadaptation = self.input.v('gridadaptation')

        #   determine whether to run certain terms
        if self.input.v('MOMADV') is not None:      # advection in the momentum equation
            self.MOMADV = self.input.v('MOMADV')
        if self.input.v('SALADV') is not None:      # advection in the salinity equation
            self.SALADV = self.input.v('SALADV')
        if self.input.v('W') is not None:           # Include w terms
            self.W = self.input.v('W')

        ## Continuation settings
        self.stepsize = self.input.v('stepsize')
        self.stepsize_original = self.input.v('stepsize')
        self.goal = self.input.v('continuationGoal')
        if self.goal == 'False':
            self.convergedToGoal = True
        else:
            self.convergedToGoal = False

        ################################################################################################################
        # Create container with parameters
        ################################################################################################################
        self.load_params()

        ################################################################################################################
        # Compute eigenvalues and eigenfunctions
        ################################################################################################################
        H = self.input.v('H', x=0, z=0, f=0)
        sf0 = np.real(self.dc.v('Roughness', x=0, z=0, f=0))
        Av0 = np.real(self.dc.v('Av', x=0, z=0, f=0))
        R = Av0/(sf0*H)
        self.lm = self.eigenvals_velocity(self.M, R).reshape((1, 1, self.M, 1))

        # Initialise eigenfunctions
        eta = -self.input.v('grid', 'axis', 'z')
        m_arr = np.arange(0, self.M).reshape((1, 1, self.M, 1))
        self.f1 = (-1)**m_arr*np.cos(m_arr*np.pi*eta.reshape((1, kmax+1, 1, 1)))                            # Eigenfunctions
        self.f2 = (-1)**m_arr*np.cos(self.lm*eta.reshape((1, kmax+1, 1, 1)))                                # Eigenfunctions
        self.pf2 = (-1)**m_arr*1/self.lm*(np.sin(self.lm*eta.reshape((1, kmax+1, 1, 1))) + np.sin(self.lm)) # Eigenfunctions - primitive

        ################################################################################################################
        # Compute Galerkin coefficients
        ################################################################################################################
        G1, G1b, G2, G3, G4, G5, G6, G6b, G7, G8, G9 = self.GalCoef(R)          # vertical
        H1, H2, H3 = self.GalCoef_time()                                        # temporal
        H3G2_p, H3G2_m, H3G3_p, H3G3_m, H3G7_p, H3G7_m, H3G8_p, H3G8_m = self.flux_splitting(merge_space_time(G2,H3), merge_space_time(G3,H3), merge_space_time(G7,H3), merge_space_time(G8,H3))    # flux splitting

        Galkeys = ('G1', 'G1b', 'G2', 'G3', 'G4', 'G5', 'G6', 'G6b', 'G7', 'G8', 'G9',
                   'H3G2_p', 'H3G2_m', 'H3G3_p', 'H3G3_m', 'H3G7_p', 'H3G7_m', 'H3G8_p', 'H3G8_m',
                   'H1', 'H2', 'H3')
        Galvalues = (G1, G1b, G2, G3, G4, G5, G6, G6b, G7, G8, G9,
                     H3G2_p, H3G2_m, H3G3_p, H3G3_m, H3G7_p, H3G7_m, H3G8_p, H3G8_m,
                     H1, H2, H3)
        self.GalMat = {}
        for i, k in enumerate(Galkeys):
            self.GalMat[k] = Galvalues[i]

        ################################################################################################################
        # Based on input, decide to initialise from scratch or try to search if hotstart if possible
        ################################################################################################################
        init_new = self.input.v('init')
        a = self.input.v('alpha')
        b = self.input.v('beta')
        c = self.input.v('zeta')

        if init_new == 'True' or a is None or b is None or c is None:
            x = self.input.v('grid', 'axis', 'x')
            xfull = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]

            self.logger.info('\tStarting from scratch')
            alpha = np.zeros((jmax+1, self.M, self.F), dtype=complex)
            beta = np.zeros((jmax+1, self.M, self.F), dtype=complex)
            zeta = np.zeros((jmax+1, 1, self.F), dtype=complex)
        else:
            x = self.input.v('adaptive_grid', 'axis', 'x')
            if len(x)==jmax+1:
                xfull = self.input.v('adaptive_grid', 'axis', 'x')*self.input.v('adaptive_grid', 'high', 'x')
            else:
                x = self.input.v('grid', 'axis', 'x')*self.input.v('grid', 'high', 'x')/self.input.v('adaptive_grid', 'high', 'x')
                xfull = self.input.v('grid', 'axis', 'x')*self.input.v('grid', 'high', 'x')
            # alpha = self.input.v('alpha', x=x, z=0, f=0).reshape((jmax+1, self.M, self.F))
            # beta = self.input.v('beta', x=x, z=0, f=0).reshape((jmax+1, self.M, self.F))
            # zeta = self.input.v('zeta', x=x, z=0, f=range(0, self.F)).reshape((jmax+1, 1, self.F))
            alpha_load = self.input.v('alpha', x=x, z=0, f=0)
            beta_load = self.input.v('beta', x=x, z=0, f=0)
            zeta_load = self.input.v('zeta', x=x, z=[0], f=range(0, beta_load.shape[-1]))
            alpha = np.zeros((jmax+1, self.M, self.F), dtype=complex)
            beta = np.zeros((jmax+1, self.M, self.F), dtype=complex)
            zeta = np.zeros((jmax+1, 1, self.F), dtype=complex)
            Fold = beta_load.shape[-1]

            alpha[:, :, :np.minimum(Fold, self.F)] = alpha_load[:, :, :np.minimum(Fold, self.F)]
            beta[:, :, :np.minimum(Fold, self.F)] = beta_load[:, :, :np.minimum(Fold, self.F)]
            zeta[:, :, :np.minimum(Fold, self.F)] = zeta_load[:, :, :np.minimum(Fold, self.F)]

        y = self.fold((alpha, beta, zeta,))

        ################################################################################################################
        # Find/confirm solution for these initial conditions
        ################################################################################################################
        d = self.solve(y, xfull, None)

        # verify that solution is indeed correct
        # self.input.merge(d)
        # x = self.input.v('adaptive_grid', 'axis', 'x')
        # alpha = self.input.v('alpha', x=x)
        # beta = self.input.v('beta', x=x)
        # zetax = self.input.v('zetax', x=x).reshape((jmax+1, 1, 1, 1))
        # y = np.concatenate((alpha, beta, zetax/self.scaling_zetax), axis=-1).flatten()
        # d2 = self.solve(self.GalMat, y, (self.Av0, self.Kv0, self.sf0))

        return d

    def run(self):
        ## Init
        self.logger.info('Salt Truncation model - iteration '+str(self.iteration))

        # grid
        if self.input.v('gridadaptation') is not None:
            self.gridadaptation = self.input.v('gridadaptation')
        jmax = self.input.v('grid', 'maxIndex', 'x')
        x = self.input.v('adaptive_grid', 'axis', 'x')
        L = self.input.v('L')

        # previous solution
        alpha = self.input.v('alpha', x=x, z=0, f=0).reshape((jmax+1, self.M, self.F))
        beta = self.input.v('beta', x=x, z=0, f=0).reshape((jmax+1, self.M, self.F))
        zeta = self.input.v('zeta', x=x, z=0, f=range(0, self.F)).reshape((jmax+1, 1, self.F))
        y = self.fold((alpha, beta, zeta,))

        # continuation settings
        continuationparam = self.input.v('continuationParam')

        ## Run
        # d = self.solve(y, x*L, continuationparam)
        self.step = 0
        while self.step < 5e-4: #0.1:
            try:
                d = self.solve(y, x*L, continuationparam)
                self.step += abs(self.stepsize)
            except Exception as e:
                self.stepsize = self.stepsize/2.
                if abs(self.stepsize) < 5e-4:
                    self.logger.info('\tERROR: stepsize too small to continue. Ending the continuation.')
                    self.convergedToGoal = True
                    self.step = np.inf
                    return {}
                else:
                    self.logger.info(('\tERROR: re-trying on a smaller stepsize: '+str(self.stepsize)))
                    self.load_params()
                    d = self.run()
        return d

    def solve(self, y, x, param):
        d = {}

        ## Continuation step and Newton-Raphson iteration
        y, y_old, x, p = self.continuation(y, x, param)
        d.update(p)

        ## Decomposition
        self.timers[8].tic()
        d.update(self.decomposition(y, y_old, x))
        self.timers[8].toc()

        ## Timing
        self.timers[0].disp('Adaptive grid - total')
        self.timers[1].disp('Step')
        self.timers[2].disp('Build matrices invariant part')
        self.timers[3].disp('Build matrices variant part')
        self.timers[4].disp('Add matrices')
        self.timers[5].disp('Preconditioning')
        self.timers[6].disp('Solve')
        self.timers[7].disp('Difference and update')
        self.timers[8].disp('Decomposition')

        return d


    def continuation(self, y, x, param):
        converged = False
        jmax = self.input.v('grid', 'maxIndex', 'x')
        p = None

        ########################################################################################################################
        ## 1. Step
        ########################################################################################################################
        self.timers[0].tic()
        if self.gridadaptation == 'semi':
            x, y = self.setgrid(x, y, relaxation=1)
        self.timers[0].toc()

        self.timers[1].tic()
        if param is not None:
            ## Load parameter value
            p = self.get_param(param)
            # p0 = copy(p)
            # y0 = copy(y)

            ## Jacobian
            MatbandedInvar, _ = self.Jac_invar(x)
            Matbanded_adv, Matbanded_adv_var = self.Jac_var(y, x)
            Jac = MatbandedInvar + Matbanded_adv + Matbanded_adv_var
            Jac_p = self.Jac_param(y, x, param)

            ## Solve
            z2 = - scipy.linalg.solve_banded((3*(self.M*2+1)*(self.ftot)-1, 3*(self.M*2+1)*(self.ftot)-1), Jac, Jac_p.flatten())

            norm = np.sqrt(self.norm_y(x,z2,z2)+1)

            ## Set derivatives wrt branch
            y0s = z2/norm
            p0s = 1./norm

            ## Update
            self.dlambda = self.stepsize*np.abs(p/p0s)
            y = y + self.dlambda*y0s
            p = p + self.dlambda*p0s
            self.update_param(param, p)

            # plt.figure(1, figsize=(1,2))
            # plt.plot(p, self.norm_y(x,y, y), 'r.')
            # plt.text(p, self.norm_y(x,y, y), str(self.counter))
            #
            # plt.figure(2, figsize=(1,2))
            # plt.plot(p, p0s*(p-p0)+self.norm_y(x,y0s, y-y0), 'ro')
            # plt.plot(p, ((p-p0))**2/self.dlambda+self.norm_y(x,(y-y0), (y-y0)/self.dlambda), '.k')
        self.timers[1].tic()

        ################################################################################################################
        ## 2. Newton iteration
        ################################################################################################################
        iteration = 0
        while not converged:
            ############################################################################################################
            ## 2a. Grid adaptation
            ############################################################################################################
            if self.gridadaptation == 'full':
                x, y = self.setgrid(x, y)

            ############################################################################################################
            ## 2b. Newton-Raphson
            ############################################################################################################
            self.timers[2].tic()
            if iteration==0 or self.gridadaptation=='full':
                ## Jacobian - invariant part
                MatbandedInvar, rhs = self.Jac_invar(x)         # possible to compute only in first iteration since parameter is kept at same value. In (pseudo) arclength continuation this should be computed each iteration.
            self.timers[2].toc()

            ## Jacobian - varying part
            self.timers[3].tic()
            Matbanded_adv, Matbanded_adv_var = self.Jac_var(y, x)
            self.timers[3].toc()

            self.timers[4].tic()
            ## Prepare matrices
            Matbanded = MatbandedInvar + Matbanded_adv
            Matbanded_var = Matbanded_adv_var

            residual = bandedMatVec(Matbanded, y)-rhs
            Jac = Matbanded + self.NEWTON*Matbanded_var
            self.timers[4].toc()

            ## Precondition
            self.timers[5].tic()
            Pvect = 1./bandedMatVec(np.abs(Jac), np.ones((Jac.shape[1])))
            P = np.zeros(Jac.shape)
            P[3*(self.M*2+1)*(self.ftot)-1, :] = Pvect
            for i in range(1, 3*(self.M*2+1)*(self.ftot)):
                P[3*(self.M*2+1)*(self.ftot)-1-i,i:]=Pvect[:-i]
                P[3*(self.M*2+1)*(self.ftot)-1+i,:-i]=Pvect[i:]
            Jac = Jac*P
            residual = residual*Pvect
            self.timers[5].toc()

            ## Solve
            self.timers[6].tic()
            z1 = - scipy.linalg.solve_banded((3*(self.M*2+1)*(self.ftot)-1, 3*(self.M*2+1)*(self.ftot)-1), Jac, residual)
            self.timers[6].toc()
            dy = z1

            # if param is not None:
            #     Jac_p = self.Jac_param(y, x, param)
            #     z2 = - scipy.linalg.solve_banded((3*(self.M*2+1)*(self.ftot)-1, 3*(self.M*2+1)*(self.ftot)-1), Jac, Jac_p.flatten())
            #
            #     r = self.dlambda - self.norm_y(x,y0s, y-y0) - np.inner(p0s, p-p0)
            #     dp = (r-self.norm_y(x,y0s, z1))/(p0s+self.norm_y(x,y0s, z2))
            #     dy += z2*dp
            dp = 0
            self.timers[6].toc()

            ############################################################################################################
            ## 2c. Difference
            ############################################################################################################
            self.timers[7].tic()
            ## Load alpha and integrate for comparison
            dx_int = np.zeros((jmax+1))
            dx_int[1:-1] = 0.5*(x[2:]-x[:-2])
            dx_int[0] = 0.5*(x[1]-x[0])
            dx_int[-1] = 0.5*(x[-1]-x[-2])
            alpha = self.unfold(y, 'alpha')
            alpha_int = np.sum(np.abs(alpha) * dx_int.reshape((jmax+1, 1, 1)), axis=0)
            alpha_new = self.unfold(y+dy, 'alpha')
            alpha_new_int = np.sum(np.abs(alpha_new) * dx_int.reshape((jmax+1, 1, 1)), axis=0)

            ## Compute difference
            diff = np.linalg.norm((alpha_new_int-alpha_int), 2)/np.linalg.norm(alpha_new_int, 2)
            if np.isnan(diff):
                raise ValueError

            ## Update
            y_old = copy(y)
            y = y + self.RELAX*dy
            # del dy
            if param is not None:
                if (self.stepsize<0 and p+self.RELAX*dp<=self.goal) or (self.stepsize>0 and p+self.RELAX*dp>=self.goal):
                    self.convergedToGoal = True
                p = p + self.RELAX*dp
                self.update_param(param, p)
                self.logger.info('\tNew value of '+param+': ' + str(p))
            self.timers[7].toc()

            ############################################################################################################
            ## 2d. Print result
            ############################################################################################################
            iteration += 1
            # sys.stdout.write("\rIteration: "+str(iteration)+' - '+"Rel. error: "+str(diff) +' ')
            self.logger.info("\tIteration: "+str(iteration)+' - '+"Rel. error: "+str(diff) +' ')
            if diff<self.TOLERANCE:
                converged = True
                self.logger.info('\tConverged after ' + str(iteration) + ' iterations')
                # sys.stdout.flush()
            elif iteration>10 and diff>0.5:
                raise ValueError
            elif iteration>self.MAXITER:
                if diff<self.TOLERANCE*10:
                    converged = True
                    self.logger.info('\tSufficiently converged after ' + str(iteration) + ' iterations')
                else:
                    # sys.stdout.flush()
                    self.logger.info('\tDid not converge after ' + str(iteration) + ' iterations')
                    raise ValueError

        return y, y_old, x, self.update_param(param, p)

    """
    #####################################################################################################################
        JACOBIANS
    ##################################################################################################################### 
    """
    def Jac_invar(self, x):
        ############################################################################################################
        ## Init
        ############################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        L = x[-1]

        g = self.input.v('G')
        BETA = self.input.v('BETA')
        H = self.input.v('H', x=x/L)
        Hx = self.input.d('H', x=x/L, dim='x')
        B = self.input.v('B', x=x/L)
        Bx  = self.input.d('B', x=x/L, dim='x')
        Av = self.dc.v('Av', x=x/L, z=0, f=0)
        Kv = self.dc.v('Kv', x=x/L, z=0, f=0)

        A = B*H
        Ax = B*Hx + Bx*H

        ssea = self.input.v('ssea')
        Q = self.input.v('Q1', x=x/L)
        Kh = self.input.v('Kh', x=x/L, z=0, f=0)
        if self.tide_type == 'discharge':
            Q_M2= self.input.v('QM2', x=x/L)
        else:
            zetaAmp = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (self.F,))+ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (self.F,))

        # Initialise eigenfunctions
        m_arr = np.arange(0, self.M).reshape((1, 1, self.M))
        nullmat = np.zeros((self.M*(self.ftot), self.M*(self.ftot)))

        xl = L*self.dxmax_frac*3.
        Avw0 = self.Avw
        weirmixing = Avw0*np.exp(-((x-L)/xl)**2)

        scaling_DAcont = 1./A[0]
        self.scaling_DAcont_L = 1./(A[0]*L/jmax+1.)
        scaling_s_x0 = 1e-4
        ############################################################################################################
        ## Jacobian - invariant part
        ############################################################################################################
        # For discretiseOperator, the numbers indicate:
        #   order of der of x
        #   equation number (salinity, momentum, DA cont)
        #   variable number (s, u, zeta)

        varshape = (self.M*(self.ftot), self.M*(self.ftot), (self.ftot)) # number of elements in each of the variables s, u, zeta

        ## Hydro - Inertia
        G1bH2 = merge_space_time(self.GalMat['G1b'], self.GalMat['H2'])
        inert = np.ones((jmax+1, 1, 1))*G1bH2
        MatbandedUt = discretiseOperator(inert, 0, 1, 1, varshape, x)

        ## Hydro - Eddy viscosity
        G1H1 = merge_space_time(self.GalMat['G1'], self.GalMat['H1'])
        dif = np.real((Av/H**2).reshape((jmax+1, 1, 1)))*G1H1

        numdif = np.real((weirmixing/H**2).reshape((jmax+1, 1, 1)))*G1H1
        MatbandedAv = discretiseOperator(dif+numdif, 0, 1, 1, varshape, x)

        ## Hydro - Barotropic
        G4H1 = merge_space_time(self.GalMat['G4'], self.GalMat['H1'])
        px = g*G4H1*np.ones((jmax+1, 1, 1))
        if self.tide_type == 'discharge':
            MatbandedZetax = discretiseOperator(px, 0, 1, 2, varshape, x)
        else:
            MatbandedZetax = discretiseOperator(px, 1, 1, 2, varshape, x)

        ## Hydro - Baroclinic
        G5H1 = merge_space_time(self.GalMat['G5'], self.GalMat['H1'])
        baroc = np.real((g*BETA*H).reshape((jmax+1, 1, 1))*G5H1)
        MatbandedBaroc = -discretiseOperator(baroc, 1, 1, 0, varshape, x, U_up=nullmat, forceCentral=True)
        MatbandedBaroc[np.where(abs(MatbandedBaroc)<1e-14*np.max(abs(MatbandedBaroc)))] = 0

        ## Sal - Inertia
        G6bH2 = merge_space_time(self.GalMat['G6b'], self.GalMat['H2'])
        inert = np.ones((jmax+1, 1, 1))*G6bH2
        MatbandedSt = discretiseOperator(inert, 0, 0, 0, varshape, x)

        ## Sal - Vertical eddy diffusity
        G6H1 = merge_space_time(self.GalMat['G6'], self.GalMat['H1'])
        G6bH1 = merge_space_time(self.GalMat['G6b'], self.GalMat['H1'])
        dif = np.real((Kv/H**2).reshape((jmax+1, 1, 1)) * G6H1)
        numdif = np.real((weirmixing/H**2).reshape((jmax+1, 1, 1)) * G6H1)
        Dirichlet_xL = G6bH1[0, Ellipsis]
        MatbandedKv = discretiseOperator(dif+numdif, 0, 0, 0, varshape, x, U_up=Dirichlet_xL)

        ## Sal - Horizontal eddy diffusity
        hdif2 = -np.real((Kh).reshape((jmax+1, 1, 1))*G6bH1)
        MatbandedKh = discretiseOperator(hdif2, 2, 0, 0, varshape, x, U_down=nullmat, U_up=nullmat)
        # if np.any(Ax!=0):
        #     hdif1 = -np.real((Kh*Ax/A).reshape((jmax+1, 1, 1))*G6bH1)
        #     MatbandedKh = MatbandedKh + discretiseOperator(hdif1, 1, 0, 0, varshape, x, U_down=nullmat, U_up=nullmat)
        if np.any(Bx!=0):
            hdif1 = -np.real((Kh*Bx/B).reshape((jmax+1, 1, 1))*G6bH1)
            MatbandedKh = MatbandedKh + discretiseOperator(hdif1, 1, 0, 0, varshape, x, U_down=nullmat, U_up=nullmat)

        ## DA Continuity
        if self.tide_type == 'discharge':
            ## .. - velocity
            G9H1 = merge_space_time(self.GalMat['G9'], self.GalMat['H1'])
            DA = G9H1*A.reshape((jmax+1, 1, 1))
            MatbandedDAu = self.scaling_DAcont_L*discretiseOperator(DA, 0, 2, 1, varshape, x)
            # MatbandedDAu = scaling_DAcont*discretiseOperator(DA, 1, 2, 1, varshape, x, U_up=0*DA[-1]) + discretiseOperator(0*DA, 0, 2, 1, varshape, x, U_up=self.scaling_DAcont_L*DA[-1])
            # MatbandedDAu = self.scaling_DAcont_L*discretiseOperator(DA, 0, 2, 1, varshape, x)
        else:
            ## .. - water level
            GH2 = merge_space_time(np.ones((1, 1, 1)), self.GalMat['H2'])
            DA = -GH2*B.reshape((jmax+1, 1, 1))
            MatbandedDAzeta = scaling_DAcont*discretiseOperator(DA, 0, 2, 2, varshape, x, U_down=np.eye(self.ftot), U_up=np.zeros((self.ftot, self.ftot)))

            ## .. - velocity
            G9H1 = merge_space_time(self.GalMat['G9'], self.GalMat['H1'])
            DA = -G9H1*A.reshape((jmax+1, 1, 1))                        # NB for numerical accuracy of the scheme use a minus sign here and a minus sign on the rhs (because the bc is upstream, information should flow downstream)
            DAx = -G9H1*Ax.reshape((jmax+1, 1, 1))                        # NB for numerical accuracy of the scheme use a minus sign here and a minus sign on the rhs (because the bc is upstream, information should flow downstream)
            MatbandedDAu = scaling_DAcont*discretiseOperator(DA, 1, 2, 1, varshape, x, U_down=0*DA[0], U_up=0*DA[-1]) + discretiseOperator(0*DA, 0, 2, 1, varshape, x, U_up=self.scaling_DAcont_L*DA[-1])
            if np.any(Ax!=0):
                MatbandedDAu = MatbandedDAu + scaling_DAcont*discretiseOperator(DAx, 0, 2, 1, varshape, x, U_down=0*DA[0], U_up=0*DA[-1])

        ## Sum
        MatbandedInvar = MatbandedUt + MatbandedAv + MatbandedZetax + MatbandedBaroc + MatbandedSt + MatbandedKv + MatbandedKh + MatbandedDAu
        if self.tide_type != 'discharge':
            MatbandedInvar += MatbandedDAzeta

        ## RHS
        rhs = np.zeros((jmax+1, self.M*2+1, self.ftot))
        if self.tide_type == 'discharge':
            rhs[:, -1, 0] = -self.scaling_DAcont_L*Q                        # river discharge for DA cont. eq.
            if self.ftot>1:
                rhs[:, -1, 1] = self.scaling_DAcont_L*np.real(Q_M2)*0.5#*np.inf                    # tidal discharge for DA cont. eq.  NB the 0.5 is a Galerkin coefficient
                rhs[:, -1, 2] = self.scaling_DAcont_L*np.imag(Q_M2)*0.5#*np.nan                     # tidal discharge for DA cont. eq.  NB the 0.5 is a Galerkin coefficient
        else:
            rhs[-1, -1, 0] = self.scaling_DAcont_L*Q[-1]                        # river discharge at x=L for DA cont. eq.
            if self.ftot>1:
                rhs[0, -1, 0] = scaling_DAcont*np.real(zetaAmp[0])         # water level at x=0 for DA cont. eq.
                rhs[0, -1, 1::2] = scaling_DAcont*np.real(zetaAmp[1:])
                rhs[0, -1, 2::2] = scaling_DAcont*np.imag(zetaAmp[1:])

        # Boundary correction for s=ssea
        if self.SEABC == 'adaptive':
            MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, 3*(2*self.M+1)*(self.ftot)-1)] = 0
            for i in range(1, self.ftot):
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0

                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), (self.ftot)),                   i+np.arange(0, self.M*(self.ftot), (self.ftot))] = 3.*scaling_s_x0
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - sum(varshape) - np.arange(0, self.M*(self.ftot), (self.ftot)),   sum(varshape) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] = -4.*scaling_s_x0
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - 2*sum(varshape) - np.arange(0, self.M*(self.ftot), (self.ftot)), 2*sum(varshape) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] = 1.*scaling_s_x0

        elif self.SEABC == 'dirichlet':
            MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1, 0] = 1*scaling_s_x0
            for i in range(1, self.ftot):
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0

                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), (self.ftot)),                   i+np.arange(0, self.M*(self.ftot), (self.ftot))] = 1.*scaling_s_x0
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - sum(varshape) - np.arange(0, self.M*(self.ftot), (self.ftot)),   sum(varshape) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] = 0
                MatbandedInvar[3*(2*self.M+1)*(self.ftot)-1 - 2*sum(varshape) - np.arange(0, self.M*(self.ftot), (self.ftot)), 2*sum(varshape) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] = 0

        rhs[0, 0, 0] = scaling_s_x0*ssea

        rhs = rhs.flatten()
        return MatbandedInvar, rhs

    def Jac_var(self, y, x):
        ############################################################################################################
        ## Init
        ############################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        L = x[-1]
        B = self.input.v('B', x=x/L)
        Bx  = self.input.d('B', x=x/L, dim='x')
        scaling_s_x0 = 1e-4

        nullmat = np.zeros((self.M*(self.ftot), self.M*(self.ftot)))

        # alpha and beta
        ytemp = y.reshape((jmax+1, (2*self.M+1)*(self.ftot)))
        alpha = ytemp[:, :self.M*(self.ftot)]
        beta = ytemp[:, self.M*(self.ftot):2*self.M*(self.ftot)]

        ############################################################################################################
        ## Jacobian - varying part
        ############################################################################################################
        varshape = (self.M*(self.ftot), self.M*(self.ftot), (self.ftot)) # number of elements in each of the variables s, u, zeta

        ## Hydro - Advection
        if self.MOMADV>0:
            # The numbers below mean: derivative_order1 (=derivative of vector in 2nd arg), axis1, derivative_order2, eqno2, varno2, shape, data
            H3G2 = merge_space_time(self.GalMat['G2'], self.GalMat['H3'])
            adv1 = discretiseTensor((self.GalMat['H3G2_p'], self.GalMat['H3G2_m'], H3G2), beta, 0, -1, 1, 1, 1, varshape, x, U_up=nullmat)
            adv2 = discretiseTensor((self.GalMat['H3G2_p'], self.GalMat['H3G2_m'], H3G2), beta, 1, -2, 0, 1, 1, varshape, x, U_up=nullmat)

            H3G3 = merge_space_time(self.GalMat['G3'], self.GalMat['H3'])
            adv3 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), beta, 0, -1, 1, 1, 1, varshape, x, U_up=nullmat)
            adv4 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), beta, 1, -2, 0, 1, 1, varshape, x, U_up=nullmat)
            if np.any(Bx!=0):
                adv5 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), (Bx/B).reshape((jmax+1, 1))*beta, 0, -2, 0, 1, 1, varshape, x, U_up=nullmat)
                adv6 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), (Bx/B).reshape((jmax+1, 1))*beta, 0, -1, 0, 1, 1, varshape, x, U_up=nullmat)
            else:
                adv5 = 0
                adv6 = 0

            MatbandedMomadv = self.MOMADV*(adv1+adv4+adv5)
            MatbandedMomadv_var = self.MOMADV*(adv2+adv3+adv6)

        else:
            MatbandedMomadv = np.zeros((sum(varshape)*6-1, sum(varshape)*(jmax+1)))
            MatbandedMomadv_var = np.zeros((sum(varshape)*6-1, sum(varshape)*(jmax+1)))

        ## Sal - Advection
        if self.SALADV>0:
            H3G7 = merge_space_time(self.GalMat['G7'], self.GalMat['H3'])
            adv1 = discretiseTensor((self.GalMat['H3G7_p'], self.GalMat['H3G7_m'], H3G7), beta,  0, -1, 1, 0, 0, varshape, x, U_up=nullmat)
            adv2 = discretiseTensor((self.GalMat['H3G7_p'], self.GalMat['H3G7_m'], H3G7), alpha, 1, -2, 0, 0, 1, varshape, x, U_up=nullmat)

            H3G8 = merge_space_time(self.GalMat['G8'], self.GalMat['H3'])
            adv3 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8),                             beta,  1, -1, 0, 0, 0, varshape, x, U_up=nullmat)
            adv5 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8),                             alpha, 0, -2, 1, 0, 1, varshape, x, U_up=nullmat, forceCentral=True)
            if np.any(Bx!=0):
                adv4 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8), (Bx/B).reshape((jmax+1, 1))*beta,  0, -1, 0, 0, 0, varshape, x, U_up=nullmat)
                adv6 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8), (Bx/B).reshape((jmax+1, 1))*alpha, 0, -2, 0, 0, 1, varshape, x, U_up=nullmat)
            else:
                adv4 = 0
                adv6 = 0

            MatbandedAdv = self.SALADV*(adv1 + adv5 + adv6)
            MatbandedAdv_var = self.SALADV*(adv2 + adv3 + adv4)
            # MatbandedAdv = self.SALADV*(adv1 + adv3 + adv6)
            # MatbandedAdv_var = self.SALADV*(adv2 + adv5 + adv4)
        else:
            MatbandedAdv = np.zeros((sum(varshape)*6-1, sum(varshape)*(jmax+1)))
            MatbandedAdv_var = np.zeros((sum(varshape)*6-1, sum(varshape)*(jmax+1)))

        # Boundaries corrections
        # s at x = 0
        # First set to zero
        for i in range(0, self.ftot):
            MatbandedAdv[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0
            MatbandedAdv_var[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0
            MatbandedMomadv[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0
            MatbandedMomadv_var[3*(2*self.M+1)*(self.ftot)-1 - np.arange(-i, 3*(2*self.M+1)*(self.ftot)-1), np.arange(0, i+3*(2*self.M+1)*(self.ftot)-1)] = 0

        # Next set the RHS

        if self.SEABC == 'adaptive':
            alpha_c = self.unfold(y, 'alpha')[0]
            t = np.linspace(0, 2*np.pi, 100).reshape((1,1,100))
            tinit = t[0,0,np.argmin(self.F_bc(t, alpha_c))]
            that = scipy.optimize.fmin(self.F_bc, tinit, (alpha_c,), xtol=1e-10, disp=False)

            cm = (1.)**np.arange(0, self.M).reshape((1,self.M, 1))  # New basis
            n = np.arange(0, self.F).reshape((1,1, self.F))
            cf = np.cos(n*that)-1j*np.sin(n*that)
            sf = -np.sin(n*that)-1j*np.cos(n*that)
            Fy = self.fold((cm*cf,))
            gy = self.fold((cm*sf,))
            Ft = -np.sum(np.real(alpha_c).reshape((1, self.M, self.F))*cm*np.sin(n*that)*n   +np.imag(alpha_c).reshape((1,self.M, self.F))*cm*np.cos(n*that)*n)
            gt = -np.sum(np.real(alpha_c).reshape((1, self.M, self.F))*cm*np.cos(n*that)*n**2-np.imag(alpha_c).reshape((1,self.M, self.F))*cm*np.sin(n*that)*n**2)

            # then determine the jacobian
            MatbandedAdv[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), 1), np.arange(0, self.M*(self.ftot), 1)] = scaling_s_x0*Fy
            if gt==0:   # and Ft==0:
                MatbandedAdv_var[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), 1), np.arange(0, self.M*(self.ftot), 1)] = 0.
            else:
                MatbandedAdv_var[3*(2*self.M+1)*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), 1), np.arange(0, self.M*(self.ftot), 1)] = -scaling_s_x0*gy*Ft/gt

        ## Sum
        Matbanded_adv = MatbandedAdv + MatbandedMomadv
        Matbanded_adv_var = MatbandedAdv_var + MatbandedMomadv_var

        return Matbanded_adv, Matbanded_adv_var

    def F_bc(self, t, alpha0):
        m = np.arange(0, self.M).reshape((self.M, 1, 1))
        n = np.arange(0, self.F).reshape((1, self.F, 1))
        cm = (1.)**m   # New basis
        F = np.sum(np.sum(np.real(alpha0).reshape((self.M, self.F, 1))*cm*np.cos(n*t) - np.imag(alpha0).reshape((self.M, self.F, 1))*cm*np.sin(n*t), axis=0),axis=0)
        return -F.flatten()

    def Jac_param(self, y, x, param):
        jmax = len(x)-1
        ytemp = y.reshape((jmax+1, (2*self.M+1)*(self.ftot)))
        alpha = ytemp[:, :self.M*(self.ftot)]
        beta = ytemp[:, self.M*(self.ftot):2*self.M*(self.ftot)]

        if param =='Av' or param=='Kv' or param =='sf':
            H = self.input.v('H', range(0, jmax+1), 0, 0)
            G1H1 = merge_space_time(self.GalMat['G1'], self.GalMat['H1'])
            G6H1 = merge_space_time(self.GalMat['G6'], self.GalMat['H1'])

            # compute jacobians
            Jac_p = np.zeros((jmax+1, (2*self.M+1)*(self.ftot)))
            Jac_p[:, :self.M*(self.ftot)] = (1./H**2).reshape((jmax+1, 1)) * np.inner(alpha.reshape((jmax+1, 1, self.M*self.ftot)), G6H1).reshape((jmax+1, self.M*(self.ftot)))
            Jac_p[-1, :self.M*(self.ftot)] = 0.
            Jac_p[:, self.M*(self.ftot):2*self.M*(self.ftot)] = (1./H**2).reshape((jmax+1, 1)) * np.inner(beta.reshape((jmax+1, 1, self.M*self.ftot)), G1H1).reshape((jmax+1, self.M*(self.ftot)))
            Jac_p[-1, self.M*(self.ftot):2*self.M*(self.ftot)] = 0.

            Jac_p[0, :self.ftot] = 0
            Jac_p[1, 1:self.ftot] = 0
            Jac_p[2, 1:self.ftot] = 0
        if param =='Q':
            Jac_p = np.zeros((jmax+1, self.M*2+1, self.ftot))
            Jac_p[-1, -1, 0] = -self.scaling_DAcont_L
            Jac_p = Jac_p.reshape((jmax+1, (2*self.M+1)*(self.ftot)))
        return Jac_p

    def flux_splitting(self, G2, G3, G7, G8):
        ## Discretise non-linear tensor operators
        G2_p, G2_m = TensorFluxSplitting(G2, -2)
        G3_p, G3_m = TensorFluxSplitting(G3, -2)
        G7_p, G7_m = TensorFluxSplitting(G7, -2)
        G8_p, G8_m = TensorFluxSplitting(G8, -1)
        return G2_p, G2_m, G3_p, G3_m, G7_p, G7_m, G8_p, G8_m

    """
    #####################################################################################################################
        CONTINUATION PARAMETERS
    ##################################################################################################################### 
    """
    def load_params(self):
        self.dc = DataContainer(self.input.slice('grid')._data)
        jmax = self.input.v('grid', 'maxIndex', 'x')
        self.dc.addData('Roughness', copy((np.real(self.input.v('Roughness', range(0, jmax+1), 0, 0)))))
        self.dc.addData('Av', copy((np.real(self.input.v('Av', range(0, jmax+1), 0, 0)))))
        self.dc.addData('Kv', copy((np.real(self.input.v('Kv', range(0, jmax+1), 0, 0)))))
        self.dc.addData('Q1', copy((np.real(self.input.v('Q1', x=0, z=0, f=0)))))
        return

    def get_param(self, param):
        if param =='Av':
            p = np.float32(np.real(self.dc.v('Av', x=0, z=0, f=0)))
        if param=='Kv':
            p = np.float32(np.real(self.dc.v('Kv', x=0, z=0, f=0)))
        if param =='sf':
            p = np.float32(np.real(self.dc.v('Roughness', x=0, z=0, f=0)))
        if param =='Q':
            p = np.float32(np.real(self.dc.v('Q1', x=0, z=0, f=0)))
        return p

    def update_param(self, param, p):
        d = {}
        jmax = self.input.v('grid', 'maxIndex', 'x')
        if param =='Av' or param=='Kv' or param =='sf':
            if param =='Av':
                pold = self.dc.v('Av', x=0, z=0, f=0)
            elif param=='Kv':
                pold = self.dc.v('Kv', x=0, z=0, f=0)
            else:
                pold = self.dc.v('Roughness', x=0, z=0, f=0)

            self.dc.addData('Av', self.dc.v('Av', range(0, jmax+1), 0, 0)*p/pold)
            self.dc.addData('Kv', self.dc.v('Kv', range(0, jmax+1), 0, 0)*p/pold)
            self.dc.addData('Roughness', self.dc.v('Roughness', range(0, jmax+1), 0, 0)*p/pold)

            d['Av'] = self.dc.v('Av', range(0, jmax+1), 0, 0)*p/pold
            d['Kv'] = self.dc.v('Kv', range(0, jmax+1), 0, 0)*p/pold
            d['Roughness'] = self.dc.v('Roughness', range(0, jmax+1), 0, 0)*p/pold
        if param =='Q':
            self.dc.addData('Q1', p)
            d['Q1'] = p
        return d

    def norm_y(self, x, y1, y2):
        jmax = len(x)-1
        dx_int = np.zeros((jmax+1))
        dx_int[1:-1] = 0.5*(x[2:]-x[:-2])
        dx_int[0] = 0.5*(x[1]-x[0])
        dx_int[-1] = 0.5*(x[-1]-x[-2])
        norm = np.sum((y1 * y2).reshape((jmax+1, 2*self.M+1, self.ftot)) * dx_int.reshape((jmax+1, 1, 1)))/(max(x)*(2*self.M+1)*self.ftot)

        # norm = np.inner(y1, y2)/len(y1)**2
        return norm
    """
    #####################################################################################################################
        DECOMPOSITION
    ##################################################################################################################### 
    """
    def decomposition(self, y, y_old, x):
        ############################################################################################################
        ## Init
        ############################################################################################################
        d = {}

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        L = x[-1]

        g = self.input.v('G')
        BETA = self.input.v('BETA')
        H = self.input.v('H', x=x/L)
        Hx = self.input.d('H', x=x/L, dim='x')
        B = self.input.v('B', x=x/L)
        Bx  = self.input.d('B', x=x/L, dim='x')
        A = B*H
        Ax = B*Hx + Bx*H
        Q = self.input.v('Q1', x=x/L)
        Q_M2= self.input.v('QM2', x=x/L)
        zetaAmp = ny.amp_phase_input(self.input.v('A0'), self.input.v('phase0'), (self.F,))+ny.amp_phase_input(self.input.v('A1'), self.input.v('phase1'), (self.F,))
        Kh = self.input.v('Kh', x=x/L, z=0, f=0)
        Av = self.dc.v('Av', x=x/L, z=0, f=0)
        Kv = self.dc.v('Kv', x=x/L, z=0, f=0)

        # Initialise eigenfunctions
        m_arr = np.arange(0, self.M).reshape((1, 1, self.M, 1))
        nullmat = np.zeros((self.M*(self.ftot), self.M*(self.ftot)))
        varshape_hydro = (self.M*(self.ftot), (self.ftot)) # number of elements in each of the variables u, zeta
        varshape_sal = (self.M*(self.ftot), ) # number of elements in s

        xl = L*self.dxmax_frac*3.
        Avw0 = self.Avw
        weirmixing = Avw0*np.exp(-((x-L)/xl)**2)

        scaling_DAcont = 1./A[0]
        # self.scaling_DAcont_L = 1./(A[0]*L/jmax+1.)
        scaling_s_x0 = 1e-4

        # alpha and beta
        alpha = self.unfold(y, 'alpha')
        beta = self.unfold(y, 'beta')
        zeta = self.unfold(y, 'zeta')

        alpha_old = self.unfold(y_old, 'alpha')
        beta_old = self.unfold(y_old, 'beta')

        # grid
        grid = deepcopy(self.input.slice('grid'))
        grid._data['grid']['axis']['x'] = x/L

        ############################################################################################################
        ## Decomposition Hydrodynamics
        ############################################################################################################
        # For discretiseOperator, the numbers indicate:
        #   order of der of x
        #   equation number (momentum, DA cont)
        #   variable number (u, zeta)

        ## System terms
        #       Inertia
        G1bH2 = merge_space_time(self.GalMat['G1b'], self.GalMat['H2'])
        inert = np.ones((jmax+1, 1, 1))*G1bH2
        MatbandedUt = discretiseOperator(inert, 0, 0, 0, varshape_hydro, x)

        #       Eddy viscosity
        G1H1 = merge_space_time(self.GalMat['G1'], self.GalMat['H1'])
        dif = np.real((Av/H**2).reshape((jmax+1, 1, 1)))*G1H1
        numdif = np.real((weirmixing/H**2).reshape((jmax+1, 1, 1)))*G1H1
        MatbandedAv = discretiseOperator(dif+numdif, 0, 0, 0, varshape_hydro, x)

        #      Barotropic
        G4H1 = merge_space_time(self.GalMat['G4'], self.GalMat['H1'])
        px = g*G4H1*np.ones((jmax+1, 1, 1))
        if self.tide_type == 'discharge':
            MatbandedZetax = discretiseOperator(px, 0, 0, 1, varshape_hydro, x)
        else:
            MatbandedZetax = discretiseOperator(px, 1, 0, 1, varshape_hydro, x)

        #       DA Continuity
        if self.tide_type =='discharge':
            #  .. - velocity
            G9H1 = merge_space_time(self.GalMat['G9'], self.GalMat['H1'])
            DA = G9H1*A.reshape((jmax+1, 1, 1))
            MatbandedDAu = self.scaling_DAcont_L*discretiseOperator(DA, 0, 1, 0, varshape_hydro, x)
        else:
            #  .. - water level
            GH2 = merge_space_time(np.ones((1, 1, 1)), self.GalMat['H2'])
            DA = -GH2*B.reshape((jmax+1, 1, 1))
            MatbandedDAzeta = scaling_DAcont*discretiseOperator(DA, 0, 1, 1, varshape_hydro, x, U_down=np.eye(self.ftot), U_up=np.zeros((self.ftot, self.ftot)))

            #  .. - velocity
            G9H1 = merge_space_time(self.GalMat['G9'], self.GalMat['H1'])
            DA = -G9H1*A.reshape((jmax+1, 1, 1))                        # NB for numerical accuracy of the scheme use a minus sign here and a minus sign on the rhs (because the bc is upstream, information should flow downstream)
            DAx = -G9H1*Ax.reshape((jmax+1, 1, 1))                        # NB for numerical accuracy of the scheme use a minus sign here and a minus sign on the rhs (because the bc is upstream, information should flow downstream)
            MatbandedDAu = scaling_DAcont*discretiseOperator(DA, 1, 1, 0, varshape_hydro, x, U_down=0*DA[0], U_up=0*DA[-1]) + discretiseOperator(0*DA, 0, 1, 0, varshape_hydro, x, U_up=self.scaling_DAcont_L*DA[-1])
            if any(Ax!=0):
                MatbandedDAu = MatbandedDAu + scaling_DAcont*discretiseOperator(DAx, 0, 1, 0, varshape_hydro, x, U_down=0*DA[0], U_up=0*DA[-1])


        Mat = MatbandedUt+MatbandedAv+MatbandedZetax+MatbandedDAu
        if self.tide_type != 'discharge':
            Mat += MatbandedDAzeta

        ## Forcing terms
        #      Advection
        if self.MOMADV>0:
            # The numbers below mean: derivative_order1 (=derivative of vector in 2nd arg), axis1, derivative_order2, eqno2, varno2, shape, data
            H3G2 = merge_space_time(self.GalMat['G2'], self.GalMat['H3'])
            H3G3 = merge_space_time(self.GalMat['G3'], self.GalMat['H3'])

            ytemp = y_old.reshape((jmax+1, (2*self.M+1)*(self.ftot)))
            beta_old_temp = ytemp[:, self.M*(self.ftot):2*self.M*(self.ftot)]
            adv1 = discretiseTensor((self.GalMat['H3G2_p'], self.GalMat['H3G2_m'], H3G2), beta_old_temp, 0, -1, 1, 0, 0, varshape_hydro, x, U_up=nullmat)
            adv4 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), beta_old_temp, 1, -2, 0, 0, 0, varshape_hydro, x, U_up=nullmat)
            adv5 = self.W*discretiseTensor((self.GalMat['H3G3_p'], self.GalMat['H3G3_m'], H3G3), (Bx/B).reshape((jmax+1, 1))*beta_old_temp, 0, -2, 0, 0, 0, varshape_hydro, x, U_up=nullmat)

            MatbandedMomadv = self.MOMADV*(adv1+adv4+adv5)
            ytemp = self.fold((beta, zeta))
            f_adv = bandedMatVec(MatbandedMomadv, ytemp.flatten())
        else:
            f_adv = np.zeros(((sum(varshape_hydro))*(jmax+1)))

        #      Baroclinic
        G5H1 = merge_space_time(self.GalMat['G5'], self.GalMat['H1'])
        baroc = np.real((g*BETA*H).reshape((jmax+1, 1, 1)) * G5H1)
        MatbandedBaroc = -discretiseOperator(baroc, 1, 0, 0, varshape_hydro, x, U_up=nullmat, forceCentral=True)
        # MatbandedBaroc = -discretiseOperator(baroc, 1, 0, 0, varshape_hydro, x, U_up=nullmat)
        MatbandedBaroc[np.where(abs(MatbandedBaroc)<1e-14*np.max(abs(MatbandedBaroc)))] = 0
        ytemp = self.fold((alpha, 0*zeta))
        f_baroc = bandedMatVec(MatbandedBaroc, ytemp)

        #       RHS
        rhsQ = np.zeros((jmax+1, self.M+1, self.ftot))
        if self.tide_type == 'discharge':
            rhsQ[:, -1, 0] = -self.scaling_DAcont_L*Q                        # river discharge for DA cont. eq.
            rhsTide = np.zeros((jmax+1, self.M+1, self.ftot))
            if self.ftot>1:
                rhsTide[:, -1, 1] = self.scaling_DAcont_L*np.real(Q_M2)*0.5                        # tidal discharge for DA cont. eq.
                rhsTide[:, -1, 2] = self.scaling_DAcont_L*np.imag(Q_M2)*0.5                        # tidal discharge for DA cont. eq.
        else:
            rhsQ[-1, -1, 0] = self.scaling_DAcont_L*Q[-1]                        # river discharge at x=L for DA cont. eq.
            rhsTide = np.zeros((jmax+1, self.M+1, self.ftot))
            rhsTide[0, -1, 0] = scaling_DAcont*np.real(zetaAmp[0])         # water level at x=0 for DA cont. eq.
            rhsTide[0, -1, 1::2] = scaling_DAcont*np.real(zetaAmp[1:])
            rhsTide[0, -1, 2::2] = scaling_DAcont*np.imag(zetaAmp[1:])

        rhs = {}
        rhs['riv'] = rhsQ.flatten()
        rhs['tide'] = rhsTide.flatten()
        rhs['gc'] = -f_baroc
        rhs['adv'] = -f_adv

        ## Compute decomposition
        d['zeta'] = {}
        d['beta'] = {}
        d['u0'] = {}
        d['w0'] = {}

        d['__variableOnGrid'] = {}
        d['__variableOnGrid']['alpha'] = 'adaptive_grid'
        d['__variableOnGrid']['beta'] = 'adaptive_grid'
        d['__variableOnGrid']['zeta'] = 'adaptive_grid'
        d['__variableOnGrid']['u0'] = 'adaptive_grid'
        d['__variableOnGrid']['w0'] = 'adaptive_grid'
        d['__variableOnGrid']['s'] = 'adaptive_grid'
        d['__variableOnGrid']['T'] = 'adaptive_grid'

        beta_total = 0
        zeta_total = 0
        for key in rhs.keys():
            temp = scipy.linalg.solve_banded((3*(self.M+1)*(self.ftot)-1, 3*(self.M+1)*(self.ftot)-1), Mat, rhs[key])
            temp = np.concatenate((np.zeros((jmax+1, self.M, self.ftot)), temp.reshape((jmax+1, self.M+1, self.ftot))), axis=1).flatten()
            beta_sub = self.unfold(temp, 'beta', True)
            zeta_sub = self.unfold(temp, 'zeta')
            beta_total += beta_sub
            zeta_total += zeta_sub
            d['beta'][key] = beta_sub
            d['zeta'][key] = zeta_sub

        # residual due to rounding errors
        beta_residual = self.unfold(y, 'beta', True) - beta_total
        d['beta']['residual'] = beta_residual
        zeta_residual = self.unfold(y, 'zeta') - zeta_total
        d['zeta']['residual'] = zeta_residual

        # compute velocity
        for key in d['beta'].keys():
            beta_sub = d['beta'][key].reshape((jmax+1, self.M, self.F))
            u = np.einsum('abc,eb->aec', beta_sub, self.f2[0,:,:,0])
            d['u0'][key] = u

            w = - H.reshape((jmax+1, 1, 1)) * np.einsum('abc,eb->aec', ((Bx/B).reshape((jmax+1, 1, 1))*beta_sub + np.gradient(beta_sub, x, edge_order=2, axis=0)), self.pf2[0,:,:,0])
            d['w0'][key] = w

        ############################################################################################################
        ## Decomposition Salinity
        ############################################################################################################
        # For discretiseOperator, the numbers indicate:
        #   order of der of x
        #   equation number (salinit,)
        #   variable number (s,)
        ## System
        #       Inertia
        G6bH2 = merge_space_time(self.GalMat['G6b'], self.GalMat['H2'])
        inert = np.ones((jmax+1, 1, 1))*G6bH2
        MatbandedSt = discretiseOperator(inert, 0, 0, 0, varshape_sal, x)

        #       Vertical eddy diffusity
        G6H1 = merge_space_time(self.GalMat['G6'], self.GalMat['H1'])
        G6bH1 = merge_space_time(self.GalMat['G6b'], self.GalMat['H1'])
        dif = np.real((Kv/H**2).reshape((jmax+1, 1, 1)) * G6H1)
        numdif = np.real((weirmixing/H**2).reshape((jmax+1, 1, 1)) * G6H1)
        Dirichlet_xL = G6bH1[0, Ellipsis]
        MatbandedKv = discretiseOperator(dif+numdif, 0, 0, 0, varshape_sal, x, U_up=Dirichlet_xL)

        #       Unit correction for the depth-average
        E1 = np.zeros((jmax+1, self.M*self.ftot, self.M*self.ftot))
        E1[:, 0, 0] = 1.
        MatbandedDA = discretiseOperator(E1, 0, 0, 0, varshape_sal, x, U_up=nullmat)

        Mat = MatbandedSt+MatbandedKv+MatbandedDA
        
        #       Boundary correction for s=ssea
        #           First set to zero
        for i in range(1, self.ftot):
            Mat[3*self.M*(self.ftot)-1 - np.arange(-i, 3*self.M*(self.ftot)-1), np.arange(0, i+3*self.M*(self.ftot)-1)] = 0
            
        #           Determine Dirichlet condition for subtidal salinity based on maximum salinity
        # alpha_c = self.unfold(y_old, 'alpha')[0]
        # t = np.linspace(0, 2*np.pi, 100).reshape((1,1,100))
        # tinit = t[0,0,np.argmin(self.F_bc(t, alpha_c))]
        # that = scipy.optimize.fmin(self.F_bc, tinit, (alpha_c,), disp=False)
        #
        # cm = (1.)**np.arange(0, self.M).reshape((1,self.M, 1))
        # n = np.arange(0, self.F).reshape((1,1, self.F))
        # cf = np.cos(n*that)-1j*np.sin(n*that)
        # Fy = self.fold((cm*cf,))
        #
        # # then determine the jacobian
        # Mat[3*self.M*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), 1), np.arange(0, self.M*(self.ftot), 1)] = scaling_s_x0*Fy

        #           Apply Neumann condition for tidal salinity
        for i in range(1, self.ftot):
            Mat[3*self.M*(self.ftot)-1 - np.arange(0, self.M*(self.ftot), (self.ftot)),                   i+np.arange(0, self.M*(self.ftot), (self.ftot))] += 3.*scaling_s_x0
            Mat[3*self.M*(self.ftot)-1 - sum(varshape_sal) - np.arange(0, self.M*(self.ftot), (self.ftot)),   sum(varshape_sal) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] += -4.*scaling_s_x0
            Mat[3*self.M*(self.ftot)-1 - 2*sum(varshape_sal) - np.arange(0, self.M*(self.ftot), (self.ftot)), 2*sum(varshape_sal) + i + np.arange(0, self.M*(self.ftot), (self.ftot))] += 1.*scaling_s_x0

        ## Forcing
        rhs = {}

        #       Horizontal eddy diffusity
        # hdif1 = -np.real((Kh*Ax/A).reshape((jmax+1, 1, 1))*G6bH1)
        hdif1 = -np.real((Kh*Bx/B).reshape((jmax+1, 1, 1))*G6bH1)
        hdif2 = -np.real((Kh).reshape((jmax+1, 1, 1))*G6bH1)
        MatbandedKh = discretiseOperator(hdif1, 1, 0, 0, varshape_sal, x, U_down=nullmat, U_up=nullmat) + discretiseOperator(hdif2, 2, 0, 0, varshape_sal, x, U_down=nullmat, U_up=nullmat)
        ytemp = self.fold((alpha, ))
        f_kh = bandedMatVec(MatbandedKh, ytemp)
        rhs['Kh'] = -f_kh

        #       Advection
        for betaLabel in d['beta'].keys():
            if self.SALADV>0:
                betaVal = self.fold((d['beta'][betaLabel].reshape((jmax+1, self.M, self.F)),)).reshape((jmax+1, self.M*self.ftot))
                beta_old_temp = self.fold((beta_old,)).reshape((jmax+1, 1, 1, self.M*self.ftot))
                alpha_old_temp = self.fold((alpha_old,)).reshape((jmax+1, self.M*self.ftot))

                H3G7 = merge_space_time(self.GalMat['G7'], self.GalMat['H3'])
                H3G8 = merge_space_time(self.GalMat['G8'], self.GalMat['H3'])
                adv1 = discretiseTensor((self.GalMat['H3G7_p'], self.GalMat['H3G7_m'], H3G7), betaVal,                                           0, -1, 1, 0, 0, varshape_sal, x, U_up=nullmat, postProcess = beta_old_temp)
                adv5 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8),                             alpha_old_temp, 0, -2, 1, 0, 0, varshape_sal, x, U_up=nullmat, forceCentral=True)
                adv6 = self.W*discretiseTensor((self.GalMat['H3G8_p'], self.GalMat['H3G8_m'], H3G8), (Bx/B).reshape((jmax+1, 1))*alpha_old_temp, 0, -2, 0, 0, 0, varshape_sal, x, U_up=nullmat)

                ytemp = self.fold((alpha,))
                f_adv = bandedMatVec(self.SALADV*adv1, ytemp)
                f_adv += bandedMatVec(self.SALADV*(adv5+adv6), betaVal)
                f_adv[:self.ftot] = 0
            else:
                f_adv = np.zeros(((sum(varshape_sal))*(jmax+1)))
            rhs[betaLabel] = -f_adv

        ## Compute decomposition
        d['s'] = {}
        d['alpha'] = {}
        alpha_total = 0
        alpha_total_da = 0
        for key in rhs.keys():
            temp = scipy.linalg.solve_banded((3*(self.M)*(self.ftot)-1, 3*(self.M)*(self.ftot)-1), Mat, rhs[key])
            temp = np.concatenate((temp.reshape((jmax+1, self.M, self.ftot)), np.zeros((jmax+1, self.M+1, self.ftot))), axis=1).flatten()
            alpha_sub = self.unfold(temp, 'alpha', True)
            alpha_total += alpha_sub

            alpha_sub_da = copy(alpha_sub)
            alpha_sub_da[:, :, :, 1:, :] = 0
            alpha_total_da += alpha_sub_da

            alpha_sub_var = copy(alpha_sub)
            alpha_sub_var[:, :, :, 0, :] = 0
            d['alpha'][key] = alpha_sub_var

        #   the depth-average subtidal alpha is saved separately; this follows from the transport balance and cannot be attributed to mechanisms this way
        alpha_temp = copy(alpha).reshape((jmax+1, 1, 1, self.M, self.F))
        alpha_temp[:, 0, 0, 1:, :] = 0.
        alpha_temp[:, 0, 0, :, 1:] = 0.
        alpha_total += alpha_temp
        d['alpha']['da'] = alpha_total_da+alpha_temp

        alpha_temp = d['alpha']['residual'] + alpha.reshape((jmax+1, 1, 1, self.M, self.F)) - alpha_total
        d['alpha']['residual'] = alpha_temp

        # compute salinity
        for key in d['alpha'].keys():
            alpha_sub = d['alpha'][key].reshape((jmax+1, self.M, self.F))
            s = np.einsum('abc,eb->aec', alpha_sub, self.f1[0,:,:,0])
            d['s'][key] = s

        ############################################################################################################
        ## Transport
        ############################################################################################################
        d['T'] = {}

        ## Advective transport
        #       Declare dictionary
        d['T']['adv'] = {}
        for f in range(0, self.F):
            d['T']['adv']['TM'+str(f)] = {}
        for k1 in d['alpha'].keys():
            for k2 in d['beta'].keys():
                name = '-'.join(sorted([k1,k2]))
                for f in range(0, self.F):
                    d['T']['adv']['TM'+str(f)][name] = 0

        #       Fill dictionary
        for k1 in d['alpha'].keys():
            for k2 in d['beta'].keys():
                name = '-'.join(sorted([k1,k2]))
                for f in range(0, self.F):
                    alpha_temp = d['alpha'][k1].reshape((jmax+1, self.M, 1, self.F))
                    beta_temp = copy(d['beta'][k2].reshape((jmax+1, 1, self.M, self.F)))
                    beta_temp[:, :, :, :f] = 0
                    beta_temp[:, :, :, f+1:] = 0
                    ab = ny.complexAmplitudeProduct(alpha_temp, beta_temp, 3)
                    G70 = self.GalMat['G7'][:, 0, :, :].reshape((1, self.M, self.M, 1))

                    d['T']['adv']['TM'+str(f)][name] += np.sum(np.sum(G70*ab, axis=-3), axis=-2, keepdims=True)

        ## Diffusive transport
        d['T']['dif'] = {}
        for k1 in d['alpha'].keys():
            alpha_temp = d['alpha'][k1][:, [0], 0, 0, :]
            alphax_0 = np.gradient(alpha_temp, x, edge_order=2, axis=0)
            d['T']['dif'][k1] = -Kh.reshape((jmax+1, 1, 1))*alphax_0

        ## DEBUG TOTAL TRANSPORT ##
        # a2 = self.fold((alpha,)).reshape((jmax+1,self.ftot*self.M))
        # b2 = self.fold((beta,)).reshape((jmax+1,1,self.ftot*self.M))
        # G7H3 = merge_space_time(self.GalMat['G7'][:, 0, :, :], self.GalMat['H3'][:, 0, :, :])
        # Tcheck1 = np.sum(np.sum(G7H3*b2, axis=-1)*a2, axis=-1)
        #
        # alphax_0 = np.gradient(alpha, x, edge_order=2, axis=0)
        # Tcheck2 = -Kh.reshape((jmax+1, 1, 1))*alphax_0
        ## END DEBUG ##

        ############################################################################################################
        ## Add the grid to the output
        ############################################################################################################
        d['adaptive_grid'] = grid._data['grid']
        d['L'] = x[-1]

        return d

    """
    #####################################################################################################################
        GALERKIN METHOD
    ##################################################################################################################### 
    """
    def GalCoef(self, R):
        # velocity eigenvalues - dimensions [x, k, m, n]
        lk = self.lm.reshape((1, self.M, 1, 1))
        lm = self.lm.reshape((1, 1, self.M, 1))
        ln = self.lm.reshape((1, 1, 1, self.M))

        # salinity eigenvalues - dimensions [x, k, m, n]
        ar = np.arange(0, self.M)
        k = ar.reshape((1, self.M, 1, 1))
        m = ar.reshape((1, 1, self.M, 1))
        n = ar.reshape((1, 1, 1, self.M))

        # G1: int l^2 cm*ck     # New basis
        G1 = np.zeros((1, self.M, self.M, 1))
        G1[0, range(0, self.M), range(0, self.M), 0] = (lk**2.*(np.sin(2*lk)+2*lk)/(4*lk)).flatten()
        G1 = G1.reshape((1, self.M, self.M))

        # G1b: int cm*ck        # New basis
        G1b = np.zeros((1, self.M, self.M, 1))
        G1b[0, range(0, self.M), range(0, self.M), 0] = ((np.sin(2*lk)+2*lk)/(4*lk)).flatten()
        G1b = G1b.reshape((1, self.M, self.M))
        
        # G2: int cn*cm*ck      # New basis
        # G2 = ((lk**3+(-lm-ln)*lk**2+(-lm**2+2*ln*lm-ln**2)*lk+lm**3-ln*lm**2-ln**2*lm+ln**3)*np.sin(lk+lm+ln)+(lk**3+(ln-lm)*lk**2+(-lm**2-2*ln*lm-ln**2)*lk+lm**3+ln*lm**2-ln**2*lm-ln**3)*np.sin(lk+lm-ln)+(lk**3+(lm-ln)*lk**2+(-lm**2-2*ln*lm-ln**2)*lk-lm**3-ln*lm**2+ln**2*lm+ln**3)*np.sin(lk-lm+ln)+(lk**3+(lm+ln)*lk**2+(-lm**2+2*ln*lm-ln**2)*lk-lm**3+ln*lm**2+ln**2*lm-ln**3)*np.sin(lk-lm-ln))/(4*lk**4+(-8*lm**2-8*ln**2)*lk**2+4*lm**4-8*ln**2*lm**2+4*ln**4)      # old basis
        G2 = (((ln**3+(-lm-lk)*ln**2+(-lm**2+2*lk*lm-lk**2)*ln+lm**3-lk*lm**2-lk**2*lm+lk**3)*np.sin(ln+lm+lk)+(ln**3+(lk-lm)*ln**2+(-lm**2-2*lk*lm-lk**2)*ln+lm**3+lk*lm**2-lk**2*lm-lk**3)*np.sin(ln+lm-lk)+(ln**3+(lm-lk)*ln**2+(-lm**2-2*lk*lm-lk**2)*ln-lm**3-lk*lm**2+lk**2*lm+lk**3)*np.sin(ln-lm+lk)+(ln**3+(lm+lk)*ln**2+(-lm**2+2*lk*lm-lk**2)*ln-lm**3+lk*lm**2+lk**2*lm-lk**3)*np.sin(ln-lm-lk))*(-1)**(n+m+k))/(4*ln**4+(-8*lm**2-8*lk**2)*ln**2+4*lm**4-8*lk**2*lm**2+4*lk**4)

        # G3: int sn*sm*sk      # New basis
        # G3 = ln/lm*(-((lk**3+(-lm-ln)*lk**2+(-lm**2+2*ln*lm-ln**2)*lk+lm**3-ln*lm**2-ln**2*lm+ln**3)*np.sin(lk+lm+ln)+(-lk**3+(lm-ln)*lk**2+(lm**2+2*ln*lm+ln**2)*lk-lm**3-ln*lm**2+ln**2*lm+ln**3)*np.sin(lk+lm-ln)+(-lk**3+(ln-lm)*lk**2+(lm**2+2*ln*lm+ln**2)*lk+lm**3+ln*lm**2-ln**2*lm-ln**3)*np.sin(lk-lm+ln)+(lk**3+(lm+ln)*lk**2+(-lm**2+2*ln*lm-ln**2)*lk-lm**3+ln*lm**2+ln**2*lm-ln**3)*np.sin(lk-lm-ln))/(4*lk**4+(-8*lm**2-8*ln**2)*lk**2+4*lm**4-8*ln**2*lm**2+4*ln**4))
        # G3_b = ln/lm*np.sin(lm)*(((ln-lk)*np.cos(ln+lk)+(ln+lk)*np.cos(ln-lk))/(2*ln**2-2*lk**2+1e-30)-ln/(ln**2-lk**2+1e-30))
        # G3_b[0, range(0, self.M), :, range(0, self.M)] = (lk/lm*np.sin(lm)*(np.cos(lk)**2/(2*lk)-1/(2*lk)))[0,:,:,0]
        # G3 += G3_b
        # G3o = copy(G3)        # Old basis

        G3 = -((((-1)**k*ln**3+(-(-1)**k*lm-(-1)**k*lk)*ln**2+(-(-1)**k*lm**2+2*(-1)**k*lk*lm-(-1)**k*lk**2)*ln+(-1)**k*lm**3-(-1)**k*lk*lm**2-(-1)**k*lk**2*lm+(-1)**k*lk**3)*np.sin(ln+lm+lk)+((-1)**k*ln**3+((-1)**k*lk-(-1)**k*lm)*ln**2+(-(-1)**k*lm**2-2*(-1)**k*lk*lm-(-1)**k*lk**2)*ln+(-1)**k*lm**3+(-1)**k*lk*lm**2-(-1)**k*lk**2*lm-(-1)**k*lk**3)*np.sin(ln+lm-lk)+(-(-1)**k*ln**3+((-1)**k*lk-(-1)**k*lm)*ln**2+((-1)**k*lm**2+2*(-1)**k*lk*lm+(-1)**k*lk**2)*ln+(-1)**k*lm**3+(-1)**k*lk*lm**2-(-1)**k*lk**2*lm-(-1)**k*lk**3)*np.sin(ln-lm+lk)+(-(-1)**k*ln**3+(-(-1)**k*lm-(-1)**k*lk)*ln**2+((-1)**k*lm**2-2*(-1)**k*lk*lm+(-1)**k*lk**2)*ln+(-1)**k*lm**3-(-1)**k*lk*lm**2-(-1)**k*lk**2*lm+(-1)**k*lk**3)*np.sin(ln-lm-lk))*(-1)**(n+m))/(4*ln**4+(-8*lm**2-8*lk**2)*ln**2+4*lm**4-8*lk**2*lm**2+4*lk**4)
        G3_b = ((((-1)**k*np.sin(lm)*ln-(-1)**k*lk*np.sin(lm))*np.cos(ln+lk)+((-1)**k*np.sin(lm)*ln+(-1)**k*lk*np.sin(lm))*np.cos(ln-lk)-2*(-1)**k*np.sin(lm)*ln)*(-1)**(n+m))/(2*ln**2-2*lk**2+1e-30)
        G3_b[0, range(0, self.M), :, range(0, self.M)] = (((np.cos(lk)**2-1)*np.sin(lm)*(-1)**m)/(2*lk))[0,:,:,0]
        G3 = ln/lm*(G3+G3_b)

        # G4: int ck            # New basis
        G4 = (-1)**k*np.sin(lk)/lk
        G4 = G4.reshape((1, self.M, 1))

        # G5: int Sm ck         # New basis
        # G5 = 1/(m*np.pi+1e-20) * (((np.pi*m-lk)*np.cos(np.pi*m+lk)+(np.pi*m+lk)*np.cos(np.pi*m-lk))/(2*np.pi**2*m**2-2*lk**2+1e-20)-(np.pi*m)/(np.pi**2*m**2-lk**2+1e-20))
        # G5[:, :, [0], :] = (1./lk**2-(lk*np.sin(lk)+np.cos(lk))/lk**2)    # Old basis

        G5 = 1/(m*np.pi+1e-20) * (-1)**(m+k) * (((np.pi*m-lk)*np.cos(np.pi*m+lk)+(np.pi*m+lk)*np.cos(np.pi*m-lk))/(2*np.pi**2*m**2-2*lk**2+1e-20)-(np.pi*m)/(np.pi**2*m**2-lk**2+1e-20))
        G5[:, :, [0], :] = (-1.)**k * (1./lk**2-(lk*np.sin(lk)+np.cos(lk))/lk**2)
        G5 = G5.reshape(1, self.M, self.M)

        # G6: int Cm Ck         # New basis
        G6 = np.zeros((1, self.M, self.M, 1))
        G6[0, range(1, self.M),range(1, self.M),0] = (m[0, 0, 1:, 0]*np.pi)**2 * 0.5
        G6 = G6.reshape((1, self.M, self.M))

        # Boundary matrices     # New basis
        G6b = np.zeros((1, self.M, self.M, 1))
        G6b[0, 0, 0, 0] = 1.
        G6b[0, range(1, self.M),range(1, self.M),0] = 0.5
        G6b = G6b.reshape((1, self.M, self.M))

        # G7: int cn Cm Ck      # New basis
        # G7 = ((np.pi**3*m**3+(-np.pi**2*ln-np.pi**3*k)*m**2+(-np.pi*ln**2+2*np.pi**2*k*ln-np.pi**3*k**2)*m+ln**3-np.pi*k*ln**2-np.pi**2*k**2*ln+np.pi**3*k**3)*np.sin(np.pi*m+ln+np.pi*k)+(np.pi**3*m**3+(np.pi**3*k-np.pi**2*ln)*m**2+(-np.pi*ln**2-2*np.pi**2*k*ln-np.pi**3*k**2)*m+ln**3+np.pi*k*ln**2-np.pi**2*k**2*ln-np.pi**3*k**3)*np.sin(np.pi*m+ln-np.pi*k)+(np.pi**3*m**3+(np.pi**2*ln-np.pi**3*k)*m**2+(-np.pi*ln**2-2*np.pi**2*k*ln-np.pi**3*k**2)*m-ln**3-np.pi*k*ln**2+np.pi**2*k**2*ln+np.pi**3*k**3)*np.sin(np.pi*m-ln+np.pi*k)+(np.pi**3*m**3+(np.pi**2*ln+np.pi**3*k)*m**2+(-np.pi*ln**2+2*np.pi**2*k*ln-np.pi**3*k**2)*m-ln**3+np.pi*k*ln**2+np.pi**2*k**2*ln-np.pi**3*k**3)*np.sin(np.pi*m-ln-np.pi*k))/(4*np.pi**4*m**4+(-8*np.pi**2*ln**2-8*np.pi**4*k**2)*m**2+4*ln**4-8*np.pi**2*k**2*ln**2+4*np.pi**4*k**4)
        G7 = (((np.pi**3*(-1)**k*m**3+(-np.pi**2*(-1)**k*ln-np.pi**3*k*(-1)**k)*m**2+(-np.pi*(-1)**k*ln**2+2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3-np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln+np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m+ln+np.pi*k)+(np.pi**3*(-1)**k*m**3+(np.pi**3*k*(-1)**k-np.pi**2*(-1)**k*ln)*m**2+(-np.pi*(-1)**k*ln**2-2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3+np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln-np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m+ln-np.pi*k)+(np.pi**3*(-1)**k*m**3+(np.pi**2*(-1)**k*ln-np.pi**3*k*(-1)**k)*m**2+(-np.pi*(-1)**k*ln**2-2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m-(-1)**k*ln**3-np.pi*k*(-1)**k*ln**2+np.pi**2*k**2*(-1)**k*ln+np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m-ln+np.pi*k)+(np.pi**3*(-1)**k*m**3+(np.pi**2*(-1)**k*ln+np.pi**3*k*(-1)**k)*m**2+(-np.pi*(-1)**k*ln**2+2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m-(-1)**k*ln**3+np.pi*k*(-1)**k*ln**2+np.pi**2*k**2*(-1)**k*ln-np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m-ln-np.pi*k))*(-1)**n)/(4*np.pi**4*m**4+(-8*np.pi**2*ln**2-8*np.pi**4*k**2)*m**2+4*ln**4-8*np.pi**2*k**2*ln**2+4*np.pi**4*k**4)

        # G8: int sn Sm Ck      # New basis
        # G8 =  -m*np.pi/ln*( ((np.pi**3*m**3+(-np.pi**2*ln-np.pi**3*k)*m**2+(-np.pi*ln**2+2*np.pi**2*k*ln-np.pi**3*k**2)*m+ln**3-np.pi*k*ln**2-np.pi**2*k**2*ln+np.pi**3*k**3)*np.sin(np.pi*m+ln+np.pi*k)+(np.pi**3*m**3+(np.pi**3*k-np.pi**2*ln)*m**2+(-np.pi*ln**2-2*np.pi**2*k*ln-np.pi**3*k**2)*m+ln**3+np.pi*k*ln**2-np.pi**2*k**2*ln-np.pi**3*k**3)*np.sin(np.pi*m+ln-np.pi*k)+(-np.pi**3*m**3+(np.pi**3*k-np.pi**2*ln)*m**2+(np.pi*ln**2+2*np.pi**2*k*ln+np.pi**3*k**2)*m+ln**3+np.pi*k*ln**2-np.pi**2*k**2*ln-np.pi**3*k**3)*np.sin(np.pi*m-ln+np.pi*k)+(-np.pi**3*m**3+(-np.pi**2*ln-np.pi**3*k)*m**2+(np.pi*ln**2-2*np.pi**2*k*ln+np.pi**3*k**2)*m+ln**3-np.pi*k*ln**2-np.pi**2*k**2*ln+np.pi**3*k**3)*np.sin(np.pi*m-ln-np.pi*k))/(4*np.pi**4*m**4+(-8*np.pi**2*ln**2-8*np.pi**4*k**2)*m**2+4*ln**4-8*np.pi**2*k**2*ln**2+4*np.pi**4*k**4))
        # G8_b = m*np.pi/ln*np.sin(ln)*((2*m*(-1)**m)/(2*np.pi*(-1)**k*m**2-2*np.pi*k**2*(-1)**k+1e-30)-m/(np.pi*m**2-np.pi*k**2+1e-30))
        # G8_b[0, range(0, self.M), range(0, self.M), :] = 0
        # G8 += G8_b
        # G80 = copy(G8)

        G8 =  -(((np.pi**3*(-1)**k*m**3+(-np.pi**2*(-1)**k*ln-np.pi**3*k*(-1)**k)*m**2+(-np.pi*(-1)**k*ln**2+2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3-np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln+np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m+ln+np.pi*k)+(np.pi**3*(-1)**k*m**3+(np.pi**3*k*(-1)**k-np.pi**2*(-1)**k*ln)*m**2+(-np.pi*(-1)**k*ln**2-2*np.pi**2*k*(-1)**k*ln-np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3+np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln-np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m+ln-np.pi*k)+(-np.pi**3*(-1)**k*m**3+(np.pi**3*k*(-1)**k-np.pi**2*(-1)**k*ln)*m**2+(np.pi*(-1)**k*ln**2+2*np.pi**2*k*(-1)**k*ln+np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3+np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln-np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m-ln+np.pi*k)+(-np.pi**3*(-1)**k*m**3+(-np.pi**2*(-1)**k*ln-np.pi**3*k*(-1)**k)*m**2+(np.pi*(-1)**k*ln**2-2*np.pi**2*k*(-1)**k*ln+np.pi**3*k**2*(-1)**k)*m+(-1)**k*ln**3-np.pi*k*(-1)**k*ln**2-np.pi**2*k**2*(-1)**k*ln+np.pi**3*k**3*(-1)**k)*(-1)**m*np.sin(np.pi*m-ln-np.pi*k))*(-1)**n)/(4*np.pi**4*m**4+(-8*np.pi**2*ln**2-8*np.pi**4*k**2)*m**2+4*ln**4-8*np.pi**2*k**2*ln**2+4*np.pi**4*k**4)
        G8_b = ((np.sin(ln)*m-np.sin(ln)*m*(-1)**(m+k))*(-1)**n)/(np.pi*m**2-np.pi*k**2+1e-30)
        G8_b[0, range(0, self.M), range(0, self.M), :] = 0
        G8 = m*np.pi/ln*(G8+G8_b)

        # G9:                   # New basis
        G9 = (-1)**m*np.sin(lm)/lm
        G9 = G9.reshape((1, 1, self.M))

        return G1, G1b, G2, G3, G4, G5, G6, G6b, G7, G8, G9

    def GalCoef_time(self):
        omega = self.input.v('OMEGA')
        ftot = self.ftot

        sine_testfunction_sign = -1.

        # time eigenvalues - dimensions [x, f0, f1, f2]
        ar = np.arange(0, self.F)
        f0 = ar.reshape((1, self.F, 1, 1))
        f1 = ar.reshape((1, 1, self.F, 1))
        f2 = ar.reshape((1, 1, 1, self.F))

        # H1
        temp = np.zeros((self.F, 2))
        if sine_testfunction_sign==1:
            temp[:, 1] = 1
        temp = temp.flatten()[1:]
        H1 = np.diag(0.5-temp).reshape((1, ftot, ftot))
        H1[0, 0, 0] = 1.

        # H2
        temp = np.zeros((self.F, 2))
        temp[:, 1] = ar
        temp = temp.flatten()[2:]
        H2 = np.diag(-0.5*temp*omega, 1).reshape((1, ftot, ftot))
        H2 += np.diag(-sine_testfunction_sign*0.5*temp*omega, -1).reshape((1, ftot, ftot))

        # H3
        T = 2*np.pi/omega
        tlen = 1000
        t=np.linspace(0, T, tlen).reshape((1, 1, 1, 1, tlen))

        phi0 = np.zeros((1,ftot,1,1,tlen))                                      # test functions
        phi0[0,0] = 1
        phi0[:,1::2,:,:,:] = np.cos(ar[1:].reshape((1,self.F-1,1,1,1))*omega*t)
        phi0[:,2::2,:,:,:] = sine_testfunction_sign*np.sin(ar[1:].reshape((1,self.F-1,1,1,1))*omega*t)

        phi1 = np.zeros((1,1,ftot,1,tlen))                                      # time dependence component 1
        phi1[0,0] = 1
        phi1[:,:, 1::2,:,:] = np.cos(ar[1:].reshape((1,1,self.F-1,1,1))*omega*t)
        phi1[:,:, 2::2,:,:] = -np.sin(ar[1:].reshape((1,1,self.F-1,1,1))*omega*t)
        phi2 = phi1.reshape(1, 1, 1, ftot, tlen)                                # time dependence component 2

        H3 = np.trapz(phi0*phi1*phi2, t, axis=-1)/T
        H3[np.where(abs(H3)<1e-10)] = 0
        # H4[np.where(abs(H4)<1e-10)] = 0
        # H5[np.where(abs(H5)<1e-10)] = 0
        return H1, H2, H3

    def eigenvals_velocity(self, N, R):
        """
        Compute eigenvalues for velocity satisfying bottom bc
        Args:
            N: Number of eigenvalues
            R: dimensionless bed friction sf*H/Av
        """
        roots = []
        for n in range(0, N):
            eps = 1e-6
            interval = (((n-1)*np.pi+0.5*np.pi+eps), (n*np.pi+0.5*np.pi-eps))
            if n==0:
                interval = (-eps, (n*np.pi+0.5*np.pi-eps))
            # sol = scipy.optimize.root_scalar(self.tanlin, (1./R,), bracket=interval)
            # roots.append(sol.root)
            sol = scipy.optimize.brentq(self.tanlin, interval[0], interval[1], (1./R,))
            roots.append(sol)
        roots = [i for i in sorted(roots)]
        return np.asarray(roots)

    """
    #####################################################################################################################
        ADAPTIVE GRID
    ##################################################################################################################### 
    """
    def setgrid(self, x, y, relaxation = 0.5):
        if np.all(y==0):
            return x, y
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        L = x[-1]

        alpha0 = self.unfold(y, 'alpha')[:, :, 0]        # only subtidal salinity
        alphax0 = np.gradient(alpha0, x, edge_order=2, axis=0)
        s = np.inner(alpha0, self.f1[:,:,:,0]).reshape((jmax+1, kmax+1))
        sx = np.max(np.abs(np.inner(alphax0, self.f1[:,:,:,0]).reshape((jmax+1, kmax+1))), axis=1)

        # old grid
        grid = deepcopy(self.input.slice('grid'))
        grid._data['grid']['axis']['x'] = x/L

        # New length
        Ls = np.max(np.where(s[:, -1]>1e-3, x, 0))
        Lnew = 2*Ls
        print('Length is set to %s'%(Lnew))
        if Lnew <L:
            move_from_index = int(np.round(0.9*np.max(np.where(x<Lnew))))
            len_move = jmax+1-move_from_index
            xnew = np.concatenate((x[:move_from_index], np.linspace(x[move_from_index], Lnew, len_move)),0)
            sx_fun = scipy.interpolate.interp1d(x, sx, bounds_error=False, fill_value=0)
            sxnew = sx_fun(xnew)
        else:
            move_from_index = int(np.round(0.9*jmax))
            len_move = jmax+1-move_from_index
            xnew = np.concatenate((x[:move_from_index], np.linspace(x[move_from_index], Lnew, len_move)),0)
            sx_fun = scipy.interpolate.interp1d(x, sx, bounds_error=False, fill_value=0)
            sxnew = sx_fun(xnew)

        # st.configure()
        # plt.figure(1, figsize=(1,2))
        # plt.plot(x, sx)
        # plt.plot(xnew, sxnew, '--')
        # st.show()

        # prepare and computate new grid
        x_new_raw = self.adaptgrid(sxnew, xnew, Lnew)
        if np.mean(np.abs(x_new_raw-xnew)) < L*self.dxmax_frac:
            x_new = copy(x_new_raw)
        else:
            x_new = relaxation*x_new_raw+(1-relaxation)*xnew

        # evaluate y and y_old on the new grid by interpolating from the old grid
        grid.addData('y', self.unfold(y, gridconform=True))
        ynew = grid.v('y', x=x_new/L).flatten()
        # if Lnew <L:
        #     ynew = grid.v('y', x=x_new/L).flatten()
        # else:
        #     xdimless = x_new/L
        #     xdimless = xdimless[np.where(xdimless<=1)]
        #     yunfold = self.unfold(y, gridconform=True)
        #     ytemp = np.zeros(yunfold.shape, dtype=yunfold.dtype)
        #     ytemp[:len(xdimless)] = grid.v('y', x=xdimless)
        #     ynew = ytemp.flatten()

        # st.configure()
        # plt.figure(1, figsize=(1,2))
        # plt.plot(x, self.unfold(y, gridconform=True)[:, 0,0,0,0], 'o-')
        # plt.plot(x_new, self.unfold(ynew, gridconform=True)[:, 0,0,0,0], 'o-')
        # st.show()

        # transfer new grid to variable x
        x = copy(x_new)
        y = copy(ynew)
        return x, y

    def adaptgrid(self, sx, x, L):
        jmax = len(x)-1

        ksi = np.linspace(0, L, jmax+1)
        maxdx = L*self.dxmax_frac
        mindx = L/float(jmax)*self.dxmin_frac
        n = 3.
        iter = 0
        diff = np.inf

        M = (abs(sx)+1./L)**n
        # M = np.max(abs(sx)+1./L, axis=1)**n
        M = np.minimum(M, np.min(M)*(maxdx/mindx))
        if not hasattr(self, 'fslope'):
            self.fslope = np.trapz(M, x)/L
        M_f = scipy.interpolate.interp1d(x, M, kind='cubic')

        x0 = copy(x)
        MAXITER = 5
        while diff>1e-3 and iter<MAXITER:
            iter +=1
            self.fslope, res, flag, m = scipy.optimize.fsolve(self.refinementfunction2, self.fslope, (L, ksi, M_f, maxdx, mindx), full_output=True)
            if flag!=1 or res['fvec']>1:
                self.fslope, res, flag, m = scipy.optimize.fsolve(self.refinementfunction2, np.trapz(M, x)/L, (L, ksi, M_f, maxdx, mindx), full_output=True)
                if flag!=1 or res['fvec']>1:
                    self.fslope, res, flag, m = scipy.optimize.fsolve(self.refinementfunction2, np.trapz(M, x)/L*100, (L, ksi, M_f, maxdx, mindx), full_output=True)
                    if flag!=1 or res['fvec']>1:
                        self.fslope, res, flag, m = scipy.optimize.fsolve(self.refinementfunction2, np.trapz(M, x)/L*0.01, (L, ksi, M_f, maxdx, mindx), full_output=True)
            xnew = self.refinementfunction(self.fslope,L, ksi, M_f, maxdx, mindx)
            xnew = xnew/xnew[-1]*L
            xnew[-1]=L
            # sx = scipy.interpolate.interp1d(x, sx, axis=0)
            # sx = sx(xnew)

            # dx = xorig[1:]-xorig[:-1]
            # dxnew = xnew[1:]-xnew[:-1]
            # plt.figure(1, figsize=(2,2))
            # plt.subplot(2,2,1)
            # plt.plot(ksi, xorig)
            # plt.plot(ksi, xnew)
            # plt.title('x')
            # plt.subplot(2,2,2)
            # plt.plot(x, xorig-xnew)
            # plt.title('diff')
            # plt.subplot(2,2,3)
            # plt.plot(x[:-1], dx, '.')
            # plt.plot(x[:-1], dxnew, '.')
            # plt.title('dx')
            # plt.subplot(2,2,4)
            # plt.plot(x[1:-1], dx[1:]/dx[:-1])
            # plt.title('$dx_i/dx_{i-1}$')
            # st.show()

            diff = np.max(abs(x-xnew))
            x = copy(xnew)

        if np.any(np.isnan(x)) or np.any(x==np.inf):
            raise ValueError

        ## report back
        if iter<MAXITER and flag==1:
            self.logger.info('\t\tGrid converged in %s iterations. Maximum change in coordinates equals %s m.'%(iter, int(np.max(abs(x0-x)))))
        elif flag!=1:
            self.logger.info('\t\tCould not find a suitable grid configuation')
        else:
            self.logger.info('\t\tGrid did not converge in %s iterations; still %s m difference in last iteration.'%(MAXITER, diff))

        return x

    def refinementfunction(self,gamma, L, ksi, M_f, dxmax, dxmin):
        dx = 0
        jmax = len(ksi)-1
        xnew = np.zeros(jmax+1)
        for i,ksi_i in enumerate(ksi):
            dxprev = copy(dx)
            dksi = ksi_i-ksi[i-1]
            if i>0:
                a = np.maximum(np.minimum(xnew[i-1], L-1),1)
                dx = gamma*dksi/M_f(a)
                if i>1:
                    MAXCHANGE = 0.1#0.06 #0.03
                    dx = np.maximum(np.minimum(dx, (1+MAXCHANGE)*dxprev), (1-MAXCHANGE)*dxprev)
                dx = np.maximum(np.minimum(dx, dxmax), dxmin)
                xnew[i] = xnew[i-1]+dx
        return xnew

    def refinementfunction2(self,gamma, L, ksi, M_f, dxmax, dxmin):
        xnew = self.refinementfunction(gamma, L, ksi, M_f, dxmax, dxmin)
        return xnew[-1]-L

    """
    Other functions
    """
    def unfold(self, y, element=None, gridconform=False):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        y = y.reshape((jmax+1, self.M*2+1, self.ftot))

        if element==None:
            if gridconform:
                y = y.reshape((jmax+1, 1, 1, self.M*2+1, self.ftot))
            return y
        else:
            if element=='alpha':
                elem = y[:, :self.M, :]
            elif element=='beta':
                elem = y[:, self.M:2*self.M, :]
            elif element=='zeta':
                elem = y[:, 2*self.M:, :]

            elem_complex = np.zeros(elem.shape[:-1]+(int((elem.shape[-1]+1)/2),), dtype=complex)
            elem_complex[:, :, 0] = elem[:, :, 0]
            elem_complex[:, :, 1:] = elem[:, :, 1::2]+1j*elem[:, :, 2::2]

            if gridconform:
                elem_complex = elem_complex.reshape((jmax+1, 1, 1, elem_complex.shape[-2], elem_complex.shape[-1]))

            return elem_complex

    def fold(self, elements):
        element_list = []
        for elem in elements:
            tlist = []
            tlist.append(np.real(elem[:, :, [0]]))
            for i in range(1, self.F):
                tlist.append(np.real(elem[:, :, [i]]))
                tlist.append(np.imag(elem[:, :, [i]]))

            element_list.append(np.concatenate(tlist, -1))

        y = np.concatenate(element_list, axis=1).flatten()
        return y

    def tanlin(self, l, c):
        return l*np.tan(l)-c

    def interpretValues(self, values):
        #inpterpret values on input as space-separated list or as pure python input
        values = ny.toList(values)

        # case 1: pure python: check for (, [, range, np.arange
        #   merge list to a string
        valString = ' '.join([str(f) for f in values])
        #   try to interpret as python string
        if any([i in valString for i in ['(', '[', ',', 'np.arange', 'range']]):
            try:
                valuespy = None
                exec('valuespy ='+valString)
                return valuespy
            except Exception as e:
                try: errorString = ': '+ e.message
                except: errorString = ''
                from src.util.diagnostics.KnownError import KnownError
                raise KnownError('Failed to interpret input as python command %s in input: %s' %(errorString, valString), e)

        # case 2: else interpret as space-separated list
        else:
            return values