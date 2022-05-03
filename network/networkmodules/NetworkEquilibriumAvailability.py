"""
Date: 12-October-2020
Authors: J. Wang
# TODO: 

"""

import logging
import numpy as np
from nifty.harmonicDecomposition import absoluteU, signU
import nifty as ny
from src.DataContainer import DataContainer
import scipy.linalg
from scipy import integrate
from copy import deepcopy
from packages.semi_analytical2DV.sediment.EquilibriumAvailability import EquilibriumAvailability
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nifty.harmonicDecomposition import absoluteU

fsize = 20
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = 0.85
plt.rcParams['figure.subplot.bottom'] = 0.12

plt.rc('xtick', labelsize=fsize) 
plt.rc('ytick', labelsize=fsize) 
plt.rc('axes', labelsize=fsize, axisbelow=True) 
plt.rc('legend', fontsize=fsize-3)
plt.rc('lines', linewidth=2)
plt.rcParams['axes.axisbelow'] = True

class NetworkEquilibriumAvailability(EquilibriumAvailability):
    # Variables
    logger = logging.getLogger(__name__)
    TOL = 10 ** -4  # Tolerance for determining the stock for the first iteration; not important for accuracy of the final answer
    TOL2 = 10 ** -13  # Tolerance for the convergence of the time-integrator for erodibility
    MAXITER = 1000  # maximum number of time steps in the time-integrator for erodibility
    dt = 3600 * 24 * 10  # time step in sec in the time-integrator
    dt = 10
    # timers = [ny.Timer(), ny.Timer(),ny.Timer(), ny.Timer(),ny.Timer(), ny.Timer()]


    def __init__(self, input):
        self.input = input
        self.channelnumber = 0
        self.OMEGA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.iteration = 0

        return

    def run(self):
        # self.timers[0].tic()
        self.logger.info('Running module NetworkEquilibriumAvailability') 

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0

        self.nch = self.input.v('networksettings', 'numberofchannels')

        
        # print('NWT')
        # for i in range(self.nch):
        #     print(self.input.v('network',str(i)).v('Q'))


        # for mod in ['baroc', 'adv', 'dp', 'river', 'stokes', 'nostress']:
        #     print(mod)
        #     for i in range(self.nch):
        #         print(self.input.v('network',str(i)).v('Q',mod))

        self.initialise_dictionaries()

        self.prepareTransport()

        self.f0uncap = self.networkMorphodynamicEquilibrium(continuous='stock')
        
        self.f = deepcopy(self.f0uncap)

        # if np.max(self.f0uncap.values()) > 1 or np.min(self.f0uncap.values()) < 0: 
            # self.networkSSCEquilibrium()
            # self.network_availability_numerical()
        # else:
            # se
    
        # self.plot()
        self.nch=3
        # self.plot_MDE()
        # 
        self.plot_manu()
        # self.plot_manu_YE()
        # self.plot_contour()
        # self.plot_contour_YE()
        # self.plot_Tcomponnets()

        plt.show()
        dictC = {}
        dictC['network'] = {} 

        # t = np.linspace(0, 2 * np.pi, 100) #  for saving the bottom stock
        for i in range(self.nch):
            
            f = self.f[str(i)]
            alpha1 = self.alpha1[str(i)]

            dictC['network'][str(i)] = DataContainer(self.dictC['network'][str(i)])
            # dictC['network'][str(i)] = self.transport[str(i)]
            dictC['network'][str(i)].addData('f', f)
            dictC['network'][str(i)].addData('alpha1', alpha1)

            # compute and save bottom stock per unit width per f
            # deposition = np.zeros(100)
            # erosion = np.zeros(100)
            # dc = self.input.v('network', str(i))
            # jmax = dc.v('grid', 'maxIndex', 'x')
            # bottom_stock = np.zeros(jmax+1)
            # u0b = dc.v('u0', 'tide', range(jmax+1), 0, 1)
            # u0b_M0 = absoluteU(u0b, 0)
            # u0b_M4 = absoluteU(u0b, 2)

            # # might be a problem if ws and sf are not constants
            # ws = self.input.v('ws0', range(jmax+1), 0, 0)
            # sf = self.input.v('Roughness', range(jmax+1), 0, 0)
            # E = self.finf * ws * self.RHOS * sf / self.GPRIME / self.DS

            # for k in range(jmax+1):
            #     deposition = np.real(ws[k] * (dc.v('hatc0', 'a', k, 0, 0) + dc.v('hatc0', 'a', k, 0, 2) * np.exp(2j*t)))
            #     erosion = E[k] * np.real(u0b_M0[k] + u0b_M4[k] * np.exp(2j*t))
            #     bottom_stock[k] = 0.5 * np.mean(np.abs(deposition - erosion))
            
            # dictC['network'][str(i)].addData('Sbed', bottom_stock)

            


            # tmp = {} 
            # for key in self.transport.getKeysOf(str(i),'T'):
            #     # tmp[key] = {}
            #     # for subkey in self.transport.getKeysOf(str(i),'T', key):
            #     #     tmp[key][subkey] = self.transport.v(str(i),'T', key, subkey)
            #     tmp[key] = self.transport.v(str(i), 'T', key)
            # dictC['network'][str(i)].addData('T', tmp)

            # tmp = {} 
            # for key in self.transport.getKeysOf(str(i),'F'):
            #     tmp[key] = self.transport.v(str(i), 'F', key)
            # dictC['network'][str(i)].addData('F', tmp)

        self.input.merge(dictC)

        # for i in range(self.nch):
        #     self.dictC['network'][str(i)]['f'] = self.f[str(i)]
        # self.input.merge(DataContainer(self.dictC))

        d = {}

        return d

    def initialise_dictionaries(self):


        self.rho0 = self.input.v('RHO0')
        self.DS = self.input.v('DS')
        self.finf = self.input.v('finf')
        self.RHOS = self.input.v('RHOS')
        self.GPRIME = self.input.v('G') * (self.RHOS - self.input.v('RHO0')) / self.input.v('RHO0')

        self.x = {}
        self.C00 = {}
        self.C04 = {}     
        self.T = {}
        self.F = {}
        self.B = {}
        self.H = {}
        self.xd = {} # dimensional x
        self.L = np.array([])
        self.T = {}
        self.Tx = {}
        self.F = {}
        self.Fx = {}
        self.p = {}
        self.q = {}
        self.h = np.zeros(self.nch)
        self.alpha1 = {}
        self.alpha2 = {}

        self.BTx = {}
        self.BFx = {}

        self.DE0 = {} # deposition - erosion
        self.DEL = {} 

        self.dictC = {}
        self.dictC['network'] = {} 



    def fS_relation(self, S, C0, C4):

        if S <= C0 - C4:
            result = S / C0
        elif S >= C0 + C4:
            result = 1.
        else:
            result = 0.5 * (1 + S/C0) + (1/np.pi) * np.arcsin((S - C0) / C4) * (1 - S/C0) - (1/np.pi) * np.sqrt((C4/C0)**2 - (S/C0 - 1)**2)
        
        return result


    def prepareTransport(self):

        transport = {}

        t = np.linspace(0, 2 * np.pi, 100)

        for channel in range(self.nch):   

            transport[str(channel)] = {}
            dc = self.input.v('network', str(channel))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')

            # prepare width
            self.B[str(channel)] = dc.v('B', range(jmax+1), 0, 0)
            self.H[str(channel)] = dc.v('H', range(jmax+1), 0, 0)
            # prepare length
            L = dc.v('L')
            self.L = np.append(self.L, L)
            # prepare x-axis
            x = dc.v('grid', 'axis', 'x')
            self.xd[str(channel)] = x * L
            # dx = x[1:] - x[:-1]

            c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), range(kmax+1), 0)) 
            c04 = np.abs(dc.v('hatc0', 'a', range(jmax+1), range(kmax+1), 2)) 
            # k = 0 bottom
            self.alpha1[str(channel)] = ny.integrate(c00, 'z', kmax, 0, dc.slice('grid'))[:, 0]
            if self.alpha1[str(channel)][-1] == 0:
                self.alpha1[str(channel)][-1] += self.alpha1[str(channel)][-2]   
            # self.alpha1[str(channel)][-1] += self.alpha1[str(channel)][-2]            
            self.alpha2[str(channel)] = ny.integrate(c04, 'z', kmax, 0, dc.slice('grid'))[:, 0]/(self.alpha1[str(channel)]+1e-10) + 1.e-10

            self.C00[str(channel)] = c00

            #TODO reference level
            zarr = ny.dimensionalAxis(dc.slice('grid'), 'z')[:, :, 0] #-dc.v('R', x=x/L).reshape((len(x), 1)) 

            # u0b = dc.v('u0', 'tide', range(jmax+1), 0, 1)
            # u0b_M0 = absoluteU(u0b, 0)
            # u0b_M4 = absoluteU(u0b, 2)

            # ws = self.input.v('ws0', range(jmax+1), 0, 0)
            # sf = self.input.v('Roughness', range(jmax+1), 0, 0)
       
            # deposition0 = np.real(ws[0] * (dc.v('hatc0', 'a', 0, 0, 0) + dc.v('hatc0', 'a', 0, 0, 2) * np.exp(2j*t)))
            # depositionL = np.real(ws[jmax] * (dc.v('hatc0', 'a', jmax, 0, 0) + dc.v('hatc0', 'a', jmax, 0, 2) * np.exp(2j*t)))

            # E = self.finf * ws * self.RHOS * sf / self.GPRIME / self.DS
            # erosion0 = E[0] * np.real(u0b_M0[0] + u0b_M4[0] * np.exp(2j*t))
            # erosionL = E[-1] * np.real(u0b_M0[jmax] + u0b_M4[jmax] * np.exp(2j*t))
           
            # self.DE0[str(channel)] = 0.5 * np.mean(np.abs(deposition0 - erosion0)) 
            # self.DEL[str(channel)] = 0.5 * np.mean(np.abs(depositionL - erosionL)) 

            # self.DE0[str(channel)] = self.alpha1[str(channel)][0]     
            # self.DEL[str(channel)] = self.alpha1[str(channel)][-1]
            #  
            self.DE0[str(channel)] = self.alpha1[str(channel)][0]  / self.H[str(channel)][0]   
            self.DEL[str(channel)] = self.alpha1[str(channel)][-1] / self.H[str(channel)][-1]

            c0 = dc.v('hatc0', 'a', range(jmax+1), range(kmax+1), range(fmax+1)) 
            c2 = dc.v('hatc2', 'a', 'erosion', 'river_river', range(jmax+1), range(kmax+1), 0) 
            c0x = dc.d('hatc0', 'a', range(jmax+1), range(kmax+1), range(fmax+1), dim='x') 
            c2x = dc.d('hatc2', 'a', 'erosion', 'river_river', range(jmax+1), range(kmax+1), 0, dim='x')

            u0 = dc.v('u0', range(jmax+1), range(kmax+1), range(fmax+1))
            # zeta0 = dc.v('zeta0', range(jmax+1), [0], range(fmax+1))

            Kh = self.input.v('Kh', range(jmax+1), 0, 0) 
            
            d = {}
            d['T'] = {}
            d['F'] = {}

            # Transport terms that are a function of the first order velocity, i.e. u1*c0 terms.
            # adv, nostress, baroc, stokes, dp, tide, river, tide_2
            for submod in dc.getKeysOf('u1'):
                # u1_comp = dc.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                d['T'] = self.dictExpand(d['T'], submod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                # calculate residual Transport terms
                for n in (0, 2):
                    tmp = dc.v('u1', submod, range(jmax+1), range(kmax+1), n)
                    if n==0:
                        if submod == 'stokes':
                            tmp = np.real(np.trapz(tmp * c0[:, :, 0], x=-zarr, axis=1))
                            # if any(tmp) > 10**-14:
                            d['T'][submod] = self.dictExpand(d['T'][submod], 'TM0', ['return', 'drift'])
                            d['T'][submod]['TM0']['return'] += tmp
                        else:
                            tmp = np.real(np.trapz(tmp * c0[:, :, 0], x=-zarr, axis=1))
                            # if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp # 2n = 0

                    elif n==2:
                        if submod == 'stokes':
                            tmp = np.real(np.trapz((tmp * np.conj(c0[:, :, 2]) + np.conj(tmp) * c0[:, :, 2]) / 4., x=-zarr, axis=1))
                            if any(tmp) > 10**-14:
                                d['T'][submod] = self.dictExpand(d['T'][submod], 'TM4', ['return', 'drift'])
                                d['T'][submod]['TM4']['return'] += tmp
                        else:
                            tmp = np.real(np.trapz((tmp * np.conj(c0[:, :, 2]) + np.conj(tmp) * c0[:, :, 2]) / 4., x=-zarr, axis=1))
                            # if any(tmp) > 10**-14:
                            d['T'][submod]['TM' + str(2 * n)] += tmp # 2n = 4, TM4

            
            # Transport terms that are a function of the first order concentration, i.e. u0*c1 terms.
            # erosion, sedadv, noflux
            for submod in dc.getKeysOf('hatc1', 'a'):
                u0 = dc.v('u0', 'tide', range(jmax+1), range(kmax+1), 1)
                if submod == 'erosion':
                    # adv, stokes, nostress, baroc, tide, river, tide 2, dp
                    for subsubmod in dc.getKeysOf('hatc1', 'a', submod):
                        d['T'] = self.dictExpand(d['T'], subsubmod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                        # M2 contribution of baroc
                        tmp = dc.v('hatc1', 'a', submod, subsubmod, range(jmax+1), range(kmax+1), 1)
                        # tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-zarr, axis=1))
                        tmp = np.real(np.trapz((u0 * np.conj(tmp) + np.conj(u0) * tmp) / 4., x=-zarr, axis=1))
                        if subsubmod == 'stokes':
                            # if any(tmp) > 10**-14:
                            d['T'][subsubmod] = self.dictExpand(d['T'][subsubmod], 'TM2', ['return', 'drift'])
                            d['T'][subsubmod]['TM2']['return'] += tmp
                        else:
                            # if any(tmp) > 10**-14:
                            d['T'][subsubmod]['TM2'] += tmp
                else:
                    d['T'] = self.dictExpand(d['T'], submod, ['TM' + str(2 * n) for n in range(0, fmax + 1)])
                    tmp = dc.v('hatc1', 'a', submod, range(0, jmax+1), range(0, kmax+1), 1)
                    # tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-zarr, axis=1))
                    tmp = np.real(np.trapz((u0 * np.conj(tmp) + np.conj(u0) * tmp) / 4., x=-zarr, axis=1))
                    # if any(tmp) > 10**-14:
                    d['T'][submod]['TM2'] += tmp

            # Transport terms that are related to diffusion, i.e. K_h*c0 or K_h*c2
            d['T'] = self.dictExpand(d['T'], 'diffusion_tide', ['TM' + str(2 * n) for n in range(fmax+1)])
            d['T']['diffusion_tide']['TM0'] = np.real(-Kh*np.trapz(c0x[:, :, 0], x=-zarr, axis=1))

            d['T'] = self.dictExpand(d['T'], 'diffusion_river', ['TM' + str(2 * n) for n in range(fmax+1)])
            tmp = np.real(-Kh*np.trapz(c2x, x=-zarr, axis=1))
            # if any(tmp) > 10**-14:
            d['T']['diffusion_river']['TM0'] = tmp

            # Transport terms that are related to Stokes drift, i.e. u0*c0*zeta0
            if 'stokes' in dc.getKeysOf('hatc1', 'a', 'erosion'):
                for n in (0, 2):
                    # u0s = u0[:, 0, 1]           # M2
                    u0s = dc.v('u0', 'tide', range(jmax+1), 0, 1)
                    # zeta0s = zeta0[:, 0, 1]     # M2
                    zeta0s = dc.v('zeta0', 'tide', range(jmax+1), 0, 1)
                    tmp = c0[:, 0, n]           # 
                    if n==0:
                        tmp = np.real(np.conj(u0s) * tmp * zeta0s + u0s * tmp * np.conj(zeta0s)) / 4
                    elif n==2:
                        tmp = np.real(u0s * np.conj(tmp) * zeta0s + np.conj(u0s) * tmp * np.conj(zeta0s)) / 8
                    # if any(tmp) > 10**-14:
                    d['T']['stokes']['TM' + str(2 * n)]['drift'] = tmp


            # Transport term that is related to the river-river interaction u1river*c2river
            d['T'] = self.dictExpand(d['T'], 'river_river', ['TM' + str(2 * n) for n in range(0, fmax + 1)])
            if dc.v('u1', 'river') is not None:
                u1_comp = dc.v('u1', 'river', range(0, jmax+1), range(0, kmax+1), 0)
                d['T']['river_river']['TM0'] = np.real(np.trapz(u1_comp * c2, x=-zarr, axis=1))

            ## Diffusion F #################################################################################################
            # Diffusive part, i.e. Kh*c00 and Kh*c20
            d['F'] = self.dictExpand(d['F'], 'diffusion_tide', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
            d['F']['diffusion_tide']['FM0'] = np.real(-Kh*np.trapz(c0[:, :, 0], x=-zarr, axis=1))

            d['F'] = self.dictExpand(d['F'], 'diffusion_river', ['FM' + str(2 * n) for n in range(0, fmax + 1)])
            d['F']['diffusion_river']['FM0'] = np.real(-Kh*np.trapz(c2, x=-zarr, axis=1))

            # Part of F that is related to sediment advection, i.e. u0*c1sedadv
            u0 = dc.v('u0', 'tide', range(jmax+1), range(kmax+1), 1)

            for submod in dc.getKeysOf('hatc1', 'ax'):
                d['F'] = self.dictExpand(d['F'], submod, ['FM' + str(2 * n) for n in range(0, fmax + 1)])
                tmp = dc.v('hatc1', 'ax', submod, range(jmax+1), range(kmax+1), 1)
                # tmp = np.real(np.trapz((u0[:, :, 1] * np.conj(tmp) + np.conj(u0[:, :, 1]) * tmp) / 4., x=-zarr, axis=1))
                tmp = np.real(np.trapz((u0 * np.conj(tmp) + np.conj(u0) * tmp) / 4., x=-zarr, axis=1))
                # if any(tmp) > 10**-14:
                d['F']['sedadv']['FM2'] += tmp

            transport[str(channel)] = d
            self.dictC['network'][str(channel)] = d

        self.transport = DataContainer(transport)

        for channel in range(self.nch):
            B = self.B[str(channel)]
            T = self.transport.v(str(channel), 'T') 
            F = self.transport.v(str(channel), 'F') 
            x = self.xd[str(channel)]
            self.T[str(channel)] = T 
            self.F[str(channel)] = F 
            self.BTx[str(channel)] = np.gradient(B * T, x, edge_order=2)
            self.BFx[str(channel)] = np.gradient(B * F, x, edge_order=2)

            self.p[str(channel)] = -(B * T + self.BFx[str(channel)]) / (B * F)
            self.q[str(channel)] = -self.BTx[str(channel)] / (B * F)

            self.h[channel] = x[1] - x[0]

        # return d

    def networkSSCEquilibrium(self):

          
        # t1 = time.time()
        # f0 = self.erodibility_stock_numerical(eps=1e-13, max_iter=10000, dt=3600*24*10)
        # t2 = time.time()
        f0 = self.Picard_iteration(eps=1e-6, max_iter=2000)
        # self.print_flux()
        # t3= time.time()
        # print(t2-t1, t3-t2, t3-t1)

    def networkMorphodynamicEquilibrium(self, continuous = 'stock'):
        # Numerically solve the morphodynamic equilibrium for the network using Chernetsky's approach
        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        Sea = self.input.v('networksettings', 'label', 'Sea')       # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')   # labels of river channels
        csea = self.input.v('networksettings', 'forcing', 'csea')
        Qsed= self.input.v('networksettings', 'forcing', 'Qsed')
        nch = self.nch  # number of channel

        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex)         
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 

        Ns = [len(self.xd[str(i)]) for i in range(nch)]    # number of grid points of each channel in the network
        N = sum(Ns)  # total number of grids in the network

        dictM = {}

        # interior mesh points of each channel, excluding boundary conditions
        # each channel: shape = (ngrid-2, ngrid)
        for channel in range(nch):
            h = self.h[channel]
            p = self.p[str(channel)]
            q = self.q[str(channel)]
            ngrid = len(self.xd[str(channel)]) # number of grids in The channel
            dictM[str(channel)] = np.zeros((ngrid-2, ngrid))
            for i in range(ngrid-2):
                dictM[str(channel)][i, [i, i+1, i+2]] = np.array([
                    -1 - h / 2 * p[i+1],
                    2 + h ** 2 * q[i+1],
                    -1 + h / 2 * p[i+1]
                ])    
        
        M = scipy.linalg.block_diag(*([dictM[str(i)] for i in range(nch)]))  # shape = (N-2*nch, N). 
        v = np.zeros(np.shape(M)[0]) 
        # The other 2*nch rows are matching and forcing conditions.
        # Do not use M = scipy.linalg.block_diag(*([dictM[key] for key in dictM.keys()])) 
        # keys in a dictionary are not ordered........

        # prescribe f at sea, x=0
        """Sea channels must have the smallest channel indices!!!!!!!"""

        for channel in Sea:
            c000 = self.alpha1[str(channel)][0] # depth integrated carrying capacity
            fsea = csea[channel] / c000 * self.H[str(channel)][0]
            row_temp = np.zeros(N)
            i = sum(Ns[k] for k in range(channel)) 
            row_temp[i] = 1
            # v_temp = min(1, fsea)
            # print(fsea)
            v_temp = fsea

            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
        
        # prescribe flux at x=L
        i_riv = 0
        for channel in River:
            B = self.B[str(channel)][-1]
            T = self.T[str(channel)][-1]
            F = self.F[str(channel)][-1]
            x = self.xd[str(channel)]
            h = x[2] - x[1]
            p = self.p[str(channel)][-1]
            i = sum(Ns[k] for k in range(channel)) + Ns[channel] - 1 # index of the last point in The channel
    
            row_temp = np.zeros(N)
            row_temp[[i, i-1, i-2]] = np.array([
                        B * T + 3 * B * F / 2 / h,
                        -2 * B * F / h,
                        B * F / 2 / h 
                    ])
            v_temp = Qsed[i_riv]

            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
            i_riv += 1
  
         # continuous bottom stock
        if continuous == 'stock':
            for j in range(len(Vertex)):
                channels = vertex_nosign[j] # indices of channels that are connected at the branching point
                for i in range(len(channels)-1): # N channels => N-1 conditions
                    row_temp = np.zeros(N)

                    channel = vertex_nosign[j][i] # channel index
                    if sign[j][i] == 1:     # x = L
                        i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                        row_temp[i_grid] = self.DEL[str(channel)]
                    elif sign[j][i] == -1:
                        i_grid = sum(Ns[k] for k in range(channel))
                        row_temp[i_grid] = self.DE0[str(channel)]
                    
                    channel = vertex_nosign[j][i-1] # the other channel index
                    if sign[j][i-1] == 1:
                        i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                        row_temp[i_grid] = -self.DEL[str(channel)]
                    elif sign[j][i-1] == -1:
                        i_grid = sum(Ns[k] for k in range(channel))
                        row_temp[i_grid] = -self.DE0[str(channel)]
                    v_temp = 0
                    M = np.vstack([M, row_temp])
                    v = np.append(v, v_temp)

        elif continuous == 'erodibility':
            for j in range(len(Vertex)):
                channels = vertex_nosign[j] # indices of channels that are connected at the branching point
                for i in range(len(channels)-1): # N channels => N-1 conditions
                    row_temp = np.zeros(N)

                    channel = vertex_nosign[j][i] # channel index
                    if sign[j][i] == 1:     # x = L
                        i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                        row_temp[i_grid] = 1
                    elif sign[j][i] == -1:
                        i_grid = sum(Ns[k] for k in range(channel))
                        row_temp[i_grid] = 1
                    
                    channel = vertex_nosign[j][i-1] # the other channel index
                    if sign[j][i-1] == 1:
                        i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                        row_temp[i_grid] = -1
                    elif sign[j][i-1] == -1:
                        i_grid = sum(Ns[k] for k in range(channel))
                        row_temp[i_grid] = -1
                    v_temp = 0
                    M = np.vstack([M, row_temp])
                    v = np.append(v, v_temp)

       
        # mass conservation for suspended sediments
        for j in range(len(Vertex)):
            channels = vertex_nosign[j] # indices of channels that are connected at the branching point
            # one junction one condition
            row_temp = np.zeros(N)
            
            for i in range(len(channels)):
                channel = vertex_nosign[j][i] # channel index

                if sign[j][i] == 1:     # x = L
                    B = self.B[str(channel)][-1]
                    T = self.T[str(channel)][-1]
                    F = self.F[str(channel)][-1]
                    x = self.xd[str(channel)]
                    h = x[2] - x[1]
                    i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1 # last grid point
                    row_temp[[i_grid, i_grid-1, i_grid-2]] = -np.array([
                        B * T + 3 * B * F / 2 / h,
                        -2 * B * F / h,
                        B * F / 2 / h 
                    ])
                    
                elif sign[j][i] == -1:  # x = 0
                    B = self.B[str(channel)][0]
                    T = self.T[str(channel)][0]
                    F = self.F[str(channel)][0]
                    x = self.xd[str(channel)]
                    h = x[2] - x[1]
                    i_grid = sum(Ns[var] for var in range(channel)) 
                    row_temp[[i_grid, i_grid+1, i_grid+2]] = np.array([
                        B * T - 3 * B * F / 2 / h,
                        +2 * B * F / h,
                        -B * F / 2 / h 
                    ])
            v_temp = 0
            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)

        f = np.linalg.solve(M, v)

        self.f = {}
        for channel in range(nch):
            x0 = sum(Ns[k] for k in range(channel))
            xL = x0 + Ns[channel] 
            self.f[str(channel)] = f[x0:xL]



        fdict = {}
        for channel in range(nch):
            # grid = 
            x0 = sum(Ns[k] for k in range(channel))
            xL = x0 + Ns[channel] 
            fdict[str(channel)] = f[x0:xL]

        return fdict


    def Picard_iteration(self, eps=1e-7, max_iter=10000):
        # Sea = self.input.v('networksettings', 'label', 'Sea') 
        # for i in Sea:
        #     self.f[str(i)][0] = min(self.f[str(i)][0], 1) 
        
        
        # f <= 1
        for i in range(self.nch):
            self.f[str(i)][self.f[str(i)] > 1] = 1
            self.f[str(i)][self.f[str(i)] < 0] = 0
  
        error = 1

        num_iter = 0
        while error > eps and num_iter < max_iter:
            self.erodibility_match()
            f_tmp = np.array(deepcopy(self.f.values()))
            for i in range(self.nch):
                self.Picard_multiSteps(i, N=100)
                # error = np.min([np.max(self.f[str(i)] - f_tmp[str(i)]), error])
            
            num_iter += 1
            error = np.max(np.abs(np.array(self.f.values()) - f_tmp))
        self.erodibility_match()
        for i in range(self.nch):
            self.Picard_multiSteps(i, N=10)
        print(error)
        print(num_iter)

    def Picard_step(self, f, x, T, Tx, F, Fx):

        N = len(x)
        h = x[1] - x[0]
        
        # update from land to sea
        for i in range(N-2, 0, -1):
            f[i] = (2 * F[i] * (f[i-1] + f[i+1]) + h * (Fx[i] + T[i]) * (f[i+1] - f[i-1])) / (2 * (2 * F[i] - Tx[i] * h**2))

            if f[i] > 1:
                f[i] = 1
            elif f[i] < 0:
                f[i] = 0

        return f

    def update_transport(self):

        for i in range(self.nch):
            dc = self.input.v('network', str(i))
            x = self.xd[str(i)]
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            f = self.f[str(i)]
            transport = (B * T * f + B * F * np.gradient(f, x, edge_order=2))[-1]
            self.G[str(i)] = transport / B


    def network_availability_numerical(self):

        # compute atil for all channel.
        self.atil = {}
        self.g, self.S = self.init_erodibility_stock_network()
        self.f0 = {}

        
        for channel in range(self.nch):
            
            x = self.xd[str(channel)]
            dx = x[1:] - x[:-1]
            f0uncap = np.exp(-integrate.cumtrapz(self.T[str(channel)] / self.F[str(channel)], dx=dx, axis=0, initial=0))
            
            f0uncap[np.where(f0uncap==np.inf)] = 0
            f0uncap[np.where(np.isnan(f0uncap))] = 0
            f0uncap = np.minimum(f0uncap, 1e4)

            self.atil[str(channel)] = f0uncap
            self.g[str(channel)][np.where(np.isnan(self.g[str(channel)]))] = 0.
            self.S[str(channel)][np.where(np.isnan(self.S[str(channel)]))] = 0.

            self.f0[str(channel)] = self.atil[str(channel)] * self.g[str(channel)]  # compute initial f

        i = 0
        difference = np.inf
        dt = self.dt
        # return
        # while difference > self.TOL2 * dt / (3600 * 24)  and i < self.MAXITER:
        while difference > self.TOL2 * dt / (3600 * 24)  and i < 4:
        # while difference > 1e-15  and i < 50:
            i += 1
            f_new, self.S = self.network_timestepping_stock(dt)

            # difference = np.max(abs(np.asarray(self.atil.values()) * np.asarray(self.g.values()) - np.asarray(self.f0.values())) / abs(np.asarray(self.f0.values()) + 10 ** -2 * np.max(np.asarray(self.f0.values()))))
            difference = np.max(
                abs(
                    np.asarray(f_new.values())  - np.asarray(self.f0.values())
                    ) / abs(
                        np.asarray(self.f0.values()) + 10 ** -2 * np.max(np.asarray(self.f0.values()))
                        )
                )

            print(difference)


            self.f0 = f_new
        
        # self.f = deepcopy(self.f0)
        self.logger.info('\t Erosion limited conditions; time iterator took %s iterations, last change: %s' % (str(i), str(difference)))
        
            

    def network_timestepping_stock(self, dt):

        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        Sea = self.input.v('networksettings', 'label', 'Sea')       # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')   # labels of river channels
        csea = self.input.v('networksettings', 'forcing', 'csea')
        Qsed= self.input.v('networksettings', 'forcing', 'Qsed')
        nch = self.nch  # number of channel

        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex)         
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 

        Ns = [len(self.xd[str(i)]) for i in range(nch)]    # number of grid points of each channel in the network
        N = sum(Ns)  # total number of grids in the network

        dictM = {}
        dictRHS = {}

        # interior mesh points of each channel, excluding boundary conditions
        # each channel: shape = (ngrid-2, ngrid)
        for channel in range(nch):
            # atil = self.atil[str(channel)]
            S = self.S[str(channel)]
            alpha1 = self.alpha1[str(channel)]
            alpha2 = self.alpha2[str(channel)]
            T = self.T[str(channel)]
            B = self.B[str(channel)]
            F = self.F[str(channel)]
            # f = self.f0[str(channel)]
            x = self.xd[str(channel)]
            dx = x[1:] - x[:-1]
            # jmax = len(B) - 1

            """ Time_integrator_Yoeri.pdf 2019-04-17 """
            htil = self.erodibility_stock_relation(alpha2, S / alpha1)
            htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1)
            BTx = np.gradient(B * T, x, edge_order=2)[1:-1]         # (BT)_x
            BF = (B * F)[1:-1]
            BT_BFx = (B * T + np.gradient(B * F, x, edge_order=2))[1:-1]    # BT + (BF)_x
            fp = np.maximum(0, BT_BFx)
            fm = np.minimum(0, BT_BFx)
            h = dx[-1]
  
            a = 0.5 * htil_der[:-2] * (-fp / h + BF / h**2) 
            b = B[1:-1] / dt + 0.5 * htil_der[1:-1] * (1 + (fp - fm) / h - 2 * BF / h**2) 
            c = 0.5 * htil_der[2:] * (fm / h + BF / h**2) 

            rhs =  + b * S[1:-1] + a * S[:-2] + c * S[2:] 
            # - 0.5 * (BTx * htil[1:-1])
            # - 0.5 * BF / h**2 * (htil[2:] - 2 * htil[1:-1] + htil[:-2])
            # - 0.5 * fm / h * (htil[2:] - htil[1:-1])
            # - 0.5 * fp / h * (htil[1:-1] - htil[:-2])
            - 0.5 * (
                BTx * htil[1:-1] 
                + fm / h * (htil[2:] - htil[1:-1])
                + fp / h * (htil[1:-1] - htil[:-2]) 
                + BF / h**2 * (htil[2:] - 2 * htil[1:-1] + htil[:-2])
            ) * 2

            ngrid = len(self.xd[str(channel)]) # number of grids in The channel
            dictM[str(channel)] = np.zeros((ngrid-2, ngrid))
            # dictRHS[str(channel)] = np.zeros(ngrid-2)
            for i in range(ngrid-2):
                dictM[str(channel)][i, [i, i+1, i+2]] = np.array([
                    a[i], b[i], c[i]
                    ])  
            dictRHS[str(channel)] = rhs  
        
        M = scipy.linalg.block_diag(*([dictM[str(i)] for i in range(nch)]))  # shape = (N-2*nch, N). 
        v = np.array([])

        for i in range(self.nch):
            v = np.append(v, dictRHS[str(i)])  

            """ The simple scheme f = atil * g does not give constant transport """
            # htil = self.erodibility_stock_relation(alpha2, S / alpha1) / (atil+1e-20)
            # htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1 * (atil+1e-20))
            
            # # transport
            # dxav = 0.5 * (dx[:-1] + dx[1:])
            # a = 0.5 * (B[:-2] + B[1:-1]) / B[1:-1] * 0.5 * (F[:-2] + F[1:-1]) * 0.5 * (atil[:-2] + atil[1:-1]) / (dx[:-1] * dxav)
            # c = 0.5 * (B[2:] + B[1:-1]) / B[1:-1] * 0.5 * (F[2:] + F[1:-1]) * 0.5 * (atil[2:] + atil[1:-1]) / (dx[1:] * dxav)
            # b = -(a + c) * htil_der[1:-1] + 1. / dt

            # # S = X[jmax+1:]
            # d = -c * (htil[2:jmax + 1] - htil[1:jmax]) + a * (htil[1:jmax] - htil[:jmax - 1])
            # d += c * (htil_der[2:jmax + 1] * S[2:jmax + 1] - htil_der[1:jmax] * S[1:jmax]) - a * (htil_der[1:jmax] * S[1:jmax] - htil_der[:jmax - 1] * S[:jmax - 1])
            # d += S[1:-1] / dt

            # a = a * htil_der[:-2]
            # c = c * htil_der[2:]

            # ngrid = len(self.xd[str(channel)]) # number of grids in The channel
            # dictM[str(channel)] = np.zeros((ngrid-2, ngrid))
            # # dictRHS[str(channel)] = np.zeros(ngrid-2)
            # for i in range(ngrid-2):
            #     dictM[str(channel)][i, [i, i+1, i+2]] = np.array([
            #         a[i], 
            #         b[i], 
            #         c[i]
            #         ])  
            # dictRHS[str(channel)] = d 
            #  
            """ implicit Crank Nicholson oscillates too much """
        #     htil = self.erodibility_stock_relation(alpha2, S / alpha1)
        #     htil_der = (self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1))
        #     x = self.xd[str(channel)]

        #     p = -F
        #     q = -1 / B * (B * T + np.gradient(B * F, x, edge_order=2))
        #     r = -1 / B * np.gradient(B * T, x, edge_order=2)

        #     h = dx[-1]
            
        #     # a_j, b_j, c_j
        #     a = (p / 2 / h**2 + q / 4 / h)[1:-1] * dt 
        #     b = (-p / h**2 + r / 2)[1:-1] * dt
        #     c = (p / 2 / h**2 - q / 4 / h)[1:-1] * dt
        #     d = S[1:-1] + a * f[2:] + b * f[1:-1] + c * f[:-2]
        #     - a * (htil_der[2:] * S[2:] - htil[2:]) 
        #     - b * (htil_der[1:-1] * S[1:-1] - htil[1:-1]) 
        #     - c * (htil_der[:-2] * S[:-2] - htil[:-2])  

        #     cp = -a * htil_der[2:] 
        #     cm = -c * htil_der[:-2] 
        #     cc = 1 - b * htil_der[1:-1] 

        #     ngrid = len(self.xd[str(channel)]) # number of grids in The channel
        #     dictM[str(channel)] = np.zeros((ngrid-2, ngrid))
        #     # dictRHS[str(channel)] = np.zeros(ngrid-2)
        #     for i in range(ngrid-2):
        #         dictM[str(channel)][i, [i, i+1, i+2]] = np.array([
        #             cm[i], 
        #             cc[i], 
        #             cp[i]
        #             ])  
        #     dictRHS[str(channel)] = d  
        
        # M = scipy.linalg.block_diag(*([dictM[str(i)] for i in range(nch)]))  # shape = (N-2*nch, N). 
        # v = np.array([])
        # for i in range(self.nch):
        #     v = np.append(v, dictRHS[str(i)])






        # The other 2*nch rows are matching and forcing conditions.
        # Do not use M = scipy.linalg.block_diag(*([dictM[key] for key in dictM.keys()])), keys in a dictionary are not ordered........

        # prescribe f at sea, x=0
        """Sea channels must have the smallest channel indices!!!!!!!"""
        for channel in Sea:
            # atil = self.atil[str(channel)]
            S = self.S[str(channel)]
            alpha1 = self.alpha1[str(channel)]
            alpha2 = self.alpha2[str(channel)]
            htil = self.erodibility_stock_relation(alpha2, S / alpha1) 
            htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1)

            c000 = self.alpha1[str(channel)][0] # depth integrated carrying capacity
            fsea = csea[channel] / c000 * self.H[str(channel)][0]
            fsea = min(1, fsea)

            row_temp = np.zeros(N)
            i = sum(Ns[k] for k in range(channel)) 
            if htil_der[0] == 0:
                row_temp[i] = 1
                v_temp = S[0]
            else:
                row_temp[i] = htil_der[0] 
                v_temp = fsea - htil[0]  + htil_der[0] * S[0]

            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
        
        # prescribe flux at x=L
        i_riv = 0
        for channel in River:
            B = self.B[str(channel)][-1]
            S = self.S[str(channel)]
            F = self.F[str(channel)][-1]
            T = self.T[str(channel)][-1]
            dx = (self.xd[str(channel)][1:] - self.xd[str(channel)][:-1])[-1]
            alpha1 = self.alpha1[str(channel)]
            alpha2 = self.alpha2[str(channel)]
            htil = self.erodibility_stock_relation(alpha2, S / alpha1) 
            htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1)
            row_temp = np.zeros(N)
                 
            BT = B * T
            BF = np.maximum(B * F, 0)

            a = BT + BF / dx
            b = BF / dx
            
            rhs = Qsed[i_riv] + a * (htil_der[-1] * S[-1] - htil[-1])
            - b * (htil_der[-2] * S[-2] - htil[-2])
            
            i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1 # last grid point

            row_temp[[i_grid, i_grid-1]] = np.array([
                        a * htil_der[-1],
                        -b * htil_der[-2]
                    ])
            v_temp = rhs

            # if htil_der[-1] == 0:
            #         htil_der[-1] = htil_der[-2]
            
            # a = B * T + 3 * B * F / 2 / dx
            # b = B * F / dx
            # # rhs = Qsed[i_riv] + a * (htil_der[-1] / alpha1[-1] * S[-1] - htil[-1])   - 2 * b * (htil_der[-2] / alpha1[-2] * S[-2] - htil[-2])    + b / 2 * (htil_der[-3] / alpha1[-3] * S[-3] - htil[-3])
            # rhs = Qsed[i_riv] + a * (htil_der[-1]  * S[-1] - htil[-1])   - 2 * b * (htil_der[-2] * S[-2] - htil[-2])    + b / 2 * (htil_der[-3] * S[-3] - htil[-3])

            # i = sum(Ns[k] for k in range(channel)) + Ns[channel] - 1 # index of the last point in The channel
    
            # row_temp = np.zeros(N)
            # row_temp[[i, i-1, i-2]] = np.array([
            #             # a * htil_der[-1] / alpha1[-1],
            #             # -2 * b * htil_der[-2] / alpha1[-2],
            #             # .5 * b * htil_der[-3] / alpha1[-3] 
            #             a * htil_der[-1] ,
            #             -2 * b * htil_der[-2],
            #             .5 * b * htil_der[-3]
            #         ])
            # v_temp = rhs
            # B = self.B[str(channel)]
            # S = self.S[str(channel)]
            # F = self.F[str(channel)]
            # T = self.T[str(channel)]
            # dx = (self.xd[str(channel)][1:] - self.xd[str(channel)][:-1])
            # alpha1 = self.alpha1[str(channel)]
            # alpha2 = self.alpha2[str(channel)]
            # atil = self.atil[str(channel)]

            # a = 0.5 * (B[jmax] + B[jmax - 1]) / B[jmax] * 0.5 * (F[jmax] + F[jmax - 1]) * 0.5 * (atil[jmax] + atil[jmax - 1]) / (dx[-1] * 0.5 * dx[-1])
            # b = -a * htil_der[-1] + 1. / dt

            # d = a * (htil[jmax] - htil[jmax - 1])
            # d += - a * (htil_der[jmax] * S[-1] - htil_der[jmax - 1] * S[-2])
            # d += S[-1] / dt

            # a = a * htil_der[-2]
            # i = sum(Ns[k] for k in range(channel)) + Ns[channel] - 1 # index of the last point in The channel
            # row_temp = np.zeros(N)
            # row_temp[[i, i-1]] = np.array([
            #         b, a
            #         ])
            # v_temp = d


            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
            i_riv += 1
  


        # continuous stock S
        for j in range(len(Vertex)):
            channels = vertex_nosign[j] # indices of channels that are connected at the branching point
            for i in range(len(channels)-1): # N channels => N-1 conditions
                row_temp = np.zeros(N)

                channel = vertex_nosign[j][i] # channel index
                if sign[j][i] == 1:     # x = L
                    i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                    row_temp[i_grid] = 1
                elif sign[j][i] == -1:
                    i_grid = sum(Ns[k] for k in range(channel))
                    row_temp[i_grid] = 1
                
                channel = vertex_nosign[j][i-1] # the other channel index
                if sign[j][i-1] == 1:
                    i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1
                    row_temp[i_grid] = -1
                elif sign[j][i-1] == -1:
                    i_grid = sum(Ns[k] for k in range(channel))
                    row_temp[i_grid] = -1
                v_temp = 0
                M = np.vstack([M, row_temp])
                v = np.append(v, v_temp)

       
        # mass conservation for suspended sediment transport
        for j in range(len(Vertex)):
            channels = vertex_nosign[j] # indices of channels that are connected at the branching point
            # one junction one condition
            row_temp = np.zeros(N)
            v_temp=0
            
            for i in range(len(channels)):
                channel = vertex_nosign[j][i] # channel index

                B = self.B[str(channel)]
              
                S = self.S[str(channel)]
                F = self.F[str(channel)]
                T = self.T[str(channel)]
                dx = self.xd[str(channel)][1:] - self.xd[str(channel)][:-1]
                alpha1 = self.alpha1[str(channel)]
                alpha2 = self.alpha2[str(channel)]
                    
                htil = self.erodibility_stock_relation(alpha2, S / alpha1) 
                htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1)
                # if htil_der[-1] == 0:
                #     htil_der[-1] = htil_der[-2]
                """ one side second order formula  """
                # if sign[j][i] == 1:     # x = L
                #     a = B[-1] * T[-1] + 3 * B[-1] * F[-1] / 2 / dx[-1]
                #     b = B[-1] * F[-1] / dx[-1]
                #     rhs = a * (htil_der[-1] * S[-1] - htil[-1])
                #     - 2 * b * (htil_der[-2] * S[-2] - htil[-2])
                #     + b / 2 * (htil_der[-3] * S[-3] - htil[-3])

                #     i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1 # last grid point

                #     row_temp[[i_grid, i_grid-1, i_grid-2]] = np.array([
                #                 a * htil_der[-1],
                #                 -2 * b * htil_der[-2],
                #                 .5 * b * htil_der[-3]
                #             ])
                #     v_temp += rhs
                    
                # elif sign[j][i] == -1:  # x = 0
                #     a = B[0] * T[0] - 3 * B[0] * F[0] / 2 / dx[0]
                #     b = B[0] * F[0] / dx[0]
                #     rhs = a * (htil_der[0] * S[0] - htil[0])
                #     + 2 * b * (htil_der[1] * S[1] - htil[1])
                #     - b / 2 * (htil_der[2] * S[2] - htil[2])

                #     i_grid = sum(Ns[var] for var in range(channel)) 

                #     row_temp[[i_grid, i_grid+1, i_grid+2]] = -np.array([
                #                 a * htil_der[0],
                #                 +2 * b * htil_der[1],
                #                 -.5 * b * htil_der[2] 
                #             ])
                #     v_temp += -rhs

                """ first order upwind """
                if sign[j][i] == 1:     # x = L
                    BT = B[-1] * T[-1]
                    BF = np.maximum(B[-1] * F[-1], 0)

                    a = BT + BF / dx[-1]
                    b = BF / dx[-1]
                    
                    rhs = + a * (htil_der[-1] * S[-1] - htil[-1])
                    - b * (htil_der[-2] * S[-2] - htil[-2])
                    
                    i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] -1 # last grid point

                    row_temp[[i_grid, i_grid-1]] = np.array([
                                a * htil_der[-1],
                                -b * htil_der[-2]
                            ])
                    v_temp += rhs
                    
                elif sign[j][i] == -1:  # x = 0
                    BT = B[0] * T[0]
                    BF = np.minimum(B[0] * F[0], 0)

                    a = BT + BF / dx[0]
                    b = BF / dx[0]
                    rhs = a * (htil_der[0] * S[0] - htil[0])
                    - b * (htil_der[1] * S[1] - htil[1])
                    
                    i_grid = sum(Ns[var] for var in range(channel)) 

                    row_temp[[i_grid, i_grid+1]] = -np.array([
                                a * htil_der[0] ,
                                - b * htil_der[1]
                            ])
                    v_temp += -rhs

            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)

        result_S = np.linalg.solve(M, v)
        # result_S = np.maximum(result_S, 0)

        f_new = {}
        S_new = {}
        for channel in range(nch):
            x0 = sum(Ns[k] for k in range(channel))
            xL = x0 + Ns[channel] 
            alpha1 = self.alpha1[str(channel)]
            alpha2 = self.alpha2[str(channel)]
            S = self.S[str(channel)]
            # atil = self.atil[str(channel)]

            htil = self.erodibility_stock_relation(alpha2, S / alpha1) 
            htil_der = self.erodibility_stock_relation_der(alpha2, S / alpha1) / (alpha1)

            # g_new[str(channel)] = htil + htil_der * (result_S[x0:xL] - S)
            # S_new[str(channel)] = 0.5 * (result_S[x0:xL] + self.S[str(channel)])
            S_new[str(channel)] = result_S[x0:xL] 
            f_new[str(channel)] = htil + htil_der * (result_S[x0:xL] - S)


        
        return f_new, S_new

    def erodibility_stock_numerical(self, eps=1e-13, max_iter=1000, dt=3600*24*10):

        #f0, Smod = 
        # self.availability_numerical(np.real(T), np.real(F), np.real(f0uncap), k, alpha1, alpha2, G) 
        #  def availability_numerical(self, T, F, f0uncap, fsea, alpha1, alpha2, G):
        # X = self.timestepping_stock(X, B, F, aeq, fsea, dx, dt, alpha1, alpha2, G, f0uncap[0])

        # aeq = f0uncap / f0uncap[0]  # scale the availability such that aeq=1 at x=0
        # k: boundary conditions  
        # BC 1: total amount of sediment in the system
        # if self.input.v('sedbc') == 'astar':
        #     astar = self.input.v('astar')
        #     k = (astar * np.trapz(B, dx=dx, axis=0) / np.trapz(B * exponent, dx=dx, axis=0))

        # # BC 2: concentration at the seaward boundary
        # elif self.input.v('sedbc') == 'csea':
        #     csea = self.input.v('csea')
        #     c000 = alpha1[0]
        #     k = csea / c000 * (self.input.v('grid', 'low', 'z', 0) - self.input.v('grid', 'high', 'z', 0))

        self.G = {}
        self.aeq = {}
        self.Smod = {}
        self.fsea = {}
        # self.f0 = {}

        self.X = self.init_erodibility_stock_network()

        # compute source G
        for i in range(self.nch):
            dc = self.input.v('network', str(i))
            x = self.xd[str(i)]
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            jmax = dc.v('grid', 'maxIndex', 'x')
            f = self.f[str(i)]
            
            self.fsea[str(i)] = min(1, self.f0uncap[str(i)][0])

            self.X[str(i)][np.where(np.isnan(self.X[str(i)]))] = 0.
            self.aeq[str(i)] = self.f0uncap[str(i)] / self.f0uncap[str(i)][0]
            self.f[str(i)] = self.aeq[str(i)] * self.X[str(i)][:jmax+1]
            

            transport = (B * T * f + B * F * np.gradient(f, x, edge_order=2))[-1]
            self.G[str(i)] =  transport / B


        error = 1
        num_iter = 0
        
        while error > eps and num_iter < max_iter:
            f_tmp = np.array(deepcopy(self.f.values()))
            self.erodibility_match()
            
            for i in range(self.nch):
                dc = self.input.v('network', str(i))
                jmax = dc.v('grid', 'maxIndex', 'x')
                x = self.xd[str(i)]
                dx = x[1:] - x[:-1]

                # f0uncap: analytical soln
                # aeq = f0uncap / f0uncap[0]
                # fsea = csea / c000 * H 

                # self.timestepping_stock(X, B, F, aeq, fsea, dx, dt, alpha1, alpha2, G, f0uncap[0])

                self.X[str(i)] = self.timestepping_stock(
                    self.X[str(i)], self.B[str(i)], self.F[str(i)], 
                    self.aeq[str(i)], self.fsea[str(i)], 
                    dx, dt, self.alpha1[str(i)], self.alpha2[str(i)], 
                    self.G[str(i)], self.f0uncap[str(i)][0]
                    )


                self.f[str(i)] = self.aeq[str(i)] * self.X[str(i)][:jmax + 1]  # compute f at beginning of time step i+1
                # Smod = self.X[str(i)][jmax + 1:]  # S at beginning of time step i+1
            self.update_transport()
            self.erodibility_match()

            num_iter += 1
            error = np.max(np.abs(np.array(self.f.values()) - f_tmp))
        print(error, num_iter)
        
        # self.erodibility_match()

    def init_erodibility_stock_network(self):

        g = {}
        S = {}
        # self.f0uncap = {}

        for channel in range(self.nch):
            dc = self.input.v('network', str(channel))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = self.xd[str(channel)]
            dx = x[1:] - x[:1]
            B = self.B[str(channel)]

            Smax = np.array([0.999])  # initial guess
            F = self.erodibility_stock_relation(self.alpha2[str(channel)][[0]], Smax) - np.array([0.999])

            while max(abs(F)) > self.TOL:
                dfdS = self.erodibility_stock_relation_der(self.alpha2[str(channel)][[0]], Smax)
                Smax = Smax - F / dfdS
                F = self.erodibility_stock_relation(self.alpha2[str(channel)][[0]], Smax) - np.array([0.999])

            Shat = self.f[str(channel)]
            F = self.erodibility_stock_relation(self.alpha2[str(channel)], Shat) - self.f[str(channel)]

            i = 0
            imax = 50
            while max(abs(F)) > self.TOL and i < imax:
                i += 1
                dfdS = self.erodibility_stock_relation_der(self.alpha2[str(channel)], Shat)
                Shat = Shat - F / (dfdS + 10 ** -22)
                F = self.erodibility_stock_relation(self.alpha2[str(channel)], Shat) - self.f[str(channel)]
                # if i == 50:
                    
            g[str(channel)] = self.erodibility_stock_relation(self.alpha2[str(channel)], Shat) * self.f[str(channel)][0] / self.f[str(channel)]

            Shat = np.minimum(Shat, Smax)

            # g[str(channel)] = np.append(g, Shat * self.alpha1[str(channel)]).reshape(2 * len(x))
        
            S[str(channel)] =  Shat * self.alpha1[str(channel)] 
            
        return g, S
            

    def erodibility_match(self):

        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        Sea = self.input.v('networksettings', 'label', 'Sea')       # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')   # labels of river channels
        csea = self.input.v('networksettings', 'forcing', 'csea')
        Qsed= self.input.v('networksettings', 'forcing', 'Qsed')
        nch = self.nch

        # prescribed sediment flux at the river heads
        i_riv = 0
        for channel in River:
            B = self.B[str(channel)][-1]
            T = self.T[str(channel)][-1]
            F = self.F[str(channel)][-1]
            x = self.xd[str(channel)]
            f = self.f[str(channel)]
            h = x[2] - x[1]
            self.f[str(channel)][-1] = max(0, min(1, (Qsed[i_riv] + 2 * B * F / h * f[-2] - B * F / 2 / h * f[-3]) / (B * T + 3 * B * F / 2 / h)))
            # self.f[str(channel)][-1] = (Qsed[i_riv] - 2 * B * F / h * f[-2] + B * F / 2 / h * f[-3]) / (B * T + 3 * B * F / 2 / h)
            
            # propagate the flux into the channel
            i_riv += 1
            self.Picard_multiSteps(str(channel), N=10)
        

        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex)         
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 
        
        for j in range(len(Vertex)):
            channels = np.asarray(vertex_nosign[j])
            # e.g. channels = [1, 6, -7]
            nc = len(channels)
            M = np.zeros((nc, nc))
            v = np.zeros(nc)
            row = 0

            # mass conservation of SSC, suspended load transport
            for i in range(len(channels)):
                channel = vertex_nosign[j][i]
                x = self.xd[str(channel)]
                f = self.f[str(channel)]
                h = x[2] - x[1]
                if sign[j][i] == 1:     # x = L, e.g. sea channels
                    B = self.B[str(channel)][-1]
                    T = self.T[str(channel)][-1]
                    F = self.F[str(channel)][-1]
                    M[row, i] = B * T + 3 * B * F / (2 * h)
                    v[row] += 2 * B * F / h * f[-2]  - B * F / (2 * h) * f[-3]
                elif sign[j][i] == -1:
                    B = self.B[str(channel)][0]
                    T = self.T[str(channel)][0]
                    F = self.F[str(channel)][0]
                    M[row, i] = -(B * T - 3 * B * F / (2 * h)) 
                    v[row] += 2 * B * F / h * f[1]  - B * F / (2 * h) * f[2]
            row += 1 


            # continuous subtidal suspended stock
            for i in range(len(channels)-1):
                channel = vertex_nosign[j][i] # channel index
                if sign[j][i] == 1:     # x = L
                    M[row, i] = self.DEL[str(channel)]
                elif sign[j][i] == -1:
                    M[row, i] = self.DE0[str(channel)] 
                
                channel = vertex_nosign[j][i-1] # the other channel index
                if sign[j][i-1] == 1:
                    M[row, i-1] = -self.DEL[str(channel)]
                elif sign[j][i-1] == -1:
                    M[row, i-1] = -self.DE0[str(channel)] 
                row += 1 

            # continuous f
            # for i in range(len(channels)-1):
            #     channel = vertex_nosign[j][i] # channel index
            #     if sign[j][i] == 1:     # x = L
            #         # B = self.B[str(channel)][-1]
            #         # T = self.T[str(channel)][-1]
            #         # F = self.F[str(channel)][-1]
            #         # x = self.xd[str(channel)]
            #         # f = self.f[str(channel)]
            #         # h = x[2] - x[1]
            #         M[row, i] = 1
            #     elif sign[j][i] == -1:
            #         M[row, i] = 1 
                
            #     channel = vertex_nosign[j][i-1] # the other channel index
            #     if sign[j][i-1] == 1:
            #         M[row, i-1] = -1
            #     elif sign[j][i-1] == -1:
            #         M[row, i-1] = -1
            #     row += 1 
            
            fbc = np.linalg.solve(M, v)
            
            # if sum(fbc>1) > 0: # then at least f > 1 in one channel
                
            #     channels_saturated = channels[fbc>=1] # channels whose f > 1
            #     channels_TBD = channels[fbc<1]      # channels to be matched again
                
            #     flux = 0    # the known flux from channel with f=1
            #     for i in channels_saturated:        # set f=1 at the junction for channels whose f>=1
            #     # for i in range(len(channels_saturated)): 
            #         # channel = channels_saturated[i]
            #         x = self.xd[str(i)]
            #         B = self.B[str(i)]
            #         T = self.T[str(i)]
            #         F = self.F[str(i)]
            #         h = x[2] - x[1]
            #         if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0]:
            #             self.f[str(i)][-1] = 1
            #             flux += B[-1] * T[-1] + B[-1] * F[-1] * (3 - 4*self.f[str(i)][-2] + self.f[str(i)][-3]) / (2 * h)
            #         elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1):
            #             self.f[str(i)][0] = 1
            #             flux += (B[0] * T[0] + B[0] * F[0] * (-3 + 4*self.f[str(i)][1] - self.f[str(i)][2]) / (2 * h))
            #         self.Picard_multiSteps(i)
                                        
            #     nc2 = len(channels_TBD)             # of channels to be matched again
            #     # f<1 in only one channel, compute f using mass conservation
            #     if nc2 == 1:                        
            #         i = channels_TBD[0] # the only channel in the set
            #         x = self.xd[str(i)]
            #         f = self.f[str(i)]
            #         h = x[2] - x[1]
            #         if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0] :
            #             B = self.B[str(i)][-1]
            #             T = self.T[str(i)][-1]
            #             F = self.F[str(i)][-1]               
            #             self.f[str(i)][-1] = (-flux - 2 * B * F / h * f[-2] + B * F / (2 * h) * f[-3]) / (B * T + 3 * B * F / (2 * h))
            #             # self.f[str(i)][-1] = (-flux + 2 * B * F / h * f[-2] - B * F / (2 * h) * f[-3]) / (B * T + 3 * B * F / (2 * h))
            #             # if f > 1 still, sediment goes into the imaginary pool
            #             # self.f[str(i)][-1] = min(1, (-flux - 2 * B * F / h * f[-2] - B * F / (2 * h) * f[-3]) / (B * T + 3 * B * F / (2 * h)))

            #         elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1)[0]:
            #             B = self.B[str(i)][0]
            #             T = self.T[str(i)][0]
            #             F = self.F[str(i)][0]
            #             self.f[str(i)][0] = (flux - 2 * B * F / h * f[1] + B * F / (2 * h) * f[2]) / (B * T - 3 * B * F / (2 * h))
            #             # if f > 1 still, sediment goes into the imaginary pool
            #             # self.f[str(i)][0] = min(1, (-flux + 2 * B * F / h * f[1] + B * F / (2 * h) * f[2]) / (B * T - 3 * B * F / (2 * h)))          
            #         self.Picard_multiSteps(i)
                
            #     # f<1 in more than one channel, compute f using mass conservation + continuous suspended stock
            #     elif nc2 > 1:                               
            #         M = np.zeros((nc2, nc2))
            #         v = np.zeros(nc2)
            #         row = 0
            #         for i in range(len(channels_TBD)):
            #             channel = channels_TBD[i]
            #             x = self.xd[str(channel)]
            #             f = self.f[str(channel)]
            #             h = x[2] - x[1]
            #             if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
            #                 B = self.B[str(channel)][-1]
            #                 T = self.T[str(channel)][-1]
            #                 F = self.F[str(channel)][-1]
        
            #                 M[row, i] = B * T + 3 * B * F / (2 * h)
            #                 v[row] += 2 * B * F / h * f[-2]  - B * F / (2 * h) * f[-3]

            #             elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
            #                 B = self.B[str(channel)][0]
            #                 T = self.T[str(channel)][0]
            #                 F = self.F[str(channel)][0]

            #                 M[row, i] = B * T - 3 * B * F / (2 * h)
            #                 v[row] += -2 * B * F / h * f[1]  + B * F / (2 * h) * f[2]
            #         # for i in range(len(channels_saturated)):

            #         v[row] -= flux                    
            #         row += 1 

            #         for i in range(len(channels_TBD)-1):
            #             channel = channels_TBD[i] # channel index
            #             if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
            #                 M[row, i] = self.DEL[str(channel)]
            #             elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
            #                 M[row, i] = self.DE0[str(channel)] 
                        
            #             channel = channels_TBD[i-1] # the other channel index
            #             if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
            #                 M[row, i-1] = -self.DEL[str(channel)]
            #             elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
            #                 M[row, i-1] = -self.DE0[str(channel)] 
            #             row += 1 

            #         fbc2 = np.linalg.solve(M, v)
            #         for i in range(len(channels_TBD)):
            #             channel = channels_TBD[i]
            #             if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
            #                 # self.f[str(channel)][-1] = min(1, fbc2[i])
            #                 self.f[str(channel)][-1] = fbc2[i]
            #             elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
            #                 # self.f[str(channel)][0] = min(1, fbc2[i])
            #                 self.f[str(channel)][0] = fbc2[i]
            #             self.Picard_multiSteps(i)
            # self.print_flux()
            if sum(fbc>=1) > 0: # then at least f > 1 in one channel

                channels_saturated = channels[fbc>=1] # channels whose f > 1
                channels_TBD = channels[fbc<1]      # channels to be matched again

                fbc2key = False
                while len(channels_TBD) > 1 and fbc2key == False:
                    nc2 = len(channels_TBD)
                    flux = 0
                    for i in channels_saturated:        # set f=1 at the junction for channels whose f>=1
                        B = self.B[str(i)]
                        T = self.T[str(i)]
                        F = self.F[str(i)]
                        h = self.h[i]
                        if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0]:  # sign = 1
                            self.f[str(i)][-1] = 1
                            flux += B[-1] * T[-1] + B[-1] * F[-1] * (3 - 4*self.f[str(i)][-2] + self.f[str(i)][-3]) / (2 * h)
                        elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1):  # sign = -1
                            self.f[str(i)][0] = 1
                            flux += -(B[0] * T[0] + B[0] * F[0] * (-3 + 4*self.f[str(i)][1] - self.f[str(i)][2]) / (2 * h))
                        # self.Picard_multiSteps(i, N=20)

                    M = np.zeros((nc2, nc2))
                    v = np.zeros(nc2)
                    row = 0
                    for i in range(len(channels_TBD)):  # horizontal flux conservation
                        channel = channels_TBD[i]
                        x = self.xd[str(channel)]
                        f = self.f[str(channel)]
                        h = x[2] - x[1]
                        if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
                            B = self.B[str(channel)][-1]
                            T = self.T[str(channel)][-1]
                            F = self.F[str(channel)][-1]
                            M[row, i] = B * T + 3 * B * F / (2 * h)
                            v[row] += 2 * B * F / h * f[-2]  - B * F / (2 * h) * f[-3]
                        elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                            B = self.B[str(channel)][0]
                            T = self.T[str(channel)][0]
                            F = self.F[str(channel)][0]
                            M[row, i] = -(B * T - 3 * B * F / (2 * h))
                            v[row] += 2 * B * F / h * f[1] - B * F / (2 * h) * f[2]
                    v[row] -= flux                    
                    row += 1 

                    for i in range(len(channels_TBD)-1):    # continuous suspended stock
                        channel = channels_TBD[i] # channel index
                        if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
                            M[row, i] = self.DEL[str(channel)]
                        elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                            M[row, i] = self.DE0[str(channel)] 
                        
                        channel = channels_TBD[i-1] # the other channel index
                        if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                            M[row, i-1] = -self.DEL[str(channel)]
                        elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                            M[row, i-1] = -self.DE0[str(channel)] 
                        row += 1 

                    fbc2 = np.linalg.solve(M, v)
                    temp = deepcopy(channels_TBD)
                    for i in range(len(channels_TBD)):
                        channel = channels_TBD[i]
                        if fbc2[i] >= 1:
                            channels_saturated = np.append(channels_saturated, channel)
                            if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                                self.f[str(channel)][-1] = 1
                            elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                self.f[str(channel)][0] = 1
                        else:
                            if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                                self.f[str(channel)][-1] = 0.5 * (fbc2[i] + self.f[str(channel)][-1])
                                # self.f[str(channel)][-1] = fbc[i]
                            elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                self.f[str(channel)][0] = 0.5 * (fbc2[i] + self.f[str(channel)][0])
                                # self.f[str(channel)][0] = fbc[i]
                        # self.Picard_multiSteps(i)
                    channels_TBD = np.setdiff1d(channels, channels_saturated)
                    if len(temp) == len(channels_TBD):
                        fbc2key = True
                    
                
                # At the end of above while loop, compute f only if len(channels_TBD) == 1, otherwise all channels are detemrined. 
                if len(channels_TBD) == 1:
                    flux = 0
                    for i in channels_saturated:        # set f=1 at the junction for channels whose f>=1
                        x = self.xd[str(i)]
                        B = self.B[str(i)]
                        T = self.T[str(i)]
                        F = self.F[str(i)]
                        h = x[2] - x[1]
                        if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0]:  # sign = 1
                            self.f[str(i)][-1] = 1
                            flux += B[-1] * T[-1] + B[-1] * F[-1] * (3 - 4*self.f[str(i)][-2] + self.f[str(i)][-3]) / (2 * h)
                        elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1):  # sign = -1
                            self.f[str(i)][0] = 1
                            flux += -(B[0] * T[0] + B[0] * F[0] * (-3 + 4*self.f[str(i)][1] - self.f[str(i)][2]) / (2 * h))
                        # self.Picard_multiSteps(i, N=100)

                    i = channels_TBD[0] # the only channel in the set
                    x = self.xd[str(i)]
                    f = self.f[str(i)]
                    h = x[2] - x[1]
                    if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0] :
                        B = self.B[str(i)][-1]
                        T = self.T[str(i)][-1]
                        F = self.F[str(i)][-1]
                        # if f > 1 still, sediment goes into the imaginary pool
                        self.f[str(i)][-1] = min(
                            1, 
                            (-flux + 2 * B * F / h * f[-2] - B * F / (2 * h) * f[-3]) / (B * T + 3 * B * F / (2 * h))
                        )
                    elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1)[0]:
                        B = self.B[str(i)][0]
                        T = self.T[str(i)][0]
                        F = self.F[str(i)][0]
                        # if f > 1 still, sediment goes into the imaginary pool
                        # self.f[str(i)][0] = min(
                        #     1, 
                        #     0.5 * (self.f[str(i)][0] + (-flux - 2 * B * F / h * f[1] + B * F / (2 * h) * f[2]) / (B * T - 3 * B * F / (2 * h)))
                        # )          
                        f_temp = (-flux - 2 * B * F / h * f[1] + B * F / (2 * h) * f[2]) / (B * T - 3 * B * F / (2 * h))
                        if f_temp > 0.95:
                            self.f[str(i)][0] = 1
                        else:
                            self.f[str(i)][0] = min(1, f_temp)   
                    

                # f > 1 in all channels
                if sum(fbc>1) == len(channels): 

                    if self.flux_check(vertex_nosign[j], sign[j]) >= -.05: # shared pool increases over time
                        # f = 1 in all channels
                        for i in range(len(channels)):
                            channel = vertex_nosign[j][i]
                            if sign[j][i] == 1:
                                self.f[str(channel)][-1] = 1
                            elif sign[j][i] == -1:
                                self.f[str(channel)][0] =  1
                            # self.Picard_multiSteps(i)
                    else:   
                        # match channels again: channels with flux into the pool have f=1, other channels need to be matched using conserved horizontal flux and continuous suspended stock
                        saturated = []
                        TBD = []
                        for i in range(len(channels)):
                            channel = vertex_nosign[j][i]
                            if sign[j][i] == 1:
                                flux_cap = self.B[str(channel)][-1] * self.T[str(channel)][-1]            
                            elif sign[j][i] == -1:
                                flux_cap = -1 * self.B[str(channel)][0] * self.T[str(channel)][0]          
                            if flux_cap >= 0:
                                saturated.append(channel)
                            else:
                                TBD.append(channel)
                        while len(TBD) > 1 and fbc2key == False:
                            nc2 = len(TBD)
                            flux = 0
                            for i in saturated:        # set f=1 at the junction for channels whose f>=1
                                B = self.B[str(i)]
                                T = self.T[str(i)]
                                F = self.F[str(i)]
                                h = self.h[i]
                                if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0]:  # sign = 1
                                    self.f[str(i)][-1] = 1
                                    flux += B[-1] * T[-1] + B[-1] * F[-1] * (3 - 4*self.f[str(i)][-2] + self.f[str(i)][-3]) / (2 * h)
                                elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1):  # sign = -1
                                    self.f[str(i)][0] = 1
                                    flux += -(B[0] * T[0] + B[0] * F[0] * (-3 + 4*self.f[str(i)][1] - self.f[str(i)][2]) / (2 * h))
                                # self.Picard_multiSteps(i, N=10)

                            M = np.zeros((nc2, nc2))
                            v = np.zeros(nc2)
                            row = 0
                            for i in range(len(TBD)):  # horizontal flux conservation
                                channel = TBD[i]
                                f = self.f[str(channel)]
                                h = self.h[channel]
                                if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
                                    B = self.B[str(channel)][-1]
                                    T = self.T[str(channel)][-1]
                                    F = self.F[str(channel)][-1]
                                    M[row, i] = B * T + 3 * B * F / (2 * h)
                                    v[row] += 2 * B * F / h * f[-2]  - B * F / (2 * h) * f[-3]
                                elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                    B = self.B[str(channel)][0]
                                    T = self.T[str(channel)][0]
                                    F = self.F[str(channel)][0]
                                    M[row, i] = -(B * T - 3 * B * F / (2 * h))
                                    v[row] += 2 * B * F / h * f[1]  - B * F / (2 * h) * f[2]
                            v[row] -= flux                    
                            row += 1 

                            for i in range(len(TBD)-1):    # continuous suspended stock
                                channel = TBD[i] # channel index
                                if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:     # x = L
                                    M[row, i] = self.DEL[str(channel)]
                                elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                    M[row, i] = self.DE0[str(channel)] 
                                
                                channel = TBD[i-1] # the other channel index
                                if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                                    M[row, i-1] = -self.DEL[str(channel)]
                                elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                    M[row, i-1] = -self.DE0[str(channel)] 
                                row += 1 

                            fbc2 = np.linalg.solve(M, v)
                            temp = deepcopy(TBD)
                            for i in range(len(TBD)):
                                channel = TBD[i]
                                if fbc2[i] >= 1:
                                    saturated.append(channel) 
                                    if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                                        self.f[str(channel)][-1] = 1
                                    elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                        self.f[str(channel)][0] = 1
                                else:
                                    if ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==1)[0]:
                                        self.f[str(channel)][-1] = 0.5 * (fbc2[i] + self.f[str(channel)][-1])
                                    elif ((np.asarray(sign[j]))[vertex_nosign[j]==channel]==-1)[0]:
                                        self.f[str(channel)][0] = 0.5 * (fbc2[i] + self.f[str(channel)][0])
                                # self.Picard_multiSteps(i, N=10)
                            TBD = list(set(channels) - set(saturated))
                            if len(temp) == len(TBD):
                                fbc2key = True

                        if len(TBD) == 1:
                            flux = 0
                            for i in saturated:        # set f=1 at the junction for channels whose f>=1
                                B = self.B[str(i)]
                                T = self.T[str(i)]
                                F = self.F[str(i)]
                                h = self.h[i]
                                if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0]:  # sign = 1
                                    self.f[str(i)][-1] = 1
                                    flux += B[-1] * T[-1] + B[-1] * F[-1] * (3 - 4*self.f[str(i)][-2] + self.f[str(i)][-3]) / (2 * h)
                                elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1):  # sign = -1
                                    self.f[str(i)][0] = 1
                                    flux += -(B[0] * T[0] + B[0] * F[0] * (-3 + 4*self.f[str(i)][1] - self.f[str(i)][2]) / (2 * h))
                                # self.Picard_multiSteps(i, N=100)

                            i = TBD[0] # the only channel in the set
                            f = self.f[str(i)]
                            h = self.h[i]
                            if ((np.asarray(sign[j]))[vertex_nosign[j]==i]==1)[0] :
                                B = self.B[str(i)][-1]
                                T = self.T[str(i)][-1]
                                F = self.F[str(i)][-1]
                                # if f > 1 still, sediment goes into the imaginary pool
                                self.f[str(i)][-1] = min(
                                    1, 
                                    (-flux + 2 * B * F / h * f[-2] - B * F / (2 * h) * f[-3]) / (B * T + 3 * B * F / (2 * h))
                                )
                            elif ((np.asarray(sign[j]))[vertex_nosign[j]==i]==-1)[0]:
                                B = self.B[str(i)][0]
                                T = self.T[str(i)][0]
                                F = self.F[str(i)][0]
                                # if f > 1 still, sediment goes into the imaginary pool
                                self.f[str(i)][0] = min(
                                    1, 
                                    (-flux - 2 * B * F / h * f[1] + B * F / (2 * h) * f[2]) / (B * T - 3 * B * F / (2 * h))
                                )          
                            # self.Picard_multiSteps(i, N=10)
                                    

            else:   # f < 1 in all channels: it is the solution, update values for f at junctions
                
                for i in range(len(channels)):
                                    
                    channel = vertex_nosign[j][i]
                    if sign[j][i] == 1:
                        # self.f[str(channel)][-1] = 0.5 * (self.f[str(channel)][-1] + fbc[i])
                        self.f[str(channel)][-1] = fbc[i]
                    elif sign[j][i] == -1:
                        # self.f[str(channel)][0] =  0.5 * (self.f[str(channel)][0] + fbc[i])
                        self.f[str(channel)][0] =  fbc[i]
                    
                    # self.Picard_multiSteps(i, N=2)
            
            # print(self.flux_check())

    def print_flux(self):      
        flux0 = self.B['0'][-1] * self.T['0'][-1] * self.f['0'][-1] + self.B['0'][-1] * self.F['0'][-1] * (3 * self.f['0'][-1] - 4*self.f['0'][-2] + self.f['0'][-3]) / (2 * self.h[0]) 

        flux1 = self.B['1'][-1] * self.T['1'][-1] * self.f['1'][-1] + self.B['1'][-1] * self.F['1'][-1] * (3 * self.f['1'][-1] - 4*self.f['1'][-2] + self.f['1'][-3]) / (2 * self.h[1]) 

        flux2 = self.B['2'][0] * self.T['2'][0] * self.f['2'][0] + self.B['2'][1] * self.F['2'][1] * (-3 * self.f['2'][0] + 4*self.f['2'][1] - self.f['2'][2]) / (2 * self.h[2])
        print(flux0 + flux1 - flux2)


    def flux_check(self, vertex_nosign, sign):

        flux = 0
        for i in range(len(vertex_nosign)):
            if sign[i] > 0:
                flux += self.B[str(i)][-1] * self.T[str(i)][-1] 
            else:
                flux -= self.B[str(i)][0] * self.T[str(i)][0]
        return flux

    def Picard_multiSteps(self, i, N=10):
        # propagate the boundary condition into the channel
        for j in range(N):
            self.f[str(i)] = self.Picard_step(self.f[str(i)], self.xd[str(i)], self.B[str(i)]*self.T[str(i)], self.BTx[str(i)], self.B[str(i)] * self.F[str(i)], self.BFx[str(i)])

    def plot_transport(self, ls='-'):
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(8, 4))
        for i in range(self.nch):
            f = self.f[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            xx = self.xd[str(i)]
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))

            plt.plot(x, trans, color=color[i], label=label[i], linestyle=ls)
            # plt.ylabel('Transport')
        plt.legend()
        plt.ylim([-.25e4,.25e4])
        # plt.ylim([0,1.1])
        # plt.xlim([-1.4e5,5e4])
        plt.xlim([-1.35e5,1e5])

    def plot_Tcomponnets(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        # dc = self.input.v('network', '0')
        keys = self.transport.getKeysOf('0', 'T')
        for mod in keys:
        # for mod in ['stokes', 'nostress']:
            plt.figure(figsize=(10, 6.18))
            for i in range(self.nch):
                f = self.f[str(i)]
                # self.alpha1[str(i)]
                dc = self.input.v('network', str(i))
                jmax = dc.v('grid', 'maxIndex', 'x')
                # T = self.transport.v(str(i), 'T', mod, range(jmax+1), 0)
                T = self.transport.v(str(i), 'T', mod)
                # T0 = self.transport.v(str(i), 'T', mod, 'TM0')
                # T2 = self.transport.v(str(i), 'T', mod, 'TM2')
                # T4 = self.transport.v(str(i), 'T', mod, 'TM4')
                # F = self.transport.v(str(i), 'F', string, freq)
        
                B = self.B[str(i)]
                # c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
                x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
                plt.plot(x, -T, color=color[i], label=label[i])
                # plt.plot(x, -T0, color=color[i], label=label[i])
                # plt.plot(x, -T2, color=color[i], ls='--')
                # plt.plot(x, -T4, color=color[i], ls='dotted')
            if mod == 'nostress':
                plt.ylabel('T ' + 'vel.-depth asym. (kg $^{-1}$ m$^{-1}$ s$^{-1}$)')
            else:
                plt.ylabel('T ' + mod + '(kg m$^{-1}$ s$^{-1}$)')
            plt.legend()
            plt.xlabel('Position (km)')
            # plt.savefig(mod+'.pdf', dpi=None, facecolor='w', edgecolor='w')


        # for mod in self.transport.getKeysOf('0', 'T', 'tide'):
        #     plt.figure()
        #     for i in range(self.nch):
        #         f = self.f[str(i)]
        #         # self.alpha1[str(i)]
        #         dc = self.input.v('network', str(i))
        #         T = self.transport.v(str(i), 'T', 'tide', mod)
        #         # F = self.transport.v(str(i), 'F', string, freq)
        
        #         jmax = dc.v('grid', 'maxIndex', 'x')
        #         B = self.B[str(i)]
        #         # c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #         x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #         plt.plot(x, B * T, color=color[i], label=label[i])
        #     plt.ylabel('BT tide' + mod)
        #     plt.legend()

        # mod = 'diffusion_tide'
        # plt.figure()
        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        #     T = self.transport.v(str(i), 'T', mod)
        #     # F = self.transport.v(str(i), 'F', string, freq)
    
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     kmax = dc.v('grid', 'maxIndex', 'z')
        #     B = self.B[str(i)]
        #     c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), 0, 0))
        #     c0x = dc.d('hatc0', 'a', range(jmax+1), 0, 0, dim='x') 
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, B * T, color=color[i], label=label[i])
        #     # plt.plot(x, np.real(c0x), color=color[i], label=label[i])
        # plt.ylabel('BT ' + mod)
        # plt.legend()


    def plot(self):
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')


        self.nch = 3
        fig, axs = plt.subplots(2,2, figsize=(9, 9))
        for i in range(self.nch):
            f = self.f0[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
            cb = dc.v('hatc0', 'a', range(jmax+1), 0, 0) * f
            axs[0,0].plot(x, cb, color=color[i], label=label[i])
        axs[0,0].legend()
        axs[0,0].set_ylabel('$c_b$ (kg/m$^3$)')    
        #    
        # for i in range(self.nch):
        #     f = self.f0[str(i)]
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)

        #     # axs[0,0].plot(x, self.g[str(i)], color=color[i], label=label[i])
        #     p = axs[0, 0].plot(x, self.g[str(i)], color=color[i], label=label[i])
        #     axs[0, 0].plot(x, self.atil[str(i)], '--', color=p[-1].get_color())
        # axs[0,0].legend()
        # # axs[0,0].set_ylabel('$\\tilde{a}$ ')
        # axs[0,0].set_ylabel('solid: $g$, dashed: atil ')

        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
        #     cb = dc.v('hatc0', 'a', range(jmax+1), -1, 0) * f
        #     axs[1].plot(x, cb, label=label[i])
        # axs[1].legend()
        # axs[1].set_ylabel('$c_b$')
        

        for i in range(self.nch):
            f = self.f0[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            xx = self.xd[str(i)]
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))
            p = axs[0, 1].plot(x, trans, color=color[i], label=label[i])
            axs[0, 1].plot(x, B*T, '--', color=p[-1].get_color())
        axs[0, 1].legend()
        axs[0, 1].set_ylabel('transport (kg/s)')


        for i in range(self.nch):
            # f = self.f0[str(i)]
            f = self.f0[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
            axs[1, 0].plot(x, f, color=color[i], label=label[i])
        axs[1, 0].legend()
        axs[1, 0].set_ylabel('$f$')

        for i in range(self.nch):
            f = self.f0[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            xx = self.xd[str(i)]
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            # dSdt = np.gradient(flux, xx, edge_order=2)
            # S = self.alpha1[str(i)] 
            # p = axs[1, 1].plot(x, S*f, color=color[i], label=label[i])
            axs[1, 1].plot(x, self.S[str(i)], color=color[i], label=label[i])

            # axs[1, 1].plot(x, S, '--', color=p[-1].get_color())
        axs[1, 1].legend()
        axs[1, 1].set_ylabel('SS (kg/m$^2$)')

        # plt.figure(figsize=(8, 4))
        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     # f = self.f0uncap[str(i)]
        #     # self.alpha1[str(i)]
        #     dc = self.input.v('network', str(i))
            
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, f, color=color[i], label=label[i])
        # plt.legend()
        # plt.ylabel('f')
        # plt.xlim([-1.4e5,5e4])
        # plt.ylim([-0.1,1.1])
        # plt.ylim([0, 0.8])


        # plt.figure(figsize=(8, 4))
        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     c00 = (dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x,  c00 * f, color=color[i], label=label[i])
        # plt.legend()
        # plt.ylabel('$c_b$')


        # plt.figure(figsize=(8, 4))
        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x,  np.angle(dc.v('u1', range(jmax+1), -1, 2)), color=color[i], label=label[i])
        #     # print(dc.v('Q', 'dp'))
        # plt.legend()
        # plt.ylabel('$u$')


        # plt.figure(figsize=(8, 8))
        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     xx = self.xd[str(i)]
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     B = self.B[str(i)]
        #     T = self.T[str(i)]
        #     F = self.F[str(i)]
        #     trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))
        #     # transx = np.gradient(trans, x, edge_order=2)

        #     plt.plot(x, trans, color=color[i], label=label[i])
        #     plt.plot(x, B * T, color=color[i], linestyle='--')
        #     # plt.plot(x, F, color=color[i], label=label[i])
        #     # plt.plot(x, T, color=color[i], linestyle='--')
        #     plt.ylabel('Transport')
        # plt.legend()
        # plt.ylim([-2e4, 1e4])
        # plt.ylim([-3e3, 3e3])
        # plt.xlim([-1.4e5,5e4])
        # plt.xlim([-1.35e5,0])
        # plt.ylim([-5,20])


        # plt.figure(figsize=(8, 4))
        # t = np.linspace(0, 2 * np.pi, 100)
        # for i in range(self.nch):
        #     deposition = np.zeros(100)
        #     erosion = np.zeros(100)
            
        #     f = self.f[str(i)]

        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     bottom_stock = np.zeros(jmax+1)
        #     u0b = dc.v('u0', 'tide', range(jmax+1), 0, 1)
        #     u0b_M0 = absoluteU(u0b, 0)
        #     u0b_M4 = absoluteU(u0b, 2)
        #     ws = self.input.v('ws0', range(jmax+1), 0, 0)
        #     sf = self.input.v('Roughness', range(jmax+1), 0, 0)
        #     E = self.finf * ws * self.RHOS * sf / self.GPRIME / self.DS
        #     for k in range(jmax+1):
        #         deposition = np.real(ws[k] * (dc.v('hatc0', 'a', k, 0, 0) + dc.v('hatc0', 'a', k, 0, 2) * np.exp(2j*t)))
        #         erosion = E[k] * np.real(u0b_M0[k] + u0b_M4[k] * np.exp(2j*t))
        #         bottom_stock[k] = 0.5 * np.mean(np.abs(deposition - erosion))

        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 

        #     plt.plot(x, bottom_stock * f * self.B[str(i)], color=color[i], label=label[i])
  
        #     plt.ylabel('subtidal bottom stock')
        # plt.legend()



        # plt.ylim([-.3e4,.3e4])
        # plt.xlim([-1.4e5,5e4])
        # plt.xlim([-1.35e5,1e5])

         

        # plt.figure()
        # string = 'tide'

        # for i in range(self.nch):
        #     f = self.f[str(i)]
        #     # self.alpha1[str(i)]
        #     dc = self.input.v('network', str(i))
        #     T = self.transport.v(str(i), 'T', string)
        #     # F = self.transport.v(str(i), 'F', string, freq)
    
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     B = self.B[str(i)]
        #     # c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, B * T, color=color[i], label=label[i])
        # plt.ylabel('BT ' + string)
        # plt.legend()


        # plot transport capacity
        # fig1 = plt.figure()
        # string = 'baroc'
        # # diffusion_river
        # # diffusion_tide
        # # sedadv
        # freq = 'TM0'
        # # freq = 'FM0'
        # for i in range(nch):
        #     f = self.f[str(i)]
        #     # self.alpha1[str(i)]
        #     dc = self.input.v('network', str(i))
        #     T = self.transport.v(str(i), 'T', string, freq)
        #     # F = self.transport.v(str(i), 'F', string, freq)
        #     # T = self.transport.v(str(i), 'T', string)
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     B = self.B[str(i)]
        #     c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, B * T * f , color=color[i], label=label[i])
        # plt.ylabel(string + ' ' + freq )
        # plt.legend()
        # plt.savefig("/Users/jinyangwang/Desktop/plots/manu3/" + string + "/TM0", dpi=150)
        # # plt.show()


        # plt.figure()
        # freq = 'TM2'
        # for i in range(nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        #     T = self.transport.v(str(i), 'T', string, freq)
        #     # F = self.transport.v(str(i), 'F', string, freq)
        #     # T = self.transport.v(str(i), 'T', string)
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     B = self.B[str(i)]
        #     c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, B * T * f , color=color[i], label=label[i])
        # plt.ylabel(string + ' ' + freq )
        # plt.legend()

        # plt.figure()
        # freq = 'TM4'
        # for i in range(nch):
        #     f = self.f[str(i)]
        #     dc = self.input.v('network', str(i))
        #     T = self.transport.v(str(i), 'T', string, freq)
        #     # F = self.transport.v(str(i), 'F', string, freq)
        #     # T = self.transport.v(str(i), 'T', string)
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     B = self.B[str(i)]
        #     c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     plt.plot(x, B * T * f , color=color[i], label=label[i])
        # plt.ylabel(string + ' ' + freq )
        # plt.legend()


        # plt.figure(figsize=(8, 8))
        # for i in range(self.nch):
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     xx = self.xd[str(i)]
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
        #     B = self.B[str(i)]
        #     T = self.transport.v(str(i), 'T', 'nostress')
        #     plt.plot(x, B * T, color=color[i], linestyle='--')
        #     # plt.plot(x, F, color=color[i], label=label[i])
        #     # plt.plot(x, T, color=color[i], linestyle='--')
        #     plt.ylabel('Transport')
        # plt.legend()


        plt.show()


    def plot_MDE(self):
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')

        # self.nch = 3
        fig, axs = plt.subplots(2,2, figsize=(9, 9))
        for i in range(self.nch):
            f = self.f[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
            cb = dc.v('hatc0', 'a', range(jmax+1), -1, 0) * f
            p = axs[0,0].plot(x, cb, color=color[i], label=label[i])
            # axs[0,0].plot(x, self.alpha1[str(i)] * f / self.H[str(i)], '--', color=p[-1].get_color())
            # axs[0,0].plot(x,  dc.v('hatc0', 'a', range(jmax+1), 0, 0) * f, '--', color=p[-1].get_color())

        axs[0,0].legend()
        axs[0,0].set_ylabel('$c_{b}$ (kg/m$^3$)')    
        #    
             

        for i in range(self.nch):
            f = self.f[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            xx = self.xd[str(i)]
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))
            p = axs[0, 1].plot(x, trans, color=color[i], label=label[i])
            axs[0, 1].plot(x, B*T*f, '--', color=p[-1].get_color())
        axs[0, 1].legend()
        axs[0, 1].set_ylabel('transport (kg/s)')


        for i in range(self.nch):
            f = self.f[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
            axs[1, 0].plot(x, f, color=color[i], label=label[i])
        axs[1, 0].legend()
        axs[1, 0].set_ylabel('$f$')

        for i in range(self.nch):
            f = self.f[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            xx = self.xd[str(i)]
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            # dSdt = np.gradient(flux, xx, edge_order=2)
            S = self.alpha1[str(i)] 
            p = axs[1, 1].plot(x, S, color=color[i], label=label[i])

            # axs[1, 1].plot(x, S, '--', color=p[-1].get_color())
        axs[1, 1].legend()
        axs[1, 1].set_ylabel('Carrying capacity (kg/m$^2$)')


    def plot_manu(self):
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        # label = ['1', '2', '3']

        channels=[0, 1, 2]

        # plt.figure(figsize=(10, 6.18))
        # plt.xlabel('Position (km)')
        # ax1 = plt.gca()
        # # ax2 = ax1.twinx()
        # for i in channels:
        #     f = self.f[str(i)]
        #     alpha1 = self.alpha1[str(i)]
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
        #     ax1.plot(x, alpha1 * f / self.H[str(i)], color=color[i], label=label[i])
        #     # ax2.plot(x, f , color=color[i], ls='--')
        # ax1.legend()
     
        # # ax1.set_ylabel('Solid: stock S (kg m$^{-2}$)')
        # ax1.set_ylabel('Solid: $\\bar{c}$ (kg m$^{-3}$)')
        # # ax2.set_ylabel('Dashed: $f$') 
        # plt.xlabel('Position (km)')
        # # plt.xlim([-23,65])
        # # ax1.set_ylim([0.25, 0.65])
        # ax1.set_ylim([0.02, 0.055])
        # ax1.set_yticks([ 0.02, 0.03,0.04,0.05])  
        # # ax2.set_ylim([0, 0.25])
        # # ax2.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25]) 
        # ax1.set_xlabel('Position (km)') 
        # plt.savefig('c_ref.pdf', dpi=None, facecolor='w', edgecolor='w')

        plt.figure(figsize=(10, 6.18))
        plt.xlabel('Position (km)')
        ax1 = plt.gca()
        # ax2 = ax1.twinx()
        for i in channels:
            f = self.f[str(i)]
            alpha1 = self.alpha1[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            trans = (B * T * f + B * F * np.gradient(f, self.xd[str(i)], edge_order=2))
            ax1.plot(x, -trans, color=color[i], label=label[i])
            ax1.plot(x, -B*T*f , color=color[i], ls='--')
        ax1.set_ylim([-10,70])
        # ax1.set_yticks([-100, -80, -60, -40, -20, 0, 20, 40])  
        # ax1.set_ylabel('Solid: $\mathcal{F}$ (kg s$^{-1}$)')
        ax1.set_ylabel('Sediment transport (kg s$^{-1}$)')
        # ax2.set_ylabel('Dashed: $BTf$ (kg s$^{-1}$)') 
        # ax2.set_yticks([-100, -80, -60, -40, -20, 0, 20, 40]) 
        ax1.set_xlabel('Position (km)')
        ax1.text(-38, 10, "solid: $\mathcal{F}$", color="k", fontsize=18)
        ax1.text(-38, 5, "dashed: $BTf$", color="k", fontsize=18) 
        ax1.text(-38, 15, "X: sediment trapping zone", color="k", fontsize=18) 
        plt.scatter(9.8,2.9, marker="x", color='k', s=300, zorder=3)
        plt.scatter(12.6,47.75, marker="x", color='k', s=300, zorder=3)
        # ax1.scatter(14.83, 204, marker="x", color='k', s=300, zorder=3)
        # ax1.scatter(44.6,1292, marker="x", color='k', s=300, zorder=3)
        # ax1.set_ylim([-150, 2200])
        # plt.savefig('NST_ref.pdf', dpi=None, facecolor='w', edgecolor='w')


        # plt.figure(figsize=(10, 6.18))
        # plt.xlabel('Position (km)')
        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # for i in channels:
        #     f = self.f[str(i)]
        #     alpha1 = self.alpha1[str(i)]
        #     dc = self.input.v('network', str(i))
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
        #     B = self.B[str(i)]
        #     f = self.f[str(i)]
        #     F = self.F[str(i)]
        #     alpha = self.alpha1[str(i)]
        #     # trans = (B * T * f + B * F * np.gradient(f, self.xd[str(i)], edge_order=2))
        #     ax1.plot(x, f, color=color[i], label=label[i])
        #     ax2.plot(x, alpha , color=color[i], ls='--')
        # # ax1.set_ylim([-2.5, 1.5])
        # ax1.set_ylim([0.04, 0.09])
        # # ax1.set_ylim([0,0.4])
        # ax1.set_yticks([0.04, 0.05, 0.06, 0.07,0.08,0.09])  
        # # ax1.set_ylabel('Solid: $T$ (kg m$^{-1}$ s$^{-1}$)')
        # ax1.set_ylabel('Solid: $f$')
        # ax2.set_ylim([6, 11])
        # ax2.set_ylabel('Dashed: $\hat{C}$ (kg m$^{-2}$)') 
        # ax2.set_yticks([6,7,8,9,10,11]) 
        # # ax2.set_yticks([0, 10, 20, 30, 40]) 

        # ax2.set_xlabel('Position (km)') 
        # plt.savefig('fhatC_ref.pdf', dpi=None, facecolor='w', edgecolor='w')



        # submods = ['river', 'baroc', 'sedadv', 'nostress', 'adv', 'stokes']
        # colors = ['wheat', 'g', 'b', 'm', 'r', 'c']
        submods = ['river', 'baroc']
        colors = ['c', 'm']


        plt.figure(figsize=(10, 6.18))
        for j in range(len(submods)):
            mod = submods[j]
            for i in channels:
                f = self.f[str(i)]
                dc = self.input.v('network', str(i))
                if mod == 'adv':
                    T = self.transport.v(str(i), 'T', mod) +  self.transport.v(str(i), 'T', 'dp')       
                else:
                    T = self.transport.v(str(i), 'T', mod)        
                jmax = dc.v('grid', 'maxIndex', 'x')

                # c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
                x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
                if i == 0:
                    p = plt.plot(x, -T, color=colors[j], ls = '--')
                else:
                    p = plt.plot(x, -T, color=colors[j])
        for i in channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            T = self.T[str(i)] - self.transport.v(str(i), 'T', 'river') - self.transport.v(str(i), 'T', 'baroc')
            if i == 0:
                plt.plot(x,-T, color='r', ls = '--')
            else:
                plt.plot(x, -T, color='r')
        for i in channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            T = self.T[str(i)]
            if i == 0:
                plt.plot(x,-T, color='k', ls = '--')
            else:
                plt.plot(x, -T, color='k')

        plt.ylabel('T (kg m$^{-1}$ s$^{-1}$)')
        plt.xlabel('Position (km)')

        # submods = ['total', 'river', 'baroc', 'sedadv', 'vel.-depth asym.', 'advection', 'Stokes']
        submods = ['total', 'river', 'baroclinic', 'other']
        # colors = ['k', 'wheat', 'g', 'b', 'm', 'r', 'c']
        colors = ['k', 'c', 'm', 'r']
        lines = [Line2D([0], [0], color=c, linewidth=2) for c in colors]
        ax = plt.gca()
        leg1 = ax.legend(lines, submods)
        leg1.get_frame().set_alpha(0.3)
        # leg1.get_frame().set_facecolor((0, 0, 1, 0.1))
        # ax.text(-38.2, -0.5, "solid: channel 2 and 3", color="k", fontsize=18)
        # ax.text(-38.2, -0.7, "dashed: channel 1", color="k", fontsize=18)
        # ax.text(-38.2, -0.6, "solid: channel 2 and 3", color="k", fontsize=20)
        # ax.text(-38.2, -0.8, "dashed: channel 1", color="k", fontsize=20)
        # ax.text(25, 8, "solid: channel 2 and 3", color="k", fontsize=16)
        # ax.text(25, 7, "dashed: channel 1", color="k", fontsize=16)
        # ax.text(36, 6, "X: trapping zone", color="k", fontsize=18)
        # plt.savefig('Tcomps_ref.pdf', dpi=None, facecolor='w', edgecolor='w')

 
    

    def plot_manu_YE(self):
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 

        plt.figure(figsize=(10, 6.18))
        plt.xlabel('Position (km)')
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        for i in range(self.nch):
            f = self.f[str(i)]
            alpha1 = self.alpha1[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            ax1.plot(x, alpha1, color=color[i], label=label[i])
            ax2.plot(x, f , color=color[i], ls='--')
        ax1.legend()
     
        ax1.set_ylabel('Solid: $\hat{C}$ (kg m$^{-2}$)')
        ax2.set_ylabel('Dashed: $f$') 
        plt.xlabel('Position (km)')
        ax1.set_ylim([7, 28])
        ax1.set_yticks([7, 14, 21, 28])  
        ax2.set_ylim([0, 0.6])
        ax2.set_yticks([0, 0.2, 0.4, 0.6]) 
        ax2.set_xlabel('Position (km)') 

        plt.figure(figsize=(10, 6.18))
        plt.xlabel('Position (km)')
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        for i in range(self.nch):
            f = self.f[str(i)]
            alpha1 = self.alpha1[str(i)]
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            B = self.B[str(i)]
            T = self.T[str(i)]
            F = self.F[str(i)]
            trans = (B * T * f + B * F * np.gradient(f, self.xd[str(i)], edge_order=2))
            ax1.plot(x, trans, color=color[i], label=label[i])
            ax2.plot(x, B*T*f , color=color[i], ls='--')
        ax1.set_ylim([-5000,3000])
        # ax1.set_yticks([-100, -80, -60, -40, -20, 0, 20, 40])  
        ax1.set_ylabel('Solid: $\mathcal{F}$ (kg s$^{-1}$)')
        ax2.set_ylim([-5000,3000])
        ax2.set_ylabel('Dashed: $BTf$ (kg s$^{-1}$)') 
        # ax2.set_yticks([-100, -80, -60, -40, -20, 0, 20, 40]) 
        ax2.set_xlabel('Position (km)') 




        submods = ['river', 'baroc', 'sedadv', 'nostress', 'adv', 'stokes']
        colors = ['wheat', 'g', 'b', 'm', 'r', 'c']
        # submods = ['river', 'baroc']
        # colors = ['wheat', 'g']


        plt.figure(figsize=(10, 6.18))
        for j in range(len(submods)):
            mod = submods[j]
            for i in range(self.nch):
                f = self.f[str(i)]
                dc = self.input.v('network', str(i))
                T = self.transport.v(str(i), 'T', mod)        
                jmax = dc.v('grid', 'maxIndex', 'x')
                # c00 = np.real(dc.v('hatc0', 'a', range(jmax+1), -1, 0))
                x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
                if i == 0:
                    p = plt.plot(x, T, color=colors[j], ls = '--')
                else:
                    p = plt.plot(x, T, color=colors[j])
        for i in range(self.nch):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            T = self.T[str(i)]
            if i == 0:
                plt.plot(x, T, color='k', ls = '--')
            else:
                plt.plot(x, T, color='k')

        plt.ylabel('T (kg m$^{-1}$ s$^{-1}$)')
        plt.xlabel('Position (km)')

        submods = ['total', 'river', 'baroc', 'sedadv', 'vel.-depth asym.', 'advection', 'Stokes']
        colors = ['k', 'wheat', 'g', 'b', 'm', 'r', 'c']
        lines = [Line2D([0], [0], color=c, linewidth=2) for c in colors]
        plt.legend(lines, submods)

    def plot_contour(self):
        plt.rc('image', cmap='YlOrBr')
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.rcParams['figure.subplot.bottom'] = 0.12
        plt.rcParams['figure.subplot.wspace'] = 0.25
        plt.rcParams['figure.subplot.top'] = 0.97
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.left'] = 0.06


        self.nch = 3
        fig, axs = plt.subplots(3,1, figsize=(18, 6))

        SSC = {}
        X = {}
        Y = {}

        # label = ['NP', 'SP', 'SC']
        channels = [0, 1, 2]
        for i in channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            z = np.linspace(-self.H[str(i)][0], 0, kmax+1)
            X[str(i)], Y[str(i)] = np.meshgrid(x, z)

            f = self.f[str(i)].reshape(jmax+1, 1)
            c = np.real(dc.v('hatc0', 'a', range(jmax+1), range(kmax+1), 0) * f)
            SSC[str(i)] = np.flipud(c.transpose())

        SSC_values = SSC.values()
        SSCmax = np.max(SSC_values)
        SSCmin = np.min(SSC_values)
        for i in channels:

            ax = plt.subplot(1, 3, i+1)
            p = ax.pcolor(X[str(i)], Y[str(i)], SSC[str(i)], vmin=SSCmin, vmax=SSCmax)

            # xpos = lo[i], lo[i]+dc.v('L')
            ax.text(0.35, 0.9, 'channel '+ str(i+1), transform=ax.transAxes, fontsize=20)
            ax.set_ylabel('Depth (m)')
            ax.set_xlabel('Position (km)')
    
        cbar = plt.colorbar(p)
        cbar.set_label('SSC (kg m$^{-3}$)')
        # plt.savefig('SSC_ref.png', facecolor='w', edgecolor='w')



    def plot_contour_YE(self):
        plt.rc('image', cmap='YlOrBr')
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.rcParams['figure.subplot.bottom'] = 0.12
        plt.rcParams['figure.subplot.wspace'] = 0.4
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.right'] = 0.97
        plt.rcParams['figure.subplot.left'] = 0.06


        self.nch = 3
        fig, axs = plt.subplots(3,1, figsize=(18, 6))

        SSC = {}
        X = {}
        Y = {}
        for i in range(3):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) / 1000
            z = np.linspace(-self.H[str(i)][0], 0, kmax+1)
            X[str(i)], Y[str(i)] = np.meshgrid(x, z)

            f = self.f[str(i)].reshape(jmax+1, 1)
            c = np.real(dc.v('hatc0', 'a', range(jmax+1), range(kmax+1), 0) * f)
            SSC[str(i)] = np.flipud(c.transpose())

        # SSC_values = SSC.values()
        # SSCmax = np.max(SSC_values)
        # SSCmin = np.min(SSC_values)
        for i in range(3):

            ax = plt.subplot(1, 3, i+1)
            p = ax.pcolor(X[str(i)], Y[str(i)], SSC[str(i)])
            cbar = plt.colorbar(p)
            cbar.set_label('SSC (kg m$^{-3}$)')

            # xpos = lo[i], lo[i]+dc.v('L')
            ax.text(0.48, 0.9, label[i], transform=ax.transAxes, fontsize=20)
            ax.set_ylabel('Depth (m)')
            ax.set_xlabel('Position (km)')
    
        # cbar = plt.colorbar(p)
        # cbar.set_label('SSC (kg m$^{-3}$)')

    

    def store_output(self):

        return 

"""
T
adv 0 4
stokes 0 4 return drift
nostress 0 4 
dp 4 0
baroc 0
river 0 2
tide 2 4
no flux 2
sedadv 2
diffusion tide 0
diffusion river 0
river river 0

F
diffusion tide 0
diffusion river 0
sedadv 2
"""