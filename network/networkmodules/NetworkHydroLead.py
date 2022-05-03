"""
Date: 03-June-2019
Authors: J.Wang and Y.M. Dijkstra
"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy
import nifty as ny
import numpy as np
from scipy.integrate import cumtrapz
from nifty.functionTemplates.NumericalFunctionWrapper import NumericalFunctionWrapper



class NetworkHydroLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        self.channelnumber = 0
        self.iteration = 0

        return

    def stopping_criterion(self, iteration):
        stop = False
        self.channelnumber += 1
        # self.channelnumber = self.channelnumber % self.input.v('networksettings', 'numberofchannels')

        numberofchannels = self.input.v('networksettings', 'numberofchannels')
        # stop if iteration number exceeds number of prescribed variables
        if self.channelnumber >= numberofchannels:
            stop = True

            # store output of last iteration
            d = self.storeOutput()
            self.input.merge(d)
            C = self.matchchannels() # Calculate coef. for channel matching
            # C_river = self.matchchannels_river() 
            #### Save the scaled/matched solutions
            for i in range(numberofchannels):
                dout = self.scaleChannels(i, C[i,:])
                # dout_river = self.scaleChannels_river(i, C_river[i,:])
                self.input.merge(dout)
                # self.input.merge(dout_river)

            # Run the following if using turbulence WJY: Cd is not None
            # if self.input.v('Cd') is not None:

            # if self.Iterno == 0:
            #     error = 1
            # else: 
            #     error = np.max(self.Av_old - self.Av_new)
            # # Check convergence of friction parameters
            # if  error < 1e-4 and self.Iterno > 0:
            #     stop = True
            # else:
            #     self.channelnumber = 1
            #     self.Iterno +=1
            #     self.logger.info('error: '+str(error) + '\nStart iteration' + str(self.Iterno+1))
            #     stop = False     
            #     self.Av_old = self.Av_new
            #     self.Av_new = np.array([])
            

        return stop

    def run_init(self):
        """
        """           
        self.logger.info('Running network Hydro Lead')

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
        self.channelnumber = 0
            
        d = self.prepareInput()

        # M2 coefficients
        self.zeta0_L = []
        self.zeta0_0 = []
        self.Q0_0 = []
        self.Q0_L = []

        # M0 coefficients

        self.Riv0 = []
        self.RivL = []

        # self.Av_old = np.array([])
        # self.Av_new = np.array([])
        return d

    def run(self):
        # self.logger.info('Running Network Hydro Lead channel ' +str(self.channelnumber+1))
        
        dout = self.storeOutput()

        d = self.prepareInput()
        d.update(dout)
        return d

    def prepareInput(self):
        """ load input variable from networksettings for geometry2DV """
        d = {}
        dc = self.input.v('network', str(self.channelnumber)) 
        
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')
        d['A0'] = [0, 1., 0]
        d['Q0'] = 1.
        d['phase0'] = [0, 0, 0]
        d['u0'] = dc.v('u0', range(jmax+1), range(kmax+1), range(fmax+1))
        d['grid'] = dc.data['grid']


        d.update(dc.data)
        # loop over turbulence d['Av'] d['Roughness']

        return d

    def storeOutput(self):
        d = {}
        dc = self.input.v('network', str(self.channelnumber-1))
        
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        # self.Av_new = np.append(self.Av_new, self.input.v('Av'))

        u0 = self.input.v('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
        w0 = self.input.v('w0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
        zeta0 = self.input.v('zeta0', 'tide', range(jmax+1), [0], range(fmax+1))
        u0_2 = self.input.v('u0', 'tide_2', range(jmax+1), range(kmax+1), range(fmax+1))
        w0_2 = self.input.v('w0', 'tide_2', range(jmax+1), range(kmax+1), range(fmax+1))
        zeta0_2 = self.input.v('zeta0', 'tide_2', range(jmax+1), [0], range(fmax+1))
        

        # if 'river' in self.input.getKeysOf('zeta0'):
        #     riverZeta = (self.input.v('zeta0', 'river', range(jmax+1), 0, 0))
        #     self.Riv0.append(riverZeta[0])
        #     self.RivL.append(riverZeta[-1])
        
        # store transport and elevation for 2 modes for each channel
        # prepare the coefficients (zeta0_0, zeta0_L, Q0_0, Q0_L) for matching channels
        # mode 1: zeta0, u0;    mode 2: zeta0_2, u0_2
        # _0: at x = 0; _L: at x = L
        
        self.zeta0_0.append([zeta0[:,:,1][0][0], zeta0_2[:,:,1][0][0]])
        self.zeta0_L.append([zeta0[:,:,1][-1][0], zeta0_2[:,:,1][-1][0]])

        # depth-integrated horizontal velocity of mode 
        u0_DA = ny.integrate(u0, 'z', kmax, 0, dc, [0], range(kmax+1), [1])[0,0,0]
        uL_DA = 0 

        # depth-integrated horizontal velocity of mode 2
        uL_2_DA = ny.integrate(u0_2, 'z', kmax, 0, dc, [jmax], range(kmax+1), [1])[0,0,0]
        u0_2_DA = 0

        B_0 = dc.v('B', 0)
        B_L = dc.v('B',jmax)

        # transport
        self.Q0_0.append([u0_DA*B_0, u0_2_DA*B_0])
        self.Q0_L.append([uL_DA*B_L, uL_2_DA*B_L])

        # store to network
        dictU = {
            'zeta0': {
                'tide': deepcopy(zeta0),
                'tide_2': deepcopy(zeta0_2)
            },
            'u0': {
                'tide': deepcopy(u0),
                'tide_2': deepcopy(u0_2)
            },
            'w0': {
                'tide': deepcopy(w0),
                'tide_2': deepcopy(w0_2)
            }
        }

        d['network'] = {} 
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].merge(dictU)
        
        return d




    def scaleChannels(self, i, C):
        # self.logger.info('Scaling channel ' + str(i+1))

        dc = self.input.v('network', str(i))
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')
        x = np.linspace(0, 1, jmax+1) 

        zeta00 = dc.v('zeta0', 'tide', range(jmax+1), 0, 1)
        zeta01 = dc.v('zeta0', 'tide_2', range(jmax+1), 0, 1)
        zeta00x = dc.d('zeta0', 'tide', x=x, dim='x')[:,:,1]
        zeta01x = dc.d('zeta0', 'tide_2', x=x, dim='x')[:,:,1]
        zeta00xx = dc.d('zeta0', 'tide', x=x, dim='xx')[:,:,1]
        zeta01xx = dc.d('zeta0', 'tide_2', x=x, dim='xx')[:,:,1]
        zeta   = zeta00 * C[0] + zeta01 * C[1]
        zetax  = zeta00x * C[0] + zeta01x * C[1]
        zetaxx = zeta00xx * C[0] + zeta01xx * C[1]
        nfzeta = NumericalFunctionWrapper(zeta, dc.slice('grid'))
        nfzeta.addDerivative(zetax, 'x')
        nfzeta.addDerivative(zetaxx, 'xx')

        u00 = dc.v('u0', 'tide', range(jmax+1), range(kmax+1), 1)
        u01 = dc.v('u0', 'tide_2', range(jmax+1), range(kmax+1), 1)   
        u00x = dc.d('u0', 'tide', x=x, dim='x')[:,:,1]
        u01x = dc.d('u0', 'tide_2', x=x, dim='x')[:,:,1]
        u00z = dc.d('u0', 'tide', x=x, dim='z')[:,:,1]
        u01z = dc.d('u0', 'tide_2', x=x, dim='z')[:,:,1]
        u00zz = dc.d('u0', 'tide', x=x, dim='zz')[:,:,1]
        u01zz = dc.d('u0', 'tide_2', x=x, dim='zz')[:,:,1]
        u00xx = dc.d('u0', 'tide', x=x, dim='xx')[:,:,1]
        u01xx = dc.d('u0', 'tide_2', x=x, dim='xx')[:,:,1]
        u   = u00 * C[0] + u01 * C[1]
        ux  = u00x * C[0] + u01x * C[1]
        uxx = u00xx * C[0] + u01xx * C[1]
        uz  = u00z * C[0] + u01z * C[1]
        uzz = u00zz * C[0] + u01zz * C[1]
        nfu = NumericalFunctionWrapper(u, self.input.slice('grid'))
        nfu.addDerivative(ux, 'x')
        nfu.addDerivative(uxx, 'xx')
        nfu.addDerivative(uz, 'z')
        nfu.addDerivative(uzz, 'zz')

        w00 = dc.v('w0', 'tide', range(jmax+1), range(kmax+1), 1)
        w01 = dc.v('w0', 'tide_2', range(jmax+1), range(kmax+1), 1) 
        w00z = dc.d('w0', 'tide', x=x, dim='z')[:,:,1]
        w01z = dc.d('w0', 'tide_2', x=x, dim='z')[:,:,1]
        w   = w00 * C[0] + w01 * C[1]
        wz  = w00z * C[0] + w01z * C[1]
        nfw = NumericalFunctionWrapper(w, self.input.slice('grid'))
        nfw.addDerivative(wz, 'z')
        

        dictU = {}
        dictU = {
            'zeta0': {
                'tide': nfzeta.function,
                'tide_2': zeta00 * 0
            },
            'u0': {
                'tide': nfu.function,
                'tide_2': u00 * 0
            },
            'w0': {
                'tide': nfw.function,
                'tide_2': w00 * 0
            }
        }
        d = {}
        d['network'] = {} 
        d['network'][str(i)] = DataContainer()
        d['network'][str(i)].merge(dictU)

        return d

    def scaleChannels_river(self, i, C):
        dc = self.input.v('network', str(i))
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        zeta0 = dc.v('zeta0', 'river', range(jmax+1), [0], range(fmax+1))
        u0 = dc.v('zeta0', 'river', range(jmax+1), range(kmax+1), range(fmax+1))

        dictU = {
            'zeta0': {
                'river': deepcopy(zeta0 * C[0] + C[1])
            },
            'u0': {
                'river': deepcopy(u0 * C[0] + C[1])
            }
        }

        d = {}
        d['network'] = {} 
        d['network'][str(i)] = DataContainer()
        d['network'][str(i)].merge(dictU)

        return d

    def matchchannels(self):
        self.logger.info('Matching channels Hydro Lead')
        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        Sea = self.input.v('networksettings', 'label', 'Sea')   # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')    # labels of river channels
        tide = self.input.v('networksettings', 'forcing', 'M2')
        discharge = self.input.v('networksettings', 'forcing', 'discharge')
        nch = self.input.v('networksettings', 'numberofchannels')  # number of channels

        # assert len(tide) == len(Sea), 'Number of sea channels and tidal forcings does not equal'
        # assert len(River) == len(discharge), 'Number of river channels and discharge forcings does not equal'

        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex) 
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 
 
        M = np.zeros((2*nch, 2*nch), dtype=complex)
        v = np.zeros(2*nch, dtype=complex)
        v[0:len(Sea)] = tide
        row = 0

        # zeta0_L[# of Channel][# of Mode]
        # prescribe tidal surface elevation at the sea
        for i in Sea:
            M[row, [2*i, 2*i+1]] = np.array([self.zeta0_0[i][0], self.zeta0_0[i][1]])
            row += 1

        # u=0 at tidal limit/weir <=> Q0 = 0 
        # !!!!! no River discharge
        for i in River:
            M[row, [2*i, 2*i+1]] = np.array([self.Q0_L[i][0], self.Q0_L[i][1]])
            row += 1

        # continuous SSH
        for j in range(len(Vertex)):
            loc = vertex_nosign[j]  # indices of channels that are connected at branching point
            # loc[i]: index for channel
            for i in range(len(loc)-1):
                if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                    M[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta0_L[loc[i]][0], self.zeta0_L[loc[i]][1]]
                elif sign[j][i] == -1:
                    M[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta0_0[loc[i]][0], self.zeta0_0[loc[i]][1]]
                if sign[j][i-1] == 1:
                    M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta0_L[loc[i-1]][0], -self.zeta0_L[loc[i-1]][1]]                
                elif sign[j][i-1] == -1:
                    M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta0_0[loc[i-1]][0], -self.zeta0_0[loc[i-1]][1]]
                row += 1

        # mass conservation
        for j in range(len(Vertex)):
            for i in range(len(vertex_nosign[j])):
                k = vertex_nosign[j][i] # channel index
                if sign[j][i] == 1:
                    M[row, [2*k, 2*k+1]] = [self.Q0_L[k][0], self.Q0_L[k][1]]
                elif sign[j][i] == -1:
                    M[row, [2*k, 2*k+1]] = [-self.Q0_0[k][0], -self.Q0_0[k][1]]
            row += 1
        C = np.reshape(np.linalg.solve(M, v), (nch, 2))

        return C


    # def matchchannels_river(self):
    #     Vertex = self.input.v('networksettings', 'label', 'Vertex')
    #     Sea = self.input.v('networksettings', 'label', 'Sea')   # labels of sea channels
    #     River = self.input.v('networksettings', 'label', 'River')    # labels of river channels
    #     tide = self.input.v('networksettings', 'forcing', 'M2')
    #     discharge = self.input.v('networksettings', 'forcing', 'discharge')
    #     nch = self.input.v('networksettings', 'numberofchannels')  # number of channels
    #     vertex_nosign = deepcopy(Vertex) 
    #     sign = deepcopy(Vertex) 

    #     Mr = np.zeros((2 * nch, 2 * nch))
    #     vr = np.zeros(2 * nch)
    #     vr[0:len(River)] = discharge    # prescribed discharge
    #     row = 0

    #     for i in River:
    #         Mr[row,[2*i]] = 1
    #         row += 1
            
    #     for i in Sea:
    #         Mr[row,[[2*i, 2*i+1]]] = [np.real(self.Riv0[i]), 1]
    #         row += 1

    #     # continuous SSH
    #     for j in range(len(Vertex)):
    #         loc = vertex_nosign[j]  # indices of channels that are connected at branching point
    #         # loc[i]: index for channel
    #         for i in range(len(loc)-1):
    #             if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
    #                 Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.RivL[loc[i]]), 1]
    #             elif sign[j][i] == -1:
    #                 Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.Riv0[loc[i]]), 1]
    #             if sign[j][i-1] == 1:
    #                 Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.RivL[loc[i-1]]), -1]                
    #             elif sign[j][i-1] == -1:
    #                 Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.Riv0[loc[i-1]]), -1]
    #             row += 1
                
    #     # mass cons.
    #     for j in range(len(Vertex)):
    #         for i in range(len(vertex_nosign[j])):
    #             k = vertex_nosign[j][i]
    #             if sign[j][i] == 1:
    #                 Mr[row, 2*k] = 1
    #             elif sign[j][i] == -1:
    #                 Mr[row, 2*k] = -1
    #         row += 1
    #     # C = np.reshape(np.linalg.solve(Mr, vr), (nch, 2))
    #     return C
        
