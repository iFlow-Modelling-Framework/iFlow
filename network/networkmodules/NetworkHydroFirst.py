"""
Network module for matching the first order water motion.

Date: 13-Feb-2020
Authors: J. Wang
"""
import copy
import logging
import numpy as np
from src.DataContainer import DataContainer
from src.util.diagnostics import KnownError
from nifty import toList
import numbers
import os
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import nifty as ny
import scipy.linalg

class NetworkHydroFirst():
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        self.channelnumber = 0
        self.iteration = 0
        self.OMEGA = self.input.v('OMEGA')
        self.G = self.input.v('G')
        self.riverZeta = {}
        self.riverU = {}

        self.run_initKey = False    # this is always false

        ## salinityNetworkKey == True and self.run_salinity = False => does not compute salinity
        ## salinityNetworkKey == False and self.run_salinity = True =>  compute salinity
        self.run_salinity = False
        # self.run_salinity = True
        self.salinityNetworkKey = 1 -  self.run_salinity
        
        

        return

    def stopping_criterion(self, iteration):
        # iteration over channels. For each channel, calculated unscaled solutions
        stop = False
        self.channelnumber += 1
        numberofchannels = self.input.v('networksettings', 'numberofchannels')

        if self.salinityNetworkKey == False:
            if self.channelnumber >= numberofchannels:
            
                # save the unscaled solutions of the last channel
                d = self.storeOutput()
                self.input.merge(d)

                # match channels, M0 and M4
                C = self._matchchannels()

                # scale channels   # loop over subtidal/overtide components
                if 'dp' in C:  # compute dp iff adv is computed
                    mod = 'dp'
                    for i in range(numberofchannels):
                        dout = self.scaleChannels(mod, i, C[mod][i,:])
                        self.input.merge(dout)
                        
                for mod in self.input.getKeysOf('zeta1'):
                    # scale tide and remove tide_2 as the last step
                    if mod not in ['tide', 'tide_2']:
                        # loop over channels
                        for i in range(numberofchannels):
                            dout = self.scaleChannels(mod, i, C[mod][i,:])
                            self.input.merge(dout)

                # scale M4 tide and set tide_2 to 0
                for i in range(numberofchannels):
                    if 'tide' in C:
                        dout = self.scaleChannels('tide', i, C['tide'][i,:])
                    self.input.merge(dout)

                # compute salinity again using river flow, compute baroc again, and merge into iFlow
                
                # Qtot = sum(C[key][:,0] for key in C.keys()) - C['baroc'][:, 0]

                self.networkSalinity(C['river'][:, 0])
                # self.networkSalinity(Qtot)

                self.salinityNetworkKey = True
                self.channelnumber = 0
                # self.run_init() # after this, return false to iFlow and run(). Prepared input is not returned to iFlow
                # self.channelnumber = 1

                
        elif self.salinityNetworkKey == True:
         # stop if iteration number exceeds number of prescribed variables
            if self.channelnumber >= numberofchannels:
                stop = True

                # save the unscaled solutions of the last channel
                d = self.storeOutput()
                self.input.merge(d)

                # match channels, M0 and M4
                C = self._matchchannels()

                # scale channels   # loop over subtidal/overtide components
                if 'dp' in C:  # compute dp iff adv is computed
                    mod = 'dp'
                    for i in range(numberofchannels):
                        dout = self.scaleChannels(mod, i, C[mod][i,:])
                        self.input.merge(dout)
                        
                for mod in self.input.getKeysOf('zeta1'):
                    # scale tide and remove tide_2 as the last step
                    if mod not in ['tide', 'tide_2']:
                        # loop over channels
                        for i in range(numberofchannels):
                            dout = self.scaleChannels(mod, i, C[mod][i,:])
                            self.input.merge(dout)

                # scale M4 tide and set tide_2 to 0
                for i in range(numberofchannels):
                    if 'tide' in C:
                        dout = self.scaleChannels('tide', i, C['tide'][i,:])
                    self.input.merge(dout)


                # Store transports into iFlow
                dictQ = {}
                dictQ['Q'] = {}
                for i in range(numberofchannels):
                    dictQ['Q'][str(i)] = DataContainer()

                    tmp = {}
                    for key in C.keys():
                        tmp[key] = np.real(C[key][i][0])
                    dictQ['Q'][str(i)].addData('Q', tmp)
                
                self.input.merge(dictQ)

        return stop

    def run_init(self):
        """
        """
        self.logger.info('Running Network Hydro First' )

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0


        d = self.prepareInput()

        # M0 coefficients
        self.coefQ_0 = []
        self.coefQ_L = []
        self.coef0 = {}
        self.coefL = {}
        # M4 coefficients
        self.zeta1_L = {}
        self.zeta1_0 = {}
        self.Q1_0 = {}
        self.Q1_L = {}

        # leading order M2 current for dynamic pressure
        self.u0da_0 = []
        self.u0da_L = []

        return d

    def run(self):
        # self.logger.info('Running Network Hydro First channel ' + str(self.channelnumber+1))
        
        if self.run_initKey == False and self.salinityNetworkKey == True and self.run_salinity == True:
            self.run_initKey = True
            return self.run_init()

        dout = self.storeOutput()
        d = self.prepareInput()
        d.update(dout)
        return d

    def prepareInput(self):
        
        d = {}
        dc = self.input.v('network', str(self.channelnumber))

        d['A1'] = [0, 0, 1.]
        d['Q1'] = 1.
        d['phase1'] = [0, 0, 0]
        d['grid'] = dc.data['grid']

        d['Q'] = {}

        d.update(dc.data)

        return d
    
    def storeOutput(self):
        d = {}
        dc = self.input.v('network', str(self.channelnumber-1))
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        # u1river = self.input.v('zeta1', 'river', range(jmax+1), range(kmax+1), range(fmax+1))

        riverZeta = self.input.v('zeta1', 'river', range(jmax+1), 0, 0)

        self.riverZeta[str(self.channelnumber-1)] = deepcopy(self.input.v('zeta1', 'river', range(jmax+1), [0], range(fmax+1)))
        
        self.riverU[str(self.channelnumber-1)] = deepcopy(self.input.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1)))

        self.coefQ_0.append(riverZeta[0])
        self.coefQ_L.append(riverZeta[-1])

        u1 =   self.input.v('u1', 'tide',   range(jmax+1), range(kmax+1), 2)
        u1_2 = self.input.v('u1', 'tide_2', range(jmax+1), range(kmax+1), 2)

        zeta1 =   self.input.v('zeta1', 'tide',   range(jmax+1), 0, 2)
        zeta1_2 = self.input.v('zeta1', 'tide_2', range(jmax+1), 0, 2)

        dictU = {}
        dictU['zeta1'] = {}
        dictU['u1'] = {}
        dictU['Q'] = {}

        # Leading order M2 cuurent will be only used by 
        u0 = self.input.v('u0', 'tide',   range(jmax+1), range(kmax+1), 1)
        u0_DA = ny.integrate(u0, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]
        uL_DA = ny.integrate(u0, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]
        self.u0da_0.append(u0_DA / dc.v('H', 0))
        self.u0da_L.append(uL_DA / dc.v('H', jmax))

        # store zeta1 at 0 and L without transport
        # loop over every contirbution, excluding dynamic pressure
        for mod in self.input.getKeysOf('zeta1'):
            if mod not in self.coef0:
                self.coef0[mod] = []
                self.Q1_0[mod] = []
                self.zeta1_0[mod] = []
            # if mod not in self.coefL:
                self.coefL[mod] = []
                self.Q1_L[mod] = []
                self.zeta1_L[mod] = []

            if mod == 'tide':
                self.zeta1_0[mod].append([zeta1[0], zeta1_2[0]])
                self.zeta1_L[mod].append([zeta1[-1], zeta1_2[-1]]) 

                ### depth-integrated horizontal velocity of mode 
                # u0_DA: u at 0
                u0_DA = ny.integrate(u1, 'z', kmax, 0, dc, [0], range(kmax+1), [2])
                uL_DA = 0 

                ### depth-integrated horizontal velocity of mode 2
                uL_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [jmax], range(kmax+1), [2])
                u0_2_DA = 0

                B_0 = dc.v('B', 0)
                B_L = dc.v('B',jmax)   
                ### transport
                self.Q1_0[mod].append([u0_DA*B_0, u0_2_DA*B_0])
                self.Q1_L[mod].append([uL_DA*B_L, uL_2_DA*B_L])

                dictU['zeta1'][mod] = deepcopy(self.input.v('zeta1', mod, range(jmax+1), [0], range(fmax+1)))
                dictU['zeta1']['tide_2'] = deepcopy(self.input.v('zeta1', 'tide_2', range(jmax+1), [0], range(fmax+1)))
                dictU['u1'][mod] = deepcopy(self.input.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1)))
                dictU['u1']['tide_2'] = deepcopy(self.input.v('u1', 'tide_2', range(jmax+1), range(kmax+1), range(fmax+1)))
                

            else: # adv, nostress, Stokes, baroc
                # first store for M0
                dictU['zeta1'][mod] = deepcopy(self.input.v('zeta1', mod, range(jmax+1), [0], range(fmax+1)))
                dictU['u1'][mod] = deepcopy(self.input.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1)))

                self.coef0[mod].append((self.input.v('zeta1', mod, 0, 0, 0)))
                self.coefL[mod].append((self.input.v('zeta1', mod, -1, 0, 0)))

                if mod in ['adv', 'stokes', 'nostress']: # add M4
                    u1_s =   self.input.v('u1', mod,  range(jmax+1), range(kmax+1), 2)
                    zeta1_s = self.input.v('zeta1', mod, range(jmax+1), 0, 2)

                    self.zeta1_0[mod].append([zeta1[0], zeta1_2[0], zeta1_s[0]])
                    self.zeta1_L[mod].append([zeta1[-1], zeta1_2[-1], zeta1_s[-1]]) 

                    ### depth-integrated horizontal velocity of mode 
                    u0_DA = ny.integrate(u1, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]
                    # uL_DA = 0 
                    uL_DA = ny.integrate(u1, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]

                    ### depth-integrated horizontal velocity of mode 2
                    uL_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]
                    # u0_2_DA = 0
                    u0_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]

                    u0sda = ny.integrate(u1_s, 'z', kmax, 0, dc, [0],    range(kmax+1), 2)[0][0]
                    uLsda = ny.integrate(u1_s, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]

                    B_0 = dc.v('B', 0)
                    B_L = dc.v('B',jmax)  

                    ### transport
                    self.Q1_0[mod].append([u0_DA*B_0, u0_2_DA*B_0, u0sda*B_0])
                    self.Q1_L[mod].append([uL_DA*B_L, uL_2_DA*B_L, uLsda*B_L])

                    # Store for dynamic pressure iff adv is activated
                    if mod == 'adv':
                        
                        mod = 'dp'
                        dictU['zeta1'][mod] = deepcopy(self.input.v('zeta1', 'river', range(jmax+1), [0], range(fmax+1)))
                        dictU['u1'][mod] = deepcopy(self.input.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1)))
            

                
        d['network'] = {} 
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].merge(dictU)

        return d


    def scaleChannels(self, mod, i, C):
        # self.logger.info('Scaling ' + mod + ' channel ' + str(i+1))
        dc = self.input.v('network', str(i))
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        d={}
        dictU = {}

        if mod == 'river':
            # uriver = dc.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1))
            dictU = {
                'zeta1': {mod: deepcopy(self.riverZeta[str(i)] * C[0] + C[1])},
                'u1':    {mod: deepcopy(self.riverU[str(i)] * C[0] )},
                'Q': {mod: np.real(C[0])}
            }
            d['network'] = {}
            d['network'][str(i)] = DataContainer()
            d['network'][str(i)].merge(dictU)

        elif mod == 'tide':
            zeta1 = dc.v('zeta1', 'tide', range(jmax+1), [0], range(fmax+1))
            zeta2 = dc.v('zeta1', 'tide_2', range(jmax+1), [0], range(fmax+1))
            u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
            u2 = dc.v('u1', 'tide_2', range(jmax+1), range(kmax+1), range(fmax+1))
            
            dictU = {
                'zeta1': {
                    mod: deepcopy(C[0]*zeta1 + C[1]*zeta2),
                    'tide_2': zeta1 * 0
                },
                'u1': {
                    mod: deepcopy(C[0]*u1 + C[1]*u2),
                    'tide_2': u1 * 0
                }
            }
            d['network'] = {}
            d['network'][str(i)] = DataContainer()
            d['network'][str(i)].merge(dictU)
        
        else: # adv, stokes, nostress, baroc
            if mod in ['adv', 'stokes', 'nostress']:
                zeta1 = dc.v('zeta1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
                zeta2 = dc.v('zeta1', 'tide_2', range(jmax+1), range(kmax+1), 2)
                zetas = dc.v('zeta1', mod, range(jmax+1), [0], range(fmax+1))

                u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
                u2 = dc.v('u1', 'tide_2', range(jmax+1), range(kmax+1), 2)
                us = dc.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1))

                # M0 component
                zeta1[:,:,0] = (zetas[:,:,0]  + self.riverZeta[str(i)][:,:,0] * C[0] + C[1])
                u1[:,:,0] = us[:,:,0]  + self.riverU[str(i)][:,:,0] * C[0]

                # M4 component
                zeta1[:,:,2] = C[2] * zeta1[:,:,2] + C[3] * zeta2 + zetas[:,:,2]
                u1[:,:,2]    = C[2] * u1[:,:,2]    + C[3] * u2    + us[:,:,2]
         
                dictU = {
                    'zeta1': {mod: deepcopy(zeta1)},
                    'u1': {mod: deepcopy(u1)},
                    'Q': {mod: np.real(C[0])}
                }

                

            elif mod == 'dp':
                zeta1 = dc.v('zeta1', 'tide', range(jmax+1), [0], range(fmax+1))
                zeta2 = dc.v('zeta1', 'tide_2', range(jmax+1), [0], 2)

                u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
                u2 = dc.v('u1', 'tide_2', range(jmax+1), range(kmax+1), 2)

                zeta1[:,:,0] = self.riverZeta[str(i)][:,:,0] * C[0] + C[1]
                zeta1[:,:,2] = C[2] * zeta1[:,:,2] + C[3] * zeta2
                u1[:,:,0] = self.riverU[str(i)][:,:,0] * C[0]
                u1[:,:,2] = C[2] * u1[:,:,2] + C[3] * u2

                dictU = {
                    'zeta1': {mod: deepcopy(zeta1)},
                    'u1': {mod: deepcopy(u1)},
                    'Q': {mod: np.real(C[0])}
                }

            elif mod == 'baroc': # baroc
                zeta1 = dc.v('zeta1', mod, range(jmax+1), range(kmax+1), range(fmax+1))
                u1 = dc.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1))
                dictU = {
                    'zeta1': {mod: deepcopy(zeta1  + self.riverZeta[str(i)] * C[0] + C[1])},
                    'u1':    {mod: deepcopy(u1 + self.riverU[str(i)] * C[0])},
                    'Q': {mod: np.real(C[0])}
                }

                
            d['network'] = {}
            d['network'][str(i)] = DataContainer()
            d['network'][str(i)].merge(dictU)           

        return d

    def _matchchannels(self):
        """
        Return the channel matching coefficients.
        For M0 components, the final solution: 
            zeta = zeta1 + C[0] * zeta_river + C[1]
            u = u1 + * C[0] * u_river
        where    
        zeta1, u1: the solution for single channel (is 0 for river and dynamic pressure)
        C[0]: subtidal transport
        C[1]: a constant s.t. water level is continuous

        If the component contributes to M4, then,
            zeta = C[2] * zeta1[:,:,2] + C[3] * zeta2 + zetas[:,:,2]
            u = C[2] * u1[:,:,2] + C[3] * u2 + us[:,:,2]
        where
        zetas, us: the solution for single channel (is 0 for river and dynamic pressure)
        zeta1: first order tide for single channel
        zeta2: as zeta1, but solved with reversed boundary conditions
        """

        self.logger.info('Matching channels hydro first')
        C = {}
        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        Sea = self.input.v('networksettings', 'label', 'Sea')   # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')    # labels of river channels
        discharge = self.input.v('networksettings', 'forcing', 'discharge')
        nch = self.input.v('networksettings', 'numberofchannels')  # number of channels
        tide = self.input.v('networksettings', 'forcing', 'M4')

        # if the external M4 forcing is not given:
        if tide is None:
            tide = 0


        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex)         
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 

        for mod in self.input.getKeysOf('zeta1'):
            # self.logger.info('Matching ' +  mod)

            if mod == 'river':
                Mr = np.zeros((2 * nch, 2 * nch))
                vr = np.zeros(2 * nch)
                vr[0:len(River)] = discharge    # prescribed discharge
                row = 0

                for i in River:
                    Mr[row,[2*i]] = 1
                    row += 1
                    
                for i in Sea:
                    Mr[row,[[2*i, 2*i+1]]] = [np.real(self.coefQ_0[i]), 1]
                    row += 1

                # continuous SSH
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.coefQ_L[loc[i]]), 1]
                        elif sign[j][i] == -1:
                            Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.coefQ_0[loc[i]]), 1]
                        if sign[j][i-1] == 1:
                            Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.coefQ_L[loc[i-1]]), -1]                
                        elif sign[j][i-1] == -1:
                            Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.coefQ_0[loc[i-1]]), -1]
                        row += 1
                # mass cons.
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i]
                        if sign[j][i] == 1:
                            Mr[row, 2*k] = 1
                        elif sign[j][i] == -1:
                            Mr[row, 2*k] = -1
                    row += 1
                C['river'] = np.reshape(np.linalg.solve(Mr, vr), (nch, 2))
            
            elif mod == 'tide':
                M = np.zeros((2*nch, 2*nch), dtype=complex)
                v = np.zeros(2*nch, dtype=complex)
                v[0:len(Sea)] = tide
                row = 0

                # zeta0_L[# of Channel][# of Mode]
                # prescribe tidal surface elevation at the sea
                for i in Sea:
                    M[row, [2*i, 2*i+1]] = np.array([self.zeta1_0[mod][i][0], self.zeta1_0[mod][i][1]])
                    row += 1

                # u=0 at tidal limit/weir <=> Q0 = 0 
                for i in River:
                    M[row, [2*i, 2*i+1]] = np.array([self.Q1_L[mod][i][0], self.Q1_L[mod][i][1]])
                    row += 1

                # continuous SSH
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            M[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_L[mod][loc[i]][0], self.zeta1_L[mod][loc[i]][1]]
                        elif sign[j][i] == -1:
                            M[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_0[mod][loc[i]][0], self.zeta1_0[mod][loc[i]][1]]
                        if sign[j][i-1] == 1:
                            M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_L[mod][loc[i-1]][0], -self.zeta1_L[mod][loc[i-1]][1]]                
                        elif sign[j][i-1] == -1:
                            M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_0[mod][loc[i-1]][0], -self.zeta1_0[mod][loc[i-1]][1]]
                        row += 1

                # mass conservation
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i] # channel index
                        if sign[j][i] == 1:
                            M[row, [2*k, 2*k+1]] = [self.Q1_L[mod][k][0], self.Q1_L[mod][k][1]]
                        elif sign[j][i] == -1:
                            M[row, [2*k, 2*k+1]] = [-self.Q1_0[mod][k][0], -self.Q1_0[mod][k][1]]
                    row += 1
                C[mod] = np.reshape(np.linalg.solve(M, v), (nch, 2))  
            
            elif mod is not 'dp':
                M = np.zeros((2 * nch, 2 * nch), dtype=complex)
                v = np.zeros(2 * nch, dtype=complex)
                row = 0

                # transport = 0 in river channels
                for i in River:
                    M[row,[2*i]] = 1
                    row += 1
                # 0 SSH at sea
                for i in Sea:
                    # M[row,[[2*i, 2*i+1]]] = [self.coefQ_L[i], 1]
                    M[row,[[2*i, 2*i+1]]] = [self.coefQ_0[i], 1]
                    # v[row] = -self.coefL[mod][row]
                    v[row] = -self.coef0[mod][row]
                    row += 1
                # continuous SSH
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            M[row, [2*loc[i], 2*loc[i]+1]] = [self.coefQ_L[loc[i]], 1]
                            v[row] -= self.coefL[mod][loc[i]]
                        elif sign[j][i] == -1:
                            M[row, [2*loc[i], 2*loc[i]+1]] = [self.coefQ_0[loc[i]], 1]
                            v[row] -= self.coef0[mod][loc[i]]
                        if sign[j][i-1] == 1:
                            M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.coefQ_L[loc[i-1]], -1]     
                            v[row] += self.coefL[mod][loc[i-1]]           
                        elif sign[j][i-1] == -1:
                            M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.coefQ_0[loc[i-1]], -1]
                            v[row] += self.coef0[mod][loc[i-1]]
                        row += 1
                # mass cons.
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i]
                        if sign[j][i] == 1:
                            M[row, 2*k] = 1
                        elif sign[j][i] == -1:
                            M[row, 2*k] = -1
                    row += 1

                C[mod] = np.reshape(np.linalg.solve(M, v), (nch, 2))

                if mod in ['adv', 'stokes', 'nostress']:
                    M4 = np.zeros((2*nch, 2*nch), dtype=complex)
                    v4 = np.zeros(2*nch, dtype=complex)
                    row = 0

                    # zeta0_L[mod][# of Channel][# of Mode]
                    # prescribe tidal surface elevation at the sea
                    for i in Sea:
                        M4[row, [2*i, 2*i+1]] = np.array([self.zeta1_0[mod][i][0], self.zeta1_0[mod][i][1]])
                        v4[row] -= self.zeta1_0[mod][i][2]
                        row += 1

                    # u=0 at tidal limit/weir  
                    for i in River:
                        M4[row, [2*i, 2*i+1]] = np.array([self.Q1_L[mod][i][0], self.Q1_L[mod][i][1]])
                        v4[row] -= self.Q1_L[mod][i][2]
                        row += 1

                    # continuous SSH
                    for j in range(len(Vertex)):
                        loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                        # loc[i]: index for channel
                        for i in range(len(loc)-1):
                            if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                                M4[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_L[mod][loc[i]][0], self.zeta1_L[mod][loc[i]][1]]
                                v4[row] -= self.zeta1_L[mod][loc[i]][2]
                            elif sign[j][i] == -1:
                                M4[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_0[mod][loc[i]][0], self.zeta1_0[mod][loc[i]][1]]
                                v4[row] -= self.zeta1_0[mod][loc[i]][2]
                            if sign[j][i-1] == 1:
                                M4[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_L[mod][loc[i-1]][0], -self.zeta1_L[mod][loc[i-1]][1]]
                                v4[row] -= -self.zeta1_L[mod][loc[i-1]][2]                
                            elif sign[j][i-1] == -1:
                                M4[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_0[mod][loc[i-1]][0], -self.zeta1_0[mod][loc[i-1]][1]]
                                v4[row] -= -self.zeta1_0[mod][loc[i-1]][2]
                            row += 1

                    # mass conservation
                    for j in range(len(Vertex)):
                        for i in range(len(vertex_nosign[j])):
                            k = vertex_nosign[j][i] # channel index
                            if sign[j][i] == 1:
                                M4[row, [2*k, 2*k+1]] = [self.Q1_L[mod][k][0], self.Q1_L[mod][k][1]]
                                v4[row] -= self.Q1_L[mod][k][2]
                            elif sign[j][i] == -1:
                                M4[row, [2*k, 2*k+1]] = [-self.Q1_0[mod][k][0], -self.Q1_0[mod][k][1]]
                                v4[row] -= -self.Q1_0[mod][k][2]
                        row += 1
                    C[mod] = np.append(C[mod], np.reshape(np.linalg.solve(M4, v4), (nch, 2)), axis=1) 
              
                # Solve for the contribution by cont. dp only if adv is solved
                if mod == 'adv':
                    u0 = np.real(0.25 * np.abs(self.u0da_0)**2)
                    uL = np.real(0.25 * np.abs(self.u0da_L)**2)
                    u0M4 = 0.25 * np.complex64(self.u0da_0)**2
                    uLM4 = 0.25 * np.complex64(self.u0da_L)**2

                    # first match M0
                    M0dp = np.zeros((2 * nch, 2 * nch))
                    v0dp = np.zeros(2 * nch)
                    row = 0
                    for i in River:
                        M0dp[row,[2*i]] = 1
                        row += 1
                        
                    for i in Sea:
                        M0dp[row,[[2*i, 2*i+1]]] = [np.real(self.coefQ_0[i]), 1]
                        row += 1

                    # continuous dynamic pressure
                    for j in range(len(Vertex)):
                        loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                        # loc[i]: index for channel
                        for i in range(len(loc)-1):
                            if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                                M0dp[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.coefQ_L[loc[i]]), 1]
                                v0dp[row] -= uL[loc[i]] / self.G
                            elif sign[j][i] == -1:
                                M0dp[row, [2*loc[i], 2*loc[i]+1]] = [np.real(self.coefQ_0[loc[i]]), 1]
                                v0dp[row] -= u0[loc[i]] / self.G
                            if sign[j][i-1] == 1:
                                M0dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.coefQ_L[loc[i-1]]), -1]   
                                v0dp[row] += uL[loc[i-1]] / self.G             
                            elif sign[j][i-1] == -1:
                                M0dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(self.coefQ_0[loc[i-1]]), -1]
                                v0dp[row] += u0[loc[i-1]] / self.G   
                            row += 1
                    # mass cons.
                    for j in range(len(Vertex)):
                        for i in range(len(vertex_nosign[j])):
                            k = vertex_nosign[j][i]
                            if sign[j][i] == 1:
                                M0dp[row, 2*k] = 1
                            elif sign[j][i] == -1:
                                M0dp[row, 2*k] = -1
                        row += 1
                    C['dp'] = np.reshape(np.linalg.solve(M0dp, v0dp), (nch, 2))


                    # match M4 by dynamic pressure
                    M4dp = np.zeros((2*nch, 2*nch), dtype=complex)
                    v4dp = np.zeros(2*nch, dtype=complex)
                    row = 0

                    # zeta0_L[# of Channel][# of Mode]
                    # 0 elevation at the sea 
                    for i in Sea:
                        M4dp[row, [2*i, 2*i+1]] = np.array([self.zeta1_0['tide'][i][0], self.zeta1_0['tide'][i][1]])
                        v4dp[row] = 0
                        row += 1

                    # u=0 at tidal limit/weir <=> Q0 = 0 
                    for i in River:
                        M4dp[row, [2*i, 2*i+1]] = np.array([self.Q1_L['tide'][i][0], self.Q1_L['tide'][i][1]])
                        row += 1

                    # continuous dynamic pressure
                    for j in range(len(Vertex)):
                        loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                        # loc[i]: index for channel
                        for i in range(len(loc)-1):
                            if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                                M4dp[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_L['tide'][loc[i]][0], self.zeta1_L['tide'][loc[i]][1]]
                                v4dp[row] -= uLM4[loc[i]] / self.G
                            elif sign[j][i] == -1:
                                M4dp[row, [2*loc[i], 2*loc[i]+1]] = [self.zeta1_0['tide'][loc[i]][0], self.zeta1_0['tide'][loc[i]][1]]
                                v4dp[row] -= u0M4[loc[i]] / self.G
                            if sign[j][i-1] == 1:
                                M4dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_L['tide'][loc[i-1]][0], -self.zeta1_L['tide'][loc[i-1]][1]]    
                                v4dp[row] += uLM4[loc[i-1]] / self.G            
                            elif sign[j][i-1] == -1:
                                M4dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.zeta1_0['tide'][loc[i-1]][0], -self.zeta1_0['tide'][loc[i-1]][1]]
                                v4dp[row] += u0M4[loc[i-1]] / self.G
                            row += 1

                    # mass conservation
                    for j in range(len(Vertex)):
                        for i in range(len(vertex_nosign[j])):
                            k = vertex_nosign[j][i] # channel index
                            if sign[j][i] == 1:
                                M4dp[row, [2*k, 2*k+1]] = [self.Q1_L['tide'][k][0], self.Q1_L['tide'][k][1]]
                            elif sign[j][i] == -1:
                                M4dp[row, [2*k, 2*k+1]] = [-self.Q1_0['tide'][k][0], -self.Q1_0['tide'][k][1]]
                        row += 1
                    CM4dp = np.reshape(np.linalg.solve(M4dp, v4dp), (nch, 2))
                    C['dp'] = np.append(C['dp'], CM4dp , axis=1) 

       
        return C


    def networkSalinity(self, Q):
        Vertex = self.input.v('networksettings', 'label', 'Vertex')
        nch = self.input.v('networksettings', 'numberofchannels')
        Sea = self.input.v('networksettings', 'label', 'Sea')       # labels of sea channels
        River = self.input.v('networksettings', 'label', 'River')   # labels of river channels
        ssea = self.input.v('networksettings', 'forcing', 'S0')[Sea]


        Kh = 250

        vertex_nosign = deepcopy(Vertex) 
        sign = deepcopy(Vertex)         
        for i in range(len(Vertex)):
            for x in range(len(Vertex[i])):
                vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
                sign[i][x] = np.sign(Vertex[i][x]) 


        dictM = {}
        Ns = []  # number of grid points of each channel in the network
        for channel in range(nch):
            dc = self.input.v('network', str(channel))
            x = dc.v('grid', 'axis', 'x') * dc.v('L') # dimensional axis
            Ns.append(len(x))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            B = dc.v('B', range(jmax+1), 0, 0)
            lc = -np.mean(B / np.gradient(B, x, edge_order=2))
            u = np.real(np.mean(dc.v('u1', 'river', range(jmax+1), range(kmax+1), 0), axis=1))
            # u = np.real(np.mean(dc.v('u1', range(jmax+1), range(kmax+1), 0), axis=1)) - np.real(np.mean(dc.v('u1', 'baroc', range(jmax+1), range(kmax+1), 0), axis=1))
            
            p = -1/lc + u / Kh
            h = x[2] - x[1]

            ngrid = len(x)
            dictM[str(channel)] = np.zeros((ngrid-2, ngrid))
            for i in range(ngrid-2):
                dictM[str(channel)][i, [i, i+1, i+2]] = np.array([
                    -1 - h / 2 * p[i+1],
                    2 , # q = 0
                    -1 + h / 2 * p[i+1]
                ]) 
        M = scipy.linalg.block_diag(*([dictM[str(i)] for i in range(nch)]))  # shape = (N-2*nch, N). 
        v = np.zeros(np.shape(M)[0]) 
        
        N = sum(Ns) # total # of grids in the network
        
        # prescribe s sea
        for channel in Sea:
            row_temp = np.zeros(N)
            i = sum(Ns[k] for k in range(channel)) 
            row_temp[i] = 1
            v_temp = ssea[channel]
            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
            
        # s = 0 from river
        i_riv = 0
        for channel in River:
            i = sum(Ns[k] for k in range(channel)) + Ns[channel] - 1 # index of the last point in The channel    
            row_temp = np.zeros(N)
            row_temp[i] = 1
            v_temp = 0

            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
            i_riv += 1
        
        # continuous s at junctions
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

        # mass conservation for salt flux at junctions
        for j in range(len(Vertex)):
            channels = vertex_nosign[j] # indices of channels that are connected at the branching point
            # one junction one condition
            row_temp = np.zeros(N)
            
            for i in range(len(channels)):
                channel = vertex_nosign[j][i] # channel index
                dc = self.input.v('network', str(channel))
                x = dc.v('grid', 'axis', 'x') * dc.v('L') # dimensional axis
                h = x[2] - x[1]
                jmax = dc.v('grid', 'maxIndex', 'x')
                kmax = dc.v('grid', 'maxIndex', 'z')
                B = dc.v('B', range(jmax+1), 0, 0)
                H = dc.v('H', 0, 0, 0)
                
                if sign[j][i] == 1:     # x = L
                    i_grid = sum(Ns[var] for var in range(channel)) + Ns[channel] - 1 # last grid point
                    row_temp[[i_grid, i_grid-1, i_grid-2]] = -np.array([
                        Q[channel] - 3 * B[-1] * H * Kh / 2 / h,
                        2 * B[-1] * H * Kh / h,
                        - B[-1] * H * Kh / 2 / h 
                    ])
                    
                elif sign[j][i] == -1:  # x = 0
                    i_grid = sum(Ns[var] for var in range(channel)) 
                    row_temp[[i_grid, i_grid+1, i_grid+2]] = np.array([
                        Q[channel] + 3 * B[0] * H * Kh / 2 / h,
                        -2 * B[0] * H * Kh / h,
                        + B[0] * H * Kh / 2 / h 
                    ])
            v_temp = 0
            M = np.vstack([M, row_temp])
            v = np.append(v, v_temp)
        s = np.linalg.solve(M, v)

        self.s = {}
        # overwirte the prescribed salinity in iFlow
        dictS = {}
        dictS['network'] = {}
        for channel in range(nch):
            dictS['network'][str(channel)] = DataContainer()
            x0 = sum(Ns[k] for k in range(channel))
            xL = x0 + Ns[channel] 
            self.s[str(channel)] = s[x0:xL]
            dictS['network'][str(channel)].addData('s0', deepcopy(s[x0:xL]))
        self.input.merge(dictS)


        # plt.figure()
        # lo = self.input.v('networksettings', 'geometry', 'lo')
        # for i in range(nch):
        #     dc = self.input.v('network', str(i))
        #     # x = dc.v('grid', 'axis', 'x') * dc.v('L') # dimensional axis
        #     x = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1)
        #     jmax = dc.v('grid', 'maxIndex', 'x')
        #     s = dc.v('s0', range(jmax+1), 0, 0)
        #     plt.plot(x, s)

        # plt.show()

        # salinity is stored in iFlow!

        # match baroc again
        # mod = 'baroc'

        # for i in range(nch):
        #     dc = self.input.v('network', str(channel))
        #     a = self.input.v('zeta1', mod, 0, 0, 0)
        #     b = dc.v('zeta1', mod, 0, 0, 0)
        #     self.coefL[mod][i] = self.input.v('zeta1', mod, -1, 0, 0)



        # M = np.zeros((2 * nch, 2 * nch), dtype=complex)
        # v = np.zeros(2 * nch, dtype=complex)
        # row = 0
        # # transport = 0 in river channels
        # for i in River:
        #     M[row,[2*i]] = 1
        #     row += 1
        # # 0 SSH at sea
        # for i in Sea:
        #     M[row,[[2*i, 2*i+1]]] = [self.coefQ_0[i], 1]
        #     v[row] = -self.coef0[mod][row]
        #     row += 1
        # # continuous SSH
        # for j in range(len(Vertex)):
        #     loc = vertex_nosign[j]  # indices of channels that are connected at branching point
        #     # loc[i]: index for channel
        #     for i in range(len(loc)-1):
        #         if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
        #             M[row, [2*loc[i], 2*loc[i]+1]] = [self.coefQ_L[loc[i]], 1]
        #             v[row] -= self.coefL[mod][loc[i]]
        #         elif sign[j][i] == -1:
        #             M[row, [2*loc[i], 2*loc[i]+1]] = [self.coefQ_0[loc[i]], 1]
        #             v[row] -= self.coef0[mod][loc[i]]
        #         if sign[j][i-1] == 1:
        #             M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.coefQ_L[loc[i-1]], -1]     
        #             v[row] += self.coefL[mod][loc[i-1]]           
        #         elif sign[j][i-1] == -1:
        #             M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-self.coefQ_0[loc[i-1]], -1]
        #             v[row] += self.coef0[mod][loc[i-1]]
        #         row += 1
        # # mass cons.
        # for j in range(len(Vertex)):
        #     for i in range(len(vertex_nosign[j])):
        #         k = vertex_nosign[j][i]
        #         if sign[j][i] == 1:
        #             M[row, 2*k] = 1
        #         elif sign[j][i] == -1:
        #             M[row, 2*k] = -1
        #     row += 1

        # C[mod] = np.reshape(np.linalg.solve(M, v), (nch, 2))



        return 

    # def exchangeFlowIntensity(self):
    
    #     numberofchannels = self.input.v('networksettings', 'numberofchannels')

    #     d = {}
    #     d['network'] = {}

    #     for i in range(numberofchannels):

    #         d['network'][str(i)] = DataContainer()

    #         dc = self.input.v('network', str(i))
    #         H = dc.v('H', 0, 0, 0)
    #         jmax = dc.v('grid', 'maxIndex', 'x')
    #         kmax = dc.v('grid', 'maxIndex', 'z')
    #         u = dc.v('u1', 'adv', range(jmax+1), range(kmax+1), 0) 
    #         # x =  dc.v('grid', 'axis', 'x')
    #         # z =  dc.v('grid', 'axis', 'z') * H
    #         z = np.linspace(-H, 0, kmax+1)
    #         x = np.linspace(0, dc.v('L'), jmax+1)
    #         X, Z = np.meshgrid(x, z)
            
    #         # exchangeIntensity = self.computeExchangeIntensity(u, z, Z, H)
    #         integrand = u.transpose() * (Z + 0.5 * H)
    #         exchangeIntensity = 4 * np.trapz(integrand, x=z, axis=0) / H**2
    #         # exchangeIntensity = u
    #         # exchangeIntensity = np.mean(u, axis=1)


    #         # u0 = self.input.v('u0', 'tide',   range(jmax+1), range(kmax+1), 1)
    #         # u0_DA = ny.integrate(u0, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]

    #         d['network'][str(i)].addData('xIntensity', exchangeIntensity)
        
    #     self.input.merge(d)



    