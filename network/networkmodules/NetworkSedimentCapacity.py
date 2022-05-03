"""
Date: 07-October-2020
Authors: J.Wang

"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy
import nifty as ny
import numpy as np
from scipy.integrate import cumtrapz


class NetworkSedimentCapacity:
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

        numberofchannels = self.input.v('networksettings', 'numberofchannels')
        # stop if iteration number exceeds number of prescribed variables
        if self.channelnumber >= numberofchannels:
            stop = True

            # store output of last iteration
            d = self.storeOutput()
            self.input.merge(d)

        return stop


    def run_init(self):
        """
        """
        self.logger.info('Running Network Sediment Capacity')

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0

        d = self.prepareInput()

        return d      

    def run(self):
        """
        """
        dout = self.storeOutput()
        # self.input.merge(d)
        d = self.prepareInput()

        d.update(dout)

        return d


    def prepareInput(self):
        
        d = {}
        dc = self.input.v('network', str(self.channelnumber))
        d['csea'] = [1]
        d['Qsed'] = [1]
        d['grid'] = dc.data['grid']
        d.update(dc.data)
        
        return d

    def storeOutput(self):
        d = {}
        dc = self.input.v('network', str(self.channelnumber-1))
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        dictC = {}

        for hatc in ['hatc0', 'hatc1', 'hatc2']:
            dictC[hatc] = {}
            for var1 in self.input.getKeysOf(hatc): # var1 is 'a' or 'ax'
                dictC[hatc][var1] = {}
                if len(self.input.getKeysOf(hatc, var1)) > 0: # if there is any key under 'a' or 'ax'
                    for var2 in self.input.getKeysOf(hatc, var1): # var2 is 'erosion', 'noflux', or 'sedadv'
                        dictC[hatc][var1][var2] = {}
                        if len(self.input.getKeysOf(hatc, var1, var2)) > 0: # if there is any key under 'erosion'
                            for var3 in self.input.getKeysOf(hatc, var1, var2): # var3 is 'adv', 'stokes', ...
                                dictC[hatc][var1][var2][var3] = {}
                                dictC[hatc][var1][var2][var3] = self.input.v(hatc, var1, var2, var3, range(jmax+1), range(kmax+1), range(fmax+1))
                        else:
                            dictC[hatc][var1][var2] = self.input.v(hatc, var1, var2, range(jmax+1), range(kmax+1), range(fmax+1))
                else: # should not come to here...
                    dictC[hatc][var1] = self.input.v(hatc, var1, range(jmax+1), range(kmax+1), range(fmax+1))
                
        d['network'] = {} 
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].merge(dictC)


        return d