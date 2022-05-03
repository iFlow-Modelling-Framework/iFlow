"""

Date: 19-01-2022

Authors: J. Wang 
"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy
from packages.analytical2DV.turbulence.profiles.UniformXF import UniformXF

class NetworkTurbulence:

    logger = logging.getLogger(__name__)

    def __init__(self, input):
        self.input = input
        self.channelnumber = 0
        self.iteration = 0
        return

    def stopping_criterion(self, iteration):

        # print('stopping')

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
        
        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0


        self.logger.info('Running network turbulence')
        d = self.prepareInput()
        return d

    def run(self):
        # self.logger.info('run geometry channel number ' +str(self.channelnumber+1))
   
        dout = self.storeOutput()
        d = self.prepareInput()
        d.update(dout)
        
        return d


    def prepareInput(self):
        """ load input variable from networksettings for geometry2DV"""
        d = {}

        return d

    def storeOutput(self):
        # dc = self.input.v('network', str(self.channelnumber-1))

        # H = dc.v('H')
        n = 1
        Av = self.input.v('networksettings', 'geometry', 'depth', self.channelnumber-1) ** n  *  self.input.v('Av_coef') 
        sf0 = self.input.v('sf0')

        d = {}
        d['network'] = {}
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].addData('Av', deepcopy(Av))
        d['network'][str(self.channelnumber-1)].addData('Kv', deepcopy(Av))
        d['network'][str(self.channelnumber-1)].addData('Roughness', deepcopy(sf0))


        return d
