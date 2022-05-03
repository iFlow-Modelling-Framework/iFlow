"""


Date:
Authors: J.Wang 
"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy


class NetworkSalinity:
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
            # self.channelnumber=0
        return stop

    def run_init(self):

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0

        self.logger.info('Running network salinity')
        d = self.prepareInput()
        return d

    def run(self):
        # self.logger.info('run salinity channel number ' +str(self.channelnumber+1))
        dout = self.storeOutput()
        d = self.prepareInput()
        d.update(dout)
        
        return d


    def prepareInput(self):
        """ load input variable from networksettings for geometry2DV"""
        d = {}

        d['ssea'] = self.input.v('networksettings', 'forcing', 'S0', self.channelnumber)
        d['xc'] = self.input.v('networksettings', 'forcing', 'xc', self.channelnumber)
        d['xl'] = self.input.v('networksettings', 'forcing', 'xL', self.channelnumber)
        d['L'] = self.input.v('networksettings', 'geometry', 'length', self.channelnumber)

        # print(d['ssea'])
         # d['H0'] = {}
        # d['H0']['type'] ='functions.Constant'
        # d['H0']['C0'] = self.input.v('networksettings', 'geometry', 'depth', self.channelnumber)

        # d['B0'] = {}
        # d['B0']['type'] = 'functions.Exponential'
        # d['B0']['C0'] = self.input.v('networksettings', 'geometry', 'width', self.channelnumber)
        # d['B0']['Lc'] = self.input.v('networksettings', 'geometry', 'lc', self.channelnumber)

        # d['L'] = self.input.v('networksettings', 'geometry', 'length', self.channelnumber)
        return d

    def storeOutput(self):

  
        s0 = self.input.v('s0')

        d = {}
        d['network'] = {}
        d['network'][str(self.channelnumber-1)] = DataContainer()
        if self.iteration == 0:
            # SaltHyperbolicTangent is getting bigger
            d['network'][str(self.channelnumber-1)].addData('s0', deepcopy(s0))

        return d
