"""

Date:

Authors: J. Wang and Y.M. Dijkstra
"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy


class NetworkGeometry:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
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


        self.logger.info('Running network geometry')
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

        d['H0'] = {}
        d['H0']['type'] ='functions.Constant'
        d['H0']['C0'] = self.input.v('networksettings', 'geometry', 'depth', self.channelnumber)

        d['B0'] = {}
        d['B0']['type'] = 'functions.Exponential'
        # d['B0']['type'] = 'functions.Constant'
        d['B0']['C0'] = self.input.v('networksettings', 'geometry', 'width', self.channelnumber)
        d['B0']['Lc'] = self.input.v('networksettings', 'geometry', 'lc', self.channelnumber)

        d['L'] = self.input.v('networksettings', 'geometry', 'length', self.channelnumber)
        return d

    def storeOutput(self):

        H = self.input.v('H')
        B = self.input.v('B')
        L = self.input.v('L')

        d = {}
        d['network'] = {}
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].addData('H', deepcopy(H))
        d['network'][str(self.channelnumber-1)].addData('B', deepcopy(B))
        d['network'][str(self.channelnumber-1)].addData('L', deepcopy(L))
        return d
