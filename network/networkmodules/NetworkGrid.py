"""


Date:
Authors:
"""
import logging
from src.DataContainer import DataContainer
from copy import deepcopy


class NetworkGrid:
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

        if self.input.v('iteration') > self.iteration:
            self.iteration = self.input.v('iteration')
            self.channelnumber = 0
            
        self.logger.info('Running network grid')
        d = self.prepareInput()
        return d

    def run(self):
        
        # self.logger.info('run grid channel number ' +str(self.channelnumber+1))
        dout = self.storeOutput()
        d = self.prepareInput()
        d.update(dout)
        return d


    def prepareInput(self):
        """ load input variable from networksettings for geometry2DV"""
        d = {}
        dc = self.input.v('network', str(self.channelnumber))

        # d['xgrid'] = [self.input.v('networksettings', 'grid', 'gridType', self.channelnumber), self.input.v('networksettings', 'grid', 'jmax', self.channelnumber)]
        d['xgrid'] = [self.input.v('networksettings', 'grid', 'gridType', self.channelnumber), self.input.v('networksettings', 'grid', 'jmax', self.channelnumber)]
        d['zgrid'] = [self.input.v('networksettings', 'grid', 'gridType', self.channelnumber), self.input.v('networksettings', 'grid', 'kmax', self.channelnumber)]
        d['fgrid'] = ['integer', self.input.v('networksettings', 'grid', 'fmax')]
        d['xoutputgrid'] = [self.input.v('networksettings', 'grid', 'gridType_out', self.channelnumber), self.input.v('networksettings', 'grid', 'jmax_out', self.channelnumber)]
        d['zoutputgrid'] = [self.input.v('networksettings', 'grid', 'gridType_out', self.channelnumber), self.input.v('networksettings', 'grid', 'kmax_out', self.channelnumber)]
        d['foutputgrid'] = ['integer', self.input.v('networksettings', 'grid', 'fmax_out')]
        d['L'] = dc.v('L')
        d['B'] = dc.v('B')
        d['H'] = dc.v('H')
        return d

    def storeOutput(self):

        grid = self.input.data['grid'] # not so nice way
        # grid = {}       # nicer way
        # gridvars = self.input.getKeysOf('grid')
        # for var in gridvars:
        #     grid[var] = self.input.v('grid', var)
    
        outputGrid = self.input.data['outputgrid']

        d = {}
        d['network'] = {}
        d['network'][str(self.channelnumber-1)] = DataContainer()
        d['network'][str(self.channelnumber-1)].addData('grid', deepcopy(grid))
        d['network'][str(self.channelnumber-1)].addData('outputGrid', deepcopy(outputGrid))

        return d
