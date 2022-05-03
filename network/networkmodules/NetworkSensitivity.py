"""
Network sensitivity

Date: 29-Jul-2021
"""

import copy
import logging
import numpy as np
from src.util.diagnostics import KnownError
import src.config_menu as cfm
from src.DataContainer import DataContainer
from copy import deepcopy
# from nifty import toList
# import numbers
import os
# from itertools import product
import cPickle as pickle


class NetworkSensitivity:
    # Variables
    logger = logging.getLogger(__name__)
    skippedfiles = 0
    

    def __init__(self, input):
        self.input = input
        self.iteration = 0
        self.data = DataContainer()
        self.experiments = []
        self.variable = {}

    def stopping_criterion(self, iteration):

        var_val = self.variable[self.var_name][self.iteration]

        self.saveData(self.var_name, var_val) 
        self.experiments.append(self.var_name + str(var_val))

        stop = False
        self.iteration += 1 #=iteration + self.skippedfiles

        # dc = self.input.v('network', '1')
        # stop if iteration number exceeds number of prescribed variables
        if self.iteration >= np.product(self.numLoops):
            stop = True

            self.data.addData('experiments', self.experiments)
            self.data.addData('label', self.input.v('networksettings', 'label', 'ChannelLabel'))
            self.data.addData('color', self.input.v('networksettings', 'label', 'ColourLabel'))
            self.data.addData('numberofchannels', self.nch)

            # filename = self.path + 'SDE' + self.var_name + '10-20.p'
            # filename = self.path + 'MDE' + self.var_name + '10-100.p'
            # filename = self.path + 'SDE' + self.var_name + 'baroc.p'
            # filename = self.path + 'MDE' + self.var_name + 'barocRect.p'
            # filename = self.path + 'MDE' + self.var_name + 'barocFull.p'
            # filename = self.path + 'MDE' + self.var_name + 'Q4-8k.p'
            # filename = self.path + 'MDE' + self.var_name + '0-6k.p'
            # filename = self.path + 'MDE' + self.var_name + '7-15nos.p'
            # filename = self.path + 'MDE' + self.var_name + 'S0.p'
            # filename = self.path + 'MDE' + self.var_name + '5-30.p'
            # filename = self.path + 'MDE' + self.var_name + '5-20H=13.p'
            # filename = self.path + 'MDE' + self.var_name + '0-100.p'
            # filename = self.path + 'MDE' + self.var_name + '-100ex.p'
            # filename = self.path + 'MDE' + self.var_name + '300-1000.p'
            # filename = self.path + 'MDE' + self.var_name + '5-20shift4.p'
            # filename = self.path + 'MDE' + self.var_name + '10-20.p'
            # filename = self.path + 'MDE' + self.var_name + '5-20b4DWP.p'
            # filename = self.path + 'MDE' + self.var_name + 'DWPB.p'
            # filename = self.path + 'MDE' + self.var_name + '2m.p'
            filename = self.path + 'MDE' + self.var_name + '.p'

            with open(filename, 'wb') as fp:
                pickle.dump(self.data, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return stop

    def run_init(self):

        self.logger.info('Running network sensitivity')
        self.nch = self.input.v('networksettings', 'numberofchannels')

        cwdpath = cfm.CWD     
        # path = 
        # self.path = cwdpath + '/output/3ch/'
        # self.path = '/Users/jinyangwang/output/3ch/'
        self.path = '/Users/jinyang/output/3ch/'
        # self.path = '/Users/jinyangwang/output/YE4ch/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        nstep = 200

        key = 'geometry'
        # key = 'forcing'
# 
        # subkey = 'depth'
        # subkey = 'length'
        # subkey = 'discharge'
        # subkey = 'Qsed'
        # subkey = 'M2'
        # subkey = 'M4'
        subkey = 'width'
        # subkey = 'SLR'
        # key = 'SLR'
        # subkey = 'S0'
        self.varyVariable(nstep, key, subkey)

        self.numLoops = np.prod([len(self.variable[key]) for key in self.variable.keys()])
        # self.numLoops = nstep
        d = self.run()

        return d

    def run(self):
        """
        return: a new networksetting
        """

        d = {}

        self.logger.info('Sensitivity Analysis iteration %i of %i' % (self.iteration+1, self.numLoops))
        # self.logger.info('\tValues:')
        # for key in d:
        #     self.logger.info('\t%s = %s' % (key, str(d[key])))

        d['iteration'] = self.iteration

        d['networksettings'] = {} 
        d['networksettings'][self.key] = {}

        if self.key == 'forcing':
            if self.var_name == 'discharge':
                d['networksettings'][self.key][self.var_name] = self.variable[self.var_name][self.iteration]
            elif self.var_name == 'Qsed':
                d['networksettings'][self.key][self.var_name] = np.array([self.variable[self.var_name][self.iteration]])
                # print(self.variable[self.var_name][self.iteration])

            elif self.var_name == 'M2':
                if self.iteration == 0:
                    self.M2Amp = self.input.v('networksettings', 'forcing', 'M2')
                    d['networksettings']['forcing']['M2'] = (self.M2Amp * self.variable['M2'][self.iteration])
                else:
                    d['networksettings']['forcing']['M2'] = deepcopy(self.M2Amp * self.variable[self.var_name][self.iteration])
            elif self.var_name == 'M4':
                if self.iteration == 0:
                    self.M4Amp = np.abs(self.input.v('networksettings', 'forcing', 'M4'))
                    d['networksettings']['forcing']['M4'] = (self.M4Amp * np.exp(1j*self.variable['M4'][self.iteration]))
                else:
                    d['networksettings']['forcing']['M4'] = deepcopy(self.M4Amp * np.exp(1j*self.variable[self.var_name][self.iteration]))

            elif self.var_name == 'S0':
                channel = 0
                d['networksettings']['forcing'][self.var_name] = self.input.v('networksettings', 'forcing', 'S0')
                d['networksettings']['forcing'][self.var_name][channel] = self.variable[self.var_name][self.iteration]
                print(d['networksettings']['forcing'][self.var_name])



        elif self.key == 'geometry':

            
            if self.var_name == 'depth':
                channel = 0
                d['networksettings'][self.key][self.var_name] = self.input.v('networksettings', 'geometry', 'depth')
                d['networksettings'][self.key][self.var_name][channel] = self.variable[self.var_name][self.iteration]

            elif self.var_name == 'length':
                channel = 0
                d['networksettings'][self.key][self.var_name] = self.input.v('networksettings', 'geometry', 'length')
                d['networksettings'][self.key]['lo'] = self.input.v('networksettings', 'geometry', 'lo')

                d['networksettings'][self.key][self.var_name][channel] = self.variable[self.var_name][self.iteration]
                d['networksettings'][self.key]['lo'][channel] = -self.variable[self.var_name][self.iteration]
            
            elif self.var_name == 'width':
                channel = 0
                d['networksettings'][self.key]['width'] = self.input.v('networksettings', 'geometry', 'width')
                d['networksettings'][self.key]['lc'] = self.input.v('networksettings', 'geometry', 'lc')
             

                d['networksettings'][self.key]['width'][channel] = self.variable['width'][self.iteration]
# -L / log(bl/bo)
                L = self.input.v('networksettings', 'geometry', 'length')[channel]
                # d['networksettings'][self.key]['lc'][channel] = -L / np.log(3500 / self.variable['width'][self.iteration] )
                d['networksettings'][self.key]['lc'][channel] = -L / np.log(500 / self.variable['width'][self.iteration] )
                
                # -self.variable[self.var_name][self.iteration]

            elif self.var_name == 'SLR':

                if self.iteration == 0:
                    self.default_depths = self.input.v('networksettings', 'geometry', 'depth')
                    d['networksettings'][self.key]['depth'] = self.input.v('networksettings', 'geometry', 'depth')
                else:
                    d['networksettings'][self.key]['depth'] = deepcopy(self.default_depths)
                    d['networksettings'][self.key]['depth'] += self.variable[self.var_name][self.iteration]
                # d['networksettings'][self.key]['depth'] += self.var_values
                print(self.default_depths)
                print(d['networksettings'][self.key]['depth'] )

        return d


    def saveData(self, varName, varValue):

        """
        Add results to self.data. The key name follows the varied variabels. 
        self.data[variable][channel]
        """

        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')

        ext = '.p'
        filename = self.path + str(self.iteration) + ext
        keyname = varName + str(varValue)

        data = {}


        for i in range(self.nch):
            
            channel = str(i)
            dc = self.input.v('network', channel)
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')

            data[channel] = {}
            data[channel]['x'] = np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            data[channel]['B'] = dc.v('B', range(jmax+1), 0, 0)
            data[channel]['H'] = dc.v('H', 0, 0, 0)
            data[channel]['jmax'] = dc.v('grid', 'maxIndex', 'x')
            data[channel]['kmax'] = dc.v('grid', 'maxIndex', 'z')
            # data[channel]['exchangeIntensity'] = dc.v('xIntensity', range(jmax+1), 0, 0)

            f = dc.v('f', range(jmax+1), kmax, 0)
            c00 = (dc.v('hatc0', 'a', range(jmax+1), -1, 0))
            # bottom_stock = dc.v('Sbed', range(jmax+1))
            alpha1 = dc.v('alpha1', range(jmax+1))

            # data[channel]['cb'] = np.real(f * c00)
            data[channel]['f'] = f
            data[channel]['alpha1'] = np.real(alpha1)
            data[channel]['Qsed'] = -self.input.v('networksettings', 'forcing', 'Qsed')[0]
            # data[channel]['Sbed'] = bottom_stock

            VariableList = ['Q', 'T', 'F']

            # data[channel]['T'] = {}
            # data[channel]['F'] = {}
            # data[channel]['Q'] = {}

            # for key in dc.getKeysOf('Q'):
            #     data[channel]['Q'][key] = dc.v('Q', key)

            # for key in dc.getKeysOf('T'):
            #     data[channel]['T'][key] = dc.v('T', key)
            
            # for key in dc.getKeysOf('F'):
            #     data[channel]['F'][key] = dc.v('F', key)

            for variable in VariableList:
                data[channel][variable] = {}
                for key in dc.getKeysOf(variable):
                    if len(dc.getKeysOf(variable, key)) > 1:
                        data[channel][variable][key] = {}
                        for subkey in dc.getKeysOf(variable, key):
                            data[channel][variable][key][subkey] = dc.v(variable, key, subkey)
                    else:
                        data[channel][variable][key] = dc.v(variable, key)


            # data[channel]['u0'] = {}
            # data[channel]['u0'] = dc.v('u0', 'tide', range(jmax+1), range(kmax+1), 1)
            # data[channel]['zeta0'] = {}
            # data[channel]['zeta0'] = dc.v('zeta0', 'tide', range(jmax+1), -1, 1)

            # data[channel]['u1'] = {}
            # data[channel]['zeta1'] = {}
            # for key in dc.getKeysOf('u1'):
            #     data[channel]['u1'][key] = dc.v('u1', key, range(jmax+1), range(kmax+1), range(3))
            #     data[channel]['zeta1'][key] = dc.v('zeta1', key, range(jmax+1), range(kmax+1), range(3))



            # data[channel]['hatc1'] = {}
            # data[channel]['hatc1']
            # for key in dc.getKeysOf('hatc1','a','erosion'):
            #     data[channel]['hatc1'] = dc.v('hatc1','a','erosion','tide', range(jmax+1), range(kmax+1), 1)

            self.data.addData(keyname, data)

# dc.v('hatc1','a','erosion','tide',range(101),range(51),2)


    def varyVariable(self, nstep, key, subkey=None):

        # depth = np.linspace(7,15, nstep)
        # depth = np.linspace(5,20, nstep)
        # depth = np.linspace(7,11, nstep)
        depth = np.linspace(5,20, nstep)
        # depth = np.linspace(5,15, nstep)
        # depth = np.linspace(10,20, nstep)
        # discharge = np.linspace(500., 1500., nstep)
        # discharge = np.linspace(4000., 8000., nstep)
        discharge = np.linspace(300., 1000., nstep)
        Qsed = np.linspace(0., -100., nstep)
        # Qsed = np.linspace(0., -3000., nstep)
        # Qsed = np.linspace(0., -6000., nstep)
        # Qsed = np.linspace(1000., -1000., nstep)
        
        SLR = np.linspace(0, 2., nstep)
        # SLR = SLR[1] - SLR[0]
        length = np.linspace(20e3, 100e3, nstep)

        # width = np.linspace(3500+1e-10, 8500, nstep)
        width = np.linspace(500+1e-10, 2000, nstep)
        
        tideAmp = np.linspace(0.5, 1.5, nstep)
        M4pha = np.linspace(0, 2*np.pi, nstep)
        S0 = np.linspace(10, 30, nstep)


        self.key = key

        if key == 'geometry':
            if subkey == 'depth':
                self.variable['depth'] = depth
                self.var_name = 'depth'
                self.var_values = depth
            elif subkey == 'SLR':
                self.variable['SLR'] = SLR
                self.var_name = 'SLR'
                self.var_values = SLR
            elif subkey == 'length':
                self.variable['length'] = length
                self.var_name = 'length'
                self.var_values = length
            elif subkey == 'width':
                self.variable['width'] = width
                self.var_name = 'width'
                self.var_values = width

        elif key == 'forcing':
            if subkey == 'discharge':
                self.variable['discharge'] = discharge
                self.var_name = 'discharge'
                self.var_values = discharge
            elif subkey == 'Qsed':
                self.variable['Qsed'] = Qsed
                self.var_name = 'Qsed'
                self.var_values = Qsed
            elif subkey == 'M2':
                self.variable['M2'] = tideAmp
                self.var_name = 'M2'
                self.var_values = tideAmp
            elif subkey == 'M4':
                self.variable['M4'] = M4pha
                self.var_name = 'M4'
                self.var_values = M4pha
            elif subkey =='S0':
                self.variable['S0'] = S0
                self.var_name = 'S0'
                self.var_values = S0
            # elif subkey == ''





