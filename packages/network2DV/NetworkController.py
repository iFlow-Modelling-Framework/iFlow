import inspect
import numpy as np
from copy import deepcopy
import logging
import nifty as ny
from .util.matchLeadingOrderHydro import matchLeadingOrderHydro
from .util.matchFirstOrderHydro import matchFirstOrderHydro


class NetworkController():
    # Variables
    hydrolead_match = 0
    hydrofirst_match = 0
    sediment_match = 0
    salinity_match = 0
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, block):
        self.input = input
        self.block = block
        return

    def run(self):
        # load networksettings file
        package, name = ny.splitModuleName(self.input.v('file'))
        main = ny.dynamicImport(package, name)
        main_inst = main()
        network_settings = main_inst.setup()
        self.input.merge(network_settings)

        # prepare
        self.numberofchannels = self.input.v('network_settings', 'numberofchannels')

        # make copies for each channel in the network
        self.channel_list = []
        for i in range(self.numberofchannels):
            newchannel, _ = self.block.deepcopy()
            self.channel_list.append(newchannel)

        # set input for each channel
        for i, channel in enumerate(self.channel_list):
            data = {}

            # geometry
            data['L'] = network_settings['network_settings']['L'][i]
            data['H0'] = {}
            data['B0'] = {}
            for key in network_settings['network_settings']['H0'].keys():
                data['H0'][key] = network_settings['network_settings']['H0'][key][i]
            for key in network_settings['network_settings']['B0'].keys():
                data['B0'][key] = network_settings['network_settings']['B0'][key][i]

            # grid
            data['xgrid'] = [network_settings['network_settings']['grid']['gridTypeX'][i], network_settings['network_settings']['grid']['jmax'][i]]
            data['zgrid'] = [network_settings['network_settings']['grid']['gridTypeZ'][i], network_settings['network_settings']['grid']['kmax'][i]]
            data['fgrid'] = ['integer', network_settings['network_settings']['grid']['fmax']]

            data['xoutputgrid'] = [network_settings['network_settings']['grid']['gridTypeX_out'][i], network_settings['network_settings']['grid']['jmax_out'][i]]
            data['zoutputgrid'] = [network_settings['network_settings']['grid']['gridTypeZ_out'][i], network_settings['network_settings']['grid']['kmax_out'][i]]
            data['foutputgrid'] = ['integer', network_settings['network_settings']['grid']['fmax_out']]

            # water motion
            data['A0'] = [0, 1, 0]
            data['A1'] = [0, 0, 1]
            data['phase0'] = [0, 0, 0]
            data['phase1'] = [0, 0, 0]
            data['Q0'] = 1
            data['Q1'] = 1

            # turbulence
            for key in network_settings['network_settings']['turbulence'].keys():
                data[key] = network_settings['network_settings']['turbulence'][key][i]

            # salinity
            for key in network_settings['network_settings']['salinity'].keys():
                data[key] = network_settings['network_settings']['salinity'][key][i]

            # sediment
            data['erosion_formulation'] = network_settings['network_settings']['sediment']['erosion_formulation'][i]
            data['finf'] = network_settings['network_settings']['sediment']['finf'][i]
            data['ws0'] = network_settings['network_settings']['sediment']['ws0'][i]
            data['Kh'] = network_settings['network_settings']['sediment']['Kh'][i]

            channel.addInputData(data)

        # run each channel, interrupt after each module to check for matching
        currentmodule = 0
        while currentmodule<len(self.channel_list[0].callStack):
            for i, channel in enumerate(self.channel_list):
                # print('channel '+str(i+1))
                channel.run(init=True, interrupt=True, startnumber=currentmodule)
                self.match_check(channel.result)
            currentmodule += 1

        # Process output
        d = {}
        d.update(network_settings)
        d['network_output'] = {}
        for i, channel in enumerate(self.channel_list):
            name = self.input.v('network_settings','label','ChannelLabel',i)
            dc = channel.getInput()
            dc.merge(channel.getOutput())
            dc.addData('channelNumber', i)
            d['network_output'][name] = dc
        return d

    def match_check(self, result):
        if 'zeta0' in result or 'zeta0_reverse' in result:
            self.hydrolead_match += 1

            if self.hydrolead_match == 2*self.numberofchannels: # account for two modules; including the return tide
                self.logger.info('Matching channels Hydro Lead')
                matchLeadingOrderHydro(self.input, self.channel_list)

        if 'zeta1' in result or 'zeta1_reverse' in result:
            self.hydrofirst_match += 1

            if self.hydrofirst_match == 2*self.numberofchannels:
                self.logger.info('Matching channels Hydro First')
                matchFirstOrderHydro(self.input, self.channel_list)

        # if 's0' in result:            # NO MATCHING FOR SALINITY, EVERYTHING SHOULD BE MATCHED ON INPUT
        #     self.salinity_match += 1
        #     if self.salinity_match == self.numberofchannels:
        #         matchSalinity()
