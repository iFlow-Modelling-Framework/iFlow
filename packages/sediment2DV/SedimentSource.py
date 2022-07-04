"""
SedimentSource
Sediment-discharge relation

Date: 01-2019
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny


class SedimentSource:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        ## fluvial sources
        #   init
        d = {}
        c0 = ny.toList(self.input.v('QC_c'))
        cexp = ny.toList(self.input.v('QC_exp'))
        Q = self.input.v('Q1')

        xsedsource = ny.toList(self.input.v('xsource'))
        Qsource = ny.toList(self.input.v('Qsource'))

        #   Upstream boundary
        d['Qsed'] = c0[-1]*Q**cexp[-1]

        #   Other fluvial sources
        d['sedsource'] = [['point', xsedsource[i], c0[i]*Qsource[i]**cexp[i]] for i in range(0, len(Qsource))]

        ## Other sources and sinks
        xsedsource = ny.toList(self.input.v('x_sedsource'))     # one value for each point source, two values for each line source
        type = ny.toList(self.input.v('type_sedsource'))        # 'point' or 'line'
        Qsource = ny.toList(self.input.v('Q_sedsource'))        # sediment source in kg/s for point source and kg/s/m for line source
        ii = 0
        for i in range(0, len(Qsource)):
            if type[i] == 'point':
                d['sedsource'].append([type[i], xsedsource[ii], Qsource[i]])
                ii += 1
            if type[i] == 'line':
                d['sedsource'].append([type[i], xsedsource[ii], xsedsource[ii+1], Qsource[i]])
                ii += 2

        return d

