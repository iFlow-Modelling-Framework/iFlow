"""
Step

Date: 06-Nov-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import matplotlib.pyplot as plt
import step_config as cf
from copy import copy
import nifty as ny
import itertools
import math


class Step:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        self.dims = list(set(input.v('grid','dimensions')+['t']))

        return

    def lineplot(self, axis1, axis2, *args, **kwargs):
        # determine which axis displays data and which grid information
        axis = [axis1, axis2]       # list of both axis
        del axis1, axis2
        gridAxisNo = [i for i, grid in enumerate(axis) if grid in self.dims][0]   # number of grid axis
        dataAxisNo = np.mod(gridAxisNo+1, 2)                                # number of data axis
        # TODO: error if no valid axis provided
        # TODO: handle time axis

        # get all keys/subkeys to data
        keyList = [axis[dataAxisNo]]+[i for i in args if isinstance(i, basestring)]     # key + subkeys to data

        # display a sub-level yes/no
        subplots = kwargs.get('subplots') or None           # which (if any) var to display in subplots
        sublevel = kwargs.get('sublevel') or False          # show sub-level data: True/False
        plotno = kwargs.get('plotno') or 1                  # set plot number (default 1)
        looplist = [dim for dim in kwargs if dim in self.dims and dim != axis[gridAxisNo]]+['sublevel']*sublevel # list of variables to loop over: i.e. grid axis that are not one of the axis + subplots

        # checks: no more than two loop variables not in subplots, maximum one var in subplots
        # TODO

        #######
        # determine values to loop plot over
        loopvalues = [None]*len(looplist)
        for i, loopvar in enumerate(looplist):
            if loopvar=='sublevel':
                loopvalues[i] = ny.toList(self.input.getKeysOf(*keyList))
            else:
                loopvalues[i] = ny.toList(kwargs[loopvar])

        # take all combinations of values
        if len(loopvalues)>1:
            permutations = itertools.product(*loopvalues)
        else:
            permutations = [(i,) for i in loopvalues[0]]

        #########
        # determine number and shape of subplots
        numberOfSubplots = [len(loopvalues[ii]) for ii in range(0, len(looplist)) if looplist[ii] in ny.toList(subplots)] or [1]
        numberOfSubplots = numberOfSubplots[0]
        if numberOfSubplots <= 1:
            subplotShape = (1, 1)
        elif numberOfSubplots == 2:
            subplotShape = (1, 2)
        elif numberOfSubplots <= 8:
            subplotShape = (int(math.ceil(numberOfSubplots/2.)), 2)
        elif numberOfSubplots <= 15:
            subplotShape = (int(math.ceil(numberOfSubplots/3.)), 3)
        else:
            subplotShape = (int(math.ceil(numberOfSubplots/4.)), 4)

        #########
        # plot loop
        plt.figure(plotno, figsize=(subplotShape[1]*cf.wunit, subplotShape[0]*cf.hunit))
        plt.hold(True)

        # loop over all combinations of the data
        for combi in permutations:
            if subplots in looplist:    # make subplot frame if there are subplots
                subplotIndex = looplist.index(subplots)
                subplot_number = loopvalues[subplotIndex].index(combi[subplotIndex])
                plt.subplot(*(subplotShape+(subplot_number+1,)))

            ## load data
            d = {}
            d[axis[gridAxisNo]] = self.input.v('grid', 'axis', axis[gridAxisNo]).reshape(self.input.v('grid', 'maxIndex', axis[gridAxisNo])+1)
            # append keylist with current sublevel
            keyListTemp = copy(keyList)
            for k in range(0, len(combi)):
                if looplist[k] == 'sublevel':
                    keyListTemp.append(combi[k])
                else:
                    d[looplist[k]] = combi[k]
            #   set axes
            axisData = [None]*2
            axisData[dataAxisNo] = self.input.v(*keyListTemp, **d)
            axisData[gridAxisNo] = ny.dimensionalAxis(self.input.slice('grid'), axis[gridAxisNo], **d)
            if kwargs.get('operation'):
                axisData[dataAxisNo] = kwargs['operation'](axisData[dataAxisNo])

            conv_grid = cf.conversion.get(axis[gridAxisNo]) or 1.   # convert size of axis depending on conversion factor in config
            conv_data = cf.conversion.get(keyListTemp[0]) or 1.
            axisData[dataAxisNo] = axisData[dataAxisNo]*conv_data
            axisData[gridAxisNo] = axisData[gridAxisNo]*conv_grid

            plt.plot(*axisData)

            ## Title and axis labels
            if numberOfSubplots > 1:
                if subplots == 'sublevel':
                    title = combi[subplotIndex]
                else:
                    title = subplots + ' = ' + str(combi[subplotIndex])
                try:
                    title = cf.names[title]
                except:
                    pass
                plt.title(title)

            # axis labels. Try to get from config file, else take plain name
            try:
                xname = cf.names[axis[0]]
                xunit = cf.units[axis[0]]
            except KeyError:
                xname = axis[0]
                xunit = ''
            try:
                yname = cf.names[axis[1]]
                yunit = cf.units[axis[1]]
            except KeyError:
                yname = axis[1]
                yunit = ''

            plt.xlabel(xname+' ('+xunit+')')
            plt.ylabel(yname+' ('+yunit+')')

            if kwargs.get('operation')==np.abs:
                if dataAxisNo==0:
                    plt.xlabel('|'+xname+'|'+' ('+xunit+')')
                else:
                    plt.ylabel('|'+yname+'|'+' ('+yunit+')')
            elif kwargs.get('operation')==np.angle:
                if dataAxisNo==0:
                    plt.xlabel('Phase('+xname+')'+' ('+cf.units['phase']+')')
                else:
                    plt.ylabel('Phase('+yname+')'+' ('+cf.units['phase']+')')
            elif kwargs.get('operation')==np.real:
                if dataAxisNo==0:
                    plt.xlabel('Re('+xname+')'+' ('+xunit+')')
                else:
                    plt.ylabel('Re('+yname+')'+' ('+yunit+')')
            elif kwargs.get('operation')==np.imag:
                if dataAxisNo==0:
                    plt.xlabel('Im('+xname+')'+' ('+xunit+')')
                else:
                    plt.ylabel('Im('+yname+')'+' ('+yunit+')')
            if kwargs.get('operation')==np.abs:
                if dataAxisNo==0:
                    plt.xlabel('|'+xname+'|'+' ('+xunit+')')
                else:
                    plt.ylabel('|'+yname+'|'+' ('+yunit+')')

        try:
            plt.suptitle(cf.names[axis[dataAxisNo]]+' over '+axis[gridAxisNo], y=1.02)
        except KeyError:
            plt.suptitle(axis[dataAxisNo]+' over '+axis[gridAxisNo], y=1.05)
        plt.draw()


        return
