"""
Step

Date: 06-Nov-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import matplotlib as mpl
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
        plt.figure(plotno, dpi=cf.dpi, figsize=subplotShape)
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
            if not kwargs.get('der'):
                axisData[dataAxisNo] = self.input.v(*keyListTemp, **d)
            else:
                d['dim'] = kwargs['der']
                axisData[dataAxisNo] = self.input.d(*keyListTemp, **d)
                d.pop('dim')
            axisData[gridAxisNo] = ny.dimensionalAxis(self.input.slice('grid'), axis[gridAxisNo], **d)
            if kwargs.get('operation'):
                if kwargs.get('operation') is not np.angle:
                    axisData[dataAxisNo] = kwargs['operation'](axisData[dataAxisNo])
                else:
                    axisData[dataAxisNo] = -kwargs['operation'](axisData[dataAxisNo]) * 180 / np.pi

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

        if axis[0] == 'x':
            plt.xlim(np.min(axisData[gridAxisNo]), np.max(axisData[gridAxisNo]))

        if kwargs.get('suptitle'):
            plt.suptitle(kwargs.get('suptitle'))
        else:
            try:
                plt.suptitle(cf.names[axis[dataAxisNo]]+' over '+axis[gridAxisNo])
            except KeyError:
                plt.suptitle(axis[dataAxisNo]+' over '+axis[gridAxisNo])
        plt.draw()

        return

    def contourplot(self, axis1, axis2, value_label, *args, **kwargs):
        # get all keys/subkeys to data
        keyList = [value_label]+[i for i in args if isinstance(i, basestring)]      # key + subkeys to data

        # display a sub-level yes/no
        subplots = kwargs.get('subplots') or False          # Variable for subplots
        plotno = kwargs.get('plotno') or 1                  # set plot number (default 1)

        loopvalues = [None]             # values to loop subplots over
        if subplots == 'sublevel':
            loopvalues = ny.toList(self.input.getKeysOf(*keyList)) 
        elif subplots:
            loopvalues = [i for i in ny.toList(kwargs[subplots])]

        # determine number and shape of subplots
        numberOfSubplots = len(loopvalues)
        if numberOfSubplots <= 1:
            subplotShape = (1, 1)
        elif numberOfSubplots == 2:
            subplotShape = (1, 2)
        elif numberOfSubplots <= 8:
            subplotShape = (int(math.ceil(numberOfSubplots / 2.)), 2)
        elif numberOfSubplots <= 15:
            subplotShape = (int(math.ceil(numberOfSubplots / 3.)), 3)
        else:
            subplotShape = (int(math.ceil(numberOfSubplots / 4.)), 4)

        # load axis data for evaluating the value to be plotted
        d = {}                      # contains all axis data for evaluating the function
        for key in kwargs:
            if key in self.input.v('grid', 'dimensions'):
                d[key] = kwargs[key]

        ## load axis data
        axis1_axis = self.input.v('grid', 'axis', axis1).reshape(self.input.v('grid', 'maxIndex', axis1) + 1)
        axis2_axis = self.input.v('grid', 'axis', axis2).reshape(self.input.v('grid', 'maxIndex', axis2) + 1)
        d[axis1] = axis1_axis
        d[axis2] = axis2_axis

        #########
        # plot loop
        plt.figure(plotno, figsize=subplotShape)
        plt.hold(True)

        # loop over all combinations of the data
        for subplot_number, sub in enumerate(loopvalues):
            plt.subplot(*(subplotShape+(subplot_number+1,)))

            # append keylist with current sublevel
            keyListTemp = copy(keyList)
            if subplots=='sublevel':
                keyListTemp.append(sub)
            elif subplots:
                d[subplots] = sub

            # load dimensional axes and the value to be plotted
            value = self.input.v(*keyListTemp, **d)
            axis1_dim = ny.dimensionalAxis(self.input.slice('grid'), axis1, **d)
            axis2_dim = ny.dimensionalAxis(self.input.slice('grid'), axis2, **d)
            if kwargs.get('operation'):
                if kwargs.get('operation') is not np.angle:
                    value = kwargs['operation'](value)
                else:
                    value = -kwargs['operation'](value) * 180 / np.pi

            conv = cf.conversion.get(axis1) or 1.   # convert size of axis depending on conversion factor in config
            axis1_dim = axis1_dim*conv
            conv = cf.conversion.get(axis2) or 1.   # convert size of axis depending on conversion factor in config
            axis2_dim = axis2_dim*conv
            conv = cf.conversion.get(value_label) or 1.   # convert size of axis depending on conversion factor in config
            value = value*conv

            # set cmap
            if 'u' in value_label:
                if np.amax(np.abs(value))==0:
                    cmap = 'RdBu_r'
                    normalisecolor = mpl.colors.Normalize(vmin=0, vmax=0)
                elif np.amin(value) >= 0:
                    normalisecolor = mpl.colors.Normalize(vmin=0, vmax=np.amax(np.abs(value)))
                    cmap = 'Reds'
                elif np.amax(value) <= 0:
                    normalisecolor = mpl.colors.Normalize(vmin=np.amin(value), vmax=0)
                    cmap = 'Blues_r'
                else:
                    normalisecolor = mpl.colors.Normalize(vmin=-np.amax(np.abs(value)), vmax=np.amax(np.abs(value)))
                    cmap = 'RdBu_r'
            elif 'c' in value_label:
                cmap = 'YlOrBr'
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=np.amax(np.abs(value)))

            # plot
            plt.pcolormesh(axis1_dim, axis2_dim, value, norm=normalisecolor, cmap=cmap, shading='gouraud') #, cmap='YlOrBr'
            plt.plot(axis1_dim[:,1], axis2_dim[:,-1], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[:,0], axis2_dim[:,0], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[-1,:], axis2_dim[-1,:], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[0,:], axis2_dim[0,:], 'k-', linewidth=0.2)
            cb = plt.colorbar()

            ## Title and axis labels
            if numberOfSubplots > 1:
                if subplots == 'sublevel':
                    title = sub
                else:
                    title = subplots + ' = ' + str(sub)
                try:
                    title = cf.names[title]
                except:
                    pass
                plt.title(title)

            # axis labels. Try to get from config file, else take plain name
            try:
                xname = cf.names[axis1]
                xunit = cf.units[axis1]
            except KeyError:
                xname = axis1
                xunit = ''
            try:
                yname = cf.names[axis2]
                yunit = cf.units[axis2]
            except KeyError:
                yname = axis2
                yunit = ''

            plt.xlabel(xname+' ('+xunit+')')
            plt.ylabel(yname+' ('+yunit+')')

            try:
                value_name = cf.names[value_label]
                value_unit = cf.units[value_label]
            except:
                value_name = value_label
                value_unit = ''
            if kwargs.get('operation')==np.abs or kwargs.get('operation')==abs:
                plt.suptitle('|'+value_name+'|'+' ('+value_unit+')')
            elif kwargs.get('operation')==np.angle:
                plt.suptitle('Phase('+value_name+')'+' ('+cf.units['phase']+')')
            elif kwargs.get('operation')==np.real:
                plt.suptitle('Re('+value_name+')'+' ('+value_unit+')')
            elif kwargs.get('operation')==np.imag:
                plt.suptitle('Im('+value_name+')'+' ('+value_unit+')')

        plt.draw()

        return

    def transportplot(self, axis1, axis2, *args, **kwargs):
        # determine which axis displays data and which grid information
        axis = [axis1, axis2]  # list of both axis
        del axis1, axis2
        gridAxisNo = [i for i, grid in enumerate(axis) if grid in self.dims][0]  # number of grid axis
        dataAxisNo = np.mod(gridAxisNo + 1, 2)  # number of data axis
        # TODO: error if no valid axis provided
        # TODO: handle time axis

        # get all keys/subkeys to data
        keyList = [axis[dataAxisNo]] + [i for i in args if isinstance(i, basestring)]  # key + subkeys to data

        # display a sub-level yes/no
        # subplots = kwargs.get('subplots') or None  # which (if any) var to display in subplots
        sublevel = kwargs.get('sublevel') or False  # show sub-level data: True/False
        plotno = kwargs.get('plotno') or 1  # set plot number (default 1)
        # looplist = [dim for dim in kwargs if dim in self.dims and dim != axis[gridAxisNo]] + ['sublevel'] * sublevel  # list of variables to loop over: i.e. grid axis that are not one of the axis + subplots

        # checks: no more than two loop variables not in subplots, maximum one var in subplots
        # TODO

        #######
        # determine values to loop plot over
        loopvalues = [None] * len(keyList)
        for i, loopvar in enumerate(keyList):
            if sublevel:
                if loopvar == 'T':
                    tempList = []
                    for key in self.input.getKeysOf(loopvar):
                        tempList.append([loopvar, key])
                    loopvalues[i] = tempList
                elif loopvar in ['TM2M0', 'TM2M2', 'TM2M4']:
                    tempList = []
                    for key in self.input.getKeysOf('T', 'TM2', loopvar):
                        tempList.append(['T', 'TM2', loopvar, key])
                    loopvalues[i] = tempList
                elif loopvar == 'TM0stokes':
                    tempList = []
                    for key in self.input.getKeysOf('T', 'TM0', loopvar):
                        tempList.append(['T', 'TM0', loopvar, key])
                    loopvalues[i] = tempList
                else:
                    tempList = []
                    for key in self.input.getKeysOf('T', loopvar):
                        tempList.append(['T', loopvar, key])
                    loopvalues[i] = tempList
            else:
                if loopvar == 'T':
                    loopvalues[i] = loopvar
                elif loopvar in ['TM2M0', 'TM2M2', 'TM2M4']:
                    loopvalues[i] = ['T', 'TM2', loopvar]
                else:
                    loopvalues[i] = ['T', loopvar]

        #########
        # determine number and shape of subplots
        numberOfSubplots = len(keyList)
        # subplotShape = (numberOfSubplots, 1)
        if numberOfSubplots <= 1:
            subplotShape = (1, 1)
        elif numberOfSubplots == 2:
            subplotShape = (1, 2)
        elif numberOfSubplots <= 8:
            subplotShape = (int(math.ceil(numberOfSubplots / 2.)), 2)
        elif numberOfSubplots <= 15:
            subplotShape = (int(math.ceil(numberOfSubplots / 3.)), 3)
        else:
            subplotShape = (int(math.ceil(numberOfSubplots / 4.)), 4)

        #########
        # plot loop
        plt.figure(plotno, dpi=cf.dpi, figsize=subplotShape)
        plt.hold(True)
        # loop over all combinations of the data
        for subplot_number, subplot_values in enumerate(loopvalues):
            pos = np.unravel_index(subplot_number, subplotShape)

            ## load data
            d = {}
            d[axis[gridAxisNo]] = self.input.v('grid', 'axis', axis[gridAxisNo]).reshape(
                self.input.v('grid', 'maxIndex', axis[gridAxisNo]) + 1)
            d['z'] = 0.
            d['f'] = 0.
            # set axes
            axisData = [None] * 2
            axisData[gridAxisNo] = ny.dimensionalAxis(self.input.slice('grid'), axis[gridAxisNo], **d)
            if kwargs.get('operation'):
                if kwargs.get('operation') is not np.angle:
                    value = kwargs['operation'](value)
                else:
                    value = -kwargs['operation'](value) * 180 / np.pi
            conv_grid = cf.conversion.get(axis[gridAxisNo]) or 1.  # convert size of axis depending on conversion factor in config
            axisData[gridAxisNo] = axisData[gridAxisNo] * conv_grid
            ## plot subplots
            if sublevel:
                sp = plt.subplot2grid((subplotShape[0], 10 * subplotShape[1]), (pos[0], 10 * pos[1]), colspan=8)
                for num, subplot_value in enumerate(subplot_values):
                    ## load labels
                    plt.axhline(0, color='k', linewidth=0.5)
                    if sum(self.input.v(*subplot_value)) != 0:
                        l = {}
                        if subplot_value[-1] is not 'tide':
                            label = subplot_value[-1]
                        else:
                            label = subplot_value[-2]
                        try:
                            label = cf.transportlabels[label]
                        except:
                            pass
                        l['label'] = label
                        axisData[dataAxisNo] = self.input.v(*subplot_value)
                        sp.plot(*axisData, **l)
                if len(loopvalues[subplot_number]) > 1 :
                    l = {}
                    l['label'] = 'total'
                    axisData[dataAxisNo] = self.input.v(*subplot_value[:-1])
                    total, = sp.plot(*axisData, **l)
                    plt.setp(total, color='k')
                plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                           labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
            else:
                sp = plt.subplot(*(subplotShape+(subplot_number+1,)))
                if sum(self.input.v(*subplot_values)) != 0:
                    plt.axhline(0, color='k', linewidth=0.5)
                    axisData[dataAxisNo] = self.input.v(*subplot_values)
                    if kwargs.get('operation'):
                        axisData[dataAxisNo] = kwargs['operation'](axisData[dataAxisNo])

                    conv_data = cf.conversion.get('T') or 1.
                    axisData[dataAxisNo] = axisData[dataAxisNo] * conv_data

                    sp.plot(*axisData)

            ## Title and axis labels
            if numberOfSubplots > 1:
                title = keyList[subplot_number]
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
                yname = cf.names['T']
                yunit = cf.units['T']
            except KeyError:
                yname = axis[1]
                yunit = ''

            plt.xlabel(xname + ' (' + xunit + ')')
            plt.ylabel('Transport (' + yunit + ')')

            if kwargs.get('operation') == np.abs:
                if dataAxisNo == 0:
                    plt.xlabel('|' + xname + '|' + ' (' + xunit + ')')
                else:
                    plt.ylabel('|' + yname + '|' + ' (' + yunit + ')')
            elif kwargs.get('operation') == np.angle:
                if dataAxisNo == 0:
                    plt.xlabel('Phase(' + xname + ')' + ' (' + cf.units['phase'] + ')')
                else:
                    plt.ylabel('Phase(' + yname + ')' + ' (' + cf.units['phase'] + ')')
            elif kwargs.get('operation') == np.real:
                if dataAxisNo == 0:
                    plt.xlabel('Re(' + xname + ')' + ' (' + xunit + ')')
                else:
                    plt.ylabel('Re(' + yname + ')' + ' (' + yunit + ')')
            elif kwargs.get('operation') == np.imag:
                if dataAxisNo == 0:
                    plt.xlabel('Im(' + xname + ')' + ' (' + xunit + ')')
                else:
                    plt.ylabel('Im(' + yname + ')' + ' (' + yunit + ')')
            if kwargs.get('operation') == np.abs:
                if dataAxisNo == 0:
                    plt.xlabel('|' + xname + '|' + ' (' + xunit + ')')
                else:
                    plt.ylabel('|' + yname + '|' + ' (' + yunit + ')')

            plt.suptitle('Sediment transport' + ' over ' + axis[gridAxisNo])
        plt.draw()

        return