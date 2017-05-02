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
from operator import itemgetter
from src.util.diagnostics import KnownError


class Step:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        self.dims = list(set(input.v('grid','dimensions')+['t']))

        return

    def lineplot(self, axis1, axis2, *args, **kwargs):
        """
        Makes a lineplot of a variable over a certain axes

        Parameters:
            axis1: string
                axis plotted on the horizontal axis
            axis2: string
                variable plotted on the vertical axis

        args:

        kwargs:
            x, z, f: integer or range of integers
                plots the given range of the associated dimension
            subplots: string
                generates subplots of the physical variable over the given string: 'sublevel' or 'f'. If 'sublevel',
                all sublevels that are associated with the physical variable appear in a subplot. If 'f', subplots will
                be generated over the frequency domain
            sublevel: boolean
                show sub-level data or not
            plotno: integer
                plot number
            operation: python function
                makes an operation on the physical variable using the python function. This function could be for
                instance a numpy function (e.g. np.abs, np.angle, np.imag or np.real) or a nifty function (e.g.
                ny.scalemax)
        """

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
        if len(loopvalues) >= 1:
            permutations = itertools.product(*loopvalues)
        else:
            permutations = [None] # 13-02-2017 YMD

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
            try:
                for k in range(0, len(combi)):
                    if looplist[k] == 'sublevel':
                        keyListTemp.append(combi[k])
                    else:
                        d[looplist[k]] = combi[k]
            except:
                pass
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
        """
        Plots 2DV contourplots of a variable over two axes

        Parameters:
            axis1: string
                axis plotted on the horizontal axis
            axis2: string
                axis plotted on the vertical axis
            value_label: string
                physical variable for which the controurplot needs to be made

        args:

        kwargs:
            x, z, f: integer or range of integers
                plots the given range of the associated dimension
            subplots: string
                generates subplots of the physical variable over the given string: 'sublevel' or 'f'. If 'sublevel',
                all sublevels that are associated with the physical variable appear in a subplot. If 'f', subplots will
                be generated over the frequency domain
            plotno: integer
                plot number
            operation: python function
                makes an operation on the physical variable using the python function. This function could be for
                instance a numpy function (e.g. np.abs, np.angle, np.imag or np.real) or a nifty function (e.g.
                ny.scalemax)
        """

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
            if 'c' in value_label:
                cmap = 'YlOrBr'
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=np.amax(np.abs(value)))
            else:
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

            # plot
            plt.pcolormesh(axis1_dim, axis2_dim, value, norm=normalisecolor, cmap=cmap, shading='gouraud') #, cmap='YlOrBr'
            plt.plot(axis1_dim[:,1], axis2_dim[:,-1], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[:,0], axis2_dim[:,0], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[-1,:], axis2_dim[-1,:], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[0,:], axis2_dim[0,:], 'k-', linewidth=0.2)
            plt.gca().set_axis_bgcolor([.75, .75, .75])
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

    def transportplot_mechanisms(self, **kwargs):
        """
        Plots the advective transport based on the physical mechanisms that force it.

        kwargs:
            sublevel: string
                displays underlying levels of the associated mechanisms: 'sublevel', 'subsublevel' or False
            plotno: integer
                plot number
            display: integer or list of strings
                displays the underlying mechanisms indicated. An integer plots the largest contributions up to that
                integer and a list of strings plots the mechanisms in that list
            scale: boolean
                scales the transport contributions to the maximum value of all contributions
            concentration: boolean
                plots the depth-mean, sub-tidal concentration in the background
       """
        ################################################################################################################
        # Extract args and/or kwargs
        ################################################################################################################
        sublevel = kwargs.get('sublevel') or kwargs.get('subsublevel') or False  # show sub-level data: True/False
        plotno = kwargs.get('plotno') or 1  # set plot number (default 1)
        display = kwargs.get('display') or 5 # display number of mechanisms (sorted in descending order) or specific mechanisms
        scale = kwargs.get('scale') or False # scale the transport contributions to the maximum: True/False
        concentration = kwargs.get('concentration') or False

        ################################################################################################################
        # Construct list of mechanisms to display and calculate these mechanisms
        ################################################################################################################
        # get keys of the transport mechanisms to display entered by the user or all mechanism
        if isinstance(display, list):
            if set(display).issubset(self.input.getKeysOf('T')):
                keyList = display
            else:
                raise KnownError('Not all transport mechanisms passed with display are available.')
        else:
            keyList = self.input.getKeysOf('T')

        # get availability and its derivative w.r.t. x
        x = self.input.v('grid', 'axis', 'x')
        a = self.input.v('a').reshape(len(x),)
        a_x = -self.input.v('T') * a / self.input.v('F')
        # construct list with values to plot
        loopvalues = [[]]
        if sublevel:
            tmp_max = []
            for key in keyList:
                if not 'diffusion' in key:
                    T = self.input.v('T', key)
                    if key in self.input.getKeysOf('F'):
                        trans = T * a + self.input.v('F', key) * a_x
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans))), key])
                    else:
                        trans = T * a
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans))), key])
                    tmp_max.append(abs(trans).max())
                    if sublevel == 'subsublevel' and len(self.input.slice('T', key).getAllKeys()[0]) > 2:
                        loopvalues.append([])
                        tmpkeys = sorted(self.input.slice('T', key).getAllKeys(), key=itemgetter(2))
                        subkeys = [tmpkeys[i*3:3+i*3] for i in range(len(tmpkeys)/3)]
                        for subkey in subkeys:
                            tmp = np.zeros(a.shape)
                            for subsubkey in subkey:
                                tmp += self.input.v('T', *subsubkey) * a
                            loopvalues[len(loopvalues)-1].append([tmp, np.sqrt(np.mean(np.square(trans))), subsubkey[-1]])
                        loopvalues[len(loopvalues)-1].append([trans, np.sqrt(np.mean(np.square(trans))), key])
            maxT = max(tmp_max)
            # Sort transport mechanisms based on the value of their root-mean-square value
            loopvalues[0] = sorted(loopvalues[0], key=itemgetter(1), reverse=True)
            # Only take the largest transport contributions indicated by the display integer. If the display integer is
            # larger than the length of the keyList, then all contributions are taken into account
            if isinstance(display, int):
                loopvalues[0] = loopvalues[0][:min(display, len(keyList))]
            # Sort alphetically so that mechanisms receive the same line color for plotting
            loopvalues[0] = sorted(loopvalues[0], key=itemgetter(2))
        else:
            Ttotal = ((self.input.v('T') - self.input.v('T', 'diffusion_tide') - self.input.v('T', 'diffusion_river')) *
                      a + (self.input.v('F') - self.input.v('F', 'diffusion_tide') - self.input.v('F', 'diffusion_river')
                           ) * a_x)
            loopvalues[0].append([Ttotal, np.sqrt(np.mean(np.square(Ttotal))), 'total'])
            maxT = abs(Ttotal).max()

        ################################################################################################################
        # determine number and shape of subplots
        ################################################################################################################
        numberOfSubplots = len(loopvalues)
        subplotShape = (numberOfSubplots, 2)

        ################################################################################################################
        # plot
        ################################################################################################################
        ## load grid data
        xdim = ny.dimensionalAxis(self.input.slice('grid'), 'x')[:, 0, 0]
        conv_grid = cf.conversion.get('x') or 1.  # convert size of axis depending on conversion factor in config
        xdim = xdim * conv_grid
        ## plot
        plt.figure(plotno, dpi=cf.dpi, figsize=subplotShape)
        plt.hold(True)
        if not sublevel:
            sp = plt.subplot()
            plt.axhline(0, color='k', linewidth=0.5)
            if scale:
                loopvalues[0][0][0] = loopvalues[0][0][0] / maxT
            ln = []
            ln += sp.plot(xdim, loopvalues[0][0][0], label='adv. transport')
            if concentration:
                c = np.real(np.mean(self.input.v('c0')[:, :, 0] + self.input.v('c1')[:, :, 0] +
                                    self.input.v('c2')[:, :, 0], axis=1))
                if scale:
                    c = c / c.max()
                    ln += sp.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                    labels = [l.get_label() for l in ln]
                    plt.legend(ln, labels, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                               labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    plt.title('Advective Transport')
                else:
                    sp2 = sp.twinx()
                    ln += sp2.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                    labels = [l.get_label() for l in ln]
                    plt.legend(ln, labels, bbox_to_anchor=(1.3, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                               labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    plt.title('Advective Transport', y=1.09)
            else:
                plt.title('Advective Transport')
            ## Axis labels
            try:
                xname = cf.names['x']
                xunit = cf.units['x']
            except KeyError:
                xname = 'x'
                xunit = ''
            plt.xlabel(xname + ' (' + xunit + ')')
            try:
                yunitT = cf.units['T']
                if concentration:
                    if scale:
                        sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$, $c$ / $c_{max}$ (-)')
                        sp.set_ylim([-1.1, 1.1])
                    else:
                        yunitc = cf.units['c']
                        sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                        sp2.set_ylabel(r'$c$ (' + yunitc + ')')
                else:
                    if scale:
                        sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$ (' + yunitT + ')')
                        sp.set_ylim([-1.1, 1.1])
                    else:
                        sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
            except KeyError:
                yname = [r'$\mathcal{T}$']
                yunit = ''
                plt.ylabel(yname + ' (' + yunit + ')')
        else:
            for subplot_number, subplot_values in enumerate(loopvalues):
                pos = np.unravel_index(subplot_number, subplotShape)
                sp = plt.subplot2grid(subplotShape, (pos[0], pos[1]))
                plt.axhline(0, color='k', linewidth=0.5)
                # loop over all combinations of the data
                ln = []
                for i, value in enumerate(subplot_values):
                    try:
                        label = cf.transportlabels[value[2]]
                    except KeyError:
                        label = value[2]
                    if scale:
                        value[0] = value[0] / maxT
                    if i == len(subplot_values)-1 and subplot_number >= 1:
                        ln += sp.plot(xdim, value[0], 'k', label=label)
                    else:
                        ln += sp.plot(xdim, value[0], label=label)
                if concentration and subplot_number == 0:
                    c = np.real(np.mean(self.input.v('c0')[:, :, 0] + self.input.v('c1')[:, :, 0] +
                                        self.input.v('c2')[:, :, 0], axis=1))
                    if scale:
                        c = c / c.max()
                        ln += sp.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                        labels = [l.get_label() for l in ln]
                        plt.legend(ln, labels, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                        if subplot_number == 0:
                            plt.title('Advective Transport')
                        else:
                            title = keyList[subplot_number]
                            try:
                                title = cf.names[title]
                            except:
                                pass
                            plt.title(title)
                    else:
                        sp2 = sp.twinx()
                        ln += sp2.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                        labels = [l.get_label() for l in ln]
                        plt.legend(ln, labels, bbox_to_anchor=(1.3, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                        if subplot_number == 0:
                            plt.title('Advective Transport', y=1.09)
                        else:
                            title = keyList[subplot_number]
                            try:
                                title = cf.names[title]
                            except:
                                pass
                            plt.title(title, y=1.09)
                else:
                    plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                               labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    if subplot_number == 0:
                        plt.title('Advective Transport')
                    else:
                        title = keyList[subplot_number]
                        try:
                            title = cf.names[title]
                        except:
                            pass
                        if concentration and subplot_number > 0:
                            plt.title(title, y=1.09)
                        else:
                            plt.title(title)
                # axis labels and limits. Try to get from config file, else take plain name
                try:
                    xname = cf.names['x']
                    xunit = cf.units['x']
                except KeyError:
                    xname = 'x'
                    xunit = ''
                plt.xlabel(xname + ' (' + xunit + ')')
                try:
                    yunitT = cf.units['T']
                    if concentration:
                        if scale:
                            if subplot_number == 0:
                                sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$, $c$ / $c_{max}$ (-)')
                            else:
                                sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$ (-)')
                            sp.set_ylim([-1.1, 1.1])
                        else:
                            yunitc = cf.units['c']
                            sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                            sp2.set_ylabel(r'$c$ (' + yunitc + ')')
                    else:
                        if scale:
                            sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$ (' + yunitT + ')')
                            sp.set_ylim([-1.1, 1.1])
                        else:
                            sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                except KeyError:
                    yname = [r'$\mathcal{T}$']
                    yunit = ''
                    plt.label(yname + ' (' + yunit + ')')
        plt.xlim(0, max(xdim))
        plt.draw()
        return
