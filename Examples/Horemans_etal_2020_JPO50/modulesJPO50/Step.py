"""
Step module to plot model output

Date: 17-Aug-20
Authors: Y.M. Dijkstra, R.L. Brouwer, D.M.L. Horemans
"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import modulesJPO50.step_config as cf
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
        # plt.hold(True)

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

            if sublevel and subplots is None:
                lab = combi[-1]
                try:
                    lab = cf.names[lab]
                except:
                    pass
                plt.plot(*axisData, label=lab)
                plt.legend(fontsize=8)
            else:
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
        # plt.hold(True)

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
            else:
                value = np.real(value)

            conv = cf.conversion.get(axis1) or 1.   # convert size of axis depending on conversion factor in config
            axis1_dim = axis1_dim*conv
            conv = cf.conversion.get(axis2) or 1.   # convert size of axis depending on conversion factor in config
            axis2_dim = axis2_dim*conv
            conv = cf.conversion.get(value_label) or 1.   # convert size of axis depending on conversion factor in config
            value = value*conv

            # set cmap
            cmap = 'rainbow'
            if 'c' in value_label:
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=350)
            elif value_label == 'pValue':
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=0.05)
            elif value_label == 'ws0ms':
                cmap = 'coolwarm'
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=np.amax(np.abs(value)))
            else:
                normalisecolor = mpl.colors.Normalize(vmin=0, vmax=np.amax(np.abs(value)))

            # plot
            value = np.ma.masked_array(value, value == 0)
            plt.cm.rainbow.set_bad('w', 1.)

            plt.pcolormesh(axis1_dim, axis2_dim, value, norm=normalisecolor, cmap=cmap, shading='gouraud') #, cmap='YlOrBr'
            plt.plot(axis1_dim[:,1], axis2_dim[:,-1], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[:,0], axis2_dim[:,0], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[-1,:], axis2_dim[-1,:], 'k-', linewidth=0.2)
            plt.plot(axis1_dim[0,:], axis2_dim[0,:], 'k-', linewidth=0.2)
            from matplotlib.ticker import MaxNLocator
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            if int(mpl.__version__.split('.')[0])>1:
                plt.gca().set_facecolor([.75, .75, .75])
            else:
                plt.gca().set_axis_bgcolor([.75, .75, .75])
            cb = plt.colorbar()
            plt.xlim(np.min(axis1_dim), np.max(axis1_dim))
            plt.ylim(np.min(axis2_dim), np.max(axis2_dim))
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
            if numberOfSubplots > 1:
                if kwargs.get('operation')==np.abs or kwargs.get('operation')==abs:
                    plt.suptitle('|'+value_name+'|'+' ('+value_unit+')')
                elif kwargs.get('operation')==np.angle:
                    plt.suptitle('Phase('+value_name+')'+' ('+cf.units['phase']+')')
                elif kwargs.get('operation')==np.real:
                    plt.suptitle('Re('+value_name+')'+' ('+value_unit+')')
                elif kwargs.get('operation')==np.imag:
                    plt.suptitle('Im('+value_name+')'+' ('+value_unit+')')
                else:
                    plt.suptitle(value_name+' ('+value_unit+')')
            else:
                if kwargs.get('operation')==np.abs or kwargs.get('operation')==abs:
                    plt.title('|'+value_name+'|'+' ('+value_unit+')')
                elif kwargs.get('operation')==np.angle:
                    plt.title('Phase('+value_name+')'+' ('+cf.units['phase']+')')
                elif kwargs.get('operation')==np.real:
                    plt.title('Re('+value_name+')'+' ('+value_unit+')')
                elif kwargs.get('operation')==np.imag:
                    plt.title('Im('+value_name+')'+' ('+value_unit+')')
                else:
                    plt.title(value_name+' ('+value_unit+')')
        plt.title('')

        if 'c' in value_label:
            plt.axvline(x=110, ymax=0.55, color='black', linestyle='dashed')
            plt.annotate(s='', xy=(50, -10.5), xytext=(110, -10.5), arrowprops=dict(arrowstyle='<->'))
            # #plt.annotate(s='', xy=(50, -10.5), xytext=(90, -10.5), arrowprops=dict(arrowstyle='<->'))
            plt.text(63, -11.5, 'ETM width', size=7, color='black')
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
        # import matplotlib.style
        # import matplotlib as mpl
        # mpl.style.use('classic')
        ################################################################################################################
        # Extract args and/or kwargs
        ################################################################################################################
        sublevel = kwargs.get('sublevel') or kwargs.get('subsublevel') or False  # show sub-level data: True/False
        plotno = kwargs.get('plotno') or 1  # set plot number (default 1)
        display = kwargs.get(
            'display') or 5  # display number of mechanisms (sorted in descending order) or specific mechanisms
        scale = kwargs.get('scale') or False  # scale the transport contributions to the maximum: True/False
        concentration = kwargs.get('concentration') or False
        legend = kwargs.get('legend') or False
        capacity = kwargs.get('capacity') or False

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
        if self.input.v('f') is not None:
            a = self.input.v('f', x=x).reshape(len(x), )
            a_x = self.input.d('f', x=x, dim='x').reshape(len(x), )
        else:
            a = self.input.v('a').reshape(len(x), )
            a_x = -self.input.v('T') * a / self.input.v('F')
        # construct list with values to plot
        loopvalues = [[]]
        if sublevel:
            tmp_max = []
            for key in keyList:
                # print key
                if not 'diffusion' in key:  # and (key == 'stokes' or key == 'river' or key == 'source'):# and (key == 'stokes'): # and key != 'source': #and (key == 'stokes' or key == 'river' or key == 'fallvel'):# and (key == 'river' or key == 'stokes'):
                    T = self.input.v('T', key)
                    if capacity:
                        trans = T
                    else:
                        if key in self.input.getKeysOf('F'):
                            trans = T * a + self.input.v('F', key) * a_x
                            loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), key])
                        else:
                            trans = T * a
                    if key == 'tide':
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), 'ext. M4 tide'])
                    elif key == 'sedadv':
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), 'adv. sediment'])
                    elif key == 'fallvel':
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))),
                                              r'M2 settling vel.' '\n' 'due to flocculation'])
                        # loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), r'add. $w^{1}_{s}$ c$^0$ forcing'])
                    elif key == 'source':
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), 'river'])
                    elif key == 'nostress':
                        loopvalues[0].append([trans, 1, 'nostress'])
                        # loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), 'nostress'])
                    elif key == 'river':
                        index_trib = int(np.where(np.asarray(loopvalues[0])[:, 2] == 'river')[0])
                        loopvalues[0][index_trib][0] = loopvalues[0][index_trib][0] + trans
                    else:
                        loopvalues[0].append([trans, np.sqrt(np.mean(np.square(trans[:-31]))), key])
                    tmp_max.append(abs(trans).max())
                    if sublevel == 'subsublevel' and len(self.input.slice('T', key).getAllKeys()[0]) > 2:
                        loopvalues.append([])
                        tmpkeys = sorted(self.input.slice('T', key).getAllKeys(), key=itemgetter(2))
                        subkeys = [tmpkeys[i * 3:3 + i * 3] for i in range(len(tmpkeys) / 3)]
                        for subkey in subkeys:
                            tmp = np.zeros(a.shape)
                            for subsubkey in subkey:
                                tmp += self.input.v('T', *subsubkey) * a
                            loopvalues[len(loopvalues) - 1].append(
                                [tmp, np.sqrt(np.mean(np.square(trans))), subsubkey[-1]])
                        loopvalues[len(loopvalues) - 1].append([trans, np.sqrt(np.mean(np.square(trans))), key])
            if capacity:
                Ttotal = self.input.v(
                    'T')  # *a+self.input.v('F')*a_x#- self.input.v('T', 'diffusion_tide') - self.input.v('T', 'diffusion_river')
            else:
                Ttotal = ((self.input.v('T') - self.input.v('T', 'diffusion_tide') - self.input.v('T',
                                                                                                  'diffusion_river')) *
                          a + (self.input.v('F') - self.input.v('F', 'diffusion_tide') - self.input.v('F',
                                                                                                      'diffusion_river')
                               ) * a_x)
            # loopvalues = [[]]
            loopvalues[0].append([Ttotal, np.sqrt(np.mean(np.square(Ttotal))), 'total net transport'])
            maxT = max(tmp_max)
            # maxT = abs(Ttotal).max()
            # Sort transport mechanisms based on the value of their root-mean-square value
            loopvalues[0] = sorted(loopvalues[0], key=itemgetter(1), reverse=True)
            # Only take the largest transport contributions indicated by the display integer. If the display integer is
            # larger than the length of the keyList, then all contributions are taken into account
            if isinstance(display, int):
                loopvalues[0] = loopvalues[0][:min(display, len(keyList))]
            # Sort alphetically so that mechanisms receive the same line color for plotting
            loopvalues[0] = sorted(loopvalues[0], key=itemgetter(2), reverse=True)
        else:
            if capacity:
                Ttotal = self.input.v(
                    'T')  # - self.input.v('T', 'diffusion_tide') - self.input.v('T', 'diffusion_river')
            else:
                Ttotal = ((self.input.v('T') - self.input.v('T', 'diffusion_tide') - self.input.v('T',
                                                                                                  'diffusion_river')) *
                          a + (self.input.v('F') - self.input.v('F', 'diffusion_tide') - self.input.v('F',
                                                                                                      'diffusion_river')
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
        # plt.hold(True)
        if not sublevel:
            sp = plt.subplot()
            plt.axhline(0, color='k', linewidth=0.5)
            if scale:
                loopvalues[0][0][0] = loopvalues[0][0][0] / maxT
            ln = []
            ln += sp.plot(xdim, loopvalues[0][0][0], label='adv. transport')
            if concentration:
                conv = cf.conversion.get('c') or 1.
                c = np.real(np.mean(self.input.v('c0')[:, :, 0] + self.input.v('c1')[:, :, 0] +
                                    self.input.v('c2')[:, :, 0], axis=1)) * conv
                if scale:
                    c = c / c.max()
                    ln += sp.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                    labels = [l.get_label() for l in ln]
                    if legend is 'out':
                        plt.legend(ln, labels, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    elif legend is 'in':
                        plt.legend(ln, labels, loc='upper left', borderaxespad=0.2, fontsize=7,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4, frameon=False)
                    plt.title('Advective Transport')
                else:
                    sp2 = sp.twinx()
                    ln += sp2.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                    labels = [l.get_label() for l in ln]
                    if legend is 'out':
                        plt.legend(ln, labels, bbox_to_anchor=(1.3, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    elif legend is 'in':
                        plt.legend(ln, labels, loc='upper left', borderaxespad=0.2, fontsize=7,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4, frameon=False)
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
                        if legend is 'in':
                            sp.set_ylim([-1.1, 1.1])
                        else:
                            sp.set_ylim([-1.1, 1.1])
                    else:
                        yunitc = cf.units['c']
                        sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                        sp2.set_ylabel(r'$c$ (' + yunitc + ')')
                else:
                    if scale:
                        sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$ (' + yunitT + ')')
                        if legend is 'in':
                            sp.set_ylim([-1.1, 1.1])
                        else:
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
                    if i == len(subplot_values) - 1 and subplot_number >= 1:
                        ln += sp.plot(xdim, value[0], 'k', label=label)
                    else:
                        ln += sp.plot(xdim, value[0], label=label)
                if concentration and subplot_number == 0:
                    conv = cf.conversion.get('c') or 1.
                    c = np.real(np.mean(self.input.v('c0')[:, :, 0] + self.input.v('c1')[:, :, 0] +
                                        self.input.v('c2')[:, :, 0], axis=1)) * conv
                    if scale:
                        c = c / c.max()
                        ln += sp.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                        labels = [l.get_label() for l in ln]
                        if legend is 'out':
                            plt.legend(ln, labels, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0.,
                                       fontsize=cf.fontsize2,
                                       labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                        elif legend is 'in':
                            plt.legend(ln, labels, loc='upper left', borderaxespad=0.2, fontsize=7,
                                       labelspacing=0.1, handlelength=0.1, handletextpad=0.4, frameon=False)
                        if subplot_number == 0:
                            plt.title('Advective Transport')
                        else:
                            title = keyList[subplot_number - 1]
                            try:
                                title = cf.names[title]
                            except:
                                pass
                            plt.title(title)
                    else:
                        sp2 = sp.twinx()
                        ln += sp2.plot(xdim, c, '--', color='grey', label=r'$\langle\bar{c}\rangle$')
                        labels = [l.get_label() for l in ln]
                        if legend is 'out':
                            plt.legend(ln, labels, bbox_to_anchor=(1.3, 0.), loc=3, borderaxespad=0.,
                                       fontsize=cf.fontsize2,
                                       labelspacing=0.1, handlelength=0.5, handletextpad=0.4, framealpha=0.0)
                        elif legend is 'in':
                            plt.legend(ln, labels, loc='upper left', borderaxespad=0.2, fontsize=7,
                                       labelspacing=0.1, handlelength=0.1, handletextpad=0.4, frameon=False)
                        if subplot_number == 0:
                            plt.title(r'Advective Transport ', y=1.09)
                        else:
                            title = keyList[subplot_number - 1]
                            try:
                                title = cf.names[title]
                            except:
                                pass
                            plt.title(title, y=1.09)
                else:
                    if legend is 'out':
                        plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0., fontsize=cf.fontsize2,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4)
                    elif legend is 'in':
                        plt.legend(ln, labels, loc='upper left', borderaxespad=0.2, fontsize=7,
                                   labelspacing=0.1, handlelength=0.1, handletextpad=0.4, frameon=False)
                    if subplot_number == 0:
                        plt.title('Advective Transport')
                    else:
                        title = keyList[subplot_number - 1]
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
                            if legend is 'in':
                                sp.set_ylim([-1.1, 1.1])
                            else:
                                sp.set_ylim([-1.1, 1.1])
                        else:
                            yunitc = cf.units['c']
                            sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                            sp2.set_ylabel(r'$<\overline{c}>$ (' + yunitc + ')')
                    else:
                        if scale:
                            sp.set_ylabel(r'$\mathcal{T}$ / $\mathcal{T}_{max}$ (' + yunitT + ')')
                            if legend is 'in':
                                sp.set_ylim([-1.1, 1.1])
                            else:
                                sp.set_ylim([-1.1, 1.1])
                        else:
                            sp.set_ylabel(r'$\mathcal{T}$ (' + yunitT + ')')
                except KeyError:
                    yname = [r'$\mathcal{T}$']
                    yunit = ''
                    plt.label(yname + ' (' + yunit + ')')
        # set x and y limits
        xlim_ = 30  # 20#37#Q=73 choose 20#40 for Q154
        plt.xlim(0, max(xdim[:-xlim_]))
        min_ = []
        max_ = []
        for i in range(0, len(loopvalues[0][:])):
            min_.append(min(np.real(loopvalues[0][i][0][:-xlim_])))
            max_.append(max(np.real(loopvalues[0][i][0][:-xlim_])))
        # print 1.5 * min(min_), 1.5 * max(max_)
        # sp.set_ylim([-0.5, 1.5*max(max_)]) # ymin-1.4593231861 -0.25
        sp.set_ylim([-1., 1.3])
        # sp.set_ylim([-0.7, 0.6/3.])
        sp.set_xlabel('x (km)')
        plt.title('')
        plt.draw()
        return