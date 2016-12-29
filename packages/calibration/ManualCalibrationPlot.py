"""
ManualCalibrationPlot

Date: 22-Apr-16
Authors: Y.M. Dijkstra
"""
from cost_function_DJ96 import cost_function_DJ96
import numpy as np
import step as st
import matplotlib.pyplot as plt
import nifty as ny
from src.util.diagnostics.KnownError import KnownError

class ManualCalibrationPlot:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        measurementset = self.input.v('measurementset')
        data = self.input.getKeysOf('experimentdata')
        calib_param = ny.toList(self.input.v('calibration_parameter'))
        label = ny.toList(self.input.v('label'))
        unit_temp = ny.toList(self.input.v('unit'))
        unit = []
        for u in unit_temp:
            if u == '-':
                unit.append('')
            else:
                unit.append('($'+u+'$)')
        if len(calib_param)==1:
            param_range = [[]]
        elif len(calib_param)==2:
            param_range = [[], []]
        else:
            raise KnownError('ManualCalibration not implemented for calibration with more than 3 parameters.')

        # inital loop to determine the parameter ranges
        for i, dat in enumerate(data):
            dc = self.input.v('experimentdata', dat)
            for j in range(0, len(calib_param)):
                param_range[j].append(dc.v(calib_param[j]))
        for j in range(0, len(calib_param)):
            param_range[j] = sorted(list(set(param_range[j])))

        # second loop to determine the values per parameter setting
        cost_range = np.nan*np.zeros([len(l) for l in param_range]+[2])
        for dat in data:
            dc = self.input.v('experimentdata', dat)
            index = [l.index(dc.v(calib_param[i])) for i, l in enumerate(param_range)]

            L = dc.v('grid', 'high', 'x')
            H0 = dc.n('grid', 'high', 'z', x=0)
            dc.merge(self.input.slice(measurementset))
            x_obs = dc.v(measurementset, 'x_waterlevel')/L
            x_ext = np.zeros(len(x_obs)+2)
            x_ext[1:-1] = x_obs
            x_ext[0] = 0
            x_ext[-1] = 1
            zeta_obs = dc.v(measurementset, 'zeta', x=x_obs, z=0, f=[1,2])


            # Start fix
            # zeta_obs = dc.data[measurementset]['zeta'].im_self.dataContainer.data['value'][:,1:3] #fix
            # from copy import deepcopy                                                                     #fix
            # newgrid = deepcopy(dc.data['zeta0']['tide'].im_self.dataContainer.data['grid']['outputgrid']) #fix
            # i = 0
            # while i is not None:
            #     try:
            #         keys = dc.data['zeta'+str(i)].keys()
            #         for key in keys:
            #             if i<2:
            #                 dc.data['zeta'+str(i)][key].im_self.dataContainer.data['grid'] = newgrid
            #             else:
            #                 keys2 =dc.data['zeta'+str(i)][key].keys()
            #                 for key2 in keys2:
            #                     dc.data['zeta'+str(i)][key][key2].im_self.dataContainer.data['grid'] = newgrid
            #         i += 1
            #     except:
            #         i = None
            # End fix

            zeta_mod = 0
            i = 0
            while True:
                if dc.v('zeta'+str(i), x=x_obs, z=0, f=[1,2]) is not None:
                    zeta_mod += dc.v('zeta'+str(i), x=x_obs, z=0, f=[1,2])
                    i += 1
                else:
                    break
                if i>3:
                    break

            cost_range[tuple(index)+(0,)] = cost_function_DJ96(x_ext, zeta_obs[:,0], zeta_mod[:,0])
            cost_range[tuple(index)+(1,)] = cost_function_DJ96(x_ext, zeta_obs[:,1], zeta_mod[:,1])


        st.configure()
        # 1D plots
        if len(calib_param) == 1:
            try:
                minlocM2 = [param_range[0][np.where(cost_range[:,0]==np.min(cost_range[:,0]))[0]]]
                minlocM4 = [param_range[0][np.where(cost_range[:,1]==np.min(cost_range[:,1]))[0]]]
            except:
                minlocM2 = [np.nan]
                minlocM4 = [np.nan]
            print 'Minumim $M_2$: '
            print calib_param[0]+' '+str(minlocM2[0])
            print 'Minumim $M_4$: '
            print calib_param[0]+' '+str(minlocM4[0])

            if self.input.v('axis')=='log':
                axis = np.log10(param_range[0])
                label = '$log_{10}$($'+label[0]+'$)' + unit[0]
                minlocM2 = np.log10(minlocM2)
                minlocM4 = np.log10(minlocM4)
            else:
                axis = param_range[0]
                label = '$'+calib_param[0] + '$' + unit[0]

            plt.figure(1, figsize=(1,1))
            plt.plot(axis, cost_range[:,0], 'k.')
            plt.plot(minlocM2[0], np.min(cost_range[:,0]), 'ro')
            plt.xlabel(label)
            plt.ylabel('Cost $M_2$')
            plt.yticks([],[])
            plt.ylim(0, max(cost_range[:,0]))

            plt.figure(2, figsize=(1,1))
            plt.plot(axis, cost_range[:,1], 'k.')
            plt.plot(minlocM4[0], np.min(cost_range[:,1]), 'ro')
            plt.xlabel(label)
            plt.ylabel('Cost $M_4$')
            plt.yticks([],[])
            plt.ylim(0, max(cost_range[:,1]))

        # 2D plots
        elif len(calib_param) == 2:
            try:
                minlocM2 = [param_range[0][np.where(cost_range[:,:,0]==np.min(cost_range[:,:,0]))[0]], param_range[1][np.where(cost_range[:,:,0]==np.min(cost_range[:,:,0]))[1]]]
                minlocM4 = [param_range[0][np.where(cost_range[:,:,1]==np.min(cost_range[:,:,1]))[0]], param_range[1][np.where(cost_range[:,:,1]==np.min(cost_range[:,:,1]))[1]]]
            except:
                minlocM2 = [np.nan, np.nan]
                minlocM4 = [np.nan, np.nan]
            print 'Minumim $M_2$: '
            print calib_param[0]+' '+str(minlocM2[0])
            print calib_param[1]+' '+str(minlocM2[1])
            print 'Minumim $M_4$: '
            print calib_param[0]+' '+str(minlocM4[0])
            print calib_param[1]+' '+str(minlocM4[1])

            if self.input.v('axis')=='log':
                axis1 = np.log10(param_range[0])
                axis2 = np.log10(param_range[1])
                label1 = '$log_{10}$($'+label[0]+'$)' + unit[0]
                label2 = '$log_{10}$($'+label[1]+'$)' + unit[1]
                minlocM2 = [np.log10(i) for i in minlocM2]
                minlocM4 = [np.log10(i) for i in minlocM4]
            else:
                axis1 = param_range[0]
                axis2 = param_range[1]
                label1 = '$'+label[0] + '$' + unit[0]
                label2 = '$'+label[1] + '$' + unit[1]
            plt.figure(1, figsize=(1,1))
            plt.hold(True)
            plt.contourf(axis1, axis2, np.transpose(cost_range[:, :, 0]), 30)

            plt.plot(minlocM2[0], minlocM2[1], 'ro')
            # plt.plot(axis1, np.log10(0.5*10**axis2*H0), 'r')
            plt.xlim(min(axis1), max(axis1))
            plt.ylim(min(axis2), max(axis2))
            # plt.plot(np.log10(0.003), np.log10(0.061), 'yo')  # best Scheldt calibration
            # plt.plot(np.log10(0.098), np.log10(0.019), 'yo')  # best Ems1981 calibration
            plt.title('Cost $M_2$')
            plt.xlabel(label1)
            plt.ylabel(label2)
            #plt.colorbar()

            plt.figure(2, figsize=(1,1))
            plt.hold(True)
            plt.plot(minlocM4[0], minlocM4[1], 'ro')
            plt.contourf(axis1, axis2, np.transpose(cost_range[:, :,1]), 30)
            # plt.plot(axis1, np.log10(0.5*10**axis2*H0), 'r')
            plt.xlim(min(axis1), max(axis1))
            plt.ylim(min(axis2), max(axis2))
            plt.title('Cost $M_4$')
            plt.xlabel(label1)
            plt.ylabel(label2)
            #plt.colorbar()


        st.show()

        d = {}
        return d