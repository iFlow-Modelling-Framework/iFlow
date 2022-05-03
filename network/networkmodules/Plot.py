"""
Date:
Authors:
"""
import copy
import logging
import numpy as np
from src.util.diagnostics import KnownError
from nifty import toList
import numbers
import os
from itertools import product
import nifty as ny
import matplotlib.pyplot as plt
import step as st
from matplotlib.lines import Line2D

plt.rcParams['axes.axisbelow'] = True
plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = 0.85
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rc('axes', axisbelow=True)
class Plot:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input

        return

    def run(self):
        self.channels = [0, 1, 2]

        # self.plotWidth()
        # self.plotLead('M2')
        # self.plotResidualCurernt()
        self.plotTidalCurrent()
        # self.plotSalinity()
        # self.plotLead2('M2')
        # self.plotLead('M4')
        # self.plotResidualComponents()

        # self.plotFirst('dp', M4=1)
        # self.plotFirst('stokes', M4=0)
        # self.plotFirst('river')
        # self.plotFirst('adv', M4=0)
        # self.plotFirst('tide', M4=1)
        # self.plotFirst('nostress', M4=1)
        # self.plotFirst('baroc', M4=0)
        # self.u1_contour('adv')

        # self.plotSed()
        plt.show()
        return

    def plotResidualComponents(self):

        fsize=20
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.top'] = 0.94
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize, axisbelow=True) 
        number_channel = len(self.input.getKeysOf('network'))

        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
         

        mods = self.input.v('network', '0').getKeysOf('u1')
        for mod in mods:
            plt.figure(figsize=(10, 6.18))
            for i in self.channels:
                dc = self.input.v('network', str(i))
                jmax = dc.v('grid', 'maxIndex', 'x')
                kmax = dc.v('grid', 'maxIndex', 'z')
                u1M0bot = dc.v('u1', mod, range(jmax+1), -1, 0)
                u1M0sur = dc.v('u1', mod, range(jmax+1), 0, 0)
                u1M0avg = np.mean(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0), axis=1)
                x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
                plt.plot(x / 1000, np.zeros(np.shape(u1M0sur)), color='k', lw=3)
                plt.plot(x / 1000, -np.real(u1M0bot), color=color[i], label=label[i])
                plt.plot(x / 1000, -np.real(u1M0sur), color=color[i], ls='--')
                plt.plot(x / 1000, -np.real(u1M0avg), color=color[i], ls='dotted')     
            if mod == 'nostress':
                mod = 'Velocity depth asymmetry.'
            elif mod == 'adv':
                mod = 'Advection'
            elif mod == 'dp':
                mod = 'Dynamic pressure'
            elif mod == 'stokes':
                mod = 'Stokes return flow'
            elif mod == 'baroc':
                mod = 'Density-driven flow'
        
            plt.ylabel(mod + ' (m s$^{-1}$)')   
            plt.xlabel('Position (km)')  
            # plt.savefig(mod+'.pdf', facecolor='w', edgecolor='w')
                  



    def plotTidalCurrent(self):
        plt.rcParams['figure.subplot.left'] = 0.12
        plt.rcParams['figure.subplot.top'] = 0.94
        plt.rcParams['figure.subplot.right'] = 0.85
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize, axisbelow=True) 
        number_channel = len(self.input.getKeysOf('network'))
        # number_channel = 3
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10, 6.18))
        # for i in range(number_channel):
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = dc.v('u0', 'tide', range(jmax+1), kmax, 1)
            u1M4 = sum([dc.v('u1', mod, range(jmax+1), kmax, 2) for mod in dc.getKeysOf('u1')])
            # u1M0 = sum([np.mean(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0), axis=1) for mod in dc.getKeysOf('u1')])
            # u1M0bot = sum([dc.v('u1', mod, range(jmax+1), -1, 0) for mod in dc.getKeysOf('u1')])
            # u1M0sur = sum([dc.v('u1', mod, range(jmax+1), 0, 0) for mod in dc.getKeysOf('u1')])
            # print(dc.getKeysOf('u1'))

            # x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            # plt.plot(x, np.abs(u0), color=color[i], label=label[i])
            # plt.plot(x, np.real(u1M0sur), color=color[i], label=label[i], ls='--')
            # plt.plot(x, np.real(u1M0bot), color=color[i], label=label[i])

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        # for i in range(number_channel):
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            u0 = np.mean(dc.v('u0', 'tide', range(jmax+1), range(kmax+1), 1), axis=1)
            u1 = sum([np.mean(dc.v('u1', mod, range(jmax+1), range(kmax+1), 2), axis=1) for mod in dc.getKeysOf('u1')])
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            ax1.plot(x / 1000, np.angle(u0), color=color[i], label=label[i])
            ax2.plot(x / 1000, np.angle(u1), color=color[i], ls='--')
        ax1.legend(loc='upper left')
        ax1.set_ylim([-np.pi,np.pi])
        ax2.set_ylim([-np.pi,np.pi])

        # ax1.set_ylabel('Solid: M$_2$ current amplitude (m s$^{-1}$)')   
        # ax1.set_xlabel('Position (km)') 
        # ax2.set_ylabel('Dashed: M$_4$ current amplitude (m s$^{-1}$)') 
        ax1.set_ylabel('Solid: M$_2$ current phase (rad)')   
        ax1.set_xlabel('Position (km)') 
        ax2.set_ylabel('Dashed: M$_4$ current phase (rad)') 
        # ax1.set_yticks([0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
        # ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax2.set_yticks([0, 0.02, 0.04, 0.06, 0.08, .1])
        # plt.savefig('uM2M4_pha_ref.pdf', facecolor='w', edgecolor='w')



    def plotResidualCurernt(self):

        fsize=20
        plt.rcParams['figure.subplot.left'] = 0.12
        plt.rcParams['figure.subplot.top'] = 0.94
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize, axisbelow=True) 
        number_channel = len(self.input.getKeysOf('network'))
        # number_channel = 3
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10, 6.18))
        # for i in range(number_channel):
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = dc.v('u0', 'tide', range(jmax+1), kmax, 1)
            u1M4 = sum([dc.v('u1', mod, range(jmax+1), kmax, 2) for mod in dc.getKeysOf('u1')])
            # u1M0 = sum([np.mean(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0), axis=1) for mod in dc.getKeysOf('u1')])
            # u1M0bot = sum([dc.v('u1', mod, range(jmax+1), -1, 0) for mod in dc.getKeysOf('u1')])
            # u1M0sur = sum([dc.v('u1', mod, range(jmax+1), 0, 0) for mod in dc.getKeysOf('u1')])
            # print(dc.getKeysOf('u1'))

            # x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            # plt.plot(x, np.abs(u0), color=color[i], label=label[i])
            # plt.plot(x, np.real(u1M0sur), color=color[i], label=label[i], ls='--')
            # plt.plot(x, np.real(u1M0bot), color=color[i], label=label[i])

        ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # for i in range(number_channel):
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            u1M0bot = sum([dc.v('u1', mod, range(jmax+1), -1, 0) for mod in dc.getKeysOf('u1')])
            u1M0sur = sum([dc.v('u1', mod, range(jmax+1), 0, 0) for mod in dc.getKeysOf('u1')])
            u1M0avg = np.mean(sum([dc.v('u1', mod, range(jmax+1), range(kmax+1), 0) for mod in dc.getKeysOf('u1')]), axis=1)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            ax1.plot(x / 1000, np.zeros(np.shape(u1M0sur)), color='k', lw=3)
            ax1.plot(x / 1000, -np.real(u1M0bot), color=color[i], label=label[i])
            # ax2.plot(x / 1000, -np.real(u1M0sur), color=color[i], ls='--')
            ax1.plot(x / 1000, -np.real(u1M0sur), color=color[i], ls='--')
            ax1.plot(x / 1000, -np.real(u1M0avg), color=color[i], ls='dotted')            
        # ax1.legend()
        ax1.set_ylim([-0.05, 0.25])
        # ax2.set_ylim([-0.05, 0.25])

        # ax1.set_ylabel('Solid: bottom residual current (m s$^{-1}$)')   
        ax1.set_ylabel('Residual current (m s$^{-1}$)')   
        ax1.set_xlabel('Position (km)') 
        
        # ax2.set_ylabel('Dashed: surface residual current (m s$^{-1}$)') 
        ax1.set_yticks([0, 0.1,0.2])
        # ax2.set_yticks([0, 0.1,0.2])

        ax1.text(-38, 0.005, "solid: bottom", color="k", fontsize=18)
        ax1.text(-38, -0.015, "dashed: surface", color="k", fontsize=18)
        # ax1.text(30, 0.32, "solid: bottom", color="k", fontsize=16)
        # ax1.text(30, 0.29, "dashed: surface", color="k", fontsize=16)
        ax1.text(-38, -0.035 , "dotted: depth-averaged", color="k", fontsize=18)
        # plt.savefig('uM0_ref.pdf', facecolor='w', edgecolor='w')
        
        # lines = [Line2D([0], [0], color=c, linewidth=2) for c in color]
        # plt.legend(lines, label)


    def plotSed(self):
        number_channel = len(self.input.getKeysOf('network'))
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10,6.18))
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            c0 = dc.v('hatc0', 'a', range(jmax+1), kmax, 0)
            f = dc.v('f', range(jmax+1), kmax, 0)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, f, color=color[i], label=label[i])
            plt.ylabel('$f$')
            plt.legend()

 
    def plotWidth(self):
        number_channel = len(self.input.getKeysOf('network'))
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            B = dc.v('B', range(jmax+1), 0, 0)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, B, color=color[i], label=label[i])
            plt.ylabel('Width (m)')
            plt.legend()

    def plotSalinity(self):
        plt.rcParams['figure.subplot.left'] = 0.1
        plt.rcParams['figure.subplot.top'] = 0.94
        plt.rcParams['figure.subplot.right'] = 0.9
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        # plt.rc('axes', labelsize=fsize, axisbelow=True) 
        # number_channel = len(self.input.getKeysOf('network'))
        number_channel = 3
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10, 6.18))
        lines = [Line2D([0], [0], color=c, linewidth=2) for c in color[:-1]]
        # lines = [Line2D([0], [0], color=c, linewidth=2) for c in ['b','r','k']]
        plt.legend(lines, ['1', '2', '3'], loc="upper left")
        # plt.legend(lines, ['NP', 'SP', 'SC'], loc="upper left")
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_ylabel('Solid: salinity (psu)')   
        ax1.set_xlabel('Position (km)') 
        ax2.set_ylabel('Dashed: salinity gradient (psu m$^{-1}$)') 
        # ax1.set_yticks(np.array([0, 6, 12, 18, 24, 30]))
        # ax2.set_yticks(np.array([0, 1,3,5,7]) *1e-4)
        # ax1.set_ylim([0, 35])
        # ax2.set_ylim([0, 2.5*1e-3])
        ax1.set_yticks(np.array([0, 7, 14, 21, 28 ,35]))
        # ax2.set_yticks(np.array([0, 0.5, 1, 1.5, 2, 2.5]) *1e-3)

        # for i in range(number_channel):
        for i in self.channels:
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            c0 = dc.v('hatc0', 'a', range(jmax+1), kmax, 0)
            # f = dc.v('f', range(jmax+1), kmax, 0)
            s = dc.v('s0', range(jmax+1), 0, 0)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            ax1.plot(x / 1000, s, color=color[i], zorder=10)
            ax2.plot(x / 1000, np.gradient(s, x, edge_order=2), color=color[i], ls='--', zorder=10)
        # ax1.set_yticks(np.array([0, 7, 14, 21, 28 ,35]))
        # ax2.set_yticks(np.array([0, 0.5, 1, 1.5, 2, 2.5]) *1e-3)

            
        # plt.savefig('salinity.pdf', facecolor='w', edgecolor='w')


    def plotLead(self, arg = 'M2'):
        if arg == 'M2':
            f = 1
        elif arg == 'M4':
            f = 2

        # xlim = [-23000,61000]
        
        number_channel = len(self.input.getKeysOf('network'))
        # number_channel = 3
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            zeta0 = dc.v('zeta0', 'tide', range(jmax+1), kmax, f)
            # zeta0 = dc.v('Av', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.abs(zeta0), color=color[i], label=label[i])
            plt.ylabel('$|\zeta^0|$')
            plt.legend()
        
        # plt.xlim(xlim)
        # plt.ylim([0,1.6])
        # plt.savefig('eta0amp.pdf', dpi=None, facecolor='w', edgecolor='w')

        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            zeta0 = dc.v('zeta0', 'tide', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.angle(zeta0), color=color[i], label=label[i])
            plt.ylabel(arg +' arg $\zeta^0$')
            plt.legend()
        # plt.xlim(xlim)
        # plt.savefig('eta0arg.pdf', dpi=None, facecolor='w', edgecolor='w')

        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = np.mean(dc.v('u0', 'tide', range(jmax+1), range(kmax+1), f), axis=1)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.abs(u0), color=color[i], label=label[i])
            plt.ylabel('|$u^0$|')
            plt.legend()
        # plt.xlim(xlim)
        # plt.savefig('u0amp.pdf', dpi=None, facecolor='w', edgecolor='w')

        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = np.mean(dc.v('u0', 'tide', range(jmax+1), range(kmax+1), f), axis=1)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.angle(u0), color=color[i], label=label[i])
            plt.ylabel(arg +' arg $u^0$')
            plt.legend()
        # plt.xlim(xlim)
        # plt.savefig('u0arg.pdf', dpi=None, facecolor='w', edgecolor='w')
        # plt.show()


    def plotLead2(self, arg = 'M2'):
        if arg == 'M2':
            f = 1
        elif arg == 'M4':
            f = 2

        number_channel = len(self.input.getKeysOf('network'))
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            zeta0 = dc.v('zeta0', 'tide_2', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.abs(zeta0), color=color[i], label=label[i])
            plt.ylabel(arg + ' $\zeta_0$')
            plt.legend()
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            zeta0 = dc.v('zeta0', 'tide_2', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.angle(zeta0), color=color[i], label=label[i])
            plt.ylabel(arg +' arg $\zeta_0$')
            plt.legend()

        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = dc.v('u0', 'tide_2', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.abs(u0), color=color[i], label=label[i])
            plt.ylabel(arg + ' $u_0$')
            plt.legend()
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u0 = dc.v('u0', 'tide_2', range(jmax+1), kmax, f)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, np.angle(u0), color=color[i], label=label[i])
            plt.ylabel(arg +' arg $u_0$')
            plt.legend()

    def plotFirst(self, mod, M4=False):
        number_channel = len(self.input.getKeysOf('network'))
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            zeta1 = np.real(dc.v('zeta1', mod, range(jmax+1), kmax, 0))
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, (zeta1), color=color[i], label=label[i])
            plt.ylabel(mod+' $\zeta^1_0$')
            plt.legend()
        # plt.savefig('eta_'+mod+'.pdf', dpi=None, facecolor='w', edgecolor='w')

        plt.figure(figsize=(10,6.18))
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            fmax = dc.v('grid', 'maxIndex', 'f')
            u1 = np.mean(np.real(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0)), axis=1)
            x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
            plt.plot(x, (u1), color=color[i], label=label[i])
            plt.ylabel(mod +' $u^1_0$')
            plt.legend()
        # plt.savefig('u_'+mod+'.pdf', dpi=None, facecolor='w', edgecolor='w')

        if M4 == 1:
            plt.figure(figsize=(10,6.18))
            for i in range(number_channel):
                dc = self.input.v('network', str(i))
                jmax = dc.v('grid', 'maxIndex', 'x')
                kmax = dc.v('grid', 'maxIndex', 'z')
                fmax = dc.v('grid', 'maxIndex', 'f')
                zeta1 = dc.v('zeta1', mod, range(jmax+1), kmax, 2)
                x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
                plt.plot(x, np.abs(zeta1), color=color[i], label=label[i])
                plt.ylabel(mod +' $|\zeta^1_2|$')
                plt.legend()

            plt.figure(figsize=(10,6.18))
            for i in range(number_channel):
                dc = self.input.v('network', str(i))
                jmax = dc.v('grid', 'maxIndex', 'x')
                kmax = dc.v('grid', 'maxIndex', 'z')
                fmax = dc.v('grid', 'maxIndex', 'f')
                u1 = (dc.v('u1', mod, range(jmax+1), kmax, 2))
                x = -np.linspace(lo[i], lo[i]+dc.v('L'), jmax+1) 
                plt.plot(x, np.abs(u1), color=color[i], label=label[i])
                plt.ylabel(mod +' $|u^1_2|$')
                plt.legend()

    def u1_contour(self,mod):
        number_channel = len(self.input.getKeysOf('network'))
        lo = self.input.v('networksettings', 'geometry', 'lo')
        color = self.input.v('networksettings', 'label', 'ColourLabel')
        label = self.input.v('networksettings', 'label', 'ChannelLabel')
        fig1, axs1 = plt.subplots(2, 4, figsize=(12, 8), edgecolor='w')
        
        vmin = 0
        vmax = 0
        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            u1 = np.real(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0))

            vmax = np.max([vmax, np.max(u1)])
            vmin = np.max([vmin, np.min(u1)])

        for i in range(number_channel):
            dc = self.input.v('network', str(i))
            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            x = np.linspace(0, dc.v('L'), jmax+1) / 1000
            z = np.linspace(-1, 0, kmax+1)
            X, Z = np.meshgrid(x, z)
            u1 = np.real(dc.v('u1', mod, range(jmax+1), range(kmax+1), 0)).T
            ax = plt.subplot(2, 4, i+1)
            c1 = ax.contour(X, Z, u1, colors='k')
            c  = ax.pcolor(X, Z, np.flipud(u1), vmin=vmin, vmax=vmax, cmap='rainbow')
        ax = plt.subplot(2, 4, 8)
        ax.axis('off')
        cbaxes = fig1.add_axes([0.79, 0.05, 0.02, 0.41]) 
        cb = plt.colorbar(c, cax = cbaxes)  
        cb.set_label('u (m/s)')

        plt.show()
