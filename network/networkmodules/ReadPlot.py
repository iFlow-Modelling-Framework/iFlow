import cPickle as pickle
import types
import logging
import os
from src.util.diagnostics.KnownError import KnownError
import src.DataContainer
import matplotlib
import matplotlib.pyplot as plt
import src.config_menu as cfm
import numpy as np
from matplotlib import colors
from scipy.interpolate import make_interp_spline
from nifty.harmonicDecomposition import absoluteU, signU
from matplotlib.lines import Line2D


plt.rcParams['axes.grid'] = True
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15

plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = .85
plt.rcParams['figure.subplot.top'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rcParams['figure.subplot.wspace'] = 0.1
plt.rcParams['figure.subplot.hspace'] = 0.2
plt.rc('image', cmap='rainbow')
# cmap = 'YlOrBr'
# matplotlib.use('pdf')
class ReadPlot:

    logger = logging.getLogger(__name__)

    def __init__(self, input):
        self.input = input

    def run(self):
        
        path = cfm.CWD + '/output/3ch/'
        # path = '/Users/jinyangwang/output/3ch/'
        # path = '/Users/jinyang/output/7ch/'
        path = '/Users/jinyang/output/3ch/'
        # path = '/Users/jinyangwang/output/YE4ch/'
        # filename = 'depth.p'
        # filename = 'discharge.p'
        # filename = 'out_2ch_H0.p'

      
        # filename = 'MDEwidth5-20.p'
        # filename = 'MDEwidthH11m.p'
        # filename = 'MDEwidthH7m.p' 


        # filename = 'MDEQsed3000.p'
        # filename = 'MDEdepth5-15.p'
        # filename = 'MDEwidthDWP7m.p' 
        # filename = 'MDEwidthDWP11m.p' # Depth of NP = 7m 
        # filename = 'MDES0S0.p'


        # filename = 'MDEQsed.p'
        # filename = 'MDEdepth.p'
        filename = 'MDEwidth.p'




        with open(path + filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # print(self.data)
        self.experiments = self.data.v('experiments')
        self.label = self.data.v('label')
        self.color = self.data.v('color')
        self.nch = self.data.v('numberofchannels')

        # self.plot_NWT()
        # self.plot_NWT(mod='river')

        self.contours_cThatcf_width() 
        # self.contours_hatcf_width()
        # self.contours_cT_width()
        # self.contours_Tcomponents_width()
        # self.plot_NWTNST_width()

        # self.contours_cT_depth()
        # self.contours_hatcf_depth()
        # self.contours_cThatcf_depth()
        # self.plot_NWTNST_depth()
        # self.contour_FT_intersect()
        # self.contours_Tcomponents_depth()

        # self.contour_Tratio_depth()
        # self.contour_Tratio_width()


        # self.plot_NWTNST_Qsed()
        # self.plot_NST_Qsed()
        # self.contours_cT_Qsed()
        # self.contours_hatcf_Qsed()
        # self.contours_Tcomponents_Qsed()
        # self.contours_cf_Qsed()
        # self.ThatC_Qsed()
        # self.plot_NWTNST_Q()
        # self.contours_cT_Q()
        # self.contours_hatcf_Q()
        # self.contours_Tcomponents_Q()

        # self.contours_cT_SLR()
        # self.contours_hatcf_SLR()
        # self.contours_Tcomponents_SLR()
        # self.plot_NWTNST_SLR()


        # self.plot_NWTNST_S0()
            
        plt.show()

        

        d = {}
        return d


    def contour_Tratio_depth(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 20
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
        T = {}
        X = {}
        Y = {}
        for i in range(2):
            channel = str(i)
            T[channel] = [] 
            y = []
            for exp in self.experiments:
                Ttmp = self.data.v(exp, str(i), 'T')
                T[channel].append(-Ttmp)
                y.append(self.data.v(exp, '0', 'H'))
            x = -self.data.v(exp, channel, 'x') / 1000
            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        Tratio = np.asarray(T['0'])/np.asarray(T['1'])
        Tlo = np.min(Tratio)
        Thi = np.max(Tratio)
        shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')

        plt.figure(figsize=(10, 6.18))
        ax = plt.gca()
        c1 = ax.pcolor(X['0'], Y['0'], Tratio, cmap = shifted_cmap, vmin=Tlo, vmax=Thi)
        c2 = ax.contour(X['0'], Y['0'], Tratio, colors='gray', linewidths=0.5)
        ax.contour(X['0'], Y['0'], Tratio, levels=[1], colors='k')
        ax.clabel(c2, inline=True, fontsize=10, fmt='%1.2f')
        cb1 = plt.colorbar(c1)
        cb1.set_label('$T_1$ / $T_2$')
        plt.xlabel('Position (km)')
        plt.ylabel('$H_1$ (m)')
        plt.yticks([5, 7.8, 8.6, 11, 13, 15.4, 18, 20])
        plt.savefig("Tratio_depth.png", facecolor='w', edgecolor='w')

    def contour_Tratio_width(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 20
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
        T = {}
        X = {}
        Y = {}
        for i in range(2):
            channel = str(i)
            T[channel] = [] 
            y = []
            for exp in self.experiments:
                Ttmp = self.data.v(exp, str(i), 'T')
                T[channel].append(-Ttmp)
                y.append(self.data.v(exp, '0', 'B')[0])
            x = -self.data.v(exp, channel, 'x') / 1000
            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        Tratio = np.asarray(T['1'])/np.asarray(T['0'])
        Tlo = np.min(Tratio)
        Thi = np.max(Tratio)
        shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')

        plt.figure(figsize=(10, 6.18))
        ax = plt.gca()
        c1 = ax.pcolor(X['0'], Y['0'], Tratio, cmap = shifted_cmap, vmin=Tlo, vmax=Thi)
        c2 = ax.contour(X['0'], Y['0'], Tratio, colors='gray', linewidths=0.5)
        ax.contour(X['0'], Y['0'], Tratio, levels=[1], colors='k')
        ax.clabel(c2, inline=True, fontsize=10, fmt='%1.2f')
        cb1 = plt.colorbar(c1)
        cb1.set_label('$T_2$ / $T_1$')
        plt.xlabel('Position (km)')
        plt.ylabel('$B_1$ at sea (m)')
        plt.yticks([500, 750, 1000, 1250, 1500, 1750, 2000]) 
        plt.savefig("Tratio_width.png", facecolor='w', edgecolor='w')


    def plot_NWT(self, mod='baroc'):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qr0 = np.array([])
        Qr1 = np.array([])
        Qo0 = np.array([])
        Qo1 = np.array([])
     
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q', mod))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q', mod))

        x = np.linspace(5, 15, len(Q0))
        # x = np.linspace(7, 11, len(Q0))
        plt.plot(x, Q0, label='NP', color='b')
        plt.plot(x, Q1, label='SP', color='r')

        for exp in self.experiments:
            Qr0 = np.append(Qr0, self.data.v(exp, '0', 'Q', 'river'))
            Qr1 = np.append(Qr1, self.data.v(exp, '1', 'Q', 'river'))

        # x = np.linspace(7, 11, len(Q0))
        plt.plot(x, Qr0, ls='--', color='b')
        plt.plot(x, Qr1, ls='--', color='r')

        for exp in self.experiments:
            Qo0 = np.append(Qo0, self.data.v(exp, '0', 'Q')-self.data.v(exp, '0', 'Q', 'river')-self.data.v(exp, '0', 'Q', 'baroc'))
            Qo1 = np.append(Qo1, self.data.v(exp, '1', 'Q')-self.data.v(exp, '1', 'Q', 'river')-self.data.v(exp, '1', 'Q', 'baroc'))

        plt.plot(x, Qo0, ls='dotted', color='b')
        plt.plot(x, Qo1, ls='dotted', color='r')
        
        plt.xlabel('$H_{NP}$ (m)')
        # plt.ylabel('$Q_{%s}$ (m$^3$ s$^{-1}$)'%(mod))
        plt.ylabel('Net water transport (m$^3$ s$^{-1}$)')
        plt.legend()
        plt.text(7, -2650, 'solid: baroclinic',fontsize=15)
        plt.text(7, -2950, 'dashed: river',fontsize=15)
        plt.text(7, -3250, 'dotted: tidally rectified',fontsize=15)
        plt.xticks([5,7,9,11,13,15])

  

    def plot_NST_Qsed(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        # channels = [2, 3, 4]
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
            # for i in channels:
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

        # x = np.linspace(0, -80, len(Q0))
        x = np.linspace(0, 100, len(Q0))

 
        plt.plot(x, -Qs0, color='b', label='1')
        plt.plot(x, -Qs1, color='r', label='2')
        # plt.ylim([-800, -150])
        # plt.ylim([150, 800])
        # plt.ylim(-200, 2500)
        plt.legend()
        plt.xlabel('Fluvial sediment input (kg s$^{-1}$)')
        plt.ylabel('Net sediment transport (kg s$^{-1}$)')
        plt.savefig('NST_Qsed.pdf', dpi=None, facecolor='w', edgecolor='w')



    def plot_NWTNST_S0(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        for exp in self.experiments:
            print(exp)
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

        # x = np.linspace(0, -80, len(Q0))
        # x = np.linspace(0, 100, len(Q0))
        # x = np.linspace(0, 1500, len(Q0))
        # x = np.linspace(1000, -1000, len(Q0))
        x = np.linspace(20,30,len(Q0))
        plt.plot(x, Q0, label='NP', color='b')
        plt.plot(x, Q1, label='SP', color='r')
        # plt.ylim([-800, -150])
        # plt.ylim([150, 800])
        plt.legend(loc="center left")
        plt.xlabel('Salinity at sea (psu)')

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        # ax1.set_yticklabels(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
        # ax1.yticks([0, 1, 2], ['January', 'February', 'March'], rotation=45)  # Set text labels and properties.
        ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 
        ax2.plot(x, -Qs0, '--', color='b')
        ax2.plot(x, -Qs1, '--', color='r')
        # plt.ylim([-110, 20])

    def plot_NWTNST_Qsed(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

        # x = np.linspace(0, -80, len(Q0))
        # x = np.linspace(0, 100, len(Q0))
        x = np.linspace(0, 1500, len(Q0))
        # x = np.linspace(1000, -1000, len(Q0))
        plt.plot(x, Q0, label='NP', color='b')
        plt.plot(x, Q1, label='SP', color='r')
        # plt.ylim([-800, -150])
        # plt.ylim([150, 800])
        plt.legend(loc="center left")
        plt.xlabel('Fluvial sediment input (kg s$^{-1}$)')

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        ax1.set_yticks(np.rint([Q0[0], Q1[0]]))  # Set label locations.
        # ax1.set_yticklabels(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
        # ax1.yticks([0, 1, 2], ['January', 'February', 'March'], rotation=45)  # Set text labels and properties.
        ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 
        ax2.plot(x, -Qs0, '--', color='b')
        ax2.plot(x, -Qs1, '--', color='r')
        # plt.ylim([-110, 20])

    def contours_hatcf_Qsed(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        alpha1 = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            alpha1[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)
                alpha1[channel].append(self.data.v(exp, channel, 'alpha1'))
                y.append(self.data.v(exp, '0', 'Qsed'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Qsed'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        fhi=1
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        label = ['NP', 'SP', 'SC']
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            # xnew = np.linspace(loc[channel].min(), loc[channel].max(), 200)
            # spl = make_interp_spline(loc[channel], y, k=3)
            # y_smooth = spl(xnew)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('Fluvial sediment input (kg s$^{-1}$)')
            ax.text(0.4, 0.9, label[i], transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('f')
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=alpha1lo,  max_val=alpha1hi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            # c = ax.contour(X[channel], Y[channel], alpha1[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=alpha1lo, vmax=alpha1hi) 
            ax.text(0.4, 0.9, label[i], transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('Fluvial sediment input (kg s$^{-1}$)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')

    def contours_cT_Qsed(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)
        fsize=15
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f')
                f[channel].append(var)
                T[channel].append(self.data.v(exp, str(i), 'T'))
                y.append(self.data.v(exp, '0', 'Qsed'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'Qsed'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('Fluvial sediment input (kg s$^{-1}$)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('S (kg m$^{-2}$)')
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            # c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap,vmin=Tlo, vmax=Thi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('Fluvial sediment input (kg s$^{-1}$)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
                # cb2.set_ticks([-15, -10, -5, 0, 2])
                # cb2.set_ticklabels([-15, -10, -5, 0, 2])
                # cb2.ax.locator_params(nbins=6)
   

    def contours_cf_Qsed(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)
        fsize=15
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
            
        S = {}
        T = {}
        X = {}
        Y = {}
        locS = {}
        locf = {}
        Smax = {}
        fmax = {}
        # channels = [2,3,4]
        for i in range(3):
        # for i in channels:
            channel = str(i)
            S[channel] = [] 
            T[channel] = [] 
            locf[str(i)] = np.array([])
            locS[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            Smax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
                H = self.data.v(exp, str(i), 'H')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                stock = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f') / H
                S[channel].append(stock)
                T[channel].append(ftmp)
                y.append(self.data.v(exp, '0', 'Qsed'))

                index = np.r_[True, stock[1:] > stock[:-1]] & np.r_[stock[:-1] > stock[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    locS[str(i)] = np.append(locS[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    Smax[str(i)] = np.append(Smax[str(i)], self.data.v(exp, '0', 'Qsed'))
                else:
                    locS[str(i)] = np.append(locS[str(i)], np.nan) 
                    Smax[str(i)] = np.append(Smax[str(i)], np.nan)
                
                index = np.r_[True, ftmp[1:] > ftmp[:-1]] & np.r_[ftmp[:-1] > ftmp[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    locf[str(i)] = np.append(locf[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'Qsed'))
                else:
                    locf[str(i)] = np.append(locf[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)



            x = -self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(S.values())
        fhi = np.max(S.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        # label = ['NP', "SP", 'SC']
        for i in range(3):
        # for i in channels:
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], S[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(-locS[channel], Smax[channel], 'k')

            if i==0:
                ax.set_ylabel('Fluvial sediment input (kg s$^{-1}$)')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$\\bar{c}$ (kg m$^{-3}$)')
                # cb1.set_ticks([0, 0.02, 0.04, 0.06, 0.08])
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            # c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = 'YlOrBr', vmin=0, vmax=Thi) 

            # spline = make_interp_spline(locf[channel], fmax[channel])
            # x = np.linspace(locf[channel].min(), locf[channel].max(), 100)
            # y = spline(x)
            # ax.plot(x, y, 'k' )
            ax.plot(-locf[channel], fmax[channel], 'k')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('Fluvial sediment input (kg s$^{-1}$)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('f')
                tick = [0, 0.2, 0.4, 0.6, 0.8, 1]
                # cb2.set_ticks(tick)
                # cb2.set_ticklabels(tick)
        plt.savefig('cf_Qsed.png', facecolor='w', edgecolor='w')


    def contours_Tcomponents_Qsed(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                Triver = self.data.v(exp, str(i), 'T', 'river')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')

                # v1 = B * Triver * ftmp
                # v2 = B * Tbaroc * ftmp
                v1 = Triver
                v2 = Tbaroc

                f[channel].append(v1)
                T[channel].append(v2)
                y.append(self.data.v(exp, '0', 'H'))

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)
            ax = plt.subplot(2, 3, i+1) 
            vmax = (np.max(f.values()))
            vmin = (np.min(f.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin, max_val=vmax, name='shifted')
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax)
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i == 0:
                ax.set_ylabel('Depth of channel 1 (m)')
            if i == 2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T$ river (kg m$^{-1}$ s$^{-1}$)')
                cb1.set_ticks([-4, -3, -2, -1, 0])
                cb1.set_ticklabels([-4, -3, -2, -1, 0])

            vmax = (np.max(T.values()))
            vmin = (np.min(T.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i == 0:
                plt.ylabel('Depth of channel 1 (m)')
            if i == 2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ baroc (kg m$^{-1}$ s$^{-1}$)')


    def plot_NWTNST_Q(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=18
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)


        x = np.linspace(300, 1000, len(Q0))
        plt.plot(x, -Q0, label='1', color='b')
        plt.plot(x, -Q1, label='2', color='r')
        # plt.ylim([-650, -300])
        plt.legend()
        plt.xlabel('River discharge (m$^{3}$ s$^{-1}$)')

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        # ax1.set_yticks(np.rint([-Q0[0], -Q1[0]]))  # Set label locations.
        # ax1.set_yticklabels(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
        # ax1.yticks([0, 1, 2], ['January', 'February', 'March'], rotation=45)  # Set text labels and properties.
        ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 
        ax2.plot(x, Qs0, '--', color='b')
        ax2.plot(x, Qs1, '--', color='r')
        # plt.ylim([-110, 20])

    def contours_hatcf_Q(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        alpha1 = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            alpha1[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)
                alpha1[channel].append(self.data.v(exp, channel, 'alpha1'))
                y.append(self.data.v(exp, '2', 'Q'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Q'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('River discharge (m$^3$ s$^{-1}$)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('f')
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=alpha1lo,  max_val=alpha1hi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            # c = ax.contour(X[channel], Y[channel], alpha1[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=alpha1lo, vmax=alpha1hi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('River discharge (m$^3$ s$^{-1}$)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')

    def contours_cT_Q(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)

        # self.data.v(exp, '2', 'Q')
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f')
                f[channel].append(var)
                T[channel].append(self.data.v(exp, str(i), 'T'))
                y.append(self.data.v(exp, '2', 'Q'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Q'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            # X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))
            # y = np.linspace((0, 2, ))
            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('River discharge (m$^3$ s$^{-1}$)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('S (kg m$^{-2}$)')
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap,vmin=Tlo, vmax=Thi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('River discharge (m$^3$ s$^{-1}$)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
                # cb2.set_ticks([-15, -10, -5, 0, 2])
                # cb2.set_ticklabels([-15, -10, -5, 0, 2])
                # cb2.ax.locator_params(nbins=6)
   
    def contours_Tcomponents_Q(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                Triver = self.data.v(exp, str(i), 'T', 'river')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')

                # v1 = B * Triver * ftmp
                # v2 = B * Tbaroc * ftmp
                v1 = Triver
                v2 = Tbaroc

                f[channel].append(v1)
                T[channel].append(v2)
                y.append(self.data.v(exp, '2', 'Q'))

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)
            ax = plt.subplot(2, 3, i+1) 
            vmax = (np.max(f.values()))
            vmin = (np.min(f.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin, max_val=vmax, name='shifted')
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax)
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i == 0:
                ax.set_ylabel('River discharge (m$^3$ s$^{-1}$)')
            if i == 2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T$ river (kg m$^{-1}$ s$^{-1}$)')
                cb1.set_ticks([-4, -3, -2, -1, 0])
                cb1.set_ticklabels([-4, -3, -2, -1, 0])

            vmax = (np.max(T.values()))
            vmin = (np.min(T.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i == 0:
                plt.ylabel('River discharge (m$^3$ s$^{-1}$)')
            if i == 2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ baroc (kg m$^{-1}$ s$^{-1}$)')



    def plot_NWTNST_width(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=18
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        # channels = [2,3,4]
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
            # for i in channels:
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * T * f + B * F * np.gradient(f, xx, edge_order=2)))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

        # x = np.linspace(500, 2000, len(Q0))
        x = np.linspace(500,2000, len(Q0))
        
        # plt.plot(x, Q0, label='1', color='b')
        # plt.plot(x, Q1, label='2', color='r')
        # plt.legend()
        plt.xlabel('$B_{1}$ at sea (m)')


        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        # ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 

        # c = np.linspace(0, 25, len(Q0))
        plt.plot(x, -Qs0,  color='b', label='1')
        plt.plot(x, -Qs1, color='r', label='2')
        plt.xticks([500, 750, 1000, 1250, 1500, 1750, 2000])
        # ax1.set_ylim([0, 1000])
        # ax2.set_ylim([-10,60])
        # ax1.set_yticks([0, 250, 500, 750, 1000])
        plt.yticks([-10, 0, 10, 20, 30,  40, 50, 60])
        plt.ylabel('Net sediment transport (kg s$^{-1}$)')
        plt.legend()
        # plt.ylim()
        # plt.savefig('NWTNST_width.pdf', dpi=None, facecolor='w', edgecolor='w')
        plt.savefig('NST_width.pdf', dpi=None, facecolor='w', edgecolor='w')
    
    def contours_hatcf_width(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        alpha1 = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            alpha1[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)
                alpha1[channel].append(self.data.v(exp, channel, 'alpha1'))
                y.append(self.data.v(exp, '0', 'B')[0])

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                # index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'B')[0])
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        tick = [500.0, 1000.0, 1500.0, 2000.0]
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')
            ax.set_yticks(tick)

            if i==0:
                ax.set_ylabel('Width of channel 1 at sea (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('f')
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=alpha1lo,  max_val=alpha1hi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], alpha1[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=alpha1lo, vmax=alpha1hi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            ax.set_yticks(tick)
            if i==0:
                plt.ylabel('Width of channel 1 at sea (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')

    def contours_cT_width(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                # B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f')
                f[channel].append(var)
                T[channel].append(self.data.v(exp, str(i), 'T'))
                y.append(self.data.v(exp, '0', 'B')[0])

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'B')[0])
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        tick = [500.0, 1000.0, 1500.0, 2000.0]
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('Width of channel 1 at sea (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_yticks(tick)
            # ax.set_yticklabels(tick)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('S (kg m$^{-2}$)')
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap,vmin=Tlo, vmax=Thi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            ax.set_yticks(tick)
            # ax.set_yticklabels(tick)
            if i==0:
                plt.ylabel('Width of channel 1 at sea (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
                ticks = [-1.2, -0.6, 0, 0.6]
                cb2.set_ticks(ticks)
                cb2.set_ticklabels(ticks)
                # cb2.ax.locator_params(nbins=6)
   
    def contours_Tcomponents_width(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                Triver = self.data.v(exp, str(i), 'T', 'river')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')

                # v1 = B * Triver * ftmp
                # v2 = B * Tbaroc * ftmp
                v1 = Triver
                v2 = Tbaroc

                f[channel].append(v1)
                T[channel].append(v2)
                y.append(self.data.v(exp, '0', 'B')[0])

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)
            ax = plt.subplot(2, 3, i+1) 
            vmax = (np.max(f.values()))
            vmin = (np.min(f.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin, max_val=vmax, name='shifted')
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax)
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i == 0:
                ax.set_ylabel('Depth of channel 1 (m)')
            if i == 2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T$ river (kg m$^{-1}$ s$^{-1}$)')
                cb1.set_ticks([-4, -3, -2, -1, 0])
                cb1.set_ticklabels([-4, -3, -2, -1, 0])

            vmax = (np.max(T.values()))
            vmin = (np.min(T.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i == 0:
                plt.ylabel('Width of channel 1 at sea (m)')
            if i == 2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ baroc (kg m$^{-1}$ s$^{-1}$)')

      

     

    def plot_NWTNST_depth(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=20
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        # channels = [2,3,4]
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
            # for i in channels:
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

        x = np.linspace(5, 20, len(Q0))
        # x = np.linspace(7, 11, len(Q0))
        # plt.plot(x, Q0, label='1', color='b')
        # plt.plot(x, Q1, label='2', color='r')
        
        plt.xlabel('$H_{1}$ (m)')

        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        # ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 

        # c = np.linspace(0,20, len(Q0))
        # plt.plot(x, -Qs0 , '--', color='b')
        # plt.plot(x, -Qs1 , '--', color='r')
        plt.plot(x, -Qs0, label='1', color='b')
        plt.plot(x, -Qs1, label='2', color='r')
        # ax1.legend()
        # ax1.set_xticks([5, 8, 11, 13, 14, 17, 20])
        # ax1.set_ylim([-500, 1500])
        # ax2.set_ylim([-75, 125])
        plt.legend()
        plt.xticks([5, 7.8, 11, 13, 15.4, 18, 20])
        plt.hlines(0, 5, 20)
        plt.ylabel('Net sediment transport (kg s$^{-1}$)')
        # plt.savefig('NWTNST_depth.pdf', facecolor='w', edgecolor='w')
        plt.savefig('NST_depth.pdf', facecolor='w', edgecolor='w')


    def contour_FT_intersect(self):
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.07
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                Ttmp = self.data.v(exp, str(i), 'T')
                flux = B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)
                tp = Ttmp - flux / B / ftmp

                T[channel].append(tp)
                y.append(self.data.v(exp, '0', 'H'))

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

  
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
        fig1, axs1 = plt.subplots(1, 3, figsize=(12, 3.7), edgecolor='w')
        ytick = [5, 10, 15, 20]
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(1, 3, i+1) 
            ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k', linewidths=2)
            c1 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=Tlo, vmax=Thi)
            # ax.plot(loc[channel], fmax[channel], 'k')
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)


            if i==0:
                ax.set_ylabel('Depth of channel 1 (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T - \mathcal{F}/(Bf)$ (kg m$^{-1}$ s$^{-1}$)')
                # tick = [0, 0.5, 1, 1.5, 2]
                # cb1.set_ticks(tick)
                # cb1.set_ticklabels(tick)
                # cb1.ax.locator_params(nbins=len(tick))
            ax.set_xlabel('Position (km)')


    def contours_hatcf_depth(self, axis2='H'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        alpha1 = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            alpha1[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)
                alpha1[channel].append(self.data.v(exp, channel, 'alpha1'))
                y.append(self.data.v(exp, '0', axis2))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'H'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        ytick = [5, 10, 15, 20]

        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=0, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')
            # ax.set_yticks(ytick)
            # ax.set_yticklabels(ytick)
            if i==0:
                ax.set_ylabel('Depth of channel 1 (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('f')
                # tick = [0, 0.05, 0.1, 0.15, 0.2]
                # cb1.set_ticks(tick)
                # cb1.set_ticklabels(tick)
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=alpha1lo,  max_val=alpha1hi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], alpha1[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=0, vmax=alpha1hi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('Depth of channel 1 (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')
                # tick = [0, 10, 20, 30]
                # cb2.set_ticks(tick)
                # cb2.set_ticklabels(tick)

    def contours_cT_depth(self, axis2='H'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        fsize=15
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                Ttmp = self.data.v(exp, str(i), 'T')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B))#[-1]
                tp = ((B * Ttmp * ftmp ) / (B*ftmp))

                var = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f')
                f[channel].append(var)
                # T[channel].append(Ttemp)
                T[channel].append(tp)
                y.append(self.data.v(exp, '0', axis2))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'H'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
        print(flo, fhi)
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        ytick = [5, 10, 15, 20]
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=0, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)


            if i==0:
                ax.set_ylabel('Depth of channel 1 (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('S (kg m$^{-2}$)')
                tick = [0, 0.5, 1, 1.5, 2]
                cb1.set_ticks(tick)
                cb1.set_ticklabels(tick)
                cb1.ax.locator_params(nbins=len(tick))
  
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap,vmin=Tlo, vmax=Thi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('Depth of channel 1 (m)')
                
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
                # cb2.set_ticks([-18, -12, -6, 0, 2])
                # cb2.set_ticklabels([-18, -12, -6, 0, 2])
                cb2.ax.locator_params(nbins=6)
   
    def contours_cThatcf_width(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        fsize=15
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
        f = {}
        alpha1 = {}
        S = {}
        T = {}
        X = {}
        Y = {}
        locf = {}
        locf2 = {}
        locS = {}
        fmax = {}
        Smax = {}
        # channels = [2,3,4]
        
        for i in range(3):
        # for i in channels:
            channel = str(i)
            f[channel] = [] 
            S[channel] = [] 
            T[channel] = [] 
            alpha1[channel] = [] 
            locf[str(i)] = np.array([])
            locf2[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            locS[str(i)] = np.array([])
            Smax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
                F = self.data.v(exp, str(i), 'F')
                Ttmp = self.data.v(exp, str(i), 'T')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                alpha1tmp = self.data.v(exp, channel, 'alpha1')
                Stmp = ftmp * alpha1tmp / 13 # depth = 13 m

                f[channel].append(ftmp)
                alpha1[channel].append(alpha1tmp)
                S[channel].append(Stmp)
                T[channel].append(-Ttmp)

                y.append(self.data.v(exp, '0', 'B')[0] / 1000)
                # print(self.data.v(exp, '0', 'B'))

                indexf = np.r_[True, ftmp[1:] > ftmp[:-1]] & np.r_[ftmp[:-1] > ftmp[1:], True]
                indexf[0] = False
                indexf[-1] = False

                if sum(indexf) > 0 :
                    locf[str(i)] = np.append(locf[str(i)], self.data.v(exp, channel, 'x')[indexf][0]/ 1000) 
                    locf2[str(i)] = np.append(locf2[str(i)], self.data.v(exp, channel, 'x')[indexf][-1]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'B')[0] / 1000)
                else:
                    locf[str(i)] = np.append(locf[str(i)], np.nan) 
                    locf2[str(i)] = np.append(locf2[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
                
                indexS = np.r_[True, Stmp[1:] > Stmp[:-1]] & np.r_[Stmp[:-1] > Stmp[1:], True]
                indexS[0] = False
                indexS[-1] = False

                if sum(indexS) > 0 :
                    locS[str(i)] = np.append(locS[str(i)], self.data.v(exp, channel, 'x')[indexS][0]/ 1000) 
                    Smax[str(i)] = np.append(Smax[str(i)], self.data.v(exp, '0', 'B')[0] / 1000)
                else:
                    locS[str(i)] = np.append(locS[str(i)], np.nan) 
                    Smax[str(i)] = np.append(Smax[str(i)], np.nan)

            x = -self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Slo = np.min(S.values())
        Shi = np.max(S.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        # ytick = [0.5, 1, 1.5, 2]
        # ytick = [3.5,4.5,5.5,6.5,7.5,8.5]
        Stick = [0, 0.4, 0.8, 1.2, 1.6]
        ftick = [0, 0.05, 0.1, 0.15,  0.2]
        # label = ['NP', 'SP', 'SC']

        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 10), edgecolor='w')
        for i in range(3):
        # for i in channels:
            channel = str(i)

            ax = plt.subplot(4, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], S[channel], cmap = 'YlOrBr', vmin=0, vmax=Shi)
            cp = ax.contour(X[channel], Y[channel], S[channel], colors='gray', linewidths=0.5)
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            ax.plot(-locS[channel], Smax[channel], 'k')
            # ax.set_yticks(ytick)
            # ax.set_yticklabels(ytick)
            if i==0:
                # ax.set_ylabel('$B_1$ at sea (km)')
                ax.set_ylabel('$B_{1}$ at sea (km)')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i==2:
                cb1 = plt.colorbar(c1)
                # cb1.set_label('S (kg m$^{-2}$)')
                cb1.set_label('$\\bar{c}$ (kg m$^{-3}$)')
                # cb1.set_ticks(Stick)
                # cb1.set_ticklabels(Stick)
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(4, 3, i+10)
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=Tlo, vmax=Thi) 
            cp =  ax.contour(X[channel], Y[channel], T[channel], colors='gray', linewidths=0.5) 
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k') 
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')

            
            # ax.set_yticks(ytick)
            # ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('$B_{1}$ at sea (km)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')

            ax = plt.subplot(4, 3, i+7)
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=0, vmax=fhi)
            cp = ax.contour(X[channel], Y[channel], f[channel], colors='gray', linewidths=0.5)
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            # ax.plot(-locf[channel][22:34], fmax[channel][22:34], 'k')
            # ax.plot(-locf[channel], fmax[channel], 'k')
            ax.plot(-locf2[channel], fmax[channel], 'k')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            # ax.set_yticks(ytick)
            # ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('$B_{1}$ at sea (km)')
            if i==2:
                cb2 = plt.colorbar(c1)
                cb2.set_label('$f$')
                cb2.set_ticks(ftick)
                cb2.set_ticklabels(ftick)

            ax = plt.subplot(4, 3, i+4)
            c1 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=0, vmax=alpha1hi)
            cp = ax.contour(X[channel], Y[channel], alpha1[channel], colors='gray', linewidths=0.5)
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            # ax.set_yticks(ytick)
            # ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('$B_{1}$ at sea (km)')
            if i==2:
                cb2 = plt.colorbar(c1)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')
        plt.savefig("cTfhatC_width.png", facecolor='w', edgecolor='w')


    def contours_cThatcf_depth(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        fsize=18
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize)
        f = {}
        alpha1 = {}
        S = {}
        T = {}
        X = {}
        Y = {}
        locf = {}
        locf2 = {}
        locS = {}
        fmax = {}
        Smax = {}
        # x4p = {}
        # channels = [2, 3, 4]
        for i in range(3):
        # for i in channels:
            channel = str(i)
            f[channel] = [] 
            S[channel] = [] 
            T[channel] = [] 
            alpha1[channel] = [] 
            locf[str(i)] = np.array([])
            locf2[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            locS[str(i)] = np.array([])
            Smax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
                F = self.data.v(exp, str(i), 'F')
                Ttmp = self.data.v(exp, str(i), 'T')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                alpha1tmp = self.data.v(exp, channel, 'alpha1')
                H = self.data.v(exp, str(i), 'H')
                # Stmp = ftmp * alpha1tmp 
                Stmp = ftmp * alpha1tmp / H

                f[channel].append(ftmp)
                alpha1[channel].append(alpha1tmp)
                S[channel].append(Stmp)
                T[channel].append(-Ttmp)

                y.append(self.data.v(exp, '0', 'H'))

                indexf = np.r_[True, ftmp[1:] > ftmp[:-1]] & np.r_[ftmp[:-1] > ftmp[1:], True]
                indexf[0] = False
                indexf[-1] = False

                if sum(indexf) > 0 :
                    locf[str(i)] = np.append(locf[str(i)], self.data.v(exp, channel, 'x')[indexf][0]/ 1000) 
                    locf2[str(i)] = np.append(locf2[str(i)], self.data.v(exp, channel, 'x')[indexf][-1]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'H'))
                else:
                    locf[str(i)] = np.append(locf[str(i)], np.nan) 
                    locf2[str(i)] = np.append(locf2[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
                
                indexS = np.r_[True, Stmp[1:] > Stmp[:-1]] & np.r_[Stmp[:-1] > Stmp[1:], True]
                indexS[0] = False
                indexS[-1] = False

                if sum(indexS) > 0 :
                    locS[str(i)] = np.append(locS[str(i)], self.data.v(exp, channel, 'x')[indexS][0]/ 1000) 
                    Smax[str(i)] = np.append(Smax[str(i)], self.data.v(exp, '0', 'H'))
                else:
                    locS[str(i)] = np.append(locS[str(i)], np.nan) 
                    Smax[str(i)] = np.append(Smax[str(i)], np.nan)

            x = -self.data.v(exp, channel, 'x') / 1000
            # x4p[channel] = x

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))


        flo = np.min(f.values())
        fhi = np.max(f.values())
        Slo = np.min(S.values())
        Shi = np.max(S.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        # ytick = [5, 7,  11, 15, 20]
        # ytick = [5, 7, 9, 11, 13, 15]
        ytick = [5, 7, 9,  11, 13,  15, 17, 20]
        # ytick = [7,8,9,10,11]
        Stick = [0, 0.5, 1, 1.5, 2]
        # ftick = [0, 0]

        # sppp=aaas
        lw = 0.5
        # label = ['NP', 'SP', 'SC']
        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 10), edgecolor='w')
        for i in range(3):
        # for i in channels:
            channel = str(i)

            ax = plt.subplot(4, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], S[channel], cmap = 'YlOrBr', vmin=0, vmax=Shi)
            cp = ax.contour(X[channel], Y[channel], S[channel], colors='gray', linewidths=0.5)
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            ax.plot(-locS[channel], Smax[channel], 'k')
            ax.hlines(11, 
            np.min(-self.data.v(exp, channel, 'x') / 1000), 
            np.max( -self.data.v(exp, channel, 'x') / 1000), 
            color='w',linewidths=0.5)

            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i==0:
                ax.set_ylabel('$H_{1}$ (m)')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i==2:
                cb1 = plt.colorbar(c1)
                # cb1.set_label('S (kg m$^{-2}$)')
                cb1.set_label('$\\bar{c}$ (kg m$^{-3}$)')
                # cb1.set_ticks(Stick)
                # cb1.set_ticklabels(Stick)
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(4, 3, i+10)
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=Tlo, vmax=Thi) 
            cp = ax.contour(X[channel], Y[channel], T[channel], colors='gray', linewidths=0.5)   
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i==0:
                plt.ylabel('$H_{1}$ (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
            ax.set_xlabel('Position (km)')

            ax = plt.subplot(4, 3, i+7)
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=0, vmax=fhi)
            cp = ax.contour(X[channel], Y[channel], f[channel], colors='gray', linewidths=0.5)   
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            # ax.plot(-locf[channel][15:19], fmax[channel][15:19], 'k')
            # ax.plot(-locf[channel][22:29], fmax[channel][22:29], 'k')
            ax.plot(-locf[channel], fmax[channel], 'k')
            ax.plot(-locf2[channel], fmax[channel], 'k')
            # ax.hlines(11, 
            # np.min(-self.data.v(exp, channel, 'x') / 1000), 
            # np.max( -self.data.v(exp, channel, 'x') / 1000), 
            # color='w',linewidths=0.5)
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_yticks(ytick)
            if i==0:
                plt.ylabel('$H_{1}$ (m)')
            if i==2:
                cb2 = plt.colorbar(c1)
                cb2.set_label('$f$')
                # cb2.set_ticks(ftick)
                # cb2.set_ticklabels(ftick)

            ax = plt.subplot(4, 3, i+4)
            c1 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=0, vmax=alpha1hi)
            cp = ax.contour(X[channel], Y[channel], alpha1[channel], colors='gray', linewidths=0.5)   
            ax.clabel(cp, inline=True, fontsize=10, fmt='%1.2f')
            ax.text(0.35, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            
            ax.set_yticks(ytick)
            if i==0:
                plt.ylabel('$H_{1}$ (m)')
            if i==2:
                cb2 = plt.colorbar(c1)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')
        plt.savefig("cTfhatC_depth.png", facecolor='w', edgecolor='w')


    def contours_Tcomponents_depth(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                Triver = self.data.v(exp, str(i), 'T', 'river')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')

                # v1 = B * Triver * ftmp
                # v2 = B * Tbaroc * ftmp
                v1 = Triver
                v2 = Tbaroc

                f[channel].append(v1)
                T[channel].append(v2)
                y.append(self.data.v(exp, '0', 'H'))

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7.4), edgecolor='w')
        ytick = [5, 10, 15, 20]
        for i in range(3):
            channel = str(i)
            ax = plt.subplot(2, 3, i+1) 
            vmax = (np.max(f.values()))
            vmin = (np.min(f.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin, max_val=vmax, name='shifted')
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax)
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i == 0:
                ax.set_ylabel('Depth of channel 1 (m)')
            if i == 2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T$ river (kg m$^{-1}$ s$^{-1}$)')
                # tick = [-6, -5, -4, -3, -2, -1, 0]
                # cb1.set_ticks(tick)
                # cb1.set_ticklabels(tick)

            vmax = (np.max(T.values()))
            vmin = (np.min(T.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            ax.set_yticks(ytick)
            ax.set_yticklabels(ytick)
            if i == 0:
                plt.ylabel('Depth of channel 1 (m)')
            if i == 2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ baroc (kg m$^{-1}$ s$^{-1}$)')

      


    def contours_hatcf_SLR(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        alpha1 = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            alpha1[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)
                alpha1[channel].append(self.data.v(exp, channel, 'alpha1'))
                y.append(self.data.v(exp, '2', 'H') - self.data.v(self.experiments[0], '2', 'H'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'H') - self.data.v(self.experiments[0], '2', 'H'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        flo = np.min(f.values())
        fhi = np.max(f.values())
        alpha1lo = np.min(alpha1.values())
        alpha1hi = np.max(alpha1.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('Sea level rise (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('f')
  
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=alpha1lo,  max_val=alpha1hi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            # c = ax.contour(X[channel], Y[channel], alpha1[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], alpha1[channel], cmap = 'YlOrBr', vmin=alpha1lo, vmax=alpha1hi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('Sea level rise (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$\hat{C}$ (kg m$^{-2}$)')

    def contours_cT_SLR(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        # y = np.linspace(0, 2, len(self.experiments))
        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []
            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')
         
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                var = self.data.v(exp, channel, 'alpha1') * self.data.v(exp, channel, 'f')
                f[channel].append(var)
                T[channel].append(self.data.v(exp, str(i), 'T'))
                y.append(self.data.v(exp, '2', 'H') - self.data.v(self.experiments[0], '2', 'H'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0 :
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'H') - self.data.v(self.experiments[0], '2', 'H'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000
            
            # X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))
            X[channel], Y[channel] = np.meshgrid(x, y)

        flo = np.min(f.values())
        fhi = np.max(f.values())
        Tlo = np.min(T.values())
        Thi = np.max(T.values())
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(2, 3, i+1) 
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr', vmin=flo, vmax=fhi)
            ax.plot(loc[channel], fmax[channel], 'k')

            if i==0:
                ax.set_ylabel('Sea level rise (m)')
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
          
            if i==2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('S (kg m$^{-2}$)')
  
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=Tlo,  max_val=Thi, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap,vmin=Tlo, vmax=Thi) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i==0:
                plt.ylabel('Sea level rise (m)')
            if i==2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ (kg m$^{-1}$ s$^{-1}$)')
                # cb2.set_ticks([-15, -10, -5, 0, 2])
                # cb2.set_ticklabels([-15, -10, -5, 0, 2])
                # cb2.ax.locator_params(nbins=6)
   
    def contours_Tcomponents_SLR(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.wspace'] = 0.2
        plt.rcParams['figure.subplot.hspace'] = 0.2
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                Triver = self.data.v(exp, str(i), 'T', 'river')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')
                Tadv = self.data.v(exp, str(i), 'T', 'adv')
                Tsedadv = self.data.v(exp, str(i), 'T', 'sedadv')
                Tstokes = self.data.v(exp, str(i), 'T', 'stokes')
                Tbaroc = self.data.v(exp, str(i), 'T', 'baroc')

                # v1 = B * Triver * ftmp
                # v2 = B * Tbaroc * ftmp
                v1 = Triver
                v2 = Tbaroc

                

                f[channel].append(v1)
                T[channel].append(v2)
                y.append(self.data.v(exp, '2', 'H') - self.data.v(self.experiments[0], '2', 'H'))

            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7.4), edgecolor='w')
        for i in range(3):
            channel = str(i)
            ax = plt.subplot(2, 3, i+1) 
            vmax = (np.max(f.values()))
            vmin = (np.min(f.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin, max_val=vmax, name='shifted')
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax)
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)

            if i == 0:
                ax.set_ylabel('Sea level rise (m)')
            if i == 2:
                cb1 = plt.colorbar(c1)
                cb1.set_label('$T$ river (kg m$^{-1}$ s$^{-1}$)')
                # cb1.set_ticks([-4, -3, -2, -1, 0])
                # cb1.set_ticklabels([-4, -3, -2, -1, 0])

            vmax = (np.max(T.values()))
            vmin = (np.min(T.values()))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(2, 3, i+4)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap, vmin=vmin, vmax=vmax) 
            ax.text(0.4, 0.9, 'channel '+str(i+1), transform=ax.transAxes, fontsize=15)
            ax.set_xlabel('Position (km)')
            if i == 0:
                plt.ylabel('Sea level rise (m)')
            if i == 2:
                cb2 = plt.colorbar(c2)
                cb2.set_label('$T$ baroc (kg m$^{-1}$ s$^{-1}$)')

    def plot_NWTNST_SLR(self):
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams['figure.subplot.bottom'] = 0.12
        fsize=18
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=fsize) 
        plt.rc('ytick', labelsize=fsize) 
        plt.rc('legend', fontsize=fsize-3)
        plt.rc('axes', labelsize=fsize) 
        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        Qs0 = np.array([])
        Qs1 = np.array([])
        for exp in self.experiments:
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            for i in range(2):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)


        x = np.linspace(0, 2, len(Q0))
        plt.plot(x, -Q0, label='1', color='b')
        plt.plot(x, -Q1, label='2', color='r')
        # plt.ylim([-650, -300])
        plt.legend()
        plt.xlabel('Sea level rise (m)')

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_ylabel('Solid: net water transport (m$^3$ s$^{-1}$)')
        # ax1.set_yticks(np.rint([-Q0[0], -Q1[0]]))  # Set label locations.
        # ax1.set_yticklabels(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
        # ax1.yticks([0, 1, 2], ['January', 'February', 'March'], rotation=45)  # Set text labels and properties.
        ax2.set_ylabel('Dashed: net sediment transport (kg s$^{-1}$)') 
        ax2.plot(x, Qs0, '--', color='b')
        ax2.plot(x, Qs1, '--', color='r')
        # plt.ylim([-110, 20])





    def contours_fT_depth(self, axis1='x', axis2='H'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.15
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                # trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B))#[-1]
                tp = ((B * Ttmp * ftmp ) / (B*ftmp))

                # jmax = self.data.v(exp, str(i), 'jmax')
                # var = self.data.v(exp, channel, 'f')
                var = np.real(self.data.v(exp, channel, 'cb'))
                f[channel].append(var)
                # f[channel].append(np.abs(self.data.v(exp, channel, 'zeta0', range(jmax+1), 0)))

                # T[channel].append(Ttmp -trans)
                T[channel].append(Ttmp)
                y.append(self.data.v(exp, '0', axis2))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                # index[-1] = False

                if sum(index) > 0 :
                    
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'H'))
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))


        
        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel], cmap = 'YlOrBr' )
            ax.plot(loc[channel], fmax[channel], 'k')

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' f')
            # cb1.set_label('channel ' + str(i+1) + ' $c_b$')

            vmax = (np.max(T[channel]))
            vmin = (np.min(T[channel]))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap) #,vmin=-vmax, vmax=vmax) 

            plt.ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')



    def contours_fT_width(self, axis1='x'):    
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((Ttmp * ftmp +  F * np.gradient(ftmp, xx, edge_order=2))/ (ftmp))#[-1]

                # var = self.data.v(exp, channel, 'f')
                var = np.real(self.data.v(exp, channel, 'cb'))
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(trans)
                # T[channel].append(Ttmp)
                y.append(self.data.v(exp, '0', 'B')[0])

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                # index[-1] = False

                if sum(index) > 0 :
                    
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '0', 'B')[0])
                else:
                    loc[str(i)] = np.append(loc[str(i)], np.nan) 
                    fmax[str(i)] = np.append(fmax[str(i)], np.nan)
            x = self.data.v(exp, channel, 'x') / 1000
            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))
        
        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])

            # fs = make_interp_spline(loc[channel], fmax[channel])

            # xs = np.linspace(loc[channel][0], loc[channel][-1], 100)
            # ys = fs(xs)
            ax.plot(loc[channel], fmax[channel], 'k')

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Width (m)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            # cb1.set_label('channel ' + str(i+1) + ' f')
            cb1.set_label('channel ' + str(i+1) + ' $c_b$')

            vmax = (np.max(T[channel]))
            vmin = (np.min(T[channel]))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')

            ax = plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap = shifted_cmap) 
            plt.ylabel('Width (m)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')



    def contours_fT_Q(self, axis1='x'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.15
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                # var = self.data.v(exp, channel, 'f')
                var = self.data.v(exp, channel, 'cb')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                y.append(self.data.v(exp, '2', 'Q'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Q'))
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])
            ax.plot(loc[channel], fmax[channel], 'k')

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Discharge (m$^3$/s)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            # cb1.set_label('channel ' + str(i+1) + ' f')
            cb1.set_label('channel ' + str(i+1) + ' cb')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Discharge (m$^3$/s)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')


    def contours_fT_Qsed(self, axis1='x'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.15
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                # var = self.data.v(exp, channel, 'f')
                var = self.data.v(exp, channel, 'cb')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                y.append(self.data.v(exp, '2', 'Qsed'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1] = False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index][0]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Qsed'))
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])
            ax.plot(loc[channel], fmax[channel], 'k')

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Qsed (m$^3$/s)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            # cb1.set_label('channel ' + str(i+1) + ' f')
            cb1.set_label('channel ' + str(i+1) + ' cb')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Qsed (m$^3$/s)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')
    def contours_fT_length(self, axis1='x'):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                L = xx[-1] - xx[0]
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                x2 = self.data.v(exp, '1', 'x')
                y.append((x2[-1]-x2[0]) / 1000) 

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], (x2[-1]-x2[0]) / 1000)
            
            x = self.data.v(exp, channel, 'x') / 1000
            # y = self.data.v()

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        for i in range(len(self.experiments)):

            X['1'][i] = self.data.v(self.experiments[i], '1','x')

        # a = b

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])
            ax.plot(loc[channel], fmax[channel], 'k')

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Length (km)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' f')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Length (km)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')

    def contours_fT_SLR(self, axis1='x'):

        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:

                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                # var = self.data.v(exp, channel, 'f')
                var = self.data.v(exp, channel, 'cb')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                y.append(self.data.v(exp, channel, 'H'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                index[-1]=False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, channel, 'H'))
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))

        # a=b

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            ax.plot(loc[channel], fmax[channel], 'k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' f')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')

    def contours_fT_M2Amp(self, axis1='x'):

        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        y = np.linspace(0.5, 1.5, len(self.experiments))

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
 
            count = 0
            for exp in self.experiments:
                
                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                # y.append(self.data.v(exp, channel, 'H'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], y[count])
                
                count += 1
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, y)

        # a=b

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            ax.plot(loc[channel], fmax[channel], 'k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Multiple of tidal amplitudes')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' f')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Multiple of tidal amplitudes')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')

    def contours_fT_M4Pha(self, axis1='x'):

        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        y = np.linspace(0, 2 * np.pi, len(self.experiments))

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
 
            count = 0
            for exp in self.experiments:
                
                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]

                var = self.data.v(exp, channel, 'f')
                f[channel].append(var)

                T[channel].append(Ttmp - trans)
                # T[channel].append(Ttmp)
                # y.append(self.data.v(exp, channel, 'H'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                if sum(index) > 0:
                    loc[str(i)] = np.append(loc[str(i)], self.data.v(exp, channel, 'x')[index]/ 1000) 
                    fmax[str(i)] = np.append(fmax[str(i)], y[count])
                
                count += 1
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, y)

        # a=b

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            ax.plot(loc[channel], fmax[channel], 'k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Phase difference')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' f')


            vmax = (np.max(T[channel]))
            vmin = (np.min(T[channel]))
            shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel], cmap=shifted_cmap) 

            plt.ylabel('Phase difference')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $T-\mathcal{F}/(Bf)$')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')

    def contours_T_M4Pha(self, axis1='x'):

        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.1
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}
        y = np.linspace(0, 2 * np.pi, len(self.experiments))
        # y = np.linspace(5,18, len(self.experiments))

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
 
            count = 0
            for exp in self.experiments:
                
                B = self.data.v(exp, str(i), 'B')
                Ttmp = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                ftmp = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = ((B * Ttmp * ftmp + B * F * np.gradient(ftmp, xx, edge_order=2)) / (B*ftmp))#[-1]
                
                # var = self.data.v(exp, channel, 'T', 'tide','TM2')
                # f[channel].append(self.data.v(exp, channel, 'T', 'tide','TM2'))
                # f[channel].append(self.data.v(exp, channel, 'T', 'sedadv'))
                
                # a=b
                # jmax = self.data.v(exp, channel, 'jmax')

                jmax = self.data.v(exp, channel, 'jmax')
                f[channel].append(np.abs(self.data.v(exp, channel, 'u1', 'tide', range(jmax+1), 0, 2)))
                # f[channel].append(np.abs(self.data.v(exp, channel, 'zeta0', range(jmax+1), 0)))
                # f[channel].append(np.abs(self.data.v(exp, channel, 'hatc1',range(jmax+1), -1)))
                
                # T[channel].append(self.data.v(exp, channel, 'T', 'noflux'))
                # T[channel].append(self.data.v(exp, channel, 'T', 'noflux'))
                T[channel].append(np.angle(self.data.v(exp, channel, 'u1', 'tide', range(jmax+1), 0, 2)))
                # T[channel].append(np.angle(self.data.v(exp, channel, 'zeta0', range(jmax+1), 0)))

                count += 1
            
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, y)

        # a=b

        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            # ax.plot(loc[channel], fmax[channel], 'k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])

            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Phase difference')
            # ax.set_ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            # cb1.set_label('channel ' + str(i+1) + ' TM2')
            # cb1.set_label('channel ' + str(i+1) + ' $|u2|$')
            cb1.set_label('channel ' + str(i+1) + ' |u| M$_4$')
            # cb1.set_label('channel ' + str(i+1) + ' sedadv')

            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Phase difference')
            # plt.ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            # cb2.set_label('channel ' + str(i+1) + ' arg u2')
            cb2.set_label('channel ' + str(i+1) + ' arg u M$_4$' )
            # cb2.set_label('channel ' + str(i+1) + ' noflux')
            # cb2.set_label('channel ' + str(i+1) + ' TM4')
            # cb2.set_label('channel ' + str(i+1) + ' $T$')

        plt.xlabel('Position (km)')

  



    def contours_M2(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.15
        plt.rcParams['figure.subplot.hspace'] = 0.1
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')

                jmax= self.data.v(exp, str(i), 'jmax')
                kmax= self.data.v(exp, str(i), 'kmax')
                # xx = self.data.v(exp, str(i), 'x')


                uda = np.real(absoluteU(np.mean(self.data.v(exp, str(i), 'u0', range(jmax+1), range(kmax+1)),axis=1), 0))
                ub = np.real(absoluteU(self.data.v(exp, str(i), 'u0',  range(jmax+1), -1), 0))

                # v2 = np.mean(u_M0, axis=1)

                f[channel].append(uda)
                T[channel].append(ub )
                y.append(self.data.v(exp, '1', 'H'))

        
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))


        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12), edgecolor='w')
        for i in range(3):
            channel = str(i)

            ax = plt.subplot(3, 2, 2*i+1) 
            # vmax = (np.max(f[channel]))
            # vmin = (np.min(f[channel]))
            
            c = ax.contour(X[channel], Y[channel], f[channel], levels=[0], colors='k')
            c1 = ax.pcolor(X[channel], Y[channel], f[channel])


            # ax.set_xlabel('Position (km)')
            ax.set_ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb1 = plt.colorbar(c1)
            cb1.set_label('channel ' + str(i+1) + ' $u_0$da')

            vmax = (np.max(T[channel]))
            vmin = (np.min(T[channel]))
            # shifted_cmap = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
            # , cmap = shifted_cmap
            ax =  plt.subplot(3, 2, 2*i+1+1)
            c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
            c2 = ax.pcolor(X[channel], Y[channel], T[channel]) 

            plt.ylabel('Depth (m)')
            # plt.title('Channel ' + str(i))
            cb2 = plt.colorbar(c2)
            cb2.set_label('channel ' + str(i+1) + ' $u_{0b}$')

        plt.xlabel('Position (km)')




    def contours_Sbed(self):
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['figure.subplot.left'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.wspace'] = 0.15
        plt.rcParams['figure.subplot.hspace'] = 0.2
        # f['channel'] = f(x,H)
            
        f = {}
        T = {}
        X = {}
        Y = {}
        loc = {}
        fmax = {}

        for i in range(3):
            channel = str(i)
            f[channel] = [] 
            T[channel] = [] 
            loc[str(i)] = np.array([])
            fmax[str(i)] = np.array([])
            y = []

            for exp in self.experiments:
                B = self.data.v(exp, str(i), 'B')

                jmax= self.data.v(exp, str(i), 'jmax')
                kmax= self.data.v(exp, str(i), 'kmax')
                # xx = self.data.v(exp, str(i), 'x')


                Sbed = self.data.v(exp, str(i), 'Sbed')
      
                f[channel].append(Sbed)
                T[channel].append(self.data.v(exp, str(i), 'T'))
                y.append(self.data.v(exp, '1', 'H'))

        
            x = self.data.v(exp, channel, 'x') / 1000

            X[channel], Y[channel] = np.meshgrid(x, np.asarray(y))


        # fig1, axs1 = plt.subplots(3, 2, figsize=(16, 12 ), edgecolor='w')


        Smax = np.max(f.values())
        Smin = np.min(f.values())
        # shifted_cmap1 = shiftedColorMap(matplotlib.cm.rainbow, min_val=Smin,   max_val=Smax, name='shifted')


        f = []
        for exp in self.experiments:
            f.append(self.data.v(exp, '1', 'Sbed')/self.data.v(exp, '0', 'Sbed'))
        plt.figure()
        plt.contour(X['1'], Y['1'], f, colors='k', levels=[1])
        plt.pcolor(X['1'], Y['1'], f)
        plt.colorbar()

        # for i in range(3):
        #     channel = str(i)

        #     ax = plt.subplot(3, 2, 2*i+1) 
        #     # vmax = (np.max(f[channel]))
        #     # vmin = (np.min(f[channel]))
            
        #     c = ax.contour(X[channel], Y[channel], f[channel], colors='k')
        #     ax.clabel(c, inline=1, fontsize=10)
        #     c1 = ax.pcolor(X[channel], Y[channel], f[channel], vmin=Smin, vmax=Smax)


        #     # ax.set_xlabel('Position (km)')
        #     ax.set_ylabel('Depth (m)')
        #     # plt.title('Channel ' + str(i))
        #     cb1 = plt.colorbar(c1)
        #     cb1.set_label('channel ' + str(i+1) + ' $S_{bed}$')

        #     vmax = (np.max(T[channel]))
        #     vmin = (np.min(T[channel]))
        #     shifted_cmap2 = shiftedColorMap(matplotlib.cm.rainbow, min_val=vmin,   max_val=vmax, name='shifted')
           
        #     ax =  plt.subplot(3, 2, 2*i+1+1)
        #     c = ax.contour(X[channel], Y[channel], T[channel], levels=[0], colors='k')
        #     c2 = ax.pcolor(X[channel], Y[channel], T[channel] , cmap = shifted_cmap2) 

        #     plt.ylabel('Depth (m)')
        #     # plt.title('Channel ' + str(i))
        #     cb2 = plt.colorbar(c2)
        #     cb2.set_label('channel ' + str(i+1) + ' $T$')

        # plt.xlabel('Position (km)')

def shiftedColorMap(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap



    def plot_Depth(self):

        loc = {}
        varmax = {}
        for i in range(self.nch):
            loc[str(i)] = np.array([])
            varmax[str(i)] = np.array([])


        plt.figure(figsize=(10, 6.18))
        for exp in self.experiments:
            for i in range(self.nch):

                x = self.data.v(exp, str(i), 'x')

                # var = np.real(self.data.v(exp, str(i), 'f'))
                var = np.real(self.data.v(exp, str(i), 'cb'))
                # var = np.real(self.data.v(exp, str(i), 'cb')) / np.real(self.data.v(exp, str(i), 'f'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False

                loc[str(i)] = np.append(loc[str(i)], x[index])
                varmax[str(i)] = np.append(varmax[str(i)], var[index])

                if i == 0:
                    p = plt.plot(x, var, '--')
                elif i == 1:
                    p = plt.plot(x, var, color=p[-1].get_color(), label=exp[5:9])
                else:
                    plt.plot(x, var, color=p[-1].get_color())

            for i in range(self.nch):
                plt.plot(loc[str(i)], varmax[str(i)], 'k', lw = 1)

        plt.legend()
        # plt.ylabel('$c_b$ (kg/m$^3$)')
        plt.ylabel('$f$')
        # plt.ylim([0, 1.05])
        plt.xlabel('$x$ (m)')

    def plot_Discharge(self):

        plt.figure()
        loc = {}
        varmax = {}
        for i in range(self.nch):
            loc[str(i)] = np.array([])
            varmax[str(i)] = np.array([])

        for exp in self.experiments:
            for i in range(self.nch):

                x = self.data.v(exp, str(i), 'x')
                # var = np.real(self.data.v(exp, str(i), 'cb'))
                var = np.real(self.data.v(exp, str(i), 'f'))

                index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                index[0] = False
                
                
                loc[str(i)] = np.append(loc[str(i)], x[index])
                varmax[str(i)] = np.append(varmax[str(i)], var[index])

                if i == 0:
                    p = plt.plot(x, var, '--')
                elif i == 1:
                    p = plt.plot(x, var, color=p[-1].get_color(), label=exp[9:-1])
                else:
                    plt.plot(x, var, color=p[-1].get_color())

            for i in range(self.nch):
                plt.plot(loc[str(i)], varmax[str(i)], 'k', lw = 1)

        plt.legend()
        # plt.ylabel('$c_b$ (kg/m$^3$)')
        plt.ylabel('$f$')
        plt.xlabel('$x$ (m)')

    def plot_T(self):
        
        path = '/Users/jinyangwang/Desktop/plots/manu3/3ch/'

        for key in self.data.getKeysOf(self.experiments[0], '0', 'T'):
            plt.figure(figsize=(10, 6.18))
            for exp in self.experiments:
                for i in range(self.nch):
                    B = self.data.v(exp, str(i), 'B')
                    T = self.data.v(exp, str(i), 'T', key)
                    f = self.data.v(exp, str(i), 'f')
                    x = self.data.v(exp, str(i), 'x')
                    if i == 0:
                        # exp[5:9] 9:-1
                        p = plt.plot(x, T, '--')
                    elif i == 1:
                        plt.plot(x, T, label=exp[5:9], color=p[-1].get_color())
                    else:
                        plt.plot(x, T, color=p[-1].get_color())
            plt.legend()
            plt.ylabel('$T $' + ' ' +  str(key))
            plt.xlabel('$x$ (m)')
            
            plt.savefig(path+str(key)+'.pdf')
    
    
    def plot_transport(self):

        plt.figure(figsize=(10, 6.18))
        for exp in self.experiments:
            for i in range(self.nch):

                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                x = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, x, edge_order=2))

                if i == 0:
                    p = plt.plot(x, trans, '--')
                    plt.plot(x, B * T, '--', color=p[-1].get_color())
                elif i == 1:
                    plt.plot(x, trans, color=p[-1].get_color(), label=exp[5:9])
                    # plt.plot(x, trans, color=p[-1].get_color(), label=exp[9::])
                    plt.plot(x, B * T, color=p[-1].get_color())
                else:
                    plt.plot(x, trans, color=p[-1].get_color())
                    plt.plot(x, B * T, color=p[-1].get_color())


        plt.legend()
        plt.ylabel('$BTf$')
    
        plt.xlabel('$x$ (m)')

 
    def plot_NWT(self):
        plt.rcParams['figure.subplot.left'] = 0.11
        plt.rcParams['figure.subplot.bottom'] = 0.1

        plt.figure(figsize=(10, 6.18))
        Q0 = np.array([])
        Q1 = np.array([])
        plt.rcParams['lines.linewidth'] = 2
        # for mod in 
        for exp in self.experiments:
            # for i in range(self.nch):
            # mod = 'baroc'
            mod = 'river'
            Q0 = np.append(Q0, self.data.v(exp, '0', 'Q'))
            Q1 = np.append(Q1, self.data.v(exp, '1', 'Q'))
            # Q0 = np.append(Q0, self.data.v(exp, '0', 'Q', mod))
            # Q1 = np.append(Q1, self.data.v(exp, '1', 'Q', mod))
        print(self.data.getKeysOf(exp, '1', 'Q'))

        x = np.linspace(5, 20, 50)
        # x = np.linspace(500, 1500, 40) # width
        # x = np.linspace(0, 2*np.pi, 20) # pd
        plt.plot(x, -Q0, label='1', color='b')
        plt.plot(x, -Q1, label='2', color='r')
        ax = plt.gca()

        plt.legend()
        plt.ylabel('Net water transport (m$^3$/s)')
        # plt.ylabel(mod + ' water transport (m$^3$/s)')
    
        # plt.xlabel('Width of channel 2 (m)')
        plt.xlabel('Depth of channel 1 (m)')


    def plot_SSC_transport(self):
        plt.rcParams['figure.subplot.left'] = 0.11
        plt.rcParams['figure.subplot.bottom'] = 0.1
        
        plt.figure(figsize=(10, 6.18))
        Qs0 = np.array([])
        Qs1 = np.array([])
        x = []
        for exp in self.experiments:
            for i in range(self.nch):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)
                # fmax[str(i)] = np.append(fmax[str(i)], self.data.v(exp, '2', 'Q'))

            # x.append(exp[5:9])
            # x.append(exp[6:9])
        x = np.linspace(5, 20, 50)
        # x = np.linspace(3500, 8500, 20)
        # x = np.linspace(500, 1500, 40)
        # x = np.linspace(0, 2*np.pi, 20) # pd
        plt.plot(x, Qs0, label='1', color='b')
        plt.plot(x, Qs1, label='2', color='r')

        plt.legend()
        plt.ylabel('Net sediment transport (kg/s)')
        plt.xlabel('Depth of channel 1 (m)')
        # plt.xlabel('sea level rise (m)')
        # plt.xlabel('NP width at sea (m)')


        # for mod in self.data.getKeysOf(self.experiments[0], '0', 'T'):
        #     if mod not in ['diffusion_river', 'diffusion_tide']:
        #     # a = b
        #         plt.figure(figsize=(10, 6.18))
        #         Qs0 = np.array([])
        #         Qs1 = np.array([])
        #         x = []
        #         for exp in self.experiments:
        #             for i in range(self.nch):
        #                 B = self.data.v(exp, str(i), 'B')
        #                 T = self.data.v(exp, str(i), 'T', mod)
        #                 F = self.data.v(exp, str(i), 'F')
        #                 f = self.data.v(exp, str(i), 'f')
        #                 xx = self.data.v(exp, str(i), 'x')
        #                 trans = (B * T * f )[-1]
                    
        #                 if i == 0:
        #                     Qs0 = np.append(Qs0, trans)
        #                 elif i == 1:
        #                     Qs1= np.append(Qs1, trans)

        #             # x.append(exp[5:9])
        #             # x.append(exp[6:9])
        #         x = np.linspace(5, 20, 20)
        #         # x = np.linspace(500, 1500, 40)
        #         # x = np.linspace(0, 2*np.pi, 20) # pd
        #         plt.plot(x, Qs0, '--', label='1')
        #         plt.plot(x, Qs1, label='2')

        #         plt.legend()
        #         # plt.ylabel('Net sediment transport (kg/s)')
        #         plt.ylabel(mod + ' sediment transport (kg/s)')
            
        #         plt.xlabel('Depth of channel 2 (m)')
        #     # plt.xlabel('Width of channel 2 (m)')
        #     # plt.xlabel('Phase difference (rad)')
        #     # plt.xlabel('Length of channel 2 (m)')


        # plot diffusive transport
        plt.figure(figsize=(10, 6.18))
        Qs0 = np.array([])
        Qs1 = np.array([])
        x = []
        for exp in self.experiments: 
            for i in range(self.nch):
                B = self.data.v(exp, str(i), 'B')
                T = self.data.v(exp, str(i), 'T', 'diffusion_tide') + self.data.v(exp, str(i), 'T', 'diffusion_river')
                F = self.data.v(exp, str(i), 'F')
                f = self.data.v(exp, str(i), 'f')
                xx = self.data.v(exp, str(i), 'x')
                trans = (B * T * f + B * F * np.gradient(f, xx, edge_order=2))[-1]
            
                if i == 0:
                    Qs0 = np.append(Qs0, trans)
                elif i == 1:
                    Qs1= np.append(Qs1, trans)

            # x.append(exp[5:9])
            # x.append(exp[6:9])
        x = np.linspace(5, 20, 50)
        # x = np.linspace(500, 1500, 40)
        # x = np.linspace(0, 2*np.pi, 20) # pd
        plt.plot(x, Qs0, '--', label='1')
        plt.plot(x, Qs1, label='2')

        plt.legend()
        plt.ylabel('Diffusive sediment transport (kg/s)')
        plt.xlabel('Depth of channel 2 (m)')

    def plot_exchangeIntensity(self):

        plt.figure(figsize=(10, 6.18))
        for exp in self.experiments:
            for i in range(self.nch):

                x = self.data.v(exp, str(i), 'x')
                H = self.data.v(exp, str(i), 'H')
                jmax = self.data.v(exp, str(i), 'jmax')
                kmax = self.data.v(exp, str(i), 'kmax')                
                L = x[-1] - x[0]
                z = np.linspace(-H, 0, kmax+1)
                X, Z = np.meshgrid(np.linspace(0, L, jmax+1), z)

                u = np.real(self.data.v(exp, str(i), 'u1'))
                integrand = u.transpose() * (Z + 0.5 * H)
                exchangeIntensity = 4 * np.trapz(integrand, x=z, axis=0) / H**2


                var = exchangeIntensity


                # index = np.r_[True, var[1:] > var[:-1]] & np.r_[var[:-1] > var[1:], True]
                # index[0] = False

                # loc[str(i)] = np.append(loc[str(i)], x[index])
                # varmax[str(i)] = np.append(varmax[str(i)], var[index])

                if i == 0:
                    p = plt.plot(x, var, '--')
                elif i == 1:
                    p = plt.plot(x, var, color=p[-1].get_color(), label=exp[5:9])
                else:
                    plt.plot(x, var, color=p[-1].get_color())

            # for i in range(self.nch):
            #     plt.plot(loc[str(i)], varmax[str(i)], 'k', lw = 1)

        plt.legend()
        plt.ylabel('exchange flow intensity')
        plt.xlabel('$x$ (m)')
    
