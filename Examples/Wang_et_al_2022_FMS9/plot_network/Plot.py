"""
Date:
Authors:
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import step as st
import nifty as ny



class Plot:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input

        return

    def run(self):
        st.configure()
        numberofchannels = self.input.v('network_settings', 'numberofchannels')
        channelList = self.input.getKeysOf('network_output')
        plotmin = -40
        plotmax = 40

        plt.figure(1, figsize=(1,2))
        plt.title('leading-order water level M2')
        plt.xlim(plotmin,plotmax)
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')

            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)

            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)
            zeta0 = dc.v('zeta0', range(0, jmax+1), 0, 1)

            plt.plot(x/1000, abs(zeta0), color=colour)

        plt.figure(2, figsize=(2,3))
        plt.suptitle('First-order water level subtidal')
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)

            for q,mod in enumerate(dc.getKeysOf('zeta1')):
                plt.subplot(2,4,q+1)
                plt.title(mod)
                zeta1 = dc.v('zeta1', mod, range(0, jmax+1), 0, 0)
                plt.plot(x/1000, np.real(zeta1), color=colour)
                plt.xlim(plotmin,plotmax)

        plt.figure(3, figsize=(2,3))
        plt.suptitle('First-order water level M4')
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)
            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)

            for q, mod in enumerate(dc.getKeysOf('zeta1')):
                plt.subplot(2,4,q+1)
                plt.title(mod)
                zeta1 = dc.v('zeta1', mod, range(0, jmax+1), 0, 2)
                plt.plot(x/1000, abs(zeta1), color=colour)
                plt.xlim(plotmin,plotmax)

        plt.figure(4, figsize=(1,2))
        plt.title('salinity')
        plt.xlim(plotmin,plotmax)
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)

            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)
            s0 = dc.v('s0', range(0, jmax+1), 0, 0)

            plt.plot(x/1000, np.real(s0), color=colour)

        plt.figure(5, figsize=(1,2))
        plt.subplot(1,2,1)
        plt.title('concentration c00')
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)

            jmax = dc.v('grid', 'maxIndex', 'x')
            kmax = dc.v('grid', 'maxIndex', 'z')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)
            H = dc.v('H', range(jmax+1))
            c0 = ny.integrate(dc.v('c0', range(0, jmax+1), range(kmax+1), 0), 'z', kmax, 0, dc)[:,0]/H

            plt.plot(x/1000, np.real(c0), color=colour)
            plt.xlim(plotmin,plotmax)
        plt.subplot(1,2,2)
        plt.title('erodibility')
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)

            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)
            f = dc.v('f', range(0, jmax+1), 0, 0)

            plt.plot(x/1000, np.real(f), color=colour)
            plt.xlim(plotmin,plotmax)
        plt.figure(6, figsize=(1,2))
        plt.title('Transport capacity')
        ## plot water level
        for channel in channelList:
            dc = self.input.v('network_output', channel)
            channelNumber = dc.v('channelNumber')
            colour = self.input.v('network_settings', 'label', 'ColourLabel', channelNumber)

            jmax = dc.v('grid', 'maxIndex', 'x')
            x = np.linspace(0, 1, jmax+1)*dc.v('L') + self.input.v('network_settings', 'L0', channelNumber)
            T = dc.v('T', range(0, jmax+1), 0, 0)

            plt.plot(x/1000, np.real(T), color=colour)
            plt.xlim(plotmin,plotmax)
        st.show()

        return
