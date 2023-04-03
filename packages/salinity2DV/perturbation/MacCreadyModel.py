"""
Salinity module based on the model presented by MacCready, 2004
Author: Y.M. Dijkstra
Date: 01-04-2023
"""
import logging
import nifty as ny
import numpy as np


class MacCreadyModel:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        jmax = self.input.v('grid', 'maxIndex', 'x')
        zeta = -self.input.v('grid', 'axis', 'z')
        x = ny.dimensionalAxis(self.input, 'x')[:, 0, 0]
        dx = x[1:]-x[:-1]

        Q = self.input.v('Q1', range(0, jmax+1), [0], 0)
        sigrho_sal = self.input.v('sigrho_sal')

        Av = np.real(self.input.v('Av', range(0, jmax+1), [0], 0))
        Kv = np.real(self.input.v('Kv', range(0, jmax+1), [0], 0))/sigrho_sal
        sf = self.input.v('Roughness', range(0, jmax+1), [0], 0)
        g = self.input.v('G')
        beta = self.input.v('BETA')
        ssea = self.input.v('ssea')

        H = self.input.v('H', range(0, jmax+1), [0], 0)+self.input.v('R', range(0, jmax+1), [0], 0)
        B = self.input.v('B', range(0, jmax+1), [0], 0)
        Kh = self.input.v('Kh')
        # Kh = np.maximum(Kh*np.sqrt(B/B[0,0]),Kh/10)

        R = (Av/(sf*H))

        P1 = (1/6-1/2*zeta**2)/(R+1/3)
        P2 = (.5*zeta**2-.5-R)*(1/8+.5*R)/(R+1/3)+1/6*(zeta**3+1)+.5*R
        P3 = (1/24*zeta**4-1/12*zeta**2+7/360)/(R+1/3)
        P4 = 1/120*zeta**5+1/24*(1/8+.5*R)/(R+1/3)*zeta**4+(1/12+1/4*R-(1/4+1/2*R)*(1/8+1/2*R)/(R+1/3))*zeta**2+1/8640*(5+36*R)/(R+1/3)

        P1P3 = np.trapz(-P1*P3, zeta) # minus for switching integration direction
        P1P4 = np.trapz(-P1*P4, zeta)
        P2P3 = np.trapz(-P2*P3, zeta)
        P2P4 = np.trapz(-P2*P4, zeta)

        A1 = -(Q/(B*H))**2*H**2/Kv*P1P3 - Kh
        B1 = Q/(B*H)*g*beta*H**5/(Av*Kv)*(P1P4-P2P3)
        C1 = g**2*beta**2*H**8/(Av**2*Kv)*P2P4
        D1 = Q/(B*H)

        # Init solution
        s = np.zeros(jmax+1)
        sx = np.zeros(jmax+1)

        # Solve BC
        temp = np.roots([C1[0,0], B1[0,0]-D1[0,0]*(g*beta*H**5/(Av*Kv)*P4[:,-1])[0,0], A1[0,0]+D1[0,0]*(Q/(B*H)*H**2/Kv*P3[:,-1])[0,0], -D1[0,0]*ssea])
        for i,r in enumerate(temp):
            if np.imag(r)==0 and r<0:
                sx[0] = np.real(r)
                break
            if i==len(temp)-1:
                self.logger.error('No solution for BC found')
        s[0] = (A1[0]/D1[0]*sx[0] +B1[0]/D1[0]*sx[0]**2 + C1[0]/D1[0]*sx[0]**3)[0]

        # Solve interior
        for j in range(1, jmax+1):
            temp = np.roots([C1[j,0], B1[j,0], A1[j,0]-dx[j-1]*D1[j,0], -D1[j,0]*s[j-1]-.0*D1[j,0]*dx[j-1]*sx[j-1]])
            for i,r in enumerate(temp):
                if np.imag(r)==0 and r<0:
                    sx[j] = np.real(r)
                    break
                # if i==len(temp)-1:
                    # self.logger.error('No solution for interior point')
            s[j] = (A1[j]/D1[j]*sx[j] + B1[j]/D1[j]*sx[j]**2 + C1[j]/D1[j]*sx[j]**3)[0]


        sriv = Q/(B*H)*H**2/Kv*sx[:,None]*P3
        sgc = -g*beta*H**5/(Av*Kv)*sx[:,None]**2*P4

        stot = s[:,None] + sriv + sgc
        return {'s0':stot}

        # import matplotlib.pyplot as plt
        # import step as st
        # tide_phase = self.input.v('tide')
        # x_saldata, saldata_bottom, saldata_surf = sal_data(Q[0,0], tide_phase)
        # # ssea = saldata_bottom[np.argmin(np.abs(x_saldata))]
        # u = -Q/(B*H)*(1+P1)
        # ugc = -g*beta*H**3/Av*sx[:,None]*P2
        #
        #
        # st.configure()
        # plt.figure(1, figsize=(2,2))
        # plt.subplot(2,3,1)
        # plt.plot(x, s)
        # plt.plot(x, stot[:,0])
        # plt.plot(x, stot[:,-1])
        # plt.plot(x_saldata, saldata_surf, 'k:')
        # plt.plot(x_saldata, saldata_bottom, 'k:')
        # # plt.plot(x, s_prescribed[:,0], 'k:')
        # plt.ylim(-1,ssea)
        #
        # plt.subplot(2,3,2)
        # plt.plot(x, stot[:,-1]-stot[:,0])
        # plt.plot(x_saldata, saldata_bottom-saldata_surf, 'k:')
        #
        # plt.subplot(2,2,3)
        # plt.plot(x/1000, np.abs(Av[:,0]), label='Av')
        # plt.plot(x/1000, np.abs(Kv[:,0]), label='Kv')
        # plt.legend()
        #
        # plt.subplot(2,3,4)
        # plt.plot(x/1000, H)
        #
        # plt.subplot(2,3,5)
        # plt.plot(x/1000, B)
        #
        # plt.subplot(2,3,6)
        # plt.plot(x/1000, B*H)
        # plt.plot(data[:,0]-data[0,0], data[:,3])
        # plt.plot(data[:,0]-data[0,0], data[:,4])
        #
        #
        # # plt.plot(data[:,0]-data[0,0], data[:,3])
        # # plt.plot(data[:,0]-data[0,0], data[:,4])
        #
        # plt.figure(2, figsize=(1,2))
        # plt.subplot(1,2,1)
        # plt.plot(u[0,:], zeta[0,:])
        # plt.plot(ugc[0,:], zeta[0,:])
        # plt.subplot(1,2,2)
        # plt.plot(sriv[0,:], zeta[0,:])
        # plt.plot(sgc[0,:], zeta[0,:])
        #
        # st.show()
        #