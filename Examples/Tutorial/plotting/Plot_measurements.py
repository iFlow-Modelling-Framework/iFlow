"""
Basic plot routine for hydrodynamics and sediment dynamics

Date: 13-Nov-17
Authors: Y.M. Dijkstra
"""
import matplotlib.pyplot as plt
import step as st
import numpy as np
import nifty as ny


class Plot_measurements:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        ################################################################################################################
        ## Get data
        ################################################################################################################
        # grid sizes
        fmax = self.input.v('grid', 'maxIndex', 'f')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        jmax = self.input.v('grid', 'maxIndex', 'x')

        # grid axes, dimensionless and with dimension
        x = self.input.v('grid', 'axis', 'x')               # dimensionless x axis between 0 and 1 (jmax+1)
        z = self.input.v('grid', 'axis', 'z', 0)            # dimensionless z axis between 0 and 1 (kmax+1)
        x_km = ny.dimensionalAxis(self.input.slice('grid'), 'x', x=x, z=0, f=0) # x axis in m between 0 and L (jmax+1)

        L = self.input.v('grid', 'high', 'x')               # length in m (1)
        B = self.input.v('B', x=x, z=[0], f=[0])            # width (jmax+1, 1, 1)

        # variables
        Av = self.input.v('Av', x=x, z=0.5, f=range(0, fmax+1))         # Eddy viscosity (jmax+1, fmax+1)
        Roughness = self.input.v('Roughness', x=x, z=0, f=range(0, fmax+1))     # Partial slip coefficient (jmax+1, fmax+1)

        zeta = self.input.v('zeta0', x=x, z=0, f=range(0, fmax+1)) + self.input.v('zeta1', x=x, z=0, f=range(0, fmax+1))    # water level (jmax+1, fmax+1)
        u = self.input.v('u0', x=x, z=z, f=range(0, fmax+1)) + self.input.v('u1', x=x, z=z, f=range(0, fmax+1))             # horizontal velocity (jmax+1, kmax+1, fmax+1)
        c = self.input.v('c0', x=x, z=z, f=range(0, fmax+1)) + self.input.v('c1', x=x, z=z, f=range(0, fmax+1)) + self.input.v('c2', x=x, z=z, f=range(0, fmax+1))  # concentration (jmax+1, kmax+1, fmax+1)

        StotalB = ny.integrate(ny.integrate(B*c, 'z', kmax, 0, self.input.slice('grid')), 'x', 0, jmax, self.input.slice('grid'))   # compute total sediment stock
        print('Total sediment stock in domain (mln kg): '+str(np.real(StotalB[0,0,0])/1.e6))

        # use data from measurements
        measurementset = self.input.v('measurementset')
        x_waterlevel = self.input.v(measurementset, 'x_waterlevel')
        x_velocity = self.input.v(measurementset, 'x_velocity')
        zeta_meas = self.input.v(measurementset, 'zeta', x=x_waterlevel/L, z=0, f=range(0, 3))
        ucomp_meas = self.input.v(measurementset, 'u_comp', x=x_velocity/L, z=0, f=range(0, 3))

        ################################################################################################################
        ## Plot
        ################################################################################################################
        st.configure()

        # Figure 1 - Water level amplitude
        plt.figure(1, figsize=(1,2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for n in range(0, 3):
            if n==0:
                p = plt.plot(x_km/1000., abs(zeta[:, n]+self.input.v('R', range(0, jmax+1))), label='$M_'+str(2*n)+'$')
            else:
                p = plt.plot(x_km/1000., abs(zeta[:, n]), label='$M_'+str(2*n)+'$')
            plt.plot(x_waterlevel/1000., abs(zeta_meas[:, n]), 'o', color=p[0].get_color())
        plt.ylabel('$|\hat{\zeta}|$ $(m)$')
        plt.xlabel('x (km)')
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.xlim(np.min(x_km/1000.), np.max(x_km/1000.))
        plt.title('Water level amplitude')

        # Figure 2 - Water level phase
        plt.figure(2, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for n in range(0, 3):
            p = plt.plot(x_km/1000., -np.angle(zeta[:, n])*180/np.pi, label='$M_'+str(2*n)+'$')
            if n == 1 or n == 2:
                plt.plot(x_waterlevel/1000., -np.angle(zeta_meas[:, n])*180/np.pi, 'o', color=p[0].get_color())
        plt.ylabel('$\phi(\hat{\zeta})$ $(deg)$')
        plt.xlabel('x (km)')
        plt.ylim(-180, 180)
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.xlim(np.min(x_km/1000.), np.max(x_km/1000.))
        plt.title('Water level phase')

        # Figure 3 - Velocity amplitude
        plt.figure(3, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        # velocity components
        for n in range(0, 3):
            p = plt.plot(x_km/1000., abs(u[:, 0, n]), label='$M_'+str(2*n)+'$')
            plt.plot(x_velocity/1000., abs(ucomp_meas[:, n]), 'o', color=p[0].get_color())
        plt.ylabel('$|\hat{u}|$ $(m/s)$')
        plt.xlabel('x (km)')
        plt.title('Surface velocity amplitude')
        plt.xlim(np.min(x_km/1000.), np.max(x_km/1000.))
        plt.legend(bbox_to_anchor=(1.15, 1.05))

        # Figure 4 - Roughness, Eddy viscosity
        plt.figure(4, figsize=(1, 2))
        plt.subplot(1,2,1)
        for n in range(0, fmax+1):
            plt.plot(x_km/1000, abs(Av[:, n]))
        plt.ylabel(r'$|\hat{A}_{\nu}|$ $(m^2/s)$')
        plt.xlabel('x (km)')

        plt.subplot(1,2,2)
        for n in range(0, fmax+1):
            plt.plot(x_km/1000, abs(Roughness[:, n]), label='$M_'+str(2*n)+'$')
        plt.ylabel(r'$|\hat{s}_{f}|$ $(m/s)$')
        plt.xlabel('x (km)')

        # Figure 5 - surface concentration
        plt.figure(5, figsize=(1,2))
        for n in range(0, fmax+1):
            p = plt.plot(x_km/1000., abs(c[:, 0, n]))
        plt.xlabel('x (km)')
        plt.ylabel('|c| $(kg/m^3)$')
        plt.title('Surface sediment concentration')

        # Figure 6 - availability
        plt.figure(11, figsize=(1,1))
        a = self.input.v('a', range(0, jmax+1), 0, 0)
        if self.input.v('a') is not None:
            plt.plot(x_km/1000., a)
            plt.legend()
            plt.xlabel('x (km)')
            plt.ylabel('a')
            plt.title('sediment availability')

        st.show()

        return {}