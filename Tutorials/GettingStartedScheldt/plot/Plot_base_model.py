"""
Plot

Date: 20-Apr-16
Authors: Y.M. Dijkstra
"""
import matplotlib.pyplot as plt
import step as st
import numpy as np
from step import Step
import nifty as ny
import logging


class Plot_base_model:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Plotting')
        ## Load data from data container

        x = self.input.v('grid', 'axis', 'x')
        z = self.input.v('grid', 'axis', 'z', 0)
        x_km = ny.dimensionalAxis(self.input.slice('grid'), 'x', x=x, z=0, f=0)

        L = self.input.v('grid', 'high', 'x')
        H = self.input.n('grid', 'high', 'z', x=x)
        R = self.input.v('grid', 'low', 'z', x=x)
        depth = H+ R

        fmax = self.input.v('grid', 'maxIndex', 'f')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        jmax = self.input.v('grid', 'maxIndex', 'x')

        Av = self.input.v('Av', x=x, z=0.5, f=range(0, fmax+1))
        Roughness = self.input.v('Roughness', x=x, z=0, f=range(0, fmax+1))

        zeta = self.input.v('zeta0', x=x, z=0, f=range(0, fmax+1)) + \
               self.input.v('zeta1', x=x, z=0, f=range(0, fmax + 1))
        u =    self.input.v('u0', x=x, z=z, f=range(0, fmax+1)) + \
               self.input.v('u1', x=x, z=z, f=range(0, fmax+1))

        # depth average of the velocity
        u = ny.integrate(u, 'z', kmax, 0, self.input.slice('grid'))/depth.reshape((len(x), 1, 1))
        u = u.reshape(u.shape[0], u.shape[-1])

        ## Plot
        st.configure()
        step = Step.Step(self.input)

        step.lineplot('x', 'H', z=0, f=0, sublevel=False, operation=np.real,plotno=1)      # depth
        step.lineplot('x', 'B', z=0, f=0, sublevel=False, operation=np.abs,plotno=2)       # width
        try:
            step.lineplot('x', 's0', z=0, f=0, sublevel=False, operation=np.abs,plotno=3)     # salinity
        except:
            pass

        # water level amplitude
        plt.figure(4, figsize=(1,2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for n in range(0, 3):
            if n==0:
                plt.plot(x_km/1000., abs(zeta[:, n]+self.input.v('R', range(0, jmax+1))), label='$M_'+str(2*n)+'$')
            else:
                plt.plot(x_km/1000., abs(zeta[:, n]), label='$M_'+str(2*n)+'$')
        plt.ylabel('$|\hat{\zeta}|$ $(m)$')
        plt.title('Water level amplitude')
        plt.xlabel('x (km)')
        plt.xlim(0, L/1000.)
        plt.legend(bbox_to_anchor=(1.15, 1.05))

        # water level phase
        plt.figure(5, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        for n in range(0, 3):
            plt.plot(x_km/1000., -np.angle(zeta[:, n])*180/np.pi, label='$M_'+str(2*n)+'$')
        plt.ylabel('$\phi(\hat{\zeta})$ $(deg)$')
        plt.xlabel('x (km)')
        plt.title('Water level phase')
        plt.ylim(-180, 180)
        plt.legend(bbox_to_anchor=(1.15, 1.05))
        plt.xlim(0, L/1000.)

        # depth-average velocity amplitude
        plt.figure(6, figsize=(1, 2))
        plt.subplot2grid((1,8), (0, 0),colspan=7)
        # velocity components
        for n in range(0, 3):
            p = plt.plot(x_km/1000., abs(u[:, n]), label='$M_'+str(2*n)+'$')
        plt.ylabel('$|\hat{u}|$ $(m/s)$')
        plt.title('Velocity amplitude')
        plt.xlabel('x (km)')
        plt.xlim(0, L/1000.)
        plt.legend(bbox_to_anchor=(1.15, 1.05))

        # eddy viscosity
        plt.figure(7, figsize=(2, 2))
        plt.subplot(2,2,1)
        plt.plot(x_km/1000, abs(Av[:,0]))
        plt.plot(x_km/1000, abs(Av[:,1]))
        plt.plot(x_km/1000, abs(Av[:,2]))
        plt.ylabel(r'$|\hat{A}_{\nu}|$ $(m^2/s)$')
        plt.xlabel('x (km)')
        plt.xlim(0, L/1000.)

        plt.subplot(2, 2, 2)
        plt.plot(x_km/1000, abs(Av[:,0])/depth)
        plt.plot(x_km/1000, abs(Av[:,1])/depth)
        plt.plot(x_km/1000, abs(Av[:,2])/depth)
        plt.ylabel(r'$|\hat{A}_{\nu}|/H$ $(m/s)$')
        plt.xlabel('x (km)')
        plt.xlim(0, L/1000.)

        plt.subplot(2,2,3)
        plt.plot(x_km/1000, abs(Roughness[:,0]), label='$M_0$')
        plt.plot(x_km/1000, abs(Roughness[:,1]), label='$M_2$')
        plt.plot(x_km/1000, abs(Roughness[:,2]), label='$M_4$')
        plt.ylabel(r'$|\hat{s}_{f}|$ $(m/s)$')
        plt.xlabel('x (km)')
        plt.legend(bbox_to_anchor=(1.35, 1.04))
        plt.xlim(0, L/1000.)

        #sediment plots
        step.contourplot('x', 'z', 'c0', f=0, sublevel=False, operation=np.abs,plotno=8)       # leading order subtidal concentration
        step.transportplot_mechanisms(sublevel='sublevel', concentration=True,plotno=9)       # transportterms


        st.show()

        d = {}
        return d