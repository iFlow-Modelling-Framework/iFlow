"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from scipy.linalg import solve_banded
from numpy.linalg import svd
import matplotlib.pyplot as plt
import step as st


class StaticAvailability:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 10**-6

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module StaticAvailability')

        ################################################################################################################
        ## Init
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')

        c0 = self.input.v('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0 = self.input.v('hatc1', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        c1_a0x = self.input.v('hatc1', 'ax', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        d = {}

        c0_int = ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
        B = self.input.v('B', range(0, jmax+1), [0], [0])
        u0 = self.input.v('u0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        zeta0 = self.input.v('zeta0', range(0, jmax+1), [0], range(0, fmax+1))
        Kh = self.input.v('Kh', range(0, jmax+1), [0], [0])

        ################################################################################################################
        ## Second order closure
        ################################################################################################################
        u1 = self.input.v('u1', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        d['T'] = {}
        d['F'] = {}
        T0 = 0
        F0 = 0

        ## Transport T  ############################################################################################
        ## T.1. - u0*c1_a0
        # Total
        c1a_f0 = c1_a0
        T0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1a_f0, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('hatc1', 'a'):
            if submod == 'erosion':
                for subsubmod in self.input.getKeysOf('hatc1', 'a', 'erosion'):
                    c1_a0_comp = self.input.v('hatc1', 'a', submod, subsubmod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                    c1a_f0_comp_res = c1_a0_comp
                    d['T'] = self.dictExpand(d['T'], subsubmod, ['TM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
                    # transport with residual availability
                    for n in range(0, fmax+1):
                        tmp = np.zeros(c1a_f0_comp_res.shape, dtype=complex)
                        tmp[:, :, n] = c1a_f0_comp_res[:, :, n]
                        tmp = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                        if any(abs(tmp)) > 10**-14:
                            d['T'][subsubmod]['TM'+str(2*n)] += tmp
            else:
                c1_a0_comp = self.input.v('hatc1', 'a', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
                c1a_f0_comp_res = c1_a0_comp
                d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
                # transport with residual availability
                for n in range(0, fmax+1):
                    tmp = np.zeros(c1a_f0_comp_res.shape, dtype=complex)
                    tmp[:, :, n] = c1a_f0_comp_res[:, :, n]
                    tmp = ny.integrate(ny.complexAmplitudeProduct(u0, tmp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp

        ## T.2. - u1*c0
        # Total
        T0 += ny.integrate(ny.complexAmplitudeProduct(u1, c0, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('u1'):
            u1_comp = self.input.v('u1', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['T'] = self.dictExpand(d['T'], submod, ['TM'+str(2*n) for n in range(0, fmax+1)]) # add submod index to dict if not already
            # transport with residual availability
            for n in range(0, fmax+1):
                tmp = np.zeros(u1_comp.shape, dtype=complex)
                tmp[:, :, n] = u1_comp[:, :, n]
                if submod == 'stokes':
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod] = self.dictExpand(d['T'][submod], 'TM'+str(2*n), ['return', 'drift'])
                        d['T'][submod]['TM0']['return'] += tmp
                else:
                    tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c0, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                    if any(abs(tmp)) > 10**-14:
                        d['T'][submod]['TM'+str(2*n)] += tmp

        ## T.5. - u0*c0*zeta0
        # Total
        T0 += ny.complexAmplitudeProduct(ny.complexAmplitudeProduct(u0[:, [0], :], c0[:, [0], :], 2), zeta0, 2)

        # Decomposition
        uzeta = ny.complexAmplitudeProduct(u0[:, [0], :], zeta0, 2)
        d['T'] = self.dictExpand(d['T'], 'stokes', ['TM'+str(2*n) for n in range(0, fmax+1)])
        # transport with residual availability
        for n in range(0, fmax+1):
            tmp = np.zeros(c0[:, [0], :].shape, dtype=complex)
            tmp[:, :, n] = c0[:, [0], n]
            tmp = ny.complexAmplitudeProduct(uzeta, tmp, 2)[:, 0, 0]
            if any(abs(tmp)) > 10**-14:
                d['T']['stokes']['TM'+str(2*n)]['drift'] += tmp

        ## T.6. - u1riv*c2rivriv
        c2 = self.input.v('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        u1riv = self.input.v('u1', 'river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        if u1riv is not None:
            d['T'] = self.dictExpand(d['T'], 'river_river', 'TM0')  # add submod index to dict if not already
            tmp = ny.integrate(ny.complexAmplitudeProduct(u1riv, c2, 2), 'z', kmax, 0, self.input.slice('grid'))
            if any(abs(tmp[:, 0,0])) > 10**-14:
                d['T']['river_river']['TM0'] = tmp[:, 0,0]

            T0 += tmp

        ## T.7. - diffusive part
        # Total
        c0x = self.input.d('hatc0', 'a', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        T0 += - Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid'))

        c2x = self.input.d('hatc2', 'a', 'erosion', 'river_river', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1), dim='x')
        T0 += - Kh*ny.integrate(c2x, 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        d['T'] = self.dictExpand(d['T'], 'diffusion_tide', ['TM0'])
        d['T'] = self.dictExpand(d['T'], 'diffusion_river', ['TM0'])
        # transport with residual availability
        tmp = - (Kh*ny.integrate(c0x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['T']['diffusion_tide']['TM0'] = tmp
        tmp = - (Kh*ny.integrate(c2x, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['T']['diffusion_river']['TM0'] = tmp


        ## Diffusion F  ############################################################################################
        ## F.1. - u0*C1ax*f0
        # Total
        F0 += ny.integrate(ny.complexAmplitudeProduct(u0, c1_a0x, 2), 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        for submod in self.input.getKeysOf('hatc1', 'ax'):
            c1_ax0_comp = self.input.v('hatc1', 'ax', submod, range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['F'] = self.dictExpand(d['F'], submod, ['FM'+str(2*n) for n in range(0, fmax+1)])  # add submod index to dict if not already
            # transport with residual availability
            for n in range(0, fmax+1):
                tmp = np.zeros(u0.shape, dtype=complex)
                tmp[:, :, n] = u0[:, :, n]
                tmp = ny.integrate(ny.complexAmplitudeProduct(tmp, c1_ax0_comp, 2), 'z', kmax, 0, self.input.slice('grid'))[:, 0, 0]
                if any(abs(tmp)) > 10**-14:
                    d['F'][submod]['FM'+str(2*n)] += tmp

        ## F.3. - diffusive part
        # Total
        F0 += - Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid'))
        F0 += - Kh*ny.integrate(c2, 'z', kmax, 0, self.input.slice('grid'))

        # Decomposition
        d['F'] = self.dictExpand(d['F'], 'diffusion_tide', ['FM0'])
        d['F'] = self.dictExpand(d['F'], 'diffusion_river', ['FM0'])
        # transport with residual availability
        tmp = - (Kh*ny.integrate(c0, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['F']['diffusion_tide']['FM0'] = tmp
        tmp = - (Kh*ny.integrate(c2, 'z', kmax, 0, self.input.slice('grid')))[:, 0, 0]
        if any(abs(tmp)) > 10**-14:
            d['F']['diffusion_river']['FM0'] = tmp

        ## Solve    ################################################################################################
        ## Add all mechanisms & compute a0c
        from src.DataContainer import DataContainer
        dc = DataContainer(d)
        dc.merge(self.input.slice('grid'))
        T_til = np.real(dc.v('T', range(0, jmax+1)))
        F_til = np.real(dc.v('F', range(0, jmax+1)))

        # DEBUG: CHECKS IF COMPOSITE T, F == total T, F
        # print np.max(abs((dc.v('T', range(0, jmax+1))-T0[:, 0, 0])/(T0[:, 0, 0]+10**-10)))
        # print np.max(abs((dc.v('F', range(0, jmax+1))-F0[:, 0, 0])/(F0[:, 0, 0]+10**-10)))

        integral = -ny.integrate(T_til/(F_til-10**-6), 'x', 0, range(0, jmax+1), self.input.slice('grid'))

        ################################################################################################################
        # Boundary condition 1
        ################################################################################################################
        if self.input.v('sedbc')=='astar':
            astar = self.input.v('astar')
            k = astar * ny.integrate(B[:, 0, 0], 'x', 0, jmax, self.input.slice('grid'))/ny.integrate(B[:, 0, 0]*np.exp(integral), 'x', 0, jmax, self.input.slice('grid'))

            f0 = k*np.exp(integral)
            f0x = -T_til/F_til*f0

        ################################################################################################################
        # Boundary condition 2
        ################################################################################################################
        elif self.input.v('sedbc')=='csea':
            csea = self.input.v('csea')
            c000 = np.real(c0_int[0,0,0])
            k = csea/c000*(self.input.v('grid', 'low', 'z', 0)-self.input.v('grid', 'high', 'z', 0))
            f0 = k*np.exp(integral)
            f0x = -T_til/F_til*f0

        else:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('sediment boundary sedbc not known: use astar or csea')

        ################################################################################################################
        # Store in dict
        ################################################################################################################
        d['a'] = f0
        d['c0'] = c0*f0.reshape((jmax+1, 1, 1))
        d['c1'] = c1_a0*f0.reshape((jmax+1, 1, 1)) + c1_a0x*f0x.reshape((jmax+1, 1, 1))
        d['c2'] = c2*f0.reshape((jmax+1, 1, 1))

        return d

    def dictExpand(self, d, subindex, subsubindices):
        if not subindex in d:
            d[subindex] = {}
        elif not isinstance(d[subindex], dict):
            d[subindex] = {}
        for ssi in ny.toList(subsubindices):
            if not ssi in d[subindex]:
                d[subindex][ssi] = 0
        return d

