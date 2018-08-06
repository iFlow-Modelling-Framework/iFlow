"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
# from cFunctionUncoupled import cFunctionUncoupled
from erosion import erosion


class SedDynamicLead:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Running module SedDynamic - leading order')
        d = {}

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        self.submodulesToRun = self.input.v('submodules')
        # H = self.input.v('H', range(0, jmax+1))
        method = self.input.v('erosion_formulation')
        frictionpar = self.input.v('friction')      # friction parameter used for the erosion, by default the total roughness
        if frictionpar == None:
            frictionpar = 'Roughness'

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        Kv = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        # NB. If Kv is not provided on input, use the module DiffusivityUndamped to compute it. This is a fix for making this module easier to use.
        if Kv is None:
            from DiffusivityUndamped import DiffusivityUndamped
            sr = self.input.v('sigma_rho')
            if sr is None:  # add Prandtl-Schmidt number if it does not exist
                self.input.addData('sigma_rho', 1.)
            md = DiffusivityUndamped(self.input)
            self.input.merge(md.run())
            Kv = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
            d['Kv'] = Kv

        ws = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax+1, kmax+1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)

        # erosion
        E = erosion(ws, 0, self.input, method, friction=frictionpar)
        Fbed[:, :, fmax:, 0] = -E



        ################################################################################################################
        # Solve equation
        ################################################################################################################
        # Coupled system
        c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
        c = c.reshape((jmax+1, kmax+1, ftot))
        hatc0 = ny.eliminateNegativeFourier(c, 2)

        # Uncoupled system
        # hatc0 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # hatc0[:, :, 0] = cFunctionUncoupled(ws, Kv, F, Fsurf, Fbed, self.input, 0).reshape((jmax+1, kmax+1))
        # hatc0[:, :, 2] = cFunctionUncoupled(ws, Kv, F, Fsurf, Fbed, self.input, 2).reshape((jmax+1, kmax+1))

        # correction at the bed (optional)
        # cbed00 = E[:, -1, 0]/ws[:, -1, 0]
        # frac = cbed00/hatc0[:, -1, 0]
        # hatc0 = hatc0*frac.reshape((jmax+1, 1, 1))

        ## analytical solutions
        # ahatc0 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # ahatc02 = np.zeros((jmax+1, kmax+1, fmax+1), dtype=complex)
        # z = ny.dimensionalAxis(self.input.slice('grid'), 'z')[:, :, 0]
        # ws = ws[:, 0, 0]
        # Kv = Kv[:, 0, 0]
        # OMEGA = self.input.v('OMEGA')
        # H = self.input.v('H', range(0, jmax+1))
        #
        # # M0
        # ahatc0[:, :, 0] = (E[:, 0, 0]/ws).reshape((jmax+1, 1))*np.exp(-(ws/Kv).reshape((jmax+1, 1))*(z+H.reshape((jmax+1, 1))))
        #
        #
        # # M4
        #
        # r1 = -ws/(2.*Kv)+np.sqrt(ws**2+8*1j*OMEGA*Kv)/(2*Kv)
        # r2 = -ws/(2.*Kv)-np.sqrt(ws**2+8*1j*OMEGA*Kv)/(2*Kv)
        # k2 = -E[:, 0, 2]*(Kv*(-r1*(ws+Kv*r2)/(ws+Kv*r1)*np.exp(-r1*H) + r2*np.exp(-r2*H)))**-1.
        # k1 = -k2*(ws+Kv*r2)/(ws+Kv*r1)
        # ahatc0[:, :, 2] = k1.reshape((jmax+1, 1))*np.exp(r1.reshape((jmax+1, 1))*z) + k2.reshape((jmax+1, 1))*np.exp(r2.reshape((jmax+1, 1))*z)

        # import step as st
        # import matplotlib.pyplot as plt
        #
        # st.configure()
        # plt.figure(1, figsize=(1,3))
        # plt.subplot(1,3,1)
        # plt.plot(np.real(ahatc0[0, :, 0]), z[0, :], label='analytical')
        # plt.plot(np.real(hatc0[0, :, 0]), z[0, :], label='numerical')
        # plt.xlim(0, np.max(np.maximum(abs(ahatc0[0, :, 0]), abs(hatc0[0, :, 0])))*1.05)
        #
        # plt.subplot(1,3,2)
        # plt.plot(np.abs(ahatc0[0, :, 2]), z[0, :], label='analytical')
        # # plt.plot(np.abs(ahatc02[0, :, 2]), z[0, :], label='analytical2')
        # plt.plot(np.abs(hatc0[0, :, 2]), z[0, :], label='numerical')
        # # plt.xlim(np.min(np.minimum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05, np.max(np.maximum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05)
        # plt.legend()
        #
        # plt.subplot(1,3,3)
        # plt.plot(np.imag(ahatc0[0, :, 2]), z[0, :], label='analytical')
        # # plt.plot(np.imag(ahatc02[0, :, 2]), z[0, :], label='analytical2')
        # plt.plot(np.imag(hatc0[0, :, 2]), z[0, :], label='numerical')
        # # plt.xlim(np.min(np.minimum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05, np.max(np.maximum(abs(ahatc0[0, :, 2]), abs(hatc0[0, :, 2])))*1.05)
        # plt.legend()
        # st.show()


        d['hatc0'] = {}
        d['hatc0']['a'] = {}
        d['hatc0']['a']['erosion'] = hatc0



        return d






