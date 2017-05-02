"""
SedDynamic

Date: 09-Nov-16
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
from cFunction import cFunction
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

        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        ftot = 2*fmax+1
        self.submodulesToRun = self.input.v('submodules')
        # H = self.input.v('H', range(0, jmax+1))
        method = self.input.v('erosion_formulation')

        ################################################################################################################
        # Left hand side
        ################################################################################################################
        # PrSchm = self.input.v('sigma_rho', range(0, jmax+1), range(0, kmax+1), [0])  # assume it is constant in time; else division with AV fails
        Av = self.input.v('Av', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))
        # Kv = Av/PrSchm
        Kv = self.input.v('Kv', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        # ws = np.zeros((jmax+1, kmax+1, fmax+1))
        ws = self.input.v('ws0', range(0, jmax+1), range(0, kmax+1), range(0, fmax+1))

        ################################################################################################################
        # Forcing terms
        ################################################################################################################
        F = np.zeros([jmax+1, kmax+1, ftot, 1], dtype=complex)
        Fsurf = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)
        Fbed = np.zeros([jmax+1, 1, ftot, 1], dtype=complex)

        # erosion
        E = erosion(ws, Av, 0, self.input, method)
        Fbed[:, :, fmax:, 0] = -E

        ################################################################################################################
        # Solve equation
        ################################################################################################################
        c, cMatrix = cFunction(ws, Kv, F, Fsurf, Fbed, self.input, hasMatrix = False)
        c = c.reshape((jmax+1, kmax+1, ftot))

        d = {}
        d['hatc0'] = {}
        d['hatc0']['a'] = {}
        d['hatc0']['a']['erosion'] = ny.eliminateNegativeFourier(c, 2)
        return d






