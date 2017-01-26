"""
Calibration

Date: 06-Mar-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from scipy.optimize import minimize


class AutoCalibration:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        stop = True
        return stop

    def run_init(self):
        # load observations
        # TODO temp for twin experiment; move to run
        self.x_obs = np.arange(0, 1, 0.1)
        self.zeta_obs = [ 1.77200000+0.j, 1.48444530-0.4940093j, 0.93002957-0.91896679j, 0.43297738-1.08699649j,
                          0.04539667-1.0939328j, -0.29089626-0.9750055j, -0.57467817-0.67652735j, -0.63564916-0.16312824j,
                          -0.26415931+0.22483038j, 0.05206857+0.09799576j]

        d = self.run()
        return d

    def run(self):
        # load observations TODO
        # self.x_obs = np.arange(0, 1, 0.1)
        # self.zeta_obs =


        # load variables to x0
        x0 = []
        self.variables = ny.toList(self.input.v('variables'))
        for var in self.variables:
            x0.append(self.input.v(var))

        # minimize
        x = minimize(self.costFunction, x0)
        print x

        # load calibration to dictionary for returning
        d = {}
        for i, var in enumerate(self.variables):
            d[var] = x[i]

        return d

    def costFunction(self, x0):
        # load calibration variables to dict
        d = {}
        for i, var in enumerate(self.variables):
            d[var] = x0[i]

        # run modules
        ny.runCallStackLoop(startUpdate=d)

        # compute cost function
        #   NB. now only take M2 component
        zeta_mod = self.input.v('zeta0', x=self.x_obs, z=0, f=1)
        dzeta = np.abs(zeta_mod)-np.abs(self.zeta_obs)
        dphi = np.angle(zeta_mod)-np.angle(self.zeta_obs)
        J = np.sum(np.sqrt( (dzeta)**2.+2.*np.abs(zeta_mod)*np.abs(self.zeta_obs)*(1.-np.cos(dphi)) ))

        return J


