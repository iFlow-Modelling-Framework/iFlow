"""
Calibration
Calibration of the M2 water level by evaluating the cost function of Jones and Davies (1996) for a single calibration parameter.
Starts from a user-supplied initial guess for the calibration parameter and then uses a divide-and-conquer algorithm to find the optimal parameter.

Date: July 2017
Authors: Y.M. Dijkstra
"""
import numpy as np
import logging
from .cost_function_DJ96 import cost_function_DJ96


class Calibration:
    # Variables
    logger = logging.getLogger(__name__)
    TOLLERANCE = 0.1

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        if not np.nan in self.calib_par:
            if (self.calib_par[2]-self.calib_par[0])/self.calib_par[1] < self.TOLLERANCE:
                self.stop = True
                if self.input.v(self.calibvar_name) == self.calib_par[1]:
                    self.logger.info('\t Calibration converged in %s iterations at %s = %s' % (iteration, self.calibvar_name, str(self.calib_par[1])))
                    # print self.calib_par
                    # print self.cost
                    return True
        return False

    def run_init(self):
        self.logger.info('Running Calibration - init')

        # initialise
        self.calib_par = np.nan*np.zeros(3)
        self.cost = np.nan*np.zeros(3)
        self.upperbound = [np.nan, np.nan]       #upper bound of the calibration_par, cost
        self.lowerbound = [np.nan, np.nan]       #lower bound of the calibration_par, cost
        self.stop = False

        # First set the centre slot
        self.calibvar_name = self.input.v('calibrationParameter')
        self.changeFactor = self.input.v('changeFactor')
        self.ignorePhase = self.input.v('ignorePhase')
        if self.ignorePhase == 'True':
            self.ignorePhase = True
        else:
            self.ignorePhase = False

        self.pos = 1
        self.calib_par[1] = self.input.v(self.calibvar_name)

        d = {}
        d[self.calibvar_name] = self.input.v(self.calibvar_name)
        return d

    def run(self):
        self.logger.info('Running Calibration')
        d = {}
        ################################################################################################################
        # Evaluate result of the last iteration
        ################################################################################################################
        L = self.input.v('L')
        measurementset = self.input.v('measurementset')
        x_obs = self.input.v(measurementset, 'x_waterlevel')/L
        x_ext = np.zeros(len(x_obs)+2)
        x_ext[1:-1] = x_obs
        x_ext[0] = 0
        x_ext[-1] = 1
        zeta_obs = self.input.v(measurementset, 'zeta', x=x_obs, z=0, f=1)

        obs_subset = self.input.obs_subset('observations_subset')
        if obs_subset is None:
            zeta_obs = self.input.v(measurementset, 'zeta', x=x_obs, z=0, f=1)
        else:
            zeta_obs = self.input.v(measurementset, 'zeta', obs_subset, x=x_obs, z=0, f=1)

        zeta = self.input.v('zeta0', x=x_obs, z=0, f=1)

        self.cost[self.pos] = cost_function_DJ96(x_ext, zeta_obs, zeta, ignorePhase=self.ignorePhase)     # Calibrate only on M2
        dif = self.cost[1] - self.cost[self.pos]        # compare last iteration to centre iteration

        if self.pos != 1 and dif > 0 and not self.stop:     #last setting was better than previous, set new centre point and open up a slot
            # move new best value to the centre
            if self.pos == 0:
                self.calib_par[1:] = self.calib_par[:-1]
                self.calib_par[0] = self.lowerbound[0]
                self.cost[1:] = self.cost[:-1]
                self.cost[0] = self.lowerbound[1]
            else:
                self.calib_par[:-1] = self.calib_par[1:]
                self.calib_par[-1] = self.upperbound[0]
                self.cost[:-1] = self.cost[1:]
                self.cost[-1] = self.upperbound[1]

        ################################################################################################################
        # Set new parameters
        ################################################################################################################
        # if converged, run with latest settings
        if self.stop:
            newvalue = self.calib_par[1]
            self.pos = 1

        # elif there is any open slot, fill it with a new value set by changeFactor
        elif np.isnan(self.calib_par[2]) or np.isnan(self.calib_par[0]):
            if np.isnan(self.calib_par[2]):
                self.pos = 2
                newvalue = self.calib_par[1]*(self.changeFactor)
            else:
                self.pos = 0
                newvalue = self.calib_par[1]*(1./self.changeFactor)

        # if there is no open slot, half the biggest interval
        else:
            if self.calib_par[1] - self.calib_par[0] > self.calib_par[2] - self.calib_par[1]:
                newvalue = 0.5*(self.calib_par[1] + self.calib_par[0])
                self.pos = 0
                self.lowerbound = [self.calib_par[0], self.cost[0]]
            else:
                newvalue = 0.5*(self.calib_par[2] + self.calib_par[1])
                self.pos = 2
                self.upperbound = [self.calib_par[2], self.cost[2]]

        d[self.calibvar_name] = newvalue
        self.calib_par[self.pos] = newvalue
        self.logger.info('\tCalibration setting new value %s=%s' %(self.calibvar_name, str(newvalue)))

        return d


