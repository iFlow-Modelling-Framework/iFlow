"""
Measurements of the Garonne Tidal River
based on Brouwer et al WP1.3, WL Borgerhout

Date: 20-Apr-16
Authors: I Jalon-Rojas (modified from Y.M. Dijkstra mesurements.Scheldt)
"""
import numpy as np


class Gironde_v1:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):

        L = self.input.v('L') #IJR

        d = {}
        d['Gironde_measurements'] = {}

        Qmeas=self.input.v('Q')
        year = self.input.v('year')
        TRm=self.input.v('TRm')

        #root_meas='packages\measurements\observations\MeasurementsGaronne\Version1'
        #datatide = np.loadtxt(root_meas + '\meas_v1_' + str(year) + '_TRm' + str(TRm) + '_Q' + str(Qmeas) + '.txt')

        root_meas = 'Examples/Jalon-Rojas_et_al_2020_JGR/measurements/observations/' + str(year) + '/'
        root_meas = r'D:\Work\PhD\iFlow\iFlow3\Examples\Jalon-Rojas_et_al_2021_JGR\measurements\observations\1953/'
        datatide = np.loadtxt(root_meas + 'obs_' + str(year) + '_TRm' + str(TRm) + '_Q' + str(Qmeas) + '.txt')


        # data
        station_names = ['Le Marquis', 'Bordeaux', 'Cadillac','La Reole']

        x_station = datatide[:,0]  #np.asarray([0., 18.5, 33.8, 49.8, 54, 61.1, 75.6, 97.3, 106.8, 119.8, 130.6, 142.7, 148.8])*1000.
        M2amp = datatide[:,1]  #np.asarray([1.77, 1.98, 2.03, 2.18, 2.19, 2.26, 2.31, 2.28, 2.22, 1.69, 1.31, 1.09, 1.02])
        M2phase = datatide[:,2]
        M4amp = datatide[:,3]
        M4phase = datatide[:,4]


        # process
        water_level = np.zeros((len(x_station), 3), dtype=complex)
        water_level[:, 1] = M2amp * np.exp(-M2phase / 180. * np.pi * 1j)
        water_level[:, 2] = M4amp * np.exp(-M4phase / 180. * np.pi * 1j)

        # grid
        grid = {}
        grid['gridtype'] = 'Regular'
        grid['dimensions'] = ['x', 'z', 'f']
        grid['axis'] = {}
        grid['maxIndex'] = {}
        grid['low'] = {}
        grid['high'] = {}
        grid['contraction'] = [[], []]

        grid['high']['x'] = L
        grid['low']['x'] = 0.
        grid['high']['z'] = 1
        grid['low']['z'] = 0.
        grid['high']['f'] = None
        grid['low']['f'] = None
        grid['axis']['f'] = np.asarray([0, 1, 2]).reshape((1,1,3))
        grid['axis']['z'] = np.asarray([0,1]).reshape((1,2))
        grid['maxIndex']['f'] = 2
        grid['maxIndex']['z'] = 1

        grid_stations = grid
        grid_stations['axis']['x'] = x_station/grid['high']['x']
        grid_stations['maxIndex']['x'] = len(x_station)-1

        d['grid_stations_Garonne'] = grid_stations

        # load to dictionary
        d['Gironde_measurements']['zeta'] = water_level.reshape((len(x_station), 1, 3))
        d['Gironde_measurements']['x_stations'] = x_station
        d['Gironde_measurements']['stations'] = station_names
        d['__variableOnGrid'] = {}
        d['__variableOnGrid']['Gironde_measurements'] = {}
        d['__variableOnGrid']['Gironde_measurements']['zeta'] = 'grid_stations_Garonne'

        return d