"""
Measurements of the Scheldt

Date: 20-Apr-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import copy
from src.DataContainer import DataContainer
import src.old.functionTemplates


class Scheldt:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        d = {}
        d['Scheldt_measurements'] = {}

        ################################################################################################################
        # data
        ################################################################################################################
        # Stations (optional)
        station_names = ['Vlissingen', 'Terneuzen', 'Hansweert', 'Bath', 'Prosperpolder', 'Liefkenshoek', 'Antwerpen', 'Temse', 'St. Amands', 'Dendermonde', 'Schoonaarde', 'Wetteren', 'Melle']
        x_station = np.asarray([0.,     18.5,           33.8,      49.8,     54,             61.1,           75.6,       97.3,     106.8,        119.8,         130.6,        142.7,     148.8])*1000.
        river_sections = {'Western Scheldt': [0, 1, 2, 3],
                          'Lower Sea Scheldt': [4, 5, 6],
                          'Upper Sea Scheldt': [7, 8, 9, 10, 11, 12]}

        # Stations (optional)
        L = 160000.

        # Water Level Measurements (mandatory)
        x_waterlevel = np.asarray([0., 18.5, 33.8, 49.8, 54, 61.1, 75.6, 97.3, 106.8, 119.8, 130.6, 142.7, 148.8])*1000.    # x-coordinate of the measurement locations (in m)
        phaseM2mouth = 0. # Phase of the M2 tidal water level at the mouth (x=0).

        M2amp = np.asarray([1.77, 1.98, 2.03, 2.18, 2.19, 2.26, 2.31, 2.28, 2.22, 1.69, 1.31, 1.09, 1.02])                  # M2 amplitude (in m)
        M2phase = np.asarray([0, 10.4, 20.5, 31.1, 33.1, 35.6, 44.3, 62.7, 73.4, 93.3, 116.4, 143.4, 157.4]) - phaseM2mouth # M2 phase (in deg)
        M4amp = np.asarray([0.14, 0.12, 0.11, 0.11, 0.12, 0.12, 0.13, 0.16, 0.24, 0.25, 0.24, 0.21, 0.22])                  # M4 amplitude (in m)
        M4phase = np.asarray([-1.3, 12.3, 39.4, 58.7, 62.2, 65.1, 74.6, 88.2, 101.7, 128.7, 164.3, 212.6, 242.9]) - 2*phaseM2mouth  # M4 amplitude (in m)

        # Velocity measurements (optional)
        x_station_vel = np.asarray([17, 28, 40, 41 ,63, 74, 85, 102, 120, 132, 140])*1000.
        M0vel = np.asarray([-0.096, -0.09, -0.011, -0.040, -0.007, -0.1, 0.005, -0.011, 0.117, -0.144, 0.037])              # subtidal velocity
        M2velamp = np.asarray([1., 1.096, 0.785, 0.818, 0.332, 0.813, 0.56, 0.627, 0.798, 0.680, 0.190])
        M2velphase = np.asarray([301, 285, 164, 176, 243, 298, 344, 291, 103, 283, 98]) - phaseM2mouth
        M4velamp = np.asarray([0.109, 0.069, 0.112, 0.066, 0.033, 0.079, 0.085, 0.154, 0.226, 0.254, 0.084])
        M4velphase = np.asarray([311, 281, 69, 67, 234, 28, 329, 248, 27, 221, 22]) - 2*phaseM2mouth


        ################################################################################################################
        # process data
        ################################################################################################################
        water_level = np.zeros((len(x_waterlevel), 3), dtype=complex)
        water_level[:, 0] = np.nan
        water_level[:, 1] = M2amp * np.exp(-M2phase / 180. * np.pi * 1j)
        water_level[:, 2] = M4amp * np.exp(-M4phase / 180. * np.pi * 1j)

        u_comp = np.zeros((len(x_station_vel), 3), dtype=complex)
        u_comp[:, 0] = M0vel
        u_comp[:, 1] = M2velamp * np.exp(-M2velphase / 180. * np.pi * 1j)
        u_comp[:, 2] = M4velamp * np.exp(-M4velphase / 180. * np.pi * 1j)

        ################################################################################################################
        # grid
        ################################################################################################################
        grid = {}
        grid['gridtype'] = 'Regular'
        grid['dimensions'] = ['x', 'f']
        grid['axis'] = {}
        grid['maxIndex'] = {}
        grid['low'] = {}
        grid['high'] = {}
        grid['contraction'] = [[], []]

        grid['high']['x'] = L
        grid['low']['x'] = 0.
        grid['high']['f'] = None
        grid['low']['f'] = None
        grid['axis']['f'] = np.asarray([0, 1, 2]).reshape((1,3))
        grid['maxIndex']['f'] = 2

        grid_waterlevel = grid
        grid_velocity = copy.deepcopy(grid)

        grid_waterlevel['axis']['x'] = x_waterlevel/grid['high']['x']
        grid_waterlevel['maxIndex']['x'] = len(x_waterlevel)-1
        grid_velocity['axis']['x'] = x_station_vel/grid['high']['x']
        grid_velocity['maxIndex']['x'] = len(x_station_vel)-1

        grid_waterlevel = DataContainer({'grid':grid_waterlevel})
        grid_velocity = DataContainer({'grid':grid_velocity})

        ################################################################################################################
        # load to dictionary
        ################################################################################################################
        # stations
        d['Scheldt_measurements']['x_stations'] = x_station
        d['Scheldt_measurements']['stations'] = station_names
        d['Scheldt_measurements']['sections'] = river_sections

        # water level
        nf = src.old.functionTemplates.NumericalFunctionWrapper(water_level, grid_waterlevel)
        d['Scheldt_measurements']['zeta'] = nf.function
        d['Scheldt_measurements']['x_waterlevel'] = x_waterlevel

        # velocity
        nfu2 = src.old.functionTemplates.NumericalFunctionWrapper(u_comp, grid_velocity)       # now saved without phase data
        d['Scheldt_measurements']['x_velocity'] = x_station_vel
        d['Scheldt_measurements']['u_comp'] = nfu2.function

        return d