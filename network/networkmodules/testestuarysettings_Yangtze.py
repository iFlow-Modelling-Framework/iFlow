import numpy as np
from numpy import pi
"""
Yangtze 
___11___10__9__8__7____1___
                    \6___2___
                      \5__3__                 
                        \__4__   
"""


def setup():
        numberofchannels = 7

        geometry = {
            'depth': np.array([5, 7, 11, 7, 9, 9, 10], dtype=float),
            'length': np.array([85, 60, 61, 54, 23, 51, 600], dtype=float) * 1000,
            'width': np.array([12000, 20000, 3500, 30000, 6200, 14000, 6600], dtype=float),
            'lc': np.array([30, 52, 470, 21, -368, 64, 1660], dtype=float) * 1000,
            'lo': -np.array([85, 111, 135, 128, 74, 51, 0,], dtype = float) * 1000,
            # 'width': np.array([705.8, 6308.42, 3074, 2292.8, 6600, 6310.3, 4600], dtype=float)
        }

        grid = {
            'jmax': np.array([100, 100, 100, 100, 100, 100, 100])*2,
            'kmax': np.array([50, 50, 50, 50, 50, 50, 50]),
            'fmax': 2,
            'gridType': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant'],
            'jmax_out': np.array([100, 100, 100, 100, 100, 100, 100])*2,
            'kmax_out': np.array([50, 50, 50, 50, 50, 50, 50]),
            'fmax_out': 2,
            'gridType_out': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant']
        }

        label = {
            'Sea': np.array([1, 2, 3, 4]) - 1,
            'River': np.array([7]) - 1,
            'Vertex': np.array([[1, 6, -7], [2, 5, -6], [3, 4, -5]]),
            'ChannelLabel': ['NB','NC','NP','SP','SC','SB','YR'],
            'ColourLabel': ['g', 'm' ,'b', 'r','k', 'c','darkorange']
        }

        forcing = {
            'M2': np.array([ 1.55*np.exp(1j*.1667*pi), 1.06*np.exp(1j*.2222*pi), 1.3*np.exp(1j*.2222*pi), 1.3*np.exp(1j*0.1667*pi) ]),
            # 'M4': np.array([ .55*np.exp(1j*.1667*pi), .06*np.exp(1j*.2222*pi), .3*np.exp(1j*.2222*pi), .3*np.exp(1j*0.1667*pi) ]),
            'discharge': np.array([10000.0]),
            
            # 'S0': np.array([26.3, 30, 35, 20.65, 4, 3, 0]),
            # 'xc': np.array([30000, 48000, 25000, 12800, 2050, 14000, 0]),
            # 'xL': np.array([13000, 10000, 8000, 12000, 7000, 12000, 1]),
            'S0': np.array([30, 25, 30, 25, 0.1, 0.1, 0]),
            'xc': np.array([30000, 48000, 38000, 12800, 2050, 14000, 0]),
            'xL': np.array([13000, 10000, 8000, 12000, 7000, 12000, 1]),
            'csea': np.array([.3, .3, .3, .3]) ,
            'Qsed': np.array([-3000.]) 
        }

        return numberofchannels, geometry, forcing, grid, label

        # numberofchannels = 11

        # geometry = {
        #     'depth': np.array([5, 7, 11, 7, 9, 9, 10, 10, 10, 10, 10], dtype=float),
        #     'width': np.array([705.8, 6308.42, 3074, 2292.8, 6600, 6310.3, 5178, 4062, 3187, 2500, 1961.3], dtype=float),
        #     'lc': -np.array([30, 52, 470, 21, -368, 64, 206, 206, 206, 206, 1660], dtype=float) * 1000,
        #     'length': np.array([85, 60, 61, 54, 23, 51, 50, 50, 50, 50, 370], dtype=float) * 1000,
        #     'lo': np.array([85, 111, 135, 128, 74, 51, 0, -50, -100, -150, -200], dtype = float) * 1000
        # }

        # grid = {
        #     'jmax': np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
        #     'kmax': np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
        #     'fmax': 2,
        #     'gridType': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant'],
        #     'jmax_out': np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
        #     'kmax_out': np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
        #     'fmax_out': 2,
        #     'gridType_out': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant', 'equidistant']
        # }

        # label = {
        #     'Sea': np.array([1, 2, 3, 4]) - 1,
        #     'River': np.array([11]) - 1,
        #     'Vertex': np.array([[1, 6, -7], [2, 5, -6], [3, 4, -5], [7, -8], [8, -9], [9, -10], [10, -11]]),
        #     'ChannelLabel': ['NB','NC','NP','SP','SC','SB','YR','YR2','YR3','YR4','YR5'],
        #     'ColourLabel': ['g', 'm' ,'b', 'r','k', 'c','darkorange','darkorange','darkorange','darkorange','darkorange']
        # }

        # forcing = {
        #     'M2': np.exp(-1j*pi/3 * 1.1) * np.array([1.55*np.exp(-1j*.1667*pi), 1.06*np.exp(-1j*.2222*pi), 1.3*np.exp(-1j*.2222*pi), 1.3*np.exp(-1j*0.1667*pi)]),
        #     'discharge': np.array([10000.0])
        # }

        # return numberofchannels, geometry, forcing, grid, label