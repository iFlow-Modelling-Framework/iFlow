import numpy as np
from numpy import pi
def setup():
        numberofchannels = 3

        # L3 = 160 km  is a nice case! Standing wave in sea channels, ETM at junction.
        # 150-160 km: ?? 2 ETMs due to tide.
        geometry = {
            'depth': np.array([12.5, 7, 9], dtype=float),
            'width': np.array([3500, 30000, 6200], dtype=float) ,
            'lc': np.array([470, 21, -368], dtype=float) * 1e3,
            # 23 km it too short
            'length': np.array([61, 54, 23], dtype=float) * 1e3,
            'lo': -np.array([61, 54, 0], dtype=float) * 1e3
        }

        grid = {
            'jmax': np.array([100, 100, 100]) ,
            'kmax': np.array([50, 50, 50]),
            'fmax': 2,
            'gridType': ['equidistant', 'equidistant', 'equidistant'],
            'jmax_out': np.array([100, 100, 100]) ,
            'kmax_out': np.array([50, 50, 50]),
            'fmax_out': 2,
            'gridType_out': ['equidistant', 'equidistant', 'equidistant']
        }

        label = {
            'Sea': np.array([1, 2]) - 1,
            'River': np.array([3]) - 1,
            'Vertex': np.array([[1, 2, -3]]),
            'ChannelLabel': ['NP', 'SP', 'SC'],
            'ColourLabel': ['b', 'r','k']
        }

        forcing = {
            'M2': np.array([1.3*np.exp(1j*.2222*pi), 1.3*np.exp(1j*0.1667*pi)], dtype=complex),
            # 'M4': np.array([.3*np.exp(pd*np.pi), .3*np.exp(pd*np.pi)], dtype=complex) ,
            'discharge': np.array([3000.0]),
            'S0': np.array([35, 20.65, 4]),
            'xc': np.array([25000, 12800, 2050]),
            'xL': np.array([8, 12, 7]) * 1e3,
            'csea': np.array([7, 7]),
            'Qsed': np.array([-500.]) 
        }

        return numberofchannels, geometry, forcing, grid, label
