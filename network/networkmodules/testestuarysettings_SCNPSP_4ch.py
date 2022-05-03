import numpy as np
from numpy import pi
def setup():
        numberofchannels = 4

        L4 = 600
        geometry = {
            'depth': np.array([11, 7, 9, 9], dtype=float) ,
            'width': np.array([3500, 30000, 6200, 6600], dtype=float) ,
            'lc': np.array([470, 21, -368, 1e10], dtype=float) * 1e3,
            # 'width': np.array([8500, 30000, 6200, 6600], dtype=float) ,
            # 'lc': np.array([72, 21, -368, 1e10], dtype=float) * 1e3,

            
            'length': np.array([61, 54, 23, L4], dtype=float) * 1e3,
            'lo': -np.array([61, 54, 0, -23], dtype=float) * 1e3
        }

        grid = {
            'jmax': np.array([100, 100, 100, 100]) * 1 ,
            'kmax': np.array([50, 50, 50, 50]) * 1,
            'fmax': 2,
            'gridType': ['equidistant', 'equidistant', 'equidistant', 'equidistant'],
            'jmax_out': np.array([100, 100, 100, 100]) * 1,
            'kmax_out': np.array([50, 50, 50, 50]) * 1,
            'fmax_out': 2,
            'gridType_out': ['equidistant', 'equidistant', 'equidistant', 'equidistant']
        }

        label = {
            'Sea': np.array([1, 2]) - 1,
            'River': np.array([4]) - 1,
            'Vertex': np.array([[1, 2, -3], [3, -4]]),
            'ChannelLabel': ['NP', 'SP', 'SC', 'YR'],
            'ColourLabel': ['b', 'r', 'k', 'darkorange']
        }

        pd = 0.5
        cs = 0.5
        forcing = {
            'M2': np.array([1.3*np.exp(1j*.2222*pi), 1.3*np.exp(1j*0.1667*pi)], dtype=complex),
            # 'M4': np.array([.2*np.exp(1j*pd*np.pi), .2*np.exp(1j*pd*np.pi)], dtype=complex) ,
            'discharge': np.array([4000.0]),
            'S0': np.array([30, 25, 0.1, .1]),
            'xc': np.array([38000, 12800, 2050, .1]),
            'xL': np.array([8, 12, 7, .1]) * 1e3,
            'csea': np.array([cs, cs]),
            'Qsed': np.array([-1500.]) 
        }

        return numberofchannels, geometry, forcing, grid, label
