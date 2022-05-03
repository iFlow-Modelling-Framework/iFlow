import numpy as np

def setup():
        numberofchannels = 3

        

        geometry = {
            'depth': np.array([15, 15, 15], dtype=float),
            'width': np.array([500, 500,1000], dtype=float) ,
            'lc': np.array([1000, 1000, 1000], dtype=float) * 1e11,
            'length': np.array([40, 40, 40], dtype=float) * 1e3,
            'lo': -np.array([40, 40, 0], dtype=float) * 1e3
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
            'ChannelLabel': ['1', '2', '3'],
            'ColourLabel': ['g', 'r', 'darkorange']
        }
        xcshift = 4
        pd = 0.6j
        forcing = {
            'M2': np.array([1*np.exp(0j*np.pi), 1*np.exp(0j*np.pi)], dtype=complex),
            # 'M4': np.array([.1*np.exp(pd*np.pi), .1*np.exp(pd*np.pi)], dtype=complex),
            'discharge': np.array([1000.0]),
            'S0': np.array([30, 30, 30]),
            'xc': (np.array([30, 30, 30-40]) - xcshift)* 1e3,
            'xL': np.array([20, 20, 20]) * 1e3,
            # 'csea': np.array([.1, .1]),
            'csea': np.array([.1, .1]),
            'Qsed': np.array([0.]) 
        }

        return numberofchannels, geometry, forcing, grid, label
