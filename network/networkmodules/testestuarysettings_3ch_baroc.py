import numpy as np

def setup():
        numberofchannels = 4

        Bsea = 500 + 1e-10
        lc = -40000 / np.log(500 / Bsea)

        geometry = {
            # 'depth': np.array([13, 13, 13], dtype=float) ,
            # 'depth': np.array([15, 11, 13], dtype=float) ,
            # 'width': np.array([Bsea, 500, 1000], dtype=float) ,
            # 'lc': np.array([lc, 1e12, 1e12], dtype=float) ,
            # 'length': np.array([40, 40, 2000], dtype=float) * 1e3,
            # 'lo': -np.array([40, 40, 0], dtype=float) * 1e3
            'depth': np.array([15, 11, 13, 13], dtype=float) ,
            # 'depth': np.array([13, 13, 13, 13], dtype=float) ,
            # 'depth': np.array([8.5, 13, 13, 13], dtype=float) ,
            'width': np.array([Bsea, 500, 1000, 1000], dtype=float) ,
            'lc': np.array([lc, 1e20, 1e20, 1e20], dtype=float) ,
            'length': np.array([40, 40, 40, 1000], dtype=float) * 1e3,
            'lo': -np.array([40, 40, 0, -40], dtype=float) * 1e3
        }

        grid = {
            'jmax': np.array([100, 100, 100, 100]) * 2,
            'kmax': np.array([50, 50, 50, 50]) * 1,
            'fmax': 2,
            'gridType': ['equidistant', 'equidistant', 'equidistant', 'equidistant'],
            'jmax_out': np.array([100, 100, 100, 100]) *2,
            'kmax_out': np.array([50, 50, 50, 50]) * 1,
            'fmax_out': 2,
            'gridType_out': ['equidistant', 'equidistant', 'equidistant', 'equidistant']
        }

        label = {
            'Sea': np.array([1, 2]) - 1,
            'River': np.array([4]) - 1,
            'Vertex': np.array([[1, 2, -3], [3, -4]]),
            'ChannelLabel': ['1', '2', '3', '4'],
            'ColourLabel': ['b', 'r', 'darkorange',  'orange']
        }
        xcshift = 10
        # xcshift = 8
        
        forcing = {
            'M2': np.array([1*np.exp(0j*np.pi), 1*np.exp(0j*np.pi)], dtype=complex),
            'discharge': np.array([1000.0]),
            'S0': np.array([30, 30, 30, 30]),
            'xc': (np.array([30, 30, 30-40, 30-80]) - xcshift)* 1e3,
            'xL': np.array([20, 20, 20, 20]) * 1e3,
            # 'csea': np.array([.1, .1]),
            'csea': np.array([.05, .05]),
            'Qsed': np.array([-50]) 
        }

        return numberofchannels, geometry, forcing, grid, label
