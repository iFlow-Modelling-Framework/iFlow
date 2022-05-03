import numpy as np

def setup():
        numberofchannels = 2

        geometry = {
            'depth': np.array([15, 15], dtype=float) * 1,
            'width': np.array([1000.,1000.], dtype=float) * 1,
            'lc': np.array([1e100, 1e100], dtype=float),
            'length': np.array([40., 40.]) * 1e3 * 1,
            'lo': -np.array([40., 0]) * 1e3 * 1
        }

        grid = {
            'jmax': np.array([100, 100]) * 4,
            'kmax': np.array([50, 50]) * 1,
            'fmax': 2,
            'gridType': ['equidistant','equidistant'],
            'jmax_out': np.array([100, 100]) * 4,
            'kmax_out': np.array([50, 50]) * 1,
            'fmax_out': 2,
            'gridType_out': ['equidistant','equidistant']
        }

        label = {
            'Sea': np.array([1]) - 1,
            'River': np.array([2]) - 1,
            'Vertex': np.array([[1, -2]]),
            'ChannelLabel': ['sea', 'river'],
            'ColourLabel': ['g', 'r']
        }

        forcing = {
            'M2': np.array([1]),
            'discharge': np.array([1000.0]),
            'S0': np.array([30, 30]),
            'xc': np.array([30000, 30000-40000]) - 4000 , # -3000: ETM in middle
            'xL': np.array([20000, 20000]),
            'csea': np.array([.1]),
            'Qsed': np.array([-0.]) 
        }

        return numberofchannels, geometry, forcing, grid, label

