import numpy as np
"""
channel network 
                ___1____
        __4____/___2____
___5___/
       \______3______

"""


def setup():
        numberofchannels = 5

        geometry = {
            'depth': np.array([10, 10, 10, 10, 10]),
            'width': np.array([500,500,500, 500, 500]),
            'lc': np.array([1e5, 1e2, 2e5, 1e5, 1e5]),
            'length': np.array([1e5, 1e5, 1e5, 1e5, 1e5]),
            'lo': np.array([3e5, 3e5, 2e5, 2e5, 1e5]),
        }

        grid = {
            'jmax': np.array([100, 10, 1000, 100, 100]),
            'kmax': np.array([50, 50, 50, 50, 50]),
            'fmax': 2,
            'gridType': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant'],
            'jmax_out': np.array([100, 100, 100, 100, 100]),
            'kmax_out': np.array([50, 50, 50, 50, 50]),
            'fmax_out': 2,
            'gridType_out': ['equidistant','equidistant', 'equidistant', 'equidistant', 'equidistant']
        }

        label = {
            'Sea': np.array([1, 2, 3]) - 1,
            'River': np.array([5]) - 1,
            'Vertex': np.array([
                [1, 2, -4],
                [3, 4, -5]
            ]),
            'ChannelLabel': ['1', '2', '3', '4', '5'],
            'ColourLabel': ['g', 'r', 'c', 'm', 'darkorange']
        }

        forcing = {
            'M2': np.array([1, 1.2*np.exp(1j*0.1), 1.2*np.exp(1j*0.2)]),
            'discharge': np.array([100.0])
        }

        return numberofchannels, geometry, forcing, grid, label
