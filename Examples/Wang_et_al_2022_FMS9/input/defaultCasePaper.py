import numpy as np


class defaultCasePaper:

    def __init__(self):
        return

    def setup(self):
            Bsea = 500 + 1e-10
            lc = -40000 / np.log(500 / Bsea)

            d = {}
            d['network_settings'] = {}
            d['network_settings']['numberofchannels'] = 4

            # Define role of channels
            d['network_settings']['label'] = {
                'ChannelLabel': ['1', '2', '3', '4'],                   # You can give names to the channels here. Output per channel is saved under these names.
                'Sea': np.array([1, 2]) - 1,                            # numbers of the channels with sea connection; make sure these are the lowest numbers. count starts from 0
                'River': np.array([4]) - 1,                             # numbers of the channels with river connection
                'Vertex': [[1, 2, -3], [3,-4]],               # numbers of the channels meeting at junction. This is a list of lists, where each element in the outer list is a junction and the inner lists contain the numbers of the channels involved. A minus sign indicates that downstream directed flow is out of the channel.
                'ColourLabel': ['b', 'r', 'darkorange',  'orange']      # colour label for plotting (optional) per channel
            }

            # Geometry
            d['network_settings']['L'] = list(np.array([40, 40, 40, 1000])*1e3)                       # Length per channel (in m)
            d['network_settings']['L0'] = list(-np.array([40, 40, 0, -40], dtype=float) * 1e3)    # locations of the x_local=0 in global coordinates (in m)
            d['network_settings']['B0'] = {
                'type': ['functions.Exponential','functions.Exponential','functions.Exponential','functions.Exponential'],
                'Lc': np.array([lc, 1e20, 1e20, 1e20]),
                'C0': np.array([Bsea, 500, 1000, 1000]),
            }
            d['network_settings']['H0'] = {
                'type': ['functions.Constant','functions.Constant','functions.Constant','functions.Constant'],
                'C0': np.array([15, 11, 13, 13], dtype=float) ,
            }

            # Grid
            d['network_settings']['grid'] = {
                'jmax': np.array([200, 200, 200, 200]),
                'kmax': np.array([50, 50, 50, 50]),
                'fmax': 2,
                'gridTypeX': ['equidistant', 'equidistant', 'equidistant', 'equidistant'],
                'gridTypeZ': ['equidistant', 'equidistant', 'equidistant', 'equidistant'],

                'jmax_out': np.array([100, 100, 100, 100]),
                'kmax_out': np.array([50, 50, 50, 50]),
                'fmax_out': 2,
                'gridTypeX_out': ['equidistant', 'equidistant', 'equidistant', 'equidistant'],
                'gridTypeZ_out': ['equidistant', 'equidistant', 'equidistant', 'equidistant']
            }

            # Turbulence, parameters in here should match the chosen turbulence model
            d['network_settings']['turbulence'] = {
                'Av0amp': [0.0005*d['network_settings']['H0']['C0'][0], 0.0005*d['network_settings']['H0']['C0'][1], 0.0005*d['network_settings']['H0']['C0'][2], 0.0005*d['network_settings']['H0']['C0'][3]],
                'Av0phase': [ 0. ]*4,
                'sf0': [ 0.001 ]*4,
                'm': [ 0 ]*4,
                'n': [ 0 ]*4
            }

            # Salinity
            xcshift = 10
            d['network_settings']['salinity'] = {
                'ssea': np.array([30, 30, 30, 30]),
                'xc': (np.array([30, 30, 30-40, 30-80]) - xcshift)* 1e3,
                'xl': np.array([20, 20, 20, 20]) * 1e3,
            }

            # Hydrodynamics
            d['network_settings']['hydrodynamics'] = {
                # tide, array length equals number of tidal channels
                'M2': np.array([1*np.exp(0j*np.pi), 1*np.exp(0j*np.pi)], dtype=complex),
                'M4': np.array([0, 0], dtype=complex),

                # river, array length equals number of river channels
                'Q0': np.array([0.0]),
                'Q1': np.array([1000.0]),
            }

            d['network_settings']['sediment'] = {
                'erosion_formulation': ['Chernetsky']*4,
                'finf': [1e-3]*4,
                'ws0': [2e-3]*4,
                'Kh': [100]*4,
                'csea': np.array([.05, .05]),   # array length equal to number of sea channels
                'Qsed': np.array([-50])         # array length equal to number of river channels
            }

            return d
