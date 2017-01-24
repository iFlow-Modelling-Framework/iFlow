"""
TurReadDPM

Date: 17-Feb-16
Authors: Y.M. Dijkstra
"""
from src.util.diagnostics.KnownError import KnownError
import numpy as np
import nifty as ny
from nifty import makeRegularGrid
from nifty.functionTemplates import NumericalFunctionWrapper

class TurReadDPM:
    # Variables

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        self.submodulesToRun = submodulesToRun
        return

    def run(self):
        d = {}

        # read data
        filePath = ' '.join(ny.toList(self.input.v('turfile')))
        try:
            with open(filePath, 'r') as f:
                f.next()
                format = f.next().split(' ')
                z = np.zeros(int(format[0]))
                Av = np.zeros((int(format[0]), (int(format[1])-1)/2), dtype=complex)

                for i, line in enumerate(f):
                    line = line.split(' ')
                    z[i] = float(line[0])
                    for j in range(0, Av.shape[1]):
                        Av[i, j] = float(line[2*j+1])+1j*float(line[2*j+2])

        except IOError as e:                       # throw exception if file is not found
            raise KnownError(('No file found at ' + filePath), e)

        # make a grid
        dimensions = ['z', 'f']
        enclosures = [(z[0], z[-1]), None]
        axisTypes = ['list', 'integer']
        axisSize = [z, Av.shape[1]-1]
        axisOther=['', '']
        grid = makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures)

        # load factors for reducing/increading M2 Av
        lambda_Avamp = self.input.v('lambda_Avamp')
        lambda_Avphase = self.input.v('lambda_Avphase')

        Av[:,1] = lambda_Avamp*Av[:,1]
        Av[:,1] = Av[:,1]*np.exp(lambda_Avphase/180.*np.pi*1j)

        # make ordering of Av if requested
        ordered = self.input.v('ordered')
        if ordered == 'True':
            Av1 = np.zeros(Av.shape, dtype=complex)
            Av1[:, 1:] = Av[:, 1:]
            Av[:, 1:] = 0

            Av1[:,1] = Av1[:,1]*np.exp(lambda_Avphase/180.*np.pi*1j)
            Av1[:,1] = lambda_Avamp*Av1[:,1]

            nf1 = NumericalFunctionWrapper(Av1, grid)
            d['Av1'] = nf1.function
        else:
            d['Av1'] = 0.

        # ad hoc shutdown of time varying Av
        # Av[:,1:] = 0
        # Av[:,2] = 0.70*Av[:,2]
        # Av[:,3:] = 0
        # Av[:,2:] = 0
        # Av[:,1] = 5*Av[:,1]


        Av[:,0]+=10**-6     # add molecular viscosity
        nf = NumericalFunctionWrapper(Av, grid)
        d['Av'] = nf.function
        d['Roughness']= 0.
        d['BottomBC'] = 'NoSlip'
        return d