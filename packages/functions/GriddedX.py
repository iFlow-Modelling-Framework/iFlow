"""
Numerical data read from file
Now only supports 1D data import from txt files.

Original date: 17-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
import scipy.interpolate
from src.util.diagnostics import KnownError
from .checkVariables import checkVariables


class GriddedX():
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.__file = data.v('file')
        self.dimNames = dimNames
        checkVariables(self.__class__.__name__, ('file', self.__file), ('L', self.L))

        if self.__file[-4:] == '.txt':
            self.function, self.der, self.der2 = self.__readtxt()
        else:
            raise KnownError('Unknown file extension of file '+str(self.__file))
        return

    def value(self, x, **kwargs):
        return self.function(x)

    def derivative(self, x, **kwargs):
        return self.der(x)

    def secondDerivative(self, x, **kwargs):
        return self.der2(x)

    def __readtxt(self):
        # read preamble
        data = np.loadtxt(self.__file)
        x = data[:, 0]

        # reshape x with L
        if x[-1]<self.L:
            raise KnownError('Depth file does not specify depth over the full length of the system.')
        var = scipy.interpolate.interp1d(data[:, 0]/self.L, data[:, 1])
        try:
            varx = scipy.interpolate.interp1d(data[:, 0]/self.L, data[:, 2])
        except:
            varx = None
        try:
            varxx = scipy.interpolate.interp1d(data[:, 0]/self.L, data[:, 3])
        except:
            varxx = None

        return var, varx, varxx
