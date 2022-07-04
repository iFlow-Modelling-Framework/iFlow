"""
Class DFunction

provides a callable function that references to taking the numerical derivative of a function. It stores the function,
axis along which to take the derivative and the grid (including grid data).

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
from .derivativeOfFunction import derivativeOfFunction, secondDerivativeOfFunction
from .diagnostics.KnownError import KnownError


class DFunction:
    def __init__(self, function, dim, grid, gridname='grid'):
        self.dimNames = function.__self__.dimNames
        self.function = function
        self.derivativeDim = dim
        self.gridname = gridname
        self.grid = grid
        return

    def dfunction(self, **kwargs):
        for dir in list(set(sorted(self.derivativeDim))): # loop over all dimensions once
            order = len([i for i in self.derivativeDim if i==dir]) # collect the number of occurances of this dimension
            if order == 1:
                value = derivativeOfFunction(self.function, dir, self.grid, gridname=self.gridname, epsilon=1e-4, **kwargs)
            elif order == 2:
                value = secondDerivativeOfFunction(self.function, dir, self.grid, gridname=self.gridname, epsilon=1e-4, **kwargs)
            else:
                raise KnownError('Numerical derivatives of order %s are not implemented' % str(order))
        return value


